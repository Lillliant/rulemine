"""Command-line interface for RuleMine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rulemine.adapters.antigravity import discover_sessions, load_turns, parse_since
from rulemine.adapters.cursor import discover_cursor_sessions
from rulemine.extract import extract_events_for_window
from rulemine.models import DecisionDraft, ExtractedEvent, RuleDraft, SkillCandidate
from rulemine.promote import promote_corrections, promote_decisions, promote_skills
from rulemine.redact import redact_secrets
from rulemine.render import render_decision_markdown, render_rules_block, render_skill_draft, update_agents_markdown
from rulemine.store import (
    DEFAULT_DB_PATH,
    compute_transcript_hash,
    get_drafts,
    get_events,
    get_store_stats,
    init_db,
    save_drafts,
    save_events,
    save_session,
    should_remine,
    update_draft_status,
)
from rulemine.window import window_turns

app = typer.Typer(
    name="rulemine",
    help="Mine local agent transcripts for rules, decisions, and portable skills using local Gemma.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def mine(
    since: str = typer.Option("7d", help="Window to scan, e.g. 7d, 24h, or YYYY-MM-DD"),
    model: str = typer.Option("gemma2:2b", help="Ollama model name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan and window transcripts without calling the model"),
    workspace: Optional[str] = typer.Option(None, help="Filter transcripts by workspace directory"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of sessions to process"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
    host: str = typer.Option("http://127.0.0.1:11434", "--host", help="Ollama host URL"),
    base_dir: Optional[Path] = typer.Option(None, "--base-dir", help="Base directory for agent session logs"),
    adapter: str = typer.Option("auto", "--adapter", "-a", help="Agent adapter: 'antigravity', 'cursor', or 'auto'/'all'"),
    timeout: float = typer.Option(180.0, "--timeout", "-t", help="Ollama socket read / idle timeout in seconds (default: 180s)"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error", help="Skip failing sessions and continue mining remaining sessions"),
    keep_alive: str = typer.Option("30m", "--keep-alive", help="Ollama model keep-alive in memory (e.g. 30m, 1h)"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Fast demo mode: process only top 3 sessions with compact few-shots"),
    no_filter: bool = typer.Option(False, "--no-filter", help="Disable candidate gating and send all windows to LLM"),
    max_fewshots: int = typer.Option(2, "--max-fewshots", help="Maximum few-shot examples in prompt (default: 2)"),
) -> None:
    """Mine local agent transcripts for corrections, decisions, and skill candidates."""
    try:
        since_dt = parse_since(since)
    except ValueError as e:
        console.print(f"[bold red]Error parsing --since option:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print(
        f"[bold green]RuleMine[/bold green] scanning transcripts since [cyan]{since}[/cyan]"
        + (f" in workspace [cyan]{workspace}[/cyan]" if workspace else "")
        + f" using [magenta]{model}[/magenta]"
        + (f" via adapter [yellow]{adapter}[/yellow]" if adapter != "auto" else "")
    )

    # Resolve sessions based on adapter selection
    sessions: list[SessionRef] = []
    adapter_lower = adapter.lower()

    if adapter_lower == "cursor":
        sessions = discover_cursor_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir)
    elif adapter_lower == "antigravity":
        sessions = discover_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir)
    else:  # "auto" or "all"
        if base_dir:
            if (base_dir / "brain").is_dir() or (base_dir / "conversation_summaries.db").is_file():
                sessions.extend(discover_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir))
            elif (base_dir / "workspaceStorage").is_dir() or (base_dir / "globalStorage").is_dir() or base_dir.name == "state.vscdb":
                sessions.extend(discover_cursor_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir))
            else:
                sessions.extend(discover_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir))
                sessions.extend(discover_cursor_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir))
        else:
            sessions.extend(discover_sessions(since=since_dt, workspace_filter=workspace))
            sessions.extend(discover_cursor_sessions(since=since_dt, workspace_filter=workspace))

    # Sort combined sessions descending by mtime
    sessions.sort(key=lambda s: s.mtime or 0.0, reverse=True)
    if not sessions:
        console.print("[yellow]No conversation sessions found matching criteria.[/yellow]")
        return

    if quick and not limit:
        limit = 3
        max_fewshots = 1

    if limit and limit > 0:
        sessions = sessions[:limit]

    # In dry-run mode: scan and display windowing stats without calling model or writing DB
    if dry_run:
        table = Table(title=f"RuleMine Dry-Run Ingest: {len(sessions)} Discovered Session(s)")
        table.add_column("Session ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white")
        table.add_column("Workspace", style="green")
        table.add_column("Turns", justify="right")
        table.add_column("Windows", justify="right")
        table.add_column("Redacted", justify="right")

        total_turns = 0
        total_windows = 0
        total_redacted = 0

        for session in sessions:
            turns = load_turns(session)
            windows = window_turns(
                turns=turns,
                session_id=session.session_id,
                workspace=session.workspace,
            )

            session_redacted = sum(redact_secrets(t.content)[1] for t in turns)

            total_turns += len(turns)
            total_windows += len(windows)
            total_redacted += session_redacted

            short_id = session.session_id[:8] + "..." if len(session.session_id) > 12 else session.session_id
            short_title = (session.title or "(untitled)")[:35]
            ws_display = Path(session.workspace).name if session.workspace else "(none)"

            table.add_row(
                short_id,
                short_title,
                ws_display,
                str(len(turns)),
                str(len(windows)),
                str(session_redacted),
            )

        console.print(table)

        summary_text = (
            f"[bold]Sessions:[/bold] {len(sessions)}  |  "
            f"[bold]Turns:[/bold] {total_turns}  |  "
            f"[bold]Windows:[/bold] {total_windows}  |  "
            f"[bold]Secrets Redacted:[/bold] {total_redacted}"
        )
        console.print(Panel(summary_text, title="[yellow]Dry-Run Summary[/yellow]", border_style="yellow"))
        console.print("[yellow]Dry-run complete: no model calls or file modifications were made.[/yellow]")
        return

    # Live extraction mode
    conn = init_db(db_path)

    sessions_skipped = 0
    sessions_mined = 0
    all_new_events: list[ExtractedEvent] = []
    events_by_type: dict[str, int] = {
        "correction": 0,
        "decision": 0,
        "dead_end": 0,
        "skill_need": 0,
    }

    console.print(f"Checking cache in [cyan]{db_path}[/cyan]...")

    timeout_config = httpx.Timeout(
        connect=10.0,
        read=timeout,
        write=10.0,
        pool=10.0,
    )

    with httpx.Client(base_url=host, timeout=timeout_config) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Mining {len(sessions)} session(s)...", total=len(sessions))

            for session in sessions:
                content_hash = compute_transcript_hash(session.transcript_path)

                if not should_remine(conn, session.session_id, content_hash, model=model):
                    sessions_skipped += 1
                    progress.advance(task)
                    continue

                turns = load_turns(session)
                windows = window_turns(
                    turns=turns,
                    session_id=session.session_id,
                    workspace=session.workspace,
                )

                session_events: list[ExtractedEvent] = []
                session_failed = False
                try:
                    for win in windows:
                        extracted = extract_events_for_window(
                            window=win,
                            model=model,
                            host=host,
                            timeout=timeout,
                            keep_alive=keep_alive,
                            max_fewshots=max_fewshots,
                            client=client,
                            filter_signals=not no_filter,
                        )
                        for ev in extracted:
                            if not ev.workspace and session.workspace:
                                ev.workspace = session.workspace
                            session_events.append(ev)
                            events_by_type[ev.type] = events_by_type.get(ev.type, 0) + 1
                except httpx.ConnectError:
                    progress.stop()
                    console.print(
                        f"\n[bold red]Error connecting to Ollama at {host}:[/bold red] "
                        f"Ensure Ollama is running ('ollama serve') and model '{model}' is installed."
                    )
                    raise typer.Exit(code=1)
                except httpx.TimeoutException as timeout_err:
                    if continue_on_error:
                        session_failed = True
                        console.print(
                            f"\n[yellow]Warning: session {session.session_id} timed out after {timeout:.0f}s, skipping...[/yellow]"
                        )
                    else:
                        progress.stop()
                        console.print(
                            f"\n[bold red]Ollama request timed out after {timeout:.0f}s[/bold red] while processing session [cyan]{session.session_id}[/cyan].\n\n"
                            f"[bold yellow]How to fix:[/bold yellow]\n"
                            f"  1. Increase the timeout: [cyan]rulemine mine --timeout {int(timeout * 2)}[/cyan]\n"
                            f"  2. Skip slow sessions and continue: [cyan]rulemine mine --continue-on-error[/cyan]\n"
                            f"  3. Check Ollama server load or GPU availability.\n\n"
                            f"[dim]Note: Already mined sessions ({sessions_mined} in this run, {sessions_skipped} cached) are saved in {db_path} and will not be re-mined.[/dim]"
                        )
                        raise typer.Exit(code=1)
                except httpx.HTTPError as http_err:
                    if continue_on_error:
                        session_failed = True
                        console.print(
                            f"\n[yellow]Warning: session {session.session_id} encountered HTTP error ({http_err}), skipping...[/yellow]"
                        )
                    else:
                        progress.stop()
                        console.print(f"\n[bold red]HTTP error calling Ollama:[/bold red] {http_err}")
                        raise typer.Exit(code=1)
                except Exception as err:
                    if continue_on_error:
                        session_failed = True
                        console.print(
                            f"\n[yellow]Warning: session {session.session_id} encountered error ({err}), skipping...[/yellow]"
                        )
                    else:
                        progress.stop()
                        console.print(f"\n[bold red]Error processing session {session.session_id}:[/bold red] {err}")
                        raise typer.Exit(code=1)

                if session_failed:
                    progress.advance(task)
                    continue

                # Persist extracted events and update session cache
                if session_events:
                    save_events(session_events, conn=conn)
                    all_new_events.extend(session_events)

                save_session(
                    conn=conn,
                    session_id=session.session_id,
                    transcript_path=session.transcript_path,
                    content_hash=content_hash,
                    workspace=session.workspace,
                    title=session.title,
                    model=model,
                )
                sessions_mined += 1
                progress.advance(task)

    # Post-run summary
    summary_lines = [
        f"[bold]Total Scanned:[/bold] {len(sessions)} session(s)",
        f"[bold]Newly Mined:[/bold] {sessions_mined} session(s)",
        f"[bold]Cached / Skipped:[/bold] {sessions_skipped} session(s)",
        f"[bold]New Events Extracted:[/bold] {len(all_new_events)}",
        f"  - [cyan]Corrections:[/cyan] {events_by_type.get('correction', 0)}",
        f"  - [green]Decisions:[/green] {events_by_type.get('decision', 0)}",
        f"  - [yellow]Dead Ends:[/yellow] {events_by_type.get('dead_end', 0)}",
        f"  - [magenta]Skill Needs:[/magenta] {events_by_type.get('skill_need', 0)}",
    ]
    console.print(Panel("\n".join(summary_lines), title="[bold green]Mining Summary[/bold green]", border_style="green"))

    if all_new_events:
        preview_table = Table(title="Newly Extracted Events Preview")
        preview_table.add_column("Type", style="bold")
        preview_table.add_column("Normalized", style="white")
        preview_table.add_column("Confidence", justify="right", style="cyan")
        preview_table.add_column("Session", style="dim")

        for ev in all_new_events[:10]:
            color = {
                "correction": "cyan",
                "decision": "green",
                "dead_end": "yellow",
                "skill_need": "magenta",
            }.get(ev.type, "white")
            preview_table.add_row(
                f"[{color}]{ev.type}[/{color}]",
                ev.normalized[:60] + ("..." if len(ev.normalized) > 60 else ""),
                f"{ev.confidence:.2f}",
                ev.session_id[:8] + "...",
            )
        console.print(preview_table)
        if len(all_new_events) > 10:
            console.print(f"[dim]... and {len(all_new_events) - 10} more event(s).[/dim]")

    console.print(
        "[bold blue]Next:[/bold blue] Run [cyan]rulemine status[/cyan] to view cached memory, "
        "or [cyan]rulemine promote[/cyan] to cluster events into draft rules, decisions, and skills."
    )


@app.command()
def status(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
) -> None:
    """Show current mining cache status and event counts."""
    if not db_path.is_file():
        console.print(f"[yellow]No store found at {db_path}.[/yellow] Run [cyan]rulemine mine[/cyan] first.")
        return

    stats = get_store_stats(db_path=db_path)
    last_mined = stats["last_mined_at"] or "Never"
    events_by_type = stats["events_by_type"]

    table = Table(title="RuleMine Cache Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="bold white")

    table.add_row("Cached Sessions", str(stats["session_count"]))
    table.add_row("Total Extracted Events", str(stats["event_count"]))
    table.add_row("  - Corrections", str(events_by_type.get("correction", 0)))
    table.add_row("  - Decisions", str(events_by_type.get("decision", 0)))
    table.add_row("  - Dead Ends", str(events_by_type.get("dead_end", 0)))
    table.add_row("  - Skill Needs", str(events_by_type.get("skill_need", 0)))
    table.add_row("Drafts", str(stats["draft_count"]))

    console.print(table)
    console.print(f"[dim]Database path: {db_path}[/dim]")
    console.print(f"[dim]Last mined at: {last_mined}[/dim]")


@app.command()
def promote(
    min_support: int = typer.Option(2, "--min-support", help="Minimum occurrences across sessions to promote a correction to a rule"),
    min_skill_support: int = typer.Option(1, "--min-skill-support", help="Minimum occurrences across sessions to promote a skill candidate"),
    min_workspaces: int = typer.Option(1, "--min-workspaces", help="Minimum distinct workspaces to promote a skill candidate"),
    min_decision_support: int = typer.Option(1, "--min-decision-support", help="Minimum occurrences across sessions to promote a decision/dead_end"),
    require_reason: bool = typer.Option(False, "--require-reason", help="Only promote decisions with an explicit rationale"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Filter events to a specific workspace directory (e.g. . for current repo)"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
) -> None:
    """Cluster extracted events into proposed rules, decisions, and skills."""
    if not db_path.is_file():
        console.print(f"[yellow]No store found at {db_path}.[/yellow] Run [cyan]rulemine mine[/cyan] first.")
        return

    events = get_events(db_path=db_path)
    if not events:
        console.print("[yellow]No extracted events found in database.[/yellow] Run [cyan]rulemine mine[/cyan] first.")
        return

    if workspace:
        target_path = Path(workspace).resolve()

        def _matches_workspace(ev_ws: str | None) -> bool:
            if not ev_ws:
                return False
            p = Path(ev_ws).resolve()
            return p == target_path or p in target_path.parents or target_path in p.parents or p.name == target_path.name

        events = [e for e in events if _matches_workspace(e.workspace)]
        console.print(f"Filtered to [cyan]{len(events)}[/cyan] event(s) matching workspace [cyan]{workspace}[/cyan]")
        if not events:
            console.print("[yellow]No matching events found for workspace.[/yellow]")
            return

    console.print(f"Running promotion algorithms on [cyan]{len(events)}[/cyan] extracted event(s)...")

    rules = promote_corrections(events, min_support=min_support)
    decisions = promote_decisions(
        events,
        min_support=min_decision_support,
        require_reason=require_reason,
    )
    skills = promote_skills(events, min_support=min_skill_support, min_workspaces=min_workspaces)

    all_drafts: list[RuleDraft | DecisionDraft | SkillCandidate] = list(rules) + list(decisions) + list(skills)
    save_drafts(all_drafts, db_path=db_path)

    # Print summary table
    table = Table(title="RuleMine Promotion Results")
    table.add_column("Draft Type", style="cyan")
    table.add_column("Proposed Count", justify="right", style="bold white")

    table.add_row("Rules (AGENTS.md)", str(len(rules)))
    table.add_row("Decisions / Dead Ends (DECISIONS.md)", str(len(decisions)))
    table.add_row("Skill Candidates (SKILL.md)", str(len(skills)))
    table.add_row("Total Proposed Drafts", str(len(all_drafts)))

    console.print(table)

    if all_drafts:
        preview_table = Table(title="Proposed Drafts Preview")
        preview_table.add_column("ID", style="dim", no_wrap=True)
        preview_table.add_column("Kind", style="bold")
        preview_table.add_column("Text / Name", style="white")
        preview_table.add_column("Support", justify="right", style="cyan")

        for d in all_drafts[:10]:
            if isinstance(d, RuleDraft):
                kind_str = "[cyan]rule[/cyan]"
                text_str = d.text
                sup_str = str(d.support_count)
            elif isinstance(d, DecisionDraft):
                kind_str = f"[green]{d.kind}[/green]"
                text_str = d.text
                sup_str = "1"
            else:
                kind_str = "[magenta]skill[/magenta]"
                text_str = d.name
                sup_str = str(d.support_count)

            preview_table.add_row(
                d.draft_id,
                kind_str,
                text_str[:60] + ("..." if len(text_str) > 60 else ""),
                sup_str,
            )
        console.print(preview_table)
        if len(all_drafts) > 10:
            console.print(f"[dim]... and {len(all_drafts) - 10} more proposed draft(s).[/dim]")

    console.print(
        "[bold blue]Next:[/bold blue] Run [cyan]rulemine review[/cyan] to inspect and approve candidates."
    )


@app.command()
def review(
    kind: Optional[str] = typer.Option(None, "--kind", "-k", help="Filter review to a specific kind: 'skill', 'rule', or 'decision'"),
    all: bool = typer.Option(False, "--all", "-a", help="Approve all proposed drafts without interactive prompts"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
) -> None:
    """Interactively review, approve, or reject proposed drafts."""
    if not db_path.is_file():
        console.print(f"[yellow]No store found at {db_path}.[/yellow] Run [cyan]rulemine mine[/cyan] first.")
        return

    proposed = get_drafts(status="proposed", db_path=db_path)
    if not proposed:
        console.print("[yellow]No pending drafts to review.[/yellow] Run [cyan]rulemine promote[/cyan] to discover candidates.")
        return

    if kind:
        kind_clean = kind.lower().strip()
        if kind_clean in ("skill", "skills"):
            proposed = [d for d in proposed if d["kind"] == "skill"]
        elif kind_clean in ("rule", "rules"):
            proposed = [d for d in proposed if d["kind"] == "rule"]
        elif kind_clean in ("decision", "decisions", "dead_end", "dead_ends"):
            proposed = [d for d in proposed if d["kind"] in ("decision", "dead_end")]
        if not proposed:
            console.print(f"[yellow]No pending drafts of kind '{kind}' found to review.[/yellow]")
            return

    if all:
        for d in proposed:
            update_draft_status(d["draft_id"], "approved", db_path=db_path)
        console.print(f"[bold green]Approved all {len(proposed)} proposed draft(s).[/bold green]")
        console.print("[bold blue]Next:[/bold blue] Run [cyan]rulemine write[/cyan] to write approved drafts.")
        return

    console.print(f"[bold]Reviewing {len(proposed)} proposed draft(s)...[/bold]")
    approved_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, d in enumerate(proposed, 1):
        content_text = d.get("text") or d.get("name") or "(untitled)"
        workspaces_str = ", ".join(d.get("workspaces", [])) or "(none)"
        panel_content = (
            f"[bold]Draft {i}/{len(proposed)}[/bold] [{d['kind']}] (ID: {d['draft_id']})\n"
            f"[cyan]Content:[/cyan] {content_text}\n"
            f"[dim]Support:[/dim] {d.get('support_count', 1)}  |  "
            f"[dim]Workspaces:[/dim] {workspaces_str}"
        )
        if d.get("reason"):
            panel_content += f"\n[dim]Reason:[/dim] {d['reason']}"
        if d.get("overlaps_existing"):
            panel_content += f"\n[yellow]Overlaps existing skills:[/yellow] {', '.join(d['overlaps_existing'])}"

        console.print(Panel(panel_content, title=f"Review Candidate ({d['kind']})"))

        try:
            choice = typer.prompt("[a]pprove / [r]eject / [s]kip / [q]uit", default="s").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Review interrupted.[/yellow]")
            break

        if choice == "a":
            update_draft_status(d["draft_id"], "approved", db_path=db_path)
            console.print("[green]Approved.[/green]")
            approved_count += 1
        elif choice == "r":
            update_draft_status(d["draft_id"], "rejected", db_path=db_path)
            console.print("[red]Rejected.[/red]")
            rejected_count += 1
        elif choice == "q":
            console.print("[yellow]Exiting review.[/yellow]")
            break
        else:
            console.print("[dim]Skipped.[/dim]")
            skipped_count += 1

    console.print(
        f"\n[bold]Review Summary:[/bold] "
        f"[green]{approved_count} approved[/green], "
        f"[red]{rejected_count} rejected[/red], "
        f"[dim]{skipped_count} skipped[/dim]."
    )
    if approved_count > 0:
        console.print("[bold blue]Next:[/bold blue] Run [cyan]rulemine write[/cyan] to write approved drafts.")


@app.command(name="write")
def write_drafts(
    rules: bool = typer.Option(False, "--rules", help="Write approved rules to AGENTS.md"),
    decisions: bool = typer.Option(False, "--decisions", help="Write approved decisions to DECISIONS.md"),
    skills: bool = typer.Option(False, "--skills", help="Write approved skill drafts"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview output without modifying files"),
    sidecar: bool = typer.Option(False, "--sidecar", help="Write rules to AGENTS.rulemine.md instead of modifying AGENTS.md"),
    out: Optional[Path] = typer.Option(None, help="Custom output directory for skills"),
    target_dir: Optional[Path] = typer.Option(None, "--target-dir", help="Directory where AGENTS.md / DECISIONS.md are located"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
) -> None:
    """Write approved drafts to repository rules, decision records, or skill files."""
    if not db_path.is_file():
        console.print(f"[yellow]No store found at {db_path}.[/yellow] Run [cyan]rulemine mine[/cyan] first.")
        return

    # If no specific target requested, handle all three
    if not (rules or decisions or skills):
        rules = True
        decisions = True
        skills = True

    approved_drafts = get_drafts(status="approved", db_path=db_path)
    if not approved_drafts:
        console.print("[yellow]No approved drafts found.[/yellow] Run [cyan]rulemine review[/cyan] to approve candidates.")
        return

    approved_rules = [RuleDraft.model_validate(d) for d in approved_drafts if d["kind"] == "rule"]
    approved_decisions = [DecisionDraft.model_validate(d) for d in approved_drafts if d["kind"] in ("decision", "dead_end")]
    approved_skills = [SkillCandidate.model_validate(d) for d in approved_drafts if d["kind"] == "skill"]

    has_work = (
        (rules and approved_rules)
        or (decisions and approved_decisions)
        or (skills and approved_skills)
    )
    if not has_work:
        console.print("[yellow]No approved drafts match the selected targets.[/yellow]")
        return

    base = target_dir or Path(".")

    if dry_run:
        console.print("[yellow]Dry-run mode: previewing file modifications without writing.[/yellow]\n")
        if rules and approved_rules:
            file_name = "AGENTS.rulemine.md" if sidecar else "AGENTS.md"
            block = render_rules_block(approved_rules)
            console.print(Panel(block, title=f"[cyan]Preview: {file_name} ({len(approved_rules)} rule(s))[/cyan]"))
        if decisions and approved_decisions:
            file_name = "DECISIONS.md"
            dec_text = "\n\n".join(render_decision_markdown(d) for d in approved_decisions)
            console.print(Panel(dec_text, title=f"[green]Preview: {file_name} ({len(approved_decisions)} decision(s))[/green]"))
        if skills and approved_skills:
            out_base = out or Path("out/skills")
            for s in approved_skills:
                skill_text = render_skill_draft(s)
                console.print(Panel(skill_text, title=f"[magenta]Preview: {out_base / s.name / 'SKILL.md'}[/magenta]"))
        console.print("[yellow]Dry-run complete: no files were modified. Pass --no-dry-run to write changes.[/yellow]")
        return

    # Non-dry-run: modify files
    if rules and approved_rules:
        target_file = base / ("AGENTS.rulemine.md" if sidecar else "AGENTS.md")
        rules_block = render_rules_block(approved_rules)
        if target_file.is_file():
            existing = target_file.read_text(encoding="utf-8")
            new_content = update_agents_markdown(existing, rules_block)
        else:
            new_content = rules_block + "\n"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(new_content, encoding="utf-8")
        console.print(f"[bold green]Wrote {len(approved_rules)} rule(s) to {target_file}[/bold green]")

    if decisions and approved_decisions:
        target_file = base / "DECISIONS.md"
        entries = [render_decision_markdown(d) for d in approved_decisions]
        append_text = "\n\n".join(entries) + "\n"
        if target_file.is_file():
            existing = target_file.read_text(encoding="utf-8").rstrip()
            new_content = existing + "\n\n" + append_text
        else:
            header = "# Architecture Decisions & Dead Ends\n\n"
            new_content = header + append_text
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(new_content, encoding="utf-8")
        console.print(f"[bold green]Wrote {len(approved_decisions)} decision record(s) to {target_file}[/bold green]")

    if skills and approved_skills:
        out_base = out or Path("out/skills")
        for s in approved_skills:
            skill_dir = out_base / s.name
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_file = skill_dir / "SKILL.md"
            skill_content = render_skill_draft(s)
            skill_file.write_text(skill_content, encoding="utf-8")
            console.print(f"[bold green]Wrote skill draft to {skill_file}[/bold green]")


@app.command()
def smoke(
    model: str = typer.Option("gemma2:2b", help="Ollama model name to smoke-test"),
    host: str = typer.Option("http://127.0.0.1:11434", help="Ollama host URL"),
    timeout: float = typer.Option(60.0, "--timeout", "-t", help="Ollama request timeout in seconds"),
) -> None:
    """Smoke-test Ollama structured JSON extraction with local Gemma."""
    from rulemine.extract import smoke_test_ollama

    console.print(f"[bold cyan]Running Ollama smoke test on {host} using {model}...[/bold cyan]")
    success, details = smoke_test_ollama(model=model, host=host, timeout=timeout)
    if success:
        console.print("[bold green]✓ Smoke test passed![/bold green]")
        console.print(details)
    else:
        console.print(f"[bold red]✗ Smoke test failed:[/bold red] {details}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
