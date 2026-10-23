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
from rulemine.extract import extract_events_for_window
from rulemine.models import ExtractedEvent
from rulemine.redact import redact_secrets
from rulemine.store import (
    DEFAULT_DB_PATH,
    compute_transcript_hash,
    get_store_stats,
    init_db,
    save_events,
    save_session,
    should_remine,
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
    base_dir: Optional[Path] = typer.Option(None, "--base-dir", help="Base directory for Antigravity sessions"),
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
    )

    sessions = discover_sessions(since=since_dt, workspace_filter=workspace, base_dir=base_dir)
    if not sessions:
        console.print("[yellow]No conversation sessions found matching criteria.[/yellow]")
        return

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
            try:
                for win in windows:
                    extracted = extract_events_for_window(
                        window=win,
                        model=model,
                        host=host,
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
            except httpx.HTTPError as http_err:
                progress.stop()
                console.print(f"\n[bold red]HTTP error calling Ollama:[/bold red] {http_err}")
                raise typer.Exit(code=1)

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
        "or [cyan]rulemine review[/cyan] to inspect candidates for AGENTS.md, DECISIONS.md, and skills."
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
def review() -> None:
    """Interactively review, approve, or reject proposed drafts."""
    console.print("[bold]Interactive Review[/bold]")
    console.print("No pending drafts to review. Run [cyan]rulemine mine[/cyan] to discover candidates.")


@app.command(name="write")
def write_drafts(
    rules: bool = typer.Option(False, "--rules", help="Write approved rules to AGENTS.md"),
    decisions: bool = typer.Option(False, "--decisions", help="Write approved decisions to DECISIONS.md"),
    skills: bool = typer.Option(False, "--skills", help="Write approved skill drafts"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Preview output without modifying files"),
    sidecar: bool = typer.Option(False, "--sidecar", help="Write rules to AGENTS.rulemine.md instead of modifying AGENTS.md"),
    out: Optional[Path] = typer.Option(None, help="Custom output directory for skills"),
) -> None:
    """Write approved drafts to repository rules, decision records, or skill files."""
    console.print("[bold]RuleMine Writer[/bold]")
    if dry_run:
        console.print("[yellow]Dry-run mode: no files will be written.[/yellow]")


@app.command()
def smoke(
    model: str = typer.Option("gemma2:2b", help="Ollama model name to smoke-test"),
    host: str = typer.Option("http://127.0.0.1:11434", help="Ollama host URL"),
) -> None:
    """Smoke-test Ollama structured JSON extraction with local Gemma."""
    from rulemine.extract import smoke_test_ollama

    console.print(f"[bold cyan]Running Ollama smoke test on {host} using {model}...[/bold cyan]")
    success, details = smoke_test_ollama(model=model, host=host)
    if success:
        console.print("[bold green]✓ Smoke test passed![/bold green]")
        console.print(details)
    else:
        console.print(f"[bold red]✗ Smoke test failed:[/bold red] {details}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
