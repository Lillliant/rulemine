"""Command-line interface for RuleMine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rulemine.adapters.antigravity import discover_sessions, load_turns, parse_since
from rulemine.redact import redact_secrets
from rulemine.store import DEFAULT_DB_PATH, init_db
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
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
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

    sessions = discover_sessions(since=since_dt, workspace_filter=workspace)
    if not sessions:
        console.print("[yellow]No conversation sessions found matching criteria.[/yellow]")
        return

    table = Table(title=f"RuleMine Ingest: {len(sessions)} Discovered Sessions")
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

    if dry_run:
        console.print(Panel(summary_text, title="[yellow]Dry-Run Summary[/yellow]", border_style="yellow"))
        console.print("[yellow]Dry-run complete: no model calls or file modifications were made.[/yellow]")
    else:
        console.print(Panel(summary_text, title="[green]Ingest Summary[/green]", border_style="green"))
        console.print("[blue]Ingest complete. Run with extraction in Milestone 3 to mine rules.[/blue]")


@app.command()
def status(
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite cache database"),
) -> None:
    """Show current mining cache status and proposed drafts."""
    console.print("[bold]RuleMine Status[/bold]")
    if not db_path.is_file():
        console.print(f"No store found at {db_path}. Run [cyan]rulemine mine[/cyan] first.")
        return

    conn = init_db(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    session_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM events")
    event_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM drafts")
    draft_count = cursor.fetchone()[0]

    console.print(f"Database: [cyan]{db_path}[/cyan]")
    console.print(f"  Sessions cached: {session_count}")
    console.print(f"  Events extracted: {event_count}")
    console.print(f"  Drafts stored: {draft_count}")


@app.command()
def review() -> None:
    """Interactively review, approve, or reject proposed drafts."""
    console.print("[bold]Interactive Review[/bold]")
    console.print("No pending drafts to review.")


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
