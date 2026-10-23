"""Command-line interface for RuleMine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console

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
) -> None:
    """Mine local agent transcripts for corrections, decisions, and skill candidates."""
    console.print(f"[bold green]RuleMine[/bold green] mining transcripts since [cyan]{since}[/cyan] using [magenta]{model}[/magenta]")
    if dry_run:
        console.print("[yellow]Dry-run mode enabled: model inference will be skipped.[/yellow]")
    # Implementation wired in subsequent milestones


@app.command()
def status() -> None:
    """Show current mining cache status and proposed drafts."""
    console.print("[bold]RuleMine Status[/bold]")
    console.print("No store found yet. Run [cyan]rulemine mine[/cyan] first.")


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
