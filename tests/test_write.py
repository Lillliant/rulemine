"""Integration tests for rulemine promote, review, and write CLI workflows."""

from pathlib import Path
from typer.testing import CliRunner

from rulemine.cli import app
from rulemine.models import ExtractedEvent
from rulemine.store import get_drafts, init_db, save_events

runner = CliRunner()


def _populate_test_events(db_path: Path):
    """Seed test events into SQLite for CLI testing."""
    ev_corr1 = ExtractedEvent(
        type="correction",
        quote="Use npm instead of pnpm",
        normalized="Use npm, not pnpm in this repository",
        confidence=0.9,
        session_id="sess-1",
        step_index=1,
        workspace="/workspace/app",
    )
    ev_corr2 = ExtractedEvent(
        type="correction",
        quote="Remember to run npm here",
        normalized="Use npm, not pnpm in this repository",
        confidence=0.88,
        session_id="sess-2",
        step_index=3,
        workspace="/workspace/app",
    )
    ev_dec = ExtractedEvent(
        type="decision",
        quote="We chose SQLite for local cache",
        normalized="Use SQLite for local metadata cache",
        reason="Zero-configuration embedded DB",
        confidence=0.85,
        session_id="sess-1",
        step_index=2,
        workspace="/workspace/app",
        timestamp="2026-10-23T10:00:00Z",
    )
    ev_skill = ExtractedEvent(
        type="skill_need",
        quote="Every time we cut a release, update changelog and bump version",
        normalized="Cut release and update changelog",
        trigger="Every time a release tag is created",
        steps_hint=["Update changelog", "Bump version"],
        confidence=0.9,
        session_id="sess-3",
        step_index=1,
        workspace="/workspace/app",
    )
    save_events([ev_corr1, ev_corr2, ev_dec, ev_skill], db_path=db_path)


def test_cli_promote_and_review_all(tmp_path: Path):
    db_file = tmp_path / "rulemine.sqlite"
    _populate_test_events(db_file)

    # 1. Run promote
    result = runner.invoke(app, ["promote", "--db", str(db_file), "--min-support", "2"])
    assert result.exit_code == 0
    assert "RuleMine Promotion Results" in result.output
    assert "Proposed Count" in result.output

    # Check store has proposed drafts
    drafts = get_drafts(db_path=db_file)
    assert len(drafts) >= 3
    assert all(d["status"] == "proposed" for d in drafts)

    # 2. Review --all
    result_review = runner.invoke(app, ["review", "--all", "--db", str(db_file)])
    assert result_review.exit_code == 0
    assert "Approved all" in result_review.output

    # Check store has approved drafts
    approved = get_drafts(status="approved", db_path=db_file)
    assert len(approved) == len(drafts)


def test_cli_write_dry_run_vs_no_dry_run(tmp_path: Path):
    db_file = tmp_path / "rulemine.sqlite"
    _populate_test_events(db_file)

    # Promote and approve all
    runner.invoke(app, ["promote", "--db", str(db_file)])
    runner.invoke(app, ["review", "--all", "--db", str(db_file)])

    target_dir = tmp_path / "target_repo"
    target_dir.mkdir()
    skills_out = tmp_path / "skills_staging"

    # 1. Dry run should NOT write any files
    res_dry = runner.invoke(
        app,
        [
            "write",
            "--db", str(db_file),
            "--target-dir", str(target_dir),
            "--out", str(skills_out),
            "--dry-run",
        ],
    )
    assert res_dry.exit_code == 0
    assert "Dry-run mode" in res_dry.output
    assert not (target_dir / "AGENTS.md").exists()
    assert not (target_dir / "DECISIONS.md").exists()
    assert not skills_out.exists()

    # 2. No dry run writes files
    res_write = runner.invoke(
        app,
        [
            "write",
            "--db", str(db_file),
            "--target-dir", str(target_dir),
            "--out", str(skills_out),
            "--no-dry-run",
        ],
    )
    assert res_write.exit_code == 0
    assert "Wrote" in res_write.output

    agents_md = target_dir / "AGENTS.md"
    assert agents_md.exists()
    agents_text = agents_md.read_text(encoding="utf-8")
    assert "## Rules mined by RuleMine" in agents_text
    assert "<!-- rulemine:start -->" in agents_text
    assert "Use npm, not pnpm in this repository" in agents_text

    decisions_md = target_dir / "DECISIONS.md"
    assert decisions_md.exists()
    dec_text = decisions_md.read_text(encoding="utf-8")
    assert "## 2026-10-23 -- Use SQLite for local metadata cache" in dec_text
    assert "- Kind: decision" in dec_text

    skill_file = skills_out / "cut-release-and-update-changelog" / "SKILL.md"
    assert skill_file.exists()
    skill_text = skill_file.read_text(encoding="utf-8")
    assert "name: cut-release-and-update-changelog" in skill_text


def test_cli_write_sidecar_and_selective(tmp_path: Path):
    db_file = tmp_path / "rulemine.sqlite"
    _populate_test_events(db_file)

    runner.invoke(app, ["promote", "--db", str(db_file)])
    runner.invoke(app, ["review", "--all", "--db", str(db_file)])

    target_dir = tmp_path / "target_repo2"
    target_dir.mkdir()

    # Write only rules with sidecar
    res = runner.invoke(
        app,
        [
            "write",
            "--rules",
            "--sidecar",
            "--no-dry-run",
            "--db", str(db_file),
            "--target-dir", str(target_dir),
        ],
    )
    assert res.exit_code == 0
    assert (target_dir / "AGENTS.rulemine.md").exists()
    assert not (target_dir / "AGENTS.md").exists()
    assert not (target_dir / "DECISIONS.md").exists()


def test_cli_review_interactive(tmp_path: Path):
    db_file = tmp_path / "rulemine_interactive.sqlite"
    _populate_test_events(db_file)

    runner.invoke(app, ["promote", "--db", str(db_file)])

    # Interactive input: 'a' for first, 'r' for second, 'q' to quit
    res = runner.invoke(app, ["review", "--db", str(db_file)], input="a\nr\nq\n")
    assert res.exit_code == 0
    assert "Review Candidate" in res.output
    assert "1 approved" in res.output
    assert "1 rejected" in res.output

    approved = get_drafts(status="approved", db_path=db_file)
    rejected = get_drafts(status="rejected", db_path=db_file)
    assert len(approved) == 1
    assert len(rejected) == 1
