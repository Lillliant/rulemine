"""Unit tests for promotion logic (corrections -> rules, decisions, skills)."""

from pathlib import Path
from rulemine.models import ExtractedEvent
from rulemine.promote import promote_corrections, promote_decisions, promote_skills


def test_promote_corrections_min_support():
    ev1 = ExtractedEvent(
        type="correction",
        quote="Do not use yarn here, use npm",
        normalized="Use npm, not yarn in this repo",
        confidence=0.85,
        session_id="sess-1",
        step_index=2,
        workspace="/workspace/repo1",
    )
    ev2 = ExtractedEvent(
        type="correction",
        quote="Remember to run npm install instead of yarn",
        normalized="Use npm, not yarn in this repo",
        confidence=0.90,
        session_id="sess-2",
        step_index=4,
        workspace="/workspace/repo1",
    )
    ev_isolated = ExtractedEvent(
        type="correction",
        quote="Format with prettier",
        normalized="Run prettier before committing",
        confidence=0.8,
        session_id="sess-3",
        step_index=1,
        workspace="/workspace/repo2",
    )
    ev_low_conf = ExtractedEvent(
        type="correction",
        quote="Maybe try pytest?",
        normalized="Use pytest for testing",
        confidence=0.45,  # Below 0.6 threshold
        session_id="sess-4",
        step_index=1,
    )

    # 1. Single correction should not promote with min_support=2
    rules_single = promote_corrections([ev_isolated], min_support=2)
    assert len(rules_single) == 0

    # 2. Low confidence should never promote
    rules_low = promote_corrections([ev_low_conf, ev_low_conf], min_support=2)
    assert len(rules_low) == 0

    # 3. Two matching corrections promote to 1 rule with support=2
    rules = promote_corrections([ev1, ev2, ev_isolated], min_support=2)
    assert len(rules) == 1
    rule = rules[0]
    assert rule.support_count == 2
    assert rule.text == "Use npm, not yarn in this repo"
    assert "/workspace/repo1" in rule.workspaces
    assert ev1.event_id in rule.source_event_ids
    assert ev2.event_id in rule.source_event_ids
    assert rule.status == "proposed"


def test_promote_corrections_clustering_fuzzy():
    # Fuzzy match via token Jaccard / SequenceMatcher
    ev1 = ExtractedEvent(
        type="correction",
        quote="Never use em dashes in notes",
        normalized="Never use em dashes in documentation",
        confidence=0.8,
        session_id="sess-1",
        workspace="/workspace/docs",
    )
    ev2 = ExtractedEvent(
        type="correction",
        quote="Please avoid em dashes in documentation",
        normalized="Never use em dashes in documentation",
        confidence=0.95,
        session_id="sess-2",
        workspace="/workspace/docs",
    )

    rules = promote_corrections([ev1, ev2], min_support=2)
    assert len(rules) == 1
    assert rules[0].support_count == 2
    assert "Never use em dashes in documentation" in rules[0].text


def test_promote_decisions():
    ev_dec = ExtractedEvent(
        type="decision",
        quote="We chose SQLite for zero configuration local storage",
        normalized="Use SQLite for local metadata caching",
        reason="Zero configuration and lightweight embedded database",
        confidence=0.88,
        session_id="sess-1",
        workspace="/workspace/app",
        timestamp="2026-10-22T14:30:00Z",
    )
    ev_dead = ExtractedEvent(
        type="dead_end",
        quote="Docker compose root mounting caused permission issues",
        normalized="Avoid root volume bind mounts in test containers",
        reason="Container user permission mismatch on macOS",
        confidence=0.82,
        session_id="sess-1",
        workspace="/workspace/app",
        timestamp="2026-10-22T15:00:00Z",
    )
    ev_low = ExtractedEvent(
        type="decision",
        quote="Uncertain about caching layer",
        normalized="Maybe cache responses",
        confidence=0.5,
        session_id="sess-2",
    )

    drafts = promote_decisions([ev_dec, ev_dead, ev_low], min_confidence=0.6)
    assert len(drafts) == 2

    dec_draft = next(d for d in drafts if d.kind == "decision")
    assert dec_draft.text == "Use SQLite for local metadata caching"
    assert dec_draft.reason == "Zero configuration and lightweight embedded database"
    assert dec_draft.status == "proposed"

    dead_draft = next(d for d in drafts if d.kind == "dead_end")
    assert dead_draft.text == "Avoid root volume bind mounts in test containers"


def test_promote_decisions_deduplication():
    # Duplicate decision in same workspace/session across different steps
    ev1 = ExtractedEvent(
        type="decision",
        quote="Use SQLite",
        normalized="Use SQLite for local caching",
        reason="Speed",
        confidence=0.8,
        session_id="sess-1",
        step_index=2,
        workspace="/workspace/app",
    )
    ev2 = ExtractedEvent(
        type="decision",
        quote="Decided on SQLite",
        normalized="Use SQLite for local caching",
        reason="Speed and simplicity",
        confidence=0.9,
        session_id="sess-1",
        step_index=5,
        workspace="/workspace/app",
    )

    drafts = promote_decisions([ev1, ev2])
    assert len(drafts) == 1
    assert len(drafts[0].source_event_ids) == 2


def test_promote_skills_multi_workspace():
    ev1 = ExtractedEvent(
        type="skill_need",
        quote="We need to log task updates to the Obsidian daily note",
        normalized="Log completed task to central Obsidian vault",
        trigger="When a task milestone is completed",
        steps_hint=["Locate daily note", "Format task item with tags", "Append to note"],
        confidence=0.85,
        session_id="sess-1",
        workspace="/workspace/repo1",
    )
    ev2 = ExtractedEvent(
        type="skill_need",
        quote="Log this completed milestone into my Obsidian daily note",
        normalized="Log completed task to central Obsidian vault",
        trigger="When a task milestone is completed",
        steps_hint=["Verify daily note exists", "Append task line"],
        confidence=0.90,
        session_id="sess-2",
        workspace="/workspace/repo2",
    )

    skills = promote_skills([ev1, ev2])
    assert len(skills) == 1
    sk = skills[0]
    assert sk.name == "log-completed-task-to-central-obsidian-vault"
    assert sk.support_count == 2
    assert len(sk.workspaces) == 2
    assert len(sk.outline_sections) >= 3
    assert sk.status == "proposed"


def test_promote_skills_heuristics(tmp_path: Path):
    # 1. Single workspace without strong language -> Not promoted
    ev_ordinary = ExtractedEvent(
        type="skill_need",
        quote="Build docker image for this repository",
        normalized="Build Docker container",
        trigger="Before deploying service",
        confidence=0.8,
        session_id="sess-1",
        workspace="/workspace/repo1",
    )
    assert len(promote_skills([ev_ordinary])) == 0

    # 2. Single workspace WITH "every time" / "whenever" -> Promoted!
    ev_strong = ExtractedEvent(
        type="skill_need",
        quote="Every time we cut a release, run changelog generator and bump pyproject",
        normalized="Cut release and update changelog",
        trigger="Every time a release tag is created",
        steps_hint=["Generate changelog", "Bump version in pyproject.toml", "Create git tag"],
        confidence=0.9,
        session_id="sess-2",
        workspace="/workspace/repo1",
    )
    promoted = promote_skills([ev_strong])
    assert len(promoted) == 1
    assert promoted[0].name == "cut-release-and-update-changelog"

    # 3. Overlap check with existing skill
    skills_dir = tmp_path / "skills"
    existing_skill = skills_dir / "release-updater"
    existing_skill.mkdir(parents=True)
    (existing_skill / "SKILL.md").write_text(
        "---\nname: cut-release-and-update-changelog\ndescription: Cut release and update changelog\n---\n# skill",
        encoding="utf-8",
    )

    promoted_with_overlap = promote_skills([ev_strong], existing_skills_dir=skills_dir)
    assert len(promoted_with_overlap) == 1
    assert "cut-release-and-update-changelog" in promoted_with_overlap[0].overlaps_existing
