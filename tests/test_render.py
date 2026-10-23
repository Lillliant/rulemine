"""Unit tests for markdown rendering (rules, decisions, and portable skills)."""

from rulemine.models import DecisionDraft, RuleDraft, SkillCandidate
from rulemine.render import (
    render_decision_markdown,
    render_rules_block,
    render_skill_draft,
    update_agents_markdown,
)


def test_render_rules_block():
    rules = [
        RuleDraft(text="Use npm, not pnpm in this repository", support_count=2),
        RuleDraft(text="Use plain hyphens -- never em dashes", support_count=3),
    ]
    rendered = render_rules_block(rules)

    assert "## Rules mined by RuleMine" in rendered
    assert "<!-- rulemine:start -->" in rendered
    assert "<!-- rulemine:end -->" in rendered
    assert "- Use npm, not pnpm in this repository" in rendered
    assert "- Use plain hyphens -- never em dashes" in rendered
    assert "—" not in rendered  # No em dashes
    assert "–" not in rendered  # No en dashes


def test_update_agents_markdown_append_and_replace():
    rules = [RuleDraft(text="Use npm, not pnpm", support_count=2)]
    rules_block = render_rules_block(rules)

    # 1. Append to existing markdown that doesn't have rulemine block
    original = "# My Project Rules\n\n- Existing rule 1\n- Existing rule 2\n"
    updated = update_agents_markdown(original, rules_block)
    assert "# My Project Rules" in updated
    assert "- Existing rule 1" in updated
    assert "<!-- rulemine:start -->" in updated
    assert "- Use npm, not pnpm" in updated

    # 2. Replace existing block cleanly
    new_rules = [RuleDraft(text="Use npm strictly", support_count=4)]
    new_block = render_rules_block(new_rules)
    replaced = update_agents_markdown(updated, new_block)

    assert replaced.count("<!-- rulemine:start -->") == 1
    assert replaced.count("<!-- rulemine:end -->") == 1
    assert replaced.count("## Rules mined by RuleMine") == 1
    assert "- Use npm strictly" in replaced
    assert "- Use npm, not pnpm" not in replaced


def test_render_decision_markdown():
    dec = DecisionDraft(
        text="Use SQLite for local metadata cache",
        kind="decision",
        reason="Zero config embedded database",
        git_sha="1a2b3c4",
        timestamp="2026-10-23T11:00:00Z",
        session_id="sess-12345",
    )
    rendered = render_decision_markdown(dec)

    assert "## 2026-10-23 -- Use SQLite for local metadata cache" in rendered
    assert "- Kind: decision" in rendered
    assert "- Summary: Use SQLite for local metadata cache" in rendered
    assert "- Reason: Zero config embedded database" in rendered
    assert "- Commit: 1a2b3c4" in rendered
    assert "- Sources: sess-12345" in rendered
    assert "—" not in rendered


def test_render_skill_draft():
    skill = SkillCandidate(
        name="log-task-to-obsidian",
        description="Automate logging completed tasks to Obsidian daily note.",
        trigger="When a task milestone is completed.",
        outline_sections=[
            "Find today's daily note",
            "Format task entry with timestamp and tags",
            "Append entry to Daily Note under Today's Tasks",
        ],
        support_count=2,
        workspaces=["/workspace/repo1", "/workspace/repo2"],
        source_event_ids=["ev1", "ev2"],
    )
    rendered = render_skill_draft(skill)

    # Frontmatter
    assert rendered.startswith("---\nname: log-task-to-obsidian\n")
    assert "description: >-\n  Automate logging completed tasks" in rendered
    assert "---\n\n# log-task-to-obsidian" in rendered
    assert "## When to use\nWhen a task milestone is completed." in rendered
    assert "## Steps" in rendered
    assert "1. Find today's daily note" in rendered
    assert "2. Format task entry with timestamp and tags" in rendered
    assert "3. Append entry to Daily Note under Today's Tasks" in rendered
    assert "## Output / constraints\nPlain hyphens, no em dashes." in rendered
    assert "<!-- rulemine: sources=ev1,ev2 workspaces=/workspace/repo1,/workspace/repo2 -->" in rendered
    assert "—" not in rendered
