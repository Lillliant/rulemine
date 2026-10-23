"""Render approved drafts into markdown formats (AGENTS.md, DECISIONS.md, SKILL.md)."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from rulemine.models import DecisionDraft, RuleDraft, SkillCandidate


def render_rules_block(rules: list[RuleDraft]) -> str:
    """Render rules section formatted for AGENTS.md."""
    lines = ["## Rules mined by RuleMine", "<!-- rulemine:start -->"]
    for r in rules:
        clean_text = r.text.strip().replace("—", "--").replace("–", "-")
        lines.append(f"- {clean_text}")
    lines.append("<!-- rulemine:end -->")
    return "\n".join(lines)


def update_agents_markdown(existing_content: str, rules_block: str) -> str:
    """Insert or replace RuleMine rules block in AGENTS.md content."""
    start_tag = "<!-- rulemine:start -->"
    end_tag = "<!-- rulemine:end -->"

    if start_tag in existing_content and end_tag in existing_content:
        pattern = re.compile(
            r"(?:## Rules mined by RuleMine\s*\n)?<!-- rulemine:start -->.*?<!-- rulemine:end -->",
            re.DOTALL,
        )
        return pattern.sub(rules_block, existing_content)
    else:
        trimmed = existing_content.strip()
        if trimmed:
            return f"{trimmed}\n\n{rules_block}\n"
        return f"{rules_block}\n"


def render_decision_markdown(decision: DecisionDraft) -> str:
    """Render decision or dead end into DECISIONS.md format."""
    # Determine YYYY-MM-DD
    date_str = None
    if decision.timestamp:
        # Match YYYY-MM-DD at start
        m = re.match(r"^(\d{4}-\d{2}-\d{2})", decision.timestamp)
        if m:
            date_str = m.group(1)

    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    clean_summary = decision.text.strip().replace("—", "--").replace("–", "-")
    lines = [
        f"## {date_str} -- {clean_summary}",
        f"- Kind: {decision.kind}",
        f"- Summary: {clean_summary}",
    ]
    if decision.reason:
        clean_reason = decision.reason.strip().replace("—", "--").replace("–", "-")
        lines.append(f"- Reason: {clean_reason}")
    else:
        lines.append("- Reason: none")

    commit_val = decision.git_sha.strip() if decision.git_sha else "none"
    lines.append(f"- Commit: {commit_val}")

    if decision.session_id:
        lines.append(f"- Sources: {decision.session_id}")
    elif decision.source_event_ids:
        lines.append(f"- Sources: {', '.join(decision.source_event_ids)}")
    else:
        lines.append("- Sources: none")

    return "\n".join(lines)


def render_skill_draft(skill: SkillCandidate) -> str:
    """Render portable skill candidate into SKILL.md format."""
    clean_desc = skill.description.strip().replace("—", "--").replace("–", "-")
    clean_trigger = skill.trigger.strip().replace("—", "--").replace("–", "-")

    lines = [
        "---",
        f"name: {skill.name}",
        "description: >-",
        f"  {clean_desc}",
        "---",
        "",
        f"# {skill.name}",
        "",
        "## When to use",
        clean_trigger,
        "",
        "## Steps",
    ]

    if skill.outline_sections:
        for i, step in enumerate(skill.outline_sections, 1):
            clean_step = step.strip().replace("—", "--").replace("–", "-")
            lines.append(f"{i}. {clean_step}")
    else:
        lines.append("1. Outline steps here.")

    lines.extend([
        "",
        "## Output / constraints",
        "Plain hyphens, no em dashes.",
    ])

    sources_str = ",".join(skill.source_event_ids) if skill.source_event_ids else "none"
    workspaces_str = ",".join(skill.workspaces) if skill.workspaces else "none"
    lines.append(f"<!-- rulemine: sources={sources_str} workspaces={workspaces_str} -->")
    lines.append("")

    return "\n".join(lines)
