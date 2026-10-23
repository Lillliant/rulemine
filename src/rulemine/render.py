"""Render approved drafts into markdown formats (AGENTS.md, DECISIONS.md, SKILL.md)."""

from __future__ import annotations

from rulemine.models import DecisionDraft, RuleDraft, SkillCandidate


def render_rules_block(rules: list[RuleDraft]) -> str:
    """Render rules section formatted for AGENTS.md."""
    lines = ["## Rules mined by RuleMine", "<!-- rulemine:start -->"]
    for r in rules:
        lines.append(f"- {r.text}")
    lines.append("<!-- rulemine:end -->")
    return "\n".join(lines)


def render_decision_markdown(decision: DecisionDraft) -> str:
    """Render decision or dead end into DECISIONS.md format."""
    lines = [
        f"- Kind: {decision.kind}",
        f"- Summary: {decision.text}",
    ]
    if decision.reason:
        lines.append(f"- Reason: {decision.reason}")
    if decision.git_sha:
        lines.append(f"- Commit: {decision.git_sha}")
    return "\n".join(lines)


def render_skill_draft(skill: SkillCandidate) -> str:
    """Render portable skill candidate into SKILL.md format."""
    return f"""---
name: {skill.name}
description: >-
  {skill.description}
---

# {skill.name}

## When to use
{skill.trigger}

## Steps
1. Outline steps here.

## Output / constraints
Plain hyphens, no em dashes.
"""
