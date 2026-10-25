"""Promotion logic: clustering extracted events into RuleDrafts, DecisionDrafts, and SkillCandidates."""

from __future__ import annotations

import difflib
from pathlib import Path
import re

from rulemine.gitlink import find_nearby_commit
from rulemine.models import DecisionDraft, ExtractedEvent, RuleDraft, SkillCandidate


def _normalize_text(text: str) -> str:
    """Lowercase, strip non-alphanumeric punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _similarity_match(norm1: str, norm2: str) -> bool:
    """Check if two normalized strings match by exact, token Jaccard (>=0.85), or SequenceMatcher (>=0.88)."""
    if norm1 == norm2:
        return True

    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    if tokens1 and tokens2:
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        if jaccard >= 0.85:
            return True

    ratio = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    if ratio >= 0.88:
        return True

    return False


def _decision_fuzzy_match(norm1: str, norm2: str) -> bool:
    """Check if two decisions share core intent, substring containment, or moderate token overlap."""
    if not norm1 or not norm2:
        return False
    if norm1 == norm2 or norm1 in norm2 or norm2 in norm1:
        return True
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    if tokens1 and tokens2:
        jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        if jaccard >= 0.45:
            return True
    return difflib.SequenceMatcher(None, norm1, norm2).ratio() >= 0.65


def _to_kebab_case(text: str, max_len: int = 64) -> str:
    """Convert text to kebab-case <= max_len characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if len(text) <= max_len:
        return text or "skill"
    # Truncate at last hyphen before max_len
    truncated = text[:max_len]
    if "-" in truncated:
        truncated = truncated.rsplit("-", 1)[0]
    return truncated.strip("-") or "skill"


def _load_existing_skills(existing_skills_dir: Path | None = None) -> list[tuple[str, str]]:
    """Scan existing skills for name and description."""
    results: list[tuple[str, str]] = []
    search_dirs: list[Path] = []
    if existing_skills_dir:
        search_dirs.append(Path(existing_skills_dir))
    search_dirs.append(Path.home() / ".agents" / "skills")
    search_dirs.append(Path(".agents/skills"))

    for base_dir in search_dirs:
        if not base_dir.is_dir():
            continue
        for path in base_dir.glob("**/SKILL.md"):
            try:
                content = path.read_text(encoding="utf-8")
                m = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                if m:
                    fm = m.group(1)
                    name_m = re.search(r"^name:\s*([^\n]+)", fm, re.MULTILINE)
                    name = name_m.group(1).strip() if name_m else path.parent.name
                    desc_m = re.search(r"^description:\s*(?:>-\s*)?\n?([^\n#]+)", fm, re.MULTILINE)
                    desc = desc_m.group(1).strip() if desc_m else ""
                    results.append((name, desc))
            except Exception:
                continue
    return results


def _check_overlap(name: str, desc: str, existing_skills: list[tuple[str, str]]) -> list[str]:
    """Check proposed skill candidate against existing skill names and descriptions."""
    overlaps: list[str] = []
    norm_name = _normalize_text(name)
    norm_desc = _normalize_text(desc)
    name_tokens = set(norm_name.split())

    for ex_name, ex_desc in existing_skills:
        ex_norm_name = _normalize_text(ex_name)
        ex_norm_desc = _normalize_text(ex_desc)
        ex_tokens = set(ex_norm_name.split())

        if norm_name == ex_norm_name:
            overlaps.append(ex_name)
            continue

        if difflib.SequenceMatcher(None, norm_name, ex_norm_name).ratio() >= 0.75:
            overlaps.append(ex_name)
            continue

        if name_tokens and ex_tokens:
            jaccard = len(name_tokens & ex_tokens) / len(name_tokens | ex_tokens)
            if jaccard >= 0.6:
                overlaps.append(ex_name)
                continue

        if norm_desc and ex_norm_desc:
            if difflib.SequenceMatcher(None, norm_desc, ex_norm_desc).ratio() >= 0.65:
                overlaps.append(ex_name)
                continue

    return sorted(list(set(overlaps)))


def promote_corrections(events: list[ExtractedEvent], min_support: int = 2) -> list[RuleDraft]:
    """Cluster correction events into proposed rules meeting minimum support threshold."""
    # 1. Filter type == correction and confidence >= 0.6
    filtered = [e for e in events if e.type == "correction" and e.confidence >= 0.6]
    if not filtered:
        return []

    # 2. Cluster
    clusters: list[list[ExtractedEvent]] = []
    for ev in filtered:
        norm = _normalize_text(ev.normalized)
        matched_cluster = None

        # Try exact match first across cluster representatives
        for c in clusters:
            rep_norm = _normalize_text(c[0].normalized)
            if norm == rep_norm:
                matched_cluster = c
                break

        # Then try similarity match (token Jaccard >= 0.85 or SequenceMatcher >= 0.88)
        if matched_cluster is None:
            for c in clusters:
                if any(_similarity_match(norm, _normalize_text(member.normalized)) for member in c):
                    matched_cluster = c
                    break

        if matched_cluster is not None:
            matched_cluster.append(ev)
        else:
            clusters.append([ev])

    # 3. Filter clusters by min_support
    rules: list[RuleDraft] = []
    for c in clusters:
        if len(c) < min_support:
            continue

        # Choose best normalized phrased as imperative rule (highest confidence, then longest)
        best_event = max(c, key=lambda e: (e.confidence, len(e.normalized)))
        raw_text = best_event.normalized.strip().rstrip(".")
        if raw_text:
            rule_text = raw_text[0].upper() + raw_text[1:]
        else:
            rule_text = raw_text
        rule_text = rule_text.replace("—", "--").replace("–", "-")

        workspaces = sorted(list({e.workspace for e in c if e.workspace}))
        source_event_ids = [e.event_id for e in c]

        rules.append(
            RuleDraft(
                text=rule_text,
                support_count=len(c),
                workspaces=workspaces,
                source_event_ids=source_event_ids,
                status="proposed",
            )
        )

    # Sort deterministically by support count descending, then text
    rules.sort(key=lambda r: (-r.support_count, r.text))
    return rules


def promote_decisions(
    events: list[ExtractedEvent],
    min_confidence: float = 0.6,
    min_support: int = 1,
    require_reason: bool = False,
) -> list[DecisionDraft]:
    """Promote decision and dead_end events into decision drafts."""
    filtered = [
        e for e in events
        if e.type in ("decision", "dead_end")
        and e.confidence >= min_confidence
        and (not require_reason or (e.reason and len(e.reason.strip()) > 3))
    ]
    if not filtered:
        return []

    drafts: list[DecisionDraft] = []
    # Deduplicate near-identical normalized text within the same session/workspace
    for ev in filtered:
        ev_norm = _normalize_text(ev.normalized)
        duplicate_found = False

        for existing in drafts:
            if existing.kind != ev.type:
                continue
            # Same workspace or session
            same_scope = (
                (ev.workspace and existing.workspace and ev.workspace == existing.workspace)
                or (ev.session_id and existing.session_id and ev.session_id == existing.session_id)
            )
            if same_scope and (
                _similarity_match(ev_norm, _normalize_text(existing.text))
                or _decision_fuzzy_match(ev_norm, _normalize_text(existing.text))
            ):
                # Merge into existing draft
                if ev.event_id not in existing.source_event_ids:
                    existing.source_event_ids.append(ev.event_id)
                if not existing.reason and ev.reason:
                    existing.reason = ev.reason.strip().replace("—", "--").replace("–", "-")
                if not existing.git_sha and ev.workspace:
                    existing.git_sha = find_nearby_commit(ev.workspace, ev.timestamp)
                duplicate_found = True
                break

        if not duplicate_found:
            clean_text = ev.normalized.strip().replace("—", "--").replace("–", "-")
            clean_reason = ev.reason.strip().replace("—", "--").replace("–", "-") if ev.reason else None
            git_sha = find_nearby_commit(ev.workspace, ev.timestamp) if ev.workspace else None

            drafts.append(
                DecisionDraft(
                    text=clean_text,
                    kind=ev.type,
                    reason=clean_reason,
                    git_sha=git_sha,
                    workspace=ev.workspace,
                    timestamp=ev.timestamp,
                    session_id=ev.session_id,
                    source_event_ids=[ev.event_id],
                    status="proposed",
                )
            )

    if min_support > 1:
        drafts = [d for d in drafts if len(d.source_event_ids) >= min_support]

    return drafts


def promote_skills(
    events: list[ExtractedEvent],
    existing_skills_dir: Path | None = None,
    min_support: int = 1,
    min_workspaces: int = 1,
) -> list[SkillCandidate]:
    """Promote recurring workflows into portable skill candidates."""
    filtered = [e for e in events if e.type == "skill_need" and e.confidence >= 0.6]
    if not filtered:
        return []

    # Cluster by normalized trigger / name seed
    clusters: list[list[ExtractedEvent]] = []
    for ev in filtered:
        seed = ev.trigger or ev.normalized
        seed_norm = _normalize_text(seed)
        matched_cluster = None

        for c in clusters:
            c_seed = c[0].trigger or c[0].normalized
            c_seed_norm = _normalize_text(c_seed)
            if _similarity_match(seed_norm, c_seed_norm):
                matched_cluster = c
                break

        if matched_cluster is not None:
            matched_cluster.append(ev)
        else:
            clusters.append([ev])

    # Promotion criteria:
    # (len(unique workspaces) >= 2) OR regex match on quotes/triggers for \bevery time\b|\bwhenever\b|\balways when\b
    every_time_re = re.compile(r"\b(every time|whenever|always when)\b", re.IGNORECASE)
    existing_skills = _load_existing_skills(existing_skills_dir)

    candidates: list[SkillCandidate] = []
    for c in clusters:
        if len(c) < min_support:
            continue

        unique_ws = {e.workspace for e in c if e.workspace}
        if len(unique_ws) < min_workspaces:
            continue

        has_multi_workspace = len(unique_ws) >= 2
        has_strong_language = any(
            every_time_re.search(e.quote or "")
            or every_time_re.search(e.trigger or "")
            or every_time_re.search(e.normalized or "")
            for e in c
        )

        if not (has_multi_workspace or has_strong_language):
            continue

        best_event = max(c, key=lambda e: (e.confidence, len(e.normalized)))
        kebab_name = _to_kebab_case(best_event.normalized, max_len=64)

        raw_desc = (
            f"Automate {best_event.normalized.strip().rstrip('.')}. "
            + (f"Use when {best_event.trigger.strip().rstrip('.')}." if best_event.trigger else "")
        ).strip().replace("—", "--").replace("–", "-")

        trigger_text = (
            best_event.trigger
            or f"Use when needing to automate {best_event.normalized.strip().rstrip('.')}."
        ).strip().replace("—", "--").replace("–", "-")

        # Combine steps hints from cluster events
        steps: list[str] = []
        for e in c:
            for step in e.steps_hint:
                clean_step = step.strip().replace("—", "--").replace("–", "-")
                if clean_step and clean_step not in steps:
                    steps.append(clean_step)

        if not steps:
            steps = ["Outline workflow steps here."]

        overlaps = _check_overlap(kebab_name, raw_desc, existing_skills)

        candidates.append(
            SkillCandidate(
                name=kebab_name,
                description=raw_desc,
                trigger=trigger_text,
                outline_sections=steps,
                support_count=len(c),
                workspaces=sorted(list(unique_ws)),
                source_event_ids=[e.event_id for e in c],
                overlaps_existing=overlaps,
                status="proposed",
            )
        )

    candidates.sort(key=lambda s: (-s.support_count, s.name))
    return candidates
