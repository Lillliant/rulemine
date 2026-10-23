"""Data models for RuleMine transcript extraction, promotion, and drafting."""

from __future__ import annotations

import hashlib
from typing import Literal
from pydantic import BaseModel, Field

EventType = Literal["correction", "decision", "dead_end", "skill_need"]
ReviewStatus = Literal["proposed", "approved", "rejected"]


class Turn(BaseModel):
    step_index: int | None = None
    source: str | None = None
    type: str | None = None
    created_at: str | None = None
    content: str = ""


class SessionRef(BaseModel):
    session_id: str
    transcript_path: str
    workspace: str | None = None
    title: str | None = None
    mtime: float | None = None


class Window(BaseModel):
    session_id: str
    workspace: str | None = None
    start_step: int | None = None
    end_step: int | None = None
    text: str  # redacted, ready for model
    turn_count: int = 0


class ExtractedEvent(BaseModel):
    type: EventType
    quote: str  # short evidence from user/agent text
    normalized: str  # imperative rule OR one-line decision OR skill title seed
    reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    session_id: str
    step_index: int | None = None
    workspace: str | None = None
    timestamp: str | None = None
    # skill_need optional extras
    trigger: str | None = None  # when agent should load the skill
    steps_hint: list[str] = Field(default_factory=list)

    @property
    def event_id(self) -> str:
        key = f"{self.session_id}|{self.step_index}|{self.type}|{self.normalized}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class ExtractResponse(BaseModel):
    events: list[ExtractedEvent] = Field(default_factory=list)


class RuleDraft(BaseModel):
    text: str
    support_count: int
    workspaces: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    status: ReviewStatus = "proposed"

    @property
    def draft_id(self) -> str:
        key = f"rule|{self.text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class DecisionDraft(BaseModel):
    text: str
    kind: Literal["decision", "dead_end"]
    reason: str | None = None
    git_sha: str | None = None
    workspace: str | None = None
    timestamp: str | None = None
    session_id: str | None = None
    source_event_ids: list[str] = Field(default_factory=list)
    status: ReviewStatus = "proposed"

    @property
    def draft_id(self) -> str:
        key = f"{self.kind}|{self.text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


class SkillCandidate(BaseModel):
    name: str  # kebab-case
    description: str  # YAML description = what + when
    trigger: str
    outline_sections: list[str] = Field(default_factory=list)
    support_count: int
    workspaces: list[str] = Field(default_factory=list)
    source_event_ids: list[str] = Field(default_factory=list)
    overlaps_existing: list[str] = Field(default_factory=list)
    status: ReviewStatus = "proposed"

    @property
    def draft_id(self) -> str:
        key = f"skill|{self.name}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
