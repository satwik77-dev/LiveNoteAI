from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)


class ActionItem(BaseModel):
    id: str
    task: str
    owner: str
    deadline: str = "unspecified"
    deadline_normalized: str = "unspecified"
    priority: str = "Medium"
    status: str = "open"
    needs_review: bool = False
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    chunk_origin: int = 0
    human_locked: bool = False
    human_value: Optional[str] = None


class Decision(BaseModel):
    id: str
    decision: str
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    chunk_origin: int = 0
    human_locked: bool = False
    human_value: Optional[str] = None


class Risk(BaseModel):
    id: str
    risk: str
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    chunk_origin: int = 0
    human_locked: bool = False
    human_value: Optional[str] = None


class TrustViolation(BaseModel):
    rule: int = Field(ge=1)
    description: str
    auto_fixed: bool = False
