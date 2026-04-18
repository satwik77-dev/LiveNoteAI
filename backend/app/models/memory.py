from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from .intelligence import ActionItem, Decision, Risk


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChunkHistoryEntry(BaseModel):
    chunk_id: int = Field(ge=0)
    processed_at: str = Field(default_factory=utc_now_iso)
    duration_sec: float = Field(default=0.0, ge=0.0)
    utterance_count: int = Field(default=0, ge=0)
    trust_violations: int = Field(default=0, ge=0)


class MemoryState(BaseModel):
    meeting_id: str
    chunk_id: int = Field(default=0, ge=0)
    started_at: str = Field(default_factory=utc_now_iso)
    meeting_start_date: str
    running_summary: str = ""
    summary_human_locked: bool = False
    summary_human_value: Optional[str] = None
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    risks: List[Risk] = Field(default_factory=list)
    known_speakers: List[str] = Field(default_factory=list)
    chunk_history: List[ChunkHistoryEntry] = Field(default_factory=list)
    is_active: bool = True
    display_transcript_buffer: List[dict] = Field(default_factory=list)
    llm_transcript_buffer: List[dict] = Field(default_factory=list)
    previous_llm_window: List[dict] = Field(default_factory=list)
    asr_chunks_since_last_llm: int = Field(default=0, ge=0)
