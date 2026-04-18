from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    speaker: str = Field(default="SPEAKER_01")
    text: str = Field(default="")
    start_time: float = Field(default=0.0, ge=0.0)
    end_time: float = Field(default=0.0, ge=0.0)
    word_count: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ChunkTranscript(BaseModel):
    meeting_id: str
    chunk_id: int = Field(ge=0)
    chunk_start: float = Field(default=0.0, ge=0.0)
    chunk_end: float = Field(default=0.0, ge=0.0)
    display_utterances: List[Utterance] = Field(default_factory=list)
    llm_utterances: List[Utterance] = Field(default_factory=list)
    processing_time_sec: float = Field(default=0.0, ge=0.0)
    asr_model: str = Field(default="pending")
