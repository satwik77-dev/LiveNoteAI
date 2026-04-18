from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel


@dataclass(slots=True)
class TranscribedSegment:
    text: str
    start: float
    end: float
    confidence: float


@dataclass(slots=True)
class ASRResult:
    segments: list[TranscribedSegment]
    model_name: str


def transcribe_audio(wav_path: str | Path, *, chunk_start: float = 0.0) -> ASRResult:
    model = _get_whisper_model()
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    language = os.getenv("WHISPER_LANGUAGE", "en")

    segments, _ = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
    )

    result_segments: list[TranscribedSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        result_segments.append(
            TranscribedSegment(
                text=text,
                start=chunk_start + float(segment.start),
                end=chunk_start + float(segment.end),
                confidence=_confidence_from_avg_logprob(float(segment.avg_logprob)),
            )
        )

    return ASRResult(
        segments=result_segments,
        model_name=os.getenv("WHISPER_MODEL_SIZE", "small"),
    )


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    return WhisperModel(model_size, device="cpu", compute_type=compute_type)


def _confidence_from_avg_logprob(avg_logprob: float) -> float:
    confidence = math.exp(avg_logprob)
    return max(0.0, min(1.0, confidence))
