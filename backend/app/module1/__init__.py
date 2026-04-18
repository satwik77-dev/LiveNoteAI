from __future__ import annotations

import time
from pathlib import Path

from ..models.utterance import ChunkTranscript, Utterance
from .alignment import assign_speakers_to_segments
from .asr import TranscribedSegment, transcribe_audio
from .diarization import SpeakerSegment, diarize_audio
from .noise_filter import filter_utterances


def process_audio_chunk(
    *,
    wav_path: str | Path,
    meeting_id: str,
    chunk_id: int,
    chunk_start: float,
    chunk_end: float,
) -> ChunkTranscript:
    """Run the full Module 1 pipeline for a single browser chunk."""

    started = time.perf_counter()
    wav_file = Path(wav_path)

    asr_result = transcribe_audio(wav_file, chunk_start=chunk_start)
    diarization_segments = diarize_audio(wav_file, chunk_start=chunk_start)
    aligned_utterances = _build_aligned_utterances(asr_result.segments, diarization_segments)
    display_utterances, llm_utterances = filter_utterances(aligned_utterances)

    return ChunkTranscript(
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        display_utterances=display_utterances,
        llm_utterances=llm_utterances,
        processing_time_sec=round(time.perf_counter() - started, 3),
        asr_model=asr_result.model_name,
    )


def _build_aligned_utterances(
    asr_segments: list[TranscribedSegment],
    diarization_segments: list[SpeakerSegment],
) -> list[Utterance]:
    if not asr_segments:
        return []

    return assign_speakers_to_segments(asr_segments, diarization_segments)
