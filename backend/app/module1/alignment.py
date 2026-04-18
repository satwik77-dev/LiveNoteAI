from __future__ import annotations

from ..models.utterance import Utterance
from .asr import TranscribedSegment
from .diarization import SpeakerSegment


def assign_speakers_to_segments(
    asr_segments: list[TranscribedSegment],
    diarization_segments: list[SpeakerSegment],
) -> list[Utterance]:
    if not diarization_segments:
        return [
            _to_utterance(segment=segment, speaker="SPEAKER_01")
            for segment in asr_segments
        ]

    utterances: list[Utterance] = []
    for segment in asr_segments:
        speaker = _max_overlap_speaker(segment, diarization_segments)
        utterances.append(_to_utterance(segment=segment, speaker=speaker))
    return utterances


def _max_overlap_speaker(
    segment: TranscribedSegment,
    diarization_segments: list[SpeakerSegment],
) -> str:
    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0.0

    for diarized in diarization_segments:
        overlap = _overlap_duration(segment.start, segment.end, diarized.start, diarized.end)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diarized.speaker

    return best_speaker if best_overlap > 0 else "SPEAKER_UNKNOWN"


def _overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _to_utterance(*, segment: TranscribedSegment, speaker: str) -> Utterance:
    text = segment.text.strip()
    return Utterance(
        speaker=speaker,
        text=text,
        start_time=segment.start,
        end_time=segment.end,
        word_count=len(text.split()),
        confidence=segment.confidence,
    )
