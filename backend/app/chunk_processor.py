from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .module1 import process_audio_chunk
from .module2 import IntelligenceExtractor
from .models.utterance import ChunkTranscript
from .session_manager import ChunkReceipt, SessionManager
from .utils.audio_utils import cleanup_temp_file, convert_webm_bytes_to_wav_path
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunkTask:
    meeting_id: str
    sequence_number: int
    chunk_start: float
    chunk_end: float
    audio_bytes: bytes
    mime_type: str = "audio/webm"
    duration_ms: int = 0


class ChunkProcessor:
    """Owns per-meeting queues and the cadence-aware background worker."""

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        websocket_manager: WebSocketManager,
        intelligence_extractor: IntelligenceExtractor | None = None,
        queue_max_size: int | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.websocket_manager = websocket_manager
        self.live_intelligence_enabled = os.getenv("LIVE_INTELLIGENCE_ENABLED", "true").lower() == "true"
        self.intelligence_extractor = intelligence_extractor
        self.queue_max_size = queue_max_size or int(os.getenv("CHUNK_QUEUE_MAX_SIZE", "3"))
        self.llm_window_chunks = int(os.getenv("LLM_WINDOW_CHUNKS", "4"))
        self._queues: dict[str, asyncio.Queue[AudioChunkTask]] = {}
        self._workers: dict[str, asyncio.Task[None]] = {}

    def ensure_worker(self, meeting_id: str) -> None:
        if meeting_id in self._workers and not self._workers[meeting_id].done():
            return
        queue: asyncio.Queue[AudioChunkTask] = asyncio.Queue(maxsize=self.queue_max_size)
        self._queues[meeting_id] = queue
        self._workers[meeting_id] = asyncio.create_task(self._worker_loop(meeting_id, queue))

    async def enqueue_chunk(self, task: AudioChunkTask) -> None:
        self.ensure_worker(task.meeting_id)
        queue = self._queues[task.meeting_id]
        if queue.full():
            await self.websocket_manager.send_to_meeting(
                task.meeting_id,
                {
                    "type": "queue_full",
                    "payload": {
                        "meeting_id": task.meeting_id,
                        "sequence_number": task.sequence_number,
                        "message": "Chunk queue is full. Rejecting newest audio chunk.",
                    },
                },
            )
            return
        await queue.put(task)

    async def final_flush(self, meeting_id: str) -> None:
        """Architecture §13 — flush partial buffer + consolidation on meeting_end."""
        try:
            session = self.session_manager.get_session(meeting_id)
        except KeyError:
            logger.warning("final_flush: unknown meeting_id %s", meeting_id)
            return

        if self.live_intelligence_enabled:
            if session.llm_transcript_buffer:
                logger.info("[%s] final_flush: %d remaining utterances", meeting_id, len(session.llm_transcript_buffer))
                try:
                    if self.intelligence_extractor is None:
                        self.intelligence_extractor = IntelligenceExtractor()
                    await asyncio.to_thread(self.intelligence_extractor.run, session)
                except Exception as exc:
                    logger.warning("[%s] final_flush LLM window failed: %s", meeting_id, exc)
                self.session_manager.mark_llm_window_processed(meeting_id)
                session = self.session_manager.get_session(meeting_id)

            logger.info("[%s] final_flush: consolidation pass", meeting_id)
            try:
                if self.intelligence_extractor is None:
                    self.intelligence_extractor = IntelligenceExtractor()
                await asyncio.to_thread(self.intelligence_extractor.consolidate, session)
            except Exception as exc:
                logger.warning("[%s] consolidation pass failed: %s", meeting_id, exc)

        bundle = self.session_manager.final_report_bundle(meeting_id)
        await self.websocket_manager.send_to_meeting(
            meeting_id,
            {
                "type": "consolidation_complete",
                "payload": {"meeting_id": meeting_id, **bundle},
            },
        )
        logger.info("[%s] final_flush complete", meeting_id)

    async def shutdown(self) -> None:
        for worker in self._workers.values():
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers.values(), return_exceptions=True)

    async def _worker_loop(self, meeting_id: str, queue: asyncio.Queue[AudioChunkTask]) -> None:
        while True:
            task = await queue.get()
            try:
                await self._process_task(task)
            finally:
                queue.task_done()

    async def _process_task(self, task: AudioChunkTask) -> None:
        wav_path: Path | None = None
        await self.websocket_manager.send_to_meeting(
            task.meeting_id,
            {
                "type": "processing_started",
                "payload": {
                    "meeting_id": task.meeting_id,
                    "sequence_number": task.sequence_number,
                },
            },
        )

        try:
            wav_path = convert_webm_bytes_to_wav_path(task.audio_bytes)
            transcript: ChunkTranscript = await asyncio.to_thread(
                process_audio_chunk,
                wav_path=wav_path,
                meeting_id=task.meeting_id,
                chunk_id=task.sequence_number,
                chunk_start=task.chunk_start,
                chunk_end=task.chunk_end,
            )
            self.session_manager.append_chunk_transcript(transcript)
            self.session_manager.record_chunk_receipt(
                task.meeting_id,
                ChunkReceipt(
                    sequence_number=task.sequence_number,
                    chunk_bytes=len(task.audio_bytes),
                    mime_type=task.mime_type,
                    duration_ms=task.duration_ms,
                ),
            )

            await self.websocket_manager.send_to_meeting(
                task.meeting_id,
                {
                    "type": "transcript_update",
                    "payload": _transcript_update_payload(transcript, self.session_manager),
                },
            )

            session = self.session_manager.get_session(task.meeting_id)
            if session.asr_chunks_since_last_llm >= self.llm_window_chunks:
                if self.live_intelligence_enabled:
                    try:
                        if self.intelligence_extractor is None:
                            self.intelligence_extractor = IntelligenceExtractor()
                        await asyncio.to_thread(self.intelligence_extractor.run, session)
                    except Exception as exc:
                        logger.warning("[%s] intelligence extractor failed: %s", task.meeting_id, exc)
                self.session_manager.mark_llm_window_processed(task.meeting_id)
                await self.websocket_manager.send_to_meeting(
                    task.meeting_id,
                    {
                        "type": "intelligence_update",
                        "payload": self.session_manager.intelligence_payload(task.meeting_id),
                    },
                )

            sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
            channels = int(os.getenv("AUDIO_CHANNELS", "1"))
            await self.websocket_manager.send_to_meeting(
                task.meeting_id,
                {
                    "type": "processing_complete",
                    "payload": {
                        "meeting_id": task.meeting_id,
                        "sequence_number": task.sequence_number,
                        "chunk_bytes": len(task.audio_bytes),
                        "mime_type": task.mime_type,
                        "duration_ms": task.duration_ms,
                        "processing_metadata": {
                            "audio_duration_sec": max(0.0, task.chunk_end - task.chunk_start),
                            "sample_rate": sample_rate,
                            "channels": channels,
                            "asr_time_ms": int(transcript.processing_time_sec * 1000),
                            "model_info": transcript.asr_model,
                            "segment_count": len(transcript.display_utterances),
                        },
                    },
                },
            )
        except Exception as exc:
            logger.exception("[%s] chunk processing failed", task.meeting_id)
            await self.websocket_manager.send_to_meeting(
                task.meeting_id,
                {
                    "type": "error",
                    "payload": {"code": "chunk_failed", "detail": str(exc)},
                },
            )
        finally:
            cleanup_temp_file(wav_path)


def _transcript_update_payload(
    transcript: ChunkTranscript,
    session_manager: SessionManager,
) -> dict:
    utterances = []
    for idx, utt in enumerate(transcript.display_utterances):
        utterances.append(
            {
                "id": f"{transcript.meeting_id}-c{transcript.chunk_id}-u{idx}",
                "chunk_id": transcript.chunk_id,
                "speaker": utt.speaker,
                "text": utt.text,
                "start_time": utt.start_time,
                "end_time": utt.end_time,
                "confidence": utt.confidence,
            }
        )
    return {
        "meeting_id": transcript.meeting_id,
        "chunk_id": transcript.chunk_id,
        "chunk_start_time": transcript.chunk_start,
        "chunk_end_time": transcript.chunk_end,
        "utterances": utterances,
        "diarization_pending": False,
        "rolling_memory": session_manager.rolling_memory_payload(transcript.meeting_id),
    }
