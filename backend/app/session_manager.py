from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha1
from typing import Any
from uuid import uuid4

from .models.intelligence import ActionItem, Decision, Risk
from .models.memory import ChunkHistoryEntry, MemoryState
from .models.utterance import ChunkTranscript
from .utils.export_utils import export_json, export_pdf


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MeetingMetadata:
    title: str = "Untitled meeting"
    mode: str = "live"
    started_at: str = field(default_factory=_utc_now_iso)

    def to_frontend(self) -> dict[str, Any]:
        return {"title": self.title, "mode": self.mode, "startedAt": self.started_at}


@dataclass(slots=True)
class ChunkReceipt:
    sequence_number: int
    chunk_bytes: int
    mime_type: str
    duration_ms: int
    received_at: str = field(default_factory=_utc_now_iso)


_ITEM_TYPE_ALIASES = {
    "action_item": "action",
    "action": "action",
    "decision": "decision",
    "risk": "risk",
}


class SessionManager:
    """Owns in-memory meeting state and applies human edits."""

    def __init__(self) -> None:
        self._sessions: dict[str, MemoryState] = {}
        self._metadata: dict[str, MeetingMetadata] = {}
        self._receipts: dict[str, list[ChunkReceipt]] = {}

    def create_session(
        self,
        meeting_id: str | None = None,
        metadata: MeetingMetadata | None = None,
    ) -> MemoryState:
        resolved_meeting_id = meeting_id or f"meeting-{uuid4().hex[:10]}"
        session = MemoryState(
            meeting_id=resolved_meeting_id,
            meeting_start_date=date.today().isoformat(),
        )
        self._sessions[resolved_meeting_id] = session
        self._metadata[resolved_meeting_id] = metadata or MeetingMetadata()
        self._receipts[resolved_meeting_id] = []
        return session

    def list_sessions(self) -> list[MemoryState]:
        return list(self._sessions.values())

    def get_session(self, meeting_id: str) -> MemoryState:
        if meeting_id not in self._sessions:
            raise KeyError(f"Unknown meeting_id: {meeting_id}")
        return self._sessions[meeting_id]

    def get_or_create_session(
        self,
        meeting_id: str,
        metadata: MeetingMetadata | None = None,
    ) -> MemoryState:
        if meeting_id in self._sessions:
            return self._sessions[meeting_id]
        return self.create_session(meeting_id, metadata)

    def get_metadata(self, meeting_id: str) -> MeetingMetadata:
        return self._metadata.get(meeting_id, MeetingMetadata())

    def end_session(self, meeting_id: str) -> MemoryState:
        session = self.get_session(meeting_id)
        session.is_active = False
        return session

    def delete_session(self, meeting_id: str) -> None:
        self._sessions.pop(meeting_id, None)
        self._metadata.pop(meeting_id, None)
        self._receipts.pop(meeting_id, None)

    def append_chunk_transcript(self, transcript: ChunkTranscript) -> MemoryState:
        session = self.get_or_create_session(transcript.meeting_id)
        session.chunk_id = transcript.chunk_id
        session.asr_chunks_since_last_llm += 1
        session.display_transcript_buffer.extend(
            utterance.model_dump() for utterance in transcript.display_utterances
        )
        session.llm_transcript_buffer.extend(
            utterance.model_dump() for utterance in transcript.llm_utterances
        )
        session.known_speakers = sorted(
            {
                *session.known_speakers,
                *(
                    utterance.speaker
                    for utterance in transcript.display_utterances
                    if utterance.speaker
                ),
            }
        )
        session.chunk_history.append(
            ChunkHistoryEntry(
                chunk_id=transcript.chunk_id,
                duration_sec=max(0.0, transcript.chunk_end - transcript.chunk_start),
                utterance_count=len(transcript.display_utterances),
                trust_violations=0,
            )
        )
        return session

    def mark_llm_window_processed(self, meeting_id: str) -> MemoryState:
        session = self.get_session(meeting_id)
        session.previous_llm_window = list(session.llm_transcript_buffer)
        session.llm_transcript_buffer = []
        session.asr_chunks_since_last_llm = 0
        return session

    def record_chunk_receipt(self, meeting_id: str, receipt: ChunkReceipt) -> None:
        self._receipts.setdefault(meeting_id, []).append(receipt)

    def chunk_receipts(self, meeting_id: str) -> list[ChunkReceipt]:
        return list(self._receipts.get(meeting_id, []))

    # ------------------------------------------------------------------
    # Human mutations
    # ------------------------------------------------------------------

    def update_summary(self, meeting_id: str, summary: str, human_locked: bool = False) -> MemoryState:
        session = self.get_session(meeting_id)
        if session.summary_human_locked and not human_locked:
            return session
        session.running_summary = summary
        if human_locked:
            session.summary_human_locked = True
            session.summary_human_value = summary
        return session

    def add_action_item(self, meeting_id: str, payload: dict[str, Any]) -> ActionItem:
        session = self.get_session(meeting_id)
        data = {**payload}
        data.setdefault("id", sha1(f"action:{data.get('task','')}:{uuid4().hex}".encode()).hexdigest()[:10])
        # Frontend may send evidence as a free-text string; backend schema expects list
        data.pop("evidence", None)
        action = ActionItem(human_locked=True, **data)
        session.action_items.append(action)
        return action

    def add_decision(self, meeting_id: str, payload: dict[str, Any]) -> Decision:
        session = self.get_session(meeting_id)
        data = {**payload}
        data.setdefault("id", sha1(f"decision:{data.get('decision','')}:{uuid4().hex}".encode()).hexdigest()[:10])
        data.pop("evidence", None)
        decision = Decision(human_locked=True, **data)
        session.decisions.append(decision)
        return decision

    def add_risk(self, meeting_id: str, payload: dict[str, Any]) -> Risk:
        session = self.get_session(meeting_id)
        data = {**payload}
        data.setdefault("id", sha1(f"risk:{data.get('risk','')}:{uuid4().hex}".encode()).hexdigest()[:10])
        data.pop("evidence", None)
        risk = Risk(human_locked=True, **data)
        session.risks.append(risk)
        return risk

    def update_item(
        self,
        meeting_id: str,
        item_type: str,
        item_id: str,
        updates: dict[str, Any],
    ) -> Any:
        collection = self._resolve_collection(self.get_session(meeting_id), item_type)
        item = self._find_item(collection, item_id)
        for key, value in (updates or {}).items():
            if hasattr(item, key):
                setattr(item, key, value)
        item.human_locked = True
        return item

    def toggle_action_status(self, meeting_id: str, item_id: str, status: str) -> ActionItem:
        action = self._find_item(self.get_session(meeting_id).action_items, item_id)
        action.status = status
        action.human_locked = True
        return action

    def set_item_deleted(
        self,
        meeting_id: str,
        item_type: str,
        item_id: str,
        deleted: bool,
    ) -> Any:
        collection = self._resolve_collection(self.get_session(meeting_id), item_type)
        item = self._find_item(collection, item_id)
        if hasattr(item, "status"):
            item.status = "deleted" if deleted else "open"
        item.human_locked = True
        return item

    # ------------------------------------------------------------------
    # Snapshots for the wire protocol
    # ------------------------------------------------------------------

    def serialize_session(self, meeting_id: str) -> dict[str, Any]:
        return self.get_session(meeting_id).model_dump()

    def session_created_payload(self, meeting_id: str) -> dict[str, Any]:
        meta = self.get_metadata(meeting_id)
        session = self.get_session(meeting_id)
        return {
            "meeting_id": meeting_id,
            "created_at": session.started_at,
            "metadata": meta.to_frontend(),
        }

    def meeting_session_snapshot(self, meeting_id: str) -> dict[str, Any]:
        session = self.get_session(meeting_id)
        receipts = self._receipts.get(meeting_id, [])
        meta = self.get_metadata(meeting_id)
        return {
            "meetingId": meeting_id,
            "active": session.is_active,
            "mode": meta.mode,
            "startedAt": session.started_at,
            "lastChunkSequence": receipts[-1].sequence_number if receipts else 0,
            "receivedChunks": len(receipts),
        }

    def rolling_memory_payload(self, meeting_id: str) -> dict[str, Any]:
        session = self.get_session(meeting_id)
        llm_window_chunks = 4
        return {
            "display_utterances": len(session.display_transcript_buffer),
            "llm_buffer_utterances": len(session.llm_transcript_buffer),
            "previous_window_utterances": len(session.previous_llm_window),
            "asr_chunks_since_last_llm": session.asr_chunks_since_last_llm,
            "llm_window_ready": session.asr_chunks_since_last_llm >= llm_window_chunks,
            "current_window_utterances": len(session.llm_transcript_buffer),
        }

    def intelligence_payload(self, meeting_id: str) -> dict[str, Any]:
        session = self.get_session(meeting_id)
        return {
            "meeting_id": meeting_id,
            "chunk_id": session.chunk_id,
            "summary": {
                "running_summary": session.running_summary,
                "current_topic_focus": "",
                "unresolved_issues": [],
                "locked": session.summary_human_locked,
            },
            "action_items": [_action_to_frontend(a) for a in session.action_items],
            "decisions": [_decision_to_frontend(d) for d in session.decisions],
            "risks": [_risk_to_frontend(r) for r in session.risks],
            "review_flags": _review_flags(session),
            "llm_metadata": _llm_metadata_hint(),
        }

    def transcript_snapshot(self, meeting_id: str) -> list[dict[str, Any]]:
        session = self.get_session(meeting_id)
        out: list[dict[str, Any]] = []
        for idx, raw in enumerate(session.display_transcript_buffer):
            out.append(
                {
                    "id": f"{meeting_id}-u{idx}",
                    "chunk_id": raw.get("chunk_id", session.chunk_id),
                    "speaker": raw.get("speaker", "SPEAKER_01"),
                    "text": raw.get("text", ""),
                    "start_time": raw.get("start_time", 0.0),
                    "end_time": raw.get("end_time", 0.0),
                    "confidence": raw.get("confidence", 0.0),
                }
            )
        return out

    def final_report_bundle(self, meeting_id: str) -> dict[str, Any]:
        session = self.get_session(meeting_id)
        meta = self.get_metadata(meeting_id)
        action_items = [_action_to_frontend(a) for a in session.action_items]
        decisions = [_decision_to_frontend(d) for d in session.decisions]
        risks = [_risk_to_frontend(r) for r in session.risks]
        report = {
            "meeting": {
                "meeting_id": meeting_id,
                "title": meta.title,
                "mode": meta.mode,
                "started_at": session.started_at,
                "generated_at": _utc_now_iso(),
            },
            "summary": {
                "running_summary": session.running_summary,
                "current_topic_focus": "",
                "unresolved_issues": [],
                "locked": session.summary_human_locked,
            },
            "action_items": action_items,
            "decisions": decisions,
            "risks": risks,
            "review_flags": _review_flags(session),
            "transcript": self.transcript_snapshot(meeting_id),
        }
        try:
            json_bytes = export_json(session)
            pdf_bytes = export_pdf(session)
        except Exception:
            json_bytes = b"{}"
            pdf_bytes = b""
        return {
            "report": report,
            "json_base64": base64.b64encode(json_bytes).decode("ascii"),
            "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
            "storage": None,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _find_item(collection: list[Any], item_id: str) -> Any:
        for item in collection:
            if item.id == item_id:
                return item
        raise KeyError(f"Unknown item id: {item_id}")

    @staticmethod
    def _resolve_collection(session: MemoryState, item_type: str) -> list[Any]:
        key = _ITEM_TYPE_ALIASES.get(item_type)
        if key == "action":
            return session.action_items
        if key == "decision":
            return session.decisions
        if key == "risk":
            return session.risks
        raise KeyError(f"Unsupported item type: {item_type}")


# ----------------------------------------------------------------------
# Item → frontend shape helpers
# ----------------------------------------------------------------------

def _action_to_frontend(a: ActionItem) -> dict[str, Any]:
    return {
        "id": a.id,
        "task": a.task,
        "owner": a.owner,
        "deadline": a.deadline,
        "normalized_deadline": a.deadline_normalized,
        "priority": a.priority,
        "status": a.status,
        "needs_review": a.needs_review,
        "evidence": "",
        "evidence_spans": [
            {"utterance_id": None, "start_time": s.start, "end_time": s.end, "text": ""}
            for s in a.evidence
        ],
        "chunk_origin": a.chunk_origin,
        "human_locked": a.human_locked,
        "deleted": a.status == "deleted",
    }


def _decision_to_frontend(d: Decision) -> dict[str, Any]:
    return {
        "id": d.id,
        "decision": d.decision,
        "evidence": "",
        "evidence_spans": [
            {"utterance_id": None, "start_time": s.start, "end_time": s.end, "text": ""}
            for s in d.evidence
        ],
        "chunk_origin": d.chunk_origin,
        "human_locked": d.human_locked,
        "deleted": False,
    }


def _risk_to_frontend(r: Risk) -> dict[str, Any]:
    return {
        "id": r.id,
        "risk": r.risk,
        "evidence": "",
        "evidence_spans": [
            {"utterance_id": None, "start_time": s.start, "end_time": s.end, "text": ""}
            for s in r.evidence
        ],
        "chunk_origin": r.chunk_origin,
        "human_locked": r.human_locked,
        "deleted": False,
    }


def _review_flags(session: MemoryState) -> list[str]:
    flags: list[str] = []
    for action in session.action_items:
        if action.needs_review:
            flags.append(f"action:{action.id}")
    return flags


def _llm_metadata_hint() -> dict[str, Any]:
    import os
    return {
        "provider": os.getenv("LLM_MODE", "ollama"),
        "model": os.getenv("LLM_MODEL", "mistral:7b"),
        "queued_windows": 0,
    }
