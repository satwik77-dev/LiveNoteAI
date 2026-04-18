from __future__ import annotations

import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .chunk_processor import AudioChunkTask, ChunkProcessor
from .session_manager import MeetingMetadata, SessionManager
from .utils.export_utils import export_json, export_pdf
from .websocket_manager import WebSocketManager

load_dotenv()
logger = logging.getLogger(__name__)


def _allowed_origins() -> list[str]:
    return [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]


session_manager = SessionManager()
websocket_manager = WebSocketManager(allowed_origins=_allowed_origins())
chunk_processor = ChunkProcessor(
    session_manager=session_manager,
    websocket_manager=websocket_manager,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await chunk_processor.shutdown()


app = FastAPI(
    title="LiveNote Backend",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins() or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# HTTP endpoints
# ----------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "livenote-backend",
        "diarization_enabled": os.getenv("DIARIZATION_ENABLED", "true").lower() == "true",
        "live_intelligence_enabled": os.getenv("LIVE_INTELLIGENCE_ENABLED", "true").lower() == "true",
        "llm_mode": os.getenv("LLM_MODE", "ollama"),
    }


@app.get("/meetings")
async def list_meetings() -> dict[str, Any]:
    return {"meetings": [session.model_dump() for session in session_manager.list_sessions()]}


@app.get("/meetings/{meeting_id}")
async def get_meeting(meeting_id: str) -> dict[str, Any]:
    try:
        return {"meeting": session_manager.serialize_session(meeting_id)}
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@app.get("/meetings/{meeting_id}/export/json")
async def export_meeting_json(meeting_id: str) -> Response:
    try:
        session = session_manager.get_session(meeting_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=export_json(session),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="meeting_{meeting_id}.json"'},
    )


@app.get("/meetings/{meeting_id}/export/pdf")
async def export_meeting_pdf(meeting_id: str) -> Response:
    try:
        session = session_manager.get_session(meeting_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(
        content=export_pdf(session),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="meeting_{meeting_id}.pdf"'},
    )


# ----------------------------------------------------------------------
# WebSocket — single endpoint, JSON envelopes
# ----------------------------------------------------------------------

@app.websocket("/ws/meeting")
async def meeting_websocket(websocket: WebSocket) -> None:
    if not websocket_manager.validate_origin(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # Accept immediately; meeting_id is unknown until meeting_start arrives.
    await websocket.accept()
    bound_meeting_id: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await _send_error(websocket, "bad_json", "Invalid JSON message.")
                continue

            msg_type = message.get("type")
            payload = message.get("payload") or {}

            try:
                if msg_type == "meeting_start":
                    bound_meeting_id = await _handle_meeting_start(websocket, payload, bound_meeting_id)
                    continue

                if msg_type == "chunk_upload":
                    await _handle_chunk_upload(websocket, payload, bound_meeting_id)
                    continue

                if msg_type == "meeting_end":
                    await _handle_meeting_end(websocket, payload, bound_meeting_id)
                    continue

                if msg_type == "state_sync_request":
                    await _handle_state_sync(websocket, bound_meeting_id)
                    continue

                if msg_type and msg_type.startswith("human_"):
                    await _handle_human_mutation(websocket, msg_type, payload, bound_meeting_id)
                    continue

                await _send_error(websocket, "unknown_message", f"Unknown message type: {msg_type}")
            except KeyError as exc:
                await _send_error(websocket, "not_found", str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.exception("WS handler error")
                await _send_error(websocket, "server_error", str(exc))

    except WebSocketDisconnect:
        if bound_meeting_id:
            websocket_manager.disconnect(bound_meeting_id, websocket)


# ----------------------------------------------------------------------
# Handlers
# ----------------------------------------------------------------------

async def _handle_meeting_start(
    websocket: WebSocket,
    payload: dict[str, Any],
    bound_meeting_id: str | None,
) -> str:
    raw_meta = payload.get("metadata") or {}
    metadata = MeetingMetadata(
        title=raw_meta.get("title") or "Untitled meeting",
        mode=raw_meta.get("mode") or "live",
        started_at=raw_meta.get("startedAt") or datetime.now(timezone.utc).isoformat(),
    )
    session = session_manager.create_session(metadata=metadata)
    meeting_id = session.meeting_id
    session.is_active = True

    if bound_meeting_id and bound_meeting_id != meeting_id:
        websocket_manager.disconnect(bound_meeting_id, websocket)
    await websocket_manager.bind(meeting_id, websocket)
    chunk_processor.ensure_worker(meeting_id)

    await websocket_manager.send_to_meeting(
        meeting_id,
        {
            "type": "session_created",
            "payload": session_manager.session_created_payload(meeting_id),
        },
    )
    return meeting_id


async def _handle_chunk_upload(
    websocket: WebSocket,
    payload: dict[str, Any],
    bound_meeting_id: str | None,
) -> None:
    meeting_id = payload.get("meeting_id") or bound_meeting_id
    if not meeting_id:
        await _send_error(websocket, "no_session", "chunk_upload before meeting_start.")
        return
    try:
        session_manager.get_session(meeting_id)
    except KeyError:
        await _send_error(websocket, "not_found", f"Unknown meeting: {meeting_id}")
        return

    sequence_number = int(payload.get("sequence_number", 0))
    mime_type = payload.get("mime_type", "audio/webm")
    duration_ms = int(payload.get("duration_ms", 0))
    audio_base64 = payload.get("audio_base64", "")
    try:
        audio_bytes = base64.b64decode(audio_base64, validate=False)
    except Exception as exc:  # noqa: BLE001
        await _send_error(websocket, "bad_audio", f"Failed to decode audio: {exc}")
        return

    chunk_sec = float(os.getenv("ASR_CHUNK_SEC", "15"))
    chunk_start = sequence_number * chunk_sec
    chunk_end = chunk_start + (duration_ms / 1000.0 if duration_ms else chunk_sec)

    await chunk_processor.enqueue_chunk(
        AudioChunkTask(
            meeting_id=meeting_id,
            sequence_number=sequence_number,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            duration_ms=duration_ms,
        )
    )


async def _handle_meeting_end(
    websocket: WebSocket,
    payload: dict[str, Any],
    bound_meeting_id: str | None,
) -> None:
    meeting_id = payload.get("meeting_id") or bound_meeting_id
    if not meeting_id:
        await _send_error(websocket, "no_session", "meeting_end without an active session.")
        return
    try:
        session_manager.get_session(meeting_id)
    except KeyError:
        await _send_error(websocket, "not_found", f"Unknown meeting: {meeting_id}")
        return

    await chunk_processor.final_flush(meeting_id)
    session_manager.end_session(meeting_id)
    await websocket_manager.send_to_meeting(
        meeting_id,
        {
            "type": "session_ended",
            "payload": {
                "meeting_id": meeting_id,
                "ended_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    )


async def _handle_state_sync(websocket: WebSocket, bound_meeting_id: str | None) -> None:
    if not bound_meeting_id:
        await websocket.send_json(
            {
                "type": "state_sync",
                "payload": {
                    "session": {
                        "meetingId": None,
                        "active": False,
                        "mode": None,
                        "startedAt": None,
                        "lastChunkSequence": 0,
                        "receivedChunks": 0,
                    },
                    "transcript": [],
                    "rolling_memory": None,
                    "intelligence": None,
                    "final_report": None,
                },
            }
        )
        return

    try:
        session_manager.get_session(bound_meeting_id)
    except KeyError:
        await _send_error(websocket, "not_found", f"Unknown meeting: {bound_meeting_id}")
        return

    await websocket.send_json(
        {
            "type": "state_sync",
            "payload": {
                "session": session_manager.meeting_session_snapshot(bound_meeting_id),
                "transcript": session_manager.transcript_snapshot(bound_meeting_id),
                "rolling_memory": session_manager.rolling_memory_payload(bound_meeting_id),
                "intelligence": session_manager.intelligence_payload(bound_meeting_id),
                "final_report": None,
            },
        }
    )


async def _handle_human_mutation(
    websocket: WebSocket,
    msg_type: str,
    payload: dict[str, Any],
    bound_meeting_id: str | None,
) -> None:
    meeting_id = payload.get("meeting_id") or bound_meeting_id
    if not meeting_id:
        await _send_error(websocket, "no_session", "human_* before meeting_start.")
        return
    try:
        session_manager.get_session(meeting_id)
    except KeyError:
        await _send_error(websocket, "not_found", f"Unknown meeting: {meeting_id}")
        return

    if msg_type == "human_update_summary":
        session_manager.update_summary(meeting_id, payload.get("summary", ""), human_locked=True)
    elif msg_type == "human_add_action":
        data = {k: v for k, v in payload.items() if k != "meeting_id"}
        session_manager.add_action_item(meeting_id, data)
    elif msg_type == "human_add_decision":
        data = {k: v for k, v in payload.items() if k != "meeting_id"}
        session_manager.add_decision(meeting_id, data)
    elif msg_type == "human_add_risk":
        data = {k: v for k, v in payload.items() if k != "meeting_id"}
        session_manager.add_risk(meeting_id, data)
    elif msg_type in ("human_update_item", "human_update_action"):
        item_type = payload.get("item_type", "action_item")
        item_id = payload.get("item_id")
        updates = payload.get("updates") or {}
        if not item_id:
            await _send_error(websocket, "bad_request", "item_id required")
            return
        session_manager.update_item(meeting_id, item_type, item_id, updates)
    elif msg_type == "human_delete_item":
        item_id = payload.get("item_id")
        item_type = payload.get("item_type", "action_item")
        if not item_id:
            await _send_error(websocket, "bad_request", "item_id required")
            return
        session_manager.set_item_deleted(meeting_id, item_type, item_id, True)
    elif msg_type == "human_restore_item":
        item_id = payload.get("item_id")
        item_type = payload.get("item_type", "action_item")
        if not item_id:
            await _send_error(websocket, "bad_request", "item_id required")
            return
        session_manager.set_item_deleted(meeting_id, item_type, item_id, False)
    else:
        await _send_error(websocket, "unknown_message", f"Unhandled: {msg_type}")
        return

    await websocket_manager.send_to_meeting(
        meeting_id,
        {
            "type": "intelligence_update",
            "payload": session_manager.intelligence_payload(meeting_id),
        },
    )


async def _send_error(websocket: WebSocket, code: str, detail: str) -> None:
    try:
        await websocket.send_json({"type": "error", "payload": {"code": code, "detail": detail}})
    except Exception:  # noqa: BLE001
        pass
