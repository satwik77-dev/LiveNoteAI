from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    """Tracks active meeting sockets and handles fan-out."""

    def __init__(self, allowed_origins: list[str] | None = None) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._allowed_origins = {origin.strip() for origin in (allowed_origins or []) if origin.strip()}

    async def connect(self, meeting_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[meeting_id].append(websocket)

    async def bind(self, meeting_id: str, websocket: WebSocket) -> None:
        """Attach an already-accepted socket to a meeting for fan-out."""
        if websocket not in self._connections[meeting_id]:
            self._connections[meeting_id].append(websocket)

    def disconnect(self, meeting_id: str, websocket: WebSocket) -> None:
        if meeting_id not in self._connections:
            return
        self._connections[meeting_id] = [
            connection for connection in self._connections[meeting_id] if connection is not websocket
        ]
        if not self._connections[meeting_id]:
            self._connections.pop(meeting_id, None)

    def validate_origin(self, websocket: WebSocket) -> bool:
        if not self._allowed_origins:
            return True
        origin = websocket.headers.get("origin", "")
        return origin in self._allowed_origins

    async def send_to_meeting(self, meeting_id: str, message: dict[str, Any]) -> None:
        stale_connections: list[WebSocket] = []
        for connection in self._connections.get(meeting_id, []):
            try:
                await connection.send_json(message)
            except Exception:
                stale_connections.append(connection)

        for connection in stale_connections:
            self.disconnect(meeting_id, connection)

    async def broadcast(self, message: dict[str, Any]) -> None:
        for meeting_id in list(self._connections.keys()):
            await self.send_to_meeting(meeting_id, message)
