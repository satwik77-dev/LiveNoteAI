"""Unit tests for export_utils — PDF and JSON export."""
import json
import pytest

from app.utils.export_utils import export_json, export_pdf
from app.models.memory import MemoryState
from app.models.intelligence import ActionItem, Decision, Risk


def _state() -> MemoryState:
    return MemoryState(
        meeting_id="export-test",
        meeting_start_date="2026-04-13",
        running_summary="This is a test meeting summary.",
        action_items=[
            ActionItem(
                id="a1",
                task="Write tests",
                owner="SPEAKER_01",
                deadline="2026-06-01",
                priority="High",
                status="open",
                needs_review=False,
                evidence=[],
                human_locked=False,
            )
        ],
        decisions=[
            Decision(
                id="d1",
                decision="Use FastAPI for backend",
                evidence=[],
                human_locked=False,
            )
        ],
        risks=[
            Risk(
                id="r1",
                risk="Timeline is tight",
                evidence=[],
                human_locked=False,
            )
        ],
        known_speakers=["SPEAKER_01", "SPEAKER_02"],
    )


class TestExportJSON:
    def test_returns_bytes(self):
        data = export_json(_state())
        assert isinstance(data, bytes)

    def test_valid_json(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert isinstance(obj, dict)

    def test_meeting_id_present(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert obj["meeting_id"] == "export-test"

    def test_exported_at_present(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert "exported_at" in obj

    def test_action_items_included(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert len(obj["action_items"]) == 1
        assert obj["action_items"][0]["task"] == "Write tests"

    def test_decisions_included(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert obj["decisions"][0]["decision"] == "Use FastAPI for backend"

    def test_risks_included(self):
        data = export_json(_state())
        obj = json.loads(data.decode())
        assert obj["risks"][0]["risk"] == "Timeline is tight"


class TestExportPDF:
    def test_returns_bytes(self):
        data = export_pdf(_state())
        assert isinstance(data, bytes)

    def test_starts_with_pdf_header(self):
        data = export_pdf(_state())
        assert data[:4] == b"%PDF"

    def test_non_empty(self):
        data = export_pdf(_state())
        assert len(data) > 1000  # real PDF is never tiny
