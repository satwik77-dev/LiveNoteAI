"""Unit tests for MemoryManager merge logic."""
import pytest

from app.module2.memory_manager import MemoryManager
from app.module2.trust_validator import ValidatedExtraction
from app.models.memory import MemoryState
from app.models.intelligence import ActionItem, Decision, Risk


def _state(**kwargs) -> MemoryState:
    defaults = dict(
        meeting_id="test",
        meeting_start_date="2026-04-13",
        known_speakers=[],
        action_items=[],
        decisions=[],
        risks=[],
        running_summary="",
    )
    defaults.update(kwargs)
    return MemoryState(**defaults)


def _extraction(**kwargs) -> ValidatedExtraction:
    defaults = dict(
        summary="New summary",
        action_items=[],
        decisions=[],
        risks=[],
        violations=[],
    )
    defaults.update(kwargs)
    return ValidatedExtraction(**defaults)


class TestMemoryManagerMerge:
    def test_summary_updated_when_not_locked(self):
        mm = MemoryManager()
        state = _state(running_summary="Old summary")
        ext = _extraction(summary="New summary")
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window=set())
        assert state.running_summary == "New summary"

    def test_summary_not_updated_when_locked(self):
        mm = MemoryManager()
        state = _state(running_summary="Human summary", summary_human_locked=True)
        ext = _extraction(summary="AI summary")
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window=set())
        assert state.running_summary == "Human summary"

    def test_new_action_item_added(self):
        mm = MemoryManager()
        state = _state()
        ext = _extraction(action_items=[{
            "task": "Deploy app",
            "owner": "SPEAKER_01",
            "deadline": "unspecified",
            "deadline_normalized": "unspecified",
            "priority": "High",
            "status": "open",
            "needs_review": False,
            "evidence": [{"start": 5.0, "end": 10.0}],
        }])
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window={"SPEAKER_01"})
        assert len(state.action_items) == 1
        assert state.action_items[0].task == "Deploy app"

    def test_duplicate_action_item_not_duplicated(self):
        mm = MemoryManager()
        state = _state()
        item_data = {
            "task": "Deploy app",
            "owner": "SPEAKER_01",
            "deadline": "unspecified",
            "deadline_normalized": "unspecified",
            "priority": "High",
            "status": "open",
            "needs_review": False,
            "evidence": [{"start": 5.0, "end": 10.0}],
        }
        ext = _extraction(action_items=[item_data])
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window={"SPEAKER_01"})
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=1, speakers_from_window={"SPEAKER_01"})
        assert len(state.action_items) == 1  # no duplicate

    def test_human_locked_action_not_overwritten(self):
        mm = MemoryManager()
        locked_item = ActionItem(
            id="lock1",
            task="Human task",
            owner="SPEAKER_01",
            deadline="unspecified",
            priority="High",
            status="open",
            needs_review=False,
            evidence=[],
            human_locked=True,
            human_value="Human task",
        )
        state = _state(action_items=[locked_item])
        ext = _extraction(action_items=[{
            "task": "Human task",
            "owner": "SPEAKER_01",
            "deadline": "2026-01-01",  # AI tries to update deadline
            "deadline_normalized": "2026-01-01",
            "priority": "Low",
            "status": "open",
            "needs_review": False,
            "evidence": [{"start": 5.0, "end": 10.0}],
        }])
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window={"SPEAKER_01"})
        # Human-locked item should remain unchanged except evidence
        item = state.action_items[0]
        assert item.human_locked
        assert item.priority == "High"  # not changed to Low

    def test_speakers_merged(self):
        mm = MemoryManager()
        state = _state(known_speakers=["SPEAKER_01"])
        mm.merge(memory_state=state, extraction=_extraction(), source_chunk_id=0, speakers_from_window={"SPEAKER_02"})
        assert "SPEAKER_02" in state.known_speakers

    def test_new_decision_added(self):
        mm = MemoryManager()
        state = _state()
        ext = _extraction(decisions=[{
            "decision": "Use React for frontend",
            "evidence": [{"start": 5.0, "end": 10.0}],
        }])
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window=set())
        assert len(state.decisions) == 1
        assert state.decisions[0].decision == "Use React for frontend"

    def test_new_risk_added(self):
        mm = MemoryManager()
        state = _state()
        ext = _extraction(risks=[{
            "risk": "Scope creep possible",
            "evidence": [{"start": 5.0, "end": 10.0}],
        }])
        mm.merge(memory_state=state, extraction=ext, source_chunk_id=0, speakers_from_window=set())
        assert len(state.risks) == 1
        assert state.risks[0].risk == "Scope creep possible"

    def test_generate_id_is_deterministic(self):
        mm = MemoryManager()
        id1 = mm.generate_id("action", "Deploy app")
        id2 = mm.generate_id("action", "Deploy app")
        assert id1 == id2

    def test_generate_id_different_for_different_inputs(self):
        mm = MemoryManager()
        id1 = mm.generate_id("action", "Deploy app")
        id2 = mm.generate_id("action", "Write tests")
        assert id1 != id2
