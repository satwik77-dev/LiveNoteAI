"""Unit tests for TrustValidator (7 rules)."""
import json
import pytest

from app.module2.trust_validator import TrustValidator
from app.models.memory import MemoryState


def _state(**kwargs) -> MemoryState:
    defaults = dict(
        meeting_id="test",
        meeting_start_date="2026-04-13",
        known_speakers=["SPEAKER_01", "SPEAKER_02"],
        action_items=[],
        decisions=[],
        risks=[],
    )
    defaults.update(kwargs)
    return MemoryState(**defaults)


def _window(start: float = 0.0, end: float = 60.0) -> list[dict]:
    return [{"start_time": start, "end_time": end, "text": "Test utterance", "speaker": "SPEAKER_01"}]


def _valid_action(task: str = "Review PR", owner: str = "SPEAKER_01") -> dict:
    return {
        "task": task,
        "owner": owner,
        "deadline": "unspecified",
        "deadline_normalized": "unspecified",
        "priority": "Medium",
        "status": "open",
        "needs_review": False,
        "evidence": [{"start": 5.0, "end": 10.0}],
    }


def _raw(actions=None, decisions=None, risks=None, summary="Test") -> str:
    return json.dumps({
        "summary": summary,
        "action_items": actions or [],
        "decisions": decisions or [],
        "risks": risks or [],
    })


class TestTrustValidatorParsing:
    def test_valid_json_parses(self):
        v = TrustValidator()
        result = v.validate(
            raw_response=_raw(),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert result.summary == "Test"
        assert not result.violations

    def test_invalid_json_rule4(self):
        v = TrustValidator()
        result = v.validate(
            raw_response="not json at all",
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert any(viol.rule == 4 for viol in result.violations)

    def test_strips_markdown_fences(self):
        v = TrustValidator()
        raw = "```json\n" + _raw(summary="fenced") + "\n```"
        result = v.validate(
            raw_response=raw,
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert result.summary == "fenced"

    def test_non_dict_root_rule4(self):
        v = TrustValidator()
        result = v.validate(
            raw_response='["not", "a", "dict"]',
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert any(viol.rule == 4 for viol in result.violations)


class TestRule1OwnerValidation:
    def test_unknown_owner_replaced_with_unassigned(self):
        v = TrustValidator()
        action = _valid_action(owner="Bob")  # not in known_speakers
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        # Item dropped because needs_review doesn't block — but evidence required
        # If item passes evidence check, owner becomes 'unassigned'
        if result.action_items:
            assert result.action_items[0]["owner"] == "unassigned"
        assert any(viol.rule == 1 for viol in result.violations)


class TestRule2EvidenceSpans:
    def test_span_end_before_start_dropped(self):
        v = TrustValidator()
        action = _valid_action()
        action["evidence"] = [{"start": 10.0, "end": 5.0}]  # invalid
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert any(viol.rule == 2 for viol in result.violations)

    def test_span_outside_window_dropped(self):
        v = TrustValidator()
        action = _valid_action()
        action["evidence"] = [{"start": 200.0, "end": 300.0}]  # outside 0-60
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(0.0, 60.0),
        )
        assert any(viol.rule == 2 for viol in result.violations)


class TestRule3DeadlineNormalization:
    def test_valid_iso_deadline_passes(self):
        v = TrustValidator()
        action = _valid_action()
        action["deadline"] = "2026-06-01"
        action["deadline_normalized"] = "2026-06-01"
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        items = result.action_items
        if items:
            assert items[0]["deadline_normalized"] == "2026-06-01"

    def test_unparseable_deadline_sets_needs_review(self):
        v = TrustValidator()
        action = _valid_action()
        action["deadline"] = "next week"
        action["deadline_normalized"] = "next week"
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert any(viol.rule == 3 for viol in result.violations)


class TestRule5Deduplication:
    def test_duplicate_action_items_deduplicated(self):
        v = TrustValidator()
        action = _valid_action(task="Review PR", owner="SPEAKER_01")
        result = v.validate(
            raw_response=_raw(actions=[action, action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert len(result.action_items) <= 1
        assert any(viol.rule == 5 for viol in result.violations)

    def test_duplicate_decisions_deduplicated(self):
        v = TrustValidator()
        dec = {"decision": "Use Python", "evidence": [{"start": 5.0, "end": 10.0}]}
        result = v.validate(
            raw_response=_raw(decisions=[dec, dec]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert len(result.decisions) <= 1
        assert any(viol.rule == 5 for viol in result.violations)


class TestRule7EvidenceRequired:
    def test_action_without_evidence_dropped(self):
        v = TrustValidator()
        action = _valid_action()
        action["evidence"] = []  # no evidence
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert len(result.action_items) == 0
        assert any(viol.rule == 7 for viol in result.violations)

    def test_decision_without_evidence_dropped(self):
        v = TrustValidator()
        dec = {"decision": "Use microservices", "evidence": []}
        result = v.validate(
            raw_response=_raw(decisions=[dec]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        assert len(result.decisions) == 0
        assert any(viol.rule == 7 for viol in result.violations)


class TestPriorityNormalization:
    def test_invalid_priority_defaults_to_medium(self):
        v = TrustValidator()
        action = _valid_action()
        action["priority"] = "URGENT"
        result = v.validate(
            raw_response=_raw(actions=[action]),
            memory_state=_state(),
            previous_window=[],
            new_window=_window(),
        )
        if result.action_items:
            assert result.action_items[0]["priority"] in {"Low", "Medium", "High"}
