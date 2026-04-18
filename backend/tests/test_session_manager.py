"""Unit tests for SessionManager."""
import pytest

from app.session_manager import SessionManager
from app.models.utterance import Utterance, ChunkTranscript


def _utterance(speaker: str = "SPEAKER_01", text: str = "Hello", start: float = 0.0, end: float = 2.0) -> Utterance:
    return Utterance(speaker=speaker, text=text, start_time=start, end_time=end, word_count=1, confidence=0.9)


def _chunk(meeting_id: str, chunk_id: int = 0, utterances: list[Utterance] | None = None) -> ChunkTranscript:
    u = utterances or [_utterance()]
    return ChunkTranscript(
        meeting_id=meeting_id,
        chunk_id=chunk_id,
        chunk_start=0.0,
        chunk_end=15.0,
        display_utterances=u,
        llm_utterances=u,
        processing_time_sec=1.0,
        asr_model="small",
    )


class TestSessionManager:
    def test_create_session_generates_id(self):
        sm = SessionManager()
        s = sm.create_session()
        assert s.meeting_id
        assert s.is_active

    def test_create_session_uses_provided_id(self):
        sm = SessionManager()
        s = sm.create_session("my-meeting")
        assert s.meeting_id == "my-meeting"

    def test_get_session_raises_on_unknown(self):
        sm = SessionManager()
        with pytest.raises(KeyError):
            sm.get_session("nope")

    def test_get_or_create_creates_if_missing(self):
        sm = SessionManager()
        s = sm.get_or_create_session("abc")
        assert s.meeting_id == "abc"
        s2 = sm.get_or_create_session("abc")
        assert s is s2

    def test_list_sessions(self):
        sm = SessionManager()
        sm.create_session("m1")
        sm.create_session("m2")
        ids = [s.meeting_id for s in sm.list_sessions()]
        assert "m1" in ids and "m2" in ids

    def test_end_session(self):
        sm = SessionManager()
        sm.create_session("end-test")
        session = sm.end_session("end-test")
        assert not session.is_active

    def test_append_chunk_transcript(self):
        sm = SessionManager()
        sm.create_session("tx")
        chunk = _chunk("tx", chunk_id=0)
        sm.append_chunk_transcript(chunk)
        session = sm.get_session("tx")
        assert len(session.display_transcript_buffer) == 1
        assert "SPEAKER_01" in session.known_speakers

    def test_asr_chunks_counter_increments(self):
        sm = SessionManager()
        sm.create_session("counter")
        sm.append_chunk_transcript(_chunk("counter", 0))
        sm.append_chunk_transcript(_chunk("counter", 1))
        session = sm.get_session("counter")
        assert session.asr_chunks_since_last_llm == 2

    def test_mark_llm_window_processed_resets_counter(self):
        sm = SessionManager()
        sm.create_session("llm-reset")
        sm.append_chunk_transcript(_chunk("llm-reset", 0))
        sm.append_chunk_transcript(_chunk("llm-reset", 1))
        sm.mark_llm_window_processed("llm-reset")
        session = sm.get_session("llm-reset")
        assert session.asr_chunks_since_last_llm == 0

    def test_update_summary_no_lock(self):
        sm = SessionManager()
        sm.create_session("sum")
        sm.update_summary("sum", "AI summary", human_locked=False)
        s = sm.get_session("sum")
        assert s.running_summary == "AI summary"
        assert not s.summary_human_locked

    def test_update_summary_with_lock(self):
        sm = SessionManager()
        sm.create_session("sum-lock")
        sm.update_summary("sum-lock", "Human summary", human_locked=True)
        s = sm.get_session("sum-lock")
        assert s.summary_human_locked

    def test_human_locked_summary_not_overwritten(self):
        sm = SessionManager()
        sm.create_session("no-overwrite")
        sm.update_summary("no-overwrite", "Human text", human_locked=True)
        sm.update_summary("no-overwrite", "AI text", human_locked=False)
        s = sm.get_session("no-overwrite")
        assert s.running_summary == "Human text"

    def test_add_action_item(self):
        sm = SessionManager()
        sm.create_session("action")
        sm.add_action_item("action", {"task": "Review PR", "owner": "Alice", "deadline": "", "priority": "High"})
        s = sm.get_session("action")
        assert len(s.action_items) == 1
        assert s.action_items[0].human_locked

    def test_toggle_action_status(self):
        sm = SessionManager()
        sm.create_session("toggle")
        sm.add_action_item("toggle", {"task": "Task", "owner": "", "deadline": "", "priority": "Low"})
        item_id = sm.get_session("toggle").action_items[0].id
        sm.toggle_action_status("toggle", item_id, "done")
        s = sm.get_session("toggle")
        assert s.action_items[0].status == "done"

    def test_set_item_deleted_action(self):
        sm = SessionManager()
        sm.create_session("del")
        sm.add_action_item("del", {"task": "Deletable", "owner": "", "deadline": "", "priority": "Low"})
        item_id = sm.get_session("del").action_items[0].id
        sm.set_item_deleted("del", "action", item_id, True)
        s = sm.get_session("del")
        assert s.action_items[0].status == "deleted"

    def test_serialize_session_returns_dict(self):
        sm = SessionManager()
        sm.create_session("serial")
        result = sm.serialize_session("serial")
        assert isinstance(result, dict)
        assert result["meeting_id"] == "serial"
