from __future__ import annotations

import json
from typing import Any

from ..models.memory import MemoryState


def build_system_prompt(*, consolidation: bool = False) -> str:
    mode_line = (
        "You are performing a consolidation pass over existing meeting memory."
        if consolidation
        else "You are extracting live meeting intelligence from a new transcript window."
    )
    return "\n".join(
        [
            mode_line,
            "You are used as a pre-trained model through inference only. No fine-tuning is being performed.",
            "Return JSON only. Do not use markdown, prose, or code fences.",
            "Do not invent people, facts, deadlines, commitments, or evidence.",
            "Owners must come from the provided known speakers list. If unknown, use 'unassigned'.",
            "Every action item, decision, and risk must include evidence spans grounded in the provided transcript windows.",
            "Prefer precision over recall. If the transcript is ambiguous, omit the item.",
            "Keep the running summary concise and factual.",
            "Output schema keys: summary, action_items, decisions, risks.",
        ]
    )


def build_user_prompt(
    *,
    memory_state: MemoryState,
    previous_window: list[dict[str, Any]],
    new_window: list[dict[str, Any]],
    consolidation: bool = False,
) -> str:
    schema = {
        "summary": "string",
        "action_items": [
            {
                "task": "string",
                "owner": "string",
                "deadline": "string",
                "deadline_normalized": "YYYY-MM-DD or unspecified",
                "priority": "Low|Medium|High",
                "status": "open",
                "needs_review": False,
                "evidence": [{"start": 0.0, "end": 0.0}],
            }
        ],
        "decisions": [{"decision": "string", "evidence": [{"start": 0.0, "end": 0.0}]}],
        "risks": [{"risk": "string", "evidence": [{"start": 0.0, "end": 0.0}]}],
    }
    memory_payload = {
        "meeting_id": memory_state.meeting_id,
        "meeting_start_date": memory_state.meeting_start_date,
        "running_summary": memory_state.summary_human_value
        if memory_state.summary_human_locked and memory_state.summary_human_value
        else memory_state.running_summary,
        "summary_human_locked": memory_state.summary_human_locked,
        "known_speakers": memory_state.known_speakers,
        "action_items": [item.model_dump() for item in memory_state.action_items],
        "decisions": [item.model_dump() for item in memory_state.decisions],
        "risks": [item.model_dump() for item in memory_state.risks],
    }
    instruction = (
        "Consolidate the existing memory. Tighten duplicates, preserve facts, and do not create new unsupported items."
        if consolidation
        else "Use the previous window only as context continuity. Extract or update items based on the new window."
    )
    return "\n\n".join(
        [
            instruction,
            f"Known speakers: {json.dumps(memory_state.known_speakers)}",
            f"Current memory state:\n{json.dumps(memory_payload, indent=2)}",
            f"Previous transcript window:\n{json.dumps(previous_window, indent=2)}",
            f"New transcript window:\n{json.dumps(new_window, indent=2)}",
            f"Required JSON schema:\n{json.dumps(schema, indent=2)}",
        ]
    )
