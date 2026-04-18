from __future__ import annotations

import hashlib
import re
from typing import Any

from ..models.intelligence import ActionItem, Decision, EvidenceSpan, Risk
from ..models.memory import MemoryState
from .trust_validator import ValidatedExtraction


class MemoryManager:
    """Merges validated model output into the rolling in-memory meeting state."""

    def merge(
        self,
        *,
        memory_state: MemoryState,
        extraction: ValidatedExtraction,
        source_chunk_id: int,
        speakers_from_window: list[str],
    ) -> MemoryState:
        memory_state.known_speakers = sorted({*memory_state.known_speakers, *speakers_from_window})
        if extraction.summary and not memory_state.summary_human_locked:
            memory_state.running_summary = extraction.summary
        elif memory_state.summary_human_locked and memory_state.summary_human_value:
            memory_state.running_summary = memory_state.summary_human_value

        memory_state.action_items = self._merge_actions(
            existing=memory_state.action_items,
            incoming=extraction.action_items,
            source_chunk_id=source_chunk_id,
        )
        memory_state.decisions = self._merge_simple_items(
            existing=memory_state.decisions,
            incoming=extraction.decisions,
            item_key="decision",
            source_chunk_id=source_chunk_id,
            model_cls=Decision,
        )
        memory_state.risks = self._merge_simple_items(
            existing=memory_state.risks,
            incoming=extraction.risks,
            item_key="risk",
            source_chunk_id=source_chunk_id,
            model_cls=Risk,
        )
        return memory_state

    def _merge_actions(
        self,
        *,
        existing: list[ActionItem],
        incoming: list[dict[str, Any]],
        source_chunk_id: int,
    ) -> list[ActionItem]:
        by_key = {self._action_key(item.owner, item.task): item for item in existing}
        merged = list(existing)
        for raw in incoming:
            key = self._action_key(raw["owner"], raw["task"])
            evidence = self._merge_evidence(
                by_key[key].evidence if key in by_key else [],
                raw.get("evidence", []),
            )
            if key in by_key:
                current = by_key[key]
                if current.human_locked:
                    current.evidence = evidence
                    continue
                current.task = raw["task"]
                current.owner = raw["owner"]
                current.deadline = raw["deadline"]
                current.deadline_normalized = raw["deadline_normalized"]
                current.priority = raw["priority"]
                current.needs_review = raw["needs_review"]
                current.evidence = evidence
                current.chunk_origin = source_chunk_id
                continue
            created = ActionItem(
                id=self.generate_id("act", key),
                task=raw["task"],
                owner=raw["owner"],
                deadline=raw["deadline"],
                deadline_normalized=raw["deadline_normalized"],
                priority=raw["priority"],
                status=raw.get("status", "open"),
                needs_review=raw.get("needs_review", False),
                evidence=evidence,
                chunk_origin=source_chunk_id,
            )
            by_key[key] = created
            merged.append(created)
        return merged

    def _merge_simple_items(
        self,
        *,
        existing: list[Any],
        incoming: list[dict[str, Any]],
        item_key: str,
        source_chunk_id: int,
        model_cls: type[Any],
    ) -> list[Any]:
        by_key = {self._normalize(getattr(item, item_key)): item for item in existing}
        merged = list(existing)
        for raw in incoming:
            text = raw[item_key]
            normalized = self._normalize(text)
            evidence = self._merge_evidence(
                by_key[normalized].evidence if normalized in by_key else [],
                raw.get("evidence", []),
            )
            if normalized in by_key:
                current = by_key[normalized]
                current.evidence = evidence
                if not current.human_locked:
                    setattr(current, item_key, text)
                    current.chunk_origin = source_chunk_id
                continue
            created = model_cls(
                id=self.generate_id(item_key[:4], normalized),
                **{item_key: text},
                evidence=evidence,
                chunk_origin=source_chunk_id,
            )
            by_key[normalized] = created
            merged.append(created)
        return merged

    @staticmethod
    def generate_id(prefix: str, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
        return f"{prefix}_{digest}"

    @staticmethod
    def _action_key(owner: str, task: str) -> str:
        return MemoryManager._normalize(f"{owner}:{task}")

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _merge_evidence(
        existing: list[EvidenceSpan],
        incoming: list[dict[str, float]],
    ) -> list[EvidenceSpan]:
        seen: set[tuple[float, float]] = set()
        merged: list[EvidenceSpan] = []
        for span in list(existing) + [EvidenceSpan(**item) for item in incoming]:
            key = (span.start, span.end)
            if key in seen:
                continue
            seen.add(key)
            merged.append(span)
        merged.sort(key=lambda item: (item.start, item.end))
        return merged
