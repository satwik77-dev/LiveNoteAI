from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..models.intelligence import TrustViolation
from ..models.memory import MemoryState


@dataclass(slots=True)
class ValidatedExtraction:
    summary: str = ""
    action_items: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    violations: list[TrustViolation] = field(default_factory=list)


class TrustValidator:
    """Applies the seven trust rules before any memory mutation."""

    def validate(
        self,
        *,
        raw_response: str,
        memory_state: MemoryState,
        previous_window: list[dict[str, Any]],
        new_window: list[dict[str, Any]],
    ) -> ValidatedExtraction:
        payload, parse_violations = self._parse_json(raw_response)
        validated = ValidatedExtraction(violations=parse_violations)
        if not payload:
            return validated

        summary = payload.get("summary", "")
        if isinstance(summary, str):
            validated.summary = summary.strip()

        window_bounds = self._window_bounds(previous_window + new_window)
        validated.action_items = self._validate_actions(
            payload.get("action_items"),
            memory_state=memory_state,
            window_bounds=window_bounds,
        )
        validated.violations.extend(self._buffered_violations)
        self._buffered_violations = []

        validated.decisions = self._validate_simple_items(
            payload.get("decisions"),
            item_key="decision",
            existing_items=memory_state.decisions,
            window_bounds=window_bounds,
        )
        validated.violations.extend(self._buffered_violations)
        self._buffered_violations = []

        validated.risks = self._validate_simple_items(
            payload.get("risks"),
            item_key="risk",
            existing_items=memory_state.risks,
            window_bounds=window_bounds,
        )
        validated.violations.extend(self._buffered_violations)
        self._buffered_violations = []
        return validated

    def __init__(self) -> None:
        self._buffered_violations: list[TrustViolation] = []

    def _parse_json(self, raw_response: str) -> tuple[dict[str, Any], list[TrustViolation]]:
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return {}, [TrustViolation(rule=4, description=f"Invalid JSON from model: {exc.msg}", auto_fixed=False)]
        if not isinstance(parsed, dict):
            return {}, [TrustViolation(rule=4, description="Model output must be a JSON object.", auto_fixed=False)]
        return parsed, []

    def _validate_actions(
        self,
        raw_items: Any,
        *,
        memory_state: MemoryState,
        window_bounds: tuple[float, float] | None,
    ) -> list[dict[str, Any]]:
        items = raw_items if isinstance(raw_items, list) else []
        validated: list[dict[str, Any]] = []
        seen: set[str] = set()
        locked_keys = {
            self._normalize(f"{item.owner}:{item.human_value or item.task}")
            for item in memory_state.action_items
            if item.human_locked
        }
        for raw_item in items:
            if not isinstance(raw_item, dict):
                self._buffered_violations.append(
                    TrustViolation(rule=4, description="Skipped non-object action item.", auto_fixed=True)
                )
                continue
            task = str(raw_item.get("task", "")).strip()
            if not task:
                self._buffered_violations.append(
                    TrustViolation(rule=4, description="Skipped action item without task text.", auto_fixed=True)
                )
                continue

            owner = str(raw_item.get("owner", "unassigned")).strip() or "unassigned"
            if owner not in set(memory_state.known_speakers):
                owner = "unassigned"
                raw_item["needs_review"] = True
                self._buffered_violations.append(
                    TrustViolation(rule=1, description=f"Unknown action owner for task '{task}'.", auto_fixed=True)
                )

            key = self._normalize(f"{owner}:{task}")
            if key in seen:
                self._buffered_violations.append(
                    TrustViolation(rule=5, description=f"Deduplicated repeated action item '{task}'.", auto_fixed=True)
                )
                continue
            seen.add(key)

            if key in locked_keys:
                raw_item["needs_review"] = True
                self._buffered_violations.append(
                    TrustViolation(rule=6, description=f"Action overlaps a human-locked item: '{task}'.", auto_fixed=True)
                )

            evidence = self._validated_evidence(raw_item.get("evidence"), window_bounds)
            if not evidence:
                self._buffered_violations.append(
                    TrustViolation(rule=7, description=f"Removed action without valid evidence: '{task}'.", auto_fixed=True)
                )
                continue

            deadline = str(raw_item.get("deadline", "unspecified")).strip() or "unspecified"
            deadline_normalized = self._validate_deadline(str(raw_item.get("deadline_normalized", deadline)).strip())
            if deadline != "unspecified" and deadline_normalized == "unspecified":
                raw_item["needs_review"] = True
                self._buffered_violations.append(
                    TrustViolation(rule=3, description=f"Could not normalize deadline for task '{task}'.", auto_fixed=True)
                )

            validated.append(
                {
                    "task": task,
                    "owner": owner,
                    "deadline": deadline,
                    "deadline_normalized": deadline_normalized,
                    "priority": self._validate_priority(str(raw_item.get("priority", "Medium"))),
                    "status": "open",
                    "needs_review": bool(raw_item.get("needs_review", False)),
                    "evidence": evidence,
                }
            )
        return validated

    def _validate_simple_items(
        self,
        raw_items: Any,
        *,
        item_key: str,
        existing_items: list[Any],
        window_bounds: tuple[float, float] | None,
    ) -> list[dict[str, Any]]:
        items = raw_items if isinstance(raw_items, list) else []
        validated: list[dict[str, Any]] = []
        seen: set[str] = set()
        locked_keys = {
            self._normalize(getattr(item, "human_value", None) or getattr(item, item_key))
            for item in existing_items
            if getattr(item, "human_locked", False)
        }
        for raw_item in items:
            if not isinstance(raw_item, dict):
                self._buffered_violations.append(
                    TrustViolation(rule=4, description=f"Skipped non-object {item_key}.", auto_fixed=True)
                )
                continue
            text = str(raw_item.get(item_key, "")).strip()
            if not text:
                self._buffered_violations.append(
                    TrustViolation(rule=4, description=f"Skipped {item_key} without text.", auto_fixed=True)
                )
                continue

            normalized = self._normalize(text)
            if normalized in seen:
                self._buffered_violations.append(
                    TrustViolation(rule=5, description=f"Deduplicated repeated {item_key} '{text}'.", auto_fixed=True)
                )
                continue
            seen.add(normalized)

            if normalized in locked_keys:
                self._buffered_violations.append(
                    TrustViolation(rule=6, description=f"{item_key.title()} overlaps a human-locked item: '{text}'.", auto_fixed=True)
                )

            evidence = self._validated_evidence(raw_item.get("evidence"), window_bounds)
            if not evidence:
                self._buffered_violations.append(
                    TrustViolation(rule=7, description=f"Removed {item_key} without valid evidence: '{text}'.", auto_fixed=True)
                )
                continue

            validated.append({item_key: text, "evidence": evidence})
        return validated

    def _validated_evidence(
        self,
        raw_evidence: Any,
        window_bounds: tuple[float, float] | None,
    ) -> list[dict[str, float]]:
        if not isinstance(raw_evidence, list):
            self._buffered_violations.append(
                TrustViolation(rule=7, description="Item missing evidence list.", auto_fixed=True)
            )
            return []
        validated: list[dict[str, float]] = []
        min_time, max_time = window_bounds if window_bounds else (0.0, float("inf"))
        for span in raw_evidence:
            if not isinstance(span, dict):
                continue
            try:
                start = float(span.get("start"))
                end = float(span.get("end"))
            except (TypeError, ValueError):
                continue
            if end < start:
                self._buffered_violations.append(
                    TrustViolation(rule=2, description="Dropped evidence span with end < start.", auto_fixed=True)
                )
                continue
            if start < min_time or end > max_time:
                self._buffered_violations.append(
                    TrustViolation(rule=2, description="Dropped evidence span outside transcript window.", auto_fixed=True)
                )
                continue
            validated.append({"start": start, "end": end})
        return validated

    @staticmethod
    def _validate_deadline(value: str) -> str:
        if not value or value.lower() == "unspecified":
            return "unspecified"
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except ValueError:
                return "unspecified"

    @staticmethod
    def _validate_priority(value: str) -> str:
        normalized = value.strip().capitalize()
        if normalized not in {"Low", "Medium", "High"}:
            return "Medium"
        return normalized

    @staticmethod
    def _window_bounds(window: list[dict[str, Any]]) -> tuple[float, float] | None:
        starts = [float(item["start_time"]) for item in window if "start_time" in item]
        ends = [float(item["end_time"]) for item in window if "end_time" in item]
        if not starts or not ends:
            return None
        return min(starts), max(ends)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())
