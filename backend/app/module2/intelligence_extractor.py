from __future__ import annotations

import os

from ..models.memory import MemoryState
from .llm_client import BaseLLMClient, create_llm_client
from .memory_manager import MemoryManager
from .prompt_builder import build_system_prompt, build_user_prompt
from .trust_validator import TrustValidator, ValidatedExtraction


class IntelligenceExtractor:
    """Prompt -> LLM -> validator -> memory merge orchestration for Module 2."""

    def __init__(
        self,
        *,
        llm_client: BaseLLMClient | None = None,
        trust_validator: TrustValidator | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.llm_client = llm_client or create_llm_client()
        self.trust_validator = trust_validator or TrustValidator()
        self.memory_manager = memory_manager or MemoryManager()
        self.consolidation_enabled = os.getenv("LLM_CONSOLIDATION_ENABLED", "false").lower() == "true"

    def run(self, memory_state: MemoryState) -> ValidatedExtraction:
        previous_window = list(memory_state.previous_llm_window)
        new_window = list(memory_state.llm_transcript_buffer)
        if not new_window:
            return ValidatedExtraction()

        extraction = self._single_pass(
            memory_state=memory_state,
            previous_window=previous_window,
            new_window=new_window,
            consolidation=False,
        )
        self.memory_manager.merge(
            memory_state=memory_state,
            extraction=extraction,
            source_chunk_id=memory_state.chunk_id,
            speakers_from_window=self._window_speakers(previous_window + new_window),
        )

        if self.consolidation_enabled and (memory_state.action_items or memory_state.decisions or memory_state.risks):
            consolidation = self._single_pass(
                memory_state=memory_state,
                previous_window=previous_window,
                new_window=new_window,
                consolidation=True,
            )
            self.memory_manager.merge(
                memory_state=memory_state,
                extraction=consolidation,
                source_chunk_id=memory_state.chunk_id,
                speakers_from_window=self._window_speakers(previous_window + new_window),
            )
            extraction.violations.extend(consolidation.violations)

        if memory_state.chunk_history:
            memory_state.chunk_history[-1].trust_violations += len(extraction.violations)
        return extraction

    def _single_pass(
        self,
        *,
        memory_state: MemoryState,
        previous_window: list[dict],
        new_window: list[dict],
        consolidation: bool,
    ) -> ValidatedExtraction:
        system_prompt = build_system_prompt(consolidation=consolidation)
        user_prompt = build_user_prompt(
            memory_state=memory_state,
            previous_window=previous_window,
            new_window=new_window,
            consolidation=consolidation,
        )
        raw_response = self.llm_client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return self.trust_validator.validate(
            raw_response=raw_response,
            memory_state=memory_state,
            previous_window=previous_window,
            new_window=new_window,
        )

    def consolidate(self, memory_state: MemoryState) -> ValidatedExtraction:
        """Final consolidation pass — always runs on meeting_end regardless of env flag.

        Refines the running summary, deduplicates items, and normalises deadlines.
        Never overwrites human-locked fields (enforced by memory_manager).
        """
        has_content = bool(
            memory_state.running_summary
            or memory_state.action_items
            or memory_state.decisions
            or memory_state.risks
        )
        if not has_content:
            return ValidatedExtraction()

        # Use previous window as context; fall back to empty list on first window
        previous_window = list(memory_state.previous_llm_window)
        # For consolidation the "new window" is the same as previous — we're refining, not extracting
        new_window = previous_window or list(memory_state.llm_transcript_buffer)

        consolidation = self._single_pass(
            memory_state=memory_state,
            previous_window=previous_window,
            new_window=new_window,
            consolidation=True,
        )
        self.memory_manager.merge(
            memory_state=memory_state,
            extraction=consolidation,
            source_chunk_id=memory_state.chunk_id,
            speakers_from_window=[],
        )
        return consolidation

    @staticmethod
    def _window_speakers(window: list[dict]) -> list[str]:
        return sorted({item.get("speaker", "") for item in window if item.get("speaker")})
