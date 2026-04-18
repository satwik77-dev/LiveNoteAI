from __future__ import annotations

from ..models.utterance import Utterance

FILLER_WORDS = {
    "ah",
    "eh",
    "er",
    "hmm",
    "hm",
    "mm",
    "mmm",
    "oh",
    "okay",
    "ok",
    "right",
    "uh",
    "uhh",
    "um",
    "umm",
    "yeah",
    "yep",
    "yes",
}


def filter_utterances(utterances: list[Utterance]) -> tuple[list[Utterance], list[Utterance]]:
    display_utterances = [utterance for utterance in utterances if _keep_for_display(utterance)]
    llm_utterances = [utterance for utterance in utterances if _keep_for_llm(utterance)]
    return display_utterances, llm_utterances


def _keep_for_display(utterance: Utterance) -> bool:
    text = utterance.text.strip()
    duration = utterance.end_time - utterance.start_time
    if duration < 0.3:
        return False
    if len(text) <= 1:
        return False
    return True


def _keep_for_llm(utterance: Utterance) -> bool:
    if not _keep_for_display(utterance):
        return False

    text = utterance.text.strip().lower()
    normalized = "".join(character for character in text if character.isalpha() or character.isspace()).strip()
    duration = utterance.end_time - utterance.start_time

    if utterance.confidence < 0.3:
        return False
    if utterance.word_count < 2:
        return False
    if duration < 0.5:
        return False
    if normalized in FILLER_WORDS:
        return False
    return True
