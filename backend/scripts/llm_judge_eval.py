#!/usr/bin/env python3
"""LLM Judge Evaluation Script — Phase 3, Step 3.10

Uses the loser LLM model (determined in Notebook 10) to judge the winner
model's extracted action items from AMI meeting transcripts.

Usage:
    cd backend
    python scripts/llm_judge_eval.py \
        --input results/winner_outputs.json \
        --report results/judge_report.json

Input JSON format (list of meeting dicts):
    [
        {
            "meeting_id": "ES2008a",
            "transcript": "full transcript text...",
            "action_items": [
                {"id": "act_...", "task": "...", "owner": "...", "deadline": "...", ...}
            ]
        },
        ...
    ]

Required env vars (see backend/.env):
    JUDGE_LLM_MODE       — ollama or groq
    JUDGE_OLLAMA_MODEL   — model name if mode=ollama (e.g. mistral:7b)
    JUDGE_GROQ_MODEL     — model name if mode=groq
    JUDGE_GROQ_API_KEY   — Groq API key if mode=groq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Support running from backend/ or repo root
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_REPO_ROOT = _BACKEND.parent
sys.path.insert(0, str(_REPO_ROOT))

load_dotenv(_BACKEND / ".env")

from backend.app.module2.llm_client import BaseLLMClient, GroqClient, OllamaClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are an expert meeting-analysis judge evaluating action items "
    "extracted by a production AI system from real meeting transcripts. "
    "You are used as a pre-trained model through inference only — no fine-tuning is performed. "
    "Return JSON only. No markdown, no prose, no code fences. "
    "Your response must be a JSON object with a single key 'judgements' "
    "containing a list of judgement objects — one per action item provided."
)

_ITEM_SCHEMA = {
    "action_item_id": "string — the id field of the evaluated item",
    "task": "string — the task text being judged (copy from input)",
    "correctness_score": "integer 1-5 — does the task accurately reflect what was committed to in the transcript?",
    "specificity_score": "integer 1-5 — is the task specific and actionable enough to execute without guessing?",
    "grounding_score": "integer 1-5 — is the extracted item clearly grounded in the transcript content?",
    "hallucination_detected": "boolean — is any part of this item invented or not supported by the transcript?",
    "reasoning": "string — 1-2 sentence explanation of the scores",
}


def build_judge_prompt(
    *,
    transcript: str,
    action_items: list[dict[str, Any]],
) -> str:
    schema = {"judgements": [_ITEM_SCHEMA]}
    sections = [
        "Evaluate the following action items extracted from this meeting transcript.",
        "Score each item honestly on the 1-5 scales. Penalise vague tasks, hallucinations, and unsupported evidence.",
        "Scoring guide: 5 = excellent, 4 = good, 3 = acceptable, 2 = poor, 1 = completely wrong.",
        f"Meeting transcript:\n{transcript[:6000]}",  # guard against very long transcripts
        f"Extracted action items:\n{json.dumps(action_items, indent=2)}",
        f"Required JSON schema:\n{json.dumps(schema, indent=2)}",
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Client factory (judge-specific env vars with fallback)
# ---------------------------------------------------------------------------

def create_judge_client() -> BaseLLMClient:
    mode = os.getenv("JUDGE_LLM_MODE", "ollama").strip().lower()
    if mode == "groq":
        return GroqClient(
            api_key=os.getenv("JUDGE_GROQ_API_KEY") or os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("JUDGE_GROQ_MODEL") or os.getenv("GROQ_MODEL", "mistral-saba-24b"),
            timeout_sec=float(os.getenv("GROQ_TIMEOUT_SEC", "30")),
        )
    return OllamaClient(
        model=os.getenv("JUDGE_OLLAMA_MODEL", "mistral:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout_sec=float(os.getenv("OLLAMA_TIMEOUT_SEC", "120")),
    )


# ---------------------------------------------------------------------------
# Per-meeting judge run
# ---------------------------------------------------------------------------

def run_judge_for_meeting(
    *,
    client: BaseLLMClient,
    meeting_id: str,
    transcript: str,
    action_items: list[dict[str, Any]],
) -> dict[str, Any]:
    if not action_items:
        logger.info("[%s] No action items — skipping.", meeting_id)
        return {"meeting_id": meeting_id, "judgements": [], "error": None}

    prompt = build_judge_prompt(transcript=transcript, action_items=action_items)
    logger.info("[%s] Judging %d action item(s)...", meeting_id, len(action_items))

    try:
        raw = client.complete_json(system_prompt=JUDGE_SYSTEM_PROMPT, user_prompt=prompt)
        parsed = json.loads(raw)
        judgements = parsed.get("judgements")
        if not isinstance(judgements, list):
            raise ValueError(f"Expected list under 'judgements', got: {type(judgements)}")
        logger.info("[%s] Received %d judgement(s).", meeting_id, len(judgements))
        return {"meeting_id": meeting_id, "judgements": judgements, "error": None}
    except Exception as exc:
        logger.warning("[%s] Judge call failed: %s", meeting_id, exc)
        return {"meeting_id": meeting_id, "judgements": [], "error": str(exc)}


# ---------------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, Any]:
    all_judgements: list[dict[str, Any]] = [
        j for r in results for j in r.get("judgements", [])
    ]
    if not all_judgements:
        return {
            "total_items_judged": 0,
            "note": "No judgements collected — check errors in per_meeting results.",
        }

    def _mean(key: str) -> float:
        vals = [j[key] for j in all_judgements if isinstance(j.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    hallucination_count = sum(
        1 for j in all_judgements if j.get("hallucination_detected") is True
    )
    error_count = sum(1 for r in results if r.get("error"))

    return {
        "total_meetings": len(results),
        "meetings_with_errors": error_count,
        "total_items_judged": len(all_judgements),
        "mean_correctness_score": _mean("correctness_score"),
        "mean_specificity_score": _mean("specificity_score"),
        "mean_grounding_score": _mean("grounding_score"),
        "hallucination_rate": round(hallucination_count / len(all_judgements), 3),
        "hallucination_count": hallucination_count,
        "composite_score": round(
            (_mean("correctness_score") + _mean("specificity_score") + _mean("grounding_score")) / 3,
            3,
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM Judge Evaluation — uses the loser model to judge winner model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="JSON file containing winner model outputs (list of meeting dicts).",
    )
    parser.add_argument(
        "--report",
        required=True,
        metavar="PATH",
        help="Output path for the judge report JSON.",
    )
    parser.add_argument(
        "--max-meetings",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N meetings (useful for quick checks).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    winner_outputs: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(winner_outputs, list):
        logger.error("Input must be a JSON array of meeting objects.")
        sys.exit(1)

    if args.max_meetings:
        winner_outputs = winner_outputs[: args.max_meetings]
        logger.info("Limiting to first %d meeting(s).", args.max_meetings)

    client = create_judge_client()
    logger.info(
        "Judge client: %s | model: %s",
        type(client).__name__,
        getattr(client, "model", "unknown"),
    )

    results: list[dict[str, Any]] = []
    for entry in winner_outputs:
        meeting_id = str(entry.get("meeting_id", "unknown"))
        transcript = str(entry.get("transcript", ""))
        action_items = entry.get("action_items", [])
        result = run_judge_for_meeting(
            client=client,
            meeting_id=meeting_id,
            transcript=transcript,
            action_items=action_items,
        )
        results.append(result)

    aggregate = aggregate_scores(results)
    report: dict[str, Any] = {
        "judge_model": getattr(client, "model", "unknown"),
        "judge_mode": os.getenv("JUDGE_LLM_MODE", "ollama"),
        "aggregate": aggregate,
        "per_meeting": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Report written → %s", report_path)
    logger.info("Aggregate:\n%s", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()