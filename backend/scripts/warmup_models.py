#!/usr/bin/env python3
"""Model warmup script — Phase 6, Docker build step.

Pre-downloads faster-whisper and (optionally) pyannote models so the first
request to the backend doesn't stall waiting for a multi-GB download.

Run at Docker image build time:
    python scripts/warmup_models.py

Or manually before first use:
    cd backend && python scripts/warmup_models.py

Required env vars (from .env):
    WHISPER_MODEL_SIZE   — base | small  (default: small)
    WHISPER_COMPUTE_TYPE — int8 | float16 (default: int8)
    DIARIZATION_ENABLED  — true | false   (default: true)
    HF_TOKEN             — required when DIARIZATION_ENABLED=true
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Repo root on sys.path ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_REPO_ROOT = _BACKEND.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_BACKEND / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def warmup_whisper() -> None:
    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    logger.info("Downloading faster-whisper model: %s (compute=%s)...", model_size, compute_type)
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        # Run a tiny transcription to confirm the model is loaded and working
        import tempfile, wave, struct
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        with wave.open(tmp_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack("<" + "h" * 160, *([0] * 160)))  # 10ms silence
        list(model.transcribe(tmp_path)[0])  # consume generator
        Path(tmp_path).unlink(missing_ok=True)
        logger.info("faster-whisper %s ready.", model_size)
    except Exception as exc:
        logger.error("faster-whisper warmup failed: %s", exc)
        sys.exit(1)


def warmup_pyannote() -> None:
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        logger.error("HF_TOKEN is required for pyannote download. Set it in .env.")
        sys.exit(1)
    logger.info("Downloading pyannote speaker-diarization-3.1 (this may take a few minutes)...")
    try:
        # Trigger the compat patches in diarization.py before importing pyannote
        from backend.app.module1.diarization import _get_diarization_pipeline
        _get_diarization_pipeline()
        logger.info("pyannote speaker-diarization-3.1 ready.")
    except Exception as exc:
        logger.error("pyannote warmup failed: %s", exc)
        sys.exit(1)


def main() -> None:
    logger.info("=== LiveNote model warmup ===")

    warmup_whisper()

    diarization_enabled = os.getenv("DIARIZATION_ENABLED", "true").lower() == "true"
    if diarization_enabled:
        warmup_pyannote()
    else:
        logger.info("DIARIZATION_ENABLED=false — skipping pyannote download.")

    logger.info("=== All models ready. ===")


if __name__ == "__main__":
    main()
