# LiveNote Backend — HuggingFace Spaces Docker image
# HF Spaces maps port 7860 externally; the app listens on 7860.

FROM python:3.10-slim

# System deps: ffmpeg for audio conversion
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached when only code changes)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# HF Spaces: non-root user required
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Pre-create temp directory
RUN mkdir -p /tmp/livenote_audio

EXPOSE 7860

# HF Spaces deployment: Groq for LLM, no diarization (no GPU/HF token needed)
ENV HOST=0.0.0.0 \
    PORT=7860 \
    LLM_MODE=groq \
    DIARIZATION_ENABLED=false \
    LIVE_INTELLIGENCE_ENABLED=true \
    WHISPER_MODEL_SIZE=base \
    WHISPER_COMPUTE_TYPE=int8 \
    TEMP_AUDIO_DIR=/tmp/livenote_audio \
    ALLOWED_ORIGINS=*

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
