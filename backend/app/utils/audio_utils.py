from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import which

# Ensure ffmpeg is found on macOS Homebrew installs where /opt/homebrew/bin
# may not be in PATH when launched from an IDE or non-login shell.
_FFMPEG_CANDIDATES = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "ffmpeg"]

def _ffmpeg_binary() -> str:
    for candidate in _FFMPEG_CANDIDATES:
        if which(candidate) or Path(candidate).is_file():
            return candidate
    return "ffmpeg"

AudioSegment.converter = _ffmpeg_binary()  # type: ignore[assignment]


class AudioConversionError(RuntimeError):
    """Raised when an incoming browser audio blob cannot be converted."""


def _temp_audio_dir() -> Path:
    temp_dir = Path(os.getenv("TEMP_AUDIO_DIR", "/tmp/livenote_audio"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def convert_webm_bytes_to_wav_path(
    audio_bytes: bytes,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path:
    sample_rate = sample_rate or int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    channels = channels or int(os.getenv("AUDIO_CHANNELS", "1"))
    input_format = os.getenv("AUDIO_FORMAT_INPUT", "webm")
    temp_dir = _temp_audio_dir()

    with tempfile.NamedTemporaryFile(
        suffix=f".{input_format}",
        dir=temp_dir,
        delete=False,
    ) as source_file:
        source_file.write(audio_bytes)
        source_path = Path(source_file.name)

    target_path = source_path.with_suffix(".wav")

    try:
        audio = AudioSegment.from_file(source_path, format=input_format)
        audio = audio.set_frame_rate(sample_rate).set_channels(channels).set_sample_width(2)
        audio.export(target_path, format="wav")
        return target_path
    except Exception:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(source_path),
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    str(channels),
                    str(target_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return target_path
        except Exception as exc:  # pragma: no cover - fallback is environment-specific
            raise AudioConversionError("Failed to convert browser audio to wav.") from exc
        finally:
            cleanup_temp_file(source_path)
    finally:
        if source_path.exists():
            cleanup_temp_file(source_path)


def load_wav_bytes(wav_path: Path) -> bytes:
    with wav_path.open("rb") as wav_file:
        return wav_file.read()


def cleanup_temp_file(path: str | Path | None) -> None:
    if not path:
        return

    file_path = Path(path)
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError:
        pass


def in_memory_audio_file(audio_bytes: bytes) -> io.BytesIO:
    return io.BytesIO(audio_bytes)
