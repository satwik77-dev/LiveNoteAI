from __future__ import annotations

import os
import inspect
import sys
import types
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
from io import IOBase
from pathlib import Path


@dataclass(slots=True)
class SpeakerSegment:
    start: float
    end: float
    speaker: str


def diarize_audio(wav_path: str | Path, *, chunk_start: float = 0.0) -> list[SpeakerSegment]:
    if not diarization_enabled():
        return []

    pipeline = _get_diarization_pipeline()
    diarization = pipeline(str(wav_path))

    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            SpeakerSegment(
                start=chunk_start + float(turn.start),
                end=chunk_start + float(turn.end),
                speaker=str(speaker),
            )
        )
    return segments


def diarization_enabled() -> bool:
    return os.getenv("DIARIZATION_ENABLED", "true").lower() == "true"


@lru_cache(maxsize=1)
def _get_diarization_pipeline():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required when diarization is enabled.")

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]
    if not hasattr(np, "NAN"):
        np.NAN = np.nan  # type: ignore[attr-defined]
    torch_load_signature = inspect.signature(torch.load)
    if "weights_only" in torch_load_signature.parameters:
        original_torch_load = torch.load

        def _compat_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        torch.load = _compat_torch_load
    if "torchaudio.backend" not in sys.modules:
        backend_module = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend_module
    if "torchaudio.backend.common" not in sys.modules:
        common_module = types.ModuleType("torchaudio.backend.common")
        common_module.AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
        sys.modules["torchaudio.backend.common"] = common_module
    audio_metadata_cls = sys.modules["torchaudio.backend.common"].AudioMetaData
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not getattr(torchaudio.load, "__name__", "") == "_compat_torchaudio_load":
        def _compat_torchaudio_load(
            uri,
            frame_offset: int = 0,
            num_frames: int = -1,
            normalize: bool = True,
            channels_first: bool = True,
            format=None,
            buffer_size: int = 4096,
            backend=None,
        ):
            if isinstance(uri, IOBase):
                waveform, sample_rate = sf.read(uri)
            else:
                waveform, sample_rate = sf.read(str(uri))

            if waveform.ndim == 1:
                waveform = waveform[:, None]

            start_idx = max(frame_offset, 0)
            end_idx = None if num_frames in (-1, 0, None) else start_idx + num_frames
            waveform = waveform[start_idx:end_idx]
            tensor = torch.from_numpy(waveform).to(torch.float32)
            if channels_first:
                tensor = tensor.transpose(0, 1)
            return tensor, int(sample_rate)

        def _compat_torchaudio_info(uri, format=None, buffer_size: int = 4096, backend=None):
            info = sf.info(uri)
            return audio_metadata_cls(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=0,
                encoding=str(getattr(info, "subtype", "UNKNOWN")),
            )

        torchaudio.load = _compat_torchaudio_load
        torchaudio.info = _compat_torchaudio_info

    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
