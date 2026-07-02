"""Offline Vietnamese ASR via sherpa-onnx Zipformer.

Reused from conversational-agent project. Accepts float32 numpy array
or int16 bytes and returns Vietnamese text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import ASRConfig

log = logging.getLogger(__name__)


class SherpaOfflineASR:
    def __init__(self, cfg: ASRConfig) -> None:
        import sherpa_onnx

        self.cfg = cfg
        for f in (cfg.encoder, cfg.decoder, cfg.joiner, cfg.tokens):
            if not Path(f).exists():
                raise FileNotFoundError(
                    f"ASR model file missing: {f}. Run scripts/download_models.sh"
                )

        log.info("loading offline Zipformer recognizer (CPU)")
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(cfg.encoder),
            decoder=str(cfg.decoder),
            joiner=str(cfg.joiner),
            tokens=str(cfg.tokens),
            num_threads=cfg.num_threads,
            sample_rate=cfg.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
        )

    def transcribe(self, audio: np.ndarray | bytes, sample_rate: int | None = None) -> str:
        """Transcribe audio to Vietnamese text.

        Args:
            audio: float32 numpy array (normalized -1..1) or int16 bytes.
            sample_rate: sample rate of audio (defaults to cfg.sample_rate).
        """
        if isinstance(audio, bytes):
            if not audio:
                return ""
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        elif audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        if len(audio) == 0:
            return ""

        stream = self._rec.create_stream()
        stream.accept_waveform(sample_rate or self.cfg.sample_rate, audio)
        self._rec.decode_stream(stream)
        return (stream.result.text or "").strip()
