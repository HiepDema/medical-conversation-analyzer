"""Audio preprocessing: noise reduction + bandpass filter for speech."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

log = logging.getLogger(__name__)


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: int = 300,
    high_hz: int = 3400,
    order: int = 5,
) -> np.ndarray:
    """Apply bandpass filter to isolate speech frequencies."""
    nyquist = sample_rate / 2
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def reduce_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply noise reduction using noisereduce library."""
    import noisereduce as nr

    reduced = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.8)
    return reduced.astype(np.float32)


def preprocess(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Full preprocessing pipeline: noise reduction + bandpass filter.

    Args:
        audio: float32 numpy array, mono or multi-channel (shape: (samples,) or (channels, samples))
        sample_rate: sample rate in Hz

    Returns:
        Preprocessed float32 audio, same shape as input.
    """
    log.info("preprocessing: noise reduction + bandpass filter (%d Hz)", sample_rate)

    if audio.ndim == 1:
        audio = reduce_noise(audio, sample_rate)
        audio = bandpass_filter(audio, sample_rate)
    else:
        for ch in range(audio.shape[0]):
            audio[ch] = reduce_noise(audio[ch], sample_rate)
            audio[ch] = bandpass_filter(audio[ch], sample_rate)

    return audio
