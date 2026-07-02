"""Speaker diarization using Independent Component Analysis (ICA).

For stereo/multi-channel audio: FastICA separates mixed sources into
independent components (doctor vs patient voices).

For mono audio: falls back to energy-based segmentation using VAD-like
silence detection to split turns (assumption: speakers alternate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA

from .config import ICAConfig

log = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    start_sample: int
    end_sample: int
    speaker: int  # 0 or 1
    audio: np.ndarray  # float32 mono


def ica_separate(audio: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Separate multi-channel audio into independent components using FastICA.

    Args:
        audio: shape (channels, samples), float32
        n_components: number of sources to separate (default 2: doctor + patient)

    Returns:
        Separated sources, shape (n_components, samples)
    """
    if audio.ndim != 2 or audio.shape[0] < 2:
        raise ValueError("ICA requires at least 2 channels. Got shape: " + str(audio.shape))

    log.info("running FastICA (channels=%d, components=%d)", audio.shape[0], n_components)
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    sources = ica.fit_transform(audio.T).T  # (n_components, samples)

    # Normalize each source
    for i in range(sources.shape[0]):
        peak = np.max(np.abs(sources[i]))
        if peak > 0:
            sources[i] /= peak

    return sources.astype(np.float32)


def segment_by_energy(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    silence_threshold: float | None = None,
    min_segment_ms: int = 500,
) -> list[SpeakerSegment]:
    """Segment mono audio into speaker turns based on energy/silence gaps.

    Assumes speakers alternate (doctor asks, patient answers, etc.).
    Splits at silence gaps and assigns alternating speaker labels.

    If silence_threshold is None, uses adaptive threshold (15% of mean energy).
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    n_frames = len(audio) // frame_size

    if n_frames == 0:
        return []

    # Compute energy per frame
    energies = np.array([
        np.sqrt(np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2))
        for i in range(n_frames)
    ])

    # Adaptive threshold: 15% of mean energy (works across different volumes)
    if silence_threshold is None:
        mean_energy = np.mean(energies)
        silence_threshold = max(mean_energy * 0.15, 0.001)
        log.info("adaptive silence threshold: %.5f (mean energy: %.5f)", silence_threshold, mean_energy)

    # Find speech frames
    is_speech = energies > silence_threshold

    # Find segment boundaries (transitions from silence to speech and back)
    segments: list[SpeakerSegment] = []
    in_speech = False
    seg_start = 0
    speaker = 0
    min_frames = max(1, int(min_segment_ms / frame_ms))

    for i in range(n_frames):
        if is_speech[i] and not in_speech:
            seg_start = i
            in_speech = True
        elif not is_speech[i] and in_speech:
            seg_len = i - seg_start
            if seg_len >= min_frames:
                segments.append(SpeakerSegment(
                    start_sample=seg_start * frame_size,
                    end_sample=i * frame_size,
                    speaker=speaker,
                    audio=audio[seg_start * frame_size:i * frame_size],
                ))
                speaker = 1 - speaker  # alternate
            in_speech = False

    # Handle last segment
    if in_speech and (n_frames - seg_start) >= min_frames:
        segments.append(SpeakerSegment(
            start_sample=seg_start * frame_size,
            end_sample=len(audio),
            speaker=speaker,
            audio=audio[seg_start * frame_size:],
        ))

    log.info("energy-based segmentation: %d segments found", len(segments))
    return segments


def segment_ica_sources(
    sources: np.ndarray,
    sample_rate: int,
    frame_ms: int = 30,
    threshold: float = 0.03,
    min_segment_ms: int = 300,
) -> list[SpeakerSegment]:
    """Given ICA-separated sources, create time-aligned speaker segments.

    Determines which source is dominant at each time frame and creates
    segments accordingly.
    """
    frame_size = int(sample_rate * frame_ms / 1000)
    n_frames = min(sources.shape[1], sources.shape[1]) // frame_size
    n_sources = sources.shape[0]

    # Compute energy per frame per source
    energies = np.zeros((n_sources, n_frames))
    for s in range(n_sources):
        for i in range(n_frames):
            chunk = sources[s, i * frame_size:(i + 1) * frame_size]
            energies[s, i] = np.sqrt(np.mean(chunk ** 2))

    # Determine dominant speaker per frame
    dominant = np.argmax(energies, axis=0)
    max_energy = np.max(energies, axis=0)

    # Silence frames (no dominant speaker)
    dominant[max_energy < threshold] = -1

    # Build segments from consecutive same-speaker frames
    segments: list[SpeakerSegment] = []
    min_frames = max(1, int(min_segment_ms / frame_ms))

    seg_start = 0
    current_speaker = dominant[0]

    for i in range(1, n_frames):
        if dominant[i] != current_speaker:
            if current_speaker >= 0 and (i - seg_start) >= min_frames:
                spk = int(current_speaker)
                segments.append(SpeakerSegment(
                    start_sample=seg_start * frame_size,
                    end_sample=i * frame_size,
                    speaker=spk,
                    audio=sources[spk, seg_start * frame_size:i * frame_size],
                ))
            seg_start = i
            current_speaker = dominant[i]

    # Last segment
    if current_speaker >= 0 and (n_frames - seg_start) >= min_frames:
        spk = int(current_speaker)
        segments.append(SpeakerSegment(
            start_sample=seg_start * frame_size,
            end_sample=sources.shape[1],
            speaker=spk,
            audio=sources[spk, seg_start * frame_size:],
        ))

    log.info("ICA segmentation: %d segments found", len(segments))
    return segments


def diarize(
    audio: np.ndarray,
    sample_rate: int,
    cfg: ICAConfig,
) -> list[SpeakerSegment]:
    """Main diarization entry point.

    - Multi-channel (>=2): uses ICA to separate speakers, then segments by dominance.
    - Mono: falls back to energy-based alternating speaker segmentation.
    """
    if audio.ndim == 2 and audio.shape[0] >= 2:
        log.info("multi-channel audio detected (%d channels), using ICA", audio.shape[0])
        sources = ica_separate(audio, n_components=cfg.n_components)
        return segment_ica_sources(sources, sample_rate)
    else:
        if audio.ndim == 2:
            audio = audio[0]  # take first channel
        if cfg.fallback_to_energy:
            log.info("mono audio, falling back to energy-based segmentation")
            return segment_by_energy(audio, sample_rate)
        else:
            raise ValueError(
                "ICA requires multi-channel audio but got mono. "
                "Set ICA_FALLBACK_TO_ENERGY=1 to use energy-based segmentation."
            )
