"""Basic tests for the medical analyzer pipeline."""

import numpy as np
import pytest

from medical_analyzer.diarization import segment_by_energy, ica_separate, SpeakerSegment
from medical_analyzer.preprocessing import bandpass_filter, reduce_noise


def _make_sine(freq: float, duration: float, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


class TestPreprocessing:
    def test_bandpass_filter_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        filtered = bandpass_filter(audio, 16000)
        assert filtered.shape == audio.shape
        assert filtered.dtype == np.float32

    def test_bandpass_removes_low_freq(self):
        sr = 16000
        low_tone = _make_sine(50, 1.0, sr)  # below 300Hz cutoff
        speech_tone = _make_sine(1000, 1.0, sr)  # within passband
        filtered_low = bandpass_filter(low_tone, sr)
        filtered_speech = bandpass_filter(speech_tone, sr)
        assert np.mean(np.abs(filtered_low)) < np.mean(np.abs(filtered_speech))


class TestDiarization:
    def test_energy_segmentation_finds_segments(self):
        sr = 16000
        # Silence + speech + silence + speech
        silence = np.zeros(sr, dtype=np.float32)
        speech1 = _make_sine(500, 1.0, sr) * 0.5
        speech2 = _make_sine(800, 1.0, sr) * 0.5
        audio = np.concatenate([silence, speech1, silence, speech2])

        segments = segment_by_energy(audio, sr)
        assert len(segments) >= 2
        assert segments[0].speaker == 0
        assert segments[1].speaker == 1

    def test_energy_segmentation_empty_audio(self):
        audio = np.zeros(16000, dtype=np.float32)
        segments = segment_by_energy(audio, 16000)
        assert len(segments) == 0

    def test_ica_requires_multichannel(self):
        mono = np.random.randn(16000).astype(np.float32)
        with pytest.raises(ValueError, match="at least 2 channels"):
            ica_separate(mono.reshape(1, -1))

    def test_ica_separates_two_sources(self):
        sr = 16000
        n = sr * 2
        # Two distinct sources
        s1 = _make_sine(400, 2.0, sr)
        s2 = _make_sine(1200, 2.0, sr)
        # Mix them (simulating 2 mics picking up both)
        mixed = np.array([
            0.7 * s1 + 0.3 * s2,
            0.3 * s1 + 0.7 * s2,
        ], dtype=np.float32)

        sources = ica_separate(mixed, n_components=2)
        assert sources.shape == (2, n)
