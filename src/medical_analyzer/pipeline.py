"""Main pipeline: audio → preprocess → diarize → ASR → LLM correct → classify.

This orchestrates the full flow from raw audio file to structured JSON output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .asr import SherpaOfflineASR
from .config import Config, load_config
from .diarization import SpeakerSegment, diarize
from .llm import LLMClient
from .preprocessing import preprocess

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    transcript: list[dict[str, str]]
    qua_trinh_benh_ly: str
    tien_su_benh_nhan: str
    tien_su_gia_dinh: str

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "qua_trinh_benh_ly": self.qua_trinh_benh_ly,
            "tien_su_benh_nhan": self.tien_su_benh_nhan,
            "tien_su_gia_dinh": self.tien_su_gia_dinh,
        }


class MedicalPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.asr = SherpaOfflineASR(self.cfg.asr)
        self.llm = LLMClient(self.cfg.llm)

    def close(self) -> None:
        self.llm.close()

    def analyze_file(self, audio_path: str | Path) -> AnalysisResult:
        """Analyze a medical conversation audio file.

        Full pipeline:
        1. Load audio
        2. Preprocess (noise reduction + bandpass)
        3. Diarize (ICA for multi-channel, energy-based for mono)
        4. ASR each segment → text
        5. LLM spell correction
        6. LLM role assignment (doctor vs patient)
        7. LLM classification into 3 medical categories

        Returns:
            AnalysisResult with transcript and classified information.
        """
        audio_path = Path(audio_path)
        log.info("analyzing: %s", audio_path)

        # 1. Load audio
        audio, sample_rate = self._load_audio(audio_path)
        log.info("loaded audio: shape=%s, sr=%d", audio.shape, sample_rate)

        # 2. Preprocess
        audio = preprocess(audio, sample_rate)

        # 3. Diarize
        segments = diarize(audio, sample_rate, self.cfg.ica)
        if not segments:
            log.warning("no speech segments found")
            return AnalysisResult(
                transcript=[],
                qua_trinh_benh_ly="Không có thông tin",
                tien_su_benh_nhan="Không có thông tin",
                tien_su_gia_dinh="Không có thông tin",
            )

        # 4. ASR each segment
        segments_text = []
        for seg in segments:
            text = self.asr.transcribe(seg.audio, sample_rate=sample_rate)
            segments_text.append(text)
        log.info("ASR complete: %d segments transcribed", len(segments_text))

        # Filter out empty segments
        valid = [(seg, text) for seg, text in zip(segments, segments_text) if text.strip()]
        if not valid:
            log.warning("all segments transcribed as empty")
            return AnalysisResult(
                transcript=[],
                qua_trinh_benh_ly="Không có thông tin",
                tien_su_benh_nhan="Không có thông tin",
                tien_su_gia_dinh="Không có thông tin",
            )
        segments, segments_text = zip(*valid)
        segments_text = list(segments_text)

        # 5. LLM spell correction
        corrected = []
        for text in segments_text:
            try:
                corrected.append(self.llm.correct_spelling(text))
            except Exception as e:
                log.warning("spell correction failed for segment: %s", e)
                corrected.append(text)
        log.info("spell correction complete")

        # 6. LLM role assignment
        roles = self.llm.assign_roles(corrected)

        # 7. Build transcript
        transcript = [
            {"role": role, "text": text}
            for role, text in zip(roles, corrected)
        ]

        # 8. LLM classification
        classification = self.llm.classify_medical_info(transcript)
        log.info("classification complete")

        return AnalysisResult(
            transcript=transcript,
            qua_trinh_benh_ly=classification.get("qua_trinh_benh_ly", "Không có thông tin"),
            tien_su_benh_nhan=classification.get("tien_su_benh_nhan", "Không có thông tin"),
            tien_su_gia_dinh=classification.get("tien_su_gia_dinh", "Không có thông tin"),
        )

    def analyze_realtime(self, audio: np.ndarray, sample_rate: int) -> AnalysisResult:
        """Analyze audio from real-time recording (same pipeline, different input)."""
        return self._analyze_audio(audio, sample_rate)

    def _analyze_audio(self, audio: np.ndarray, sample_rate: int) -> AnalysisResult:
        """Internal: run pipeline on numpy audio array."""
        audio = preprocess(audio, sample_rate)
        segments = diarize(audio, sample_rate, self.cfg.ica)

        if not segments:
            return AnalysisResult(
                transcript=[],
                qua_trinh_benh_ly="Không có thông tin",
                tien_su_benh_nhan="Không có thông tin",
                tien_su_gia_dinh="Không có thông tin",
            )

        segments_text = [self.asr.transcribe(seg.audio, sample_rate=sample_rate) for seg in segments]
        valid = [(seg, text) for seg, text in zip(segments, segments_text) if text.strip()]
        if not valid:
            return AnalysisResult(
                transcript=[],
                qua_trinh_benh_ly="Không có thông tin",
                tien_su_benh_nhan="Không có thông tin",
                tien_su_gia_dinh="Không có thông tin",
            )

        segments, segments_text = zip(*valid)
        segments_text = list(segments_text)

        corrected = []
        for text in segments_text:
            try:
                corrected.append(self.llm.correct_spelling(text))
            except Exception:
                corrected.append(text)

        roles = self.llm.assign_roles(corrected)
        transcript = [{"role": role, "text": text} for role, text in zip(roles, corrected)]
        classification = self.llm.classify_medical_info(transcript)

        return AnalysisResult(
            transcript=transcript,
            qua_trinh_benh_ly=classification.get("qua_trinh_benh_ly", "Không có thông tin"),
            tien_su_benh_nhan=classification.get("tien_su_benh_nhan", "Không có thông tin"),
            tien_su_gia_dinh=classification.get("tien_su_gia_dinh", "Không có thông tin"),
        )

    def _load_audio(self, path: Path) -> tuple[np.ndarray, int]:
        """Load audio file, resample to ASR sample rate if needed."""
        audio, sr = sf.read(str(path), dtype="float32")

        # soundfile returns (samples, channels) — transpose to (channels, samples)
        if audio.ndim == 2:
            audio = audio.T  # (channels, samples)

        # Resample if needed
        target_sr = self.cfg.asr.sample_rate
        if sr != target_sr:
            from scipy.signal import resample

            if audio.ndim == 1:
                n_samples = int(len(audio) * target_sr / sr)
                audio = resample(audio, n_samples).astype(np.float32)
            else:
                n_samples = int(audio.shape[1] * target_sr / sr)
                resampled = np.zeros((audio.shape[0], n_samples), dtype=np.float32)
                for ch in range(audio.shape[0]):
                    resampled[ch] = resample(audio[ch], n_samples).astype(np.float32)
                audio = resampled
            sr = target_sr

        return audio, sr
