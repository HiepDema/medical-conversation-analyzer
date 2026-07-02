from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw not in (None, "") else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "http://127.0.0.1:8899/v1"))
    model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))
    api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", "not-needed"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.3))
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 2048))


@dataclass(frozen=True)
class ASRConfig:
    encoder: Path = field(default_factory=lambda: Path(_env("ASR_ENCODER", "./models/zipformer-offline/encoder-epoch-20-avg-10.int8.onnx")))
    decoder: Path = field(default_factory=lambda: Path(_env("ASR_DECODER", "./models/zipformer-offline/decoder-epoch-20-avg-10.int8.onnx")))
    joiner: Path = field(default_factory=lambda: Path(_env("ASR_JOINER", "./models/zipformer-offline/joiner-epoch-20-avg-10.int8.onnx")))
    tokens: Path = field(default_factory=lambda: Path(_env("ASR_TOKENS", "./models/zipformer-offline/config.json")))
    sample_rate: int = field(default_factory=lambda: _env_int("ASR_SAMPLE_RATE", 16000))
    num_threads: int = field(default_factory=lambda: _env_int("ASR_NUM_THREADS", 2))


@dataclass(frozen=True)
class ICAConfig:
    n_components: int = field(default_factory=lambda: _env_int("ICA_N_COMPONENTS", 2))
    fallback_to_energy: bool = field(default_factory=lambda: _env("ICA_FALLBACK_TO_ENERGY", "1") == "1")


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)


def load_config() -> Config:
    return Config()
