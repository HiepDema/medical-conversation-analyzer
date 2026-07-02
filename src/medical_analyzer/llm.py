"""LLM client for spell correction and medical classification.

Reused from conversational-agent project — async streaming client
for OpenAI-compatible endpoints.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient:
    def __init__(self, cfg: LLMConfig, *, request_timeout: float = 120.0) -> None:
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=httpx.Timeout(request_timeout, connect=10.0),
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Synchronous chat completion — returns full response text."""
        payload = {
            "model": self.cfg.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self.cfg.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
            "stream": False,
        }
        resp = self._client.post("/chat/completions", json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def correct_spelling(self, text: str) -> str:
        """Use LLM to correct Vietnamese spelling errors from ASR output."""
        if not text.strip():
            return text

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Bạn là công cụ sửa chính tả tiếng Việt cho văn bản y khoa. "
                    "Chỉ sửa lỗi chính tả, KHÔNG thay đổi nội dung hay ý nghĩa. "
                    "Trả về văn bản đã sửa, không giải thích gì thêm."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        return self.complete(messages, temperature=0.1, max_tokens=len(text) * 3)

    def classify_medical_info(self, transcript: list[dict[str, str]]) -> dict[str, str]:
        """Classify transcript into 3 medical categories.

        Args:
            transcript: list of {"role": "bác_sĩ"|"bệnh_nhân", "text": "..."}

        Returns:
            {
                "qua_trinh_benh_ly": "...",
                "tien_su_benh_nhan": "...",
                "tien_su_gia_dinh": "..."
            }
        """
        conversation_text = "\n".join(
            f"[{turn['role']}]: {turn['text']}" for turn in transcript
        )

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Bạn là trợ lý y khoa. Phân tích đoạn hội thoại giữa bác sĩ và bệnh nhân, "
                    "trích xuất thông tin vào 3 mục sau. Chỉ trả về JSON, không giải thích.\n\n"
                    "Output JSON format:\n"
                    "{\n"
                    '  "qua_trinh_benh_ly": "mô tả quá trình bệnh lý hiện tại",\n'
                    '  "tien_su_benh_nhan": "tiền sử bệnh của bệnh nhân",\n'
                    '  "tien_su_gia_dinh": "tiền sử bệnh trong gia đình"\n'
                    "}\n\n"
                    "Nếu không có thông tin cho mục nào, ghi \"Không có thông tin\"."
                ),
            ),
            ChatMessage(role="user", content=conversation_text),
        ]

        raw = self.complete(messages, temperature=0.1)

        # Parse JSON from response (handle markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON, returning raw text")
            result = {
                "qua_trinh_benh_ly": raw,
                "tien_su_benh_nhan": "Không thể phân tích",
                "tien_su_gia_dinh": "Không thể phân tích",
            }

        return result

    def assign_roles(self, segments_text: list[str]) -> list[str]:
        """Use LLM to determine which speaker is doctor vs patient.

        Given a list of transcribed segments (alternating speakers),
        returns role labels for each segment.
        """
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(segments_text))

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Phân tích đoạn hội thoại y khoa sau. Xác định vai trò của từng câu: "
                    "\"bác_sĩ\" hoặc \"bệnh_nhân\". Bác sĩ thường hỏi triệu chứng, tiền sử; "
                    "bệnh nhân thường mô tả triệu chứng, trả lời câu hỏi.\n\n"
                    "Trả về JSON array chứa role cho mỗi câu, ví dụ:\n"
                    "[\"bác_sĩ\", \"bệnh_nhân\", \"bác_sĩ\", ...]\n"
                    "Chỉ trả JSON, không giải thích."
                ),
            ),
            ChatMessage(role="user", content=numbered),
        ]

        raw = self.complete(messages, temperature=0.1)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            roles = json.loads(raw)
            if isinstance(roles, list) and len(roles) == len(segments_text):
                return roles
        except json.JSONDecodeError:
            pass

        # Fallback: alternate bác_sĩ/bệnh_nhân
        log.warning("role assignment failed, using alternating fallback")
        return ["bác_sĩ" if i % 2 == 0 else "bệnh_nhân" for i in range(len(segments_text))]
