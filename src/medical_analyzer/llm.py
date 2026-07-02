"""LLM client for spell correction and medical classification.

Reused from conversational-agent project — sync client for
OpenAI-compatible endpoints. Handles long conversations via chunking.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)

CHUNK_SIZE = 15  # max segments per LLM call (keeps within ~2k tokens)


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

    def correct_spelling_batch(self, texts: list[str]) -> list[str]:
        """Batch spell correction — groups multiple segments into one LLM call.

        Sends up to CHUNK_SIZE segments per call to reduce total LLM invocations.
        """
        results: list[str] = []
        for i in range(0, len(texts), CHUNK_SIZE):
            chunk = texts[i:i + CHUNK_SIZE]
            numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(chunk))
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "Sửa chính tả tiếng Việt cho các câu y khoa sau. "
                        "Chỉ sửa lỗi chính tả, KHÔNG thay đổi nội dung. "
                        "Trả về đúng số dòng, mỗi dòng bắt đầu bằng số thứ tự. "
                        "Ví dụ:\n1. câu đã sửa\n2. câu đã sửa"
                    ),
                ),
                ChatMessage(role="user", content=numbered),
            ]
            try:
                raw = self.complete(messages, temperature=0.1)
                lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
                # Parse numbered lines
                parsed = []
                for line in lines:
                    # Remove "1. ", "2. " prefix
                    parts = line.split(". ", 1)
                    if len(parts) == 2 and parts[0].isdigit():
                        parsed.append(parts[1])
                    else:
                        parsed.append(line)
                # Ensure we got the right count
                if len(parsed) == len(chunk):
                    results.extend(parsed)
                else:
                    log.warning("batch correction returned %d lines for %d input, falling back", len(parsed), len(chunk))
                    results.extend(chunk)
            except Exception as e:
                log.warning("batch correction failed: %s", e)
                results.extend(chunk)
        return results

    def classify_medical_info(self, transcript: list[dict[str, str]]) -> dict[str, str]:
        """Classify transcript into 3 medical categories.

        For long conversations (>CHUNK_SIZE turns), processes in chunks and
        merges partial results into a final classification.

        Args:
            transcript: list of {"role": "bác_sĩ"|"bệnh_nhân", "text": "..."}

        Returns:
            {
                "qua_trinh_benh_ly": "...",
                "tien_su_benh_nhan": "...",
                "tien_su_gia_dinh": "..."
            }
        """
        if len(transcript) <= CHUNK_SIZE:
            return self._classify_chunk(transcript)

        # Process in chunks, collect partial results
        partials: list[dict[str, str]] = []
        for i in range(0, len(transcript), CHUNK_SIZE):
            chunk = transcript[i:i + CHUNK_SIZE]
            partial = self._classify_chunk(chunk)
            partials.append(partial)

        # Merge partials into final result
        return self._merge_classifications(partials)

    def _classify_chunk(self, transcript: list[dict[str, str]]) -> dict[str, str]:
        """Classify a single chunk of transcript."""
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
        return self._parse_json_response(raw)

    def _merge_classifications(self, partials: list[dict[str, str]]) -> dict[str, str]:
        """Merge partial classification results from multiple chunks."""
        # Collect all non-empty findings per field
        fields = ["qua_trinh_benh_ly", "tien_su_benh_nhan", "tien_su_gia_dinh"]
        collected = {f: [] for f in fields}

        for p in partials:
            for f in fields:
                val = p.get(f, "").strip()
                if val and val != "Không có thông tin":
                    collected[f].append(val)

        if not any(collected.values()):
            return {f: "Không có thông tin" for f in fields}

        # Ask LLM to synthesize the partial findings
        summary_input = "\n\n".join(
            f"## {f}\n" + "\n".join(f"- {v}" for v in vals)
            for f, vals in collected.items() if vals
        )

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "Tổng hợp các thông tin y khoa sau thành 3 mục hoàn chỉnh. "
                    "Loại bỏ trùng lặp, giữ nguyên nội dung quan trọng. Trả về JSON.\n\n"
                    "Output JSON format:\n"
                    "{\n"
                    '  "qua_trinh_benh_ly": "...",\n'
                    '  "tien_su_benh_nhan": "...",\n'
                    '  "tien_su_gia_dinh": "..."\n'
                    "}"
                ),
            ),
            ChatMessage(role="user", content=summary_input),
        ]

        raw = self.complete(messages, temperature=0.1)
        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> dict[str, str]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM returned invalid JSON, returning raw text")
            return {
                "qua_trinh_benh_ly": raw,
                "tien_su_benh_nhan": "Không thể phân tích",
                "tien_su_gia_dinh": "Không thể phân tích",
            }

    def assign_roles(self, segments_text: list[str]) -> list[str]:
        """Use LLM to determine which speaker is doctor vs patient.

        For long conversations, processes in chunks with context overlap
        to maintain role consistency across boundaries.
        """
        if len(segments_text) <= CHUNK_SIZE:
            return self._assign_roles_chunk(segments_text)

        # Process in chunks with 2-segment overlap for context
        all_roles: list[str] = []
        overlap = 2

        for i in range(0, len(segments_text), CHUNK_SIZE - overlap):
            chunk = segments_text[i:i + CHUNK_SIZE]
            chunk_roles = self._assign_roles_chunk(chunk)

            if i == 0:
                all_roles.extend(chunk_roles)
            else:
                # Skip the overlap portion (already assigned)
                all_roles.extend(chunk_roles[overlap:])

        # Trim to exact length
        return all_roles[:len(segments_text)]

    def _assign_roles_chunk(self, segments_text: list[str]) -> list[str]:
        """Assign roles for a single chunk."""
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

        log.warning("role assignment failed for chunk, using alternating fallback")
        return ["bác_sĩ" if i % 2 == 0 else "bệnh_nhân" for i in range(len(segments_text))]
