"""DeepSeek-backed rubric scorer for MCM/ICM papers."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None


DETAIL_MAX = {
    "content_score": 30,
    "length_style_score": 20,
    "visual_score": 25,
    "conclusion_score": 15,
    "writing_score": 10,
}

AWARD_KEYS = ["O", "F", "M", "H", "S"]


class DeepSeekRubricScorer:
    """Call DeepSeek JSON mode and validate the returned rubric payload."""

    provider = "deepseek"

    def __init__(self, prompt_path: str | None = None):
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.enabled = os.getenv("LLM_RUBRIC_ENABLED", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", "1500"))
        self.timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "45"))
        self.prompt_path = Path(prompt_path) if prompt_path else self._default_prompt_path()

    @staticmethod
    def _default_prompt_path() -> Path:
        # src/llm_rubric_scorer.py -> project root
        return Path(__file__).resolve().parents[1] / "改进后prompt.txt"

    def unavailable(self, reason: str) -> Dict[str, Any]:
        return {
            "status": "unavailable",
            "provider": self.provider,
            "model": self.model,
            "score": 0,
            "details": {key: 0 for key in DETAIL_MAX},
            "award_prediction": "S/U",
            "probabilities": {"O": 0, "F": 0, "M": 0, "H": 0, "S": 100},
            "strengths": [],
            "weaknesses": [],
            "comments": reason,
        }

    def failed(self, reason: str) -> Dict[str, Any]:
        payload = self.unavailable(reason)
        payload["status"] = "failed"
        return payload

    def score(
        self,
        *,
        abstract: str,
        full_text: str = "",
        structure: Dict[str, Any],
        image_result: Dict[str, Any],
        image_count: int,
        page_count: int,
        ref_count: int,
        problem: str = "auto",
        contest: str = "auto",
        year: int | str = "auto",
    ) -> Dict[str, Any]:
        if not self.enabled:
            return self.unavailable("AI 评分已通过 LLM_RUBRIC_ENABLED=0 关闭")
        if not self.api_key:
            return self.unavailable("未设置评审 API Key，AI 评分未启用")
        if OpenAI is None:
            return self.unavailable("未安装 openai Python SDK，无法调用评审 API")
        if not self.prompt_path.exists():
            return self.unavailable(f"未找到 prompt 文件: {self.prompt_path}")

        prompt = self._build_prompt(
            abstract=abstract,
            full_text=full_text,
            structure=structure,
            image_result=image_result,
            image_count=image_count,
            page_count=page_count,
            ref_count=ref_count,
            problem=problem,
            contest=contest,
            year=year,
        )

        try:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict MCM/ICM Outstanding Winner judge. Output only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = response.choices[0].message.content
        except Exception as exc:
            return self.failed(f"评审 API 调用失败: {exc}")

        if not content or not content.strip():
            return self.failed("评审模型返回空内容")

        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            return self.failed(f"评审模型返回内容不是合法 JSON: {exc}")

        try:
            return self._validate_payload(raw)
        except ValueError as exc:
            return self.failed(f"评审 JSON 字段校验失败: {exc}")

    def _build_prompt(
        self,
        *,
        abstract: str,
        full_text: str,
        structure: Dict[str, Any],
        image_result: Dict[str, Any],
        image_count: int,
        page_count: int,
        ref_count: int,
        problem: str,
        contest: str,
        year: int | str,
    ) -> str:
        text = abstract or ""
        template = self.prompt_path.read_text(encoding="utf-8")
        values = {
            "text": text,
            "word_count": str(len(text.split())),
            "full_word_count": str(len((full_text or "").split())),
            "full_text_evidence": self._full_text_evidence(full_text),
            "caption_evidence": self._caption_evidence(full_text),
            "problem": str(problem or "auto"),
            "contest": str(contest or "auto"),
            "year": str(year or "auto"),
            "page_count": str(page_count or 0),
            "image_count": str(image_count or 0),
            "figure_count": str(structure.get("figure_caption_count", 0) or image_count or 0),
            "figure_caption_count": str(structure.get("figure_caption_count", 0)),
            "table_count": str(structure.get("table_count", 0)),
            "citation_count": str(structure.get("citation_count", 0)),
            "ref_count": str(ref_count or 0),
            "formula_count": str(structure.get("formula_count", 0)),
            "structure_completeness": f"{float(structure.get('structure_completeness', 0) or 0):.0%}",
            "has_sensitivity_analysis": self._yes_no(structure.get("has_sensitivity_analysis")),
            "has_model_validation": self._yes_no(structure.get("has_model_validation")),
            "has_strengths_weaknesses": self._yes_no(structure.get("has_strengths_weaknesses")),
            "has_assumption_justification": self._yes_no(structure.get("has_assumption_justification")),
            "has_model_comparison": self._yes_no(structure.get("has_model_comparison")),
            "has_error_analysis": self._yes_no(structure.get("has_error_analysis")),
            "image_quality_summary": self._image_quality_summary(image_result),
        }
        for key, value in values.items():
            template = template.replace("{" + key + "}", value)
        return template

    @staticmethod
    def _full_text_evidence(full_text: str) -> str:
        text = full_text or ""
        if not text.strip():
            return "无可用全文文本"

        chunks = []
        chunks.append(text[:12000])

        for pattern in [
            r"(?is)\bSensitivity\s+Analysis\b.{0,5000}",
            r"(?is)\b(Model\s+Validation|Validation|Testing)\b.{0,4000}",
            r"(?is)\b(Strengths?\s+and\s+Weaknesses?|Weaknesses?|Limitations?)\b.{0,4000}",
            r"(?is)\bConclusion\b.{0,5000}",
            r"(?is)\b(Methodology|Method|Approach|Model)\b.{0,4000}",
        ]:
            match = re.search(pattern, text)
            if match:
                chunks.append(match.group(0))

        evidence = "\n\n---\n\n".join(chunks)
        return evidence[:35000]

    @staticmethod
    def _caption_evidence(full_text: str) -> str:
        text = full_text or ""
        matches = re.findall(
            r"(?mi)^\s*((?:fig\.?|figure|tab\.?|table)\s+\d+[\s.:：-].{0,180})",
            text,
        )
        cleaned = [re.sub(r"\s+", " ", item).strip() for item in matches[:50]]
        return "\n".join(cleaned) if cleaned else "未提取到明确 Figure/Table 标题"

    @staticmethod
    def _yes_no(value: Any) -> str:
        return "是" if bool(value) else "否"

    @staticmethod
    def _image_quality_summary(image_result: Dict[str, Any]) -> str:
        stats = image_result.get("statistical_features")
        if stats is None or len(stats) < 18:
            return "无可用图像统计特征"

        def pct(index: int) -> str:
            return f"{float(stats[index]):.0%}"

        return (
            f"图表类图片占比 {pct(6)}；彩色图占比 {pct(9)}；"
            f"版式/专业度评分 {pct(13)}；含色条线索占比 {pct(14)}；"
            f"视觉多样性 {pct(16)}；高分辨率图片占比 {pct(17)}"
        )

    def _validate_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("根对象必须是 JSON object")
        details = raw.get("details")
        if not isinstance(details, dict):
            raise ValueError("details 必须是 object")

        normalized_details = {}
        for key, max_score in DETAIL_MAX.items():
            normalized_details[key] = self._clamped_int(details.get(key), 0, max_score, key)

        score = sum(normalized_details.values())
        probabilities = self._validated_probabilities(raw.get("probabilities"), score)

        award_prediction = self._award_from_probabilities(probabilities)

        return {
            "status": "ok",
            "provider": self.provider,
            "model": self.model,
            "score": score,
            "details": normalized_details,
            "award_prediction": award_prediction,
            "probabilities": probabilities,
            "strengths": self._string_list(raw.get("strengths")),
            "weaknesses": self._string_list(raw.get("weaknesses")),
            "comments": str(raw.get("comments", "")).strip()[:600],
        }

    @staticmethod
    def _clamped_int(value: Any, min_score: int, max_score: int, field: str) -> int:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            raise ValueError(f"{field} 必须是数字")
        return max(min_score, min(max_score, numeric))

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value.strip()[:120]] if value.strip() else []
        if not isinstance(value, Iterable):
            return []
        items = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text[:120])
            if len(items) >= 5:
                break
        return items

    @classmethod
    def _validated_probabilities(cls, value: Any, score: int) -> Dict[str, int]:
        if not isinstance(value, dict):
            return cls._probabilities_from_score(score)

        raw_values: Dict[str, float] = {}
        for key in AWARD_KEYS:
            source_key = "S/U" if key == "S" and "S/U" in value else key
            try:
                raw_values[key] = max(0.0, float(value.get(source_key, 0)))
            except (TypeError, ValueError):
                raw_values[key] = 0.0

        total = sum(raw_values.values())
        if total <= 0:
            return cls._probabilities_from_score(score)

        if total <= 1.5:
            raw_values = {key: val * 100 for key, val in raw_values.items()}
            total = sum(raw_values.values())

        normalized = {key: raw_values[key] / total * 100 for key in AWARD_KEYS}
        rounded = {key: int(round(normalized[key])) for key in AWARD_KEYS}
        diff = 100 - sum(rounded.values())
        if diff:
            adjust_key = max(AWARD_KEYS, key=lambda key: normalized[key])
            rounded[adjust_key] = max(0, rounded[adjust_key] + diff)

        return rounded

    @staticmethod
    def _award_from_probabilities(probabilities: Dict[str, int]) -> str:
        award = max(AWARD_KEYS, key=lambda key: probabilities.get(key, 0))
        return "S/U" if award == "S" else award

    @staticmethod
    def _probabilities_from_score(score: int) -> Dict[str, int]:
        centers = {"O": 95, "F": 88, "M": 80, "H": 65, "S": 42}
        scales = {"O": 5.0, "F": 6.0, "M": 7.0, "H": 9.0, "S": 12.0}
        weights = {
            key: math.exp(-0.5 * ((score - centers[key]) / scales[key]) ** 2)
            for key in AWARD_KEYS
        }
        total = sum(weights.values()) or 1.0
        raw = {key: weights[key] / total * 100 for key in AWARD_KEYS}
        rounded = {key: int(round(raw[key])) for key in AWARD_KEYS}
        diff = 100 - sum(rounded.values())
        if diff:
            adjust_key = max(AWARD_KEYS, key=lambda key: raw[key])
            rounded[adjust_key] = max(0, rounded[adjust_key] + diff)
        return rounded

    @staticmethod
    def _award_from_score(score: int) -> str:
        if score >= 90:
            return "O"
        if score >= 82:
            return "F"
        if score >= 70:
            return "M"
        if score >= 60:
            return "H"
        return "S/U"
