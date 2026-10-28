"""Model-output parsing & normalisation.

Two output forms are supported (design doc section 四):

* ``json``  — strictly one JSON object; keys per task type.
* ``text``  — free text with fixed markers:
  ``一致性=一致/不一致``, ``声母=…``, ``韵母=…``, ``声调=1-4``, ``选择=A/B``.

Anything unparseable is reported as a ``parse_error`` / ``refused`` / ``empty``
status and excluded from metric denominators, the failure rate is reported
separately (design doc section 三).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .normalize import (
    MappingEntry,
    normalize_output_pair,
    normalize_tone,
    normalize_toneless_string,
)

REFUSAL_PATTERNS = (
    "无法判断", "无法确定", "无法识别", "无法区分", "不能判断", "不能确定",
    "无法回答", "抱歉", "听不清", "没有音频", "缺少音频", "拒绝",
    "unable to", "cannot determine", "can't determine", "i'm sorry", "i am sorry",
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

TONE_ALIASES = ("tone", "声调", "tone_number", "tonal")
INITIAL_ALIASES = ("initial", "声母", "initial_consonant", "onset")
FINAL_ALIASES = ("final", "韵母", "final_vowel", "rime")
CONSISTENT_ALIASES = ("consistent", "consistency", "一致", "一致性", "is_consistent")
CHOICE_ALIASES = ("choice", "选择", "answer", "pick")


@dataclass(slots=True)
class ParsedPrediction:
    status: str
    consistent: bool | None = None
    initial: str | None = None
    final: str | None = None
    tone: int | None = None
    choice: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "consistent": self.consistent,
            "initial": self.initial,
            "final": self.final,
            "tone": self.tone,
            "choice": self.choice,
            "reason": self.reason,
        }


class OutputNormalizer:
    """Normalise predicted fields against the target mapping table."""

    def __init__(self, mapping: Mapping[str, MappingEntry]):
        self.mapping: dict[str, MappingEntry] = dict(mapping)

    def normalize_pair(self, initial: str | None, final: str | None) -> tuple[str, str]:
        return normalize_output_pair(initial, final, self.mapping)

    @staticmethod
    def normalize_tone(value: object) -> int | None:
        return normalize_tone(value)

    @staticmethod
    def normalize_toneless(value: str | None) -> str:
        return normalize_toneless_string(value or "")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from arbitrary text."""
    if not text:
        return None
    candidates = [text.strip(), _strip_code_fences(text)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            pass
    # fall back to the first {...} block (balanced scan)
    for match in _JSON_OBJECT_RE.finditer(text):
        block = match.group(0)
        depth = 0
        in_string = False
        escaped = False
        for i, ch in enumerate(block):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = block[: i + 1]
                    try:
                        value = json.loads(snippet)
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
    return None


def _lookup(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    lowered = {str(k).strip().lower(): v for k, v in mapping.items()}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(int(value))
        return None
    if value is None:
        return None
    text = str(value).strip().lower()
    positives = {"true", "yes", "y", "1", "是", "一致", "对", "正确", "没有错误", "无明显错误"}
    negatives = {"false", "no", "n", "0", "否", "不一致", "错", "错误", "有错误", "有误", "不匹配"}
    if text in positives:
        return True
    if text in negatives:
        return False
    # tolerate sentences: 不一致 must be checked before 一致
    if "不一致" in text or "不匹配" in text or "不一致" in text:
        return False
    if "一致" in text or "匹配" in text:
        return True
    return None


def _is_empty(text: str | None) -> bool:
    return not (text or "").strip()


def _looks_refused(text: str) -> bool:
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


_MARKER_SEP = r"\s*[:：=]\s*"


def _extract_marker_value(text: str, marker: str) -> str | None:
    pattern = re.compile(re.escape(marker) + _MARKER_SEP + r"([^\n，,。;；]+)")
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Task-specific parsers
# ---------------------------------------------------------------------------


def _parse_tone_json(obj: dict[str, Any], normalizer: OutputNormalizer) -> ParsedPrediction:
    raw_tone = _lookup(obj, TONE_ALIASES)
    tone = normalizer.normalize_tone(raw_tone)
    if tone is None:
        return ParsedPrediction("parse_error", reason="missing/invalid tone field")
    return ParsedPrediction("ok", tone=tone)


def _parse_tone_text(text: str, normalizer: OutputNormalizer) -> ParsedPrediction:
    tone_text = _extract_marker_value(text, "声调")
    if tone_text is None:
        match = re.search(r"声调[^\n0-9一二三四]{0,8}([1-4一二三四])", text)
        tone_text = match.group(1) if match else None
    tone = normalizer.normalize_tone(tone_text)
    if tone is None:
        return ParsedPrediction("parse_error", reason="missing tone marker")
    return ParsedPrediction("ok", tone=tone)


def _parse_detect_json(obj: dict[str, Any], normalizer: OutputNormalizer) -> ParsedPrediction:
    raw_consistent = _lookup(obj, CONSISTENT_ALIASES)
    consistent = _parse_bool(raw_consistent)
    tone = normalizer.normalize_tone(_lookup(obj, TONE_ALIASES))
    if consistent is None:
        return ParsedPrediction("parse_error", tone=tone, reason="missing/invalid consistent field")
    return ParsedPrediction("ok", consistent=consistent, tone=tone)


def _parse_detect_text(text: str, normalizer: OutputNormalizer) -> ParsedPrediction:
    consistent: bool | None = None
    # preferred marker from the design doc: 一致=是 / 一致=否
    marker = re.search(r"一致\s*[=:：]\s*(是|否|一致|不一致|true|false|True|False)", text)
    if marker:
        consistent = _parse_bool(marker.group(1))
    if consistent is None:
        # tolerate 一致性=... and bare 一致/不一致 sentences (negative first)
        if re.search(r"不一致|不匹配|不吻合", text):
            consistent = False
        elif re.search(r"一致|匹配|吻合", text):
            consistent = True
    tone_text = _extract_marker_value(text, "声调")
    tone = normalizer.normalize_tone(tone_text)
    if consistent is None:
        return ParsedPrediction("parse_error", tone=tone, reason="missing consistent marker")
    return ParsedPrediction("ok", consistent=consistent, tone=tone)


def _parse_free_json(obj: dict[str, Any], normalizer: OutputNormalizer) -> ParsedPrediction:
    raw_initial = _lookup(obj, INITIAL_ALIASES)
    raw_final = _lookup(obj, FINAL_ALIASES)
    raw_tone = _lookup(obj, TONE_ALIASES)
    initial_s, final_s = normalizer.normalize_pair(
        str(raw_initial) if raw_initial is not None else "",
        str(raw_final) if raw_final is not None else "",
    )
    tone = normalizer.normalize_tone(raw_tone)
    if tone is None or (raw_initial is None and raw_final is None):
        return ParsedPrediction("parse_error", initial=initial_s, final=final_s, tone=tone,
                                reason="missing fields")
    return ParsedPrediction("ok", initial=initial_s, final=final_s, tone=tone)


def _parse_free_text(text: str, normalizer: OutputNormalizer) -> ParsedPrediction:
    initial_text = _extract_marker_value(text, "声母")
    final_text = _extract_marker_value(text, "韵母")
    tone_text = _extract_marker_value(text, "声调")
    if tone_text is None:
        match = re.search(r"声调[^\n0-9一二三四]{0,8}([1-4一二三四])", text)
        tone_text = match.group(1) if match else None
    initial_s, final_s = normalizer.normalize_pair(initial_text, final_text)
    tone = normalizer.normalize_tone(tone_text)
    if initial_text is None and final_text is None:
        return ParsedPrediction("parse_error", reason="missing initial/final markers")
    if initial_text is None:
        initial_s = ""
    if final_text is None:
        final_s = ""
    if tone is None:
        return ParsedPrediction("parse_error", initial=initial_s, final=final_s,
                                reason="missing tone marker")
    return ParsedPrediction("ok", initial=initial_s, final=final_s, tone=tone)


def _parse_ab_json(obj: dict[str, Any]) -> ParsedPrediction:
    raw_choice = _lookup(obj, CHOICE_ALIASES)
    choice = _normalise_choice(raw_choice)
    if choice is None:
        return ParsedPrediction("parse_error", reason="missing/invalid choice")
    return ParsedPrediction("ok", choice=choice)


def _parse_ab_text(text: str) -> ParsedPrediction:
    raw = _extract_marker_value(text, "选择")
    if raw is None:
        match = re.search(r"选择[^\nABab]{0,8}([ABab])", text)
        raw = match.group(1) if match else None
    if raw is None:
        match = re.search(r"\b([AB])\b", text)
        raw = match.group(1) if match else None
    choice = _normalise_choice(raw)
    if choice is None:
        return ParsedPrediction("parse_error", reason="missing choice")
    return ParsedPrediction("ok", choice=choice)


def _normalise_choice(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"A", "B"}:
        return text
    match = re.search(r"\b([AB])\b", text) if len(text) <= 8 else None
    if match:
        return match.group(1)
    if text in {"甲", "选项甲"}:
        return "A"
    if text in {"乙", "选项乙"}:
        return "B"
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


JSON_PARSERS = {
    "free": _parse_free_json,
    "tone": _parse_tone_json,
    "detect": _parse_detect_json,
    "ab": _parse_ab_json,
}
TEXT_PARSERS = {
    "free": _parse_free_text,
    "tone": _parse_tone_text,
    "detect": _parse_detect_text,
    "ab": _parse_ab_text,
}


def parse_prediction(
    text: str,
    *,
    task_type: str,
    output_form: str,
    normalizer: OutputNormalizer,
) -> ParsedPrediction:
    """Parse one model response according to its task type and output form."""
    raw = text or ""
    if _is_empty(raw):
        return ParsedPrediction("empty", reason="empty response")

    if output_form == "json":
        obj = extract_json_object(raw)
        if obj is None:
            if _looks_refused(raw):
                return ParsedPrediction("refused", reason="json output missing, refusal-like text")
            return ParsedPrediction("parse_error", reason="no JSON object found")
        parser = JSON_PARSERS.get(task_type)
        if parser is None:
            raise ValueError(f"unknown task_type {task_type!r}")
        return parser(obj, normalizer) if task_type != "ab" else parser(obj)
    if output_form == "text":
        parser = TEXT_PARSERS.get(task_type)
        if parser is None:
            raise ValueError(f"unknown task_type {task_type!r}")
        parsed = parser(raw, normalizer) if task_type != "ab" else parser(raw)
        if not parsed.ok and _looks_refused(raw):
            parsed.status = "refused"
            parsed.reason = parsed.reason or "refusal-like text"
        return parsed

    raise ValueError(f"unknown output_form {output_form!r}")


def parse_response(
    text: str,
    *,
    task_type: str,
    output_form: str,
    normalizer: OutputNormalizer,
) -> ParsedPrediction:
    return parse_prediction(
        text, task_type=task_type, output_form=output_form, normalizer=normalizer,
    )


__all__ = [
    "OutputNormalizer",
    "ParsedPrediction",
    "extract_json_object",
    "parse_prediction",
    "parse_response",
]
