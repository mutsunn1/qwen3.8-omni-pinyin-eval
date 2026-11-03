"""Parsing of model outputs for the sentence-level tasks (design §4).

Every parser returns a plain dict (the contract consumed by ``scoring``):

* A: ``{"parse_status": ..., "syllables": [{"char","initial","final","tone"}], "detail": ...}``
* B: ``{"parse_status": ..., "words": [{"text","tone_ok","consonant_ok","vowel_ok"}], "detail": ...}``
* C: ``{"parse_status": ..., "scores": {"accuracy","fluency","prosody"}, "detail": ...}``

``parse_status`` is one of ``ok / parse_error / refused / empty / api_error``.
Only ``ok`` records enter metric denominators (design §6.3, §10.2).  Missing
boolean verdicts in task B are treated as ``False`` (the model failed to claim
the dimension is correct) and counted in the parse-failure statistics.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..normalize import INITIALS
from .normalize import Syllable, normalize_pinyin_token, split_surface

__all__ = [
    "PARSE_STATUSES",
    "base_status",
    "extract_json",
    "normalize_predicted_pair",
    "parse_generation",
    "parse_task_a",
    "parse_task_b",
    "parse_task_c",
]

PARSE_STATUSES: tuple[str, ...] = ("ok", "parse_error", "refused", "empty", "api_error")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_REFUSAL_HINTS = ("抱歉", "无法", "不能", "sorry", "cannot", "not able", "无法完成", "无法评估")


def base_status(text: str) -> str:
    """Classify the raw reply before JSON extraction: empty / refused / ok."""
    body = (text or "").strip()
    if not body:
        return "empty"
    if "{" in body or "[" in body:
        return "ok"
    lowered = body.lower()
    if any(hint.lower() in lowered for hint in _REFUSAL_HINTS):
        return "refused"
    return "ok"


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of the first JSON value in ``text``.

    Tries the *earliest* opening bracket first so that a JSON array is not
    mistaken for one of the objects nested inside it.
    """
    body = _strip_fences(text)
    if not body:
        return None
    candidates: list[tuple[int, str, str]] = []
    for opener, closer in (("{", "}"), ("[", "]")):
        start = body.find(opener)
        if start >= 0:
            candidates.append((start, opener, closer))
    for start, _opener, closer in sorted(candidates):
        candidate = body[start:]
        for end in range(len(candidate), 0, -1):
            chunk = candidate[:end]
            if not chunk.endswith(closer):
                continue
            repaired = _TRAILING_COMMA_RE.sub(r"\1", chunk)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def _unwrap_array_wrapper(obj: Any, *keys: str) -> Any | None:
    """Unwrap ``[{"words": [...]}]`` / ``[{"syllables": [...]}]`` replies.

    The model occasionally wraps the requested JSON object in a one-element
    array (observed on task B C2).  Returns the inner list when the wrapper
    matches, otherwise ``None``.
    """
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        inner = _first(obj[0], *keys)
        if isinstance(inner, list):
            return inner
    return None


def _first(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def _as_tone(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        tone = int(value)
        return tone if 1 <= tone <= 5 else 0
    text = str(value).strip()
    # Chinese tone words / digits
    table = {"一": 1, "1": 1, "二": 2, "2": 2, "三": 3, "3": 3, "四": 4, "4": 4,
             "五": 5, "5": 5, "轻声": 5, "neutral": 5, "0": 5}
    if text in table:
        return table[text]
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else 0


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "是", "正确", "对", "ok", "good", "correct"}:
        return True
    if text in {"false", "0", "no", "否", "错误", "不对", "bad", "wrong", "incorrect"}:
        return False
    return None


def _clean_token(value: Any) -> str:
    text = _as_text(value).lower().replace("v", "ü").replace("u:", "ü")
    return re.sub(r"[^a-zü]", "", text)


def normalize_predicted_pair(initial: Any, final: Any, tone: Any) -> Syllable:
    """Normalise the model's separate ``initial`` / ``final`` fields.

    Handles the common model behaviours: writing the whole syllable in
    ``initial`` or ``final``, y/w surfaces, ``j/q/x + u``, apical finals, and
    ü/v spellings.  A missing / invalid tone becomes ``0`` (never matches gold).
    """
    tone_n = _as_tone(tone)
    ini = _clean_token(initial)
    fin = _clean_token(final)
    if ini and fin:
        surface = ini + fin
        # y/w initials are not real initials: let the syllable splitter resolve
        if ini in {"y", "w"} or ini not in INITIALS:
            ini2, fin2 = split_surface(surface) if ini not in INITIALS else (ini, fin)
            if ini2 or fin2:
                return Syllable(initial=ini2, final=fin2, tone=tone_n)
        return Syllable(initial=ini, final=fin, tone=tone_n)
    if ini and not fin:
        if ini in INITIALS or ini in {"zh", "ch", "sh"}:
            return Syllable(initial=ini, final="", tone=tone_n)
        syl = normalize_pinyin_token(ini)
        return Syllable(initial=syl.initial, final=syl.final, tone=tone_n)
    if fin and not ini:
        syl = normalize_pinyin_token(fin)
        return Syllable(initial=syl.initial, final=syl.final, tone=tone_n)
    return Syllable(initial="", final="", tone=tone_n)


def parse_task_a(text: str) -> dict[str, Any]:
    """Parse task-A output into ordered per-character syllables."""
    status = base_status(text)
    result: dict[str, Any] = {"parse_status": status, "syllables": [], "detail": ""}
    if status != "ok":
        return result
    obj = extract_json(text)
    entries: Any = None
    if isinstance(obj, dict):
        entries = _first(obj, "syllables", "words", "items", "chars", "拼音", "音节")
    elif isinstance(obj, list):
        entries = _unwrap_array_wrapper(obj, "syllables", "words", "items") or obj
    if not isinstance(entries, list) or not entries:
        result["parse_status"] = "parse_error"
        result["detail"] = "no syllable array found"
        return result
    syllables: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        char = _as_text(_first(entry, "char", "字", "汉字", "character", "text", "zi"))
        initial = _first(entry, "initial", "声母", "ini", "shengmu")
        final = _first(entry, "final", "韵母", "fin", "yunmu")
        tone = _first(entry, "tone", "声调", "tone_number", "diao")
        syl = normalize_predicted_pair(initial, final, tone)
        syllables.append(
            {"char": char, "initial": syl.initial, "final": syl.final, "tone": syl.tone}
        )
    if not syllables:
        result["parse_status"] = "parse_error"
        result["detail"] = "empty syllable entries"
        return result
    result["syllables"] = syllables
    return result


def parse_task_b(text: str) -> dict[str, Any]:
    """Parse task-B output into ordered word verdicts."""
    status = base_status(text)
    result: dict[str, Any] = {"parse_status": status, "words": [], "detail": ""}
    if status != "ok":
        return result
    obj = extract_json(text)
    entries: Any = None
    if isinstance(obj, dict):
        entries = _first(obj, "words", "syllables", "items", "chars", "逐字")
    elif isinstance(obj, list):
        entries = _unwrap_array_wrapper(obj, "words", "syllables", "items") or obj
    if not isinstance(entries, list) or not entries:
        result["parse_status"] = "parse_error"
        result["detail"] = "no word verdict array found"
        return result
    words: list[dict[str, Any]] = []
    n_with_verdict = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        word = _as_text(_first(entry, "text", "char", "字", "汉字", "word", "字"))
        tone_ok = _as_bool(_first(entry, "tone_ok", "tone", "声调正确", "tone_correct"))
        consonant_ok = _as_bool(
            _first(entry, "consonant_ok", "consonant", "声母正确", "initial_ok", "initial")
        )
        vowel_ok = _as_bool(
            _first(entry, "vowel_ok", "vowel", "韵母正确", "final_ok", "final")
        )
        if any(v is not None for v in (tone_ok, consonant_ok, vowel_ok)):
            n_with_verdict += 1
        words.append({
            "text": word,
            "tone_ok": bool(tone_ok),
            "consonant_ok": bool(consonant_ok),
            "vowel_ok": bool(vowel_ok),
        })
    if not words or n_with_verdict == 0:
        result["parse_status"] = "parse_error"
        result["detail"] = "no boolean verdicts found"
        return result
    result["words"] = words
    return result


def parse_task_c(text: str) -> dict[str, Any]:
    """Parse task-C output into ``accuracy`` / ``fluency`` / ``prosody``."""
    status = base_status(text)
    result: dict[str, Any] = {"parse_status": status, "scores": {}, "detail": ""}
    if status != "ok":
        return result
    obj = extract_json(text)
    if not isinstance(obj, dict):
        result["parse_status"] = "parse_error"
        result["detail"] = "no JSON object found"
        return result
    nested = _first(obj, "scores", "dimensions", "ratings", "评分")
    source = nested if isinstance(nested, dict) else obj
    aliases = {
        "accuracy": ("accuracy", "准确度", "准确率", "acc", "accuracy_score"),
        "fluency": ("fluency", "流畅度", "流利度", "flu", "fluency_score"),
        "prosody": ("prosody", "韵律", "韵律感", "pros", "prosody_score", "intonation"),
    }
    scores: dict[str, float] = {}
    for dimension, keys in aliases.items():
        raw = _first(source, *keys)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            match = re.search(r"\d+(?:\.\d+)?", str(raw))
            if not match:
                continue
            value = float(match.group())
        scores[dimension] = round(min(5.0, max(1.0, value)), 2)
    if len(scores) < 3:
        result["parse_status"] = "parse_error"
        result["detail"] = f"missing score dimensions: {sorted(set(aliases) - set(scores))}"
        return result
    result["scores"] = {k: scores[k] for k in ("accuracy", "fluency", "prosody")}
    return result


def parse_generation(task: str, record: dict[str, Any]) -> dict[str, Any]:
    """Parse one stored ``GenerationRecord`` dict into the task-specific result."""
    if record.get("error"):
        return {"parse_status": "api_error", "detail": str(record.get("error"))[:200]}
    text = record.get("response_text") or ""
    task = task.upper()
    if task == "A":
        return parse_task_a(text)
    if task == "B":
        return parse_task_b(text)
    if task == "C":
        return parse_task_c(text)
    raise KeyError(f"unknown sentence task {task!r}")
