#!/usr/bin/env python
"""Independent verifier for a task-D run (design v0.2, 附录 A).

Task D conditions the Qwen evaluator on one of three frozen texts over the same
1,758 OMPAL learner utterances (``D-gold`` / ``D-asr`` / ``D-inject-10``).

Like ``scripts/verify_sentence_run.py`` (A/B/C), this script intentionally does
**not** import the pipeline's scoring / metrics / report / ``d_task`` modules.
It re-implements, from raw artifacts only:

* design-rule text normalisation (NFKC -> drop punctuation/separators/control &
  whitespace -> OpenCC t2s), used identically for gold and input text;
* character-level Levenshtein alignment of the input text T against the OMPAL
  gold text G (own DP), incl. the design's ``text_error_gold`` definition
  (substitution *and* insertion);
* task-D reply parsing (JSON object, per-design ``error_type`` vocabulary
  ``none|tone|consonant|vowel`` — English or Chinese), mapping the model
  alignment array onto input positions, alignment precision/recall/F1,
  text-error P/R/F1, per-dimension localization macro-F1 (only
  text-correct + aligned characters), zero-initial consonant sub-counts and
  whole-sentence Pearson/Spearman/MAE/RMSE;
* ASR text quality (CER, S/D/I, ASR normalization rate, per-learner CER) from
  the archived raw ASR outputs + OMPAL gold labels;
* deterministic error-injection integrity (gold + archived ops == injected
  text) and aggregate ratio / mix;
* cost from API ``usage`` and the design pricing snapshot (input ¥0.8/M,
  cache-hit ¥0.1/M, output ¥2.7/M; audio tokens already inside prompt tokens);
* acceptance gates (design §A.8): parse success >= 95%, alignment coverage
  >= 95%, ASR text 100% generated + archived, task-D cost <= ¥10.

Usage::

    .venv/bin/python scripts/verify_sentence_d_run.py results/sentence/sentence-d-v1
    .venv/bin/python scripts/verify_sentence_d_run.py results/sentence/sentence-d-v1 \\
        --json-out /tmp/verify_d.json --skip-data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Reuse ONLY the frozen normalisation primitive (OpenCC t2s fallback handled there).
from pinyin_eval.sentence.normalize import t2s  # noqa: E402

PRICING = {
    "currency": "CNY",
    "input_per_mtok": 0.8,
    "cache_hit_input_per_mtok": 0.1,
    "output_per_mtok": 2.7,
    "audio_tokens_per_second": 7.0,
    "audio_min_seconds": 1.0,
}

SOURCES = ("D-gold", "D-asr", "D-inject-10")
SOURCE_SLUG = {"D-gold": "gold", "D-asr": "asr", "D-inject-10": "inject10"}
D_DIMENSIONS = ("tone", "consonant", "vowel")
D_SCORE_DIMENSIONS = ("accuracy", "fluency", "prosody")
ERROR_TYPES = ("none", "tone", "consonant", "vowel")
PARSE_STATUSES = ("ok", "parse_error", "refused", "empty", "api_error", "missing")
ACCEPT_PARSE_MIN = 0.95
ACCEPT_COVERAGE_MIN = 0.95
ACCEPT_COST_MAX = 10.0
EXPECTED_UTTS = 1758
FLOAT_TOL = 1e-3  # 0.1 percentage point, independent vs reported estimate

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_REFUSAL_HINTS = ("抱歉", "无法", "不能", "sorry", "cannot", "not able")
_JSON_KEYS_ALIGNMENT = ("alignment", "words", "items")
_TONE_TABLE = {
    "一": 1, "1": 1, "二": 2, "2": 2, "三": 3, "3": 3, "四": 4,
    "4": 4, "五": 5, "5": 5, "轻声": 5, "neutral": 5, "0": 5,
}
_TRUE_STRINGS = {"true", "1", "yes", "是", "正确", "对", "ok", "good", "correct"}
_FALSE_STRINGS = {"false", "0", "no", "否", "错误", "不对", "bad", "wrong", "incorrect"}
_ERROR_TYPE_ALIASES = {
    "none": "none", "无": "none", "ok": "none", "correct": "none",
    "tone": "tone", "声调": "tone", "调": "tone",
    "consonant": "consonant", "声母": "consonant", "initial": "consonant",
    "vowel": "vowel", "韵母": "vowel", "final": "vowel",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def json_load(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(num: float | int | None, den: float | int | None) -> float | None:
    if not den:
        return None
    return float(num or 0) / float(den)


def mean(values: Iterable[float | int | None]) -> float | None:
    defined = [float(v) for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy default style)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = varx = vary = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx, dy = x - mx, y - my
        cov += dx * dy
        varx += dx * dx
        vary += dy * dy
    if varx <= 0.0 or vary <= 0.0:
        return None
    return max(-1.0, min(1.0, cov / math.sqrt(varx * vary)))


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos
        while end + 1 < len(order) and values[order[end + 1]] == values[order[pos]]:
            end += 1
        rank = (pos + end) / 2.0 + 1.0
        for k in range(pos, end + 1):
            ranks[order[k]] = rank
        pos = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def mae(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or len(xs) != len(ys):
        return None
    return sum(abs(x - y) for x, y in zip(xs, ys, strict=True)) / len(xs)


def rmse(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or len(xs) != len(ys):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys, strict=True)) / len(xs))


def f1_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision, "recall": recall, "f1": f1,
    }


def macro_f1(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    f1_error = f1_counts(tp, fp, fn)["f1"]
    f1_ok = f1_counts(tn, fn, fp)["f1"]
    values = [v for v in (f1_error, f1_ok) if v is not None]
    return {
        "macro_f1": (sum(values) / len(values)) if values else None,
        "f1_error": f1_error,
        "f1_ok": f1_ok,
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def add_check(
    checks: list[dict[str, Any]],
    item: str,
    expected: Any,
    actual: Any,
    ok: bool | None = None,
    detail: str = "",
) -> None:
    if ok is None:
        ok = str(expected) == str(actual)
    checks.append(
        {
            "item": item,
            "expected": expected,
            "actual": actual,
            "ok": bool(ok),
            "detail": detail,
        }
    )


# ---------------------------------------------------------------------------
# Text normalisation (design A.2 rule) and character DP
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Design A.2 / A.5 normalisation (identical for gold and input text).

    NFKC -> drop punctuation (``P*``), separator (``Z*``), control (``C*``) and
    whitespace -> OpenCC t2s.  Letters/digits are kept (ASR can emit Latin
    tokens for mispronounced characters); this matches ``run_d_asr.normalize``.
    """
    body = unicodedata.normalize("NFKC", text or "")
    kept = "".join(
        ch for ch in body if not unicodedata.category(ch).startswith(("P", "Z", "C"))
    )
    return t2s(kept)


def text_chars(text: str) -> list[str]:
    return list(normalize_text(text))


def edit_distance_ops(
    gold: Sequence[str], hyp: Sequence[str]
) -> tuple[int, dict[str, int], list[tuple[str, int | None, int | None]]]:
    """Levenshtein edit script with the design's deterministic tie order.

    Tie order: match < substitution < deletion (gold-only) < insertion
    (hyp-only), matching ``run_d_asr.align_chars``' documented CSR order.
    Returns ``(distance, {"sub":n,"del":n,"ins":n}, ops)`` where each op is
    ``(kind, gold_index|None, hyp_index|None)``.
    """
    n, m = len(gold), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gold[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
            0 if gold[i - 1] == hyp[j - 1] else 1
        ):
            kind = "match" if gold[i - 1] == hyp[j - 1] else "sub"
            ops.append((kind, i - 1, j - 1))
            i -= 1
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
            continue
        ops.append(("ins", None, j - 1))
        j -= 1
    ops.reverse()
    distance = sum(1 for kind, _g, _h in ops if kind != "match")
    counts = {"sub": 0, "del": 0, "ins": 0}
    for kind, _g, _h in ops:
        if kind in counts:
            counts[kind] += 1
    return distance, counts, ops


def align_pairs(gold: Sequence[str], hyp: Sequence[str]) -> list[tuple[int, int | None]]:
    """Gold-index -> hyp-index alignment (None = deletion); own DP."""
    n, m = len(gold), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gold[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    pairs: list[tuple[int, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if gold[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
            continue
        j -= 1
    pairs.reverse()
    return pairs


def gold_alignment(
    text: str, gold_text: str, gold_labels: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independent re-implementation of design A.4 gold alignment."""
    input_chars = text_chars(text)
    gold_chars = text_chars(gold_text)
    pairs = align_pairs(gold_chars, input_chars)
    matched: dict[int, int] = {h: g for g, h in pairs if h is not None}
    deleted = [g for g, h in pairs if h is None]
    per_char: list[dict[str, Any]] = []
    for index, char in enumerate(input_chars):
        gold_index = matched.get(index)
        if gold_index is None:
            per_char.append(
                {
                    "aligned": False,
                    "text_correct": None,
                    "text_error": True,  # design A.4: insertion counts as text_error
                    "error_type": None,
                    "labels": None,
                    "gold_index": None,
                }
            )
            continue
        labels = gold_labels[gold_index] if gold_index < len(gold_labels) else None
        correct = char == gold_chars[gold_index]
        error_type: str | None = None
        if correct and labels is not None:
            if int(labels.get("tone", 1)) == 0:
                error_type = "tone"
            elif int(labels.get("consonant", 1)) == 0:
                error_type = "consonant"
            elif int(labels.get("vowel", 1)) == 0:
                error_type = "vowel"
            else:
                error_type = "none"
        per_char.append(
            {
                "aligned": True,
                "text_correct": correct,
                "text_error": not correct,
                "error_type": error_type,
                "labels": labels,
                "gold_index": gold_index,
            }
        )
    stats = {
        "n_text_chars": len(input_chars),
        "n_gold_chars": len(gold_chars),
        "n_insertions_text": sum(1 for entry in per_char if not entry["aligned"]),
        "n_deletions_gold": len(deleted),
        "n_aligned": sum(1 for entry in per_char if entry["aligned"]),
        "n_text_correct": sum(1 for entry in per_char if entry["text_correct"]),
    }
    return per_char, stats


# ---------------------------------------------------------------------------
# Reply parsing (independent of pinyin_eval.sentence.d_task)
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def extract_json(text: str) -> Any | None:
    body = _strip_fences(text)
    if not body:
        return None
    decoder = json.JSONDecoder()
    for start in sorted(pos for pos in (body.find("{"), body.find("[")) if pos >= 0):
        try:
            value, _end = decoder.raw_decode(body[start:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


def as_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group())
    if math.isnan(number) or math.isinf(number):
        return None
    return round(min(5.0, max(1.0, number)), 2)


def normalize_error_type(value: Any) -> str | None:
    """Parse the design's ``error_type`` vocabulary (English *or* Chinese).

    NOTE: this is deliberately the design-correct parse: the pipeline bug found
    in phase A (English values collapsed to ``none``) shows up as a metric
    discrepancy instead of being replicated here.
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in ("", "null", "none", "无"):
        return "none"
    if raw in _ERROR_TYPE_ALIASES:
        return _ERROR_TYPE_ALIASES[raw]
    chinese = normalize_text(str(value))
    return _ERROR_TYPE_ALIASES.get(chinese)


def parse_d_reply(text: str) -> dict[str, Any]:
    """Parse one task-D reply into alignment entries + sentence scores."""
    body = (text or "").strip()
    if not body:
        return {"parse_status": "empty", "alignment": [], "scores": {}}
    obj = extract_json(body)
    if obj is None:
        if "{" not in body and "[" not in body:
            lowered = body.lower()
            if any(hint.lower() in lowered for hint in _REFUSAL_HINTS):
                return {"parse_status": "refused", "alignment": [], "scores": {}}
        return {"parse_status": "parse_error", "detail": "no JSON value found"}
    wrapper_unwrapped = False
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        inner = obj[0]
        if any(isinstance(inner.get(key), list) for key in _JSON_KEYS_ALIGNMENT):
            obj = inner
            wrapper_unwrapped = True
    if not isinstance(obj, Mapping):
        return {"parse_status": "parse_error", "detail": "no JSON object found"}
    entries: Any = None
    for key in _JSON_KEYS_ALIGNMENT:
        if isinstance(obj.get(key), list):
            entries = obj[key]
            break
    if not isinstance(entries, list) or not entries:
        return {"parse_status": "parse_error", "detail": "no alignment array found"}
    alignment: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        alignment.append(
            {
                "text": str(entry.get("text") or entry.get("char") or ""),
                "aligned": as_bool(entry.get("aligned")),
                "text_error": as_bool(entry.get("text_error")),
                "error_type": normalize_error_type(entry.get("error_type")),
                "score": as_score(entry.get("score")),
            }
        )
    if not alignment:
        return {"parse_status": "parse_error", "detail": "empty alignment entries"}
    nested = obj.get("scores")
    source = nested if isinstance(nested, Mapping) else obj
    scores: dict[str, float] = {}
    for dimension in D_SCORE_DIMENSIONS:
        value = as_score(source.get(dimension))
        if value is not None:
            scores[dimension] = value
    if len(scores) < 3:
        return {
            "parse_status": "parse_error",
            "detail": f"missing scores: {sorted(set(D_SCORE_DIMENSIONS) - set(scores))}",
        }
    return {
        "parse_status": "ok",
        "alignment": alignment,
        "scores": {dimension: scores[dimension] for dimension in D_SCORE_DIMENSIONS},
        "wrapper_unwrapped": wrapper_unwrapped,
    }


def map_hyp_entries(
    hyp: Sequence[Mapping[str, Any]], input_chars: Sequence[str]
) -> tuple[list[Mapping[str, Any] | None], list[Mapping[str, Any]]]:
    """Map model alignment entries onto input-text positions (own DP).

    Returns ``(mapped, extras)`` where ``extras`` are model entries with no
    input-text position (design A.3 "多字"); an aligned=True claim on an extra
    entry is an alignment false positive.
    """
    if len(hyp) == len(input_chars):
        return list(hyp), []
    gold_texts = [normalize_text(ch) or "?" for ch in input_chars]
    hyp_texts = [normalize_text(str(entry.get("text") or "")) or "?" for entry in hyp]
    pairs = align_pairs(gold_texts, hyp_texts)
    mapped: list[Mapping[str, Any] | None] = [None] * len(input_chars)
    matched_hyp: set[int] = set()
    for text_index, hyp_index in pairs:
        if hyp_index is None:
            continue
        mapped[text_index] = hyp[hyp_index]
        matched_hyp.add(hyp_index)
    extras = [entry for index, entry in enumerate(hyp) if index not in matched_hyp]
    return mapped, extras


# ---------------------------------------------------------------------------
# Per-item scoring (independent re-implementation of design A.3/A.4/A.5)
# ---------------------------------------------------------------------------


def score_d_item(
    item: Mapping[str, Any], parsed: Mapping[str, Any], gold: Mapping[str, Any]
) -> dict[str, Any]:
    """Score one D item given ``parsed`` reply and independently aligned gold."""
    gold_per_char = gold["per_char"]
    stats = gold["stats"]
    input_chars = gold["input_chars"]
    row: dict[str, Any] = {
        "source": str(item.get("meta", {}).get("text_source") or ""),
        "utt_id": str(item.get("meta", {}).get("utt_id") or ""),
        "n_text_chars": len(input_chars),
        "n_gold_chars": len(gold_per_char),
        "n_gold_chars_source": stats["n_gold_chars"],
        "n_gold_deletions": stats["n_deletions_gold"],
        "n_outputs": 0,
        "n_mapped": 0,
        "n_localizable": 0,
        "parse_ok": parsed.get("parse_status") == "ok",
        "parse_status": str(parsed.get("parse_status") or "parse_error"),
        "n_outputs_extra": 0,
        "n_extra_aligned": 0,
        "n_text_copy_mismatch": 0,
        "n_score": 0,
        "score_sum": 0.0,
        "n_score_error": 0,
        "score_sum_error": 0.0,
        "n_score_correct": 0,
        "score_sum_correct": 0.0,
    }
    counts = {
        "align_tp": 0, "align_fp": 0, "align_fn": 0, "align_tn": 0,
        "terr_tp": 0, "terr_fp": 0, "terr_fn": 0, "terr_tn": 0,
        "terr_missing": 0,
    }
    for dimension in D_DIMENSIONS:
        for key in ("tp", "fp", "fn", "tn"):
            counts[f"loc_{dimension}_{key}"] = 0
    for key in ("tp", "fp", "fn", "tn"):
        counts[f"loc_zero_cons_{key}"] = 0
    row.update(counts)
    for dimension in D_SCORE_DIMENSIONS:
        row[f"gold_{dimension}"] = (item.get("scores") or {}).get(dimension)
        row[f"pred_{dimension}"] = None
    if not row["parse_ok"]:
        return row
    hyp = list(parsed.get("alignment") or [])
    row["n_outputs"] = len(hyp)
    mapped, extras = map_hyp_entries(hyp, input_chars)
    row["n_outputs_extra"] = len(extras)
    row["n_extra_aligned"] = sum(1 for entry in extras if entry.get("aligned"))
    row["n_mapped"] = sum(1 for entry in mapped if entry is not None)
    row["n_text_copy_mismatch"] = sum(
        1
        for index, entry in enumerate(mapped)
        if entry is not None
        and normalize_text(str(entry.get("text") or ""))
        != normalize_text(input_chars[index])
    )
    for index, gold_entry in enumerate(gold_per_char):
        gold_aligned = bool(gold_entry["aligned"])
        gold_error = bool(gold_entry["text_error"]) if gold_entry["aligned"] else True
        entry = mapped[index] if index < len(mapped) else None
        pred_aligned = bool(entry.get("aligned")) if entry is not None else False
        if gold_aligned:
            counts["align_tp" if pred_aligned else "align_fn"] += 1
        else:
            counts["align_fp" if pred_aligned else "align_tn"] += 1
        # --- text_error (design A.4: substitution and insertion are errors) ---
        if entry is None:
            if gold_error:
                counts["terr_fn"] += 1
            else:
                counts["terr_tn"] += 1
            counts["terr_missing"] += 1
        else:
            pred_text_error = entry.get("text_error")
            if pred_text_error is None:
                counts["terr_fn" if gold_error else "terr_tn"] += 1
            elif pred_text_error:
                counts["terr_tp" if gold_error else "terr_fp"] += 1
            else:
                counts["terr_fn" if gold_error else "terr_tn"] += 1
        # --- localization: only text-correct + aligned characters -------------
        if not gold_aligned or not gold_entry["text_correct"]:
            continue
        row["n_localizable"] += 1
        labels = gold_entry["labels"] or {}
        pred_error_type = entry.get("error_type") if entry is not None else None
        for dimension in D_DIMENSIONS:
            label = labels.get(dimension)
            if label is None:
                continue
            gold_error = int(label) == 0
            pred_error = pred_error_type == dimension
            key = f"loc_{dimension}_"
            if gold_error:
                counts[key + ("tp" if pred_error else "fn")] += 1
            else:
                counts[key + ("fp" if pred_error else "tn")] += 1
            if dimension == "consonant" and bool(labels.get("zero_initial")):
                zkey = "loc_zero_cons_"
                if gold_error:
                    counts[zkey + ("tp" if pred_error else "fn")] += 1
                else:
                    counts[zkey + ("fp" if pred_error else "tn")] += 1
        # --- descriptive per-char score --------------------------------------
        if entry is not None and entry.get("score") is not None:
            value = float(entry["score"])
            row["n_score"] += 1
            row["score_sum"] += value
            label_error = any(int(labels.get(dim, 1)) == 0 for dim in D_DIMENSIONS)
            if label_error:
                row["n_score_error"] += 1
                row["score_sum_error"] += value
            else:
                row["n_score_correct"] += 1
                row["score_sum_correct"] += value
    # model entries without an input position: aligned=True claims are FP (A.3)
    row["align_fp"] += row["n_extra_aligned"]
    row.update(counts)
    for dimension in D_SCORE_DIMENSIONS:
        row[f"pred_{dimension}"] = (parsed.get("scores") or {}).get(dimension)
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _sum_counts(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, int]:
    return {
        key: sum(int(row.get(f"{prefix}{key}", 0) or 0) for row in rows)
        for key in ("tp", "fp", "fn", "tn")
    }


def _scoring_pairs(
    rows: Sequence[Mapping[str, Any]], dimension: str
) -> tuple[list[float], list[float]]:
    gold: list[float] = []
    pred: list[float] = []
    for row in rows:
        if not row.get("parse_ok"):
            continue
        if dimension == "mean_dimension":
            golds = [row.get(f"gold_{dim}") for dim in D_SCORE_DIMENSIONS]
            preds = [row.get(f"pred_{dim}") for dim in D_SCORE_DIMENSIONS]
            if any(value is None for value in (*golds, *preds)):
                continue
            gold.append(sum(float(v) for v in golds) / 3.0)
            pred.append(sum(float(v) for v in preds) / 3.0)
        else:
            gold_value = row.get(f"gold_{dimension}")
            pred_value = row.get(f"pred_{dimension}")
            if gold_value is None or pred_value is None:
                continue
            gold.append(float(gold_value))
            pred.append(float(pred_value))
    return pred, gold


def _scoring_stats(rows: Sequence[Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    pred, gold = _scoring_pairs(rows, dimension)
    return {
        "n": len(gold),
        "pearson": pearson(pred, gold),
        "spearman": spearman(pred, gold),
        "mae": mae(pred, gold),
        "rmse": rmse(pred, gold),
    }


def aggregate_d(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one text source's scored rows (own implementation of A.5)."""
    ok_rows = [row for row in rows if row.get("parse_ok")]
    align = _sum_counts(ok_rows, "align_")
    terr = _sum_counts(ok_rows, "terr_")
    alignment = f1_counts(align["tp"], align["fp"], align["fn"])
    text_error = f1_counts(terr["tp"], terr["fp"], terr["fn"])
    text_error["n_missing_outputs"] = sum(
        int(row.get("terr_missing", 0) or 0) for row in ok_rows
    )
    localization: dict[str, Any] = {}
    for dimension in D_DIMENSIONS:
        counts = _sum_counts(ok_rows, f"loc_{dimension}_")
        localization[dimension] = macro_f1(
            counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        )
    zero_counts = _sum_counts(ok_rows, "loc_zero_cons_")
    n_text = sum(int(row.get("n_text_chars", 0) or 0) for row in ok_rows)
    n_mapped = sum(int(row.get("n_mapped", 0) or 0) for row in ok_rows)
    n_outputs = sum(int(row.get("n_outputs", 0) or 0) for row in ok_rows)
    n_extra = sum(int(row.get("n_outputs_extra", 0) or 0) for row in ok_rows)
    n_extra_aligned = sum(int(row.get("n_extra_aligned", 0) or 0) for row in ok_rows)
    n_text_copy_mismatch = sum(
        int(row.get("n_text_copy_mismatch", 0) or 0) for row in ok_rows
    )
    n_gold_deletions = sum(int(row.get("n_gold_deletions", 0) or 0) for row in ok_rows)
    n_gold_chars_source = sum(
        int(row.get("n_gold_chars_source", 0) or 0) for row in ok_rows
    )
    n_score = sum(int(row.get("n_score", 0) or 0) for row in ok_rows)
    score_sum = sum(float(row.get("score_sum", 0.0) or 0.0) for row in ok_rows)
    n_correct = sum(int(row.get("n_score_correct", 0) or 0) for row in ok_rows)
    sum_correct = sum(float(row.get("score_sum_correct", 0.0) or 0.0) for row in ok_rows)
    n_error = sum(int(row.get("n_score_error", 0) or 0) for row in ok_rows)
    sum_error = sum(float(row.get("score_sum_error", 0.0) or 0.0) for row in ok_rows)
    result: dict[str, Any] = {
        "n_items": len(rows),
        "n_parse_ok": len(ok_rows),
        "n_parse_failures": len(rows) - len(ok_rows),
        "parse_success_rate": ratio(len(ok_rows), len(rows)),
        "parse_failure_rate": ratio(len(rows) - len(ok_rows), len(rows)),
        "n_text_chars": n_text,
        "n_gold_chars": sum(int(row.get("n_gold_chars", 0) or 0) for row in ok_rows),
        "alignment_coverage": ratio(n_mapped, n_text),
        "output_coverage": ratio(n_outputs, n_text),
        "n_extra_outputs": n_extra,
        "n_extra_aligned": n_extra_aligned,
        "n_text_copy_mismatch": n_text_copy_mismatch,
        "gold_side_coverage": (
            None
            if not n_gold_chars_source
            else 1.0 - (n_gold_deletions / n_gold_chars_source)
        ),
        "n_localizable": sum(int(row.get("n_localizable", 0) or 0) for row in ok_rows),
        "alignment": alignment,
        "text_error": text_error,
        "localization": localization,
        "zero_initial_consonant": macro_f1(
            zero_counts["tp"], zero_counts["fp"], zero_counts["fn"], zero_counts["tn"]
        ),
        "scoring": {
            dimension: _scoring_stats(ok_rows, dimension)
            for dimension in (*D_SCORE_DIMENSIONS, "mean_dimension")
        },
        "per_char_score": {
            "n": n_score,
            "mean": ratio(score_sum, n_score),
            "mean_correct": ratio(sum_correct, n_correct),
            "mean_error": ratio(sum_error, n_error),
        },
    }
    return result


def _headline_metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    agg = aggregate_d(rows)
    values: dict[str, float | None] = {
        "alignment_f1": agg["alignment"]["f1"],
        "text_error_f1": agg["text_error"]["f1"],
        "mean_dimension_pearson": agg["scoring"]["mean_dimension"]["pearson"],
        "mean_dimension_mae": agg["scoring"]["mean_dimension"]["mae"],
    }
    for dimension in D_DIMENSIONS:
        values[f"{dimension}_macro_f1"] = agg["localization"][dimension]["macro_f1"]
    return values


def compare_sources(
    rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]], baseline: str = "D-gold"
) -> dict[str, Any]:
    """Paired point-estimate deltas vs the D-gold baseline (own implementation)."""
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        return {
            str(row.get("utt_id")): row
            for row in rows
            if row.get("parse_ok") and row.get("utt_id")
        }

    base = index(rows_by_source.get(baseline, []))
    result: dict[str, Any] = {"baseline": baseline, "comparisons": {}}
    for source, rows in rows_by_source.items():
        if source == baseline:
            continue
        other = index(rows)
        common = sorted(set(base) & set(other))
        paired_base = [base[utt] for utt in common]
        paired_other = [other[utt] for utt in common]
        diffs: dict[str, dict[str, Any]] = {}
        base_values = _headline_metric_values(paired_base)
        other_values = _headline_metric_values(paired_other)
        for metric in base_values:
            left, right = other_values.get(metric), base_values.get(metric)
            diffs[metric] = {
                "estimate": (left - right) if left is not None and right is not None else None
            }
        result["comparisons"][source] = {
            "n_paired": len(common),
            "difference": diffs,
        }
    return result


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def usage_of(record: Mapping[str, Any]) -> dict[str, int]:
    usage = record.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(details.get("cached_tokens") or 0),
        "audio_tokens": int(details.get("audio_tokens") or 0),
        "reasoning_tokens": int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ),
    }


def record_cost(usage: Mapping[str, int]) -> float:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = max(0, min(int(usage.get("cached_tokens") or 0), prompt))
    return (
        (prompt - cached) * PRICING["input_per_mtok"]
        + cached * PRICING["cache_hit_input_per_mtok"]
        + int(usage.get("completion_tokens") or 0) * PRICING["output_per_mtok"]
    ) / 1_000_000.0


# ---------------------------------------------------------------------------
# Data verification (d_items / ASR / injection)
# ---------------------------------------------------------------------------


def _probe_audio(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "ok": True,
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "duration_s": round(info.frames / info.samplerate, 6)
            if info.samplerate
            else 0.0,
        }
    except Exception as exc:  # noqa: BLE001 - report, never abort
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def apply_injection_ops(
    gold_chars: Sequence[str], ops: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    """Apply archived injection ops sequentially; returns (chars, problems)."""
    chars = list(gold_chars)
    problems: list[str] = []
    for index, op in enumerate(ops):
        kind = op.get("op")
        position = op.get("position")
        if not isinstance(position, int) or position < 0 or position > len(chars):
            problems.append(f"op {index}: invalid position {position!r}")
            continue
        if kind == "sub":
            if position >= len(chars):
                problems.append(f"op {index}: sub position out of range")
                continue
            if chars[position] != op.get("gold"):
                problems.append(
                    f"op {index}: sub gold mismatch {chars[position]!r} != {op.get('gold')!r}"
                )
                continue
            chars[position] = str(op.get("new"))
        elif kind == "del":
            if position >= len(chars) or chars[position] != op.get("gold"):
                problems.append(f"op {index}: del mismatch")
                continue
            chars.pop(position)
        elif kind == "ins":
            chars.insert(position, str(op.get("new")))
        else:
            problems.append(f"op {index}: unknown op {kind!r}")
    return chars, problems


def verify_d_data(
    manifest_dir: Path,
    data_dir: Path,
    repo_root: Path,
    *,
    check_audio: bool = False,
) -> dict[str, Any]:
    """Independently verify d_items.jsonl, ASR archive and injection integrity."""
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {"checks": checks}

    b_rows = load_jsonl(manifest_dir / "ompal_b_items.jsonl")
    c_rows = load_jsonl(manifest_dir / "ompal_c_items.jsonl")
    b_by_utt = {
        str((row.get("meta") or {}).get("utt_id")): row for row in b_rows
    }
    c_by_utt = {
        str((row.get("meta") or {}).get("utt_id")): row for row in c_rows
    }
    d_items = load_jsonl(manifest_dir / "d_items.jsonl")
    if not d_items:
        add_check(checks, "d_items.jsonl present", True, False,
                  detail=f"missing {manifest_dir / 'd_items.jsonl'}")
        result["n_checks"] = len(checks)
        result["n_failed"] = sum(1 for check in checks if not check["ok"])
        return result

    add_check(checks, "d_items count", EXPECTED_UTTS * 3, len(d_items))
    add_check(
        checks,
        "d_items unique item ids",
        len(d_items),
        len({str(row.get("item_id")) for row in d_items}),
    )
    add_check(
        checks,
        "d_items tasks",
        ["D"],
        sorted({str(row.get("task")) for row in d_items}),
    )
    by_source: Counter[str] = Counter()
    by_utt: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in d_items:
        meta = row.get("meta") or {}
        source = str(meta.get("text_source") or "")
        utt = str(meta.get("utt_id") or "")
        by_source[source] += 1
        by_utt[utt][source] = row
    add_check(checks, "per-source item counts", {source: EXPECTED_UTTS for source in SOURCES},
              dict(by_source))
    add_check(checks, "distinct utterances", EXPECTED_UTTS, len(by_utt))
    add_check(checks, "three sources per utterance", 0,
              sum(1 for utt, sources in by_utt.items() if set(sources) != set(SOURCES)))

    # --- gold / scores / labels consistency with the OMPAL manifests ----------
    missing_gold: list[str] = []
    gold_text_mismatch: list[str] = []
    label_mismatch: list[str] = []
    score_mismatch: list[str] = []
    native_items = 0
    alignment_mismatch: list[str] = []
    n_alignment_checked = 0
    for utt, sources in by_utt.items():
        gold_row = b_by_utt.get(utt)
        if gold_row is None:
            missing_gold.append(utt)
            continue
        gold_text = str(gold_row.get("chars") or "")
        gold_labels = list(gold_row.get("labels") or [])
        gold_scores = dict((c_by_utt.get(utt) or {}).get("scores") or {})
        for source in SOURCES:
            item = sources.get(source)
            if item is None:
                continue
            native_items += bool(item.get("native"))
            if source == "D-gold" and str(item.get("text") or "") != gold_text:
                gold_text_mismatch.append(f"{utt}:D-gold text != OMPAL chars")
            if list(item.get("labels") or []) != gold_labels:
                label_mismatch.append(f"{utt}:{source}")
            if dict(item.get("scores") or {}) != gold_scores:
                score_mismatch.append(f"{utt}:{source}")
            per_char, _stats = gold_alignment(
                str(item.get("text") or ""), gold_text, gold_labels
            )
            archived = (item.get("meta") or {}).get("gold_alignment") or []
            n_alignment_checked += 1
            if len(per_char) != len(archived):
                alignment_mismatch.append(f"{utt}:{source}:len")
                continue
            for index, (mine, theirs) in enumerate(zip(per_char, archived, strict=True)):
                if (
                    bool(mine["aligned"]) != bool(theirs.get("aligned"))
                    or mine["text_correct"] != theirs.get("text_correct")
                    or mine["error_type"] != theirs.get("error_type")
                ):
                    alignment_mismatch.append(f"{utt}:{source}:{index}")
                    break
    add_check(checks, "utterances present in OMPAL B manifest", 0, len(missing_gold))
    add_check(checks, "D-gold text equals OMPAL reference text", 0, len(gold_text_mismatch))
    add_check(checks, "items carry OMPAL labels", 0, len(label_mismatch))
    add_check(checks, "items carry OMPAL sentence scores", 0, len(score_mismatch))
    add_check(checks, "no native items in D", 0, native_items)
    add_check(
        checks,
        "gold alignment recomputed == archived (aligned/text_correct/error_type)",
        0,
        len(alignment_mismatch),
        detail=f"checked={n_alignment_checked}",
    )

    # --- ASR archive + quality ----------------------------------------------
    asr_rows = load_jsonl(data_dir / "asr_outputs.jsonl")
    asr_summary = json_load(data_dir / "d_asr_summary.json", {}) or {}
    asr_by_utt = {
        str(row.get("utt_id")): row for row in asr_rows if row.get("ok")
    }
    add_check(checks, "ASR output rows", EXPECTED_UTTS, len(asr_rows))
    add_check(checks, "ASR ok rows", EXPECTED_UTTS, len(asr_by_utt))
    add_check(checks, "ASR summary n_items", EXPECTED_UTTS, asr_summary.get("n_items"))
    add_check(checks, "ASR summary n_ok", EXPECTED_UTTS, asr_summary.get("n_ok"))
    add_check(checks, "ASR summary n_fail", 0, asr_summary.get("n_fail"))
    add_check(
        checks,
        "ASR provenance recorded",
        True,
        bool(asr_summary.get("model_id"))
        and bool(asr_summary.get("funasr_version"))
        and bool(asr_summary.get("decode_params")),
    )
    missing_asr = [utt for utt in b_by_utt if utt not in asr_by_utt]
    empty_asr = [
        utt for utt, row in asr_by_utt.items()
        if not str(row.get("raw_text") or "").strip()
    ]
    add_check(checks, "every utterance has an ASR row", 0, len(missing_asr),
              detail=json.dumps(missing_asr[:5]))
    add_check(checks, "ASR raw text non-empty", 0, len(empty_asr),
              detail=json.dumps(empty_asr[:5]))
    asr_text_mismatch: list[str] = []
    for utt, sources in by_utt.items():
        item = sources.get("D-asr")
        if item is None:
            continue
        expected = normalize_text(
            str(asr_by_utt.get(utt, {}).get("raw_text") or "")
        )
        if str(item.get("text") or "") != expected:
            asr_text_mismatch.append(utt)
    add_check(checks, "D-asr item text == normalise(archived raw ASR)", 0,
              len(asr_text_mismatch), detail=json.dumps(asr_text_mismatch[:5]))

    # independent ASR quality recomputation
    quality = recompute_asr_quality(b_by_utt, asr_by_utt)
    reported_quality = (
        asr_summary.get("text_quality") or asr_summary.get("quality") or {}
    )
    reported_sdi = reported_quality.get("sdi") or {}
    reported_norm = reported_quality.get("normalization_rate") or {}
    reported_norm_rate = (
        reported_norm.get("rate")
        if isinstance(reported_norm, Mapping)
        else reported_norm
    )
    add_check(
        checks,
        "ASR CER (micro) recomputed",
        _round_or_none(reported_quality.get("cer_micro")),
        _round_or_none(quality["cer_micro"]),
        ok=_close(reported_quality.get("cer_micro"), quality["cer_micro"])
        if reported_quality else False,
    )
    for reported_key, my_key in (
        ("substitutions", "sub"),
        ("deletions", "del"),
        ("insertions", "ins"),
    ):
        reported = reported_sdi.get(reported_key)
        if reported is None:  # older/alternate schema
            reported = reported_sdi.get({"substitutions": "n_sub", "deletions": "n_del",
                                          "insertions": "n_ins"}[reported_key])
        add_check(
            checks,
            f"ASR S/D/I {reported_key} recomputed",
            reported,
            quality["sdi"][my_key],
            ok=reported == quality["sdi"][my_key],
        )
    add_check(
        checks,
        "ASR normalization rate recomputed",
        _round_or_none(reported_norm_rate),
        _round_or_none(quality["normalization_rate"]),
        ok=_close(reported_norm_rate, quality["normalization_rate"])
        if reported_norm_rate is not None else False,
    )
    for reported_key, my_key in (
        ("cer_mean_utterance", "cer_mean_utterance"),
        ("cer_median_utterance", "cer_median_utterance"),
        ("cer_p90_utterance", "cer_p90_utterance"),
        ("sentence_exact_rate", "sentence_exact_rate"),
    ):
        reported = reported_quality.get(reported_key)
        add_check(
            checks,
            f"ASR {reported_key} recomputed",
            _round_or_none(reported),
            _round_or_none(quality.get(my_key)),
            ok=_close(reported, quality.get(my_key), 1e-6)
            if reported is not None else False,
        )
    reported_learner = reported_quality.get("per_learner_cer") or {}
    for key in ("median", "p90", "max"):
        reported = reported_learner.get(key)
        add_check(
            checks,
            f"ASR per-learner CER {key} recomputed",
            _round_or_none(reported),
            _round_or_none(quality["per_learner_cer"].get(key)),
            ok=_close(reported, quality["per_learner_cer"].get(key), 1e-6)
            if reported is not None else False,
        )
    result["asr_quality"] = {
        "independent": quality,
        "reported": reported_quality,
    }

    # --- injection integrity ------------------------------------------------
    injection_checks: list[dict[str, Any]] = []
    inject_rows = [
        (utt, sources["D-inject-10"]) for utt, sources in by_utt.items()
        if "D-inject-10" in sources
    ]
    bad_reconstruction: list[str] = []
    bad_gold: list[str] = []
    total_ops = Counter()
    total_chars = 0
    n_items_with_ops = 0
    for utt, item in inject_rows:
        gold_row = b_by_utt.get(utt) or {}
        gold_text = str(gold_row.get("chars") or "")
        ops = list((item.get("meta") or {}).get("injection_ops") or [])
        rebuilt, problems = apply_injection_ops(list(gold_text), ops)
        if problems or "".join(rebuilt) != str(item.get("text") or ""):
            bad_reconstruction.append(utt)
        if str(item.get("text") or "") == gold_text:
            bad_gold.append(utt)
        for op in ops:
            total_ops[str(op.get("op"))] += 1
        total_chars += len(text_chars(gold_text))
        n_items_with_ops += 1
    add_check(checks, "injection ops reconstruct the injected text", 0,
              len(bad_reconstruction), detail=json.dumps(bad_reconstruction[:5]))
    add_check(checks, "injected text differs from gold", 0, len(bad_gold),
              detail=json.dumps(bad_gold[:5]))
    n_perturbed = sum(total_ops.values())
    aggregate_ratio = ratio(n_perturbed, total_chars)
    mix = {
        key: round(total_ops.get(key, 0) / n_perturbed, 4) if n_perturbed else None
        for key in ("sub", "del", "ins")
    }
    result["injection"] = {
        "n_items": n_items_with_ops,
        "n_chars": total_chars,
        "n_perturbed": n_perturbed,
        "ops": dict(total_ops),
        "ratio": aggregate_ratio,
        "mix": mix,
    }
    checks.extend(injection_checks)
    injected_archive = load_jsonl(manifest_dir / "d_injected_10.jsonl")
    if injected_archive:
        archive_problems: list[str] = []
        seen: set[str] = set()
        for row in injected_archive:
            utt = str(row.get("utt_id") or "")
            seen.add(utt)
            gold_row = b_by_utt.get(utt) or {}
            rebuilt, problems = apply_injection_ops(
                list(str(gold_row.get("chars") or "")), list(row.get("ops") or [])
            )
            if problems or "".join(rebuilt) != str(row.get("injected_text") or ""):
                archive_problems.append(utt)
        add_check(checks, "d_injected_10.jsonl rows == 1,758", EXPECTED_UTTS,
                  len(injected_archive))
        add_check(checks, "d_injected_10.jsonl ops reconstruct injected_text", 0,
                  len(archive_problems), detail=json.dumps(archive_problems[:5]))
        add_check(checks, "d_injected_10.jsonl covers every utterance", 0,
                  len(set(b_by_utt) - seen))
    else:
        add_check(
            checks,
            "d_injected_10.jsonl archived (design A.2)",
            True,
            False,
            detail="file missing (per-item injection_ops are still in d_items.jsonl)",
        )
    prep_summary = json_load(manifest_dir / "d_prep_summary.json", {}) or {}
    prep_injection = prep_summary.get("injection") or {}
    add_check(checks, "d_prep_summary n_items", EXPECTED_UTTS * 3, prep_summary.get("n_items"))
    add_check(checks, "d_prep_summary per_source",
              {source: EXPECTED_UTTS for source in SOURCES},
              prep_summary.get("per_source"))
    add_check(checks, "d_prep_summary n_asr_missing", 0, prep_summary.get("n_asr_missing"))
    add_check(checks, "d_prep_summary injection n_items", n_items_with_ops,
              prep_injection.get("n_items", n_items_with_ops)
              if prep_injection else None,
              ok=bool(prep_injection))
    if prep_injection:
        add_check(
            checks,
            "d_prep_summary injection ops match d_items",
            {key: int(prep_injection.get(key, 0) or 0) for key in ("sub", "del", "ins")},
            {key: total_ops.get(key, 0) for key in ("sub", "del", "ins")},
        )
        add_check(
            checks,
            "d_prep_summary injection ratio ≈ 10%",
            True,
            (aggregate_ratio is not None and 0.09 <= aggregate_ratio <= 0.13),
            detail=f"ratio={aggregate_ratio:.4f}",
        )
    # --- audio (optional) ---------------------------------------------------
    if check_audio:
        audio_problems: list[str] = []
        for utt in list(b_by_utt)[:EXPECTED_UTTS]:
            row = b_by_utt[utt]
            path = repo_root / str(row.get("audio_path") or "")
            probe = _probe_audio(path)
            if not probe.get("ok"):
                audio_problems.append(f"{utt}:missing")
            elif (probe["samplerate"], probe["channels"], probe["subtype"]) != (
                16000, 1, "PCM_16",
            ):
                audio_problems.append(f"{utt}:format")
        add_check(checks, "OMPAL learner audio (16 kHz/PCM_16/mono)", 0,
                  len(audio_problems), detail=json.dumps(audio_problems[:5]))
    result["n_checks"] = len(checks)
    result["n_failed"] = sum(1 for check in checks if not check["ok"])
    return result


def _round_or_none(value: Any, digits: int = 8) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _close(left: Any, right: Any, tol: float = 1e-6) -> bool:
    try:
        return left is not None and right is not None and abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def recompute_asr_quality(
    b_by_utt: Mapping[str, Mapping[str, Any]],
    asr_by_utt: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Independent ASR CER / S-D-I / normalization-rate / per-learner CER."""
    n_gold = n_hyp = edit_sum = 0
    sdi = {"sub": 0, "del": 0, "ins": 0}
    norm_error = {"error_chars": 0, "same": 0}
    per_learner: dict[str, dict[str, int]] = {}
    per_item_cer: list[float] = []
    missing = 0
    for utt, gold_row in b_by_utt.items():
        asr_row = asr_by_utt.get(utt)
        if asr_row is None:
            missing += 1
            continue
        gold = text_chars(str(gold_row.get("chars") or ""))
        hyp = text_chars(str(asr_row.get("raw_text") or ""))
        labels = list(gold_row.get("labels") or [])
        distance, counts, ops = edit_distance_ops(gold, hyp)
        n_gold += len(gold)
        n_hyp += len(hyp)
        edit_sum += distance
        for key in sdi:
            sdi[key] += counts[key]
        if gold:
            per_item_cer.append(distance / len(gold))
        speaker = str(gold_row.get("speaker") or "unknown")
        bucket = per_learner.setdefault(speaker, {"edit": 0, "n": 0})
        bucket["edit"] += distance
        bucket["n"] += len(gold)
        aligned = {g: h for kind, g, h in ops if g is not None and h is not None}
        for index, label in enumerate(labels):
            if index >= len(gold):
                break
            error = any(int(label.get(dim, 1)) == 0 for dim in D_DIMENSIONS)
            if not error:
                continue
            norm_error["error_chars"] += 1
            hyp_index = aligned.get(index)
            if hyp_index is not None and hyp[hyp_index] == gold[index]:
                norm_error["same"] += 1
    learner_values = sorted(
        bucket["edit"] / bucket["n"] for bucket in per_learner.values() if bucket["n"]
    )
    n_exact = sum(1 for value in per_item_cer if value == 0.0)
    return {
        "n_missing": missing,
        "cer_micro": ratio(edit_sum, n_gold),
        "sdi": sdi,
        "n_gold_chars": n_gold,
        "n_hyp_chars": n_hyp,
        "normalization_rate": ratio(norm_error["same"], norm_error["error_chars"]),
        "normalization_n": dict(norm_error),
        "per_learner_cer": {
            "n_learners": len(learner_values),
            "median": median(learner_values),
            "p90": percentile(learner_values, 0.9),
            "max": learner_values[-1] if learner_values else None,
        },
        "cer_mean_utterance": mean(per_item_cer),
        "cer_median_utterance": median(per_item_cer),
        "cer_p90_utterance": percentile(sorted(per_item_cer), 0.9),
        "sentence_exact_rate": ratio(n_exact, len(per_item_cer)),
    }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def verify_reproducibility(
    repo_root: Path, run_dir: Path, manifest_dir: Path
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_manifest = json_load(run_dir / "run_manifest.json", {}) or {}
    d_items_path = manifest_dir / "d_items.jsonl"
    d_provenance = run_manifest.get("d_provenance") or {}
    result: dict[str, Any] = {"checks": checks}
    add_check(checks, "run_manifest.json present", True, bool(run_manifest))
    required = (
        "run_id", "created_at", "model", "provider", "prompt_version",
        "prompt_fingerprints", "config_sha256", "seed", "pricing", "budget_cny",
        "n_planned", "git_commit", "items_manifests",
    )
    missing = [key for key in required if run_manifest.get(key) in (None, "", {}, [])]
    add_check(checks, "run_manifest required fields non-empty", [], missing)
    try:
        from pinyin_eval.sentence.prompts import (
            D_PROMPT_VERSION,
            TEMPLATE_D,
            prompt_fingerprints,
            prompt_sha256,
        )

        expected_version = D_PROMPT_VERSION
        fingerprints = prompt_fingerprints()
    except Exception:  # pragma: no cover - defensive
        expected_version, TEMPLATE_D = "sentence-d-v1", ""
        fingerprints, prompt_sha256 = {}, None  # type: ignore[assignment]
    add_check(checks, "prompt_version == sentence-d-v1", expected_version,
              run_manifest.get("prompt_version"))
    add_check(checks, "D prompt fingerprint matches code", fingerprints.get("D"),
              (run_manifest.get("prompt_fingerprints") or {}).get("D"))
    add_check(checks, "model recorded", "qwen3.8-omni-flash", run_manifest.get("model"))
    pricing = run_manifest.get("pricing") or {}
    for key, value in (
        ("input_per_mtok", PRICING["input_per_mtok"]),
        ("cache_hit_input_per_mtok", PRICING["cache_hit_input_per_mtok"]),
        ("output_per_mtok", PRICING["output_per_mtok"]),
    ):
        add_check(checks, f"pricing.{key}", value, pricing.get(key))
    add_check(checks, "D budget (run_manifest) ≤ ¥10", 10.0,
              run_manifest.get("budget_cny"),
              ok=run_manifest.get("budget_cny") is not None
              and float(run_manifest["budget_cny"]) <= 10.0)
    manifests = run_manifest.get("items_manifests") or {}
    add_check(checks, "d_items sha256 recorded",
              sha256_file(d_items_path), (manifests.get("D") or {}).get("sha256"))
    if isinstance(d_provenance, Mapping) and d_provenance:
        asr_prov = d_provenance.get("asr") or {}
        prep_prov = d_provenance.get("d_prep_summary") or {}
        injected_prov = d_provenance.get("injected_10") or {}
        add_check(
            checks,
            "d_provenance: d_prep_summary sha256",
            sha256_file(manifest_dir / "d_prep_summary.json"),
            prep_prov.get("sha256"),
        )
        add_check(
            checks,
            "d_provenance: d_injected_10 sha256",
            sha256_file(manifest_dir / "d_injected_10.jsonl"),
            injected_prov.get("sha256"),
        )
        add_check(
            checks,
            "d_provenance: ASR outputs sha256",
            sha256_file(manifest_dir.parent / "d_asr" / "asr_outputs.jsonl"),
            asr_prov.get("outputs_sha256"),
        )
        add_check(
            checks,
            "d_provenance: ASR summary sha256",
            sha256_file(manifest_dir.parent / "d_asr" / "d_asr_summary.json"),
            asr_prov.get("summary_sha256"),
        )
        add_check(
            checks,
            "d_provenance: ASR model/version/decode recorded",
            True,
            bool(asr_prov.get("model_id")),
        )
        injection_prov = d_provenance.get("injection") or {}
        add_check(
            checks,
            "d_provenance: injection seed/ratio/mix recorded",
            True,
            injection_prov.get("seed") is not None
            and injection_prov.get("ratio") is not None
            and bool(injection_prov.get("mix")),
        )
        add_check(
            checks,
            "run seed == sentence.d.seed (20261103)",
            int(injection_prov.get("seed") or 0),
            int(run_manifest.get("seed") or 0),
        )
    else:
        add_check(
            checks,
            "d_provenance recorded in run_manifest",
            True,
            False,
            detail="missing d_provenance block (ASR/injection provenance)",
        )
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        head, status = None, ""
    add_check(checks, "git_commit == current HEAD", head, run_manifest.get("git_commit"))
    result["git"] = {
        "head": head,
        "dirty_files": [line for line in status.splitlines() if line.strip()][:20],
        "n_dirty": len([line for line in status.splitlines() if line.strip()]),
    }
    planned = 3 * EXPECTED_UTTS
    add_check(checks, "n_planned == 3 × 1,758", planned, run_manifest.get("n_planned"))
    conditions = run_manifest.get("conditions") or {}
    result["conditions"] = {name: conditions.get(name) for name in conditions}
    if prompt_sha256 is not None:
        result["prompt_sha256_available"] = bool(TEMPLATE_D)
    result["n_checks"] = len(checks)
    result["n_failed"] = sum(1 for check in checks if not check["ok"])
    return result


# ---------------------------------------------------------------------------
# Run verification
# ---------------------------------------------------------------------------


def verify_run(run_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    d_items = load_jsonl(manifest_dir / "d_items.jsonl")
    if not d_items:
        return {"error": "d_items.jsonl missing", "checks": checks, "n_failed": 1}
    b_rows = load_jsonl(manifest_dir / "ompal_b_items.jsonl")
    b_by_utt = {str((row.get("meta") or {}).get("utt_id")): row for row in b_rows}
    item_by_id = {str(row.get("item_id")): row for row in d_items}
    records = load_jsonl(run_dir / "generations.jsonl")
    run_manifest = json_load(run_dir / "run_manifest.json", {}) or {}
    result: dict[str, Any] = {}
    expected_keys = {f"D|C1|{row.get('item_id')}" for row in d_items}
    result["records"] = {
        "n_rows": len(records),
        "n_unique": len({str(row.get("request_id")) for row in records}),
        "n_expected": len(expected_keys),
    }
    final: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    for record in records:
        key = str(record.get("request_id"))
        if key in final:
            duplicates[key] += 1
        final[key] = record
    present = set(final)
    missing = sorted(expected_keys - present)
    extra = sorted(present - expected_keys)
    result["records"].update(
        {
            "n_present": len(present),
            "n_missing": len(missing),
            "missing_examples": missing[:10],
            "n_extra": len(extra),
            "extra_examples": extra[:10],
            "duplicate_request_keys": len(duplicates),
        }
    )
    add_check(checks, "generation records for every planned request",
              len(expected_keys), len(present))
    add_check(checks, "no unexpected request ids", 0, len(extra))

    # --- envelope + prompt fingerprint -------------------------------------
    envelope = {
        "missing_usage": [], "missing_finish_reason": [], "missing_timestamp": [],
        "model_mismatch": [], "prompt_version_mismatch": [], "prompt_sha_mismatch": [],
        "condition_mismatch": [], "max_tokens_mismatch": [], "thinking_mismatch": [],
        "audio_path_mismatch": [], "error_records": [],
    }
    try:
        from pinyin_eval.sentence.prompts import (
            TEMPLATE_D,
            prompt_sha256,
        )

        can_check_prompt = True
    except Exception:  # pragma: no cover - defensive
        can_check_prompt = False
        TEMPLATE_D = ""

    def expected_prompt_sha(item: Mapping[str, Any]) -> str | None:
        if not can_check_prompt:
            return None
        # runner passes the task-specific prompt version into the fingerprint
        from pinyin_eval.sentence.prompts import D_PROMPT_VERSION

        return prompt_sha256("", TEMPLATE_D.format(text=item.get("text") or ""),
                             D_PROMPT_VERSION)

    for key, record in final.items():
        item = item_by_id.get(str(record.get("item_id")))
        if record.get("error"):
            envelope["error_records"].append(key)
        else:
            if not record.get("usage"):
                envelope["missing_usage"].append(key)
            if not record.get("finish_reason"):
                envelope["missing_finish_reason"].append(key)
            if not record.get("timestamp"):
                envelope["missing_timestamp"].append(key)
        if str(record.get("model")) != str(run_manifest.get("model") or "qwen3.8-omni-flash"):
            envelope["model_mismatch"].append(key)
        if str(record.get("prompt_version")) != "sentence-d-v1":
            envelope["prompt_version_mismatch"].append(key)
        if str(record.get("condition")) != "C1":
            envelope["condition_mismatch"].append(key)
        if int(record.get("max_tokens") or 0) != 1536:
            envelope["max_tokens_mismatch"].append(key)
        if bool(record.get("thinking")):
            envelope["thinking_mismatch"].append(key)
        if item is not None:
            if str(record.get("audio_path") or "") != str(item.get("audio_path") or ""):
                envelope["audio_path_mismatch"].append(key)
            expected_sha = expected_prompt_sha(item)
            if expected_sha is not None and str(record.get("prompt_sha256")) != expected_sha:
                envelope["prompt_sha_mismatch"].append(key)
    result["envelope"] = {
        key: (len(value) if isinstance(value, list) else value)
        for key, value in envelope.items()
    }
    result["envelope"]["examples"] = {
        key: value[:3] for key, value in envelope.items() if isinstance(value, list) and value
    }
    add_check(checks, "usage/finish_reason/timestamp recorded", [],
              {key: result["envelope"][key] for key in (
                  "missing_usage", "missing_finish_reason", "missing_timestamp")},
              ok=not any(result["envelope"][key] for key in (
                  "missing_usage", "missing_finish_reason", "missing_timestamp")))
    add_check(checks, "C1-only + D prompt version + max_tokens + thinking", [],
              {key: result["envelope"][key] for key in (
                  "condition_mismatch", "prompt_version_mismatch",
                  "max_tokens_mismatch", "thinking_mismatch")},
              ok=not any(result["envelope"][key] for key in (
                  "condition_mismatch", "prompt_version_mismatch",
                  "max_tokens_mismatch", "thinking_mismatch")))
    add_check(checks, "per-record prompt fingerprint", 0,
              result["envelope"]["prompt_sha_mismatch"])
    add_check(checks, "model id recorded per call", 0,
              result["envelope"]["model_mismatch"])

    # --- parse + score ------------------------------------------------------
    scored: dict[str, list[dict[str, Any]]] = {source: [] for source in SOURCES}
    status_counts: dict[str, Counter[str]] = {source: Counter() for source in SOURCES}
    wrapper_rows: Counter[str] = Counter()
    failure_examples: list[dict[str, Any]] = []
    for item in d_items:
        item_id = str(item.get("item_id"))
        meta = item.get("meta") or {}
        source = str(meta.get("text_source") or "")
        utt = str(meta.get("utt_id") or "")
        record = final.get(f"D|C1|{item_id}")
        if record is None:
            parsed = {"parse_status": "missing", "alignment": [], "scores": {}}
        else:
            parsed = parse_d_reply(str(record.get("response_text") or ""))
            if parsed.get("wrapper_unwrapped"):
                wrapper_rows[source] += 1
        status = str(parsed.get("parse_status") or "parse_error")
        status_counts[source][status] += 1
        if status != "ok" and len(failure_examples) < 30:
            failure_examples.append(
                {
                    "request": f"D|C1|{item_id}",
                    "source": source,
                    "status": status,
                    "detail": str(parsed.get("detail") or "")[:160],
                    "finish_reason": (record or {}).get("finish_reason"),
                }
            )
        gold_row = b_by_utt.get(utt) or {}
        gold = {
            "per_char": [],
            "stats": {},
            "input_chars": text_chars(str(item.get("text") or "")),
        }
        per_char, stats = gold_alignment(
            str(item.get("text") or ""),
            str(gold_row.get("chars") or ""),
            list(gold_row.get("labels") or []),
        )
        gold = {"per_char": per_char, "stats": stats, "input_chars": text_chars(str(item.get("text") or ""))}
        row = score_d_item(item, parsed, gold)
        scored[source].append(row)
    result["parse"] = {
        source: {
            "status_counts": dict(status_counts[source]),
            "n_items": sum(status_counts[source].values()),
            "n_parse_ok": status_counts[source].get("ok", 0),
            "parse_success_rate": ratio(
                status_counts[source].get("ok", 0), sum(status_counts[source].values())
            ),
            "wrapper_unwrapped": wrapper_rows.get(source, 0),
        }
        for source in SOURCES
    }
    result["parse"]["failure_examples"] = failure_examples
    for source in SOURCES:
        if not status_counts[source]:
            continue
        add_check(checks, f"{source} parse success ≥95%", True,
                  result["parse"][source]["parse_success_rate"],
                  ok=(result["parse"][source]["parse_success_rate"] or 0.0)
                  >= ACCEPT_PARSE_MIN,
                  detail=f"rate={result['parse'][source]['parse_success_rate']}")

    metrics: dict[str, Any] = {}
    for source in SOURCES:
        if not scored[source]:
            continue
        metrics[source] = aggregate_d(scored[source])
    result["metrics"] = metrics
    result["comparisons"] = compare_sources(scored) if metrics else {"comparisons": {}}
    # headline deltas from d_comparisons.csv are compared later

    # --- cost ---------------------------------------------------------------
    per_condition: dict[str, dict[str, Any]] = {}
    cost_floats = defaultdict(float)
    for record in records:
        task = str(record.get("task") or "")
        condition = str(record.get("condition") or "")
        if task != "D":
            continue
        usage = usage_of(record)
        key = f"{task}|{condition}"
        entry = per_condition.setdefault(
            key,
            {
                "n_calls": 0, "n_errors": 0, "n_retries": 0,
                "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
                "cost_cny": 0.0,
                "finish_reasons": Counter(),
            },
        )
        cost = record_cost(usage)
        entry["n_calls"] += 1
        entry["n_errors"] += 0 if record.get("error") is None else 1
        entry["n_retries"] += int(record.get("retries") or 0)
        entry["prompt_tokens"] += usage["prompt_tokens"]
        entry["completion_tokens"] += usage["completion_tokens"]
        entry["reasoning_tokens"] += usage["reasoning_tokens"]
        entry["cost_cny"] += cost
        entry["finish_reasons"][str(record.get("finish_reason"))] += 1
        for key_name, value in (
            ("prompt_tokens", usage["prompt_tokens"]),
            ("completion_tokens", usage["completion_tokens"]),
            ("reasoning_tokens", usage["reasoning_tokens"]),
            ("cached_tokens", usage["cached_tokens"]),
            ("audio_tokens", usage["audio_tokens"]),
        ):
            cost_floats[key_name] += value
        cost_floats["input_cost_cny"] += (
            (usage["prompt_tokens"] - usage["cached_tokens"])
            * PRICING["input_per_mtok"]
            + usage["cached_tokens"] * PRICING["cache_hit_input_per_mtok"]
        ) / 1_000_000.0
        cost_floats["output_cost_cny"] += (
            usage["completion_tokens"] * PRICING["output_per_mtok"] / 1_000_000.0
        )
        cost_floats["cost_cny"] += cost
    for entry in per_condition.values():
        entry["cost_cny"] = round(entry["cost_cny"], 6)
        entry["finish_reasons"] = dict(entry["finish_reasons"])
    total_dict = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in cost_floats.items()
    }
    total_dict["n_calls"] = sum(entry["n_calls"] for entry in per_condition.values())
    total_dict["n_errors"] = sum(entry["n_errors"] for entry in per_condition.values())
    total_dict["n_retries"] = sum(entry["n_retries"] for entry in per_condition.values())
    result["cost"] = {
        "per_task_condition": dict(sorted(per_condition.items())),
        "totals": total_dict,
        "pricing_used": PRICING,
        "cost_summary": json_load(run_dir / "cost_summary.json"),
    }
    add_check(checks, "D total cost ≤ ¥10", True, total_dict.get("cost_cny"),
              ok=(total_dict.get("cost_cny") is not None
                  and total_dict["cost_cny"] <= ACCEPT_COST_MAX))
    cost_summary = result["cost"]["cost_summary"]
    if isinstance(cost_summary, Mapping):
        reported_total = (cost_summary.get("totals") or {}).get("cost_cny")
        add_check(checks, "usage-based cost == cost_summary.json",
                  _round_or_none(reported_total),
                  _round_or_none(total_dict.get("cost_cny")),
                  ok=_close(reported_total, total_dict.get("cost_cny"), 0.01))
        add_check(checks, "usage-based n_calls == cost_summary.json",
                  (cost_summary.get("totals") or {}).get("n_calls"),
                  total_dict.get("n_calls"))
    finish_reasons = Counter(
        str(record.get("finish_reason")) for record in final.values()
    )
    result["finish_reasons"] = dict(finish_reasons)
    result["retries"] = {
        "n_records_with_retries": sum(
            1 for record in final.values() if int(record.get("retries") or 0) > 0
        ),
        "retries_total": total_dict.get("n_retries"),
        "retries_max": max(
            (int(record.get("retries") or 0) for record in final.values()), default=0
        ),
    }
    latencies = [
        float(record.get("latency_s") or 0.0) for record in final.values()
        if record.get("error") is None
    ]
    result["latency_s"] = {
        "n": len(latencies),
        "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3)
        if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }
    result["checks"] = checks
    result["n_checks"] = len(checks)
    result["n_failed"] = sum(1 for check in checks if not check["ok"])
    return result


# ---------------------------------------------------------------------------
# Acceptance (design A.8)
# ---------------------------------------------------------------------------


def acceptance(
    run: Mapping[str, Any],
    data: Mapping[str, Any] | None,
    *,
    asr_completeness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parse_success: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    for source in SOURCES:
        info = (run.get("parse") or {}).get(source) or {}
        rate = info.get("parse_success_rate")
        parse_success[source] = {
            "rate": rate,
            "n_items": info.get("n_items"),
            "n_parse_ok": info.get("n_parse_ok"),
            "threshold": ACCEPT_PARSE_MIN,
            "passed": rate is not None and rate >= ACCEPT_PARSE_MIN,
        }
        metrics = (run.get("metrics") or {}).get(source) or {}
        value = metrics.get("alignment_coverage")
        coverage[source] = {
            "value": value,
            "threshold": ACCEPT_COVERAGE_MIN,
            "passed": value is not None and value >= ACCEPT_COVERAGE_MIN,
        }
    total_cost = ((run.get("cost") or {}).get("totals") or {}).get("cost_cny")
    cost_block = {
        "value": total_cost,
        "threshold": ACCEPT_COST_MAX,
        "passed": total_cost is not None and float(total_cost) <= ACCEPT_COST_MAX,
    }
    completeness = dict(asr_completeness or {})
    asr_block = {
        "archived": bool(completeness.get("archived")),
        "n_archived": completeness.get("n_archived"),
        "n_expected": EXPECTED_UTTS,
        "passed": (
            bool(completeness.get("archived"))
            and completeness.get("n_archived") == EXPECTED_UTTS
            and not completeness.get("missing")
        ),
    }
    passed = (
        all(entry["passed"] for entry in parse_success.values())
        and all(entry["passed"] for entry in coverage.values())
        and cost_block["passed"]
        and asr_block["passed"]
    )
    return {
        "parse_success": parse_success,
        "alignment_coverage": coverage,
        "cost_cny": cost_block,
        "asr_archived": asr_block,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Comparison against reported artifacts
# ---------------------------------------------------------------------------


def _parse_csv_number(value: Any) -> float | None:
    if value is None or str(value).strip() in ("", "None", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dig(data: Any, path: str) -> Any:
    node = data
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _flatten_numeric(data: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            out.update(_flatten_numeric(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            out.update(_flatten_numeric(value, f"{prefix}[{index}]"))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        out[prefix] = data
    return out


def compare_reported(
    run_dir: Path, run: Mapping[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "d_metrics_csv": {"available": False},
        "d_acceptance_json": {"available": False},
        "d_report_summary_json": {"available": False},
        "d_comparisons_csv": {"available": False},
        "d_prep_summary_json": {"available": False},
        "d_report_md": {"available": False},
        "discrepancies": [],
    }
    metrics = run.get("metrics") or {}
    lookup: dict[tuple[str, str], Any] = {}
    for source, block in metrics.items():
        lookup[(source, "alignment_f1")] = (block.get("alignment") or {}).get("f1")
        lookup[(source, "text_error_f1")] = (block.get("text_error") or {}).get("f1")
        for dimension in D_DIMENSIONS:
            lookup[(source, f"{dimension}_macro_f1")] = (
                (block.get("localization") or {}).get(dimension) or {}
            ).get("macro_f1")
        scoring = block.get("scoring") or {}
        for dimension in (*D_SCORE_DIMENSIONS, "mean_dimension"):
            stats = scoring.get(dimension) or {}
            for metric in ("pearson", "mae", "rmse", "spearman"):
                lookup[(source, f"{dimension}_{metric}")] = stats.get(metric)
        lookup[(source, "alignment_coverage")] = block.get("alignment_coverage")
        lookup[(source, "parse_success_rate")] = block.get("parse_success_rate")
        lookup[(source, "n_parse_ok")] = block.get("n_parse_ok")
        lookup[(source, "n_items")] = block.get("n_items")

    def check_value(label: str, reported: Any, independent: Any,
                    *, tolerance: float = FLOAT_TOL, source: str = "d_metrics.csv") -> None:
        rep = _parse_csv_number(reported)
        ind = _parse_csv_number(independent)
        if rep is None and ind is None:
            return
        if rep is None or ind is None:
            comparison["discrepancies"].append(
                {"source": source, "key": label, "reported": rep, "independent": ind,
                 "delta_pp": None, "kind": "null_mismatch"}
            )
            return
        delta = ind - rep
        if abs(delta) > tolerance:
            comparison["discrepancies"].append(
                {"source": source, "key": label, "reported": rep, "independent": ind,
                 "delta_pp": round(delta * 100, 4), "kind": "value"}
            )

    metrics_path = run_dir / "d_metrics.csv"
    if metrics_path.exists():
        mismatches: list[dict[str, Any]] = []
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        compared = 0
        for row in rows:
            source = str(row.get("source") or "")
            metric = str(row.get("metric") or "")
            if (source, metric) not in lookup:
                continue
            compared += 1
            before = len(comparison["discrepancies"])
            check_value(f"{source}|{metric}", row.get("estimate"),
                        lookup[(source, metric)])
            mismatches.extend(comparison["discrepancies"][before:])
        comparison["d_metrics_csv"] = {
            "available": True,
            "n_rows": len(rows),
            "n_compared": compared,
            "n_mismatch": len(mismatches),
            "mismatches": mismatches[:40],
        }
    # acceptance.json
    acceptance_path = run_dir / "d_acceptance.json"
    reported_acceptance = json_load(acceptance_path)
    if isinstance(reported_acceptance, Mapping):
        before = len(comparison["discrepancies"])
        for source in SOURCES:
            rep = ((reported_acceptance.get("parse_success") or {}).get(source) or {})
            ind = ((run.get("acceptance") or {}).get("parse_success") or {}).get(source) or {}
            if rep.get("rate") is not None and ind.get("rate") is not None:
                if abs(float(rep["rate"]) - float(ind["rate"])) > 0.0005:
                    comparison["discrepancies"].append(
                        {"source": "d_acceptance.json",
                         "key": f"parse_success.{source}", "reported": rep["rate"],
                         "independent": ind["rate"],
                         "delta_pp": round((ind["rate"] - rep["rate"]) * 100, 4)}
                    )
            rep_cov = ((reported_acceptance.get("alignment_coverage") or {}).get(source) or {})
            ind_cov = ((run.get("acceptance") or {}).get("alignment_coverage") or {}).get(source) or {}
            if rep_cov.get("value") is not None and ind_cov.get("value") is not None:
                if abs(float(rep_cov["value"]) - float(ind_cov["value"])) > 0.0005:
                    comparison["discrepancies"].append(
                        {"source": "d_acceptance.json",
                         "key": f"alignment_coverage.{source}",
                         "reported": rep_cov["value"], "independent": ind_cov["value"],
                         "delta_pp": round((ind_cov["value"] - rep_cov["value"]) * 100, 4)}
                    )
        rep_cost = (reported_acceptance.get("cost_cny") or {}).get("value")
        ind_cost = ((run.get("acceptance") or {}).get("cost_cny") or {}).get("value")
        if rep_cost is not None and ind_cost is not None and abs(rep_cost - ind_cost) > 0.01:
            comparison["discrepancies"].append(
                {"source": "d_acceptance.json", "key": "cost_cny",
                 "reported": rep_cost, "independent": ind_cost}
            )
        if bool(reported_acceptance.get("passed")) != bool(
            (run.get("acceptance") or {}).get("passed")
        ):
            comparison["discrepancies"].append(
                {"source": "d_acceptance.json", "key": "passed",
                 "reported": reported_acceptance.get("passed"),
                 "independent": (run.get("acceptance") or {}).get("passed")}
            )
        if bool(reported_acceptance.get("asr_archived")) != bool(
            ((run.get("acceptance") or {}).get("asr_archived") or {}).get("archived")
        ):
            comparison["discrepancies"].append(
                {"source": "d_acceptance.json", "key": "asr_archived",
                 "reported": reported_acceptance.get("asr_archived"),
                 "independent": (run.get("acceptance") or {}).get("asr_archived")}
            )
        comparison["d_acceptance_json"] = {
            "available": True,
            "n_mismatch": len(comparison["discrepancies"]) - before,
        }
    # comparisons.csv
    comparisons_path = run_dir / "d_comparisons.csv"
    if comparisons_path.exists() and run.get("comparisons"):
        reported = {
            (row.get("source"), row.get("metric")): row
            for row in csv.DictReader(comparisons_path.open(encoding="utf-8"))
        }
        my_comparisons = (run.get("comparisons") or {}).get("comparisons") or {}
        before = len(comparison["discrepancies"])
        compared = 0
        for source, entry in my_comparisons.items():
            for metric, diff in (entry.get("difference") or {}).items():
                if (source, metric) not in reported:
                    continue
                compared += 1
                check_value(f"{source}|{metric}", reported[(source, metric)].get("estimate"),
                            diff.get("estimate"), source="d_comparisons.csv")
        comparison["d_comparisons_csv"] = {
            "available": True,
            "n_compared": compared,
            "n_mismatch": len(comparison["discrepancies"]) - before,
        }
    # report_summary.json (full payload)
    summary_path = run_dir / "d_report_summary.json"
    summary = json_load(summary_path)
    if isinstance(summary, Mapping):
        before = len(comparison["discrepancies"])
        expected: dict[str, Any] = {
            "comparisons.baseline": (run.get("comparisons") or {}).get("baseline"),
        }
        for source in SOURCES:
            block = metrics.get(source) or {}
            for path, value in (
                ("alignment.f1", (block.get("alignment") or {}).get("f1")),
                ("text_error.f1", (block.get("text_error") or {}).get("f1")),
                ("parse_success_rate", block.get("parse_success_rate")),
                ("n_parse_ok", block.get("n_parse_ok")),
                ("n_items", block.get("n_items")),
                ("alignment_coverage", block.get("alignment_coverage")),
                ("zero_initial_consonant.macro_f1",
                 (block.get("zero_initial_consonant") or {}).get("macro_f1")),
            ):
                expected[f"metrics.{source}.{path}"] = value
            for dimension in D_DIMENSIONS:
                expected[f"metrics.{source}.localization.{dimension}.macro_f1"] = (
                    (block.get("localization") or {}).get(dimension) or {}
                ).get("macro_f1")
            for dimension in (*D_SCORE_DIMENSIONS, "mean_dimension"):
                expected[f"metrics.{source}.scoring.{dimension}.pearson"] = (
                    (block.get("scoring") or {}).get(dimension) or {}
                ).get("pearson")
        expected["acceptance.passed"] = (run.get("acceptance") or {}).get("passed")
        expected["acceptance.cost_cny.value"] = (
            (run.get("acceptance") or {}).get("cost_cny") or {}
        ).get("value")
        expected["cost.totals.cost_cny"] = ((run.get("cost") or {}).get("totals") or {}).get(
            "cost_cny"
        )
        for path, independent in expected.items():
            reported_value = _dig(summary, path)
            if isinstance(independent, bool):
                if bool(reported_value) != independent:
                    comparison["discrepancies"].append(
                        {"source": "d_report_summary.json", "key": path,
                         "reported": reported_value, "independent": independent}
                    )
                continue
            check_value(path, reported_value, independent,
                        source="d_report_summary.json")
        comparison["d_report_summary_json"] = {
            "available": True,
            "n_compared": len(expected),
            "n_mismatch": len(comparison["discrepancies"]) - before,
        }
    # report.md tables (headline cells)
    report_path = run_dir / "d_report.md"
    if report_path.exists():
        before = len(comparison["discrepancies"])
        md_block = check_report_md(
            report_path.read_text(encoding="utf-8"), metrics, run, comparison
        )
        comparison["discrepancies"].extend(
            {"source": "d_report.md", **issue} for issue in md_block.get("issues") or []
        )
        md_block["n_mismatch"] = len(comparison["discrepancies"]) - before
        comparison["d_report_md"] = md_block
    comparison["n_discrepancies"] = len(comparison["discrepancies"])
    return comparison


def _split_row(line: str) -> list[str]:
    cells: list[str] = []
    buffer = ""
    index = 0
    while index < len(line):
        char = line[index]
        if char == "\\" and index + 1 < len(line) and line[index + 1] == "|":
            buffer += "|"
            index += 2
            continue
        if char == "|":
            cells.append(buffer)
            buffer = ""
            index += 1
            continue
        buffer += char
        index += 1
    cells.append(buffer)
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def _tables(lines: Sequence[str]) -> list[tuple[list[str], list[list[str]]]]:
    tables: list[tuple[list[str], list[list[str]]]] = []
    index = 0
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("|") and set(
            lines[index + 1].replace("|", "").replace("-", "").replace(":", "").strip()
        ) <= {" "}:
            header = _split_row(line)
            rows: list[list[str]] = []
            cursor = index + 2
            while cursor < len(lines) and lines[cursor].startswith("|"):
                rows.append(_split_row(lines[cursor]))
                cursor += 1
            tables.append((header, rows))
            index = cursor
        else:
            index += 1
    return tables


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def check_report_md(
    text: str,
    metrics: Mapping[str, Any],
    run: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare headline report.md table cells against independent numbers."""
    lines = text.splitlines()
    tables = _tables(lines)
    issues: list[dict[str, Any]] = []
    checked = 0

    def find(pred) -> tuple[list[str], list[list[str]]] | tuple[None, None]:
        for header, rows in tables:
            if pred(header):
                return header, rows
        return None, None

    def check(cell: Any, expected: Any, label: str) -> None:
        nonlocal checked
        if expected is None:
            return
        checked += 1
        if str(cell).strip() != str(expected).strip():
            issues.append({"key": label, "reported": cell, "independent": expected})

    header, rows = find(lambda h: h[:2] == ["文本条件", "items"])
    if rows:
        for row in rows:
            source = row[0]
            block = metrics.get(source) or {}
            check(row[1], block.get("n_items"), f"{source}|n_items")
            check(row[2], block.get("n_parse_ok"), f"{source}|n_parse_ok")
            parse_fail = block.get("parse_failure_rate")
            if parse_fail is not None:
                check(row[3], f"{100 * float(parse_fail):.2f}%", f"{source}|parse_fail")
            coverage = block.get("alignment_coverage")
            if coverage is not None:
                check(row[4], f"{100 * float(coverage):.2f}%", f"{source}|coverage")
    header, rows = find(lambda h: h[:2] == ["文本条件", "对齐 precision"])
    if rows:
        for row in rows:
            source = row[0]
            alignment = ((metrics.get(source) or {}).get("alignment") or {})
            for column, key in ((1, "precision"), (2, "recall"), (3, "f1")):
                check(row[column], _fmt(alignment.get(key)), f"{source}|align_{key}")
    header, rows = find(lambda h: h[:2] == ["文本条件", "tone macro-F1"])
    if rows:
        for row in rows:
            source = row[0]
            localization = (metrics.get(source) or {}).get("localization") or {}
            for offset, dimension in enumerate(D_DIMENSIONS):
                stats = localization.get(dimension) or {}
                base = 1 + offset * 4
                check(row[base], _fmt(stats.get("macro_f1")),
                      f"{source}|{dimension}_macro_f1")
                check(row[base + 1], _fmt(stats.get("f1_error")),
                      f"{source}|{dimension}_f1_error")
                check(row[base + 2], _fmt(stats.get("fpr")), f"{source}|{dimension}_fpr")
                check(row[base + 3], _fmt(stats.get("fnr")), f"{source}|{dimension}_fnr")
    header, rows = find(lambda h: h[:2] == ["文本条件", "accuracy r"])
    if rows:
        for row in rows:
            source = row[0]
            scoring = (metrics.get(source) or {}).get("scoring") or {}
            for offset, dimension in enumerate((*D_SCORE_DIMENSIONS, "mean_dimension")):
                stats = scoring.get(dimension) or {}
                base = 1 + offset * 5
                check(row[base], _fmt(stats.get("pearson")),
                      f"{source}|{dimension}_pearson")
                check(row[base + 2], _fmt(stats.get("spearman")),
                      f"{source}|{dimension}_spearman")
                check(row[base + 3], _fmt(stats.get("mae")),
                      f"{source}|{dimension}_mae")
                check(row[base + 4], _fmt(stats.get("rmse")),
                      f"{source}|{dimension}_rmse")
    header, rows = find(lambda h: h[:2] == ["检查项", "阈值"])
    acceptance = run.get("acceptance") or {}
    if rows:
        for row in rows:
            label = row[0]
            if label.startswith("解析成功率："):
                source = label.split("：", 1)[1]
                rate = ((acceptance.get("parse_success") or {}).get(source) or {}).get("rate")
                if rate is not None:
                    check(row[2], f"{100 * float(rate):.2f}%", f"acc|parse|{source}")
            elif label.startswith("输入字覆盖率："):
                source = label.split("：", 1)[1]
                value = ((acceptance.get("alignment_coverage") or {}).get(source) or {}).get(
                    "value"
                )
                if value is not None:
                    check(row[2], f"{100 * float(value):.2f}%", f"acc|coverage|{source}")
            elif label == "D 主实验成本":
                cost = (acceptance.get("cost_cny") or {}).get("value")
                if cost is not None:
                    check(row[2], f"¥{_fmt(cost)}", "acc|cost")
    return {
        "available": True,
        "n_cells_checked": checked,
        "issues": issues[:40],
        "n_issue": len(issues),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    run_dir: Path,
    *,
    manifest_dir: Path | None = None,
    data_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    check_data: bool = True,
    check_audio: bool = False,
) -> dict[str, Any]:
    manifest_dir = manifest_dir or (repo_root / "data" / "sentence" / "manifests")
    data_dir = data_dir or (repo_root / "data" / "sentence" / "d_asr")
    report: dict[str, Any] = {
        "schema": "sentence-d-run-verification/1",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "manifest_dir": str(manifest_dir),
        "data_dir": str(data_dir),
        "pricing": PRICING,
        "acceptance_thresholds": {
            "parse_success_min": ACCEPT_PARSE_MIN,
            "alignment_coverage_min": ACCEPT_COVERAGE_MIN,
            "max_cost_cny": ACCEPT_COST_MAX,
            "expected_utterances": EXPECTED_UTTS,
        },
    }
    if not (run_dir / "generations.jsonl").exists():
        report["error"] = "generations.jsonl missing"
        report["verdict"] = "FAIL"
        report["exit_code"] = 2
        return report
    data: dict[str, Any] | None = None
    if check_data:
        data = verify_d_data(manifest_dir, data_dir, repo_root, check_audio=check_audio)
        report["data"] = data
    asr_completeness: dict[str, Any] = {"archived": False}
    if data is not None:
        checks = {check["item"]: check for check in data.get("checks") or []}
        archived = checks.get("ASR ok rows")
        asr_completeness = {
            "archived": bool(archived and archived["ok"])
            and bool((checks.get("d_items count") or {}).get("ok")),
            "n_archived": (archived or {}).get("actual"),
            "missing": [],
        }
    report["reproducibility"] = verify_reproducibility(repo_root, run_dir, manifest_dir)
    report["run"] = verify_run(run_dir, manifest_dir)
    report["run"]["acceptance"] = acceptance(
        report["run"], data, asr_completeness=asr_completeness
    )
    report["comparison"] = compare_reported(run_dir, report["run"])
    failed: list[dict[str, Any]] = []
    for section in ("data", "reproducibility", "run"):
        block = report.get(section) or {}
        for check in block.get("checks") or []:
            if not check.get("ok"):
                failed.append({"section": section, **check})
    report["failed_checks"] = failed
    checks = report["run"].get("checks") or []
    report["run"]["n_failed"] = sum(1 for check in checks if not check.get("ok"))
    discrepancies = report["comparison"].get("discrepancies") or []
    report["warnings"] = []
    if not (report["run"].get("acceptance") or {}).get("passed"):
        report["warnings"].append("design A.8 acceptance gates FAILED")
    if discrepancies:
        report["warnings"].append(f"{len(discrepancies)} independent-vs-reported discrepancies")
    ok = (
        (report["run"].get("acceptance") or {}).get("passed")
        and not failed
        and not discrepancies
    )
    report["verdict"] = "PASS" if ok else "FAIL"
    report["exit_code"] = 0 if ok else 1
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path,
                        help="task-D run directory (e.g. results/sentence/sentence-d-v1)")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="item manifest directory (default: data/sentence/manifests)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="ASR output directory (default: data/sentence/d_asr)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--skip-data", action="store_true",
                        help="skip d_items/ASR/injection checks")
    parser.add_argument("--check-audio", action="store_true",
                        help="probe the 1,758 learner wavs (slower)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        args.run_dir.resolve(),
        manifest_dir=args.manifest_dir.resolve() if args.manifest_dir else None,
        data_dir=args.data_dir.resolve() if args.data_dir else None,
        repo_root=args.repo_root.resolve(),
        check_data=not args.skip_data,
        check_audio=args.check_audio,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    if args.quiet:
        acceptance_block = report.get("run", {}).get("acceptance", {})
        print(
            f"verdict={report['verdict']} acceptance_passed={acceptance_block.get('passed')} "
            f"failed_checks={len(report.get('failed_checks') or [])} "
            f"discrepancies={len((report.get('comparison') or {}).get('discrepancies') or [])}"
        )
    else:
        print(payload)
    return int(report.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
