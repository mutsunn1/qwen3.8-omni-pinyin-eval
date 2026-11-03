#!/usr/bin/env python
"""Independent verifier for a sentence-level pinyin experiment run.

This script deliberately does **not** import the pipeline's scoring / metrics
modules (``pinyin_eval.sentence.scoring`` / ``metrics`` / ``report``).  It
recomputes the headline numbers from the raw artifacts:

* ``data/sentence/manifests/*.jsonl`` (items / gold labels) and
  ``prep_summary.json`` (data-prep accounting);
* ``<run>/generations.jsonl`` (raw model replies + usage);
* the repo sources for the reproducibility audit.

Only the pure normalisation primitives of ``pinyin_eval.sentence.normalize``
(and the prompt templates for the fingerprint audit) are reused; the scoring
logic, alignment DP, metric aggregation, acceptance gates and cost accounting
are re-implemented here so that a pipeline bug cannot hide behind a shared
code path.

Usage::

    .venv/bin/python scripts/verify_sentence_run.py results/sentence/sentence-v1
    .venv/bin/python scripts/verify_sentence_run.py results/sentence/sentence-v1 \\
        --json-out /tmp/verification.json --skip-data
    .venv/bin/python scripts/verify_sentence_run.py results/sentence/sentence-v1 \\
        --skip-audio --manifest-dir data/sentence/manifests

The JSON report is written to stdout (or ``--json-out``); ``--quiet`` prints
only the verdict line.  Exit codes: 0 = no discrepancy / failed gate,
1 = discrepancies or failed acceptance gate, 2 = missing required artifact.
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

# Reuse ONLY the normalisation primitives (design §6); never scoring/metrics.
from pinyin_eval.normalize import INITIALS  # noqa: E402
from pinyin_eval.sentence.normalize import (  # noqa: E402
    GoldSyllable,
    align_gold,
    apply_n1_sandhi,
    is_erhua_surface,
    normalize_pinyin_token,
    split_surface,
    t2s,
)

#: pricing snapshot from the design §10.1 / project pricing module (hard-coded
#: on purpose: the verifier must not trust the run's own pricing snapshot).
PRICING = {
    "currency": "CNY",
    "input_per_mtok": 0.8,
    "cache_hit_input_per_mtok": 0.1,
    "output_per_mtok": 2.7,
    "audio_tokens_per_second": 7.0,
    "audio_min_seconds": 1.0,
}

TASKS = ("A", "B", "C")
CONDITIONS = ("C1", "C2")
DIMENSIONS_B = ("tone", "consonant", "vowel")
DIMENSIONS_C = ("accuracy", "fluency", "prosody")
A_METRIC_ORDER = (
    "joint_accuracy",
    "initial_accuracy",
    "final_accuracy",
    "tone_accuracy",
    "sentence_exact_rate",
    "insertion_rate",
    "deletion_rate",
    "cer_tone",
    "cer_stripped",
    "alignment_coverage",
)
A_SUBSETS = ("stable", "neutral", "yibu", "erhua")
PARSE_STATUSES = ("ok", "parse_error", "refused", "empty", "api_error", "missing")
AUDIO_EXPECT = {
    "aishell3": {"samplerate": 44100, "channels": 1, "subtype": "PCM_16"},
    "ompal": {"samplerate": 16000, "channels": 1, "subtype": "PCM_16"},
}
DESIGN_COUNTS = {"A": 300, "B": 1754, "C": 1754, "C_native": 82}
ACCEPT_PARSE_MIN = 0.95
ACCEPT_COVERAGE_MIN = 0.95
ACCEPT_COST_MAX = 20.0
FLOAT_TOL = 1e-3  # 0.1 percentage point, design acceptance for reported vs recomputed


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


def mean(values: Sequence[float | int | None]) -> float | None:
    defined = [float(v) for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


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


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Plain Levenshtein distance (own implementation; own DP)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i]
        for j, bj in enumerate(b, 1):
            cur.append(min(prev[j - 1] + (ai != bj), prev[j] + 1, cur[j - 1] + 1))
        prev = cur
    return prev[-1]


def align_sequences(
    gold: Sequence[str], hyp: Sequence[str]
) -> tuple[list[tuple[int, int | None]], int, int]:
    """Levenshtein alignment gold -> hyp; returns ``(pairs, insertions, deletions)``.

    Backtrack priority mirrors the conventional implementation (diagonal first,
    then deletion, then insertion) so that ties resolve deterministically; the
    DP itself is implemented here, not imported.
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
    pairs: list[tuple[int, int | None]] = []
    insertions = 0
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
        insertions += 1
        j -= 1
    pairs.reverse()
    deletions = sum(1 for _g, h in pairs if h is None)
    return pairs, insertions, deletions


def add_check(
    checks: list[dict[str, Any]],
    item: str,
    expected: Any,
    actual: Any,
    ok: bool | None = None,
    detail: str = "",
    severity: str = "error",
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
            "severity": severity,
        }
    )


# ---------------------------------------------------------------------------
# Reply parsing (independent of pinyin_eval.sentence.parsing)
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_REFUSAL_HINTS = ("抱歉", "无法", "不能", "sorry", "cannot", "not able")
_TONE_CHARS = frozenset("āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ")
_JSON_KEYS_A = ("syllables", "words", "items", "chars", "拼音", "音节")
_JSON_KEYS_B = ("words", "syllables", "items", "chars", "逐字")
_TONE_TABLE = {
    "一": 1, "1": 1, "二": 2, "2": 2, "三": 3, "3": 3, "四": 4,
    "4": 4, "五": 5, "5": 5, "轻声": 5, "neutral": 5, "0": 5,
}
_TRUE_STRINGS = {"true", "1", "yes", "是", "正确", "对", "ok", "good", "correct"}
_FALSE_STRINGS = {"false", "0", "no", "否", "错误", "不对", "bad", "wrong", "incorrect"}


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def extract_json(text: str) -> Any | None:
    """Best-effort first JSON value extraction (own implementation)."""
    body = _strip_fences(text)
    if not body:
        return None
    decoder = json.JSONDecoder()
    positions = sorted(
        pos for pos in (body.find("{"), body.find("[")) if pos >= 0
    )
    for start in positions:
        try:
            value, _end = decoder.raw_decode(body[start:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _first(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return str(value).strip()


def as_tone(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        tone = int(value)
        return tone if 1 <= tone <= 5 else 0
    text = str(value).strip()
    if text in _TONE_TABLE:
        return _TONE_TABLE[text]
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else 0


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


def classify_reply(task: str, text: str, *, unwrap_wrapper: bool = True) -> dict[str, Any]:
    """Parse one raw reply into ``{"parse_status", ...payload}`` (own parser)."""
    body = (text or "").strip()
    if not body:
        return {"parse_status": "empty"}
    obj = extract_json(body)
    if obj is None:
        if "{" not in body and "[" not in body:
            lowered = body.lower()
            if any(hint.lower() in lowered for hint in _REFUSAL_HINTS):
                return {"parse_status": "refused"}
        return {"parse_status": "parse_error", "detail": "no JSON value found"}
    task = task.upper()
    if task == "A":
        return _parse_a(obj, unwrap_wrapper=unwrap_wrapper)
    if task == "B":
        return _parse_b(obj, unwrap_wrapper=unwrap_wrapper)
    if task == "C":
        return _parse_c(obj)
    raise KeyError(f"unknown task {task!r}")


def _unwrap_single_object_array(obj: Any, keys: Sequence[str]) -> Any | None:
    """Own implementation of the pipeline's one-element array-wrapper rule.

    ``[{"words": [...]}]`` (observed on task B C2) is accepted; every other
    list shape keeps the historical meaning (bare array for task A).
    """
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], Mapping):
        inner = _first(obj[0], *keys)
        if isinstance(inner, list):
            return inner
    return None


def _parse_a(obj: Any, *, unwrap_wrapper: bool = True) -> dict[str, Any]:
    entries: Any = None
    if isinstance(obj, dict):
        entries = _first(obj, *_JSON_KEYS_A)
    elif isinstance(obj, list):
        entries = None
        if unwrap_wrapper:
            entries = _unwrap_single_object_array(obj, _JSON_KEYS_A)
        entries = entries if entries is not None else obj
    if not isinstance(entries, list) or not entries:
        return {"parse_status": "parse_error", "detail": "no syllable array"}
    syllables: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        initial = _first(entry, "initial", "声母", "ini", "shengmu")
        final = _first(entry, "final", "韵母", "fin", "yunmu")
        tone = _first(entry, "tone", "声调", "tone_number", "diao")
        surface = _clean(str(initial or "")) + _clean(str(final or ""))
        initial_n, final_n = split_surface(surface) if surface else ("", "")
        syllables.append(
            {
                "char": _as_text(_first(entry, "char", "字", "汉字", "character", "text", "zi")),
                "initial": initial_n,
                "final": final_n,
                "tone": as_tone(tone),
            }
        )
    if not syllables:
        return {"parse_status": "parse_error", "detail": "no dict entries"}
    return {"parse_status": "ok", "syllables": syllables}


def _clean(value: str) -> str:
    return re.sub(r"[^a-zü]", "", unicodedata.normalize("NFKC", value or "").lower()
                  .replace("v", "ü").replace("u:", "ü"))


def _parse_b(obj: Any, *, unwrap_wrapper: bool = True) -> dict[str, Any]:
    entries: Any = None
    if isinstance(obj, dict):
        entries = _first(obj, *_JSON_KEYS_B)
    elif isinstance(obj, list):
        entries = None
        if unwrap_wrapper:
            entries = _unwrap_single_object_array(obj, _JSON_KEYS_B)
        entries = entries if entries is not None else obj
    if not isinstance(entries, list) or not entries:
        return {"parse_status": "parse_error", "detail": "no word array"}
    words: list[dict[str, Any]] = []
    n_with_verdict = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw = {
            "tone_ok": _first(entry, "tone_ok", "tone", "声调正确", "tone_correct"),
            "consonant_ok": _first(
                entry, "consonant_ok", "consonant", "声母正确", "initial_ok", "initial"
            ),
            "vowel_ok": _first(entry, "vowel_ok", "vowel", "韵母正确", "final_ok", "final"),
        }
        verdicts = {key: as_bool(value) for key, value in raw.items()}
        if any(value is not None for value in verdicts.values()):
            n_with_verdict += 1
        words.append(
            {
                "text": _as_text(_first(entry, "text", "char", "字", "汉字", "word")),
                **{key: bool(value) for key, value in verdicts.items()},
            }
        )
    if not words or n_with_verdict == 0:
        return {"parse_status": "parse_error", "detail": "no boolean verdicts"}
    return {"parse_status": "ok", "words": words}


def _parse_c(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        return {"parse_status": "parse_error", "detail": "no JSON object"}
    nested = _first(obj, "scores", "dimensions", "ratings", "评分")
    source = nested if isinstance(nested, Mapping) else obj
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
        value = as_score(raw)
        if value is not None:
            scores[dimension] = value
    if len(scores) < 3:
        return {
            "parse_status": "parse_error",
            "detail": f"missing dimensions: {sorted(set(aliases) - set(scores))}",
        }
    return {"parse_status": "ok", "scores": {key: scores[key] for key in DIMENSIONS_C}}


def parse_record(
    task: str, record: Mapping[str, Any], *, unwrap_wrapper: bool = True
) -> dict[str, Any]:
    if record.get("error"):
        return {"parse_status": "api_error", "detail": str(record.get("error"))[:200]}
    return classify_reply(
        task, str(record.get("response_text") or ""), unwrap_wrapper=unwrap_wrapper
    )


# ---------------------------------------------------------------------------
# Per-item scoring (independent re-implementation of design §6/§7)
# ---------------------------------------------------------------------------


def score_a_item(item: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    gold = align_gold(str(item.get("chars") or ""), list(item.get("pinyin_surface") or []))
    if gold is None:
        raise ValueError(f"gold not alignable for {item.get('item_id')}")
    n_gold = len(gold)
    row: dict[str, Any] = {"n_gold": n_gold}
    if parsed.get("parse_status") != "ok":
        return row
    hyp = [
        {
            "char": str(entry.get("char") or ""),
            "initial": str(entry.get("initial") or ""),
            "final": str(entry.get("final") or ""),
            "tone": int(entry.get("tone") or 0),
        }
        for entry in parsed.get("syllables") or []
    ]
    gold_chars = [t2s(syllable.char) for syllable in gold]
    hyp_chars = [t2s(syllable["char"]) for syllable in hyp]
    pairs, insertions, deletions = align_sequences(gold_chars, hyp_chars)
    n_matched = 0
    n_initial = n_final = n_tone = n_joint = 0
    gold_n1 = apply_n1_sandhi(list(gold))
    hyp_n1 = apply_n1_sandhi(
        [
            GoldSyllable(
                initial=syllable["initial"], final=syllable["final"],
                tone=syllable["tone"], char=syllable["char"],
            )
            for syllable in hyp
        ]
    )
    n1_initial = n1_final = n1_tone = n1_joint = 0
    for gold_index, hyp_index in pairs:
        gold_syllable = gold[gold_index]
        hyp_syllable = hyp[hyp_index] if hyp_index is not None else None
        initial_ok = hyp_syllable is not None and gold_syllable.initial == hyp_syllable["initial"]
        final_ok = hyp_syllable is not None and gold_syllable.final == hyp_syllable["final"]
        tone_ok = (
            hyp_syllable is not None
            and gold_syllable.tone in (1, 2, 3, 4, 5)
            and gold_syllable.tone == hyp_syllable["tone"]
        )
        if hyp_syllable is None:
            continue
        n_matched += 1
        n_initial += initial_ok
        n_final += final_ok
        n_tone += tone_ok
        n_joint += initial_ok and final_ok and tone_ok
        g1, h1 = gold_n1[gold_index], hyp_n1[hyp_index]
        n1_initial += g1.initial == h1.initial
        n1_final += g1.final == h1.final
        n1_tone += g1.tone in (1, 2, 3, 4, 5) and g1.tone == h1.tone
        n1_joint += (
            g1.initial == h1.initial
            and g1.final == h1.final
            and g1.tone in (1, 2, 3, 4, 5)
            and g1.tone == h1.tone
        )
    n_insertions = len(hyp) - n_matched
    row.update(
        {
            "n_hyp": len(hyp),
            "n_matched": n_matched,
            "n_deletions": deletions,
            "n_insertions": n_insertions,
            "n_initial_correct": n_initial,
            "n_final_correct": n_final,
            "n_tone_correct": n_tone,
            "n_joint_correct": n_joint,
            "n1_n_initial_correct": n1_initial,
            "n1_n_final_correct": n1_final,
            "n1_n_tone_correct": n1_tone,
            "n1_n_joint_correct": n1_joint,
            "cer_tone_num": edit_distance(
                [syllable.with_tone for syllable in gold],
                [f"{s['initial']}{s['final']}{s['tone']}" if s["tone"] else
                 f"{s['initial']}{s['final']}" for s in hyp],
            ),
            "cer_stripped_num": edit_distance(
                [syllable.surface for syllable in gold],
                [f"{s['initial']}{s['final']}" for s in hyp],
            ),
            "n1_cer_tone_num": edit_distance(
                [syllable.with_tone for syllable in gold_n1],
                [syllable.with_tone for syllable in hyp_n1],
            ),
            "sentence_exact": bool(
                n_gold > 0
                and deletions == 0
                and n_insertions == 0
                and n_joint == n_gold
            ),
            "n1_sentence_exact": bool(
                n_gold > 0
                and deletions == 0
                and n_insertions == 0
                and n1_joint == n_gold
            ),
        }
    )
    return row


def _gold_label(unit: Mapping[str, Any], dimension: str) -> int | None:
    value = unit.get(dimension)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    return None


def score_b_item(item: Mapping[str, Any], parsed: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = [unit for unit in (item.get("labels") or []) if isinstance(unit, Mapping)]
    words = [word for word in (parsed.get("words") or []) if isinstance(word, Mapping)]
    gold_texts = [t2s(str(unit.get("text") or "")) for unit in labels]
    hyp_texts = [t2s(str(word.get("text") or "")) for word in words]
    pairs, _ins, _del = align_sequences(gold_texts, hyp_texts)
    matched: dict[int, int] = {}
    for gold_index, hyp_index in pairs:
        if hyp_index is None:
            continue
        if gold_texts[gold_index] == hyp_texts[hyp_index]:
            matched[gold_index] = hyp_index
    n_units = len(labels)
    n_inserted = len(words) - len(set(matched.values()))
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS_B:
        tp = fp = fn = tn = 0
        ztp = zfp = zfn = ztn = 0
        for gold_index, unit in enumerate(labels):
            zero_initial = dimension == "consonant" and bool(unit.get("zero_initial"))
            gold_value = _gold_label(unit, dimension)
            hyp_index = matched.get(gold_index)
            if hyp_index is None:
                fn += 1
                zfn += zero_initial
                continue
            pred_error = not bool(words[hyp_index].get(f"{dimension}_ok"))
            if gold_value == 0:
                if pred_error:
                    tp += 1
                    ztp += zero_initial
                else:
                    fn += 1
                    zfn += zero_initial
            elif gold_value == 1:
                if pred_error:
                    fp += 1
                    zfp += zero_initial
                else:
                    tn += 1
                    ztn += zero_initial
            else:
                fn += 1
                zfn += zero_initial
        rows.append(
            {
                "dimension": dimension,
                "n_units": n_units,
                "n_matched": len(matched),
                "n_missing": n_units - len(matched),
                "n_inserted": n_inserted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_zero_initial_tp": ztp,
                "n_zero_initial_fp": zfp,
                "n_zero_initial_fn": zfn,
                "n_zero_initial_tn": ztn,
            }
        )
    return rows


def score_c_item(item: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    gold_source = item.get("scores") or {}
    gold = {dimension: _finite_float(gold_source.get(dimension)) for dimension in DIMENSIONS_C}
    if parsed.get("parse_status") == "ok":
        parsed_scores = parsed.get("scores") or {}
        pred = {dimension: _finite_float(parsed_scores.get(dimension)) for dimension in DIMENSIONS_C}
    else:
        pred = dict.fromkeys(DIMENSIONS_C)
    row: dict[str, Any] = {"native": bool(item.get("native"))}
    row.update({f"gold_{dimension}": gold[dimension] for dimension in DIMENSIONS_C})
    row.update({f"pred_{dimension}": pred[dimension] for dimension in DIMENSIONS_C})
    row["gold_mean"] = (
        sum(gold.values()) / 3.0
        if all(value is not None for value in gold.values())
        else None
    )
    row["pred_mean"] = (
        sum(pred.values()) / 3.0
        if all(value is not None for value in pred.values())
        else None
    )
    return row


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


# ---------------------------------------------------------------------------
# Metric aggregation (own implementation of design §7)
# ---------------------------------------------------------------------------


def aggregate_a(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("parse_ok")]
    n_items = len(rows)
    n_ok = len(ok_rows)
    n_gold_ok = sum(int(row.get("n_gold") or 0) for row in ok_rows)
    n_matched = sum(int(row.get("n_matched") or 0) for row in ok_rows)
    n_insertions = sum(int(row.get("n_insertions") or 0) for row in ok_rows)
    n_deletions = sum(int(row.get("n_deletions") or 0) for row in ok_rows)
    headline = {
        "joint_accuracy": ratio(sum(int(r.get("n_joint_correct") or 0) for r in ok_rows), n_gold_ok),
        "initial_accuracy": ratio(
            sum(int(r.get("n_initial_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "final_accuracy": ratio(
            sum(int(r.get("n_final_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "tone_accuracy": ratio(
            sum(int(r.get("n_tone_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "sentence_exact_rate": ratio(
            sum(1 for r in ok_rows if r.get("sentence_exact")), len(ok_rows)
        ),
        "insertion_rate": ratio(n_insertions, n_gold_ok),
        "deletion_rate": ratio(n_deletions, n_gold_ok),
        "cer_tone": ratio(sum(float(r.get("cer_tone_num") or 0) for r in ok_rows), n_gold_ok),
        "cer_stripped": ratio(
            sum(float(r.get("cer_stripped_num") or 0) for r in ok_rows), n_gold_ok
        ),
        "alignment_coverage": ratio(n_matched, n_gold_ok),
    }
    n1 = {
        "n1_joint_accuracy": ratio(
            sum(int(r.get("n1_n_joint_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "n1_initial_accuracy": ratio(
            sum(int(r.get("n1_n_initial_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "n1_final_accuracy": ratio(
            sum(int(r.get("n1_n_final_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "n1_tone_accuracy": ratio(
            sum(int(r.get("n1_n_tone_correct") or 0) for r in ok_rows), n_gold_ok
        ),
        "n1_cer_tone": ratio(
            sum(float(r.get("n1_cer_tone_num") or 0) for r in ok_rows), n_gold_ok
        ),
        "n1_sentence_exact_rate": ratio(
            sum(1 for r in ok_rows if r.get("n1_sentence_exact")), len(ok_rows)
        ),
    }
    return {
        "n_items": n_items,
        "n_parse_ok": n_ok,
        "n_parse_failures": n_items - n_ok,
        "n_gold": n_gold_ok,
        "n_gold_ok": n_gold_ok,
        "n_matched": n_matched,
        "n_insertions": n_insertions,
        "n_deletions": n_deletions,
        "headline": headline,
        "n1": n1,
    }


def aggregate_a_subset(rows: Sequence[Mapping[str, Any]], subset: str) -> dict[str, Any]:
    if subset == "stable":
        selected = [row for row in rows if row.get("stable")]
    else:
        selected = [row for row in rows if bool((row.get("flags") or {}).get(subset))]
    ok_rows = [row for row in selected if row.get("parse_ok")]
    n_gold = sum(int(row.get("n_gold") or 0) for row in ok_rows)
    result = {
        "n_items": len(selected),
        "n_parse_ok": len(ok_rows),
        "joint_accuracy": ratio(
            sum(int(r.get("n_joint_correct") or 0) for r in ok_rows), n_gold
        ),
        "initial_accuracy": ratio(
            sum(int(r.get("n_initial_correct") or 0) for r in ok_rows), n_gold
        ),
        "final_accuracy": ratio(
            sum(int(r.get("n_final_correct") or 0) for r in ok_rows), n_gold
        ),
        "tone_accuracy": ratio(
            sum(int(r.get("n_tone_correct") or 0) for r in ok_rows), n_gold
        ),
        "sentence_exact_rate": ratio(
            sum(1 for r in ok_rows if r.get("sentence_exact")), len(ok_rows)
        ),
        "insertion_rate": ratio(
            sum(int(r.get("n_insertions") or 0) for r in ok_rows), n_gold
        ),
        "deletion_rate": ratio(
            sum(int(r.get("n_deletions") or 0) for r in ok_rows), n_gold
        ),
        "cer_tone": ratio(sum(float(r.get("cer_tone_num") or 0) for r in ok_rows), n_gold),
        "cer_stripped": ratio(
            sum(float(r.get("cer_stripped_num") or 0) for r in ok_rows), n_gold
        ),
        "alignment_coverage": ratio(
            sum(int(r.get("n_matched") or 0) for r in ok_rows), n_gold
        ),
    }
    return result


def _confusion_stats(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    f1_error = ratio(2 * tp, 2 * tp + fp + fn)
    f1_ok = ratio(2 * tn, 2 * tn + fn + fp)
    f1_values = [value for value in (f1_error, f1_ok) if value is not None]
    return {
        "n_units": int(tp + fp + fn + tn),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "macro_f1": (sum(f1_values) / len(f1_values)) if f1_values else None,
        "f1_error": f1_error,
        "f1_ok": f1_ok,
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
    }


def aggregate_b(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate task-B per-dimension rows (own sum + own statistics)."""
    result: dict[str, Any] = {}
    total_matched = total_units = 0
    for dimension in DIMENSIONS_B:
        dim_rows = [row for row in rows if row.get("dimension") == dimension]
        tp = sum(int(row.get("tp") or 0) for row in dim_rows)
        fp = sum(int(row.get("fp") or 0) for row in dim_rows)
        fn = sum(int(row.get("fn") or 0) for row in dim_rows)
        tn = sum(int(row.get("tn") or 0) for row in dim_rows)
        stats = _confusion_stats(tp, fp, fn, tn)
        stats["n_items"] = len({row.get("item_id", i) for i, row in enumerate(dim_rows)})
        result[dimension] = stats
        if dimension == "tone":
            total_matched = sum(int(row.get("n_matched") or 0) for row in dim_rows)
            total_units = sum(int(row.get("n_units") or 0) for row in dim_rows)
    result["alignment_coverage"] = ratio(total_matched, total_units)
    zero_rows = [row for row in rows if row.get("dimension") == "consonant"]
    ztp = sum(int(row.get("n_zero_initial_tp") or 0) for row in zero_rows)
    zfp = sum(int(row.get("n_zero_initial_fp") or 0) for row in zero_rows)
    zfn = sum(int(row.get("n_zero_initial_fn") or 0) for row in zero_rows)
    ztn = sum(int(row.get("n_zero_initial_tn") or 0) for row in zero_rows)
    result["zero_initial_consonant"] = _confusion_stats(ztp, zfp, zfn, ztn)
    result["n_items"] = len({row.get("item_id", i) for i, row in enumerate(rows)})
    return result


def _c_pairs(rows: Iterable[Mapping[str, Any]], dimension: str) -> tuple[list[float], list[float]]:
    gold_values: list[float] = []
    pred_values: list[float] = []
    for row in rows:
        if dimension == "mean_dimension":
            golds = [_finite_float(row.get(f"gold_{name}")) for name in DIMENSIONS_C]
            preds = [_finite_float(row.get(f"pred_{name}")) for name in DIMENSIONS_C]
            if any(value is None for value in golds + preds):
                continue
            gold_values.append(sum(golds) / 3.0)  # type: ignore[arg-type]
            pred_values.append(sum(preds) / 3.0)  # type: ignore[arg-type]
        else:
            gold = _finite_float(row.get(f"gold_{dimension}"))
            pred = _finite_float(row.get(f"pred_{dimension}"))
            if gold is None or pred is None:
                continue
            gold_values.append(gold)
            pred_values.append(pred)
    return gold_values, pred_values


def aggregate_c(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Task-C correlations from learner (non-native) rows + native anchor."""
    learner_rows = [row for row in rows if not row.get("native")]
    native_rows = [row for row in rows if row.get("native")]
    result: dict[str, Any] = {}
    for dimension in (*DIMENSIONS_C, "mean_dimension"):
        gold_values, pred_values = _c_pairs(learner_rows, dimension)
        result[dimension] = {
            "n": len(gold_values),
            "pearson": pearson(gold_values, pred_values),
            "spearman": spearman(gold_values, pred_values),
            "mae": mae(gold_values, pred_values),
            "rmse": rmse(gold_values, pred_values),
        }
    gold_means: list[float] = []
    pred_means: list[float] = []
    per_dim: dict[str, dict[str, list[float]]] = {
        dimension: {"gold": [], "pred": []} for dimension in DIMENSIONS_C
    }
    for row in native_rows:
        golds = [_finite_float(row.get(f"gold_{name}")) for name in DIMENSIONS_C]
        preds = [_finite_float(row.get(f"pred_{name}")) for name in DIMENSIONS_C]
        if all(value is not None for value in golds):
            gold_means.append(sum(golds) / 3.0)  # type: ignore[arg-type]
        if all(value is not None for value in preds):
            pred_means.append(sum(preds) / 3.0)  # type: ignore[arg-type]
        for dimension, gold, pred in zip(DIMENSIONS_C, golds, preds, strict=True):
            if gold is not None:
                per_dim[dimension]["gold"].append(gold)
            if pred is not None:
                per_dim[dimension]["pred"].append(pred)
    result["native_anchor"] = {
        "n": len(native_rows),
        "mean_gold": mean(gold_means),
        "mean_pred": mean(pred_means),
        "per_dimension": {
            dimension: {
                "mean_gold": mean(values["gold"]),
                "mean_pred": mean(values["pred"]),
            }
            for dimension, values in per_dim.items()
        },
    }
    ok_rows = [row for row in rows if row.get("parse_ok", True)]
    n_ok = len(ok_rows)
    n_aligned = sum(
        1
        for row in ok_rows
        if all(
            _finite_float(row.get(f"{side}_{dimension}")) is not None
            for side in ("gold", "pred")
            for dimension in DIMENSIONS_C
        )
    )
    result["n_items"] = len(rows)
    result["n_parse_ok"] = n_ok
    result["alignment_coverage"] = ratio(n_aligned, n_ok)
    return result


# ---------------------------------------------------------------------------
# Cost accounting (own implementation of the design §10.1 / pricing formulas)
# ---------------------------------------------------------------------------


def usage_of(record: Mapping[str, Any]) -> dict[str, int]:
    usage = record.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens") or 0),
        "audio_tokens": int(prompt_details.get("audio_tokens") or 0),
        "reasoning_tokens": int((usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens") or 0),
    }


def record_cost(usage: Mapping[str, int]) -> dict[str, float]:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = max(0, min(int(usage.get("cached_tokens") or 0), prompt))
    uncached = prompt - cached
    input_cost = (
        uncached * PRICING["input_per_mtok"]
        + cached * PRICING["cache_hit_input_per_mtok"]
    ) / 1_000_000.0
    output_cost = int(usage.get("completion_tokens") or 0) * PRICING["output_per_mtok"] / 1_000_000.0
    return {
        "input_cost_cny": input_cost,
        "output_cost_cny": output_cost,
        "total_cost_cny": input_cost + output_cost,
    }


# ---------------------------------------------------------------------------
# Data (manifests) verification — Phase 1
# ---------------------------------------------------------------------------


def _parse_aishell_labels(text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Own parser for AISHELL-3 ``label_train-set.txt`` (design data source)."""
    lines: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("|")
        if len(parts) != 3:
            malformed.append({"line_no": line_no, "error": f"fields={len(parts)}"})
            continue
        utt_id, pinyin, chars = parts
        chars_clean = "".join(
            ch for ch in unicodedata.normalize("NFKC", chars) if ch not in "%$"
        )
        chars_clean = re.sub(r"\s+", "", chars_clean)
        pinyin_clean = "".join(
            ch for ch in unicodedata.normalize("NFKC", pinyin) if ch not in "%$"
        )
        tokens = [token for token in pinyin_clean.split() if token]
        if not utt_id or not tokens or not chars_clean:
            malformed.append({"line_no": line_no, "error": "empty field"})
            continue
        if utt_id in seen:
            malformed.append({"line_no": line_no, "error": "duplicate id"})
            continue
        seen.add(utt_id)
        lines.append({"utt_id": utt_id, "tokens": tokens, "chars": chars_clean})
    return lines, malformed


def _parse_spk_info(text: str) -> dict[str, dict[str, str]]:
    infos: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split("\t")]
        if len(fields) != 4:
            continue
        speaker, age, gender, accent = fields
        infos[speaker] = {"age": age, "gender": gender, "accent": accent}
    return infos


def _probe_audio(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        return {
            "ok": True,
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "frames": int(info.frames),
            "duration_s": round(duration, 6),
        }
    except Exception as exc:  # noqa: BLE001 - report, never abort
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _bare_id(item_id: Any) -> str:
    """Strip the ``TASK|`` prefix of a manifest item id (OMPAL utt id)."""
    return str(item_id).split("|", 1)[-1]


def verify_data(
    manifest_dir: Path,
    repo_root: Path,
    *,
    check_audio: bool = True,
) -> dict[str, Any]:
    """Independently verify the prepared manifests + source accounting."""
    checks: list[dict[str, Any]] = []
    data: dict[str, Any] = {"checks": checks}
    prep = json_load(manifest_dir / "prep_summary.json", {}) or {}
    data["prep_summary_present"] = bool(prep)
    data_dir = manifest_dir.parent

    # -- source file hashes -------------------------------------------------
    hash_mismatches: list[dict[str, Any]] = []
    sources = prep.get("sources") or {}
    for _source_name, source in sources.items():
        for relative, expected_hash in (source.get("files_sha256") or {}).items():
            actual = sha256_file(repo_root / relative)
            if actual != expected_hash:
                hash_mismatches.append(
                    {"file": relative, "expected": expected_hash, "actual": actual}
                )
    add_check(
        checks,
        "source file sha256",
        "all match prep_summary.sources",
        {"files": sum(len(s.get("files_sha256") or {}) for s in sources.values()),
         "mismatches": len(hash_mismatches)},
        ok=bool(sources) and not hash_mismatches,
        detail=json.dumps(hash_mismatches[:5], ensure_ascii=False),
    )
    data["source_hashes"] = {"mismatches": hash_mismatches, "checked": True}

    # -- AISHELL-3 sampling frame -------------------------------------------
    aishell_dir = data_dir / "aishell3"
    label_lines: list[dict[str, Any]] = []
    label_malformed: list[dict[str, Any]] = []
    spk_infos: dict[str, dict[str, str]] = {}
    if (aishell_dir / "label_train-set.txt").exists():
        label_lines, label_malformed = _parse_aishell_labels(
            (aishell_dir / "label_train-set.txt").read_text(encoding="utf-8")
        )
    if (aishell_dir / "spk-info.txt").exists():
        spk_infos = _parse_spk_info(
            (aishell_dir / "spk-info.txt").read_text(encoding="utf-8")
        )
    frame = (prep.get("aishell3") or {}).get("frame") or {}
    label_speakers = {line["utt_id"][:7] for line in label_lines}
    add_check(checks, "AISHELL label lines", frame.get("label_train_utt_lines"), len(label_lines),
              detail="independent parse of label_train-set.txt")
    add_check(checks, "AISHELL malformed label lines", 0, len(label_malformed))
    add_check(checks, "AISHELL label speakers", frame.get("label_train_speakers"),
              len(label_speakers))
    add_check(checks, "AISHELL spk-info rows", frame.get("spk_info_rows"), len(spk_infos))

    unalignable: list[dict[str, Any]] = []
    length_filtered = 0
    eligible: list[dict[str, Any]] = []
    for line in label_lines:
        gold = align_gold(line["chars"], list(line["tokens"]))
        if gold is None:
            unalignable.append(
                {
                    "utt_id": line["utt_id"],
                    "n_chars": len(line["chars"]),
                    "n_tokens": len(line["tokens"]),
                }
            )
            continue
        if not (5 <= len(line["chars"]) <= 30):
            length_filtered += 1
            continue
        eligible.append(line)
    eligible_by_speaker: Counter[str] = Counter(line["utt_id"][:7] for line in eligible)
    add_check(checks, "AISHELL unalignable rows", frame.get("unalignable_rows"), len(unalignable))
    unalignable_ids_match = sorted(u["utt_id"] for u in unalignable) == sorted(
        u["utt_id"] for u in frame.get("unalignable") or []
    )
    add_check(
        checks,
        "AISHELL unalignable ids",
        "prep_summary list",
        "match" if unalignable_ids_match else "mismatch",
        ok=unalignable_ids_match,
        detail=f"independent={len(unalignable)} prep={len(frame.get('unalignable') or [])}",
    )
    add_check(checks, "AISHELL length-filtered rows (outside 5-30)",
              frame.get("length_filtered_rows"), length_filtered)
    add_check(checks, "AISHELL eligible utterances", frame.get("eligible_utterances"),
              len(eligible))
    add_check(checks, "AISHELL eligible speakers", frame.get("eligible_speakers"),
              len(eligible_by_speaker))
    speakers_min_utts = sum(1 for count in eligible_by_speaker.values() if count >= 10)
    add_check(checks, "AISHELL speakers with >=10 eligible utterances",
              frame.get("speakers_with_min_utts"), speakers_min_utts)
    data["aishell3_frame"] = {
        "label_lines": len(label_lines),
        "label_speakers": len(label_speakers),
        "malformed": len(label_malformed),
        "spk_info_rows": len(spk_infos),
        "unalignable": len(unalignable),
        "unalignable_ids_match": sorted(u["utt_id"] for u in unalignable)
        == sorted(u["utt_id"] for u in frame.get("unalignable") or []),
        "length_filtered": length_filtered,
        "eligible": len(eligible),
        "eligible_speakers": len(eligible_by_speaker),
        "speakers_with_min_utts": speakers_min_utts,
        "eligible_by_speaker": dict(eligible_by_speaker),
    }

    # -- task A manifest ----------------------------------------------------
    a_items = load_jsonl(manifest_dir / "aishell3_items.jsonl")
    a_speakers = Counter(str(item.get("speaker") or "") for item in a_items)
    a_attrs: dict[str, dict[str, str]] = {}
    a_attr_mismatch: list[dict[str, Any]] = []
    a_gold_unalignable: list[str] = []
    a_bad_chars: list[str] = []
    a_too_short_long: list[str] = []
    coverage_initials: Counter[str] = Counter()
    coverage_finals: Counter[str] = Counter()
    coverage_tones: Counter[int] = Counter()
    flag_combos: Counter[str] = Counter()
    flag_mismatch: list[str] = []
    for item in a_items:
        item_id = str(item.get("item_id") or "")
        speaker = str(item.get("speaker") or "")
        meta = item.get("meta") or {}
        a_attrs[speaker] = {key: str(meta.get(key)) for key in ("age", "gender", "accent")}
        chars = str(item.get("chars") or "")
        text = str(item.get("text") or "")
        if chars != text:
            a_bad_chars.append(item_id)
        if not 5 <= len(chars) <= 30:
            a_too_short_long.append(item_id)
        gold = align_gold(chars, list(item.get("pinyin_surface") or []))
        if gold is None or len(gold) != len(chars):
            a_gold_unalignable.append(item_id)
            continue
        for syllable in gold:
            if syllable.initial:
                coverage_initials[syllable.initial] += 1
            coverage_finals[syllable.final] += 1
            if syllable.tone in (1, 2, 3, 4):
                coverage_tones[syllable.tone] += 1
        flags = {
            "neutral": any(syllable.tone == 5 for syllable in gold),
            "yibu": any(ch in "一不" for ch in chars),
            "erhua": any(
                is_erhua_surface(normalize_pinyin_token(token).surface)
                for token in item.get("pinyin_surface") or []
            ),
        }
        combo = "/".join(name for name in ("neutral", "yibu", "erhua") if flags[name]) or "none"
        flag_combos[combo] += 1
        item_flags = item.get("flags") or {}
        if any(bool(item_flags.get(name)) != flags[name] for name in flags):
            flag_mismatch.append(item_id)
    # independent strata from spk-info
    for speaker, attrs in a_attrs.items():
        expected = spk_infos.get(speaker)
        if expected is None or any(expected.get(key) != attrs.get(key) for key in attrs):
            a_attr_mismatch.append({"speaker": speaker, "manifest": attrs, "spk_info": expected})
    strata = {
        "gender": Counter(a_attrs[s]["gender"] for s in a_attrs),
        "accent": Counter(a_attrs[s]["accent"] for s in a_attrs),
        "age": Counter(a_attrs[s]["age"] for s in a_attrs),
    }
    add_check(checks, "A item count", 300, len(a_items))
    add_check(checks, "A distinct speakers", 30, len(a_speakers))
    add_check(checks, "A items per speaker", {"10": 30},
              dict(sorted((str(n), count) for n, count in
                          Counter(a_speakers.values()).items())))
    add_check(checks, "A speaker attributes vs spk-info", 0, len(a_attr_mismatch))
    add_check(checks, "A strata gender", {"female": 18, "male": 12}, dict(strata["gender"]))
    add_check(checks, "A strata accent", {"north": 22, "south": 8}, dict(strata["accent"]))
    add_check(checks, "A strata age", {"B": 24, "C": 5, "D": 1}, dict(strata["age"]))
    add_check(checks, "A chars == text", 0, len(a_bad_chars))
    add_check(checks, "A lengths within 5-30", 0, len(a_too_short_long))
    add_check(checks, "A gold pinyin alignable", 0, len(a_gold_unalignable))
    add_check(checks, "A sampled speakers have >=10 eligible utts", 0,
              sum(1 for s in a_speakers if eligible_by_speaker.get(s, 0) < 10))
    prep_coverage = (prep.get("aishell3") or {}).get("coverage") or {}
    canonical_initials = list(INITIALS)
    coverage_finals_canonical = (
        "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
        "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu",
        "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ueng", "ui",
        "un", "uo", "ü", "üan", "üe", "ün",
    )
    missing_initials = [name for name in canonical_initials
                        if not coverage_initials.get(name)]
    missing_finals = [name for name in coverage_finals_canonical
                      if not coverage_finals.get(name)]
    missing_tones = [tone for tone in (1, 2, 3, 4) if not coverage_tones.get(tone)]
    add_check(checks, "A coverage initials/finals/tones present",
              {"initials": len(canonical_initials), "finals": len(coverage_finals_canonical),
               "tones": 4,
               "missing": {"initials": [], "finals": [], "tones": []}},
              {"initials": len(canonical_initials) - len(missing_initials),
               "finals": len(coverage_finals_canonical) - len(missing_finals),
               "tones": 4 - len(missing_tones),
               "missing": {"initials": missing_initials, "finals": missing_finals,
                           "tones": missing_tones}})
    add_check(
        checks,
        "A coverage counts match prep",
        "equal initial/final/tone counters",
        {
            "initials_equal": dict(coverage_initials)
            == {k: v for k, v in (prep_coverage.get("initials") or {}).items() if v},
            "finals_equal": dict(coverage_finals)
            == {k: v for k, v in (prep_coverage.get("finals") or {}).items() if v},
            "tones_equal": {str(k): v for k, v in coverage_tones.items()}
            == {k: v for k, v in (prep_coverage.get("tones") or {}).items() if v},
        },
        ok=(
            dict(coverage_initials)
            == {k: v for k, v in (prep_coverage.get("initials") or {}).items() if v}
            and dict(coverage_finals)
            == {k: v for k, v in (prep_coverage.get("finals") or {}).items() if v}
            and {str(k): v for k, v in coverage_tones.items()}
            == {k: v for k, v in (prep_coverage.get("tones") or {}).items() if v}
        ),
    )
    special = (prep.get("aishell3") or {}).get("special_subset") or {}
    add_check(checks, "A special subset flags recomputed", 0, len(flag_mismatch),
              ok=not flag_mismatch)
    add_check(checks, "A special subset count >= 50", True, special.get("count", 0) >= 50,
              detail=f"count={special.get('count')}")
    add_check(checks, "A special subset breakdown matches prep",
              special.get("breakdown"),
              {key: value for key, value in sorted(flag_combos.items()) if key != "none"})
    a_audio = _check_audio(
        checks, "A", a_items, repo_root,
        expected_format=AUDIO_EXPECT["aishell3"], check_audio=check_audio,
    )
    data["aishell3_manifest"] = {
        "n": len(a_items),
        "n_speakers": len(a_speakers),
        "items_per_speaker": dict(Counter(a_speakers.values())),
        "speakers": sorted(a_speakers),
        "strata": {key: dict(value) for key, value in strata.items()},
        "attr_mismatches": a_attr_mismatch[:5],
        "bad_chars": a_bad_chars[:5],
        "gold_unalignable": a_gold_unalignable[:5],
        "coverage": {
            "initials": dict(sorted(coverage_initials.items())),
            "finals": dict(sorted(coverage_finals.items())),
            "tones": {str(k): v for k, v in sorted(coverage_tones.items())},
        },
        "special_subset": {"count": special.get("count"),
                           "breakdown": dict(sorted(flag_combos.items()))},
        "flag_mismatches": flag_mismatch[:5],
        "audio": a_audio,
    }

    # -- OMPAL source + manifests ------------------------------------------
    repo_dir = data_dir / "ompal" / "repo"
    learner_src = json_load(repo_dir / "non-native_scores.json", {}) or {}
    native_src = json_load(repo_dir / "native_scores.json", {}) or {}
    cleaning = (prep.get("ompal") or {}).get("cleaning") or {}
    anomaly = {
        "learner_multi_slot": [], "learner_joined_mismatch": [],
        "native_multi_slot": [], "native_joined_mismatch": [],
    }
    for tag, source in (("learner", learner_src), ("native", native_src)):
        for utt_id, record in source.items():
            slots: list[str] = []
            multi = False
            for word in record.get("words") or []:
                slot_text = word.get("text")
                if isinstance(slot_text, list):
                    if len(slot_text) > 1:
                        multi = True
                    slots.extend(str(part) for part in slot_text)
                else:
                    slots.append(str(slot_text))
            if multi:
                anomaly[f"{tag}_multi_slot"].append(utt_id)
            if "".join(slots) != str(record.get("text") or ""):
                anomaly[f"{tag}_joined_mismatch"].append(utt_id)
    learner_excluded = set(anomaly["learner_multi_slot"]) | set(
        anomaly["learner_joined_mismatch"]
    )
    prep_excluded = {entry.get("utt_id") for entry in cleaning.get("exclusions") or []}
    add_check(checks, "OMPAL raw learner rows", 1768, len(learner_src))
    add_check(checks, "OMPAL raw native rows", 82, len(native_src))
    add_check(checks, "OMPAL learner corrupted word-slot rows", 9,
              len(anomaly["learner_multi_slot"]))
    add_check(checks, "OMPAL learner text/slot mismatch rows", 10,
              len(anomaly["learner_joined_mismatch"]))
    add_check(checks, "OMPAL learner exclusions (union)",
              sorted(str(value) for value in prep_excluded),
              sorted(str(value) for value in learner_excluded))
    add_check(checks, "OMPAL cleaned learner rows", 1758,
              len(learner_src) - len(learner_excluded))
    add_check(checks, "OMPAL native corrupted word-slot rows", 3,
              len(anomaly["native_multi_slot"]))
    add_check(checks, "OMPAL native text/slot mismatch rows", 4,
              len(anomaly["native_joined_mismatch"]))
    add_check(checks, "OMPAL cleaning count matches prep", cleaning.get("excluded_learner"),
              len(learner_excluded))

    b_items = load_jsonl(manifest_dir / "ompal_b_items.jsonl")
    c_items = load_jsonl(manifest_dir / "ompal_c_items.jsonl")
    native_items = load_jsonl(manifest_dir / "ompal_native_items.jsonl")
    b_by_id = {str(item.get("item_id")): item for item in b_items}
    c_by_id = {str(item.get("item_id")): item for item in c_items}
    learners = Counter(str(item.get("learner") or "") for item in b_items)
    add_check(checks, "B item count", 1758, len(b_items))
    add_check(checks, "C item count", 1758, len(c_items))
    add_check(checks, "native anchor item count", 82, len(native_items))
    add_check(checks, "B learners", 46, len(learners))
    b_bare = {_bare_id(item_id) for item_id in b_by_id}
    c_bare = {_bare_id(item_id) for item_id in c_by_id}
    add_check(checks, "B/C same item ids", sorted(b_bare), sorted(c_bare))
    b_bad_labels: list[str] = []
    b_bad_text: list[str] = []
    b_units = 0
    zero_units = 0
    source_label_mismatch: list[str] = []
    source_score_mismatch: list[str] = []
    for item in b_items:
        item_id = str(item.get("item_id"))
        source = learner_src.get(_bare_id(item_id))
        text = str(item.get("text") or "")
        labels = item.get("labels") or []
        if source is None:
            source_label_mismatch.append(f"{item_id}:missing_source")
            continue
        expected_labels: list[dict[str, Any]] = []
        for word in source.get("words") or []:
            slot_text = word.get("text")
            parts = slot_text if isinstance(slot_text, list) else [slot_text]
            for part in parts:
                expected_labels.append(
                    {
                        "text": str(part),
                        "tone": int(word.get("tone")),
                        "consonant": int(word.get("phoneme_consonant")),
                        "vowel": int(word.get("phoneme_vowel")),
                    }
                )
        if len(labels) != len(text):
            b_bad_text.append(f"{item_id}:units={len(labels)} chars={len(text)}")
        for index, unit in enumerate(labels):
            if str(unit.get("text") or "") != text[index:index + 1]:
                b_bad_text.append(f"{item_id}:pos{index}")
            for dimension in DIMENSIONS_B:
                value = unit.get(dimension)
                if value not in (0, 1) or isinstance(value, bool):
                    b_bad_labels.append(f"{item_id}:{dimension}={value!r}")
            if not isinstance(unit.get("zero_initial"), bool):
                b_bad_labels.append(f"{item_id}:zero_initial")
            b_units += 1
            zero_units += bool(unit.get("zero_initial"))
        compact = [
            {key: unit.get(key) for key in ("text", "tone", "consonant", "vowel")}
            for unit in labels
        ]
        if compact != expected_labels:
            source_label_mismatch.append(item_id)
        if item.get("scores"):
            source_score_mismatch.append(f"{item_id}:unexpected_scores")
    add_check(checks, "B per-char labels in {0,1}", 0, len(b_bad_labels))
    add_check(checks, "B label texts == item chars", 0, len(b_bad_text))
    add_check(checks, "B labels match source word labels", 0,
              len(source_label_mismatch))
    add_check(checks, "B unit count", 19598, b_units)
    add_check(checks, "B zero-initial units", 2970, zero_units)
    add_check(checks, "B scores empty", 0, len(source_score_mismatch))
    # recompute zero_initial with pypinyin (prep's documented definition)
    zero_recomputed_ok = _verify_zero_initials(b_items)
    add_check(checks, "B zero_initial flags recomputed (pypinyin)", 0,
              zero_recomputed_ok["mismatches"],
              detail=f"checked={zero_recomputed_ok['checked']}")
    # C items: same items/scores from the source file
    c_bad_scores: list[str] = []
    c_units = 0
    for item in c_items:
        item_id = str(item.get("item_id"))
        source = learner_src.get(_bare_id(item_id))
        scores = item.get("scores") or {}
        if source is None:
            c_bad_scores.append(f"{item_id}:missing_source")
            continue
        if item.get("labels"):
            c_bad_scores.append(f"{item_id}:labels_not_empty")
        for dimension in DIMENSIONS_C:
            expected = source.get(dimension)
            actual = scores.get(dimension)
            if actual is None or not (1.0 <= float(actual) <= 5.0):
                c_bad_scores.append(f"{item_id}:{dimension}={actual!r}")
            elif abs(float(actual) - float(expected)) > 1e-9:
                c_bad_scores.append(f"{item_id}:{dimension}_mismatch")
        c_units += 1
    add_check(checks, "C scores in [1,5] and equal 3-rater means", 0,
              len(c_bad_scores))
    add_check(checks, "C learner items with scores", 1758, c_units)
    c_text_by_bare = {_bare_id(key): value.get("text") for key, value in c_by_id.items()}
    texts_identical = all(
        str(value.get("text")) == str(c_text_by_bare.get(_bare_id(key)))
        for key, value in b_by_id.items()
    )
    add_check(checks, "B/C texts identical", True, texts_identical,
              ok=texts_identical)
    native_bad: list[str] = []
    for item in native_items:
        item_id = str(item.get("item_id"))
        source = native_src.get(_bare_id(item_id))
        scores = item.get("scores") or {}
        if source is None:
            native_bad.append(f"{item_id}:missing_source")
            continue
        if not item.get("native"):
            native_bad.append(f"{item_id}:native_flag")
        if str(item.get("text")) != str(source.get("text")):
            native_bad.append(f"{item_id}:text")
        for dimension in DIMENSIONS_C:
            if scores.get(dimension) != 5.0:
                native_bad.append(f"{item_id}:{dimension}={scores.get(dimension)!r}")
    add_check(checks, "native items: text/5.0 scores/native flag", 0, len(native_bad))
    add_check(checks, "native scores all 5.0", True, not native_bad)
    ompal_audio = _check_audio(
        checks, "OMPAL learner", b_items, repo_root,
        expected_format=AUDIO_EXPECT["ompal"], check_audio=check_audio,
    )
    native_audio = _check_audio(
        checks, "OMPAL native", native_items, repo_root,
        expected_format=AUDIO_EXPECT["ompal"], check_audio=check_audio,
    )
    # audio provenance map covers every manifest wav
    provenance = json_load(data_dir / "audio_sources.json", {}) or {}
    all_audio_paths = {
        str(item.get("audio_path")) for item in (*a_items, *b_items, *native_items)
    }
    missing_provenance = sorted(all_audio_paths - set(provenance))
    add_check(checks, "audio_sources.json covers every manifest wav", 0,
              len(missing_provenance), detail=json.dumps(missing_provenance[:3]))
    data["ompal"] = {
        "raw_learner": len(learner_src),
        "raw_native": len(native_src),
        "anomaly_scan": {key: sorted(value) for key, value in anomaly.items()},
        "excluded": sorted(learner_excluded),
        "cleaned_learner": len(learner_src) - len(learner_excluded),
        "b_items": len(b_items),
        "c_items": len(c_items),
        "native_items": len(native_items),
        "learners": len(learners),
        "b_units": b_units,
        "zero_initial_units": zero_units,
        "zero_initial_recompute": zero_recomputed_ok,
        "source_label_mismatches": source_label_mismatch[:5],
        "c_bad_scores": c_bad_scores[:5],
        "native_bad": native_bad[:5],
        "audio": {"learner": ompal_audio, "native": native_audio},
        "audio_provenance_missing": missing_provenance[:5],
    }

    # -- clean-up + design deltas ------------------------------------------
    data["checks"] = checks
    data["n_checks"] = len(checks)
    data["n_failed"] = sum(1 for check in checks if not check["ok"])
    data["design_deltas"] = _design_deltas(data)
    return data


def _verify_zero_initials(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception:  # pragma: no cover - dependency present in this project
        return {"available": False, "checked": 0, "mismatches": 0}
    mismatches = 0
    checked = 0
    examples: list[str] = []
    for item in items:
        text = str(item.get("text") or "")
        labels = item.get("labels") or []
        syllables = lazy_pinyin(text, style=Style.TONE3, errors="default")
        if len(syllables) != len(text):
            syllables = [
                lazy_pinyin(char, style=Style.TONE3, errors="default")[0] for char in text
            ]
        for index, unit in enumerate(labels):
            if index >= len(syllables):
                mismatches += 1
                continue
            expected = normalize_pinyin_token(str(syllables[index])).initial == ""
            checked += 1
            if bool(unit.get("zero_initial")) != expected:
                mismatches += 1
                if len(examples) < 5:
                    examples.append(f"{item.get('item_id')}:{index}")
    return {"available": True, "checked": checked, "mismatches": mismatches,
            "examples": examples}


def _check_audio(
    checks: list[dict[str, Any]],
    label: str,
    items: Sequence[Mapping[str, Any]],
    repo_root: Path,
    *,
    expected_format: Mapping[str, Any],
    check_audio: bool,
) -> dict[str, Any]:
    audio: dict[str, Any] = {
        "checked": 0, "ok": 0, "missing": [], "format_violations": [],
        "duration_mismatches": [], "total_duration_s": 0.0, "formats": {},
    }
    if not check_audio:
        audio["skipped"] = True
        return audio
    formats: Counter[str] = Counter()
    for item in items:
        path = repo_root / str(item.get("audio_path") or "")
        probe = _probe_audio(path)
        audio["checked"] += 1
        if not probe.get("ok"):
            audio["missing"].append({"audio_path": item.get("audio_path"),
                                     "error": probe.get("error")})
            continue
        formats[f"{probe['samplerate']}Hz/{probe['channels']}ch/{probe['subtype']}"] += 1
        audio["total_duration_s"] += float(probe["duration_s"])
        if any(probe.get(key) != value for key, value in expected_format.items()):
            audio["format_violations"].append(
                {"audio_path": item.get("audio_path"),
                 "observed": {key: probe.get(key) for key in expected_format}}
            )
        recorded = float(item.get("duration_s") or 0.0)
        if abs(recorded - float(probe["duration_s"])) > 1e-6:
            audio["duration_mismatches"].append(
                {"audio_path": item.get("audio_path"), "recorded": recorded,
                 "actual": probe["duration_s"]}
            )
        audio["ok"] += 1
    audio["formats"] = dict(sorted(formats.items()))
    audio["total_duration_s"] = round(audio["total_duration_s"], 3)
    ok = (
        audio["checked"] > 0
        and not audio["missing"]
        and not audio["format_violations"]
        and not audio["duration_mismatches"]
        and len(formats) <= 1
    )
    add_check(
        checks,
        f"{label} audio files exist + format + duration",
        f"{len(items)} files, {expected_format['samplerate']}Hz/"
        f"{expected_format['channels']}ch/{expected_format['subtype']}, "
        "duration == manifest duration",
        {
            "checked": audio["checked"],
            "missing": len(audio["missing"]),
            "format_violations": len(audio["format_violations"]),
            "duration_mismatches": len(audio["duration_mismatches"]),
            "formats": audio["formats"],
        },
        ok=ok,
    )
    return audio


def _design_deltas(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    deltas.append(
        {
            "item": "B/C learner item count",
            "design": DESIGN_COUNTS["B"],
            "actual": data.get("ompal", {}).get("b_items"),
            "documented": "data/sentence/manifests/README.md §ompal/cleaning-count + "
                          "prep_summary.deviations",
        }
    )
    deltas.append(
        {
            "item": "AISHELL unalignable rows",
            "design": "2 known ids (design §3.2 note for OMPAL; AISHELL not specified)",
            "actual": data.get("aishell3_frame", {}).get("unalignable"),
            "documented": "prep_summary.deviations aishell3/unalignable",
        }
    )
    deltas.append(
        {
            "item": "AISHELL strata",
            "design": "approximate quotas",
            "actual": data.get("aishell3_manifest", {}).get("strata"),
            "documented": "exact quotas met",
        }
    )
    return deltas


# ---------------------------------------------------------------------------
# Reproducibility audit
# ---------------------------------------------------------------------------


def verify_reproducibility(
    repo_root: Path, run_dir: Path, manifest_dir: Path
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_manifest = json_load(run_dir / "run_manifest.json", {}) or {}
    result: dict[str, Any] = {"checks": checks, "run_manifest_present": bool(run_manifest)}
    config_path = repo_root / "configs" / "sentence.yaml"
    config_data: dict[str, Any] = {}
    try:
        import yaml

        base = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text(encoding="utf-8")) or {}
        overlay = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_data = dict(base)
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(config_data.get(key), dict):
                config_data[key] = {**config_data[key], **value}
            else:
                config_data[key] = value
    except Exception as exc:  # noqa: BLE001 - report, do not abort
        config_data = {}
        result["config_load_error"] = str(exc)
    expected = {
        "model": str((config_data.get("model") or {}).get("model"))
        if config_data else None,
        "prompt_version": None,
        "pricing": {
            "input_per_mtok": 0.8,
            "cache_hit_input_per_mtok": 0.1,
            "output_per_mtok": 2.7,
            "audio_tokens_per_second": 7.0,
        },
    }
    try:
        from pinyin_eval.sentence.prompts import PROMPT_VERSION, prompt_fingerprints

        expected["prompt_version"] = PROMPT_VERSION
        fingerprints = prompt_fingerprints()
    except Exception:  # pragma: no cover
        fingerprints = {}
    add_check(checks, "run_manifest.json present", True, bool(run_manifest))
    required_fields = (
        "run_id", "created_at", "model", "provider", "prompt_version",
        "prompt_fingerprints", "config_sha256", "manifests", "seed",
        "pricing", "budget_cny", "n_planned", "git_commit",
    )
    missing_fields = [
        field for field in required_fields
        if run_manifest.get(field) in (None, "", {}, [])
        and field not in ("manifests",)
    ]
    add_check(checks, "run_manifest required fields non-empty", [], missing_fields)
    add_check(checks, "model id matches config", expected["model"],
              run_manifest.get("model"))
    add_check(checks, "prompt_version matches code", expected["prompt_version"],
              run_manifest.get("prompt_version"))
    add_check(checks, "prompt fingerprints match code", fingerprints,
              run_manifest.get("prompt_fingerprints"))
    pricing = run_manifest.get("pricing") or {}
    for key, value in expected["pricing"].items():
        add_check(checks, f"pricing.{key}", value, pricing.get(key))
    seed_expected = (config_data.get("project") or {}).get("seed")
    add_check(checks, "seed matches config project.seed", seed_expected,
              run_manifest.get("seed"))
    budget_expected = float((config_data.get("sentence") or {}).get("budget_cny") or 0.0)
    add_check(checks, "budget_cny matches config", budget_expected,
              run_manifest.get("budget_cny"))
    # config sha256 == sha256 of the merged config dict (project Config contract)
    try:
        from pinyin_eval.config import Config

        recomputed = Config.load(config_path).config_sha256()
    except Exception as exc:  # noqa: BLE001
        recomputed = f"error: {exc}"
    add_check(checks, "config_sha256 matches merged config", run_manifest.get("config_sha256"),
              recomputed)
    # item manifest hashes
    items_manifests = run_manifest.get("items_manifests") or {}
    for task, filename in (("A", "aishell3_items.jsonl"), ("B", "ompal_b_items.jsonl"),
                           ("C", "ompal_c_items.jsonl")):
        recorded = (items_manifests.get(task) or {}).get("sha256")
        actual = sha256_file(manifest_dir / filename)
        add_check(checks, f"manifests.{task} sha256", recorded, actual)
    native_file = manifest_dir / "ompal_native_items.jsonl"
    if native_file.exists():
        result["native_manifest_sha256"] = sha256_file(native_file)
        result["native_manifest_recorded"] = False
    # git commit
    git_commit = run_manifest.get("git_commit")
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
        tracked_sentence = subprocess.run(
            ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only",
             str(git_commit or "HEAD"), "--", "src/pinyin_eval/sentence"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        head, status, tracked_sentence = None, "", ""
    add_check(checks, "git_commit == current HEAD", head, git_commit)
    result["git"] = {
        "manifest_commit": git_commit,
        "head": head,
        "dirty_files": [line for line in status.splitlines() if line.strip()][:20],
        "n_dirty": len([line for line in status.splitlines() if line.strip()]),
        "sentence_package_tracked_at_commit": bool(tracked_sentence),
    }
    if not tracked_sentence:
        result["warnings"] = [
            "run_manifest.git_commit does not contain src/pinyin_eval/sentence "
            "(the sentence code was uncommitted when the run started); "
            "reproducibility relies on the working tree, not the commit hash"
        ]
    # planned call count
    n_planned = len(load_jsonl(manifest_dir / "aishell3_items.jsonl")) * 2
    n_planned += len(load_jsonl(manifest_dir / "ompal_b_items.jsonl")) * 2
    n_planned += len(load_jsonl(manifest_dir / "ompal_c_items.jsonl")) * 2
    if native_file.exists():
        n_planned += len(load_jsonl(native_file)) * 2
    add_check(checks, "n_planned == manifests x 2 conditions", n_planned,
              run_manifest.get("n_planned"))
    result["n_planned_independent"] = n_planned
    result["n_checks"] = len(checks)
    result["n_failed"] = sum(1 for check in checks if not check["ok"])
    return result


# ---------------------------------------------------------------------------
# Run artifact recomputation + acceptance
# ---------------------------------------------------------------------------


def _load_items(manifest_dir: Path) -> dict[str, list[dict[str, Any]]]:
    items: dict[str, list[dict[str, Any]]] = {}
    for task, filename in (("A", "aishell3_items.jsonl"), ("B", "ompal_b_items.jsonl"),
                           ("C", "ompal_c_items.jsonl")):
        items[task] = load_jsonl(manifest_dir / filename)
    native = load_jsonl(manifest_dir / "ompal_native_items.jsonl")
    items["C"] = list(items["C"]) + native
    return items


def verify_run(
    run_dir: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    run_manifest = json_load(run_dir / "run_manifest.json", {}) or {}
    conditions = run_manifest.get("conditions") or {
        "C1": {"thinking": False, "max_tokens": 256, "max_tokens_by_task": {}},
        "C2": {"thinking": True, "max_tokens": 1024, "max_tokens_by_task": {}},
    }
    items = _load_items(manifest_dir)
    item_by_id: dict[str, dict[str, Any]] = {
        item_id: item
        for task_items in items.values()
        for item_id, item in ((str(row.get("item_id")), row) for row in task_items)
    }
    records = load_jsonl(run_dir / "generations.jsonl")
    result: dict[str, Any] = {}
    result["records"] = {
        "n_rows": len(records),
        "n_unique_request_ids": len({str(row.get("request_id")) for row in records}),
    }
    # final record per (task, condition, item) - same resume semantics as the scorer
    final: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_keys = Counter()
    for record in records:
        key = (str(record.get("task") or ""), str(record.get("condition") or ""),
               str(record.get("item_id") or ""))
        if key in final:
            duplicate_keys[key] += 1
        final[key] = record
    result["records"]["duplicate_request_keys"] = len(duplicate_keys)
    result["records"]["duplicate_examples"] = [
        "|".join(key) for key in list(duplicate_keys)[:5]
    ]
    # expected request set
    expected_keys: set[tuple[str, str, str]] = set()
    for task in TASKS:
        for condition in CONDITIONS:
            for item in items.get(task, []):
                expected_keys.add((task, condition, str(item.get("item_id"))))
    actual_keys = set(final)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    result["records"]["n_expected"] = len(expected_keys)
    result["records"]["n_present"] = len(actual_keys)
    result["records"]["n_missing"] = len(missing_keys)
    result["records"]["missing_examples"] = ["|".join(key) for key in missing_keys[:10]]
    result["records"]["n_extra"] = len(extra_keys)
    result["records"]["extra_examples"] = ["|".join(key) for key in extra_keys[:10]]
    # envelope audit
    envelope = {
        "missing_usage": [], "missing_finish_reason": [], "missing_timestamp": [],
        "missing_prompt_sha256": [], "model_mismatch": [], "prompt_version_mismatch": [],
        "max_tokens_mismatch": [], "thinking_mismatch": [], "audio_path_mismatch": [],
        "error_records": [],
    }
    try:
        from pinyin_eval.sentence.prompts import (
            TEMPLATE_A,
            TEMPLATE_B,
            TEMPLATE_C,
            prompt_sha256,
        )
    except Exception:  # pragma: no cover
        prompt_sha256 = None  # type: ignore[assignment]
        TEMPLATE_A = TEMPLATE_B = TEMPLATE_C = ""  # type: ignore[assignment]
    for (task, condition, item_id), record in final.items():
        item = item_by_id.get(item_id)
        if record.get("error"):
            envelope["error_records"].append(
                {"request": "|".join((task, condition, item_id)),
                 "error": str(record.get("error"))[:200]}
            )
        else:
            if not record.get("usage"):
                envelope["missing_usage"].append("|".join((task, condition, item_id)))
            if not record.get("finish_reason"):
                envelope["missing_finish_reason"].append("|".join((task, condition, item_id)))
            if not record.get("timestamp"):
                envelope["missing_timestamp"].append("|".join((task, condition, item_id)))
            if not record.get("prompt_sha256"):
                envelope["missing_prompt_sha256"].append(
                    "|".join((task, condition, item_id))
                )
        if str(record.get("model")) != str(run_manifest.get("model") or "qwen3.8-omni-flash"):
            envelope["model_mismatch"].append("|".join((task, condition, item_id)))
        spec = conditions.get(condition) or {}
        expected_max = int((spec.get("max_tokens_by_task") or {}).get(
            task, spec.get("max_tokens", 0)))
        if expected_max and int(record.get("max_tokens") or 0) != expected_max:
            envelope["max_tokens_mismatch"].append("|".join((task, condition, item_id)))
        if bool(record.get("thinking")) != bool(spec.get("thinking")):
            envelope["thinking_mismatch"].append("|".join((task, condition, item_id)))
        if item is not None and str(record.get("audio_path") or "") != str(
            item.get("audio_path") or ""
        ):
            envelope["audio_path_mismatch"].append(
                "|".join((task, condition, item_id))
            )
        if prompt_sha256 is not None and item is not None:
            if task == "A":
                expected_sha = prompt_sha256("", TEMPLATE_A.format())
            elif task == "B":
                expected_sha = prompt_sha256("", TEMPLATE_B.format(text=item.get("text")))
            else:
                expected_sha = prompt_sha256("", TEMPLATE_C.format(text=item.get("text")))
            if str(record.get("prompt_sha256")) != expected_sha:
                envelope["prompt_version_mismatch"].append(
                    "|".join((task, condition, item_id))
                )
    result["envelope"] = {
        key: (len(value) if isinstance(value, list) else value)
        for key, value in envelope.items()
    }
    result["envelope"]["examples"] = {
        key: value[:3] for key, value in envelope.items() if isinstance(value, list) and value
    }
    add_check(checks, "generation records for every planned request",
              result["records"]["n_expected"], result["records"]["n_present"])
    add_check(checks, "usage/finish_reason/timestamp/prompt_sha256 recorded", [],
              {
              "usage": result["envelope"]["missing_usage"],
              "finish_reason": result["envelope"]["missing_finish_reason"],
              "timestamp": result["envelope"]["missing_timestamp"],
              "prompt_sha256": result["envelope"]["missing_prompt_sha256"],
              },
              ok=not any(result["envelope"][key] for key in (
                  "missing_usage", "missing_finish_reason", "missing_timestamp",
                  "missing_prompt_sha256")))
    add_check(checks, "per-record model/prompt_version/max_tokens/thinking/audio_path",
              [], {
                  "model_mismatch": result["envelope"]["model_mismatch"],
                  "prompt_sha256_mismatch": result["envelope"]["prompt_version_mismatch"],
                  "max_tokens_mismatch": result["envelope"]["max_tokens_mismatch"],
                  "thinking_mismatch": result["envelope"]["thinking_mismatch"],
                  "audio_path_mismatch": result["envelope"]["audio_path_mismatch"],
              },
              ok=not any(result["envelope"][key] for key in (
                  "model_mismatch", "prompt_version_mismatch", "max_tokens_mismatch",
                  "thinking_mismatch", "audio_path_mismatch")))

    # -- per-record parsing + scoring --------------------------------------
    scored: dict[str, dict[str, list[dict[str, Any]]]] = {
        task: {condition: [] for condition in CONDITIONS} for task in TASKS
    }
    status_counts: dict[str, dict[str, Counter]] = {
        task: {condition: Counter() for condition in CONDITIONS} for task in TASKS
    }
    failure_examples: list[dict[str, Any]] = []
    wrapper_unwrapped: Counter[tuple[str, str]] = Counter()
    for task in TASKS:
        for condition in CONDITIONS:
            for item in items.get(task, []):
                key = (task, condition, str(item.get("item_id")))
                item_id = str(item.get("item_id"))
                record = final.get(key)
                if record is None:
                    status_counts[task][condition]["missing"] += 1
                    continue
                parsed = parse_record(task, record)
                status = str(parsed.get("parse_status") or "parse_error")
                if task in ("A", "B") and status == "ok" and not record.get("error"):
                    strict = parse_record(task, record, unwrap_wrapper=False)
                    if strict.get("parse_status") != "ok":
                        wrapper_unwrapped[(task, condition)] += 1
                status_counts[task][condition][status] += 1
                parse_ok = status == "ok"
                if not parse_ok and len(failure_examples) < 40:
                    failure_examples.append(
                        {
                            "request": "|".join(key),
                            "status": status,
                            "detail": str(parsed.get("detail") or parsed.get("error") or "")[:160],
                            "finish_reason": record.get("finish_reason"),
                        }
                    )
                if task == "A":
                    # A always emits one row per item (parse failures keep n_gold)
                    row = score_a_item(item, parsed)
                    row["parse_ok"] = parse_ok
                    row["flags"] = {
                        name: bool((item.get("flags") or {}).get(name))
                        for name in ("neutral", "yibu", "erhua")
                    }
                    row["stable"] = not any(row["flags"].values())
                    scored["A"][condition].append(row)
                elif task == "B":
                    if not parse_ok:
                        continue
                    rows = score_b_item(item, parsed)
                    for row in rows:
                        row["item_id"] = item_id
                    scored["B"][condition].extend(rows)
                else:
                    # C emits one row per item (parse failures keep gold, pred None)
                    row = score_c_item(item, parsed)
                    row["item_id"] = item_id
                    row["parse_ok"] = parse_ok
                    scored["C"][condition].append(row)
    result["parse"] = {
        task: {
            condition: {
                "n_calls": sum(status_counts[task][condition].values()),
                "status_counts": dict(status_counts[task][condition]),
                "n_ok": status_counts[task][condition].get("ok", 0),
                "n_errors": status_counts[task][condition].get("api_error", 0),
                "n_parse_failures": sum(
                    count for status, count in status_counts[task][condition].items()
                    if status not in ("ok", "missing")
                ),
                "n_missing": status_counts[task][condition].get("missing", 0),
                "parse_success_rate": ratio(
                    status_counts[task][condition].get("ok", 0),
                    sum(status_counts[task][condition].values()),
                ),
            }
            for condition in CONDITIONS
        }
        for task in TASKS
    }
    result["parse"]["failure_examples"] = failure_examples
    result["parse"]["wrapper_unwrapped"] = {
        f"{task}|{condition}": count
        for (task, condition), count in sorted(wrapper_unwrapped.items())
    }
    result["parse"]["parse_policy"] = (
        "one-element array wrappers ([{...}]) are unwrapped for A/B (mirrors the "
        "documented parser); a strict re-parse count is reported in wrapper_unwrapped"
    )

    # -- task metrics -------------------------------------------------------
    metrics: dict[str, dict[str, Any]] = {"A": {}, "B": {}, "C": {}}
    for condition in CONDITIONS:
        a_block = aggregate_a(scored["A"][condition])
        a_block["subsets"] = {
            subset: aggregate_a_subset(scored["A"][condition], subset)
            for subset in A_SUBSETS
        }
        metrics["A"][condition] = a_block
        metrics["B"][condition] = aggregate_b(scored["B"][condition])
        metrics["C"][condition] = aggregate_c(scored["C"][condition])
    result["metrics"] = metrics

    # -- cost ---------------------------------------------------------------
    cost_by_key: dict[str, dict[str, Any]] = {}
    cost_floats = defaultdict(float)
    for record in records:
        task = str(record.get("task") or "")
        condition = str(record.get("condition") or "")
        if task not in TASKS or condition not in CONDITIONS:
            continue
        usage = usage_of(record)
        cost = record_cost(usage)
        key = f"{task}|{condition}"
        entry = cost_by_key.setdefault(
            key,
            {
                "n_calls": 0, "n_errors": 0, "n_retries": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "reasoning_tokens": 0, "cost_cny": 0.0,
                "finish_reasons": Counter(),
            },
        )
        entry["n_calls"] += 1
        entry["n_errors"] += 0 if record.get("error") is None else 1
        entry["n_retries"] += int(record.get("retries") or 0)
        entry["prompt_tokens"] += usage["prompt_tokens"]
        entry["completion_tokens"] += usage["completion_tokens"]
        entry["reasoning_tokens"] += usage["reasoning_tokens"]
        entry["cost_cny"] += cost["total_cost_cny"]
        entry["finish_reasons"][str(record.get("finish_reason"))] += 1
        cost_floats["prompt_tokens"] += usage["prompt_tokens"]
        cost_floats["completion_tokens"] += usage["completion_tokens"]
        cost_floats["cached_tokens"] += usage["cached_tokens"]
        cost_floats["audio_tokens"] += usage["audio_tokens"]
        cost_floats["reasoning_tokens"] += usage["reasoning_tokens"]
        cost_floats["input_cost_cny"] += cost["input_cost_cny"]
        cost_floats["output_cost_cny"] += cost["output_cost_cny"]
        cost_floats["cost_cny"] += cost["total_cost_cny"]
    for entry in cost_by_key.values():
        entry["cost_cny"] = round(entry["cost_cny"], 6)
        entry["finish_reasons"] = dict(entry["finish_reasons"])
    cost_totals_dict = {
        key: (round(value, 6) if isinstance(value, float) else value)
        for key, value in cost_floats.items()
    }
    cost_totals_dict["n_calls"] = sum(entry["n_calls"] for entry in cost_by_key.values())
    cost_totals_dict["n_errors"] = sum(entry["n_errors"] for entry in cost_by_key.values())
    cost_totals_dict["n_retries"] = sum(entry["n_retries"] for entry in cost_by_key.values())
    result["cost"] = {
        "per_task_condition": dict(sorted(cost_by_key.items())),
        "totals": cost_totals_dict,
        "pricing_used": PRICING,
        "cost_summary": json_load(run_dir / "cost_summary.json"),
    }
    cost_summary = result["cost"]["cost_summary"]
    if cost_summary:
        reported_total = (cost_summary.get("totals") or {}).get("cost_cny")
        add_check(checks, "usage-based cost matches cost_summary.json total",
                  reported_total, round(cost_totals_dict["cost_cny"], 6))
        add_check(checks, "usage-based n_calls matches cost_summary.json",
                  (cost_summary.get("totals") or {}).get("n_calls"),
                  cost_totals_dict["n_calls"])
    result["retries"] = {
        "n_records_with_retries": sum(
            1 for record in final.values() if int(record.get("retries") or 0) > 0
        ),
        "retries_total": cost_totals_dict["n_retries"],
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
        "max": round(max(latencies), 3) if latencies else None,
        "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3)
        if latencies else None,
    }
    finish_reasons = Counter()
    for record in final.values():
        finish_reasons[str(record.get("finish_reason"))] += 1
    result["finish_reasons"] = dict(finish_reasons)

    # -- acceptance ---------------------------------------------------------
    acceptance = _acceptance(result, cost_totals_dict["cost_cny"], expected_keys)
    result["acceptance"] = acceptance
    add_check(checks, "acceptance gates (design §10.2)", True, acceptance["passed"],
              detail=json.dumps(
                  {
                      "parse_success": {task: value["rate"]
                                        for task, value in acceptance["parse_success"].items()},
                      "alignment_coverage": {
                          task: value["value"]
                          for task, value in acceptance["alignment_coverage"].items()},
                      "cost_cny": acceptance["cost_cny"]["value"],
                  },
                  ensure_ascii=False,
              ))
    result["checks"] = checks
    result["n_checks"] = len(checks)
    result["n_failed"] = sum(1 for check in checks if not check["ok"])
    return result


def _acceptance(
    run: Mapping[str, Any],
    total_cost: float | None,
    expected_keys: set[tuple[str, str, str]],
) -> dict[str, Any]:
    """Design §10.2 gates recomputed independently (C1 condition, main analysis)."""
    parse = run.get("parse") or {}
    metrics = run.get("metrics") or {}
    parse_success: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        n_calls = sum(
            (parse.get(task) or {}).get(condition, {}).get("n_calls", 0)
            for condition in CONDITIONS
        )
        n_ok = sum(
            (parse.get(task) or {}).get(condition, {}).get("n_ok", 0)
            for condition in CONDITIONS
        )
        n_missing = sum(
            (parse.get(task) or {}).get(condition, {}).get("n_missing", 0)
            for condition in CONDITIONS
        )
        # status_counts already include "missing" planned calls; they count as
        # failures because they were never generated/scored
        planned = n_calls
        rate = (n_ok / planned) if planned else None
        parse_success[task] = {
            "rate": rate,
            "n_calls": planned,
            "n_scored_calls": n_calls,
            "n_ok": n_ok,
            "n_failures": planned - n_ok,
            "n_missing": n_missing,
            "threshold": ACCEPT_PARSE_MIN,
            "passed": rate is not None and rate >= ACCEPT_PARSE_MIN,
        }
    alignment: dict[str, dict[str, Any]] = {}
    a_headline = ((metrics.get("A") or {}).get("C1") or {}).get("headline") or {}
    b_c1 = (metrics.get("B") or {}).get("C1") or {}
    c_c1 = (metrics.get("C") or {}).get("C1") or {}
    values = {
        "A": a_headline.get("alignment_coverage"),
        "B": b_c1.get("alignment_coverage"),
        "C": c_c1.get("alignment_coverage"),
    }
    for task, value in values.items():
        alignment[task] = {
            "value": value,
            "threshold": ACCEPT_COVERAGE_MIN,
            "passed": value is not None and value >= ACCEPT_COVERAGE_MIN,
        }
    cost_block = {
        "value": total_cost,
        "threshold": ACCEPT_COST_MAX,
        "passed": total_cost is not None and total_cost <= ACCEPT_COST_MAX,
    }
    passed = (
        all(entry["passed"] for entry in parse_success.values())
        and all(entry["passed"] for entry in alignment.values())
        and cost_block["passed"]
    )
    return {
        "parse_success": parse_success,
        "alignment_coverage": alignment,
        "cost_cny": cost_block,
        "passed": passed,
        "expected_requests": len(expected_keys),
    }


# ---------------------------------------------------------------------------
# Comparison against the pipeline's reported artifacts
# ---------------------------------------------------------------------------


def _metric_lookup(metrics: Mapping[str, Any]) -> dict[tuple[str, str, str, str, str, str], Any]:
    """Flatten my metrics into the ``metrics.csv`` long-format key space."""
    lookup: dict[tuple[str, str, str, str, str, str], Any] = {}
    dimensions = {
        "A": {
            "joint_accuracy": "joint",
            "initial_accuracy": "initial",
            "final_accuracy": "final",
            "tone_accuracy": "tone",
            "cer_tone": "",
            "cer_stripped": "",
            "sentence_exact_rate": "",
            "insertion_rate": "",
            "deletion_rate": "",
            "alignment_coverage": "",
        },
    }
    for condition in CONDITIONS:
        a_block = ((metrics.get("A") or {}).get(condition) or {})
        headline = a_block.get("headline") or {}
        for metric in A_METRIC_ORDER:
            lookup[("A", condition, "headline", "",
                    dimensions["A"].get(metric, ""), metric)] = headline.get(metric)
        for subset in A_SUBSETS:
            subset_block = (a_block.get("subsets") or {}).get(subset) or {}
            for metric in A_METRIC_ORDER:
                lookup[("A", condition, "subset", subset,
                        dimensions["A"].get(metric, ""), metric)] = subset_block.get(metric)
        n1 = a_block.get("n1") or {}
        for metric, value in n1.items():
            if metric == "delta_vs_n0":
                continue
            lookup[("A", condition, "n1_diagnostic", "", "", metric)] = value
        b_block = ((metrics.get("B") or {}).get(condition) or {})
        lookup[("B", condition, "overall", "", "", "alignment_coverage")] = b_block.get(
            "alignment_coverage"
        )
        for dimension in DIMENSIONS_B:
            stats = b_block.get(dimension) or {}
            for metric in ("macro_f1", "f1_error", "f1_ok", "fpr", "fnr", "accuracy"):
                lookup[("B", condition, "dimension", "", dimension, metric)] = stats.get(metric)
            for count_name in ("tp", "fp", "fn", "tn"):
                lookup[("B", condition, "confusion", "", dimension, count_name)] = (
                    stats.get("confusion") or {}
                ).get(count_name)
        zero = b_block.get("zero_initial_consonant") or {}
        for metric in ("macro_f1", "f1_error", "fpr", "fnr", "accuracy"):
            lookup[("B", condition, "zero_initial_consonant", "", "consonant", metric)] = (
                zero.get(metric)
            )
        c_block = ((metrics.get("C") or {}).get(condition) or {})
        lookup[("C", condition, "overall", "", "", "alignment_coverage")] = c_block.get(
            "alignment_coverage"
        )
        for dimension in (*DIMENSIONS_C, "mean_dimension"):
            stats = c_block.get(dimension) or {}
            for metric in ("pearson", "spearman", "mae", "rmse"):
                lookup[("C", condition, "dimension", "", dimension, metric)] = stats.get(
                    metric
                )
        native = c_block.get("native_anchor") or {}
        lookup[("C", condition, "native_anchor", "", "", "n")] = native.get("n")
        lookup[("C", condition, "native_anchor", "", "", "mean_gold")] = native.get(
            "mean_gold"
        )
    return lookup


def compare_reported(run_dir: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    """Compare independently recomputed numbers against the run's own artifacts."""
    comparison: dict[str, Any] = {
        "metrics_csv": {"available": False},
        "acceptance_json": {"available": False},
        "report_summary_json": {"available": False},
        "cost_summary_json": {"available": False},
        "discrepancies": [],
    }
    lookup = _metric_lookup(run.get("metrics") or {})
    # -- metrics.csv --------------------------------------------------------
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        reported: dict[tuple[str, str, str, str, str, str], Any] = {}
        with metrics_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (
                    row.get("task", ""),
                    row.get("condition", ""),
                    row.get("scope", ""),
                    row.get("subset", ""),
                    row.get("dimension", ""),
                    row.get("metric", ""),
                )
                reported[key] = row.get("estimate")
        mismatches: list[dict[str, Any]] = []
        missing: list[str] = []
        compared = 0
        for key, independent in lookup.items():
            if key not in reported:
                missing.append("|".join(key))
                continue
            reported_value = _parse_csv_number(reported[key])
            compared += 1
            if reported_value is None and independent is None:
                continue
            if reported_value is None or independent is None:
                mismatches.append(
                    {
                        "key": "|".join(key),
                        "reported": reported_value,
                        "independent": independent,
                        "delta_pp": None,
                        "kind": "null_mismatch",
                    }
                )
                continue
            if key[2] == "confusion" or key[5] in ("tp", "fp", "fn", "tn", "n"):
                if int(round(float(reported_value))) != int(round(float(independent))):
                    mismatches.append(
                        {
                            "key": "|".join(key),
                            "reported": reported_value,
                            "independent": independent,
                            "delta": float(independent) - float(reported_value),
                            "kind": "count",
                        }
                    )
                continue
            delta = float(independent) - float(reported_value)
            if abs(delta) > FLOAT_TOL:
                mismatches.append(
                    {
                        "key": "|".join(key),
                        "reported": reported_value,
                        "independent": independent,
                        "delta_pp": round(delta * 100, 4),
                        "kind": "value",
                    }
                )
        comparison["metrics_csv"] = {
            "available": True,
            "n_rows": len(reported),
            "n_compared": compared,
            "n_mismatch": len(mismatches),
            "n_missing_in_csv": len(missing),
            "missing_examples": missing[:10],
            "mismatches": mismatches[:40],
        }
        comparison["discrepancies"].extend(
            {"source": "metrics.csv", **item} for item in mismatches
        )
    # -- acceptance.json ----------------------------------------------------
    acceptance_path = run_dir / "acceptance.json"
    if acceptance_path.exists():
        reported = json_load(acceptance_path, {}) or {}
        independent = run.get("acceptance") or {}
        mismatches: list[dict[str, Any]] = []
        for task in TASKS:
            rep = ((reported.get("parse_success") or {}).get(task) or {})
            ind = ((independent.get("parse_success") or {}).get(task) or {})
            if rep.get("rate") is not None and ind.get("rate") is not None:
                if abs(float(rep["rate"]) - float(ind["rate"])) > 0.0005:
                    mismatches.append(
                        {
                            "key": f"parse_success.{task}",
                            "reported": rep["rate"],
                            "independent": ind["rate"],
                            "delta_pp": round((ind["rate"] - rep["rate"]) * 100, 4),
                        }
                    )
            if bool(rep.get("passed")) != bool(ind.get("passed")):
                mismatches.append(
                    {"key": f"parse_success.{task}.passed",
                     "reported": rep.get("passed"), "independent": ind.get("passed")}
                )
            rep_cov = ((reported.get("alignment_coverage") or {}).get(task) or {})
            ind_cov = ((independent.get("alignment_coverage") or {}).get(task) or {})
            if rep_cov.get("value") is not None and ind_cov.get("value") is not None:
                if abs(float(rep_cov["value"]) - float(ind_cov["value"])) > 0.0005:
                    mismatches.append(
                        {
                            "key": f"alignment_coverage.{task}",
                            "reported": rep_cov["value"],
                            "independent": ind_cov["value"],
                            "delta_pp": round(
                                (ind_cov["value"] - rep_cov["value"]) * 100, 4
                            ),
                        }
                    )
        rep_cost = (reported.get("cost_cny") or {}).get("value")
        ind_cost = (independent.get("cost_cny") or {}).get("value")
        if rep_cost is not None and ind_cost is not None and abs(rep_cost - ind_cost) > 0.01:
            mismatches.append(
                {"key": "cost_cny", "reported": rep_cost, "independent": ind_cost,
                 "delta": round(ind_cost - rep_cost, 6)}
            )
        if bool(reported.get("passed")) != bool(independent.get("passed")):
            mismatches.append(
                {"key": "passed", "reported": reported.get("passed"),
                 "independent": independent.get("passed")}
            )
        comparison["acceptance_json"] = {
            "available": True,
            "n_mismatch": len(mismatches),
            "mismatches": mismatches[:20],
        }
        comparison["discrepancies"].extend(
            {"source": "acceptance.json", **item} for item in mismatches
        )
    # -- report_summary.json ------------------------------------------------
    summary_path = run_dir / "report_summary.json"
    if summary_path.exists():
        reported = json_load(summary_path, {}) or {}
        summary_mismatches: list[dict[str, Any]] = []
        c1_metrics = run.get("metrics") or {}
        a_c1 = (c1_metrics.get("A") or {}).get("C1") or {}
        b_c1 = (c1_metrics.get("B") or {}).get("C1") or {}
        c_c1 = (c1_metrics.get("C") or {}).get("C1") or {}
        expected_values: dict[str, Any] = {
            "headline.A.n_items": a_c1.get("n_items"),
            "headline.A.n_parse_ok": a_c1.get("n_parse_ok"),
            "headline.A.joint_accuracy": (a_c1.get("headline") or {}).get("joint_accuracy"),
            "headline.A.tone_accuracy": (a_c1.get("headline") or {}).get("tone_accuracy"),
            "headline.A.sentence_exact_rate": (a_c1.get("headline") or {}).get(
                "sentence_exact_rate"
            ),
            "headline.A.cer_tone": (a_c1.get("headline") or {}).get("cer_tone"),
            "headline.B.n_items": b_c1.get("n_items"),
            "headline.C.n_items": c_c1.get("n_items"),
            "headline.C.native_anchor.n": (c_c1.get("native_anchor") or {}).get("n"),
            "headline.C.native_anchor.mean_gold": (c_c1.get("native_anchor") or {}).get(
                "mean_gold"
            ),
            "headline.C.native_anchor.mean_pred": (c_c1.get("native_anchor") or {}).get(
                "mean_pred"
            ),
        }
        for dimension in DIMENSIONS_B:
            expected_values[f"headline.B.{dimension}_macro_f1"] = (
                b_c1.get(dimension) or {}
            ).get("macro_f1")
        for dimension in DIMENSIONS_C:
            expected_values[f"headline.C.{dimension}_pearson"] = (
                c_c1.get(dimension) or {}
            ).get("pearson")
        independent_acceptance = run.get("acceptance") or {}
        for task in TASKS:
            expected_values[f"acceptance.parse_success.{task}"] = (
                (independent_acceptance.get("parse_success") or {}).get(task) or {}
            ).get("rate")
            expected_values[f"acceptance.alignment_coverage.{task}"] = (
                (independent_acceptance.get("alignment_coverage") or {}).get(task) or {}
            ).get("value")
        expected_values["acceptance.cost_cny"] = (
            independent_acceptance.get("cost_cny") or {}
        ).get("value")
        expected_values["acceptance.passed"] = independent_acceptance.get("passed")
        usage_totals = (run.get("cost") or {}).get("totals") or {}
        expected_values["usage_totals.cost_cny"] = usage_totals.get("cost_cny")
        for path, independent in expected_values.items():
            reported_value = _dig(reported, path)
            if reported_value is None and independent is None:
                continue
            if isinstance(independent, bool) or isinstance(reported_value, bool):
                if bool(reported_value) != bool(independent):
                    summary_mismatches.append(
                        {"key": path, "reported": reported_value, "independent": independent}
                    )
                continue
            if reported_value is None or independent is None:
                summary_mismatches.append(
                    {"key": path, "reported": reported_value, "independent": independent}
                )
                continue
            if isinstance(independent, int) and not isinstance(independent, bool):
                if int(reported_value) != independent:
                    summary_mismatches.append(
                        {"key": path, "reported": reported_value, "independent": independent}
                    )
                continue
            delta = float(independent) - float(reported_value)
            if abs(delta) > FLOAT_TOL and not (
                path.startswith("usage_totals") and abs(delta) <= 0.01
            ):
                summary_mismatches.append(
                    {
                        "key": path,
                        "reported": reported_value,
                        "independent": independent,
                        "delta_pp": round(delta * 100, 4),
                    }
                )
        comparison["report_summary_json"] = {
            "available": True,
            "n_compared": len(expected_values),
            "n_mismatch": len(summary_mismatches),
            "mismatches": summary_mismatches[:20],
        }
        comparison["discrepancies"].extend(
            {"source": "report_summary.json", **item} for item in summary_mismatches
        )
    # -- cost_summary.json ---------------------------------------------------
    cost_summary = json_load(run_dir / "cost_summary.json")
    if isinstance(cost_summary, Mapping):
        reported_total = (cost_summary.get("totals") or {}).get("cost_cny")
        independent_total = (run.get("cost") or {}).get("totals", {}).get("cost_cny")
        mismatch: dict[str, Any] | None = None
        if reported_total is not None and independent_total is not None:
            if abs(float(reported_total) - float(independent_total)) > 0.01:
                mismatch = {
                    "key": "totals.cost_cny",
                    "reported": reported_total,
                    "independent": round(float(independent_total), 6),
                    "delta": round(float(independent_total) - float(reported_total), 6),
                }
        per_condition_mismatch: list[dict[str, Any]] = []
        reported_per = cost_summary.get("per_task_condition") or {}
        independent_per = (run.get("cost") or {}).get("per_task_condition") or {}
        for key, independent in independent_per.items():
            reported_entry = reported_per.get(key) or {}
            for field in ("n_calls", "n_errors", "n_retries"):
                if int(reported_entry.get(field) or 0) != int(independent.get(field) or 0):
                    per_condition_mismatch.append(
                        {
                            "key": f"{key}.{field}",
                            "reported": reported_entry.get(field),
                            "independent": independent.get(field),
                        }
                    )
            reported_cost = reported_entry.get("cost_cny")
            if reported_cost is not None and abs(
                float(reported_cost) - float(independent.get("cost_cny") or 0.0)
            ) > 0.01:
                per_condition_mismatch.append(
                    {
                        "key": f"{key}.cost_cny",
                        "reported": reported_cost,
                        "independent": independent.get("cost_cny"),
                    }
                )
        comparison["cost_summary_json"] = {
            "available": True,
            "total_mismatch": mismatch,
            "per_condition_mismatches": per_condition_mismatch[:20],
        }
        if mismatch:
            comparison["discrepancies"].append({"source": "cost_summary.json", **mismatch})
        comparison["discrepancies"].extend(
            {"source": "cost_summary.json", **item} for item in per_condition_mismatch
        )
    comparison["n_discrepancies"] = len(comparison["discrepancies"])
    return comparison


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_report(
    run_dir: Path,
    *,
    manifest_dir: Path | None = None,
    repo_root: Path = REPO_ROOT,
    check_audio: bool = True,
    check_data: bool = True,
) -> dict[str, Any]:
    manifest_dir = manifest_dir or (repo_root / "data" / "sentence" / "manifests")
    report: dict[str, Any] = {
        "schema": "sentence-run-verification/1",
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "manifest_dir": str(manifest_dir),
        "pricing": PRICING,
        "acceptance_thresholds": {
            "parse_success_min": ACCEPT_PARSE_MIN,
            "alignment_coverage_min": ACCEPT_COVERAGE_MIN,
            "max_cost_cny": ACCEPT_COST_MAX,
            "reported_vs_independent_tolerance_pp": FLOAT_TOL * 100,
        },
    }
    if not (run_dir / "generations.jsonl").exists():
        report["error"] = "generations.jsonl missing"
        report["verdict"] = "FAIL"
        report["exit_code"] = 2
        return report
    if check_data:
        report["data"] = verify_data(manifest_dir, repo_root, check_audio=check_audio)
    report["reproducibility"] = verify_reproducibility(repo_root, run_dir, manifest_dir)
    report["run"] = verify_run(run_dir, manifest_dir)
    report["comparison"] = compare_reported(run_dir, report["run"])
    failed: list[dict[str, Any]] = []
    for section in ("data", "reproducibility", "run"):
        block = report.get(section) or {}
        for check in block.get("checks") or []:
            if not check.get("ok"):
                failed.append({"section": section, **check})
    report["failed_checks"] = failed
    discrepancies = report["comparison"].get("discrepancies") or []
    warnings: list[str] = []
    if not report["run"]["acceptance"]["passed"]:
        warnings.append("design §10.2 acceptance gates FAILED (see run.acceptance)")
    if discrepancies:
        warnings.append(f"{len(discrepancies)} independent-vs-reported discrepancies")
    report["warnings"] = warnings
    ok = (
        report["run"]["acceptance"]["passed"]
        and not failed
        and not discrepancies
    )
    report["verdict"] = "PASS" if ok else "FAIL"
    report["exit_code"] = 0 if ok else 1
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="run directory (e.g. results/sentence/sentence-v1)")
    parser.add_argument("--manifest-dir", type=Path, default=None,
                        help="item manifest directory (default: data/sentence/manifests)")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--json-out", type=Path, default=None, help="write JSON report here")
    parser.add_argument("--skip-data", action="store_true", help="skip manifest/data checks")
    parser.add_argument("--skip-audio", action="store_true", help="skip audio probes")
    parser.add_argument("--quiet", action="store_true", help="print only the verdict line")
    args = parser.parse_args(argv)
    report = build_report(
        args.run_dir.resolve(),
        manifest_dir=args.manifest_dir.resolve() if args.manifest_dir else None,
        repo_root=args.repo_root.resolve(),
        check_audio=not args.skip_audio,
        check_data=not args.skip_data,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    if args.quiet:
        acceptance = report.get("run", {}).get("acceptance", {})
        print(
            f"verdict={report['verdict']} acceptance_passed={acceptance.get('passed')} "
            f"failed_checks={len(report.get('failed_checks') or [])} "
            f"discrepancies={len((report.get('comparison') or {}).get('discrepancies') or [])}"
        )
    else:
        print(payload)
    return int(report.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
