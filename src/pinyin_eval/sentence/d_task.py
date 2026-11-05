"""Task D: ASR-text-conditioned alignment & scoring (design v0.2, 附录 A).

Three text sources over the same 1,758 OMPAL learner utterances:

* ``D-gold``      — OMPAL reference text (oracle upper bound);
* ``D-asr``       — fixed local ASR text (FunASR Paraformer-zh, independent of
  Qwen; produced by ``scripts/run_d_asr.py``);
* ``D-inject-10`` — gold text + 10% controlled character errors (6:2:2
  substitution / deletion / insertion, fixed seed).

The model input is *audio + one text*; it must output a per-input-character
alignment array plus whole-sentence 1–5 scores:

``{"alignment": [{"text", "aligned", "text_error", "error_type", "score"}], "scores": {...}}``

This module owns: text normalisation, gold alignment (input text vs OMPAL gold),
deterministic error injection, prompt-independent parsing, per-item scoring,
aggregate metrics with cluster bootstrap, source-degradation comparisons,
acceptance and the standalone Markdown report.  The prompt template lives in
:mod:`pinyin_eval.sentence.prompts` (``TEMPLATE_D``); the runner/CLI only need the
``D`` task entry in ``ITEM_MANIFESTS`` and ``D_PROMPT_VERSION``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..stats import (
    bootstrap_ci,
    holm_correct,
    mcnemar_test,
    paired_bootstrap_difference,
)
from .normalize import align_char_sequences, t2s
from .records import SentenceItem, load_jsonl

__all__ = [
    "D_METRIC_BOOTSTRAP",
    "D_PROMPT_VERSION",
    "D_TASK",
    "TEXT_SOURCES",
    "DReportError",
    "acceptance_d",
    "build_d_items",
    "build_d_report",
    "build_substitution_pools",
    "compare_d_sources",
    "gold_alignment",
    "inject_errors",
    "map_hyp_entries",
    "normalize_text",
    "parse_task_d",
    "render_d_markdown",
    "score_d",
    "task_d_metrics",
]

D_TASK = "D"
D_PROMPT_VERSION = "sentence-d-v1"
TEXT_SOURCES: tuple[str, ...] = ("D-gold", "D-asr", "D-inject-10")
SOURCE_SLUG: dict[str, str] = {"D-gold": "gold", "D-asr": "asr", "D-inject-10": "inject10"}
SOURCE_LABELS: dict[str, str] = {
    "D-gold": "D-gold（OMPAL 原文，上界）",
    "D-asr": "D-asr（FunASR Paraformer-zh 文本）",
    "D-inject-10": "D-inject-10（gold + 10% 受控错误）",
}
ERROR_TYPES: tuple[str, ...] = ("none", "tone", "consonant", "vowel")
D_DIMENSIONS: tuple[str, ...] = ("tone", "consonant", "vowel")
D_SCORE_DIMENSIONS: tuple[str, ...] = ("accuracy", "fluency", "prosody")
D_METRIC_BOOTSTRAP: tuple[str, ...] = (
    "alignment_f1",
    "text_error_f1",
    "tone_macro_f1",
    "consonant_macro_f1",
    "vowel_macro_f1",
    "accuracy_pearson",
    "fluency_pearson",
    "prosody_pearson",
    "mean_dimension_pearson",
    "mean_dimension_mae",
)

D_ACCEPTANCE = {"parse_success_min": 0.95, "alignment_coverage_min": 0.95, "max_cost_cny": 10.0}
D_PREP_SEED = 20261103
_INJECT_RATIO = 0.10
_INJECT_MIX = {"sub": 0.6, "del": 0.2, "ins": 0.2}

_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_PUNCT_SPACE_RE = re.compile(r"[\s\W_]+", re.UNICODE)
_JSON_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class DReportError(RuntimeError):
    """Raised when task-D inputs/artifacts are missing or inconsistent."""


# ---------------------------------------------------------------------------
# Text normalisation / char helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """D input-text normalisation: NFKC, t2s, strip punctuation/whitespace.

    Keeps Han **and** letters/digits (the ASR occasionally emits Latin tokens,
    e.g. ``我the需...``): the design (A.2) only asks for NFKC + removal of
    punctuation/whitespace + OpenCC t2s.  Deleting Latin characters would make
    the input text shorter than the model's per-character alignment array and
    silently turn ASR substitutions into deletions.  For ``D-asr`` the raw ASR
    output is archived separately and never edited.
    """
    body = unicodedata.normalize("NFKC", text or "")
    body = t2s(body)
    out: list[str] = []
    for ch in body:
        category = unicodedata.category(ch)
        if category.startswith(("P", "Z", "C")) or ch.isspace():
            continue
        out.append(ch)
    return "".join(out)


def _chars(text: str) -> list[str]:
    return list(normalize_text(text))


def _pinyin_syllables(chars: Iterable[str]) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin

        return [lazy_pinyin(ch, style=Style.TONE3, errors="default")[0] for ch in chars]
    except Exception:  # pragma: no cover - pypinyin is a declared dependency
        return [ch for ch in chars]


# ---------------------------------------------------------------------------
# Gold alignment: input text T vs OMPAL reference text G
# ---------------------------------------------------------------------------


def gold_alignment(
    text: str,
    gold_text: str,
    gold_labels: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Char-level Levenshtein alignment of the input text T against gold G.

    Returns ``(per_text_char, stats)`` where ``per_text_char[i]`` describes the
    gold label of the *i*-th input character:

    * ``aligned``      — T char has a G counterpart (match or substitution);
    * ``text_correct`` — counterpart exists **and** the characters are equal
      (``None`` for an insertion: no counterpart, no equality to judge);
    * ``error_type``   — only for ``aligned and text_correct`` characters:
      ``none | tone | consonant | vowel`` from the OMPAL 0/1 labels
      (priority tone > consonant > vowel), otherwise ``None``;
    * ``labels``       — the OMPAL per-char labels when there is a counterpart.
    """
    text_chars = _chars(text)
    gold_chars = _chars(gold_text)
    pairs, n_insertions, _n_deletions = align_char_sequences(gold_chars, text_chars)
    # pairs: (gold_index, text_index | None) for every gold char
    matched: dict[int, int] = {
        text_idx: gold_idx for gold_idx, text_idx in pairs if text_idx is not None
    }
    deleted_gold = [gold_idx for gold_idx, text_idx in pairs if text_idx is None]
    per_char: list[dict[str, Any]] = []
    for text_idx, text_char in enumerate(text_chars):
        gold_idx = matched.get(text_idx)
        if gold_idx is None:
            per_char.append(
                {
                    "aligned": False,
                    "text_correct": None,
                    "error_type": None,
                    "labels": None,
                    "gold_index": None,
                }
            )
            continue
        labels = gold_labels[gold_idx] if gold_idx < len(gold_labels) else None
        correct = text_char == gold_chars[gold_idx]
        error_type: str | None = None
        if correct and labels is not None:
            tone = int(labels.get("tone", 1))
            consonant = int(labels.get("consonant", 1))
            vowel = int(labels.get("vowel", 1))
            if tone == 0:
                error_type = "tone"
            elif consonant == 0:
                error_type = "consonant"
            elif vowel == 0:
                error_type = "vowel"
            else:
                error_type = "none"
        per_char.append(
            {
                "aligned": True,
                "text_correct": correct,
                "error_type": error_type,
                "labels": labels,
                "gold_index": gold_idx,
            }
        )
    stats = {
        "n_text_chars": len(text_chars),
        "n_gold_chars": len(gold_chars),
        "n_insertions_text": n_insertions,
        "n_deletions_gold": len(deleted_gold),
        "deleted_gold_indices": deleted_gold,
        "n_aligned": sum(1 for entry in per_char if entry["aligned"]),
        "n_text_correct": sum(1 for entry in per_char if entry["text_correct"]),
    }
    return per_char, stats


# ---------------------------------------------------------------------------
# Deterministic controlled error injection (D-inject-10)
# ---------------------------------------------------------------------------


def _opencc_s2t():
    try:  # pragma: no cover - optional dependency path
        from opencc import OpenCC

        return OpenCC("s2t")
    except Exception:  # noqa: BLE001
        return None


def _aishell_homophone_pool(cfg: Config) -> dict[str, list[str]]:
    """Build a large traditional-character homophone pool from AISHELL-3 labels.

    The OMPAL corpus alone has only 337 distinct characters, so same-syllable
    substitutions often fall back to frequent characters.  AISHELL-3's released
    train label file gives ~60k char/pinyin pairs; converting the (simplified)
    characters to traditional via OpenCC s2t grows the homophone pool while
    staying inside the "同音/近音/常见混淆字表" rule of design A.2.
    """
    label_path = cfg.repo_root / "data/sentence/aishell3/label_train-set.txt"
    if not label_path.exists():
        return {}
    import re as _re

    from .normalize import align_gold

    converter = _opencc_s2t()
    pool: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    marks = _re.compile(r"[%$]")
    with label_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            _uid, pinyin, text = parts
            chars = marks.sub("", text)
            tokens = marks.sub("", pinyin).split()
            gold = align_gold(chars, tokens)
            if gold is None:
                continue
            for syllable in gold:
                if not syllable.surface or syllable.tone not in (1, 2, 3, 4, 5):
                    continue
                char = syllable.char
                if converter is not None:
                    char = converter.convert(char)
                key = f"{syllable.surface}{syllable.tone}"
                entry = (key, char)
                if entry in seen:
                    continue
                seen.add(entry)
                pool.setdefault(key, []).append(char)
    return pool


def build_substitution_pools(texts: Iterable[str]) -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    """Build homophone / near-homophone substitution pools from the corpus.

    Returns ``(toned, toneless, insertion_pool)`` where ``toned[syllable]`` lists
    corpus characters sharing the pypinyin TONE3 syllable, ``toneless`` groups by
    syllable without tone, and ``insertion_pool`` is the most frequent characters
    (natural-looking insertions).
    """
    chars: Counter[str] = Counter()
    for text in texts:
        chars.update(_chars(text))
    char_list = sorted(chars)
    syllables = _pinyin_syllables(char_list)
    toned: dict[str, list[str]] = {}
    toneless: dict[str, list[str]] = {}
    for ch, syl in zip(char_list, syllables, strict=False):
        toned.setdefault(syl, []).append(ch)
        toneless.setdefault(re.sub(r"[1-5]$", "", syl), []).append(ch)
    insertion_pool = [ch for ch, _count in chars.most_common(64)]
    return toned, toneless, insertion_pool


def inject_errors(
    gold_chars: Sequence[str],
    *,
    seed: int,
    toned_pool: Mapping[str, Sequence[str]],
    toneless_pool: Mapping[str, Sequence[str]],
    insertion_pool: Sequence[str],
    ratio: float = _INJECT_RATIO,
    mix: Mapping[str, float] = _INJECT_MIX,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Apply deterministic 10% character errors (6:2:2 sub / del / ins)."""
    chars = list(gold_chars)
    n = len(chars)
    if n == 0:
        return [], []
    rng = random.Random(seed)
    k = max(1, int(round(n * ratio)))
    positions = rng.sample(range(n), min(n, k))
    # one operation per perturbed character, drawn with the design's 6:2:2 mix;
    # expected aggregate mix is 6:2:2 (2:3:5 style rounding artifacts avoided by
    # drawing the op stochastically instead of splitting k deterministically).
    choices = ("sub", "del", "ins")
    weights = (float(mix.get("sub", 0.6)), float(mix.get("del", 0.2)), float(mix.get("ins", 0.2)))
    decisions = [(position, rng.choices(choices, weights=weights, k=1)[0]) for position in positions]
    ops: list[dict[str, Any]] = []
    syllables = _pinyin_syllables(chars)
    # apply right-to-left so earlier indices stay valid
    for position, op in sorted(decisions, key=lambda item: item[0], reverse=True):
        original = chars[position]
        if op == "sub":
            syllable = syllables[position]
            candidates = [c for c in toned_pool.get(syllable, []) if c != original]
            if not candidates:
                toneless = re.sub(r"[1-5]$", "", syllable)
                candidates = [c for c in toneless_pool.get(toneless, []) if c != original]
            if not candidates:
                candidates = [c for c in insertion_pool if c != original]
            replacement = rng.choice(sorted(candidates)) if candidates else original
            chars[position] = replacement
            ops.append({"op": "sub", "position": position, "gold": original, "new": replacement})
        elif op == "del":
            removed = chars.pop(position)
            ops.append({"op": "del", "position": position, "gold": removed, "new": None})
        else:  # insertion before the selected character
            inserted = rng.choice(sorted(insertion_pool)) if insertion_pool else "的"
            chars.insert(position, inserted)
            ops.append({"op": "ins", "position": position, "gold": None, "new": inserted})
    return chars, ops


# ---------------------------------------------------------------------------
# Data preparation: build the D item manifest
# ---------------------------------------------------------------------------


def _asr_quality(source_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise the ASR summary's text-quality block into the report schema.

    ``scripts/run_d_asr.py`` writes ``text_quality`` with
    ``sdi={substitutions, deletions, insertions, substitution_rate, ...}`` and
    ``normalization_rate={..., "rate": float}``; this helper maps it (and tolerates
    older/renamed keys) to the flat keys used by the task-D report.
    """
    summary = source_summary or {}
    quality = summary.get("quality") or summary.get("text_quality") or {}
    if not isinstance(quality, Mapping):
        return {}
    sdi = quality.get("sdi") if isinstance(quality.get("sdi"), Mapping) else {}
    normalization_raw = quality.get("normalization_rate")
    if isinstance(normalization_raw, Mapping):
        normalization = normalization_raw.get("rate")
    else:
        normalization = normalization_raw
    per_learner = quality.get("per_learner_cer")
    per_learner = per_learner if isinstance(per_learner, Mapping) else {}

    def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                return value
        return None

    return {
        "cer_micro": quality.get("cer_micro"),
        "cer_mean": _first(quality, "cer_mean_utterance", "cer_mean"),
        "cer_median": _first(quality, "cer_median_utterance", "cer_median"),
        "cer_p90": _first(quality, "cer_p90_utterance", "cer_p90"),
        "sentence_exact_rate": quality.get("sentence_exact_rate"),
        "n_sub": _first(sdi, "substitutions", "n_sub"),
        "n_del": _first(sdi, "deletions", "n_del"),
        "n_ins": _first(sdi, "insertions", "n_ins"),
        "sub_ratio": _first(sdi, "substitution_rate", "sub_ratio"),
        "del_ratio": _first(sdi, "deletion_rate", "del_ratio"),
        "ins_ratio": _first(sdi, "insertion_rate", "ins_ratio"),
        "normalization_rate": normalization,
        "learner_cer_median": per_learner.get("median"),
        "learner_cer_p90": per_learner.get("p90"),
        "learner_cer_max": per_learner.get("max"),
    }


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_source_fields(item: SentenceItem) -> dict[str, Any]:
    meta = item.meta or {}
    return {
        "utt_id": str(meta.get("utt_id") or item.item_id.split("|", 1)[-1]),
        "speaker": item.speaker,
        "sentence_id": meta.get("sentence_id"),
        "learner_id": meta.get("learner_id") or item.speaker,
    }


def build_d_items(
    cfg: Config,
    *,
    limit: int | None = None,
    require_asr: bool = True,
) -> tuple[list[SentenceItem], dict[str, Any]]:
    """Build ``d_items.jsonl`` content (3 text sources x 1,758 utterances).

    Reads the cleaned OMPAL B manifest (per-char labels), the C manifest
    (sentence-level human scores) and the fixed ASR outputs.
    """
    manifest_dir = cfg.path("sentence.paths.manifest_dir", "data/sentence/manifests")
    b_rows = load_jsonl(manifest_dir / "ompal_b_items.jsonl")
    c_rows = load_jsonl(manifest_dir / "ompal_c_items.jsonl")
    if not b_rows:
        raise DReportError(f"missing B manifest under {manifest_dir}")
    c_by_utt = {str(row.get("meta", {}).get("utt_id")): row for row in c_rows}
    asr_outputs = Path(cfg.get("sentence.d.asr_outputs", "data/sentence/d_asr/asr_outputs.jsonl"))
    asr_summary_path = Path(cfg.get("sentence.d.asr_summary", "data/sentence/d_asr/d_asr_summary.json"))
    asr_by_utt: dict[str, dict[str, Any]] = {}
    asr_summary: dict[str, Any] = {}
    if Path(asr_outputs).exists():
        asr_by_utt = {str(row["utt_id"]): row for row in load_jsonl(asr_outputs) if row.get("ok")}
    if Path(asr_summary_path).exists():
        asr_summary = json.loads(Path(asr_summary_path).read_text(encoding="utf-8"))
    if require_asr and not asr_by_utt:
        raise DReportError(
            f"ASR outputs missing ({asr_outputs}); run scripts/run_d_asr.py first "
            "or build only the gold/inject sources with require_asr=False"
        )
    if require_asr and limit is None and len(asr_by_utt) < len(b_rows):
        raise DReportError(
            f"ASR outputs incomplete: {len(asr_by_utt)} ok of {len(b_rows)} learner "
            "utterances; finish scripts/run_d_asr.py before building the full D manifest "
            "(use --limit for an explicit smoke build)"
        )

    seed = int(cfg.get("sentence.d.seed", D_PREP_SEED))
    source_names = list(cfg.get("sentence.d.sources") or TEXT_SOURCES)
    toned_pool, toneless_pool, insertion_pool = build_substitution_pools(
        row.get("chars") or row.get("text") or "" for row in b_rows
    )
    extra_pool = _aishell_homophone_pool(cfg)
    for syllable, chars in extra_pool.items():
        merged = sorted(set(toned_pool.get(syllable, [])) | set(chars))
        toned_pool[syllable] = merged
        toneless_key = re.sub(r"[1-5]$", "", syllable)
        toneless_pool[toneless_key] = sorted(
            set(toneless_pool.get(toneless_key, [])) | set(merged)
        )

    items: list[SentenceItem] = []
    injection_stats: Counter[str] = Counter()
    injection_chars = 0
    asr_missing: list[str] = []
    skipped: list[dict[str, Any]] = []
    for row in b_rows:
        base = SentenceItem.from_dict(row)
        fields = _item_source_fields(base)
        utt_id = fields["utt_id"]
        gold_text = base.chars or base.text or ""
        gold_labels = list(base.labels or [])
        c_row = c_by_utt.get(utt_id)
        scores = dict((c_row or {}).get("scores") or {})
        asr_row = asr_by_utt.get(utt_id)
        if asr_row is None:
            asr_missing.append(utt_id)
            if require_asr:
                continue
        texts: dict[str, tuple[str, list[dict[str, Any]]]] = {}
        # D-gold
        texts["D-gold"] = (gold_text, [])
        # D-asr
        if asr_row is not None:
            asr_text = str(asr_row.get("text_norm") or normalize_text(asr_row.get("raw_text") or ""))
            texts["D-asr"] = (asr_text, [])
        # D-inject-10
        injected_chars, ops = inject_errors(
            list(gold_text),
            seed=seed + int(hashlib.sha256(utt_id.encode()).hexdigest()[:8], 16),
            toned_pool=toned_pool,
            toneless_pool=toneless_pool,
            insertion_pool=insertion_pool,
        )
        texts["D-inject-10"] = ("".join(injected_chars), ops)
        for source in source_names:
            if source not in texts:
                continue
            text, ops = texts[source]
            per_char, align_stats = gold_alignment(text, gold_text, gold_labels)
            if not text:
                skipped.append({"utt_id": utt_id, "source": source, "reason": "empty text"})
                continue
            if source == "D-inject-10":
                injection_stats["n_items"] += 1
                injection_chars += len(gold_text)
                for op in ops:
                    injection_stats[op["op"]] += 1
            meta = dict(base.meta or {})
            meta.update(
                {
                    "text_source": source,
                    "source_slug": SOURCE_SLUG[source],
                    "utt_id": utt_id,
                    "learner_id": fields["learner_id"],
                    "sentence_id": fields["sentence_id"],
                    "gold_text": gold_text,
                    "gold_alignment": per_char,
                    "gold_alignment_stats": align_stats,
                    "injection_ops": ops,
                    "asr_text_raw": (asr_row or {}).get("raw_text") if source == "D-asr" else None,
                }
            )
            items.append(
                SentenceItem(
                    item_id=f"D|{SOURCE_SLUG[source]}|{utt_id}",
                    task=D_TASK,
                    source=base.source,
                    speaker=base.speaker,
                    audio_path=base.audio_path,
                    duration_s=base.duration_s,
                    text=text,
                    chars=gold_text,
                    pinyin_surface=[],
                    labels=gold_labels,
                    scores=scores,
                    native=base.native,
                    learner=base.learner,
                    flags={"native": bool(base.native), "learner": not bool(base.native)},
                    meta=meta,
                )
            )
        if limit is not None and len(items) >= limit:
            break

    summary = {
        "task": D_TASK,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "sources": source_names,
        "n_items": len(items),
        "n_utterances": len({item.meta.get("utt_id") for item in items}),
        "per_source": {
            source: sum(1 for item in items if item.meta.get("text_source") == source)
            for source in source_names
        },
        "n_asr_missing": len(asr_missing),
        "asr_missing": asr_missing[:100],
        "skipped": skipped[:100],
        "injection": {
            **dict(injection_stats),
            "n_chars": injection_chars,
            "n_perturbed": sum(injection_stats.get(op, 0) for op in ("sub", "del", "ins")),
            "ratio": (
                sum(injection_stats.get(op, 0) for op in ("sub", "del", "ins")) / injection_chars
                if injection_chars else None
            ),
        },
        "asr": {
            "model_id": asr_summary.get("model_id"),
            "device": asr_summary.get("device"),
            "n_ok": asr_summary.get("n_ok"),
            "n_expected": len(b_rows),
            "n_missing": len(asr_missing),
            "quality": _asr_quality(asr_summary),
            "outputs_sha256": _sha256_file(Path(asr_outputs)),
            "summary_sha256": _sha256_file(Path(asr_summary_path)),
        },
        "gold_alignment": {
            "n_deletions_gold": sum(
                int(item.meta.get("gold_alignment_stats", {}).get("n_deletions_gold", 0) or 0)
                for item in items
            ),
            "n_insertions_text": sum(
                int(item.meta.get("gold_alignment_stats", {}).get("n_insertions_text", 0) or 0)
                for item in items
            ),
        },
    }
    return items, summary


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _extract_json(text: str) -> Any | None:
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
            repaired = re.sub(r",\s*([}\]])", r"\1", chunk)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "是", "正确", "对", "ok"}:
        return True
    if text in {"false", "0", "no", "否", "错误", "不对", "wrong"}:
        return False
    return None


def _as_score(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group())
    if not math.isfinite(number):
        return None
    return round(min(5.0, max(1.0, number)), 2)


_ERROR_TYPE_TABLE: dict[str, str] = {
    "tone": "tone", "声调": "tone",
    "consonant": "consonant", "声母": "consonant", "initial": "consonant",
    "vowel": "vowel", "韵母": "vowel", "final": "vowel",
}
_NO_ERROR_VALUES = {"", "none", "null", "nil", "ok", "correct", "无", "没有", "无错误", "正确"}


def _as_error_type(value: Any) -> str | None:
    """Parse the design A.3 ``error_type`` value (English or Chinese labels).

    ``none`` / ``null`` mean "no pronunciation error"; an unknown label is
    ``None`` (excluded from the localization confusion counts).
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in _NO_ERROR_VALUES:
        return "none"
    if raw in _ERROR_TYPE_TABLE:
        return _ERROR_TYPE_TABLE[raw]
    han = normalize_text(raw)
    if han in _ERROR_TYPE_TABLE:
        return _ERROR_TYPE_TABLE[han]
    for token, mapped in _ERROR_TYPE_TABLE.items():
        if token in raw or token in han:
            return mapped
    return None


def parse_task_d(text: str) -> dict[str, Any]:
    """Parse one task-D reply into ``alignment`` + ``scores``."""
    from .parsing import base_status

    status = base_status(text)
    result: dict[str, Any] = {"parse_status": status, "alignment": [], "scores": {}, "detail": ""}
    if status != "ok":
        return result
    obj = _extract_json(text)
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        # some replies wrap the requested object in a one-element array
        obj = obj[0]
    if not isinstance(obj, dict):
        result["parse_status"] = "parse_error"
        result["detail"] = "no JSON object found"
        return result
    entries = obj.get("alignment") or obj.get("words") or obj.get("items")
    if not isinstance(entries, list) or not entries:
        result["parse_status"] = "parse_error"
        result["detail"] = "no alignment array found"
        return result
    alignment: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        alignment.append(
            {
                "text": str(entry.get("text") or entry.get("char") or ""),
                "aligned": _as_bool(entry.get("aligned")),
                "text_error": _as_bool(entry.get("text_error")),
                "error_type": _as_error_type(entry.get("error_type")),
                "score": _as_score(entry.get("score")),
            }
        )
    if not alignment:
        result["parse_status"] = "parse_error"
        result["detail"] = "empty alignment entries"
        return result
    nested = obj.get("scores")
    source = nested if isinstance(nested, dict) else obj
    scores: dict[str, float] = {}
    for dimension in D_SCORE_DIMENSIONS:
        value = source.get(dimension)
        if value is None:
            continue
        parsed = _as_score(value)
        if parsed is not None:
            scores[dimension] = parsed
    if len(scores) < 3:
        result["parse_status"] = "parse_error"
        result["detail"] = f"missing score dimensions: {sorted(set(D_SCORE_DIMENSIONS) - set(scores))}"
        return result
    result["alignment"] = alignment
    result["scores"] = {dimension: scores[dimension] for dimension in D_SCORE_DIMENSIONS}
    return result


# ---------------------------------------------------------------------------
# Scoring one item
# ---------------------------------------------------------------------------


def map_hyp_entries_with_extras(
    hyp: Sequence[Mapping[str, Any]], text_chars: Sequence[str]
) -> tuple[list[Mapping[str, Any] | None], list[Mapping[str, Any]]]:
    """Map the model's alignment array onto the input text positions.

    Returns ``(mapped, extras)`` where ``mapped[i]`` is the entry for input char
    ``i`` (or ``None``) and ``extras`` are model entries with no input position
    (design A.3: 多字单独统计).
    """
    if len(hyp) == len(text_chars):
        return list(hyp), []
    hyp_texts = [normalize_text(str(entry.get("text") or "")) or "?" for entry in hyp]
    pairs, _ins, _dele = align_char_sequences(  # align hyp texts to the input text
        [normalize_text(ch) or "?" for ch in text_chars], hyp_texts
    )
    out: list[Mapping[str, Any] | None] = [None] * len(text_chars)
    used: set[int] = set()
    for text_idx, hyp_idx in pairs:
        if hyp_idx is None:
            continue
        out[text_idx] = hyp[hyp_idx]
        used.add(hyp_idx)
    extras = [entry for index, entry in enumerate(hyp) if index not in used]
    return out, extras


def map_hyp_entries(
    hyp: Sequence[Mapping[str, Any]], text_chars: Sequence[str]
) -> list[Mapping[str, Any] | None]:
    """Backwards-compatible wrapper returning only the mapped entries."""
    return map_hyp_entries_with_extras(hyp, text_chars)[0]


def _row_envelope(item: SentenceItem, condition: str, parse_status: str) -> dict[str, Any]:
    meta = item.meta or {}
    return {
        "item_id": item.item_id,
        "task": D_TASK,
        "condition": condition,
        "parse_status": parse_status,
        "cluster_id": item.speaker,
        "speaker": item.speaker,
        "source": str(meta.get("text_source") or ""),
        "source_slug": str(meta.get("source_slug") or ""),
        "utt_id": str(meta.get("utt_id") or ""),
        "duration_s": item.duration_s,
    }


def score_d(item: SentenceItem, parsed: Mapping[str, Any], condition: str = "") -> dict[str, Any]:
    """Score one task-D item (alignment, text-error, error localization, scores)."""
    meta = item.meta or {}
    parse_status = str(parsed.get("parse_status") or "parse_error")
    row = _row_envelope(item, condition, parse_status)
    gold_alignment = list(meta.get("gold_alignment") or [])
    text_chars = _chars(item.text or "")
    row["n_text_chars"] = len(text_chars)
    row["n_gold_chars"] = len(gold_alignment)
    alignment_stats = meta.get("gold_alignment_stats") or {}
    row["n_gold_chars_source"] = int(alignment_stats.get("n_gold_chars", 0) or 0)
    row["n_gold_deletions"] = int(alignment_stats.get("n_deletions_gold", 0) or 0)
    rows_fields = (
        ["align_tp", "align_fp", "align_fn", "align_tn",
         "terr_tp", "terr_fp", "terr_fn", "terr_tn"]
        + [f"loc_{dim}_{key}" for dim in D_DIMENSIONS for key in ("tp", "fp", "fn", "tn")]
        + [f"loc_zero_cons_{key}" for key in ("tp", "fp", "fn", "tn")]
    )
    row.update(dict.fromkeys(rows_fields, 0))
    row.update(
        {
            "n_outputs": 0,
            "n_mapped": 0,
            "n_outputs_extra": 0,
            "n_extra_aligned": 0,
            "n_text_copy_mismatch": 0,
            "output_coverage": 0.0,
            "n_localizable": 0,
            "n_score": 0,
            "score_sum": 0.0,
            "n_score_correct": 0,
            "score_sum_correct": 0.0,
            "n_score_error": 0,
            "score_sum_error": 0.0,
        }
    )
    for dimension in D_SCORE_DIMENSIONS:
        row[f"gold_{dimension}"] = (item.scores or {}).get(dimension)
        row[f"pred_{dimension}"] = None
    if parse_status != "ok":
        return row
    hyp = list(parsed.get("alignment") or [])
    row["n_outputs"] = len(hyp)
    mapped, extras = map_hyp_entries_with_extras(hyp, text_chars)
    row["n_mapped"] = sum(1 for entry in mapped if entry is not None)
    row["n_outputs_extra"] = len(extras)
    row["n_extra_aligned"] = sum(1 for entry in extras if entry.get("aligned"))
    row["output_coverage"] = row["n_mapped"] / len(text_chars) if text_chars else 0.0
    for position, entry in zip(text_chars, mapped, strict=False):
        if entry is None:
            continue
        if normalize_text(str(entry.get("text") or "")) != normalize_text(position):
            row["n_text_copy_mismatch"] += 1
    for index, gold in enumerate(gold_alignment):
        entry = mapped[index] if index < len(mapped) else None
        gold_aligned = bool(gold.get("aligned"))
        gold_text_correct = gold.get("text_correct")
        pred_aligned = bool(entry.get("aligned")) if entry is not None else False
        if gold_aligned:
            if pred_aligned:
                row["align_tp"] += 1
            else:
                row["align_fn"] += 1
        else:
            if pred_aligned:
                row["align_fp"] += 1
            else:
                row["align_tn"] += 1
        # text-error confusion (design A.4): insertion -> text_error_gold = True,
        # substitution -> text_error_gold = not text_correct, matched -> same.
        if gold_aligned:
            gold_text_error = gold_text_correct is not None and not bool(gold_text_correct)
        else:
            gold_text_error = True
        pred_text_error = bool(entry.get("text_error")) if entry is not None else False
        if gold_text_error:
            if pred_text_error:
                row["terr_tp"] += 1
            else:
                row["terr_fn"] += 1
        else:
            if pred_text_error:
                row["terr_fp"] += 1
            else:
                row["terr_tn"] += 1
        # error localization: gold text-correct + aligned chars only
        if not gold_aligned or not gold_text_correct:
            continue
        row["n_localizable"] += 1
        labels = gold.get("labels") or {}
        pred_error_type = entry.get("error_type") if entry is not None else None
        for dimension in D_DIMENSIONS:
            label = labels.get(dimension)
            if label is None:
                continue
            gold_error = int(label) == 0
            pred_error = pred_error_type == dimension
            keys = f"loc_{dimension}_"
            if gold_error:
                if pred_error:
                    row[keys + "tp"] += 1
                else:
                    row[keys + "fn"] += 1
            else:
                if pred_error:
                    row[keys + "fp"] += 1
                else:
                    row[keys + "tn"] += 1
            if dimension == "consonant" and bool(labels.get("zero_initial")):
                zkeys = "loc_zero_cons_"
                if gold_error:
                    row[zkeys + ("tp" if pred_error else "fn")] += 1
                else:
                    row[zkeys + ("fp" if pred_error else "tn")] += 1
        # descriptive per-char score
        if entry is not None and entry.get("score") is not None:
            value = float(entry["score"])
            row["n_score"] += 1
            row["score_sum"] += value
            gold_any_error = any(int(labels.get(d, 1)) == 0 for d in D_DIMENSIONS)
            if gold_any_error:
                row["n_score_error"] += 1
                row["score_sum_error"] += value
            else:
                row["n_score_correct"] += 1
                row["score_sum_correct"] += value
    # model entries with no input position: an "aligned=True" claim for a
    # non-existent input character is an alignment false positive (A.3 多字统计)
    row["align_fp"] += row["n_extra_aligned"]
    scores = parsed.get("scores") or {}
    for dimension in D_SCORE_DIMENSIONS:
        row[f"pred_{dimension}"] = scores.get(dimension)
    return row


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def _f1(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _macro_f1(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    """Macro-F1 over the two classes (positive = pronunciation error)."""
    f1_error = _f1(tp, fp, fn)["f1"]
    f1_ok = _f1(tn, fn, fp)["f1"]  # swap roles: positive = ok
    values = [value for value in (f1_error, f1_ok) if value is not None]
    macro = sum(values) / len(values) if values else None
    return {
        "macro_f1": macro,
        "f1_error": f1_error,
        "f1_ok": f1_ok,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        j = index
        while j + 1 < len(order) and values[order[j + 1]] == values[order[index]]:
            j += 1
        rank = (index + j) / 2.0 + 1.0
        for k in range(index, j + 1):
            ranks[order[k]] = rank
        index = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _scoring_pairs(rows: Sequence[Mapping[str, Any]], dimension: str) -> tuple[list[float], list[float]]:
    gold: list[float] = []
    pred: list[float] = []
    for row in rows:
        if str(row.get("parse_status")) != "ok":
            continue
        value = row.get(f"pred_{dimension}")
        gold_value = row.get(f"gold_{dimension}")
        if value is None or gold_value is None:
            continue
        gold.append(float(gold_value))
        pred.append(float(value))
    return pred, gold


def _scoring_stats(rows: Sequence[Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    pred, gold = _scoring_pairs(rows, dimension)
    pearson = _pearson(pred, gold)
    spearman = _spearman(pred, gold)
    if gold:
        errors = [p - g for p, g in zip(pred, gold, strict=False)]
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    else:
        mae = rmse = None
    return {"n": len(gold), "pearson": pearson, "spearman": spearman, "mae": mae, "rmse": rmse}


def _mean_dimension_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return copies with mean-of-three gold/pred scores (``*_mean`` keys)."""
    out: list[Mapping[str, Any]] = []
    for row in rows:
        if str(row.get("parse_status")) != "ok":
            continue
        if any(row.get(f"pred_{dim}") is None for dim in D_SCORE_DIMENSIONS):
            continue
        if any(row.get(f"gold_{dim}") is None for dim in D_SCORE_DIMENSIONS):
            continue
        copied = dict(row)
        copied["gold_mean"] = sum(float(row[f"gold_{dim}"]) for dim in D_SCORE_DIMENSIONS) / 3.0
        copied["pred_mean"] = sum(float(row[f"pred_{dim}"]) for dim in D_SCORE_DIMENSIONS) / 3.0
        out.append(copied)
    return out


def _sum_counts(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, int]:
    return {
        key: int(sum(int(row.get(f"{prefix}{key}", 0) or 0) for row in rows))
        for key in ("tp", "fp", "fn", "tn")
    }


def _headline_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if str(row.get("parse_status")) == "ok"]
    align = _sum_counts(ok_rows, "align_")
    terr = _sum_counts(ok_rows, "terr_")
    alignment = _f1(align["tp"], align["fp"], align["fn"])
    text_error = _f1(terr["tp"], terr["fp"], terr["fn"])
    localization: dict[str, Any] = {}
    for dimension in D_DIMENSIONS:
        counts = _sum_counts(ok_rows, f"loc_{dimension}_")
        localization[dimension] = _macro_f1(counts["tp"], counts["fp"], counts["fn"], counts["tn"])
    zero_counts = _sum_counts(ok_rows, "loc_zero_cons_")
    zero_initial = _macro_f1(
        zero_counts["tp"], zero_counts["fp"], zero_counts["fn"], zero_counts["tn"]
    )
    scoring = {dimension: _scoring_stats(ok_rows, dimension) for dimension in D_SCORE_DIMENSIONS}
    mean_rows = _mean_dimension_rows(ok_rows)
    scoring["mean_dimension"] = _scoring_stats(mean_rows, "mean") if mean_rows else {
        "n": 0, "pearson": None, "spearman": None, "mae": None, "rmse": None,
    }
    n_score = sum(int(row.get("n_score", 0) or 0) for row in ok_rows)
    score_sum = sum(float(row.get("score_sum", 0.0) or 0.0) for row in ok_rows)
    n_correct = sum(int(row.get("n_score_correct", 0) or 0) for row in ok_rows)
    sum_correct = sum(float(row.get("score_sum_correct", 0.0) or 0.0) for row in ok_rows)
    n_error = sum(int(row.get("n_score_error", 0) or 0) for row in ok_rows)
    sum_error = sum(float(row.get("score_sum_error", 0.0) or 0.0) for row in ok_rows)
    n_text = sum(int(row.get("n_text_chars", 0) or 0) for row in ok_rows)
    n_mapped = sum(int(row.get("n_mapped", 0) or 0) for row in ok_rows)
    n_localizable = sum(int(row.get("n_localizable", 0) or 0) for row in ok_rows)
    n_gold_source = sum(int(row.get("n_gold_chars_source", 0) or 0) for row in ok_rows)
    n_gold_deletions = sum(int(row.get("n_gold_deletions", 0) or 0) for row in ok_rows)
    return {
        "n_items": len(rows),
        "n_parse_ok": len(ok_rows),
        "n_parse_failures": len(rows) - len(ok_rows),
        "parse_failure_rate": _ratio(len(rows) - len(ok_rows), len(rows)),
        "alignment_coverage": _ratio(n_mapped, n_text),
        "output_coverage_rate": _ratio(n_mapped, n_text),
        "n_text_chars": n_text,
        "n_localizable": n_localizable,
        "gold_side_coverage": (
            None if not n_gold_source else 1.0 - (n_gold_deletions / n_gold_source)
        ),
        "alignment": alignment,
        "text_error": text_error,
        "localization": localization,
        "zero_initial_consonant": zero_initial,
        "scoring": scoring,
        "per_char_score": {
            "n": n_score,
            "mean": _ratio(score_sum, n_score),
            "mean_correct": _ratio(sum_correct, n_correct),
            "mean_error": _ratio(sum_error, n_error),
        },
    }


def _ci_headline(rows: Sequence[Mapping[str, Any]], *, n_boot: int, alpha: float, seed: int) -> dict[str, Any]:
    def metric_fn(sample: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
        headline = _headline_metrics(sample)
        values: dict[str, float] = {}
        for key in ("f1",):
            value = headline["alignment"].get(key)
            if value is not None:
                values["alignment_f1"] = float(value)
            value = headline["text_error"].get(key)
            if value is not None:
                values["text_error_f1"] = float(value)
        for dimension in D_DIMENSIONS:
            value = headline["localization"][dimension].get("macro_f1")
            if value is not None:
                values[f"{dimension}_macro_f1"] = float(value)
        for dimension in D_SCORE_DIMENSIONS:
            stats = headline["scoring"][dimension]
            if stats.get("pearson") is not None:
                values[f"{dimension}_pearson"] = float(stats["pearson"])
        mean_stats = headline["scoring"]["mean_dimension"]
        if mean_stats.get("pearson") is not None:
            values["mean_dimension_pearson"] = float(mean_stats["pearson"])
        if mean_stats.get("mae") is not None:
            values["mean_dimension_mae"] = float(mean_stats["mae"])
        return values

    try:
        result = bootstrap_ci(
            list(rows), metric_fn, cluster_key="cluster_id", n_boot=n_boot, alpha=alpha,
            seed=seed, stratify_key=None,
        )
        return {name: dict(entry) for name, entry in result.items()}
    except (ValueError, TypeError):  # pragma: no cover - degenerate inputs
        return {}


def _per_learner(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    learners = sorted({str(row.get("cluster_id")) for row in rows})
    out: list[dict[str, Any]] = []
    for learner in learners:
        subset = [row for row in rows if str(row.get("cluster_id")) == learner]
        headline = _headline_metrics(subset)
        entry = {"learner": learner, "n_items": len(subset), "n_parse_ok": headline["n_parse_ok"]}
        entry["alignment_f1"] = headline["alignment"].get("f1")
        for dimension in D_DIMENSIONS:
            entry[f"{dimension}_macro_f1"] = headline["localization"][dimension].get("macro_f1")
        for dimension in D_SCORE_DIMENSIONS:
            entry[f"{dimension}_pearson"] = headline["scoring"][dimension].get("pearson")
        out.append(entry)
    return out


def task_d_metrics(
    rows: Sequence[Mapping[str, Any]],
    source: str,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = D_PREP_SEED,
    text_quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Metrics for one text source (all rows should belong to it already)."""
    headline = _headline_metrics(rows)
    headline["source"] = source
    headline["source_label"] = SOURCE_LABELS.get(source, source)
    headline["with_ci"] = _ci_headline(rows, n_boot=n_boot, alpha=alpha, seed=seed)
    headline["per_learner"] = _per_learner(rows)
    if text_quality is not None:
        headline["text_quality"] = dict(text_quality)
    return headline


# ---------------------------------------------------------------------------
# Source comparisons (paired by utterance)
# ---------------------------------------------------------------------------


def _rows_by_utt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("parse_status")) != "ok":
            continue
        utt = str(row.get("utt_id"))
        if utt:
            out[utt] = row
    return out


def _pair_rows(
    rows_a: dict[str, Mapping[str, Any]], rows_b: dict[str, Mapping[str, Any]]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    common = sorted(set(rows_a) & set(rows_b))
    paired_a: list[Mapping[str, Any]] = []
    paired_b: list[Mapping[str, Any]] = []
    for utt in common:
        row_a = dict(rows_a[utt])
        row_b = dict(rows_b[utt])
        row_a["item_id"] = f"utt|{utt}"
        row_b["item_id"] = f"utt|{utt}"
        paired_a.append(row_a)
        paired_b.append(row_b)
    return paired_a, paired_b


def _headline_metric_fn(sample: Sequence[Mapping[str, Any]]) -> Mapping[str, float]:
    headline = _headline_metrics(sample)
    values: dict[str, float] = {}
    for key in ("alignment_f1", "tone_macro_f1", "consonant_macro_f1", "vowel_macro_f1",
                "mean_dimension_pearson", "mean_dimension_mae"):
        values[key] = None  # type: ignore[assignment]
    if headline["alignment"]["f1"] is not None:
        values["alignment_f1"] = float(headline["alignment"]["f1"])
    for dimension in D_DIMENSIONS:
        value = headline["localization"][dimension].get("macro_f1")
        values[f"{dimension}_macro_f1"] = None if value is None else float(value)
    mean_stats = headline["scoring"]["mean_dimension"]
    values["mean_dimension_pearson"] = (
        None if mean_stats.get("pearson") is None else float(mean_stats["pearson"])
    )
    values["mean_dimension_mae"] = (
        None if mean_stats.get("mae") is None else float(mean_stats["mae"])
    )
    return values


def compare_d_sources(
    rows_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline: str = "D-gold",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = D_PREP_SEED,
) -> dict[str, Any]:
    """Paired (utterance-level) degradation vs the ``D-gold`` baseline."""
    base_rows = _rows_by_utt(rows_by_source.get(baseline, []))
    comparisons: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for source, rows in rows_by_source.items():
        if source == baseline:
            continue
        other = _rows_by_utt(rows)
        paired_base, paired_other = _pair_rows(base_rows, other)
        if not paired_base:
            continue
        try:
            diff = paired_bootstrap_difference(
                paired_base, paired_other, _headline_metric_fn,
                cluster_key="cluster_id", n_boot=n_boot, alpha=alpha, seed=seed,
            )
        except (ValueError, TypeError):  # pragma: no cover - degenerate
            diff = {}
        entry = {"source": source, "baseline": baseline, "n_paired": len(paired_base),
                 "difference": dict(diff)}
        comparisons[source] = entry
        # McNemar-style paired outcome test on per-item alignment exactness
        outcomes = []
        for base_row, other_row in zip(paired_base, paired_other, strict=False):
            base_exact = _row_alignment_exact(base_row)
            other_exact = _row_alignment_exact(other_row)
            if base_exact is not None and other_exact is not None:
                outcomes.append((base_exact, other_exact))
        if outcomes:
            b_count = sum(1 for base, other in outcomes if base and not other)
            c_count = sum(1 for base, other in outcomes if other and not base)
            mcnemar = mcnemar_test([pair[0] for pair in outcomes], [pair[1] for pair in outcomes])
            entry["alignment_exact"] = {
                "n_pairs": len(outcomes),
                "only_baseline": b_count,
                "only_other": c_count,
                "p_value": mcnemar.get("p_value"),
                "method": mcnemar.get("method"),
            }
            pvalues[source] = float(mcnemar["p_value"])
    # Holm correction on any p-values produced (currently none from bootstrap diffs)
    metrics = sorted({metric for entry in comparisons.values()
                      for metric in entry.get("difference", {})})
    ordered_sources = sorted(pvalues)
    holm_values = holm_correct([pvalues[source] for source in ordered_sources]) if pvalues else []
    return {
        "baseline": baseline,
        "comparisons": comparisons,
        "metrics": metrics,
        "alignment_exact_holm": dict(zip(ordered_sources, holm_values, strict=False)),
    }


def _row_alignment_exact(row: Mapping[str, Any]) -> bool | None:
    if str(row.get("parse_status")) != "ok":
        return None
    if int(row.get("align_fp", 0) or 0) or int(row.get("align_fn", 0) or 0):
        return False
    if "align_tp" not in row and "align_tn" not in row:
        return None
    return True


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def _usage_for_source(cost: Mapping[str, Any], source: str) -> dict[str, Any]:
    slug = SOURCE_SLUG.get(source, "")
    per = cost.get("per_task_condition") or {}
    entry = per.get(f"D|{slug}") or {}
    return {
        "n_calls": int(entry.get("n_calls", 0) or 0),
        "n_errors": int(entry.get("n_errors", 0) or 0),
        "n_parse_failures": int(entry.get("n_parse_failures", 0) or 0),
        "cost_cny": float(entry.get("cost_cny", 0.0) or 0.0),
    }


def acceptance_d(
    metrics_by_source: Mapping[str, Mapping[str, Any]],
    cost: Mapping[str, Any],
    asr_archived: bool | Mapping[str, Any],
) -> dict[str, Any]:
    """Design §A.8 acceptance gates for task D."""
    parse_success: dict[str, Any] = {}
    for source, metrics in metrics_by_source.items():
        ok = int(metrics.get("n_parse_ok", 0) or 0)
        total = int(metrics.get("n_items", 0) or 0)
        rate = (ok / total) if total else None
        parse_success[source] = {
            "rate": rate,
            "n_items": total,
            "n_parse_ok": ok,
            "threshold": D_ACCEPTANCE["parse_success_min"],
            "passed": rate is not None and rate >= D_ACCEPTANCE["parse_success_min"],
        }
    coverage: dict[str, Any] = {}
    for source, metrics in metrics_by_source.items():
        value = metrics.get("alignment_coverage")
        coverage[source] = {
            "value": value,
            "threshold": D_ACCEPTANCE["alignment_coverage_min"],
            "passed": value is not None and value >= D_ACCEPTANCE["alignment_coverage_min"],
        }
    totals = cost.get("totals") or {}
    total_cost = totals.get("cost_cny")
    cost_block = {
        "value": total_cost,
        "threshold": D_ACCEPTANCE["max_cost_cny"],
        "passed": total_cost is not None and float(total_cost) <= D_ACCEPTANCE["max_cost_cny"],
    }
    if isinstance(asr_archived, Mapping):
        n_expected = int(asr_archived.get("n_expected") or 0)
        n_missing = int(asr_archived.get("n_missing", 0) or 0)
        n_ok = int(asr_archived.get("n_ok") or 0)
        complete = n_missing == 0 and (n_expected == 0 or n_ok >= n_expected)
        asr_block = {
            "archived": True,
            "n_expected": n_expected,
            "n_ok": n_ok,
            "n_missing": n_missing,
            "outputs_sha256": asr_archived.get("outputs_sha256"),
            "summary_sha256": asr_archived.get("summary_sha256"),
            "complete": complete,
            "passed": complete,
        }
    else:
        asr_block = {"archived": bool(asr_archived), "complete": bool(asr_archived),
                     "passed": bool(asr_archived)}
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
# Report
# ---------------------------------------------------------------------------


def run_dir_for(cfg: Config, run_id: str) -> Path:
    return cfg.path("sentence.paths.results_dir", "results/sentence") / run_id


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _fmt_ci(entry: Mapping[str, Any] | None) -> str:
    if not isinstance(entry, Mapping):
        return "—"
    low, high = entry.get("lo"), entry.get("hi")
    if low is None or high is None:
        return "—"
    return f"[{_fmt(low)}, {_fmt(high)}]"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def render_d_markdown(payload: Mapping[str, Any]) -> str:
    """Render the task-D Markdown report (pure function)."""
    lines: list[str] = []
    manifest = payload.get("manifest") or {}
    lines.append(f"# 任务 D 报告：ASR 文本条件下的对齐与评分（run `{payload.get('run_id')}`）")
    lines.append("")
    lines.append(
        "> 设计：`docs/sentence-experiment-design.md` v0.2 附录 A · "
        f"生成时间：{payload.get('generated_at')}"
    )
    lines.append(
        f"> 模型：`{manifest.get('model')}` · prompt：`{manifest.get('prompt_version') or D_PROMPT_VERSION}` "
        f"· 文本条件：{', '.join(payload.get('sources') or [])}"
    )
    lines.append(
        "> 本报告由 `pinyin_eval.sentence.d_task` 自动生成；指标来自逐项评分行，未做手工计算。"
    )
    lines.append("")

    lines.append("## 1. 运行元信息")
    lines.append("")
    meta_rows = [
        ("run_id", f"`{payload.get('run_id')}`"),
        ("模型", f"`{manifest.get('model')}`"),
        ("prompt_version", f"`{manifest.get('prompt_version') or D_PROMPT_VERSION}`"),
        ("ASR 模型", f"`{(payload.get('asr') or {}).get('model_id')}`"),
        ("ASR device / 版本", f"{(payload.get('asr') or {}).get('device')} / `{(payload.get('asr') or {}).get('funasr_version')}`"),
        ("计划调用数", _fmt((payload.get('cost') or {}).get('totals', {}).get('n_calls'))),
        ("实际成本", f"¥{_fmt((payload.get('cost') or {}).get('totals', {}).get('cost_cny'))}"),
        ("预算上限", f"¥{_fmt(manifest.get('budget_cny'))}"),
        ("验收总结果", "通过" if (payload.get("acceptance") or {}).get("passed") else "未通过"),
    ]
    lines.extend(_md_table(("项", "值"), meta_rows))
    lines.append("")

    lines.append("## 2. ASR 文本质量（描述，不用于筛选）")
    lines.append("")
    asr_quality = (payload.get("asr") or {}).get("quality") or {}
    if asr_quality:
        quality_rows = [
            ("CER（micro，字符）", _fmt(asr_quality.get("cer_micro"))),
            ("CER（按句平均）", _fmt(asr_quality.get("cer_mean"))),
            ("句级完全一致率（ASR vs gold）", _fmt_pct(asr_quality.get("sentence_exact_rate"))),
            ("CER 中位 / p90（按句）", f"{_fmt(asr_quality.get('cer_median'))} / {_fmt(asr_quality.get('cer_p90'))}"),
            ("S / D / I", f"{_fmt(asr_quality.get('n_sub'))} / {_fmt(asr_quality.get('n_del'))} / {_fmt(asr_quality.get('n_ins'))}"),
            ("S/D/I 比例", f"{_fmt(asr_quality.get('sub_ratio'))} / {_fmt(asr_quality.get('del_ratio'))} / {_fmt(asr_quality.get('ins_ratio'))}"),
            ("ASR normalization rate", _fmt_pct(asr_quality.get("normalization_rate"))),
            ("逐 learner CER 中位 / p90 / max", f"{_fmt(asr_quality.get('learner_cer_median'))} / {_fmt(asr_quality.get('learner_cer_p90'))} / {_fmt(asr_quality.get('learner_cer_max'))}"),
        ]
        lines.extend(_md_table(("指标", "值"), quality_rows))
        lines.append("")
        lines.append(
            "- 口径：ASR 文本与 OMPAL gold 文本先 NFKC/t2s/去标点空格，再做字符 Levenshtein；"
            "normalization rate 仅在 gold 对齐位置、OMPAL 标注为发音错误的字上统计，"
            "描述 ASR 自动纠错程度，不归因单个机制。"
        )
    else:
        lines.append("_ASR 质量摘要不可用。_")
    lines.append("")

    inject_quality = next(
        (source.get("text_quality") for source in (payload.get("metrics") or {}).values()
         if isinstance(source, Mapping) and source.get("source") == "D-inject-10"),
        None,
    )
    if inject_quality:
        lines.append("**D-inject-10 受控错误统计**")
        lines.append("")
        lines.extend(_md_table(
            ("项", "值"),
            [
                ("扰动字符 / 总字符", f"{_fmt(inject_quality.get('n_perturbed'))} / {_fmt(inject_quality.get('n_chars'))}"),
                ("实际扰动比例", _fmt_pct(inject_quality.get("inject_ratio"))),
                ("sub / del / ins", f"{_fmt(inject_quality.get('n_sub'))} / {_fmt(inject_quality.get('n_del'))} / {_fmt(inject_quality.get('n_ins'))}"),
            ],
        ))
        lines.append("")
    lines.append("## 3. 输出格式质量与对齐")
    lines.append("")
    format_rows = []
    alignment_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        format_rows.append(
            [
                source,
                _fmt(metrics.get("n_items")),
                _fmt(metrics.get("n_parse_ok")),
                _fmt_pct(metrics.get("parse_failure_rate")),
                _fmt_pct(metrics.get("alignment_coverage")),
                _fmt_pct(metrics.get("gold_side_coverage")),
            ]
        )
        alignment = metrics.get("alignment") or {}
        with_ci = (metrics.get("with_ci") or {}).get("alignment_f1") or {}
        alignment_rows.append(
            [
                source,
                _fmt(alignment.get("precision")),
                _fmt(alignment.get("recall")),
                _fmt(alignment.get("f1")),
                _fmt_ci(with_ci),
            ]
        )
    lines.extend(_md_table(
        ("文本条件", "items", "parse-ok", "解析失败率", "输入字覆盖率", "gold-side 覆盖率"),
        format_rows,
    ))
    lines.append("")
    lines.extend(_md_table(("文本条件", "对齐 precision", "recall", "F1", "F1 95% CI"), alignment_rows))
    lines.append("")
    lines.append("_对齐正类 = gold aligned（输入字有 OMPAL gold 对应）。_")
    lines.append("")
    terr_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        text_error = metrics.get("text_error") or {}
        with_ci = (metrics.get("with_ci") or {}).get("text_error_f1") or {}
        terr_rows.append([source, _fmt(text_error.get("precision")),
                          _fmt(text_error.get("recall")), _fmt(text_error.get("f1")),
                          _fmt_ci(with_ci)])
    lines.extend(_md_table(("文本条件", "text_error precision", "recall", "F1", "F1 95% CI"), terr_rows))
    lines.append("")
    lines.append(
        "_text_error 正类 = gold text_error（substitution 与 insertion 都算文本错误，"
        "见设计 A.4）；模型漏输出的输入字按「未声称文本错误」计入。_"
    )
    lines.append("")

    lines.append("## 4. 发音错误定位（仅 text-correct + aligned 字）")
    lines.append("")
    loc_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        localization = metrics.get("localization") or {}
        entry = [source]
        for dimension in D_DIMENSIONS:
            stats = localization.get(dimension) or {}
            entry.extend([_fmt(stats.get("macro_f1")), _fmt(stats.get("f1_error")),
                          _fmt(stats.get("fpr")), _fmt(stats.get("fnr"))])
        loc_rows.append(entry)
    lines.extend(_md_table(
        ("文本条件", "tone macro-F1", "tone error-F1", "tone FPR", "tone FNR",
         "consonant macro-F1", "consonant error-F1", "consonant FPR", "consonant FNR",
         "vowel macro-F1", "vowel error-F1", "vowel FPR", "vowel FNR"),
        loc_rows,
    ))
    lines.append("")
    zero_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        zero = metrics.get("zero_initial_consonant") or {}
        zero_rows.append([source, _fmt(zero.get("macro_f1")), _fmt(zero.get("f1_error")),
                          _fmt(zero.get("fpr")), _fmt(zero.get("fnr"))])
    lines.extend(_md_table(("文本条件", "零声母 consonant macro-F1", "error-F1", "FPR", "FNR"), zero_rows))
    lines.append("")

    lines.append("## 5. 整句评分相关性")
    lines.append("")
    score_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        scoring = metrics.get("scoring") or {}
        with_ci = metrics.get("with_ci") or {}
        entry = [source]
        for dimension in (*D_SCORE_DIMENSIONS, "mean_dimension"):
            stats = scoring.get(dimension) or {}
            pearson_ci = (with_ci.get(f"{dimension}_pearson") if dimension != "mean_dimension"
                          else with_ci.get("mean_dimension_pearson")) or {}
            entry.extend([_fmt(stats.get("pearson")), _fmt_ci(pearson_ci),
                          _fmt(stats.get("spearman")), _fmt(stats.get("mae")), _fmt(stats.get("rmse"))])
        score_rows.append(entry)
    lines.extend(_md_table(
        ("文本条件", "accuracy r", "r 95% CI", "accuracy ρ", "accuracy MAE", "accuracy RMSE",
         "fluency r", "r 95% CI", "fluency ρ", "fluency MAE", "fluency RMSE",
         "prosody r", "r 95% CI", "prosody ρ", "prosody MAE", "prosody RMSE",
         "mean r", "r 95% CI", "mean ρ", "mean MAE", "mean RMSE"),
        score_rows,
    ))
    lines.append("")
    lines.append("_目标为 OMPAL 3 位评分者均值；native 不进入目标（D 只用 learner）。_")
    lines.append("")

    lines.append("## 6. 文本条件退化（配对，按 learner 聚类 bootstrap）")
    lines.append("")
    comparisons = payload.get("comparisons") or {}
    cmp_rows = []
    for source, entry in (comparisons.get("comparisons") or {}).items():
        differences = entry.get("difference") or {}
        for metric in ("alignment_f1", "tone_macro_f1", "consonant_macro_f1", "vowel_macro_f1",
                       "mean_dimension_pearson", "mean_dimension_mae"):
            diff = differences.get(metric) or {}
            cmp_rows.append([source, metric, _fmt(diff.get("estimate")), _fmt_ci(diff)])
    if cmp_rows:
        lines.extend(_md_table(("文本条件", "指标", "Δ(D-gold − 条件)", "95% CI"), cmp_rows))
    else:
        lines.append("_退化对比不可用。_")
    lines.append("")
    lines.append(
        f"- 配对单位：同一 utterance 在两个文本条件下都有 parse-ok 行；"
        f"Δ = {comparisons.get('baseline') or 'D-gold'} − 该条件。"
        "对 alignment/macro-F1/Pearson：**正值 = 该条件相对上界退化**；"
        "对 MAE：**负值 = 该条件误差更大（退化）**，正值表示误差更小。"
    )
    lines.append("")
    exact_rows = []
    holm = comparisons.get("alignment_exact_holm") or {}
    for source, entry in (comparisons.get("comparisons") or {}).items():
        exact = entry.get("alignment_exact") or {}
        if not exact:
            continue
        exact_rows.append([
            source,
            _fmt(exact.get("n_pairs")),
            _fmt(exact.get("only_baseline")),
            _fmt(exact.get("only_other")),
            _fmt(exact.get("p_value")),
            _fmt(holm.get(source)),
            str(exact.get("method") or "—"),
        ])
    if exact_rows:
        lines.append("**整句对齐完全正确（逐 utterance）的配对检验**")
        lines.append("")
        lines.extend(_md_table(
            ("文本条件", "n_pairs", "仅 D-gold 完全正确", "仅该条件完全正确",
             "McNemar p", "Holm p", "方法"),
            exact_rows,
        ))
        lines.append("")
        lines.append("_族内 Holm 校正覆盖各文本条件 × alignment-exact 一项。_")
        lines.append("")

    lines.append("## 7. 逐字 score（描述性）")
    lines.append("")
    score_desc_rows = []
    for source in payload.get("sources") or []:
        metrics = (payload.get("metrics") or {}).get(source) or {}
        desc = metrics.get("per_char_score") or {}
        score_desc_rows.append([source, _fmt(desc.get("n")), _fmt(desc.get("mean")),
                                _fmt(desc.get("mean_correct")), _fmt(desc.get("mean_error"))])
    lines.extend(_md_table(("文本条件", "n 字", "均值", "gold 正确字均值", "gold 错误字均值"), score_desc_rows))
    lines.append("")
    lines.append("_OMPAL 只有 0/1 维度标签，逐字 1–5 无直接 gold；此处仅作描述，不做主推断。_")
    lines.append("")

    lines.append("## 8. 验收检查（设计 §A.8）")
    lines.append("")
    acceptance = payload.get("acceptance") or {}
    acceptance_rows = []
    for source, entry in (acceptance.get("parse_success") or {}).items():
        acceptance_rows.append([f"解析成功率：{source}", f"≥{_fmt_pct(entry.get('threshold'))}",
                                _fmt_pct(entry.get("rate")), "✅ 通过" if entry.get("passed") else "❌ 未过"])
    for source, entry in (acceptance.get("alignment_coverage") or {}).items():
        acceptance_rows.append([f"输入字覆盖率：{source}", f"≥{_fmt_pct(entry.get('threshold'))}",
                                _fmt_pct(entry.get("value")), "✅ 通过" if entry.get("passed") else "❌ 未过"])
    cost_entry = acceptance.get("cost_cny") or {}
    acceptance_rows.append(["D 主实验成本", f"≤¥{_fmt(cost_entry.get('threshold'))}",
                            f"¥{_fmt(cost_entry.get('value'))}", "✅ 通过" if cost_entry.get("passed") else "❌ 未过"])
    asr_entry = acceptance.get("asr_archived") or {}
    acceptance_rows.append(["ASR 文本已留档", "是", "是" if asr_entry.get("archived") else "否",
                            "✅ 通过" if asr_entry.get("passed") else "❌ 未过"])
    lines.extend(_md_table(("检查项", "阈值", "实际值", "结果"), acceptance_rows))
    lines.append("")
    lines.append(f"- **验收总结果：{'✅ 通过' if acceptance.get('passed') else '❌ 未通过'}**")
    lines.append("")

    lines.append("## 9. 局限与解释边界")
    lines.append("")
    lines.extend(
        [
            "- ASR 自动纠错可能抹掉学习者错误信号：D-gold vs D-asr 的退化量只作描述，不归因单个机制。",
            "- 无时间戳：D 的“对齐”是文本-音频逐字对应，不是 start/end 强制对齐。",
            "- ASR substitution 字的 OMPAL 发音标签不可直接解释：错误定位只取 text-correct 字。",
            "- 注入错误是受控压力测试，不代表真实 ASR 错误分布；10% 为主条件，5%/20% 未运行。",
            "- OMPAL 只有 3 位评分者、法语 L1 learner 子集；D 的结论只限定该子集。",
            "- 逐字 score 无人工 1–5 gold，只作描述。",
            "- 文本简繁/标点差异已用 OpenCC t2s 与规范化处理，但仍可能引入对齐误差。",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def build_d_report(cfg: Config, run_id: str, *, n_boot: int | None = None) -> dict[str, Any]:
    """Build metrics + artifacts for a task-D run; writes ``d_report.md`` etc."""
    run_dir = run_dir_for(cfg, run_id)
    rows_path = run_dir / "rows_d.parquet"
    if not rows_path.exists():
        raise DReportError(f"missing task-D scored rows: {rows_path}")
    import pandas as pd

    frame = pd.read_parquet(rows_path)
    rows = frame.to_dict(orient="records")
    prepared = _load_json_if_exists(run_dir / "d_prep_summary.json") or {}
    manifest = _load_json_if_exists(run_dir / "run_manifest.json") or {}
    gen_summary = _load_json_if_exists(run_dir / "generate_summary.json") or {}
    rows_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        source = str(row.get("source") or "")
        if not source:
            continue
        rows_by_source.setdefault(source, []).append(row)
    sources = [source for source in TEXT_SOURCES if source in rows_by_source]
    if not sources:
        raise DReportError("no task-D sources found in rows_d.parquet")
    stats_cfg = cfg.get("sentence.stats") or {}
    n_boot = int(n_boot or stats_cfg.get("n_boot", 2000))
    alpha = float(stats_cfg.get("alpha", 0.05))
    seed = int(cfg.get("sentence.d.seed", D_PREP_SEED))
    asr_summary = _load_json_if_exists(Path(cfg.get("sentence.d.asr_summary", "data/sentence/d_asr/d_asr_summary.json")))
    asr_quality = _asr_quality(asr_summary)
    metrics_by_source: dict[str, Any] = {}
    for source in sources:
        text_quality = None
        if source == "D-asr":
            text_quality = asr_quality
        elif source == "D-inject-10":
            injection = prepared.get("injection") or {}
            text_quality = {
                "n_items": int(injection.get("n_items", 0) or 0),
                "n_sub": int(injection.get("sub", 0) or 0),
                "n_del": int(injection.get("del", 0) or 0),
                "n_ins": int(injection.get("ins", 0) or 0),
                "n_chars": int(injection.get("n_chars", 0) or 0),
                "n_perturbed": int(injection.get("n_perturbed", 0) or 0),
                "inject_ratio": injection.get("ratio"),
            }
        metrics_by_source[source] = task_d_metrics(
            rows_by_source[source], source, n_boot=n_boot, alpha=alpha, seed=seed,
            text_quality=text_quality,
        )
    comparisons = compare_d_sources(rows_by_source, baseline="D-gold",
                                    n_boot=n_boot, alpha=alpha, seed=seed)
    cost = _load_json_if_exists(run_dir / "cost_summary.json") or _cost_summary_from_generations(cfg, run_dir)
    acceptance = acceptance_d(
        metrics_by_source, cost, asr_archived=prepared.get("asr") or bool(asr_summary)
    )
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": sources,
        "manifest": manifest,
        "prepared": prepared,
        "asr": {
            "model_id": (asr_summary or {}).get("model_id"),
            "device": (asr_summary or {}).get("device"),
            "funasr_version": (asr_summary or {}).get("funasr_version"),
            "quality": asr_quality,
        },
        "run": gen_summary,
        "metrics": metrics_by_source,
        "comparisons": comparisons,
        "cost": cost,
        "acceptance": acceptance,
        "bootstrap": {"n_boot": n_boot, "alpha": alpha, "seed": seed, "cluster_key": "cluster_id"},
    }
    markdown = render_d_markdown(payload)
    (run_dir / "d_report.md").write_text(markdown, encoding="utf-8")
    _write_d_csvs(payload, run_dir)
    (run_dir / "d_report_summary.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "d_acceptance.json").write_text(
        json.dumps(_jsonable(acceptance), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"run_id": run_id, "sources": sources, "acceptance": _jsonable(acceptance),
            "artifacts": ["d_report.md", "d_metrics.csv", "d_comparisons.csv",
                          "d_per_learner.csv", "d_acceptance.json", "d_report_summary.json"]}


def _write_d_csvs(payload: Mapping[str, Any], run_dir: Path) -> None:
    import csv

    metric_rows: list[dict[str, Any]] = []
    for source, metrics in (payload.get("metrics") or {}).items():
        with_ci = metrics.get("with_ci") or {}
        for metric in ("alignment_f1", "text_error_f1", "tone_macro_f1", "consonant_macro_f1",
                       "vowel_macro_f1", "accuracy_pearson", "fluency_pearson", "prosody_pearson",
                       "mean_dimension_pearson", "mean_dimension_mae"):
            ci = with_ci.get(metric) or {}
            metric_rows.append({
                "source": source, "metric": metric,
                "estimate": ci.get("estimate"), "ci_lo": ci.get("lo"), "ci_hi": ci.get("hi"),
            })
        for dimension in D_DIMENSIONS:
            stats = (metrics.get("localization") or {}).get(dimension) or {}
            metric_rows.append({"source": source, "metric": f"{dimension}_macro_f1",
                                "estimate": stats.get("macro_f1"), "ci_lo": None, "ci_hi": None})
        for dimension in D_SCORE_DIMENSIONS:
            stats = (metrics.get("scoring") or {}).get(dimension) or {}
            metric_rows.append({"source": source, "metric": f"{dimension}_mae",
                                "estimate": stats.get("mae"), "ci_lo": None, "ci_hi": None})
    if metric_rows:
        with (run_dir / "d_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metric_rows[0]))
            writer.writeheader()
            writer.writerows(metric_rows)
    cmp_rows: list[dict[str, Any]] = []
    for source, entry in ((payload.get("comparisons") or {}).get("comparisons") or {}).items():
        for metric, diff in (entry.get("difference") or {}).items():
            cmp_rows.append({
                "source": source, "metric": metric,
                "estimate": diff.get("estimate"), "ci_lo": diff.get("lo"), "ci_hi": diff.get("hi"),
                "baseline": entry.get("baseline"), "n_paired": entry.get("n_paired"),
            })
    if cmp_rows:
        with (run_dir / "d_comparisons.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(cmp_rows[0]))
            writer.writeheader()
            writer.writerows(cmp_rows)
    learner_rows: list[dict[str, Any]] = []
    for source, metrics in (payload.get("metrics") or {}).items():
        for entry in metrics.get("per_learner") or []:
            learner_rows.append({"source": source, **entry})
    if learner_rows:
        with (run_dir / "d_per_learner.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(learner_rows[0]))
            writer.writeheader()
            writer.writerows(learner_rows)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return None


def _cost_summary_from_generations(cfg: Config, run_dir: Path) -> dict[str, Any]:
    from .runner import cost_summary

    return cost_summary(cfg, run_dir.name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
