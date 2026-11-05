#!/usr/bin/env python
"""Task D (ASR-text condition): fixed local ASR transcripts for the 1,758 OMPAL learner wavs.

Design: ``docs/sentence-experiment-design.md`` v0.2, 附录 A (A.2 数据与文本源 / A.5 指标).
Task D conditions the Qwen evaluator on a *fixed* ASR transcript instead of the OMPAL
gold text.  This script produces that transcript once, locally, and archives it verbatim:

* renderer — **FunASR Paraformer-zh**, model id
  ``iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch``
  (ModelScope, ``AutoModel``), 16 kHz frontend, no VAD and no punctuation model, greedy
  decoding (``ctc_weight=0.0``, no LM, no ITN).  Fixed decode parameters are recorded in
  the summary; ``raw_text`` is the model output exactly as returned (no human correction,
  no hand edits);
* device — try ``mps`` first, fall back to ``cpu``; the device that actually ran is
  recorded together with every attempt's error.  Correctness is preferred over speed;
* provenance — model id / hub / revision, model-file hashes, FunASR + torch versions,
  decode parameters, manifest hash and the run date are written to
  ``d_asr_summary.json``.

Outputs (under ``data/sentence/d_asr/`` unless ``--out-dir`` overrides):

* ``asr_outputs.jsonl`` — one row per learner utterance::

      {"utt_id", "speaker", "audio_path", "raw_text", "text_norm", "duration_s",
       "sample_rate", "wall_s", "rtf", "device", "ok", "error", "model_id",
       "funasr_version", "decode_params", "timestamp"}

  ``raw_text`` is the verbatim model output; ``text_norm`` is the normalisation the task D
  pipeline uses before alignment (see below).  Rows are appended as the run progresses, so
  an interrupted run can be resumed with ``--resume``;
* ``d_asr_summary.json`` — counts, failures with reasons, device/attempts, model and
  decode provenance, wall time / RTF, and the ASR text-quality description (CER, S/D/I,
  ASR normalization rate, per-learner CER distribution).

Text normalisation (identical rule for gold and ASR text, applied before every metric)::

    normalize_text(raw):
        1. unicodedata.normalize("NFKC", raw)
        2. drop every character whose Unicode category starts with "P" (punctuation),
           "Z" (separators, including ordinary spaces) or "C" (control), plus all
           whitespace;
        3. OpenCC "t2s" (traditional -> simplified); identity fallback if OpenCC is
           unavailable, which never happens for the committed run (recorded).

Character error rate / S-D-I (design A.5 "ASR 文本 vs gold 的字符错误率（CER）、S/D/I 比例"):

* gold and hypothesis are aligned with a character-level Levenshtein edit script
  (:func:`align_chars`), ties resolved deterministically towards match < substitution <
  deletion < insertion;
* ``S`` = aligned pair with different characters, ``D`` = gold character without a
  hypothesis partner, ``I`` = hypothesis character without a gold partner;
  ``edit_distance = S + D + I``;
* ``cer_micro`` = total edit distance / total gold characters;
* ``cer_mean_utterance`` = mean of per-utterance CER (``edit_distance / len(gold)``,
  utterances with an empty gold string are skipped).

ASR normalization rate (design A.5):

* OMPAL labels every gold character with tone / consonant / vowel 0/1 flags; a gold
  character is an *OMPAL-error character* when at least one of the three flags is 0
  (the primary definition; per-dimension variants are reported as well);
* we align the normalised ASR text to the normalised gold text with the same Levenshtein
  edit script and, for every OMPAL-error gold character, ask whether the ASR character
  aligned to it is the **same character**.  The rate is
  ``#same / #OMPAL-error gold chars``; a deleted error character counts as *not* same
  (the ASR text does not show the gold character at that position);
* this is a *description* of how often the ASR text still spells an OMPAL-flagged error
  character as the gold character (i.e. did not rewrite it).  It is not evidence of
  automatic correction and supports no causal claim.

CLI::

    # smoke: 3 files into a scratch dir
    .venv/bin/python scripts/run_d_asr.py --limit 3 --out-dir /tmp/d_asr_smoke
    # full resumable run (CPU only in this environment, see --device)
    MODELSCOPE_CACHE=/tmp/dasr_modelscope HTTPS_PROXY=http://127.0.0.1:7897 \\
        .venv/bin/python scripts/run_d_asr.py --resume --device auto

Sandboxed environments only allow writes under the repo and /tmp, so
``MODELSCOPE_CACHE`` (and ``TMPDIR``) must point somewhere writable before the first
run; later runs reuse the cache and need no network.  ``funasr``/``torch`` are imported
lazily inside :func:`load_model`, so importing this module (for the tests) never
downloads anything.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import statistics
import sys
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/sentence/manifests/ompal_b_items.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "data/sentence/d_asr"

#: fixed ASR model (design A.2: FunASR Paraformer-zh, independent of the Qwen evaluator)
MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_HUB = "ms"  # ModelScope; the HF mirror is not used by the primary model
MODEL_REVISION = "master"

#: fixed decode parameters (no VAD / punctuation model / hotwords / LM / ITN)
DECODE_PARAMS: dict[str, Any] = {
    "batch_size_s": 1,  # one file per generate() call -> exact per-file wall time
    "hotword": "",
    "decoding_ctc_weight": 0.0,
    "lm_weight": 0.0,
    "nbest": 1,
    "use_itn": False,
    "sentence_timestamp": False,
    "vad_model": None,
    "punc_model": None,
}

DIMENSIONS: tuple[str, ...] = ("tone", "consonant", "vowel")

#: characters that are never part of the content sequence (design A.2: 去标点/空格)
_DROP_CATEGORY_PREFIXES = ("P", "Z", "C")


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _opencc():
    """OpenCC ``t2s`` converter (``None`` when the optional package is missing)."""
    try:  # pragma: no cover - depends on the optional package
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:  # noqa: BLE001 - identity fallback
        return None


def t2s(text: str) -> str:
    """Traditional -> simplified conversion (identity fallback)."""
    if not text:
        return ""
    converter = _opencc()
    if converter is None:  # pragma: no cover - the installed env has OpenCC
        return text
    return converter.convert(text)


def strip_punctuation_and_space(text: str) -> str:
    """NFKC-normalise and drop punctuation / separators / control / whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    out = []
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char)[0] in _DROP_CATEGORY_PREFIXES:
            continue
        out.append(char)
    return "".join(out)


def normalize_text(raw: str) -> str:
    """Canonical task-D text form: NFKC -> drop punctuation/space -> OpenCC t2s."""
    return t2s(strip_punctuation_and_space(raw))


# ---------------------------------------------------------------------------
# Character alignment, CER and S/D/I
# ---------------------------------------------------------------------------


def align_chars(gold: list[str], hyp: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Character-level Levenshtein alignment.

    Returns the edit script as a list of ``(op, gold_idx, hyp_idx)`` in sequence order,
    with ``op`` one of ``"match"``, ``"sub"``, ``"del"``, ``"ins"``.  ``del`` has
    ``hyp_idx=None``; ``ins`` has ``gold_idx=None``.  Ties are resolved deterministically
    (match < substitution < deletion < insertion), so the script is reproducible.
    """
    n, m = len(gold), len(hyp)
    # dp[i][j] = edit distance between gold[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        gi = gold[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if gi == hyp[j - 1] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)

    ops: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if gold[i - 1] == hyp[j - 1] else 1):
            op = "match" if gold[i - 1] == hyp[j - 1] else "sub"
            ops.append((op, i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    ops.reverse()
    return ops


def edit_distance(gold: list[str], hyp: list[str]) -> int:
    """Plain Levenshtein distance between two character sequences."""
    n, m = len(gold), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        gi = gold[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j - 1] + (0 if gi == hyp[j - 1] else 1), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[m]


def count_sdi(ops: list[tuple[str, int | None, int | None]]) -> dict[str, int]:
    """Count substitutions / deletions / insertions in an edit script."""
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for op, _gold_idx, _hyp_idx in ops:
        if op == "sub":
            counts["substitutions"] += 1
        elif op == "del":
            counts["deletions"] += 1
        elif op == "ins":
            counts["insertions"] += 1
    return counts


def char_cer(gold: list[str], hyp: list[str]) -> float | None:
    """Per-utterance character error rate (``None`` for an empty gold string)."""
    if not gold:
        return None
    return edit_distance(gold, hyp) / len(gold)


def normalization_rate(
    gold: list[str],
    hyp: list[str],
    error_flags: list[bool],
) -> dict[str, Any]:
    """ASR normalization rate over OMPAL-error gold characters.

    See the module docstring for the exact definition.  ``error_flags[i]`` marks
    ``gold[i]`` as an OMPAL-error character.  A deleted gold character counts as *not*
    equal; an inserted hypothesis character is ignored (it has no gold position).
    """
    if len(error_flags) != len(gold):
        raise ValueError("error_flags must align 1:1 with gold characters")
    ops = align_chars(gold, hyp)
    aligned = {gold_idx: hyp_idx for _op, gold_idx, hyp_idx in ops if gold_idx is not None}
    n_error = n_same = 0
    for idx, is_error in enumerate(error_flags):
        if not is_error:
            continue
        n_error += 1
        hyp_idx = aligned.get(idx)
        if hyp_idx is not None and hyp[hyp_idx] == gold[idx]:
            n_same += 1
    return {
        "n_ompal_error_chars": n_error,
        "n_asr_same_char": n_same,
        "rate": (n_same / n_error) if n_error else None,
    }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (numpy's default definition), pure Python."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


# ---------------------------------------------------------------------------
# Manifest / gold loading
# ---------------------------------------------------------------------------


def utt_id_from_item_id(item_id: str) -> str:
    """``B|00200101`` -> ``00200101`` (the OMPAL utterance id)."""
    if "|" in item_id:
        return item_id.split("|", 1)[1]
    return item_id


def load_items(
    manifest_path: Path,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load learner items from the manifest, dropping rows whose audio is missing.

    Returns ``(items, missing_audio_ids)``; a missing wav is reported (never silently
    transcribed) and the item is excluded from the run.
    """
    items: list[dict[str, Any]] = []
    missing: list[str] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["utt_id"] = utt_id_from_item_id(str(item["item_id"]))
            audio_path = Path(item["audio_path"])
            if not audio_path.is_absolute():
                audio_path = REPO_ROOT / audio_path
            item["_audio_abs"] = str(audio_path)
            if not audio_path.exists():
                missing.append(item["utt_id"])
                continue
            items.append(item)
    if limit is not None:
        items = items[:limit]
    return items, missing


def build_gold(item: dict[str, Any]) -> tuple[list[str], list[bool], dict[str, list[bool]]]:
    """Normalised gold characters + per-character OMPAL error flags.

    Returns ``(gold_chars, error_any, error_by_dimension)``.  The manifest texts are
    length-preserving under normalisation (checked); if that ever changes, the
    per-character fallback keeps the labels attached to the right characters.
    """
    labels = item.get("labels") or []
    text = str(item.get("text") or item.get("chars") or "")
    gold_chars = list(normalize_text(text))
    if len(gold_chars) != len(labels):
        gold_chars = [normalize_text(str(label.get("text") or "")) for label in labels]
        gold_chars = [char for char in gold_chars if char]
        if len(gold_chars) != len(labels):
            raise ValueError(f"{item.get('item_id')}: cannot align gold text with labels")
    error_by_dim: dict[str, list[bool]] = {dim: [] for dim in DIMENSIONS}
    error_any: list[bool] = []
    for label in labels:
        flags = {dim: int(label.get(dim, 1)) == 0 for dim in DIMENSIONS}
        for dim in DIMENSIONS:
            error_by_dim[dim].append(flags[dim])
        error_any.append(any(flags.values()))
    return gold_chars, error_any, error_by_dim


# ---------------------------------------------------------------------------
# Model (lazy import: importing this module never touches funasr / torch)
# ---------------------------------------------------------------------------


def funasr_version() -> str:  # pragma: no cover - requires the installed package
    try:
        from importlib.metadata import version

        return version("funasr")
    except Exception:  # noqa: BLE001
        return "unknown"


def load_model(device: str, *, log_level: str = "ERROR"):  # pragma: no cover - needs model
    """Construct the fixed FunASR Paraformer-zh model on ``device``."""
    from funasr import AutoModel

    t0 = time.time()
    model = AutoModel(
        model=MODEL_ID,
        hub=MODEL_HUB,
        device=device,
        disable_update=True,
        disable_pbar=True,
        log_level=log_level,
    )
    return model, time.time() - t0


def transcribe_file(model, wav_path: str) -> tuple[str, float]:
    """Transcribe one wav file; returns ``(raw_text, wall_seconds)``."""
    start = time.perf_counter()
    result = model.generate(
        input=wav_path,
        batch_size_s=DECODE_PARAMS["batch_size_s"],
        hotword=DECODE_PARAMS["hotword"],
        decoding_ctc_weight=DECODE_PARAMS["decoding_ctc_weight"],
        lm_weight=DECODE_PARAMS["lm_weight"],
        use_itn=DECODE_PARAMS["use_itn"],
        sentence_timestamp=DECODE_PARAMS["sentence_timestamp"],
        disable_pbar=True,
    )
    wall = time.perf_counter() - start
    if not result:
        return "", wall
    text = result[0].get("text", "")
    return text if isinstance(text, str) else str(text), wall


def audio_info(path: str) -> tuple[float, int]:
    """``(duration_s, sample_rate)`` using soundfile (no model dependency)."""
    import soundfile as sf

    info = sf.info(path)
    return float(info.duration), int(info.samplerate)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_rows(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    """Read an existing ``asr_outputs.jsonl`` (resume); returns ``(rows_by_utt, bad_lines)``."""
    rows: dict[str, dict[str, Any]] = {}
    bad = 0
    if not path.exists():
        return rows, bad
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            rows[str(row.get("utt_id"))] = row
    return rows, bad


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def model_provenance(model) -> dict[str, Any]:  # pragma: no cover - needs model
    """Model path / hashes for reproducibility (missing files are reported as ``None``)."""
    paths: dict[str, Any] = {"model_files_sha256": {}}
    model_path = getattr(model, "model_path", None)
    paths["model_path"] = str(model_path) if model_path else None
    if model_path:
        root = Path(str(model_path))
        for name in ("config.yaml", "configuration.json", "tokens.json", "am.mvn"):
            paths["model_files_sha256"][name] = _sha256_file(root / name)
        # model.pt is ~881 MB; hashing it once per run is acceptable provenance overhead
        paths["model_files_sha256"]["model.pt"] = _sha256_file(root / "model.pt")
    return paths


# ---------------------------------------------------------------------------
# ASR text-quality description (design A.5; computed from archived rows)
# ---------------------------------------------------------------------------


def text_quality_metrics(
    items: list[dict[str, Any]],
    rows_by_utt: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """CER / S-D-I / ASR normalization rate / per-learner CER distribution.

    Only ``ok`` rows with a non-empty ``raw_text`` are scored (matching the design's
    "100% ASR 文本生成并留档" acceptance); their count is reported explicitly.
    """
    n_items = 0
    n_gold_chars = 0
    n_hyp_chars = 0
    edit_sum = 0
    sdi_total = {"substitutions": 0, "deletions": 0, "insertions": 0}
    per_item_cer: list[float] = []
    exact = 0
    skipped: list[str] = []

    norm_error = {"n_ompal_error_chars": 0, "n_asr_same_char": 0}
    norm_rate_per_item: list[float] = []
    norm_by_dim = {dim: {"n_ompal_error_chars": 0, "n_asr_same_char": 0} for dim in DIMENSIONS}
    norm_correct = {"n_ompal_correct_chars": 0, "n_asr_same_char": 0}
    learner_dist: dict[str, dict[str, int]] = {}

    for item in items:
        utt_id = item["utt_id"]
        row = rows_by_utt.get(utt_id)
        if row is None or not row.get("ok") or not str(row.get("raw_text") or "").strip():
            skipped.append(utt_id)
            continue
        gold, error_any, error_by_dim = build_gold(item)
        hyp = list(normalize_text(str(row["raw_text"])))
        n_items += 1
        n_gold_chars += len(gold)
        n_hyp_chars += len(hyp)
        ops = align_chars(gold, hyp)
        dist = edit_distance(gold, hyp)
        edit_sum += dist
        sdi = count_sdi(ops)
        for key, value in sdi.items():
            sdi_total[key] += value
        item_cer = (dist / len(gold)) if gold else None
        if item_cer is not None:
            per_item_cer.append(item_cer)
            if dist == 0:
                exact += 1
        speaker = str(item.get("speaker") or "unknown")
        bucket = learner_dist.setdefault(speaker, {"edit_distance": 0, "n_gold_chars": 0})
        bucket["edit_distance"] += dist
        bucket["n_gold_chars"] += len(gold)

        # ASR normalization rate (same alignment as CER, per OMPAL-flagged character)
        aligned = {gi: hi for _op, gi, hi in ops if gi is not None}
        item_n_error = item_n_same = 0
        for idx, is_error in enumerate(error_any):
            hyp_idx = aligned.get(idx)
            same = hyp_idx is not None and hyp[hyp_idx] == gold[idx]
            if is_error:
                item_n_error += 1
                norm_error["n_ompal_error_chars"] += 1
                if same:
                    item_n_same += 1
                    norm_error["n_asr_same_char"] += 1
                for dim in DIMENSIONS:
                    if error_by_dim[dim][idx]:
                        norm_by_dim[dim]["n_ompal_error_chars"] += 1
                        if same:
                            norm_by_dim[dim]["n_asr_same_char"] += 1
            else:
                norm_correct["n_ompal_correct_chars"] += 1
                if same:
                    norm_correct["n_asr_same_char"] += 1
        if item_n_error:
            norm_rate_per_item.append(item_n_same / item_n_error)

    learner_cer = {
        speaker: (bucket["edit_distance"] / bucket["n_gold_chars"])
        for speaker, bucket in learner_dist.items()
        if bucket["n_gold_chars"]
    }
    learner_values = sorted(learner_cer.values())
    per_item_sorted = sorted(per_item_cer)

    def _rate(hit_key: str, total_key: str, bucket: dict[str, int]) -> float | None:
        total = bucket[total_key]
        return (bucket[hit_key] / total) if total else None

    return {
        "method": (
            "Char-level Levenshtein alignment (CSR) of normalized ASR text (NFKC, "
            "punctuation/whitespace removed, OpenCC t2s) against normalized OMPAL gold "
            "text. CER = edit_distance/len(gold). S/D/I from the same edit script; "
            "ratios use total gold characters as denominator. ASR normalization rate = "
            "share of OMPAL-error gold chars (label 0 on >=1 of tone/consonant/vowel) "
            "whose aligned ASR char is the same char; deleted error chars count as not "
            "same. Descriptive only, no causal claim."
        ),
        "n_scored_utterances": n_items,
        "n_skipped_utterances": len(skipped),
        "skipped_utterance_ids": skipped[:50],
        "n_gold_chars": n_gold_chars,
        "n_hyp_chars": n_hyp_chars,
        "cer_micro": (edit_sum / n_gold_chars) if n_gold_chars else None,
        "cer_mean_utterance": statistics.fmean(per_item_cer) if per_item_cer else None,
        "cer_median_utterance": statistics.median(per_item_cer) if per_item_cer else None,
        "cer_p90_utterance": _percentile(per_item_sorted, 0.9),
        "cer_max_utterance": per_item_sorted[-1] if per_item_sorted else None,
        "sentence_exact_rate": (exact / n_items) if n_items else None,
        "sdi": {
            **sdi_total,
            "total_errors": sum(sdi_total.values()),
            "substitution_rate": (sdi_total["substitutions"] / n_gold_chars)
            if n_gold_chars
            else None,
            "deletion_rate": (sdi_total["deletions"] / n_gold_chars) if n_gold_chars else None,
            "insertion_rate": (sdi_total["insertions"] / n_gold_chars) if n_gold_chars else None,
        },
        "normalization_rate": {
            "primary_definition": "any of tone/consonant/vowel == 0 (OMPAL error char)",
            "n_ompal_error_chars": norm_error["n_ompal_error_chars"],
            "n_asr_same_char": norm_error["n_asr_same_char"],
            "rate": _rate("n_asr_same_char", "n_ompal_error_chars", norm_error),
            "rate_mean_utterance": statistics.fmean(norm_rate_per_item)
            if norm_rate_per_item
            else None,
            "by_dimension": {
                dim: {
                    "n_ompal_error_chars": bucket["n_ompal_error_chars"],
                    "n_asr_same_char": bucket["n_asr_same_char"],
                    "rate": _rate("n_asr_same_char", "n_ompal_error_chars", bucket),
                }
                for dim, bucket in norm_by_dim.items()
            },
            "on_ompal_correct_chars": {
                "n_ompal_correct_chars": norm_correct["n_ompal_correct_chars"],
                "n_asr_same_char": norm_correct["n_asr_same_char"],
                "rate": _rate("n_asr_same_char", "n_ompal_correct_chars", norm_correct),
            },
        },
        "per_learner_cer": {
            "n_learners": len(learner_values),
            "definition": "micro per learner = sum edit distance / sum gold chars",
            "min": learner_values[0] if learner_values else None,
            "median": statistics.median(learner_values) if learner_values else None,
            "mean": statistics.fmean(learner_values) if learner_values else None,
            "p90": _percentile(learner_values, 0.9),
            "max": learner_values[-1] if learner_values else None,
            "per_learner": dict(sorted(learner_cer.items())),
        },
    }


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(
    requested: str,
    *,
    probe_audio: str,
) -> tuple[Any, str, list[dict[str, Any]]]:  # pragma: no cover - needs model
    """Build the model, trying ``mps`` before ``cpu`` when ``requested == "auto"``.

    Returns ``(model, device, attempts)``.  A device counts as usable only when a probe
    transcription returns a string without raising; otherwise the next candidate is
    tried (design: correctness over speed).
    """
    candidates = ["mps", "cpu"] if requested == "auto" else [requested]
    attempts: list[dict[str, Any]] = []
    for candidate in candidates:
        attempt: dict[str, Any] = {"device": candidate}
        try:
            model, load_s = load_model(candidate)
            attempt["load_s"] = round(load_s, 3)
            try:
                raw_text, wall = transcribe_file(model, probe_audio)
                attempt.update(ok=True, probe_text=raw_text, probe_wall_s=round(wall, 3))
                if not isinstance(raw_text, str):
                    raise TypeError(f"unexpected probe output type: {type(raw_text)!r}")
                attempts.append(attempt)
                return model, candidate, attempts
            except Exception as exc:  # noqa: BLE001 - record and fall back
                attempt.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                attempts.append(attempt)
                del model
        except Exception as exc:  # noqa: BLE001 - model construction failed
            attempt.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            attempts.append(attempt)
    raise RuntimeError(f"no usable ASR device: {attempts}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _row_template(
    item: dict[str, Any],
    *,
    model_id: str,
    version: str,
    device: str,
) -> dict[str, Any]:
    return {
        "utt_id": item["utt_id"],
        "speaker": item.get("speaker"),
        "audio_path": item.get("audio_path"),
        "raw_text": None,
        "text_norm": None,
        "duration_s": None,
        "sample_rate": None,
        "wall_s": None,
        "rtf": None,
        "device": device,
        "ok": False,
        "error": None,
        "model_id": model_id,
        "funasr_version": version,
        "decode_params": dict(DECODE_PARAMS),
        "timestamp": None,
    }


def run(args: argparse.Namespace) -> int:  # pragma: no cover - needs model / long run
    """Execute the transcription run and write the two deliverables."""
    started_at = _now_iso()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs_path = out_dir / "asr_outputs.jsonl"
    summary_path = out_dir / "d_asr_summary.json"

    items, missing_audio = load_items(Path(args.input_manifest), limit=args.limit)
    if not items:
        print("no items to transcribe", file=sys.stderr)
        return 1
    print(
        f"[d-asr] {len(items)} items to consider, {len(missing_audio)} missing audio, "
        f"manifest={args.input_manifest}",
        file=sys.stderr,
        flush=True,
    )

    existing: dict[str, dict[str, Any]] = {}
    n_bad_lines = 0
    if args.resume:
        existing, n_bad_lines = read_rows(outputs_path)
        # drop rows that are not usable so they get retried
        existing = {
            utt_id: row
            for utt_id, row in existing.items()
            if row.get("ok") and str(row.get("raw_text") or "").strip()
        }
        if existing:
            ordered = [existing[item["utt_id"]] for item in items if item["utt_id"] in existing]
            write_rows_atomic(outputs_path, ordered)
        else:
            write_rows_atomic(outputs_path, [])
        print(
            f"[d-asr] resume: {len(existing)} ok rows already present "
            f"({n_bad_lines} malformed lines ignored)",
            file=sys.stderr,
            flush=True,
        )
    else:
        write_rows_atomic(outputs_path, [])

    todo = [item for item in items if item["utt_id"] not in existing]
    print(f"[d-asr] {len(todo)} files to transcribe", file=sys.stderr, flush=True)

    model = device = None
    attempts: list[dict[str, Any]] = []
    model_load_s = 0.0
    provenance: dict[str, Any] = {"model_path": None, "model_files_sha256": {}}
    version = funasr_version()
    if todo:
        probe_audio = todo[0]["_audio_abs"]
        model, device, attempts = select_device(args.device, probe_audio=probe_audio)
        # the device probe transcribes its file once; that file is re-run in the loop so
        # every archived row keeps an independent per-file wall time.
        model_load_s = sum(float(a.get("load_s") or 0.0) for a in attempts)
        provenance = model_provenance(model)
        print(
            f"[d-asr] device={device} attempts={json.dumps(attempts, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )
    else:
        device = args.device if args.device != "auto" else "cpu"
        print("[d-asr] nothing to do; writing summary from existing rows", file=sys.stderr)

    rows_by_utt: dict[str, dict[str, Any]] = dict(existing)
    n_ok = n_fail = 0
    failures: list[dict[str, Any]] = []
    loop_start = time.time()
    progress_every = max(1, min(100, len(todo) // 10 or 1))

    for idx, item in enumerate(todo, start=1):
        row = _row_template(item, model_id=MODEL_ID, version=version, device=str(device))
        row["timestamp"] = _now_iso()
        try:
            duration, sample_rate = audio_info(item["_audio_abs"])
            row["duration_s"] = round(duration, 3)
            row["sample_rate"] = sample_rate
            raw_text, wall = transcribe_file(model, item["_audio_abs"])
            row["wall_s"] = round(wall, 4)
            row["rtf"] = round(wall / duration, 4) if duration else None
            row["raw_text"] = raw_text
            row["text_norm"] = normalize_text(raw_text)
            if not raw_text.strip():
                row["ok"] = False
                row["error"] = "empty transcript (model returned no tokens)"
                print(f"[d-asr] WARN empty transcript for {item['utt_id']}", file=sys.stderr)
            else:
                row["ok"] = True
            if row["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                failures.append({"utt_id": item["utt_id"], "error": row["error"]})
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            row["ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            n_fail += 1
            failures.append({"utt_id": item["utt_id"], "error": row["error"]})
            print(
                f"[d-asr] ERROR {item['utt_id']}: {row['error']}", file=sys.stderr, flush=True
            )
        rows_by_utt[item["utt_id"]] = row
        append_row(outputs_path, row)
        if idx % progress_every == 0 or idx == len(todo):
            elapsed = time.time() - loop_start
            rate = idx / elapsed if elapsed else 0.0
            eta = (len(todo) - idx) / rate if rate else float("nan")
            print(
                f"[d-asr] {idx}/{len(todo)} ok={n_ok} fail={n_fail} "
                f"{rate:.2f} files/s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                file=sys.stderr,
                flush=True,
            )

    loop_wall = time.time() - loop_start
    # final file: deterministic manifest order, one row per considered item
    final_rows = [rows_by_utt[item["utt_id"]] for item in items if item["utt_id"] in rows_by_utt]
    write_rows_atomic(outputs_path, final_rows)

    quality = text_quality_metrics(items, rows_by_utt)
    # summary counts / timing describe the archived rows (this run + resumed rows), so a
    # resumed invocation still reports the whole-corpus numbers.
    n_ok = sum(1 for row in final_rows if row.get("ok"))
    n_fail = len(items) - n_ok
    archived_wall = sum(row.get("wall_s") or 0.0 for row in final_rows)
    archived_audio = sum(row.get("duration_s") or 0.0 for row in final_rows)
    archived_rtf = [
        (row["wall_s"] / row["duration_s"])
        for row in final_rows
        if row.get("wall_s") is not None and row.get("duration_s")
    ]
    summary: dict[str, Any] = {
        "task": "D-asr",
        "design": "docs/sentence-experiment-design.md v0.2, 附录 A (A.2 / A.5)",
        "input_manifest": str(Path(args.input_manifest)),
        "manifest_sha256": _sha256_file(Path(args.input_manifest)),
        "outputs_file": str(outputs_path),
        "run_date": started_at,
        "finished_at": _now_iso(),
        "n_items": len(items),
        "n_ok": n_ok,
        "n_fail": n_fail,
        "n_missing_audio": len(missing_audio),
        "missing_audio_ids": missing_audio[:50],
        "failure_count": len(failures),
        "failures": failures,
        "device": str(device),
        "device_requested": args.device,
        "device_attempts": attempts,
        "model_id": MODEL_ID,
        "model_hub": MODEL_HUB,
        "model_revision": MODEL_REVISION,
        "funasr_version": version,
        "torch_version": _torch_version(),
        "decode_params": dict(DECODE_PARAMS),
        "model_provenance": provenance,
        "limit": args.limit,
        "partial_run": args.limit is not None,
        "resume": bool(args.resume),
        "timing": {
            "this_run_model_load_s": round(model_load_s, 2),
            "this_run_inference_loop_wall_s": round(loop_wall, 2),
            "archived_total_wall_s": round(archived_wall, 2),
            "archived_total_audio_duration_s": round(archived_audio, 2),
            "rtf_micro": (archived_wall / archived_audio) if archived_audio else None,
            "rtf_median_per_file": statistics.median(archived_rtf) if archived_rtf else None,
            "wall_per_file_s_median": statistics.median(
                [row["wall_s"] for row in final_rows if row.get("wall_s") is not None]
            ),
        },
        "text_quality": quality,
        "notes": [
            "ASR is FunASR Paraformer-zh, fixed id/version/decode params; raw output "
            "archived verbatim in asr_outputs.jsonl (no human correction).",
            "Device policy: try mps, fall back to cpu; every attempt's error is in "
            "device_attempts and the device that actually ran is 'device'.",
            "Metrics are descriptive ASR text-quality descriptions (design A.5); no "
            "causal claim about automatic correction is made.",
            "text_norm = NFKC + punctuation/whitespace removal + OpenCC t2s; gold and "
            "ASR text are normalised with the same function.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[d-asr] done: n_ok={n_ok} n_fail={n_fail} cer_micro={quality['cer_micro']} "
        f"norm_rate={quality['normalization_rate']['rate']} -> {out_dir}",
        file=sys.stderr,
        flush=True,
    )
    return 0 if n_fail == 0 else 2


def _torch_version() -> str:  # pragma: no cover - installed env
    try:
        from importlib.metadata import version

        return version("torch")
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task D ASR: fixed FunASR Paraformer-zh transcripts for OMPAL learner wavs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--limit", type=int, default=None, help="transcribe only the first N items (smoke)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cpu"],
        help="auto = try mps then cpu; the device that actually ran is recorded",
    )
    parser.add_argument(
        "--resume", action="store_true", help="keep ok rows from an existing asr_outputs.jsonl"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DECODE_PARAMS["batch_size_s"],
        help="forwarded to FunASR as batch_size_s (dynamic batch seconds; 1 = per file)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    DECODE_PARAMS["batch_size_s"] = args.batch_size
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
