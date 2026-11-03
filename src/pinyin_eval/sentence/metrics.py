"""Aggregation and inference for the sentence-level pinyin experiment (design §7, §9, §10).

Every public function takes the flat rows produced by
:mod:`pinyin_eval.sentence.scoring` and returns a JSON-serialisable ``dict``
(numpy scalars are converted to Python scalars).  Only rows whose
``parse_status`` is ``"ok"`` enter a numerator/denominator unless explicitly
stated; a metric whose denominator is zero is reported as ``None`` (never NaN,
never a silently-wrong ``0``).

Statistical conventions (design §9.2):

* bootstrap = cluster bootstrap over ``cluster_id`` (AISHELL speaker / OMPAL
  learner), 2000 draws, percentile 95 % interval, ``stratify_key=None`` (the
  single-syllable pipeline's ``voice`` stratification does not apply here);
* C1 vs C2 comparisons are **paired** on the same items and resample the same
  clusters on both sides; the reported difference is always
  ``metric(C1) - metric(C2)``;
* McNemar p-values are Holm-corrected within a task family; paired-bootstrap
  differences are reported as effect + CI with ``p_value = None``.

Task A metric definitions (all over parse-ok rows)
--------------------------------------------------
``joint_accuracy`` / ``initial_accuracy`` / ``final_accuracy`` / ``tone_accuracy``
    ``Σ correct / Σ n_gold`` over aligned gold syllables.
``sentence_exact_rate``       fraction of items with ``sentence_exact == True``.
``insertion_rate`` / ``deletion_rate``   ``Σ n_insertions / Σ n_gold`` and
    ``Σ n_deletions / Σ n_gold``.
``cer_tone`` / ``cer_stripped``          ``Σ numerator / Σ denominator``.
``alignment_coverage``        ``Σ n_matched / Σ n_gold``.
N1 diagnostics mirror these from the N1 sandhi fields; ``delta_vs_n0`` is
``n1 - n0`` for joint accuracy and tone-sensitive CER.

Task B metric definitions
-------------------------
Positive class = gold label ``0`` (wrong pronunciation); ``pred_error = not *_ok``
(see :mod:`pinyin_eval.sentence.scoring`).  Per dimension:

``f1_error``   ``2tp / (2tp + fp + fn)`` (F1 of the *error* class);
``f1_ok``      ``2tn / (2tn + fn + fp)`` (F1 of the *correct* class);
``macro_f1``   mean of the F1s that are mathematically defined (a class with no
               support and no predictions is skipped, like sklearn's behaviour on
               observed labels);
``fpr``        ``fp / (fp + tn)``; ``fnr`` ``fn / (fn + tp)``;
``accuracy``   ``(tp + tn) / (tp + fp + fn + tn)``.
``zero_initial_consonant`` repeats the consonant-dimension statistics on the
zero-initial sub-counts (design §6.4: reported separately).

Task C metric definitions
-------------------------
Native items are excluded from every correlation / error metric and reported
once as the ``native_anchor``: the native row count, the mean of the per-item
three-dimension gold / predicted means (``mean_gold`` / ``mean_pred``) and the
per-dimension gold / predicted means (``per_dimension``).  ``mean_dimension``
uses the per-item mean of the three gold and predicted scores.  Pearson and
Spearman (Pearson of average ranks, ties handled) return ``None`` when either
side has zero variance or ``n < 2``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..stats import bootstrap_ci, holm_correct, mcnemar_test, paired_bootstrap_difference

__all__ = [
    "DEFAULT_SEED",
    "acceptance_checks",
    "compare_conditions",
    "task_a_metrics",
    "task_b_metrics",
    "task_c_metrics",
]

#: fixed bootstrap seed (design §9 / §12); the cli may pass a config value.
DEFAULT_SEED: int = 20261101
DEFAULT_N_BOOT: int = 2000
DEFAULT_ALPHA: float = 0.05

#: acceptance thresholds (design §10.2)
PARSE_SUCCESS_THRESHOLD: float = 0.95
ALIGNMENT_COVERAGE_THRESHOLD: float = 0.95
BUDGET_CNY: float = 20.0

_B_DIMENSIONS: tuple[str, ...] = ("tone", "consonant", "vowel")
_C_DIMENSIONS: tuple[str, ...] = ("accuracy", "fluency", "prosody")
_C_MEAN: str = "mean_dimension"
_SPECIAL_SUBSETS: tuple[str, ...] = ("stable", "neutral", "yibu", "erhua")

_A_HEADLINE_NAMES: tuple[str, ...] = (
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_py(value: Any) -> Any:
    """Convert numpy scalars to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas NA; empty strings are **not** missing."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def _as_float(value: Any) -> float | None:
    """Coerce a metric value to ``float``; missing / non-numeric becomes ``None``."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair_float(left: Any, right: Any) -> float | None:
    """``left - right`` for two optional floats (``None`` when either is missing)."""
    value_left = _as_float(left)
    value_right = _as_float(right)
    if value_left is None or value_right is None:
        return None
    return value_left - value_right


def _row_num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Numeric row field with a default (missing / NaN -> default)."""
    value = _as_float(row.get(key))
    return default if value is None else value


def _row_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    """Integer row field with a default."""
    value = _as_float(row.get(key))
    return default if value is None else int(value)


def _sum_num(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    """Sum one numeric row field over ``rows``."""
    return float(sum(_row_num(row, key) for row in rows))


def _sum_int(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    """Sum one integer row field over ``rows``."""
    return int(sum(_row_int(row, key) for row in rows))


def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is zero."""
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _parse_ok(row: Mapping[str, Any]) -> bool:
    """Whether the row is parseable; a missing ``parse_status`` counts as ``ok``."""
    status = row.get("parse_status", "ok")
    if _is_missing(status):
        return False
    return _to_py(status) == "ok"


def _jsonable(value: Any) -> Any:
    """Recursively convert a result structure into JSON-serialisable values."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    if not values:
        return None
    return float(sum(values)) / len(values)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson product-moment correlation; ``None`` for ``n < 2`` or zero variance."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mean_x
        dy = y - mean_y
        covariance += dx * dy
        variance_x += dx * dx
        variance_y += dy * dy
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    correlation = covariance / math.sqrt(variance_x * variance_y)
    return float(max(-1.0, min(1.0, correlation)))


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks with tied values replaced by their average rank."""
    array = np.asarray(values, dtype=float)
    count = array.size
    order = np.argsort(array, kind="stable")
    ranks = np.empty(count, dtype=float)
    position = 0
    while position < count:
        end = position
        while end + 1 < count and array[order[end + 1]] == array[order[position]]:
            end += 1
        ranks[order[position:end + 1]] = (position + end) / 2.0 + 1.0
        position = end + 1
    return [float(rank) for rank in ranks]


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _mae(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Mean absolute error."""
    if not xs or len(xs) != len(ys):
        return None
    return float(sum(abs(x - y) for x, y in zip(xs, ys, strict=True)) / len(xs))


def _rmse(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Root mean squared error."""
    if not xs or len(xs) != len(ys):
        return None
    return math.sqrt(
        sum((x - y) ** 2 for x, y in zip(xs, ys, strict=True)) / len(xs)
    )


def _regression_stats(
    xs: Sequence[float], ys: Sequence[float]
) -> dict[str, float | int | None]:
    """``{"n", "pearson", "spearman", "mae", "rmse"}`` for one gold/pred pair set."""
    return {
        "n": len(xs),
        "pearson": _pearson(xs, ys),
        "spearman": _spearman(xs, ys),
        "mae": _mae(xs, ys),
        "rmse": _rmse(xs, ys),
    }


# ---------------------------------------------------------------------------
# Task A — whole-utterance pinyin transcription
# ---------------------------------------------------------------------------


def _a_headline(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Headline task-A metrics over (already parse-ok) rows (see module docstring)."""
    n_gold = _sum_int(rows, "n_gold")
    n_items = len(rows)
    return {
        "joint_accuracy": _ratio(_sum_int(rows, "n_joint_correct"), n_gold),
        "initial_accuracy": _ratio(_sum_int(rows, "n_initial_correct"), n_gold),
        "final_accuracy": _ratio(_sum_int(rows, "n_final_correct"), n_gold),
        "tone_accuracy": _ratio(_sum_int(rows, "n_tone_correct"), n_gold),
        "sentence_exact_rate": _ratio(
            sum(1 for row in rows if bool(row.get("sentence_exact"))), n_items
        ),
        "insertion_rate": _ratio(_sum_int(rows, "n_insertions"), n_gold),
        "deletion_rate": _ratio(_sum_int(rows, "n_deletions"), n_gold),
        "cer_tone": _ratio(
            _sum_num(rows, "cer_tone_num"), _sum_num(rows, "cer_tone_den")
        ),
        "cer_stripped": _ratio(
            _sum_num(rows, "cer_stripped_num"), _sum_num(rows, "cer_stripped_den")
        ),
        "alignment_coverage": _ratio(_sum_int(rows, "n_matched"), n_gold),
    }


def _a_n1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """N1 sandhi diagnostic block + ``delta_vs_n0`` (see module docstring)."""
    n_gold = _sum_int(rows, "n_gold")
    n_items = len(rows)
    n0 = _a_headline(rows)
    n1 = {
        "n1_joint_accuracy": _ratio(_sum_int(rows, "n1_n_joint_correct"), n_gold),
        "n1_initial_accuracy": _ratio(_sum_int(rows, "n1_n_initial_correct"), n_gold),
        "n1_final_accuracy": _ratio(_sum_int(rows, "n1_n_final_correct"), n_gold),
        "n1_tone_accuracy": _ratio(_sum_int(rows, "n1_n_tone_correct"), n_gold),
        "n1_cer_tone": _ratio(
            _sum_num(rows, "n1_cer_tone_num"), _sum_num(rows, "n1_cer_tone_den")
        ),
        "n1_sentence_exact_rate": _ratio(
            sum(1 for row in rows if bool(row.get("n1_sentence_exact"))), n_items
        ),
    }
    n1["delta_vs_n0"] = {
        "joint_accuracy": _pair_float(
            n1["n1_joint_accuracy"], n0["joint_accuracy"]
        ),
        "cer_tone": _pair_float(n1["n1_cer_tone"], n0["cer_tone"]),
    }
    return n1


def _subset_rows(
    rows: Sequence[Mapping[str, Any]], name: str
) -> list[Mapping[str, Any]]:
    """Filter rows to one design subset: stable / neutral / yibu / erhua."""
    if name == "stable":
        return [row for row in rows if bool(row.get("stable"))]
    return [
        row
        for row in rows
        if bool((row.get("flags") or {}).get(name))
    ]


def _per_speaker(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive per-speaker task-A statistics (parse-ok rows only)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row.get("speaker")
        key = str(key) if not _is_missing(key) else str(row.get("cluster_id") or "")
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for speaker in sorted(groups):
        group = groups[speaker]
        n_gold = _sum_int(group, "n_gold")
        result.append(
            {
                "speaker": speaker,
                "n_items": len(group),
                "joint_accuracy": _ratio(
                    _sum_int(group, "n_joint_correct"), n_gold
                ),
                "initial_accuracy": _ratio(
                    _sum_int(group, "n_initial_correct"), n_gold
                ),
                "final_accuracy": _ratio(
                    _sum_int(group, "n_final_correct"), n_gold
                ),
                "tone_accuracy": _ratio(
                    _sum_int(group, "n_tone_correct"), n_gold
                ),
            }
        )
    return result


def task_a_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "cluster_id",
) -> dict[str, Any]:
    """Aggregate task-A rows into the design §7.1 report block.

    Top-level keys: ``n_items``, ``n_parse_ok``, ``n_parse_failures``,
    ``parse_failure_rate``, ``n_skipped_no_gold``, ``headline``, ``with_ci``,
    ``subsets`` (stable / neutral / yibu / erhua), ``n1``, ``per_speaker``.

    ``with_ci`` uses the cluster bootstrap over ``cluster_key`` with
    ``stratify_key=None``; every headline metric gets ``{estimate, lo, hi}``.
    Subsets are flat dicts ``{n_items, n_parse_ok, <headline metrics>}``; their
    ``n_items`` count all filtered rows, the headline metrics only parse-ok ones.
    ``headline`` and ``with_ci`` use parse-ok rows.  ``n_skipped_no_gold`` counts
    parse-ok items without gold syllables (should be 0; gold-unalignable items
    are excluded at data prep).
    """
    all_rows = list(rows)
    ok_rows = [row for row in all_rows if _parse_ok(row)]
    n_items = len(all_rows)
    n_parse_ok = len(ok_rows)
    n_parse_failures = n_items - n_parse_ok

    headline = _a_headline(ok_rows)
    with_ci = bootstrap_ci(
        ok_rows,
        _a_headline,
        cluster_key=cluster_key,
        stratify_key=None,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    subsets: dict[str, dict[str, Any]] = {}
    for name in _SPECIAL_SUBSETS:
        subset = _subset_rows(all_rows, name)
        subset_ok = [row for row in subset if _parse_ok(row)]
        block = _a_headline(subset_ok)
        block["n_items"] = len(subset)
        block["n_parse_ok"] = len(subset_ok)
        subsets[name] = block

    return _jsonable(
        {
            "n_items": n_items,
            "n_parse_ok": n_parse_ok,
            "n_parse_failures": n_parse_failures,
            "parse_failure_rate": _ratio(n_parse_failures, n_items),
            "n_skipped_no_gold": sum(
                1 for row in ok_rows if _row_int(row, "n_gold") <= 0
            ),
            "headline": headline,
            "with_ci": {name: with_ci.get(name) for name in _A_HEADLINE_NAMES},
            "subsets": subsets,
            "n1": _a_n1(ok_rows),
            "per_speaker": _per_speaker(ok_rows),
        }
    )


# ---------------------------------------------------------------------------
# Task B — whole-utterance pronunciation evaluation
# ---------------------------------------------------------------------------


def _b_confusion_stats(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    """Confusion-matrix statistics for the task-B *error* positive class."""
    f1_error = _ratio(2 * tp, 2 * tp + fp + fn)
    f1_ok = _ratio(2 * tn, 2 * tn + fn + fp)
    f1_values = [value for value in (f1_error, f1_ok) if value is not None]
    macro_f1 = _mean(f1_values)
    total = tp + fp + fn + tn
    return {
        "n_units": int(total),
        "macro_f1": macro_f1,
        "f1_error": f1_error,
        "f1_ok": f1_ok,
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "accuracy": _ratio(tp + tn, total),
        "confusion": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }


def _b_sum_confusion(rows: Iterable[Mapping[str, Any]]) -> tuple[int, int, int, int]:
    """Sum ``tp/fp/fn/tn`` over rows."""
    tp = fp = fn = tn = 0
    for row in rows:
        tp += _row_int(row, "tp")
        fp += _row_int(row, "fp")
        fn += _row_int(row, "fn")
        tn += _row_int(row, "tn")
    return tp, fp, fn, tn


def _b_dimension_rows(
    rows: Sequence[Mapping[str, Any]], dimension: str
) -> list[Mapping[str, Any]]:
    """Parse-ok rows belonging to one task-B dimension."""
    return [
        row
        for row in rows
        if str(row.get("dimension") or "") == dimension
    ]


def _b_dimension_stats(
    rows: Sequence[Mapping[str, Any]], dimension: str
) -> dict[str, Any]:
    """Statistics dict for one task-B dimension (including ``n_items``)."""
    dimension_rows = _b_dimension_rows(rows, dimension)
    tp, fp, fn, tn = _b_sum_confusion(dimension_rows)
    stats = _b_confusion_stats(tp, fp, fn, tn)
    items = {
        str(row.get("item_id"))
        for row in dimension_rows
        if not _is_missing(row.get("item_id"))
    }
    stats["n_items"] = len(items)
    return stats


def _b_zero_initial_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Zero-initial consonant statistics from the ``n_zero_initial_*`` sub-counts."""
    consonant_rows = _b_dimension_rows(rows, "consonant")
    tp = fp = fn = tn = 0
    n_items = 0
    for row in consonant_rows:
        ztp = _row_int(row, "n_zero_initial_tp")
        zfp = _row_int(row, "n_zero_initial_fp")
        zfn = _row_int(row, "n_zero_initial_fn")
        ztn = _row_int(row, "n_zero_initial_tn")
        tp += ztp
        fp += zfp
        fn += zfn
        tn += ztn
        if ztp + zfp + zfn + ztn > 0:
            n_items += 1
    stats = _b_confusion_stats(tp, fp, fn, tn)
    stats["n_items"] = n_items
    return stats


def _b_macro_metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """Bootstrap metric function: macro-F1 of every dimension in one pass."""
    result: dict[str, float | None] = {}
    for dimension in _B_DIMENSIONS:
        tp, fp, fn, tn = _b_sum_confusion(_b_dimension_rows(rows, dimension))
        result[f"{dimension}_macro_f1"] = _b_confusion_stats(tp, fp, fn, tn)[
            "macro_f1"
        ]
    return result


def _b_per_learner(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-learner task-B macro-F1 / F1 / FPR / FNR (parse-ok rows)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row.get("learner")
        if _is_missing(key) or not str(key):
            key = row.get("cluster_id")
        if _is_missing(key) or not str(key):
            key = row.get("speaker")
        key = str(key) if not _is_missing(key) else ""
        groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for learner in sorted(groups):
        group = groups[learner]
        item_ids = {
            str(row.get("item_id"))
            for row in group
            if not _is_missing(row.get("item_id"))
        }
        entry: dict[str, Any] = {
            "learner": learner,
            "n_items": len(item_ids),
            "macro_f1": {},
            "f1_error": {},
            "fpr": {},
            "fnr": {},
        }
        for dimension in _B_DIMENSIONS:
            stats = _b_dimension_stats(group, dimension)
            entry["macro_f1"][dimension] = stats["macro_f1"]
            entry["f1_error"][dimension] = stats["f1_error"]
            entry["fpr"][dimension] = stats["fpr"]
            entry["fnr"][dimension] = stats["fnr"]
        result.append(entry)
    return result


def task_b_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "cluster_id",
) -> dict[str, Any]:
    """Aggregate task-B rows into the design §7.2 report block.

    Top-level keys: ``n_items``, ``n_parse_ok``, ``n_skipped_no_gold``,
    ``alignment_coverage``, ``tone`` / ``consonant`` / ``vowel`` (each
    ``{n_items, n_units, macro_f1, f1_error, f1_ok, fpr, fnr, accuracy,
    confusion}``), ``zero_initial_consonant`` (same shape, from the zero-initial
    sub-counts), ``with_ci`` (``{dimension: {"macro_f1": {estimate, lo, hi}}}``,
    cluster bootstrap, ``stratify_key=None``), ``per_learner``.

    ``alignment_coverage`` = ``Σ n_matched / Σ n_units`` over the tone rows
    (identical on every dimension).  Rows with ``parse_status != "ok"`` never
    exist for task B (a parse failure produces no per-unit verdicts); the count
    is therefore ``n_items`` of the rows that *are* present.
    """
    all_rows = list(rows)
    ok_rows = [row for row in all_rows if _parse_ok(row)]
    item_ids = {
        str(row.get("item_id"))
        for row in ok_rows
        if not _is_missing(row.get("item_id"))
    }
    tone_rows = _b_dimension_rows(ok_rows, "tone")
    skipped_items: set[str] = {
        str(row.get("item_id"))
        for row in tone_rows
        if _row_int(row, "n_units") <= 0 and not _is_missing(row.get("item_id"))
    }

    result: dict[str, Any] = {
        "n_items": len(item_ids),
        "n_parse_ok": len(item_ids),
        "n_skipped_no_gold": len(skipped_items),
        "alignment_coverage": _ratio(
            _sum_int(tone_rows, "n_matched"), _sum_int(tone_rows, "n_units")
        ),
    }
    for dimension in _B_DIMENSIONS:
        result[dimension] = _b_dimension_stats(ok_rows, dimension)
    result["zero_initial_consonant"] = _b_zero_initial_stats(ok_rows)

    ci = bootstrap_ci(
        ok_rows,
        _b_macro_metric,
        cluster_key=cluster_key,
        stratify_key=None,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    result["with_ci"] = {
        dimension: {"macro_f1": ci.get(f"{dimension}_macro_f1")}
        for dimension in _B_DIMENSIONS
    }
    result["per_learner"] = _b_per_learner(ok_rows)
    return _jsonable(result)


# ---------------------------------------------------------------------------
# Task C — whole-utterance reading scoring
# ---------------------------------------------------------------------------


def _c_float(value: Any) -> float | None:
    """Optional float accessor for task-C score fields."""
    return _as_float(value)


def _c_pairs(
    rows: Iterable[Mapping[str, Any]], dimension: str
) -> tuple[list[float], list[float]]:
    """Gold/pred paired values for one task-C dimension or for the dimension mean.

    Rows missing either side (parse failure / incomplete prediction) are skipped,
    which is what excludes native items from the correlations only if the caller
    filtered them out; the callers below always pass non-native rows.
    """
    gold_values: list[float] = []
    pred_values: list[float] = []
    for row in rows:
        if dimension == _C_MEAN:
            golds = [_c_float(row.get(f"gold_{name}")) for name in _C_DIMENSIONS]
            preds = [_c_float(row.get(f"pred_{name}")) for name in _C_DIMENSIONS]
            if any(value is None for value in golds + preds):
                continue
            gold_values.append(float(sum(golds) / 3.0))  # type: ignore[arg-type]
            pred_values.append(float(sum(preds) / 3.0))  # type: ignore[arg-type]
        else:
            gold = _c_float(row.get(f"gold_{dimension}"))
            pred = _c_float(row.get(f"pred_{dimension}"))
            if gold is None or pred is None:
                continue
            gold_values.append(gold)
            pred_values.append(pred)
    return gold_values, pred_values


def _c_regression_metric(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Bootstrap metric function: Pearson r and MAE for every task-C dimension."""
    result: dict[str, float | None] = {}
    for dimension in (*_C_DIMENSIONS, _C_MEAN):
        gold_values, pred_values = _c_pairs(rows, dimension)
        result[f"{dimension}_pearson"] = _pearson(gold_values, pred_values)
        result[f"{dimension}_mae"] = _mae(gold_values, pred_values)
    return result


def _c_per_learner(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-learner task-C correlations and errors (non-native, parse-ok rows)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = row.get("learner")
        if _is_missing(key) or not str(key):
            key = row.get("cluster_id")
        if _is_missing(key) or not str(key):
            key = row.get("speaker")
        key = str(key) if not _is_missing(key) else ""
        groups.setdefault(key, []).append(row)

    result: list[dict[str, Any]] = []
    for learner in sorted(groups):
        group = groups[learner]
        items = {
            str(row.get("item_id"))
            for row in group
            if not _is_missing(row.get("item_id"))
        }
        entry: dict[str, Any] = {
            "learner": learner,
            "n_items": len(items),
            "pearson": {},
            "spearman": {},
            "mae": {},
            "rmse": {},
        }
        for dimension in (*_C_DIMENSIONS, _C_MEAN):
            gold_values, pred_values = _c_pairs(group, dimension)
            stats = _regression_stats(gold_values, pred_values)
            entry["pearson"][dimension] = stats["pearson"]
            entry["spearman"][dimension] = stats["spearman"]
            entry["mae"][dimension] = stats["mae"]
            entry["rmse"][dimension] = stats["rmse"]
        result.append(entry)
    return result


def _native_anchor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Human-ceiling anchor for the native utterances (excluded from correlations).

    ``mean_gold`` / ``mean_pred`` are the means over native items of the per-item
    three-dimension mean; an item enters each mean only when all three of its
    scores on that side are present (so the two means may be over different
    counts, e.g. a native item whose reply failed to parse still enters
    ``mean_gold`` but not ``mean_pred``).  ``per_dimension`` gives the mean gold
    and predicted score of each individual dimension over the native rows where
    that value is available.  ``n`` is the number of native rows, independent of
    parse status.
    """
    native_rows = [row for row in rows if bool(row.get("native"))]
    gold_means: list[float] = []
    pred_means: list[float] = []
    per_dimension_values: dict[str, tuple[list[float], list[float]]] = {
        dimension: ([], []) for dimension in _C_DIMENSIONS
    }
    for row in native_rows:
        golds = [_c_float(row.get(f"gold_{name}")) for name in _C_DIMENSIONS]
        preds = [_c_float(row.get(f"pred_{name}")) for name in _C_DIMENSIONS]
        if all(value is not None for value in golds):
            gold_means.append(float(sum(golds) / 3.0))  # type: ignore[arg-type]
        if all(value is not None for value in preds):
            pred_means.append(float(sum(preds) / 3.0))  # type: ignore[arg-type]
        for dimension, gold, pred in zip(_C_DIMENSIONS, golds, preds, strict=True):
            if gold is not None:
                per_dimension_values[dimension][0].append(gold)
            if pred is not None:
                per_dimension_values[dimension][1].append(pred)
    return {
        "n": len(native_rows),
        "mean_gold": _mean(gold_means),
        "mean_pred": _mean(pred_means),
        "per_dimension": {
            dimension: {
                "mean_gold": _mean(gold_values),
                "mean_pred": _mean(pred_values),
            }
            for dimension, (gold_values, pred_values) in per_dimension_values.items()
        },
    }


def task_c_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "cluster_id",
) -> dict[str, Any]:
    """Aggregate task-C rows into the design §7.3 report block.

    Top-level keys: ``n_items``, ``n_parse_ok``, ``n_parse_failures``,
    ``parse_failure_rate``, ``n_skipped_no_gold``, ``alignment_coverage``,
    ``accuracy`` / ``fluency`` / ``prosody`` / ``mean_dimension`` (each
    ``{n, pearson, spearman, mae, rmse}``), ``native_anchor``
    (``{"n", "mean_gold", "mean_pred", "per_dimension": {dimension:
    {"mean_gold", "mean_pred"}}}`` — the native items' mean three-dimension
    scores plus per-dimension means; native items are excluded from every other
    metric),
    ``with_ci`` (``{dimension: {"pearson": {estimate, lo, hi}, "mae": ...}}``,
    cluster bootstrap, ``stratify_key=None``), ``per_learner``.

    ``alignment_coverage`` = fraction of parse-ok rows whose three gold and three
    predicted scores are all present (task C returns no per-character text, so
    this is the available text/score alignment notion).  Correlations use
    non-native parse-ok rows only; ``mean_dimension`` is the per-item mean of the
    three dimensions.
    """
    all_rows = list(rows)
    ok_rows = [row for row in all_rows if _parse_ok(row)]
    non_native_rows = [row for row in ok_rows if not bool(row.get("native"))]
    n_items = len(all_rows)
    n_parse_ok = len(ok_rows)

    n_aligned = 0
    for row in ok_rows:
        golds = [_c_float(row.get(f"gold_{name}")) for name in _C_DIMENSIONS]
        preds = [_c_float(row.get(f"pred_{name}")) for name in _C_DIMENSIONS]
        if all(value is not None for value in golds + preds):
            n_aligned += 1

    result: dict[str, Any] = {
        "n_items": n_items,
        "n_parse_ok": n_parse_ok,
        "n_parse_failures": n_items - n_parse_ok,
        "parse_failure_rate": _ratio(n_items - n_parse_ok, n_items),
        "n_skipped_no_gold": sum(
            1
            for row in ok_rows
            if all(_c_float(row.get(f"gold_{name}")) is None for name in _C_DIMENSIONS)
        ),
        "alignment_coverage": _ratio(n_aligned, n_parse_ok),
    }
    for dimension in (*_C_DIMENSIONS, _C_MEAN):
        gold_values, pred_values = _c_pairs(non_native_rows, dimension)
        result[dimension] = _regression_stats(gold_values, pred_values)
    result["native_anchor"] = _native_anchor(all_rows)

    ci = bootstrap_ci(
        non_native_rows,
        _c_regression_metric,
        cluster_key=cluster_key,
        stratify_key=None,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )
    result["with_ci"] = {
        dimension: {
            "pearson": ci.get(f"{dimension}_pearson"),
            "mae": ci.get(f"{dimension}_mae"),
        }
        for dimension in (*_C_DIMENSIONS, _C_MEAN)
    }
    result["per_learner"] = _c_per_learner(non_native_rows)
    return _jsonable(result)


# ---------------------------------------------------------------------------
# C1 vs C2 paired comparison
# ---------------------------------------------------------------------------


def _unique_by_item(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    """Index one task-A/C row per ``item_id``; duplicate ids raise ``ValueError``."""
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if _is_missing(item_id):
            raise ValueError(f"{label}: row without item_id: {dict(row)!r}")
        key = str(item_id)
        if key in index:
            raise ValueError(f"{label}: duplicate row for item_id {key!r}")
        index[key] = row
    return index


def _assert_same_items(
    index_c1: Mapping[str, Mapping[str, Any]],
    index_c2: Mapping[str, Mapping[str, Any]],
    label: str,
) -> list[str]:
    """Require identical item-id sets (task A/C score rows exist for every item)."""
    items_c1 = set(index_c1)
    items_c2 = set(index_c2)
    if items_c1 != items_c2:
        only_c1 = sorted(items_c1 - items_c2)[:5]
        only_c2 = sorted(items_c2 - items_c1)[:5]
        raise ValueError(
            f"{label}: item_id sets differ between the two conditions "
            f"(only C1: {only_c1}, only C2: {only_c2})"
        )
    return sorted(items_c1)


def _with_p_none(
    entries: Mapping[str, Mapping[str, Any] | None]
) -> dict[str, dict[str, Any]]:
    """Add ``"p_value": None`` to every paired-bootstrap entry (CI instead of p)."""
    result: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        if entry is None:
            result[name] = {"estimate": None, "lo": None, "hi": None, "p_value": None}
            continue
        result[name] = {**dict(entry), "p_value": None}
    return result


def _compare_task_a(
    rows_c1: Sequence[Mapping[str, Any]],
    rows_c2: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
    cluster_key: str,
) -> dict[str, Any]:
    """Task-A paired comparison: bootstrap diffs + McNemar on sentence exactness."""
    index_c1 = _unique_by_item(rows_c1, "compare_conditions(task='A', C1)")
    index_c2 = _unique_by_item(rows_c2, "compare_conditions(task='A', C2)")
    item_ids = _assert_same_items(index_c1, index_c2, "compare_conditions(task='A')")
    paired_ids = [
        item_id
        for item_id in item_ids
        if _parse_ok(index_c1[item_id]) and _parse_ok(index_c2[item_id])
    ]

    paired_c1 = [index_c1[item_id] for item_id in paired_ids]
    paired_c2 = [index_c2[item_id] for item_id in paired_ids]
    differences = paired_bootstrap_difference(
        paired_c1,
        paired_c2,
        _a_headline,
        cluster_key=cluster_key,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )

    sentence_exact_c1 = [bool(index_c1[item_id].get("sentence_exact")) for item_id in paired_ids]
    sentence_exact_c2 = [bool(index_c2[item_id].get("sentence_exact")) for item_id in paired_ids]
    mcnemar = {"sentence_exact": mcnemar_test(sentence_exact_c1, sentence_exact_c2)}

    joint_correct_c1: list[bool] = []
    joint_correct_c2: list[bool] = []
    for item_id in paired_ids:
        for row, outcomes in (
            (index_c1[item_id], joint_correct_c1),
            (index_c2[item_id], joint_correct_c2),
        ):
            n_gold = _row_int(row, "n_gold")
            outcomes.append(n_gold > 0 and _row_int(row, "n_joint_correct") == n_gold)

    counts = {
        "n_items": len(paired_ids),
        "joint_all_correct_c1": sum(joint_correct_c1),
        "joint_all_correct_c2": sum(joint_correct_c2),
        "both": sum(1 for left, right in zip(joint_correct_c1, joint_correct_c2, strict=True) if left and right),
        "c1_only": sum(1 for left, right in zip(joint_correct_c1, joint_correct_c2, strict=True) if left and not right),
        "c2_only": sum(1 for left, right in zip(joint_correct_c1, joint_correct_c2, strict=True) if right and not left),
        "neither": sum(1 for left, right in zip(joint_correct_c1, joint_correct_c2, strict=True) if not left and not right),
    }

    pvalues = [mcnemar["sentence_exact"]["p_value"]]
    adjusted = holm_correct(pvalues)
    return {
        "task": "A",
        "direction": "C1 - C2",
        "n_items_c1": len(index_c1),
        "n_items_c2": len(index_c2),
        "n_items_paired": len(paired_ids),
        "differences": _with_p_none(differences),
        "mcnemar": mcnemar,
        "holm": {"sentence_exact": adjusted[0]},
        "per_item_joint_counts": counts,
    }


def _b_records_by_item(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Index task-B rows as ``item_id -> {dimension: row}``; duplicates raise."""
    index: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if _is_missing(item_id):
            raise ValueError(f"{label}: row without item_id: {dict(row)!r}")
        dimension = str(row.get("dimension") or "")
        if not dimension:
            raise ValueError(f"{label}: row without dimension: {dict(row)!r}")
        item_index = index.setdefault(str(item_id), {})
        if dimension in item_index:
            raise ValueError(
                f"{label}: duplicate row for item_id {item_id!r} dimension {dimension!r}"
            )
        item_index[dimension] = row
    return index


def _compare_task_b(
    rows_c1: Sequence[Mapping[str, Any]],
    rows_c2: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
    cluster_key: str,
) -> dict[str, Any]:
    """Task-B paired comparison: bootstrap CI of the macro-F1 difference per dimension.

    Task-B score rows do not exist for parse failures (no per-character verdicts),
    so the paired item set is the intersection and the number of items present in
    only one condition is reported instead of raising.  There is no per-item
    macro-F1, hence no McNemar test for this task (design §7.2 + §9.2).
    """
    index_c1 = _b_records_by_item(rows_c1, "compare_conditions(task='B', C1)")
    index_c2 = _b_records_by_item(rows_c2, "compare_conditions(task='B', C2)")
    ids_c1 = set(index_c1)
    ids_c2 = set(index_c2)
    paired_ids = sorted(ids_c1 & ids_c2)

    differences: dict[str, dict[str, Any]] = {}
    for dimension in _B_DIMENSIONS:
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for item_id in paired_ids:
            row_c1 = index_c1[item_id].get(dimension)
            row_c2 = index_c2[item_id].get(dimension)
            if row_c1 is None or row_c2 is None:
                continue
            if not _parse_ok(row_c1) or not _parse_ok(row_c2):
                continue
            pairs.append((row_c1, row_c2))
        paired_c1 = [left for left, _ in pairs]
        paired_c2 = [right for _, right in pairs]

        def metric(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
            tp, fp, fn, tn = _b_sum_confusion(rows)
            return {"macro_f1": _b_confusion_stats(tp, fp, fn, tn)["macro_f1"]}

        result = paired_bootstrap_difference(
            paired_c1,
            paired_c2,
            metric,
            cluster_key=cluster_key,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        differences[dimension] = _with_p_none({"macro_f1": result.get("macro_f1")})[
            "macro_f1"
        ]

    return {
        "task": "B",
        "direction": "C1 - C2",
        "n_items_c1": len(ids_c1),
        "n_items_c2": len(ids_c2),
        "n_items_paired": len(paired_ids),
        "n_items_missing_in_c1": len(ids_c2 - ids_c1),
        "n_items_missing_in_c2": len(ids_c1 - ids_c2),
        "differences": {"macro_f1": differences},
        "mcnemar": {},
        "holm": {},
    }


def _compare_task_c(
    rows_c1: Sequence[Mapping[str, Any]],
    rows_c2: Sequence[Mapping[str, Any]],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
    cluster_key: str,
) -> dict[str, Any]:
    """Task-C paired comparison: bootstrap CI of Pearson-r and MAE differences."""
    index_c1 = _unique_by_item(rows_c1, "compare_conditions(task='C', C1)")
    index_c2 = _unique_by_item(rows_c2, "compare_conditions(task='C', C2)")
    item_ids = _assert_same_items(index_c1, index_c2, "compare_conditions(task='C')")
    paired_ids = [
        item_id
        for item_id in item_ids
        if _parse_ok(index_c1[item_id]) and _parse_ok(index_c2[item_id])
    ]
    paired_c1 = [index_c1[item_id] for item_id in paired_ids]
    paired_c2 = [index_c2[item_id] for item_id in paired_ids]

    pearson_differences: dict[str, Any] = {}
    mae_differences: dict[str, Any] = {}
    for dimension in (*_C_DIMENSIONS, _C_MEAN):

        def metric(
            rows: Sequence[Mapping[str, Any]], dimension: str = dimension
        ) -> dict[str, float | None]:
            gold_values, pred_values = _c_pairs(
                (row for row in rows if not bool(row.get("native"))), dimension
            )
            return {
                "pearson": _pearson(gold_values, pred_values),
                "mae": _mae(gold_values, pred_values),
            }

        result = paired_bootstrap_difference(
            paired_c1,
            paired_c2,
            metric,
            cluster_key=cluster_key,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        with_p = _with_p_none(result)
        pearson_differences[dimension] = with_p.get(
            "pearson", {"estimate": None, "lo": None, "hi": None, "p_value": None}
        )
        mae_differences[dimension] = with_p.get(
            "mae", {"estimate": None, "lo": None, "hi": None, "p_value": None}
        )

    return {
        "task": "C",
        "direction": "C1 - C2",
        "n_items_c1": len(index_c1),
        "n_items_c2": len(index_c2),
        "n_items_paired": len(paired_ids),
        "differences": {"pearson": pearson_differences, "mae": mae_differences},
        "mcnemar": {},
        "holm": {},
    }


def compare_conditions(
    rows_c1: Sequence[Mapping[str, Any]],
    rows_c2: Sequence[Mapping[str, Any]],
    task: str,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    cluster_key: str = "cluster_id",
) -> dict[str, Any]:
    """Paired C1-vs-C2 comparison for one task (design §9.2).

    Difference direction is always ``metric(C1) - metric(C2)``.  Only items
    parse-ok in **both** conditions enter the paired statistics.

    * A: paired cluster bootstrap of every headline metric (joint / initial /
      final / tone accuracy, CERs, insertion/deletion, coverage, sentence-exact
      rate) + McNemar on per-item ``sentence_exact`` + per-item all-joint-correct
      counts.  McNemar p-values are Holm-corrected.
    * B: paired cluster bootstrap of the macro-F1 difference per dimension
      (no McNemar: there is no per-item macro-F1 quantity).
    * C: paired cluster bootstrap of the Pearson-r and MAE differences per
      dimension and for the three-dimension mean.

    Returns ``{"task", "direction", "n_items_c1", "n_items_c2",
    "n_items_paired", "differences", "mcnemar", "holm"}``; each bootstrap entry
    is ``{"estimate", "lo", "hi", "p_value": None}`` (CI instead of p), the
    McNemar entries are the :func:`~pinyin_eval.stats.mcnemar_test` dicts.
    """
    task_id = str(task).upper()
    if task_id == "A":
        block = _compare_task_a(
            rows_c1, rows_c2, n_boot=n_boot, alpha=alpha, seed=seed,
            cluster_key=cluster_key,
        )
    elif task_id == "B":
        block = _compare_task_b(
            rows_c1, rows_c2, n_boot=n_boot, alpha=alpha, seed=seed,
            cluster_key=cluster_key,
        )
    elif task_id == "C":
        block = _compare_task_c(
            rows_c1, rows_c2, n_boot=n_boot, alpha=alpha, seed=seed,
            cluster_key=cluster_key,
        )
    else:
        raise ValueError(f"compare_conditions: unknown sentence task {task!r}")
    return _jsonable(block)


# ---------------------------------------------------------------------------
# Acceptance checks (design §10.2)
# ---------------------------------------------------------------------------


def _metric_alignment_coverage(metrics: Any, task: str) -> float | None:
    """Pull the task-specific alignment coverage out of a metrics dict."""
    if not isinstance(metrics, Mapping):
        return None
    if task == "A":
        headline = metrics.get("headline")
        if isinstance(headline, Mapping):
            return _as_float(headline.get("alignment_coverage"))
        return None
    return _as_float(metrics.get("alignment_coverage"))


def _parse_success_block(entry: Any) -> dict[str, Any]:
    """Parse-success block for one task from its usage dict."""
    usage = entry if isinstance(entry, Mapping) else {}
    n_calls = int(_as_float(usage.get("n_calls")) or 0.0)
    n_errors = int(_as_float(usage.get("n_errors")) or 0.0)
    n_parse_failures = int(_as_float(usage.get("n_parse_failures")) or 0.0)
    rate: float | None = None
    if n_calls > 0:
        rate = max(0.0, min(1.0, (n_calls - n_errors - n_parse_failures) / n_calls))
    return {
        "rate": rate,
        "n_calls": n_calls,
        "n_errors": n_errors,
        "n_parse_failures": n_parse_failures,
        "threshold": PARSE_SUCCESS_THRESHOLD,
        "passed": rate is not None and rate >= PARSE_SUCCESS_THRESHOLD,
    }


def acceptance_checks(
    a_metrics: Mapping[str, Any] | None,
    b_metrics: Mapping[str, Any] | None,
    c_metrics: Mapping[str, Any] | None,
    usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Design §10.2 acceptance gates for the sentence-level pilot.

    Parameters
    ----------
    a_metrics, b_metrics, c_metrics:
        Outputs of :func:`task_a_metrics` / :func:`task_b_metrics` /
        :func:`task_c_metrics` (``None`` makes the corresponding alignment check
        fail rather than crash).
    usage:
        ``{task: {"n_calls", "n_errors", "n_parse_failures", "cost_cny"}}``.
        ``parse_success_rate = (n_calls - n_errors - n_parse_failures) / n_calls``.

    Returns
    -------
    dict
        ``{"parse_success": {"A"/"B"/"C": {rate, n_calls, n_errors,
        n_parse_failures, threshold, passed}}, "alignment_coverage": {"A"/"B"/
        "C": {value, threshold, passed}}, "cost_cny": {value, threshold,
        passed}, "passed": bool}``.  A missing metric / usage value fails the
        corresponding gate (``passed = False``) instead of being ignored.
    """
    usage_map = usage if isinstance(usage, Mapping) else {}
    parse_success = {
        task: _parse_success_block(usage_map.get(task)) for task in ("A", "B", "C")
    }
    alignment_coverage: dict[str, dict[str, Any]] = {}
    for task, metrics in (("A", a_metrics), ("B", b_metrics), ("C", c_metrics)):
        value = _metric_alignment_coverage(metrics, task)
        alignment_coverage[task] = {
            "value": value,
            "threshold": ALIGNMENT_COVERAGE_THRESHOLD,
            "passed": value is not None and value >= ALIGNMENT_COVERAGE_THRESHOLD,
        }

    costs: list[float] = []
    for task in ("A", "B", "C"):
        entry = usage_map.get(task)
        if not isinstance(entry, Mapping):
            continue
        cost = _as_float(entry.get("cost_cny"))
        if cost is not None:
            costs.append(cost)
    total_cost = float(sum(costs)) if costs else None
    cost_block = {
        "value": total_cost,
        "threshold": BUDGET_CNY,
        "passed": total_cost is not None and total_cost <= BUDGET_CNY,
    }

    passed = (
        all(entry["passed"] for entry in parse_success.values())
        and all(entry["passed"] for entry in alignment_coverage.values())
        and cost_block["passed"]
    )
    return _jsonable(
        {
            "parse_success": parse_success,
            "alignment_coverage": alignment_coverage,
            "cost_cny": cost_block,
            "passed": passed,
        }
    )
