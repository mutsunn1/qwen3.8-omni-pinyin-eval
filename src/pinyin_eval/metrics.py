"""Metric computations for the pinyin recognition / detection experiments.

Contract: ``docs/contracts.md`` section 6.

Inputs are rows produced by :meth:`pinyin_eval.records.ItemResult.to_row` -- either
plain ``dict`` objects or ``pandas.Series`` (anything supporting ``.get``).  Outputs are
plain JSON-serialisable objects (``dict`` / ``list`` / ``float`` / ``int`` / ``None``).
The single exception is :func:`confusion_matrix`, which returns a ``pandas.DataFrame``
as agreed in the contract.

Conventions used throughout this module (see the contract):

* A prediction of ``None`` / ``NaN`` means "parse failure".  Such rows are removed
  from every numerator/denominator and counted in ``n_failures``.
* A rate whose denominator is ``0`` is reported as ``None`` instead of raising or
  silently returning ``0``.
* ``macro_f1`` follows ``sklearn.metrics.f1_score(average="macro")``: one F1 per label
  in the label set (labels observed in gold **or** pred, sorted, unless ``labels`` is
  given), averaged with equal weights.  Labels with zero support get F1 ``0.0`` when
  ``labels`` is passed explicitly, exactly like scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ab_metrics",
    "classification_metrics",
    "confusion_matrix",
    "confusion_rows",
    "detection_chance_baseline",
    "detection_metrics",
    "majority_baseline",
    "random_baseline",
    "recognition_metric_pack",
    "recognition_metrics",
]

_RECOGNITION_DIMENSIONS: tuple[str, ...] = ("initial", "final", "tone")

#: ``dimension`` -> (gold key, pred key) for :func:`confusion_rows`.
_CONFUSION_COLUMNS: dict[str, tuple[str, str]] = {
    "initial": ("gold_initial", "pred_initial"),
    "final": ("gold_final", "pred_final"),
    "tone": ("gold_tone", "pred_tone"),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_py(value: Any) -> Any:
    """Convert numpy scalars to plain Python values (JSON friendliness)."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas NA; empty strings are **not** missing."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):  # e.g. pandas.NA for which bool() raises
        return True


def _sorted_labels(values: Sequence[Any]) -> list[Any]:
    """Unique labels in deterministic sorted order (fallback: by type/str)."""
    unique = list(dict.fromkeys(_to_py(value) for value in values))
    try:
        return sorted(unique)
    except TypeError:
        return sorted(unique, key=lambda value: (type(value).__name__, str(value)))


def _parse_ok(row: Mapping[str, Any]) -> bool:
    """Whether the row is parseable.  A missing ``parse_status`` counts as ``ok``."""
    status = row.get("parse_status", "ok")
    if _is_missing(status):
        return False
    return _to_py(status) == "ok"


def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a stored prediction to ``bool``; anything unexpected becomes ``None``."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1, 0.0, 1.0):
            return bool(value)
    return None


# ---------------------------------------------------------------------------
# Recognition metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    gold: Sequence[Any],
    pred: Sequence[Any | None],
    *,
    labels: Sequence[Any] | None = None,
) -> dict:
    """Accuracy + macro-F1 for a single classification dimension.

    Returns ``{"n", "n_failures", "accuracy", "macro_f1", "labels", "per_class"}``.
    ``pred`` entries that are missing (``None`` / NaN) are dropped from the scored
    set and counted in ``n_failures``.
    """
    gold_list = list(gold)
    pred_list = list(pred)
    if len(gold_list) != len(pred_list):
        raise ValueError(
            "classification_metrics: gold and pred must have the same length, "
            f"got {len(gold_list)} and {len(pred_list)}"
        )

    n_failures = 0
    pairs: list[tuple[Any, Any]] = []
    for gold_value, pred_value in zip(gold_list, pred_list, strict=True):
        if _is_missing(pred_value):
            n_failures += 1
            continue
        pairs.append((_to_py(gold_value), _to_py(pred_value)))
    n = len(pairs)

    if labels is None:
        label_list = _sorted_labels([g for g, _ in pairs] + [p for _, p in pairs])
    else:
        label_list = list(dict.fromkeys(_to_py(label) for label in labels))

    gold_counts = Counter(g for g, _ in pairs)
    pred_counts = Counter(p for _, p in pairs)
    tp_counts = Counter(g for g, p in pairs if g == p)
    correct = sum(tp_counts.values())

    per_class: dict[Any, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in label_list:
        tp = tp_counts.get(label, 0)
        support = gold_counts.get(label, 0)
        predicted = pred_counts.get(label, 0)
        false_positives = predicted - tp
        false_negatives = support - tp
        precision = tp / (tp + false_positives) if tp + false_positives else 0.0
        recall = tp / (tp + false_negatives) if tp + false_negatives else 0.0
        if precision + recall:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        f1_values.append(float(f1))

    return {
        "n": int(n),
        "n_failures": int(n_failures),
        "accuracy": _ratio(correct, n),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "labels": label_list,
        "per_class": per_class,
    }


def recognition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Initial / final / tone classification metrics plus joint accuracy.

    Only rows with ``parse_status == "ok"`` are used; the others are reported in the
    top-level ``n_failures`` (nested per-dimension dicts may additionally report a
    prediction-level failure if an ``ok`` row lacks one of the prediction fields).
    ``joint_accuracy`` requires all three dimensions to match (a missing prediction
    counts as incorrect).
    """
    row_list = list(rows)
    ok_rows = [row for row in row_list if _parse_ok(row)]
    n = len(ok_rows)

    result: dict[str, Any] = {"n": int(n), "n_failures": int(len(row_list) - n)}
    for dimension in _RECOGNITION_DIMENSIONS:
        gold = [row.get(f"gold_{dimension}") for row in ok_rows]
        pred = [row.get(f"pred_{dimension}") for row in ok_rows]
        result[dimension] = classification_metrics(gold, pred)

    joint_correct = 0
    for row in ok_rows:
        if all(
            not _is_missing(row.get(f"pred_{dimension}"))
            and _to_py(row.get(f"pred_{dimension}")) == _to_py(row.get(f"gold_{dimension}"))
            for dimension in _RECOGNITION_DIMENSIONS
        ):
            joint_correct += 1
    result["joint_accuracy"] = _ratio(joint_correct, n)
    return result


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def detection_metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Error detection metrics over rows carrying ``detection_role`` / ``pred_consistent``.

    ``pred_consistent`` is the model's *consistency* judgement about the audio and
    the claimed target: ``True`` means the model judged them **consistent**, ``False``
    means it judged them **inconsistent**.  The positive class of the detection task
    is "inconsistent", hence::

        judged_inconsistent = pred_consistent is False
        TP = pos & judged_inconsistent    FP = neg & judged_inconsistent
        TN = neg & judged_consistent      FN = pos & judged_consistent

    Rows with an unparseable or missing ``pred_consistent`` only contribute to
    ``n_failures_neg`` / ``n_failures_pos``.  Any rate with a zero denominator is
    ``None``.
    """
    true_positives = false_positives = true_negatives = false_negatives = 0
    n_neg = n_pos = n_failures_neg = n_failures_pos = 0

    for row in rows:
        role = row.get("detection_role")
        if role not in ("neg", "pos"):
            raise ValueError(
                "detection_metrics: every row needs detection_role in {'neg', 'pos'}, "
                f"got {role!r}"
            )
        judged_consistent = _coerce_bool(row.get("pred_consistent")) if _parse_ok(row) else None
        judged_inconsistent = None if judged_consistent is None else not judged_consistent
        if judged_inconsistent is None:
            if role == "neg":
                n_failures_neg += 1
            else:
                n_failures_pos += 1
            continue
        if role == "neg":
            n_neg += 1
            if judged_inconsistent:
                false_positives += 1
            else:
                true_negatives += 1
        else:
            n_pos += 1
            if judged_inconsistent:
                true_positives += 1
            else:
                false_negatives += 1

    error_precision = _ratio(true_positives, true_positives + false_positives)
    error_recall = _ratio(true_positives, true_positives + false_negatives)
    if (
        error_precision is None
        or error_recall is None
        or error_precision + error_recall == 0
    ):
        error_f1 = None
    else:
        error_f1 = float(
            2.0 * error_precision * error_recall / (error_precision + error_recall)
        )
    specificity = _ratio(true_negatives, true_negatives + false_positives)
    if error_recall is None or specificity is None:
        balanced_accuracy = None
    else:
        balanced_accuracy = float((error_recall + specificity) / 2.0)

    return {
        "n_neg": int(n_neg),
        "n_pos": int(n_pos),
        "n_failures_neg": int(n_failures_neg),
        "n_failures_pos": int(n_failures_pos),
        "tp": int(true_positives),
        "fp": int(false_positives),
        "tn": int(true_negatives),
        "fn": int(false_negatives),
        "error_precision": error_precision,
        "error_recall": error_recall,
        "error_f1": error_f1,
        "fpr": _ratio(false_positives, false_positives + true_negatives),
        "fnr": _ratio(false_negatives, false_negatives + true_positives),
        "accuracy": _ratio(
            true_positives + true_negatives,
            true_positives + false_positives + true_negatives + false_negatives,
        ),
        "balanced_accuracy": balanced_accuracy,
    }


# ---------------------------------------------------------------------------
# A/B choice metrics
# ---------------------------------------------------------------------------


def ab_metrics(rows: Sequence[Mapping[str, Any]]) -> dict:
    """A/B forced-choice accuracy and choice counts.

    Returns ``{"n", "n_failures", "accuracy", "choice_counts"}``.  ``n`` counts rows
    with ``parse_status == "ok"``; an ``ok`` row without a ``pred_choice`` counts as
    wrong but is not re-counted as a parse failure.
    """
    row_list = list(rows)
    ok_rows = [row for row in row_list if _parse_ok(row)]
    n = len(ok_rows)
    correct = 0
    choice_counts = {"A": 0, "B": 0}
    for row in ok_rows:
        choice = _to_py(row.get("pred_choice"))
        if choice in ("A", "B"):
            choice_counts[choice] += 1
        if not _is_missing(choice) and choice == _to_py(row.get("correct_choice")):
            correct += 1
    return {
        "n": int(n),
        "n_failures": int(len(row_list) - n),
        "accuracy": _ratio(correct, n),
        "choice_counts": choice_counts,
    }


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------


def confusion_matrix(
    gold: Sequence[Any],
    pred: Sequence[Any | None],
    *,
    labels: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Confusion matrix as a DataFrame (rows = gold, columns = pred).

    Missing predictions are skipped (they cannot be placed in the matrix); use
    :func:`classification_metrics` for the failure count.
    """
    gold_list = list(gold)
    pred_list = list(pred)
    if len(gold_list) != len(pred_list):
        raise ValueError(
            "confusion_matrix: gold and pred must have the same length, "
            f"got {len(gold_list)} and {len(pred_list)}"
        )
    pairs = [
        (_to_py(gold_value), _to_py(pred_value))
        for gold_value, pred_value in zip(gold_list, pred_list, strict=True)
        if not _is_missing(pred_value)
    ]
    if labels is None:
        label_list = _sorted_labels([g for g, _ in pairs] + [p for _, p in pairs])
    else:
        label_list = list(dict.fromkeys(_to_py(label) for label in labels))

    counts = Counter(pairs)
    data = [
        [counts.get((gold_label, pred_label), 0) for pred_label in label_list]
        for gold_label in label_list
    ]
    return pd.DataFrame(
        data,
        index=pd.Index(label_list, name="gold"),
        columns=pd.Index(label_list, name="pred"),
        dtype="int64",
    )


def confusion_rows(
    rows: Sequence[Mapping[str, Any]], *, dimension: str
) -> list[dict[str, Any]]:
    """Long-format confusion counts for one recognition dimension.

    ``dimension`` is ``"initial"``, ``"final"`` or ``"tone"``.  Returns a
    deterministically sorted list of ``{"gold", "pred", "count"}`` entries.  Only
    parseable rows with both a gold and a predicted value are counted.
    """
    if dimension not in _CONFUSION_COLUMNS:
        raise ValueError(
            "confusion_rows: dimension must be one of "
            f"{sorted(_CONFUSION_COLUMNS)}, got {dimension!r}"
        )
    gold_key, pred_key = _CONFUSION_COLUMNS[dimension]

    counts: Counter[tuple[Any, Any]] = Counter()
    for row in rows:
        if not _parse_ok(row):
            continue
        gold_value = row.get(gold_key)
        pred_value = row.get(pred_key)
        if _is_missing(gold_value) or _is_missing(pred_value):
            continue
        counts[(_to_py(gold_value), _to_py(pred_value))] += 1

    pairs = list(counts)
    try:
        pairs.sort()
    except TypeError:
        pairs.sort(key=lambda pair: tuple((type(v).__name__, str(v)) for v in pair))
    return [
        {"gold": gold_label, "pred": pred_label, "count": int(counts[(gold_label, pred_label)])}
        for gold_label, pred_label in pairs
    ]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def majority_baseline(gold: Sequence[Any]) -> dict:
    """Majority-class baseline over the gold labels.

    Returns ``{"label", "accuracy", "n"}``; ties are broken by sorted label order
    (same convention as ``sklearn.dummy.DummyClassifier(strategy="most_frequent")``).
    ``label`` / ``accuracy`` are ``None`` for an empty input.
    """
    values = [_to_py(value) for value in gold if not _is_missing(value)]
    n = len(values)
    if n == 0:
        return {"label": None, "accuracy": None, "n": 0}
    counts = Counter(values)
    top_count = max(counts.values())
    candidates = _sorted_labels([label for label, count in counts.items() if count == top_count])
    return {"label": candidates[0], "accuracy": float(top_count / n), "n": int(n)}


def random_baseline(n_classes: int) -> float:
    """Chance-level accuracy for a balanced ``n_classes``-way task."""
    if n_classes <= 0:
        raise ValueError(f"random_baseline: n_classes must be positive, got {n_classes!r}")
    return 1.0 / float(n_classes)


def detection_chance_baseline() -> float:
    """Chance-level accuracy on the balanced (1:1) detection set."""
    return 0.5


# ---------------------------------------------------------------------------
# Convenience pack
# ---------------------------------------------------------------------------


def recognition_metric_pack(row_groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict:
    """Convenience wrapper around :func:`recognition_metrics`.

    ``row_groups`` maps a condition id (or any group label) to its rows.  Returns::

        {"overall": recognition_metrics(all rows),
         "by_voice": {voice: recognition_metrics(rows of that voice)},
         "by_condition": {group label: recognition_metrics(rows of that group)}}

    Group iteration order is deterministic (sorted by type/str before building the
    output dict).
    """
    groups = sorted(
        row_groups.items(), key=lambda item: (type(item[0]).__name__, str(item[0]))
    )
    all_rows: list[Mapping[str, Any]] = []
    by_condition: dict[Any, dict] = {}
    for key, group_rows in groups:
        rows = list(group_rows)
        by_condition[_to_py(key)] = recognition_metrics(rows)
        all_rows.extend(rows)

    by_voice_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in all_rows:
        voice = row.get("voice")
        voice_key = "unknown" if _is_missing(voice) else str(_to_py(voice))
        by_voice_rows.setdefault(voice_key, []).append(row)
    by_voice = {
        voice: recognition_metrics(by_voice_rows[voice]) for voice in sorted(by_voice_rows)
    }
    return {
        "overall": recognition_metrics(all_rows),
        "by_voice": by_voice,
        "by_condition": by_condition,
    }
