"""Cluster bootstrap, paired comparisons, McNemar tests and multiple comparisons.

Contract: ``docs/contracts.md`` section 7.

Statistical protocol (from the design document): bootstrap resamples **clusters**
(``cluster_id`` = ``audio_id``), not individual rows -- all rows of the same cluster
are drawn together so the within-audio correlation is preserved -- and ``voice`` is
used as a stratification variable (each stratum is resampled independently with the
same number of clusters as observed, then the strata are merged).  Pairwise
comparisons resample the same clusters on both sides.

Only ``numpy`` and the standard library are used; scipy is deliberately avoided.
For the large-sample McNemar approximation the chi-square(1) survival function is
computed with :func:`math.erfc`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

MetricFn = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, float]]

__all__ = [
    "bh_correct",
    "bootstrap_ci",
    "holm_correct",
    "mcnemar_test",
    "paired_bootstrap_difference",
    "paired_outcomes",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_py(value: Any) -> Any:
    """Convert numpy scalars to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas NA; empty strings stay valid."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return True


def _to_float(value: Any) -> float | None:
    """Convert a metric value to ``float``; missing / non-numeric becomes ``None``."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    """Coerce a stored correctness value to ``bool``; anything unexpected is ``None``."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1, 0.0, 1.0):
            return bool(value)
    return None


def _parse_ok(row: Mapping[str, Any]) -> bool:
    """Whether the row is parseable; a missing ``parse_status`` counts as ``ok``."""
    status = row.get("parse_status", "ok")
    if _is_missing(status):
        return False
    return _to_py(status) == "ok"


def _sorted_scalars(values: Sequence[Any]) -> list[Any]:
    values = list(values)
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: (type(value).__name__, str(value)))


def _validate_bootstrap_args(n_boot: int, alpha: float) -> None:
    if not isinstance(n_boot, int) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _call_metric(
    metric_fn: MetricFn, rows: Sequence[Mapping[str, Any]]
) -> Mapping[str, float]:
    result = metric_fn(rows)
    if not isinstance(result, Mapping):
        raise TypeError(
            "metric_fn must return a mapping of metric name -> value, "
            f"got {type(result).__name__}"
        )
    return result


# ---------------------------------------------------------------------------
# Cluster resampling
# ---------------------------------------------------------------------------


def _build_strata(
    rows: Sequence[Mapping[str, Any]],
    cluster_key: str,
    stratify_key: str | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group row positions into ``stratum -> (cluster sizes, flattened positions)``.

    Cluster order is the order of first appearance in ``rows``; strata are sorted
    deterministically.  A row without a ``cluster_key`` value becomes its own
    singleton cluster.  The stratum of a cluster is taken from its first row.
    """
    clusters: dict[Any, list[int]] = {}
    for index, row in enumerate(rows):
        cluster_id = row.get(cluster_key)
        if _is_missing(cluster_id):
            cluster_id = ("__row__", index)
        else:
            cluster_id = _to_py(cluster_id)
        clusters.setdefault(cluster_id, []).append(index)

    if not clusters:
        return []

    strata: dict[Any, list[list[int]]] = {}
    for positions in clusters.values():
        if stratify_key is None:
            stratum: Any = ""
        else:
            value = rows[positions[0]].get(stratify_key)
            stratum = "" if _is_missing(value) else _to_py(value)
        strata.setdefault(stratum, []).append(positions)

    layout: list[tuple[np.ndarray, np.ndarray]] = []
    for key in _sorted_scalars(list(strata)):
        positions_per_cluster = strata[key]
        sizes = np.array([len(positions) for positions in positions_per_cluster], dtype=np.int64)
        flattened = np.fromiter(
            (index for positions in positions_per_cluster for index in positions),
            dtype=np.int64,
            count=int(sizes.sum()),
        )
        layout.append((sizes, flattened))
    return layout


def _resample_indices(
    layout: Sequence[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> np.ndarray:
    """One cluster-bootstrap draw: independently resample each stratum's clusters.

    Cluster ``c`` of a stratum is drawn ``k`` times with replacement, and all of its
    rows are repeated ``k`` times; the result concatenates every sampled cluster.
    """
    parts: list[np.ndarray] = []
    for sizes, flattened in layout:
        cluster_count = len(sizes)
        chosen = rng.integers(0, cluster_count, size=cluster_count)
        draw_counts = np.bincount(chosen, minlength=cluster_count)
        repeats = np.repeat(draw_counts, sizes)
        parts.append(np.repeat(flattened, repeats))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _as_object_array(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    array = np.empty(len(rows), dtype=object)
    for index, row in enumerate(rows):
        array[index] = row
    return array


def _collect_rows(row_array: np.ndarray, indices: np.ndarray) -> list[Mapping[str, Any]]:
    return row_array[indices].tolist()


def _percentile_interval(values: Sequence[float], alpha: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    low, high = np.percentile(array, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    metric_fn: MetricFn,
    *,
    cluster_key: str = "cluster_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    stratify_key: str | None = "voice",
) -> dict[str, dict[str, float | None]]:
    """Cluster-bootstrap percentile confidence interval for every metric.

    ``metric_fn`` receives the resampled rows (a cluster drawn ``k`` times appears
    ``k`` times) and must return ``{metric_name: value}``.  The point estimate is
    ``metric_fn(rows)`` on the full input.  Resamples for which a metric is undefined
    (``None``) are skipped when computing the interval.

    Returns ``{metric_name: {"estimate", "lo", "hi"}}``.
    """
    row_list = list(rows)
    _validate_bootstrap_args(n_boot, alpha)

    estimate = _call_metric(metric_fn, row_list)
    metric_names = list(estimate)
    result: dict[str, dict[str, float | None]] = {
        name: {"estimate": _to_float(estimate[name]), "lo": None, "hi": None}
        for name in metric_names
    }
    layout = _build_strata(row_list, cluster_key, stratify_key)
    if not layout:
        return result

    row_array = _as_object_array(row_list)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_boot):
        sample = _collect_rows(row_array, _resample_indices(layout, rng))
        resampled = _call_metric(metric_fn, sample)
        for name in metric_names:
            value = _to_float(resampled.get(name))
            if value is not None:
                draws[name].append(value)

    for name in metric_names:
        low, high = _percentile_interval(draws[name], alpha)
        result[name]["lo"] = low
        result[name]["hi"] = high
    return result


def _check_paired_alignment(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    cluster_key: str,
) -> None:
    """Sanity-check that two row sequences are positionally aligned."""
    for position, (row_a, row_b) in enumerate(zip(rows_a, rows_b, strict=True)):
        item_a = row_a.get("item_id")
        item_b = row_b.get("item_id")
        if not _is_missing(item_a) and not _is_missing(item_b):
            if _to_py(item_a) != _to_py(item_b):
                raise ValueError(
                    "rows_a/rows_b are not aligned by item_id at position "
                    f"{position}: {item_a!r} != {item_b!r}"
                )
        cluster_a = row_a.get(cluster_key)
        cluster_b = row_b.get(cluster_key)
        if not _is_missing(cluster_a) and not _is_missing(cluster_b):
            if _to_py(cluster_a) != _to_py(cluster_b):
                raise ValueError(
                    "rows_a/rows_b carry different clusters at position "
                    f"{position}: {cluster_a!r} != {cluster_b!r}"
                )


def paired_bootstrap_difference(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    metric_fn: MetricFn,
    *,
    cluster_key: str = "cluster_id",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float | None]]:
    """Paired cluster-bootstrap CI for ``metric(rows_a) - metric(rows_b)``.

    The two inputs must be aligned by position on the same items (the caller sorts /
    filters both sides identically); this is validated when ``item_id`` or the cluster
    id is present.  Each bootstrap draw resamples clusters once and applies the same
    draw to both sides.  ``voice`` is used for stratification, matching
    :func:`bootstrap_ci`.

    Returns ``{metric_name: {"estimate", "lo", "hi"}}``.
    """
    rows_a_list = list(rows_a)
    rows_b_list = list(rows_b)
    _validate_bootstrap_args(n_boot, alpha)
    if len(rows_a_list) != len(rows_b_list):
        raise ValueError(
            "paired_bootstrap_difference: inputs must be aligned, got lengths "
            f"{len(rows_a_list)} and {len(rows_b_list)}"
        )
    _check_paired_alignment(rows_a_list, rows_b_list, cluster_key)

    estimate_a = _call_metric(metric_fn, rows_a_list)
    estimate_b = _call_metric(metric_fn, rows_b_list)
    metric_names = list(estimate_a) + [name for name in estimate_b if name not in estimate_a]
    result: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        value_a = _to_float(estimate_a.get(name))
        value_b = _to_float(estimate_b.get(name))
        difference = None if value_a is None or value_b is None else value_a - value_b
        result[name] = {"estimate": difference, "lo": None, "hi": None}

    layout = _build_strata(rows_a_list, cluster_key, "voice")
    if not layout:
        return result

    rows_a_array = _as_object_array(rows_a_list)
    rows_b_array = _as_object_array(rows_b_list)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_boot):
        indices = _resample_indices(layout, rng)
        resampled_a = _call_metric(metric_fn, _collect_rows(rows_a_array, indices))
        resampled_b = _call_metric(metric_fn, _collect_rows(rows_b_array, indices))
        for name in metric_names:
            value_a = _to_float(resampled_a.get(name))
            value_b = _to_float(resampled_b.get(name))
            if value_a is not None and value_b is not None:
                draws[name].append(value_a - value_b)

    for name in metric_names:
        low, high = _percentile_interval(draws[name], alpha)
        result[name]["lo"] = low
        result[name]["hi"] = high
    return result


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """McNemar test on two aligned sequences of correctness outcomes.

    ``b`` = number of pairs where A is correct and B wrong, ``c`` = the opposite.
    Pairs with a missing / non-boolean entry are dropped (parse failures), and
    ``n_pairs`` reports the number of valid pairs that were compared.

    * ``b + c <= 1000``: exact two-sided binomial test
      ``p = min(1, 2 * P(X <= min(b, c)))`` with ``X ~ Binomial(b + c, 0.5)``,
      ``statistic = min(b, c)``, ``method = "exact_binomial"``.
    * ``b + c > 1000``: chi-square approximation with continuity correction
      ``statistic = max(0, |b - c| - 1) ** 2 / (b + c)`` and
      ``p = erfc(sqrt(statistic / 2))``, ``method = "chi2_continuity_corrected"``.
    """
    correct_a_list = list(correct_a)
    correct_b_list = list(correct_b)
    if len(correct_a_list) != len(correct_b_list):
        raise ValueError(
            "mcnemar_test: inputs must be aligned, got lengths "
            f"{len(correct_a_list)} and {len(correct_b_list)}"
        )

    b = c = n_pairs = 0
    for value_a, value_b in zip(correct_a_list, correct_b_list, strict=True):
        outcome_a = _coerce_bool(value_a)
        outcome_b = _coerce_bool(value_b)
        if outcome_a is None or outcome_b is None:
            continue
        n_pairs += 1
        if outcome_a and not outcome_b:
            b += 1
        elif outcome_b and not outcome_a:
            c += 1

    discordant = b + c
    if discordant == 0:
        return {
            "n_pairs": int(n_pairs),
            "b": 0,
            "c": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "exact_binomial",
        }
    if discordant > 1000:
        statistic = max(0.0, abs(b - c) - 1.0) ** 2 / discordant
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        method = "chi2_continuity_corrected"
    else:
        minority = min(b, c)
        tail = sum(math.comb(discordant, k) for k in range(minority + 1))
        p_value = min(1.0, 2.0 * tail / float(2**discordant))
        statistic = float(minority)
        method = "exact_binomial"
    return {
        "n_pairs": int(n_pairs),
        "b": int(b),
        "c": int(c),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Multiple-comparison corrections
# ---------------------------------------------------------------------------


def holm_correct(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (same order as the input)."""
    values = [float(pvalue) for pvalue in pvalues]
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    previous = 0.0
    for rank, index in enumerate(order, start=1):
        candidate = min(1.0, (count - rank + 1) * values[index])
        previous = max(previous, candidate)
        adjusted[index] = previous
    return adjusted


def bh_correct(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjusted p-values (same order as the input)."""
    values = [float(pvalue) for pvalue in pvalues]
    count = len(values)
    if count == 0:
        return []
    order = sorted(range(count), key=lambda index: values[index])
    adjusted = [1.0] * count
    previous = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        candidate = min(1.0, count / rank * values[index])
        previous = min(previous, candidate)
        adjusted[index] = previous
    return adjusted


# ---------------------------------------------------------------------------
# Paired outcome extraction
# ---------------------------------------------------------------------------


def paired_outcomes(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    item_key: str = "item_id",
    correctness_key: str = "is_correct",
) -> tuple[list[bool], list[bool]]:
    """Align two processed result sets by ``item_key`` and extract correctness.

    Items missing on either side, rows failing to parse (``parse_status != "ok"``)
    and rows whose ``correctness_key`` is missing/``None`` are dropped from both
    sides.  Output order follows ``rows_a``; duplicate keys in ``rows_a`` are used
    once.  The returned sequences are suitable inputs for :func:`mcnemar_test`.
    """
    rows_a_list = list(rows_a)
    rows_b_list = list(rows_b)

    by_key_b: dict[Any, Mapping[str, Any]] = {}
    for row in rows_b_list:
        key = row.get(item_key)
        if _is_missing(key):
            continue
        by_key_b.setdefault(_to_py(key), row)

    outcomes_a: list[bool] = []
    outcomes_b: list[bool] = []
    seen: set[Any] = set()
    for row in rows_a_list:
        key = row.get(item_key)
        if _is_missing(key):
            continue
        key = _to_py(key)
        if key in seen:
            continue
        seen.add(key)
        other = by_key_b.get(key)
        if other is None:
            continue
        if not _parse_ok(row) or not _parse_ok(other):
            continue
        correct_a = _coerce_bool(row.get(correctness_key))
        correct_b = _coerce_bool(other.get(correctness_key))
        if correct_a is None or correct_b is None:
            continue
        outcomes_a.append(correct_a)
        outcomes_b.append(correct_b)
    return outcomes_a, outcomes_b
