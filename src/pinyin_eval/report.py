"""Report generation for a scored run.

Reads ``results/runs/<run_id>/items.parquet`` (+ ``detection_rows.parquet`` and
``run_manifest.json``) and writes a Chinese markdown report plus machine-readable
tables.  Contract: ``docs/contracts.md`` section 8; design:
``docs/experiment-design.md`` section 三.

Analysis protocol implemented here (mirrors the design document):

* main results use the run's *main generation condition* (manifest
  ``config.generation.main_condition``, default ``think1__clean__json``);
* accuracies / macro-F1 / joint accuracy come straight from
  :mod:`pinyin_eval.metrics` (never recomputed from scratch);
* 95% confidence intervals use :func:`pinyin_eval.stats.bootstrap_ci` with
  ``cluster_id`` (audio) as resampling unit, ``voice`` as stratum and
  ``stats.n_boot`` draws (reproducible via ``seed``);
* detection metrics pair exp1 negatives with exp2 positives at the same
  generation condition; typed subsets are reported separately, parse failures
  are reported but never enter an accuracy denominator;
* ablation conditions are compared with the main condition on the **same item
  subset** (the per-condition item intersection, optionally intersected with the
  ``ablation_*_items.txt`` subsets produced by ``build-tasks``);
* paired condition comparisons (thinking on/off, clean/noise, json/text) use
  McNemar tests plus paired cluster-bootstrap CIs, with Holm correction inside
  each metric family;
* confusion matrices are written for the main condition's free-recognition view
  (experiment 3) per dimension (and per voice when configured);
* token usage / CNY cost is summarised from ``generations.jsonl`` with the
  manifest ``pricing`` snapshot (input / cache-hit / output rates) and compared
  against the manifest ``budget_cny`` guard when present.

All numbers written to CSV/JSON are plain Python scalars (``NaN`` -> ``None``)
and the module does no network / model calls: it only reads scored files.

When parts of a run are missing (no detection rows, only one generation
condition, no ablation subset lists, ...) the corresponding section is skipped
and the reason is recorded in the report / ``summary.json`` instead of raising.
"""

from __future__ import annotations

import json
import logging
import math
import operator
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics, stats
from .config import Config
from .pricing import DEFAULT_PRICING, Pricing, budget_status, summarize_generations
from .records import EXPERIMENTS, PROMPT_CONDITION_TITLES
from .scoring import attach_correctness, generation_key

log = logging.getLogger(__name__)

__all__ = ["ReportError", "write_report"]

# ---------------------------------------------------------------------------
# File names / artifact layout
# ---------------------------------------------------------------------------

ITEMS_NAME = "items.parquet"
DETECTION_ROWS_NAME = "detection_rows.parquet"
GENERATIONS_NAME = "generations.jsonl"
MANIFEST_NAME = "run_manifest.json"
AUDIO_MANIFEST_NAME = "audio.parquet"
TASK_DIR = "tasks"
ABLATION_CORE_NAME = "ablation_core_items.txt"
ABLATION_PAIRS_NAME = "ablation_pairs_items.txt"
MINIMAL_PAIR_COUNTS_NAME = "minimal_pair_counts.json"

REPORT_NAME = "report.md"
METRICS_NAME = "metrics.csv"
DETECTION_NAME = "detection.csv"
ABLATION_NAME = "ablation.csv"
COMPARISONS_NAME = "comparisons.csv"
BASELINES_NAME = "baselines.csv"
USAGE_NAME = "usage.csv"
SUMMARY_NAME = "summary.json"
CONFUSION_DIR = "confusion"

#: long-format metric table written to ``metrics.csv`` (column order is frozen)
METRICS_COLUMNS: tuple[str, ...] = (
    "scope",
    "experiment",
    "prompt_condition",
    "condition_id",
    "generation_condition",
    "voice",
    "pair_type",
    "subset",
    "dimension",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "n",
    "n_failures",
)

#: long-format detection table written to ``detection.csv`` (column order frozen)
DETECTION_COLUMNS: tuple[str, ...] = (
    "scope",
    "generation_condition",
    "voice",
    "subset",
    "dimension",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "n",
    "n_failures",
    "tp",
    "fp",
    "tn",
    "fn",
    "n_neg",
    "n_pos",
)

#: ablation table written to ``ablation.csv`` (column order frozen)
ABLATION_COLUMNS: tuple[str, ...] = (
    "group",
    "compared_with",
    "is_main",
    "condition_id",
    "generation_condition",
    "experiment",
    "prompt_condition",
    "dimension",
    "metric",
    "value",
    "n",
    "n_failures",
)

#: paired condition comparisons written to ``comparisons.csv``
COMPARISON_COLUMNS: tuple[str, ...] = (
    "family",
    "metric",
    "dimension",
    "axis",
    "condition_a",
    "condition_b",
    "n_pairs",
    "acc_a",
    "acc_b",
    "diff",
    "diff_lo",
    "diff_hi",
    "mcnemar_b",
    "mcnemar_c",
    "p",
    "p_holm",
    "method",
)

#: per-run / per-condition token & cost table written to ``usage.csv``
USAGE_COLUMNS: tuple[str, ...] = (
    "scope",
    "condition_id",
    "generation_condition",
    "n_calls",
    "n_errors",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "input_cost_cny",
    "output_cost_cny",
    "total_cost_cny",
    "avg_latency_s",
)

#: baseline table written to ``baselines.csv``
BASELINE_COLUMNS: tuple[str, ...] = (
    "scope",
    "experiment",
    "subset",
    "dimension",
    "baseline",
    "value",
    "label",
    "n",
)

#: design-document minimal-pair counts (docs/experiment-design.md table)
DESIGN_PAIR_COUNTS: dict[str, dict[str, int]] = {
    "sh_zh": {"n_pairs": 36, "n_instances": 576},
    "aspiration": {"n_pairs": 67, "n_instances": 1072},
    "nasal_lateral": {"n_pairs": 21, "n_instances": 336},
    "front_back_nasal_ptr": {"n_pairs": 40, "n_instances": 640},
    "i_ü": {"n_pairs": 5, "n_instances": 80},
    "ian_üan": {"n_pairs": 3, "n_instances": 48},
    "tone": {"n_pairs": 2268, "n_instances": 9072},
}

PAIR_TYPE_TITLES: dict[str, str] = {
    "sh_zh": "平翘舌",
    "aspiration": "送气",
    "nasal_lateral": "鼻音/边音",
    "front_back_nasal_ptr": "前后鼻",
    "i_ü": "i/ü",
    "ian_üan": "ian/üan",
    "tone": "声调",
}

DIMENSION_TITLES: dict[str, str] = {
    "initial": "声母",
    "final": "韵母",
    "tone": "声调",
    "joint": "联合",
    "consistency": "一致性判定",
    "detection": "检测",
}

AXIS_TITLES: dict[str, str] = {
    "thinking": "thinking on/off",
    "audio_quality": "clean/noise",
    "output_form": "json/text",
}

RECOGNITION_DIMS: tuple[str, ...] = ("initial", "final", "tone")
EXPERIMENT_ORDER: tuple[str, ...] = EXPERIMENTS

#: detection subsets (design: main + per error type)
DEFAULT_DETECTION_SUBSETS: tuple[str, ...] = ("main", "initial", "final", "tone")

#: metrics reported for detection cells (all produced by metrics.detection_metrics)
DETECTION_METRICS: tuple[str, ...] = (
    "error_precision",
    "error_recall",
    "error_f1",
    "fpr",
    "fnr",
    "accuracy",
    "balanced_accuracy",
)

#: per-cell descriptor for the main-results section
#: (experiment, prompt_condition, mode, dimensions, include_joint)
#:
#: v3 design: experiments 1/2 are tone-only, experiment 3 is free recognition and
#: experiments 4/5/6 each have **three sub-experiments** (free / correct info /
#: wrong info) that only report their own dimension.  There is no A/B task.
MAIN_CELL_DEFS: tuple[tuple[str, str, str, tuple[str, ...], bool], ...] = (
    ("exp1", "correct_target", "recognition", ("tone",), False),
    ("exp2", "wrong_target", "recognition", ("tone",), False),
    ("exp3", "free", "recognition", ("initial", "final", "tone"), True),
    ("exp4", "free", "recognition", ("initial",), False),
    ("exp4", "correct_target", "recognition", ("initial",), False),
    ("exp4", "wrong_target", "recognition", ("initial",), False),
    ("exp5", "free", "recognition", ("final",), False),
    ("exp5", "correct_target", "recognition", ("final",), False),
    ("exp5", "wrong_target", "recognition", ("final",), False),
    ("exp6", "free", "recognition", ("tone",), False),
    ("exp6", "correct_target", "recognition", ("tone",), False),
    ("exp6", "wrong_target", "recognition", ("tone",), False),
)

#: comparison metric families: (family label, metric name, dimension, prompt c.)
#:
#: v3 (2026-10-28 20:37) design:
#: * experiments 1/2 only measure tone; the dedicated ``detect`` experiment
#:   carries the consistency metric;
#: * experiments 4/5/6 are three sub-experiments each (free / correct / wrong)
#:   and only report their own dimension -> three comparison families each;
#: * there is no A/B task and no exp6 ``wrong_target`` tone-only family.
COMPARISON_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("exp1", "tone_accuracy", "tone", "correct_target"),
    ("exp2", "tone_accuracy", "tone", "wrong_target"),
    ("exp3", "initial_accuracy", "initial", "free"),
    ("exp3", "final_accuracy", "final", "free"),
    ("exp3", "tone_accuracy", "tone", "free"),
    ("exp3", "joint_accuracy", "joint", "free"),
    ("detect", "consistency_accuracy", "consistency", "detect"),
    ("exp4.free", "initial_accuracy", "initial", "free"),
    ("exp4.correct_target", "initial_accuracy", "initial", "correct_target"),
    ("exp4.wrong_target", "initial_accuracy", "initial", "wrong_target"),
    ("exp5.free", "final_accuracy", "final", "free"),
    ("exp5.correct_target", "final_accuracy", "final", "correct_target"),
    ("exp5.wrong_target", "final_accuracy", "final", "wrong_target"),
    ("exp6.free", "tone_accuracy", "tone", "free"),
    ("exp6.correct_target", "tone_accuracy", "tone", "correct_target"),
    ("exp6.wrong_target", "tone_accuracy", "tone", "wrong_target"),
)

#: design "实验 × 指标矩阵" (v3); used by the markdown overview table.
#: Each row: (experiment, title, Initial, Final, Tone, Joint, EP/ER/F1, FPR/FNR).
EXPERIMENT_METRIC_MATRIX: tuple[tuple[str, ...], ...] = (
    ("exp1", "实验一（正确信息，只评声调）", "—", "—", "✓", "—", "—", "—"),
    ("exp2", "实验二（错误信息，只评声调）", "—", "—", "✓", "—", "—", "—"),
    ("detect", "检测实验（target + target_tone）", "—", "—", "—", "—", "✓", "✓"),
    ("exp3", "实验三（自由检测/识别）", "✓", "✓", "✓", "✓", "—", "—"),
    ("exp4", "实验四（声母最小对；free/正确/错误 3 子实验）", "✓", "—", "—", "—", "—", "—"),
    ("exp5", "实验五（韵母最小对；free/正确/错误 3 子实验）", "—", "✓", "—", "—", "—", "—"),
    ("exp6", "实验六（声调最小对；free/正确/错误 3 子实验）", "—", "—", "✓", "—", "—", "—"),
)

#: note recorded in the report header / appendix: data & task schema revision
DESIGN_REVISION_NOTE = (
    "数据/任务结构对应 2026-10-28 v3 设计更新：取消 A/B 辨别任务；提示条件统一为三水平"
    "（自由检测 / 正确信息 / 错误信息）；实验一/二只评声调；检测实验独立"
    "（target + target_tone → {\"consistent\"}，2 改声母 / 2 改韵母 / 1 改声调）且**不参与消融**；"
    "实验四/五/六各为三个子实验（自由检测 / 正确信息 / 错误信息），分别只输出并评估"
    " initial / final / tone。检测实验中**正类 = 不一致**，`pred_consistent=True` 表示模型"
    "判定“一致”；实验二为 5 个只改声母/韵母的错误无调目标。"
)

DEFAULT_MAIN_CONDITION: dict[str, Any] = {
    "thinking": True,
    "audio_quality": "clean",
    "output_form": "json",
}


class ReportError(RuntimeError):
    """Raised when a report cannot be produced at all (e.g. no scored items)."""


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas NA (empty strings are valid)."""
    if value is None:
        return True
    try:
        result = value != value
    except (TypeError, ValueError):
        return True
    if isinstance(result, bool):
        return result
    try:
        return bool(result)
    except (TypeError, ValueError):
        return True


def _num(value: Any) -> float | None:
    """Convert a metric value to JSON-friendly ``float`` (``None`` for NaN)."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_int_or_none(value: Any) -> int | None:
    """``int`` conversion that keeps missing values as ``None``."""
    if _is_missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    """``float`` conversion that keeps missing values as ``None``."""
    if _is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round4(value: Any) -> float | None:
    """Round a CNY amount / latency to 4 decimals (``None`` stays ``None``)."""
    number = _as_float_or_none(value)
    return None if number is None else round(number, 4)


def _fmt_count(value: Any) -> str:
    """Format a token/call counter with thousands separators."""
    number = _as_int_or_none(value)
    return "—" if number is None else f"{number:,}"


def _pricing_from_mapping(raw: Mapping[str, Any]) -> Pricing | None:
    """Build a :class:`~pinyin_eval.pricing.Pricing` from a manifest snapshot."""
    values = dict(DEFAULT_PRICING)
    for key in values:
        if key in raw and raw[key] is not None:
            values[key] = raw[key]
    try:
        return Pricing(**values)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas / NaN values to plain JSON types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _fmt(value: Any, digits: int = 3) -> str:
    """Human readable formatting for markdown tables."""
    if value is None or _is_missing(value):
        return "—"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_ci(lo: Any, hi: Any, digits: int = 3) -> str:
    if lo is None or _is_missing(lo) or hi is None or _is_missing(hi):
        return "—"
    return f"[{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _ci_text(
    uncertainty: Mapping[str, Mapping[str, Any]] | None, metric_name: str
) -> str:
    """Format a metric's bootstrap interval as ``[lo, hi]`` (``—`` if missing)."""
    entry = (uncertainty or {}).get(metric_name) or {}
    return _fmt_ci(entry.get("lo"), entry.get("hi"))


def _fmt_rate(value: Any, digits: int = 1) -> str:
    if value is None or _is_missing(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _sorted_strings(values: Any) -> list[str]:
    out = sorted({str(value) for value in values if not _is_missing(value)})
    return out


def _coerce_bool(value: Any) -> bool | None:
    """Same coercion rule as :mod:`pinyin_eval.metrics` for stored booleans."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1, 0.0, 1.0):
            return bool(value)
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    """Coerce a (possibly nullable / object) series to a strict boolean mask."""
    try:
        return series.astype("boolean").fillna(False).astype(bool)
    except (TypeError, ValueError):
        return series.map(lambda value: False if _is_missing(value) else bool(value)).astype(bool)


def _eq_mask(gold: pd.Series, pred: pd.Series) -> pd.Series:
    """Element-wise equality with ``False`` for any missing / incomparable value."""
    try:
        result = gold.eq(pred)
    except (TypeError, ValueError):
        return pd.Series(False, index=gold.index, dtype=bool)
    return _bool_series(result)


def _joint_mask(frame: pd.DataFrame) -> pd.Series:
    """``parse ok`` AND all three dimensions match (missing prediction => False)."""
    mask = _bool_series(frame["parse_status"].eq("ok"))
    for dim in RECOGNITION_DIMS:
        gold = frame[f"gold_{dim}"]
        pred = frame[f"pred_{dim}"]
        mask = mask & _eq_mask(gold, pred)
    return _bool_series(mask)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, str):
                cells.append(value.replace("|", "\\|"))
            else:
                cells.append("" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _generation_key_for(condition: Mapping[str, Any]) -> str:
    thinking = bool(condition.get("thinking", True))
    quality = str(condition.get("audio_quality", "clean"))
    form = str(condition.get("output_form", "json"))
    return f"think{1 if thinking else 0}__{quality}__{form}"


def _parse_generation_key(key: str) -> dict[str, Any] | None:
    parts = str(key).split("__")
    if len(parts) != 3:
        return None
    thinking_raw, quality, form = parts
    mapping = {"think1": True, "think0": False}
    if thinking_raw not in mapping:
        return None
    return {"thinking": mapping[thinking_raw], "audio_quality": quality, "output_form": form}


def _condition_axis(condition: Mapping[str, Any]) -> tuple[bool, str, str]:
    return (
        bool(condition.get("thinking", True)),
        str(condition.get("audio_quality", "clean")),
        str(condition.get("output_form", "json")),
    )


def _normalise_main_condition(raw: Any, cfg: Config) -> dict[str, Any]:
    candidate: Any = None
    if isinstance(raw, Mapping):
        nested = raw.get("generation")
        if isinstance(nested, Mapping):
            candidate = nested.get("main_condition")
        if candidate is None and "thinking" in raw:
            candidate = raw
    if not isinstance(candidate, Mapping):
        candidate = cfg.main_condition
    return {
        "thinking": bool(candidate.get("thinking", DEFAULT_MAIN_CONDITION["thinking"])),
        "audio_quality": str(
            candidate.get("audio_quality", DEFAULT_MAIN_CONDITION["audio_quality"])
        ),
        "output_form": str(candidate.get("output_form", DEFAULT_MAIN_CONDITION["output_form"])),
    }


# ---------------------------------------------------------------------------
# Fast (vectorised) metric functions used inside bootstrap resamples
# ---------------------------------------------------------------------------
#
# ``stats.bootstrap_ci`` calls the metric function ~n_boot times, so the plain
# ``metrics.recognition_metrics`` (a pure-python row loop) is too slow for large
# runs.  The closures below re-implement exactly the same definitions on top of
# integer label codes pre-encoded once per row:
#   * accuracy   == metrics.classification_metrics(...)["accuracy"]
#   * macro_f1   == sklearn-style macro F1 over the labels observed in the draw
#   * joint      == metrics.recognition_metrics(...)["joint_accuracy"]
# The point estimates written to the tables always come from the real
# ``metrics`` functions; the closures are only used for the CI draws.
_IG, _IP = "_ig", "_ip"
_FG, _FP = "_fg", "_fp"
_TG, _TP = "_tg", "_tp"
_JOINT, _OK = "_joint", "_ok"
_DET_CODE = "_dcode"


def _make_recognition_fn(
    dims: Sequence[str],
    label_sizes: Mapping[str, int],
    include_joint: bool,
) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, float | None]]:
    specs: list[tuple[str, str, str, int]] = []
    key_map = {"initial": (_IG, _IP), "final": (_FG, _FP), "tone": (_TG, _TP)}
    for dim in dims:
        gold_key, pred_key = key_map[dim]
        specs.append((dim, gold_key, pred_key, max(int(label_sizes.get(dim, 0)), 1)))
    do_joint = bool(include_joint)

    def metric_fn(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        n = len(rows)
        out: dict[str, float | None] = {}
        if n == 0:
            for dim in dims:
                out[f"{dim}_accuracy"] = None
                out[f"{dim}_macro_f1"] = None
            if do_joint:
                out["joint_accuracy"] = None
            return out
        for dim, gold_key, pred_key, n_labels in specs:
            gold = np.fromiter(map(operator.itemgetter(gold_key), rows), np.int64, n)
            pred = np.fromiter(map(operator.itemgetter(pred_key), rows), np.int64, n)
            valid = pred >= 0
            n_valid = int(valid.sum())
            if n_valid == 0:
                out[f"{dim}_accuracy"] = None
                out[f"{dim}_macro_f1"] = None
                continue
            gold_valid = gold[valid]
            pred_valid = pred[valid]
            pairs = gold_valid * n_labels + pred_valid
            table = np.bincount(pairs, minlength=n_labels * n_labels).reshape(
                n_labels, n_labels
            ).astype(np.float64)
            true_positive = np.diag(table)
            support = table.sum(axis=1)
            predicted = table.sum(axis=0)
            out[f"{dim}_accuracy"] = float(true_positive.sum() / n_valid)
            denominator = support + predicted
            f1 = np.zeros(n_labels, dtype=np.float64)
            nonzero = denominator > 0
            f1[nonzero] = 2.0 * true_positive[nonzero] / denominator[nonzero]
            observed = nonzero
            out[f"{dim}_macro_f1"] = float(f1[observed].mean()) if observed.any() else None
        if do_joint:
            joint = np.fromiter(map(operator.itemgetter(_JOINT), rows), np.int64, n)
            parsed = np.fromiter(map(operator.itemgetter(_OK), rows), np.int64, n)
            denominator = int(parsed.sum())
            out["joint_accuracy"] = float(joint.sum() / denominator) if denominator else None
        return out

    return metric_fn


def _detection_metric_fn(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    n = len(rows)
    if n == 0:
        return {metric: None for metric in DETECTION_METRICS}
    codes = np.fromiter(map(operator.itemgetter(_DET_CODE), rows), np.int64, n)
    counts = np.bincount(codes, minlength=5)
    true_negative, false_positive, true_positive, false_negative = (
        int(counts[0]),
        int(counts[1]),
        int(counts[2]),
        int(counts[3]),
    )
    error_precision = _ratio(true_positive, true_positive + false_positive)
    error_recall = _ratio(true_positive, true_positive + false_negative)
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
    specificity = _ratio(true_negative, true_negative + false_positive)
    balanced_accuracy = (
        None
        if error_recall is None or specificity is None
        else float((error_recall + specificity) / 2.0)
    )
    return {
        "error_precision": error_precision,
        "error_recall": error_recall,
        "error_f1": error_f1,
        "fpr": _ratio(false_positive, false_positive + true_negative),
        "fnr": _ratio(false_negative, false_negative + true_positive),
        "accuracy": _ratio(
            true_positive + true_negative,
            true_positive + false_positive + true_negative + false_negative,
        ),
        "balanced_accuracy": balanced_accuracy,
    }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class _Reporter:
    """Stateful report builder; instantiate via :func:`write_report`."""

    def __init__(
        self,
        cfg: Config,
        run_id_or_dir: str | Path,
        *,
        out_dir: str | Path | None = None,
        n_boot: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_dir = self._resolve_run_dir(run_id_or_dir)
        self.out_dir = (
            Path(out_dir).resolve()
            if out_dir is not None
            else self.run_dir
        )
        default_n_boot = cfg.get("stats.n_boot", 2000)
        self.n_boot = int(n_boot if n_boot is not None else default_n_boot or 2000)
        if self.n_boot < 1:
            raise ReportError(f"n_boot must be a positive integer, got {self.n_boot!r}")
        default_seed = cfg.get("project.seed", 0)
        self.seed = int(seed if seed is not None else (default_seed or 0))
        self.alpha = float(cfg.get("stats.alpha", 0.05) or 0.05)
        stratify = cfg.get("stats.stratify_key", "voice")
        self.stratify_key: str | None = "voice" if stratify is None else str(stratify)
        self.per_voice = bool(cfg.get("report.per_voice", True))
        self.confusion_enabled = bool(cfg.get("report.confusion_matrices", True))
        self.comparisons_enabled = bool(cfg.get("report.mcnemar", True))
        self.multiple_comparison = str(cfg.get("report.multiple_comparison", "holm") or "holm")
        detection_subsets = cfg.get("tasks.detection.subsets", DEFAULT_DETECTION_SUBSETS)
        self.detection_subsets: tuple[str, ...] = (
            tuple(str(item) for item in detection_subsets)
            if detection_subsets
            else DEFAULT_DETECTION_SUBSETS
        )

        # loaded inputs
        self.manifest: dict[str, Any] = {}
        self.items_full = pd.DataFrame()
        self.items = pd.DataFrame()  # prepared (+ correctness columns)
        self.detection = pd.DataFrame()
        self.main_condition: dict[str, Any] = {}
        self.main_key = ""
        self.condition_keys: list[str] = []
        self.mainrows: list[dict[str, Any]] = []
        self.label_sizes: dict[str, int] = {}
        self.detection_mainrows: list[dict[str, Any]] = []
        self.detection_keys: list[str] = []
        self.ablation_source = ""
        self.ablation_universe: set[str] | None = None
        self.item_subset_source: str | None = None

        # collected tables / report parts
        self.metrics_rows: list[dict[str, Any]] = []
        self.detection_rows: list[dict[str, Any]] = []
        self.ablation_rows: list[dict[str, Any]] = []
        self.comparison_rows: list[dict[str, Any]] = []
        self.usage_rows: list[dict[str, Any]] = []
        self.usage_summary: dict[str, Any] = {}
        self.cost_summary: dict[str, Any] = {}
        self._usage_notes: list[str] = []
        self.baseline_rows: list[dict[str, Any]] = []
        self.headline: dict[str, Any] = {}
        self.detection_headline: dict[str, Any] = {}
        self.detection_unbalanced_subsets: list[str] = []
        self.confusion_tables: dict[str, dict[str, Any]] = {}
        self.md: dict[str, list[str]] = {}
        self.warnings: list[str] = []
        self.artifacts: dict[str, str] = {}
        self.summary: dict[str, Any] = {}
        self.generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    # -- path resolution ---------------------------------------------------
    def _resolve_run_dir(self, run_id_or_dir: str | Path) -> Path:
        path = Path(run_id_or_dir)
        if path.is_absolute():
            return path
        if path.exists():
            return path.resolve()
        return (self.cfg.path("paths.results_dir") / "runs" / str(run_id_or_dir)).resolve()

    def _warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        run_dir = self.run_dir
        if not run_dir.exists():
            raise ReportError(f"run directory not found: {run_dir}")
        items_path = run_dir / ITEMS_NAME
        if not items_path.exists():
            raise ReportError(
                f"items.parquet not found in {run_dir}; run `pinyin-eval score` first"
            )
        self.manifest = self._read_manifest(run_dir)
        subset_source = self.manifest.get("item_subset_source") if self.manifest else None
        if subset_source:
            self.item_subset_source = str(subset_source)
            self._warn(
                "该 run 只覆盖消融/子集 items（item_subset_source="
                f"{self.item_subset_source}）：主结果与检测指标不是全量数据口径，"
                "全量口径请使用主实验 run。"
            )
        raw_items = pd.read_parquet(items_path)
        self.items_full = raw_items.copy()
        self.items = self._prepare_items(raw_items)
        detection_path = run_dir / DETECTION_ROWS_NAME
        if detection_path.exists() and detection_path.stat().st_size > 0:
            try:
                self.detection = self._prepare_detection(pd.read_parquet(detection_path))
            except Exception as exc:  # pragma: no cover - corrupt parquet edge case
                self._warn(f"无法读取 detection_rows.parquet（{exc}）；检测分析跳过")
                self.detection = pd.DataFrame()
        else:
            self._warn("run 目录中没有 detection_rows.parquet；检测分析跳过（分数可能不完整）")
        log.info(
            "report inputs: items=%d, detection=%d, manifest=%s",
            len(self.items),
            len(self.detection),
            bool(self.manifest),
        )

    def _read_manifest(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / MANIFEST_NAME
        if not path.exists():
            self._warn("run_manifest.json 缺失；主条件/元信息回退到 configs 默认值")
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            self._warn(f"run_manifest.json 读取失败（{exc}）；回退到 configs 默认值")
            return {}
        return data if isinstance(data, dict) else {}

    def _prepare_items(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        defaults: dict[str, Any] = {
            "item_id": "",
            "experiment": "",
            "prompt_condition": "",
            "condition_id": "",
            "audio_id": "",
            "cluster_id": "",
            "voice": "",
            "gold_initial": "",
            "gold_final": "",
            "gold_tone": np.nan,
            "pred_initial": None,
            "pred_final": None,
            "pred_tone": None,
            "parse_status": "parse_error",
            "consistent_gold": None,
            "pred_consistent": None,
            "correct_choice": None,
            "pred_choice": None,
            "pair_id": None,
            "pair_type": None,
            "pair_side": None,
            "variant_type": None,
            "task_type": "",
        }
        for column, default in defaults.items():
            if column not in frame.columns:
                frame[column] = default
        if frame.empty:
            frame["generation_condition"] = pd.Series(dtype="object")
            return frame
        # stable string ids / cluster fallback
        for column in ("item_id", "audio_id", "cluster_id", "voice", "experiment",
                       "prompt_condition", "condition_id", "parse_status"):
            frame[column] = frame[column].map(
                lambda value: "" if _is_missing(value) else str(value)
            )
        cluster = frame["cluster_id"]
        audio = frame["audio_id"]
        frame["cluster_id"] = [
            c if c else a for c, a in zip(cluster.tolist(), audio.tolist(), strict=True)
        ]
        condition = frame["condition_id"]
        frame["generation_condition"] = condition.map(
            lambda value: generation_key(str(value)) if value else ""
        )
        frame["task_type"] = self._resolve_task_type(frame)
        # attach the derived correctness columns when an older/partial run lacks them
        derived = (
            "initial_correct",
            "final_correct",
            "tone_correct",
            "joint_correct",
            "consistent_correct",
            "choice_correct",
            "consistency_tone_correct",
        )
        if any(column not in frame.columns for column in derived):
            try:
                frame = attach_correctness(frame)
            except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover
                self._warn(f"attach_correctness 失败（{exc}）；正确性派生列不可用")
        return frame

    @staticmethod
    def _resolve_task_type(frame: pd.DataFrame) -> pd.Series:
        """Return the output-schema column, deriving it for older run directories.

        ``task_type`` (``free`` / ``tone`` / ``detect`` / ``ab``) was added by the
        2026-10-28 design update; for runs scored before that, the prompt
        condition fully determines the output schema.
        """
        prompt_condition = frame["prompt_condition"].map(
            lambda value: "" if _is_missing(value) else str(value)
        )
        derived = prompt_condition.map(
            {
                "free": "free",
                "correct_target": "tone",
                "wrong_target": "tone",
                "detect": "detect",
            }
        ).fillna("")
        if "task_type" not in frame.columns:
            return derived
        existing = frame["task_type"].map(
            lambda value: "" if _is_missing(value) else str(value)
        )
        return existing.where(existing.ne(""), derived)

    def _prepare_detection(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        defaults: dict[str, Any] = {
            "item_id": "",
            "generation_condition": "",
            "condition_id": "",
            "detection_role": "",
            "detection_subset": "",
            "cluster_id": "",
            "audio_id": "",
            "voice": "",
            "parse_status": "ok",
            "pred_consistent": None,
        }
        for column, default in defaults.items():
            if column not in frame.columns:
                frame[column] = default
        if frame.empty:
            return frame
        for column in ("item_id", "cluster_id", "voice", "detection_role",
                       "detection_subset", "parse_status"):
            frame[column] = frame[column].map(
                lambda value: "" if _is_missing(value) else str(value)
            )
        if "generation_condition" in frame.columns:
            frame["generation_condition"] = frame["generation_condition"].map(
                lambda value: "" if _is_missing(value) else str(value)
            )
        else:
            frame["generation_condition"] = frame["condition_id"].map(
                lambda value: generation_key(str(value)) if value else ""
            )
        has_condition = frame["generation_condition"].ne("").any()
        if not has_condition:
            frame["generation_condition"] = frame["condition_id"].map(
                lambda value: generation_key(str(value)) if value else ""
            )
        cluster = frame["cluster_id"]
        audio = frame["audio_id"]
        frame["cluster_id"] = [
            c if c else a for c, a in zip(cluster.tolist(), audio.tolist(), strict=True)
        ]
        return frame

    # -- main condition ----------------------------------------------------
    def _select_main(self) -> None:
        manifest_config = self.manifest.get("config") if self.manifest else None
        self.main_condition = _normalise_main_condition(manifest_config, self.cfg)
        requested = _generation_key_for(self.main_condition)
        if self.items.empty or "generation_condition" not in self.items.columns:
            self.main_key = requested
            self.condition_keys = []
            self._warn("items.parquet 为空或缺少 condition_id；无法计算主结果")
            return
        available = _sorted_strings(self.items["generation_condition"].tolist())
        self.condition_keys = available
        if requested in available:
            self.main_key = requested
        elif available:
            counts = self.items["generation_condition"].value_counts()
            best = max(
                available,
                key=lambda key: (int(counts.get(key, 0)), key),
            )
            self.main_key = best
            self._warn(
                f"主条件 {requested} 不在 items.parquet 中；回退到覆盖行数最多的条件 {best}"
            )
        else:
            self.main_key = requested
            self._warn("items.parquet 中没有可识别的 generation condition；无法计算主结果")
        main_frame = self.items[self.items["generation_condition"] == self.main_key]
        # the dedicated detect experiment is analysed from detection_rows.parquet in
        # the detection section; the recognition analysis below uses the v3 three
        # prompt conditions (free / correct info / wrong info) of experiments 1-6.
        analysis_mask = (
            main_frame["experiment"].isin(["exp1", "exp2", "exp3"])
            & main_frame["prompt_condition"].isin(["free", "correct_target", "wrong_target"])
        ) | (
            main_frame["experiment"].isin(["exp4", "exp5", "exp6"])
            & main_frame["prompt_condition"].isin(["free", "correct_target", "wrong_target"])
        )
        main_frame = main_frame[analysis_mask]
        self.mainrows, self.label_sizes = self._encode_items(main_frame)

    # -- row encodings -----------------------------------------------------
    def _encode_items(self, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, int]]:
        rows = frame.to_dict("records")
        label_maps: dict[str, dict[Any, int]] = {dim: {} for dim in RECOGNITION_DIMS}

        def encode(dim: str, value: Any) -> int:
            if _is_missing(value):
                return -1
            mapping = label_maps[dim]
            code = mapping.get(value)
            if code is None:
                code = len(mapping)
                mapping[value] = code
            return code

        parsed = _bool_series(frame["parse_status"].eq("ok")).tolist() if len(frame) else []
        gold = {dim: frame[f"gold_{dim}"].tolist() for dim in RECOGNITION_DIMS}
        pred = {dim: frame[f"pred_{dim}"].tolist() for dim in RECOGNITION_DIMS}
        joint = _joint_mask(frame).tolist()
        for index, row in enumerate(rows):
            ok = bool(parsed[index])
            row[_OK] = 1 if ok else 0
            row[_JOINT] = 1 if joint[index] else 0
            for dim in RECOGNITION_DIMS:
                row[f"_{dim[0]}g"] = encode(dim, gold[dim][index])
                pred_value = pred[dim][index]
                if ok and not _is_missing(pred_value):
                    row[f"_{dim[0]}p"] = encode(dim, pred_value)
                else:
                    row[f"_{dim[0]}p"] = -1
        label_sizes = {dim: len(mapping) for dim, mapping in label_maps.items()}
        return rows, label_sizes

    def _encode_detection(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows = frame.to_dict("records")
        parsed = _bool_series(frame["parse_status"].eq("ok")).tolist() if len(frame) else []
        roles = frame["detection_role"].tolist()
        preds = frame["pred_consistent"].tolist()
        for index, row in enumerate(rows):
            role = str(roles[index])
            predicted = _coerce_bool(preds[index]) if parsed[index] else None
            if predicted is None:
                code = 4  # failure (not scored)
            elif role == "neg":
                code = 0 if predicted else 1  # TN / FP
            elif role == "pos":
                code = 2 if not predicted else 3  # TP / FN
            else:  # pragma: no cover - scoring always tags a valid role
                code = 4
            row[_DET_CODE] = code
        return rows

    # -- generic cell helpers ---------------------------------------------
    def _bootstrap(
        self,
        rows: Sequence[Mapping[str, Any]],
        metric_fn: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not rows:
            return {}
        try:
            return stats.bootstrap_ci(
                rows,
                metric_fn,
                cluster_key="cluster_id",
                n_boot=self.n_boot,
                alpha=self.alpha,
                seed=self.seed,
                stratify_key=self.stratify_key,
            )
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            self._warn(f"bootstrap 计算失败（{exc}）；该单元置信区间置空")
            return {}

    @staticmethod
    def _metric_row(
        *,
        scope: str,
        experiment: str,
        prompt_condition: str,
        condition_id: str,
        generation_condition: str,
        voice: str = "",
        pair_type: str = "",
        subset: str = "",
        dimension: str,
        metric: str,
        value: Any,
        ci: Mapping[str, Any] | None = None,
        n: int = 0,
        n_failures: int = 0,
    ) -> dict[str, Any]:
        ci = ci or {}
        return {
            "scope": scope,
            "experiment": experiment,
            "prompt_condition": prompt_condition,
            "condition_id": condition_id,
            "generation_condition": generation_condition,
            "voice": voice,
            "pair_type": pair_type,
            "subset": subset,
            "dimension": dimension,
            "metric": metric,
            "value": _num(value),
            "ci_lo": _num(ci.get("lo")),
            "ci_hi": _num(ci.get("hi")),
            "n": int(n),
            "n_failures": int(n_failures),
        }

    def _append_recognition_cell(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        scope: str,
        experiment: str,
        prompt_condition: str,
        condition_id: str,
        generation_condition: str,
        voice: str = "",
        pair_type: str = "",
        subset: str = "",
        dims: Sequence[str],
        include_joint: bool,
    ) -> None:
        values = metrics.recognition_metrics(rows)
        metric_fn = _make_recognition_fn(dims, self.label_sizes, include_joint)
        uncertainty = self._bootstrap(rows, metric_fn)
        n = _int(values.get("n"))
        n_failures = _int(values.get("n_failures"))
        for dim in dims:
            entry = values.get(dim) or {}
            for metric_name in ("accuracy", "macro_f1"):
                key = f"{dim}_{metric_name}"
                self.metrics_rows.append(
                    self._metric_row(
                        scope=scope,
                        experiment=experiment,
                        prompt_condition=prompt_condition,
                        condition_id=condition_id,
                        generation_condition=generation_condition,
                        voice=voice,
                        pair_type=pair_type,
                        subset=subset,
                        dimension=dim,
                        metric=metric_name,
                        value=entry.get(metric_name),
                        ci=uncertainty.get(key),
                        n=n,
                        n_failures=n_failures,
                    )
                )
        if include_joint:
            self.metrics_rows.append(
                self._metric_row(
                    scope=scope,
                    experiment=experiment,
                    prompt_condition=prompt_condition,
                    condition_id=condition_id,
                    generation_condition=generation_condition,
                    voice=voice,
                    pair_type=pair_type,
                    subset=subset,
                    dimension="joint",
                    metric="accuracy",
                    value=values.get("joint_accuracy"),
                    ci=uncertainty.get("joint_accuracy"),
                    n=n,
                    n_failures=n_failures,
                )
            )

    def _append_ablation_metric(
        self,
        *,
        group: str,
        compared_with: str,
        is_main: bool,
        condition_id: str,
        generation_condition: str,
        experiment: str,
        prompt_condition: str,
        dimension: str,
        metric: str,
        value: Any,
        n: int,
        n_failures: int,
    ) -> None:
        self.ablation_rows.append(
            {
                "group": group,
                "compared_with": compared_with,
                "is_main": bool(is_main),
                "condition_id": condition_id,
                "generation_condition": generation_condition,
                "experiment": experiment,
                "prompt_condition": prompt_condition,
                "dimension": dimension,
                "metric": metric,
                "value": _num(value),
                "n": int(n),
                "n_failures": int(n_failures),
            }
        )

    # ------------------------------------------------------------------
    # Main results (overall / per voice / per minimal-pair type)
    # ------------------------------------------------------------------
    def _main_results(self) -> None:
        lines: list[str] = []
        if self.items.empty:
            lines.append("_items.parquet 为空：没有可分析的结果行。_")
            self.md["main"] = lines
            return
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.mainrows:
            groups[
                (str(row.get("experiment", "")), str(row.get("prompt_condition", "")))
            ].append(row)
        any_cell = False
        for experiment, prompt_condition, _mode, dims, include_joint in MAIN_CELL_DEFS:
            cell_rows = groups.get((experiment, prompt_condition), [])
            if not cell_rows:
                continue
            any_cell = True
            condition_id = f"{prompt_condition}__{self.main_key}"
            self._append_recognition_cell(
                cell_rows,
                scope="overall",
                experiment=experiment,
                prompt_condition=prompt_condition,
                condition_id=condition_id,
                generation_condition=self.main_key,
                dims=dims,
                include_joint=include_joint,
            )
            self._record_headline(
                experiment, prompt_condition, cell_rows, dims, include_joint
            )
            if self.per_voice:
                for voice in _sorted_strings(row.get("voice") for row in cell_rows):
                    voice_rows = [
                        row for row in cell_rows if str(row.get("voice", "")) == voice
                    ]
                    if not voice_rows:
                        continue
                    self._append_recognition_cell(
                        voice_rows,
                        scope="voice",
                        experiment=experiment,
                        prompt_condition=prompt_condition,
                        condition_id=condition_id,
                        generation_condition=self.main_key,
                        voice=voice,
                        dims=dims,
                        include_joint=include_joint,
                    )
            if experiment in ("exp4", "exp5", "exp6"):
                for pair_type in _sorted_strings(
                    row.get("pair_type") for row in cell_rows
                ):
                    type_rows = [
                        row
                        for row in cell_rows
                        if str(row.get("pair_type", "")) == pair_type
                    ]
                    if not type_rows:
                        continue
                    self._append_recognition_cell(
                        type_rows,
                        scope="pair_type",
                        experiment=experiment,
                        prompt_condition=prompt_condition,
                        condition_id=condition_id,
                        generation_condition=self.main_key,
                        pair_type=pair_type,
                        dims=dims,
                        include_joint=include_joint,
                    )
        if not any_cell:
            lines.append(
                "_当前 run 中没有实验一~六的识别任务行（可能只跑了检测实验或消融子集）。_"
            )

        lines.extend(self._render_main_table(groups))
        self.md["main"] = lines

    def _record_headline(
        self,
        experiment: str,
        prompt_condition: str,
        cell_rows: Sequence[Mapping[str, Any]],
        dims: Sequence[str],
        include_joint: bool,
    ) -> None:
        values = metrics.recognition_metrics(cell_rows)
        entry: dict[str, Any] = {
            "n": _int(values.get("n")),
            "n_failures": _int(values.get("n_failures")),
        }
        for dim in dims:
            sub = values.get(dim) or {}
            entry[f"{dim}_accuracy"] = _num(sub.get("accuracy"))
            entry[f"{dim}_macro_f1"] = _num(sub.get("macro_f1"))
        if include_joint:
            entry["joint_accuracy"] = _num(values.get("joint_accuracy"))
        self.headline.setdefault(experiment, {})[prompt_condition] = entry

    def _render_main_table(
        self, groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]
    ) -> list[str]:
        lines: list[str] = ["### 2.1 实验 × 指标矩阵", ""]
        lines.append(
            "> 依据 2026-10-28 v3 设计：取消 A/B 任务；实验四/五/六各含三个子实验"
            "（自由检测 / 正确信息 / 错误信息），分别只评估 initial / final / tone；"
            "检测实验不参与消融。"
        )
        lines.append("")
        lines.append(
            "| 实验/任务 | Initial | Final | Tone | Joint | EP/ER/F1 | FPR/FNR |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in EXPERIMENT_METRIC_MATRIX:
            lines.append(
                f"| {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | "
                f"{row[6]} | {row[7]} |"
            )
        lines.append("")
        lines.extend(["### 2.2 总体", ""])
        overall = [row for row in self.metrics_rows if row["scope"] == "overall"]
        if overall:
            rows = [
                [
                    row["experiment"],
                    PROMPT_CONDITION_TITLES.get(row["prompt_condition"], row["prompt_condition"]),
                    DIMENSION_TITLES.get(row["dimension"], row["dimension"]),
                    row["metric"],
                    _fmt(row["value"]),
                    _fmt_ci(row["ci_lo"], row["ci_hi"]),
                    row["n"],
                    row["n_failures"],
                ]
                for row in overall
            ]
            lines.extend(
                _md_table(
                    ["实验", "提示条件", "维度", "指标", "值", "95% CI", "n", "解析失败"],
                    rows,
                )
            )
        else:
            lines.append("_没有可报告的主结果单元。_")
        lines.append("")
        lines.append(f"> 主条件：`{self.main_key}`（generation 维度）；置信区间为 audio 聚类、voice 分层的 "
                     f"percentile bootstrap（n_boot={self.n_boot}，seed={self.seed}）。")
        if self.per_voice:
            lines.extend(["", "### 2.3 逐 voice", ""])
            voice_rows = [
                row for row in self.metrics_rows if row["scope"] == "voice"
            ]
            if voice_rows:
                rows = [
                    [
                        row["experiment"],
                        row["voice"],
                        DIMENSION_TITLES.get(row["dimension"], row["dimension"]),
                        row["metric"],
                        _fmt(row["value"]),
                        _fmt_ci(row["ci_lo"], row["ci_hi"]),
                        row["n"],
                    ]
                    for row in voice_rows
                ]
                lines.extend(
                    _md_table(
                        ["实验", "voice", "维度", "指标", "值", "95% CI", "n"], rows
                    )
                )
            else:
                lines.append("_没有逐 voice 数据。_")
        pair_rows = [row for row in self.metrics_rows if row["scope"] == "pair_type"]
        if pair_rows:
            lines.extend(
                [
                    "",
                    "### 2.4 逐最小对类型（实验四～六，3 个提示条件）",
                    "",
                    "| 实验 | 最小对类型 | 维度 | 指标 | 值 | 95% CI | n |",
                    "| --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            for row in pair_rows:
                pair_type = row["pair_type"]
                title = PAIR_TYPE_TITLES.get(pair_type, pair_type)
                lines.append(
                    f"| {row['experiment']} | {pair_type} ({title}) | "
                    f"{DIMENSION_TITLES.get(row['dimension'], row['dimension'])} | "
                    f"{row['metric']} | {_fmt(row['value'])} | "
                    f"{_fmt_ci(row['ci_lo'], row['ci_hi'])} | {row['n']} |"
                )
        return lines

    # ------------------------------------------------------------------
    # Baselines (random / majority class)
    # ------------------------------------------------------------------
    def _baselines(self) -> None:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.mainrows:
            groups[
                (str(row.get("experiment", "")), str(row.get("prompt_condition", "")))
            ].append(row)

        def add(
            scope: str,
            experiment: str,
            subset: str,
            dimension: str,
            baseline: str,
            value: Any,
            label: Any = None,
            n: int = 0,
        ) -> None:
            self.baseline_rows.append(
                {
                    "scope": scope,
                    "experiment": experiment,
                    "subset": subset,
                    "dimension": dimension,
                    "baseline": baseline,
                    "value": _num(value),
                    "label": None if _is_missing(label) else str(label),
                    "n": int(n),
                }
            )

        def add_tone_baseline(experiment: str, rows: Sequence[Mapping[str, Any]]) -> None:
            gold = [
                row.get("gold_tone")
                for row in rows
                if not _is_missing(row.get("gold_tone"))
            ]
            if not gold:
                return
            add(
                "overall",
                experiment,
                "",
                "tone",
                "random",
                metrics.random_baseline(4),
                n=len(gold),
            )
            majority = metrics.majority_baseline(gold)
            add(
                "overall",
                experiment,
                "",
                "tone",
                "majority",
                majority.get("accuracy"),
                label=majority.get("label"),
                n=_int(majority.get("n")),
            )

        # tone-only experiments 1/2
        for experiment, prompt_condition in (
            ("exp1", "correct_target"),
            ("exp2", "wrong_target"),
        ):
            add_tone_baseline(experiment, groups.get((experiment, prompt_condition), []))

        # experiment 3: initial/final majority + tone random/majority
        exp3_rows = groups.get(("exp3", "free"), [])
        if exp3_rows:
            for dimension, column in (
                ("initial", "gold_initial"),
                ("final", "gold_final"),
                ("tone", "gold_tone"),
            ):
                gold = [
                    row.get(column)
                    for row in exp3_rows
                    if not _is_missing(row.get(column))
                ]
                if not gold:
                    continue
                if dimension == "tone":
                    add(
                        "overall",
                        "exp3",
                        "",
                        "tone",
                        "random",
                        metrics.random_baseline(4),
                        n=len(gold),
                    )
                majority = metrics.majority_baseline(gold)
                add(
                    "overall",
                    "exp3",
                    "",
                    dimension,
                    "majority",
                    majority.get("accuracy"),
                    label=majority.get("label"),
                    n=_int(majority.get("n")),
                )

        # experiments 4/5/6: one output dimension each (v3); the gold distribution is
        # the same for the free / correct-info / wrong-info sub-experiments, so one
        # baseline row per dimension is enough.
        for experiment, dimension in (
            ("exp4", "initial"),
            ("exp5", "final"),
            ("exp6", "tone"),
        ):
            rows = groups.get((experiment, "free"), [])
            if not rows:
                continue
            if dimension == "tone":
                add_tone_baseline(experiment, rows)
                continue
            gold = [
                row.get(f"gold_{dimension}")
                for row in rows
                if not _is_missing(row.get(f"gold_{dimension}"))
            ]
            if not gold:
                continue
            majority = metrics.majority_baseline(gold)
            add(
                "overall",
                experiment,
                "",
                dimension,
                "majority",
                majority.get("accuracy"),
                label=majority.get("label"),
                n=_int(majority.get("n")),
            )

    # ------------------------------------------------------------------
    # Detection analysis
    # ------------------------------------------------------------------
    def _detection_results(self) -> None:
        lines: list[str] = []
        if self.detection.empty:
            lines.append("_没有 detection_rows.parquet：跳过检测分析。_")
            self.md["detection"] = lines
            return
        detection = self.detection
        detection = detection[detection["detection_role"].isin(["neg", "pos"])]
        if detection.empty:
            lines.append("_detection_rows.parquet 中没有 neg/pos 行：跳过检测分析。_")
            self.md["detection"] = lines
            return
        self.detection_keys = _sorted_strings(detection["generation_condition"].tolist())
        used_key = self.main_key if self.main_key in self.detection_keys else None
        if used_key is None and self.detection_keys:
            counts = detection["generation_condition"].value_counts()
            used_key = max(self.detection_keys, key=lambda key: (int(counts.get(key, 0)), key))
            self._warn(f"检测分析没有主条件 {self.main_key} 的行；使用 {used_key}")
        if used_key is None:
            lines.append("_检测行缺少 generation_condition：跳过检测分析。_")
            self.md["detection"] = lines
            return
        self.detection_key = used_key
        main_detection = detection[detection["generation_condition"] == used_key]
        self.detection_mainrows = self._encode_detection(main_detection)
        subsets_in_data = _sorted_strings(main_detection["detection_subset"].tolist())
        ordered_subsets = [
            subset for subset in self.detection_subsets if subset in subsets_in_data
        ] + [subset for subset in subsets_in_data if subset not in self.detection_subsets]
        imbalanced_subsets: list[str] = []
        overall_lines = [
            "| 子集 | n(neg/pos) | 失败(neg/pos) | TP | FP | TN | FN | EP (95% CI) | ER (95% CI) "
            "| Error-F1 (95% CI) | FPR (95% CI) | FNR (95% CI) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for subset in ordered_subsets:
            subset_rows = [
                row for row in self.detection_mainrows
                if str(row.get("detection_subset", "")) == subset
            ]
            if not subset_rows:
                continue
            values = metrics.detection_metrics(subset_rows)
            if (
                subset != "main"
                and _int(values.get("n_neg")) != _int(values.get("n_pos"))
            ):
                imbalanced_subsets.append(subset)
            metric_fn = _detection_metric_fn
            uncertainty = self._bootstrap(subset_rows, metric_fn)
            n = (
                _int(values.get("tp"))
                + _int(values.get("fp"))
                + _int(values.get("tn"))
                + _int(values.get("fn"))
            )
            n_failures = _int(values.get("n_failures_neg")) + _int(
                values.get("n_failures_pos")
            )
            for metric_name in DETECTION_METRICS:
                self.detection_rows.append(
                    {
                        "scope": "overall",
                        "generation_condition": used_key,
                        "voice": "",
                        "subset": subset,
                        "dimension": "detection",
                        "metric": metric_name,
                        "value": _num(values.get(metric_name)),
                        "ci_lo": _num((uncertainty.get(metric_name) or {}).get("lo")),
                        "ci_hi": _num((uncertainty.get(metric_name) or {}).get("hi")),
                        "n": n,
                        "n_failures": n_failures,
                        "tp": _int(values.get("tp")),
                        "fp": _int(values.get("fp")),
                        "tn": _int(values.get("tn")),
                        "fn": _int(values.get("fn")),
                        "n_neg": _int(values.get("n_neg")),
                        "n_pos": _int(values.get("n_pos")),
                    }
                )
            self.detection_headline[subset] = {
                "n": n,
                "n_failures": n_failures,
                "n_neg": _int(values.get("n_neg")),
                "n_pos": _int(values.get("n_pos")),
                "error_precision": _num(values.get("error_precision")),
                "error_recall": _num(values.get("error_recall")),
                "error_f1": _num(values.get("error_f1")),
                "fpr": _num(values.get("fpr")),
                "fnr": _num(values.get("fnr")),
                "accuracy": _num(values.get("accuracy")),
                "balanced_accuracy": _num(values.get("balanced_accuracy")),
            }
            overall_lines.append(
                "| {} | {} / {} | {} / {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    subset,
                    _int(values.get("n_neg")),
                    _int(values.get("n_pos")),
                    _int(values.get("n_failures_neg")),
                    _int(values.get("n_failures_pos")),
                    _int(values.get("tp")),
                    _int(values.get("fp")),
                    _int(values.get("tn")),
                    _int(values.get("fn")),
                    f"{_fmt(values.get('error_precision'))} {_ci_text(uncertainty, 'error_precision')}",
                    f"{_fmt(values.get('error_recall'))} {_ci_text(uncertainty, 'error_recall')}",
                    f"{_fmt(values.get('error_f1'))} {_ci_text(uncertainty, 'error_f1')}",
                    f"{_fmt(values.get('fpr'))} {_ci_text(uncertainty, 'fpr')}",
                    f"{_fmt(values.get('fnr'))} {_ci_text(uncertainty, 'fnr')}",
                )
            )
            # baselines: balanced detection chance + majority class
            parsed_subset = metrics.detection_metrics(subset_rows)
            n_neg = _int(parsed_subset.get("n_neg"))
            n_pos = _int(parsed_subset.get("n_pos"))
            total = n_neg + n_pos
            self.baseline_rows.append(
                {
                    "scope": "detection",
                    "experiment": "",
                    "subset": subset,
                    "dimension": "detection",
                    "baseline": "random",
                    "value": _num(metrics.detection_chance_baseline()),
                    "label": None,
                    "n": total,
                }
            )
            if total:
                majority = max(n_neg, n_pos) / total
                self.baseline_rows.append(
                    {
                        "scope": "detection",
                        "experiment": "",
                        "subset": subset,
                        "dimension": "detection",
                        "baseline": "majority",
                        "value": _num(majority),
                        "label": "neg" if n_neg >= n_pos else "pos",
                        "n": total,
                    }
                )
            if self.per_voice:
                for voice in _sorted_strings(row.get("voice") for row in subset_rows):
                    voice_rows = [
                        row for row in subset_rows if str(row.get("voice", "")) == voice
                    ]
                    if not voice_rows:
                        continue
                    voice_values = metrics.detection_metrics(voice_rows)
                    voice_uncertainty = self._bootstrap(voice_rows, _detection_metric_fn)
                    voice_n = (
                        _int(voice_values.get("tp"))
                        + _int(voice_values.get("fp"))
                        + _int(voice_values.get("tn"))
                        + _int(voice_values.get("fn"))
                    )
                    voice_failures = _int(voice_values.get("n_failures_neg")) + _int(
                        voice_values.get("n_failures_pos")
                    )
                    for metric_name in DETECTION_METRICS:
                        self.detection_rows.append(
                            {
                                "scope": "voice",
                                "generation_condition": used_key,
                                "voice": voice,
                                "subset": subset,
                                "dimension": "detection",
                                "metric": metric_name,
                                "value": _num(voice_values.get(metric_name)),
                                "ci_lo": _num(
                                    (voice_uncertainty.get(metric_name) or {}).get("lo")
                                ),
                                "ci_hi": _num(
                                    (voice_uncertainty.get(metric_name) or {}).get("hi")
                                ),
                                "n": voice_n,
                                "n_failures": voice_failures,
                                "tp": _int(voice_values.get("tp")),
                                "fp": _int(voice_values.get("fp")),
                                "tn": _int(voice_values.get("tn")),
                                "fn": _int(voice_values.get("fn")),
                                "n_neg": _int(voice_values.get("n_neg")),
                                "n_pos": _int(voice_values.get("n_pos")),
                            }
                        )
        lines.append("### 3.1 检测实验总体（main / initial / final / tone）")
        lines.append("")
        lines.append(
            "> 检测实验输出 `{\"consistent\": true|false}`；**正类 = “不一致”**，"
            "neg 来自一致样本（原始 target + target_tone），pos 来自不一致变体"
            "（2 改声母 / 2 改韵母 / 1 改声调）。"
            f"生成维度 `{used_key}`；失败行只计失败率，不进入指标分母。"
        )
        lines.append("")
        lines.extend(overall_lines)
        lines.append("")
        lines.append(
            "> 基线：平衡检测集随机基线 50%（见 `baselines.csv`）。逐 voice 的检测指标与置信区间"
            "见 `detection.csv`。"
        )
        if imbalanced_subsets:
            lines.append("")
            lines.append(
                "> **注意**：本 run 的检测 typed 子集（{}）neg/pos 不配平（见 "
                "`detection.csv` 的 `n_neg`/`n_pos`），多半来自消融子集 run；这些子集的 "
                "EP/ER/F1 不应按“平衡检测集”解释。平衡检测集应看全量主实验 run"
                "（initial/final/tone 各 6160:6160）或消融节中配平的 main 子集（256:256）。"
                .format(" / ".join(imbalanced_subsets))
            )
        self.detection_unbalanced_subsets = list(imbalanced_subsets)
        self.md["detection"] = lines

    # ------------------------------------------------------------------
    # Ablation analysis
    # ------------------------------------------------------------------
    def _ablation_item_ids(self) -> set[str] | None:
        candidates: list[Path] = []
        task_dir = self.cfg.get("paths.task_dir")
        if task_dir:
            base = self.cfg.path("paths.task_dir")
            candidates.extend(
                [
                    base / ABLATION_CORE_NAME,
                    base / ABLATION_PAIRS_NAME,
                ]
            )
        subset_source = self.manifest.get("item_subset_source") if self.manifest else None
        if subset_source:
            for part in str(subset_source).split(","):
                part = part.strip()
                if not part:
                    continue
                path = Path(part)
                candidates.append(path if path.is_absolute() else self.cfg.repo_root / path)
        ids: set[str] = set()
        found = False
        for path in candidates:
            try:
                if path.exists():
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if line:
                                ids.add(line)
                    found = True
            except OSError as exc:  # pragma: no cover - unreadable file
                self._warn(f"读取消融子集 {path} 失败（{exc}）")
        if not found:
            return None
        return ids

    def _ablation_results(self) -> None:
        lines: list[str] = []
        condition_keys = [key for key in self.condition_keys if key != self.main_key]
        if not condition_keys:
            lines.append("_只有主条件，没有消融条件；跳过消融分析。_")
            self.md["ablation"] = lines
            return
        if self.items.empty:
            lines.append("_没有可用的 items 行；跳过消融分析。_")
            self.md["ablation"] = lines
            return
        self.ablation_universe = self._ablation_item_ids()
        if self.ablation_universe is None:
            self.ablation_source = "条件公共 items（未找到 ablation_*_items.txt，按交集推断）"
            self._warn("未找到消融子集清单；消融分析按主条件与各条件的公共 items 推断")
        else:
            self.ablation_source = (
                "ablation_core_items.txt / ablation_pairs_items.txt（build-tasks 产物）"
            )
        main_frame = self.items[self.items["generation_condition"] == self.main_key]
        main_ids = set(main_frame["item_id"].astype(str))
        lines.append("### 4.1 消融条件 × 子集")
        lines.append("")
        lines.append(
            f"> 消融子集来源：{self.ablation_source}。每个消融条件都与主条件 "
            f"`{self.main_key}` 在与该条件的 **相同 items** 上比较"
            "（下表同时给出主条件在同一子集上的数值）。"
        )
        lines.append(
            "> v3 消融只有两组：**core**（实验一/二/三：正确信息 tone、错误信息 tone、"
            "自由检测识别指标）与 **pairs**（实验四/五/六的三个子实验：自由检测 / 正确信息 / "
            "错误信息，分别评估 initial / final / tone）。**检测实验不参与消融**，"
            "检测指标只在主实验的第 3 节报告。"
        )
        lines.append("")
        for key in condition_keys:
            condition_frame = self.items[self.items["generation_condition"] == key]
            common = set(condition_frame["item_id"].astype(str)) & main_ids
            if self.ablation_universe is not None:
                common &= self.ablation_universe
            if not common:
                self._warn(f"消融条件 {key} 与主条件没有公共 items；跳过")
                continue
            for frame, is_main in (
                (main_frame[main_frame["item_id"].isin(common)], True),
                (condition_frame[condition_frame["item_id"].isin(common)], False),
            ):
                if frame.empty:
                    continue
                self._collect_ablation_cells(
                    frame,
                    compared_with=key,
                    is_main=is_main,
                )
        # markdown table
        ablation_rows = list(self.ablation_rows)
        if not ablation_rows:
            lines.append("_没有可报告的消融单元（公共 items 为空）。_")
        else:
            by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in ablation_rows:
                by_group[(row["compared_with"], row["group"])].append(row)
            for (compared_with, group), rows in by_group.items():
                lines.extend(["", f"#### 消融条件 `{compared_with}` — {group} 组", ""])
                lines.append("| 来源 | 实验 | 提示条件 | 维度 | 指标 | 值 | n | 失败 |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for row in rows:
                    source = "主条件" if row["is_main"] else "消融条件"
                    prompt_title = PROMPT_CONDITION_TITLES.get(
                        row["prompt_condition"], row["prompt_condition"]
                    )
                    dimension_title = DIMENSION_TITLES.get(
                        row["dimension"], row["dimension"]
                    )
                    lines.append(
                        f"| {source} | {row['experiment'] or '—'} | "
                        f"{prompt_title or '—'} | {dimension_title} | {row['metric']} | "
                        f"{_fmt(row['value'])} | {row['n']} | {row['n_failures']} |"
                    )
        self.md["ablation"] = lines

    def _collect_ablation_cells(
        self,
        frame: pd.DataFrame,
        *,
        compared_with: str,
        is_main: bool,
    ) -> None:
        if frame.empty:
            return
        generation = self.main_key if is_main else compared_with

        def subset_rows(experiment: str, prompt_condition: str) -> list[dict[str, Any]]:
            mask = frame["experiment"].eq(experiment)
            if prompt_condition:
                mask = mask & frame["prompt_condition"].eq(prompt_condition)
            return frame[mask].to_dict("records")

        def append_recognition(
            experiment: str,
            prompt_condition: str,
            dims: Sequence[str],
            include_joint: bool,
            group: str,
        ) -> None:
            rows = subset_rows(experiment, prompt_condition)
            if not rows:
                return
            values = metrics.recognition_metrics(rows)
            condition_id = f"{prompt_condition}__{generation}"
            for dim in dims:
                entry = values.get(dim) or {}
                for metric_name in ("accuracy", "macro_f1"):
                    self._append_ablation_metric(
                        group=group,
                        compared_with=compared_with,
                        is_main=is_main,
                        condition_id=condition_id,
                        generation_condition=generation,
                        experiment=experiment,
                        prompt_condition=prompt_condition,
                        dimension=dim,
                        metric=metric_name,
                        value=entry.get(metric_name),
                        n=_int(values.get("n")),
                        n_failures=_int(values.get("n_failures")),
                    )
            if include_joint:
                self._append_ablation_metric(
                    group=group,
                    compared_with=compared_with,
                    is_main=is_main,
                    condition_id=condition_id,
                    generation_condition=generation,
                    experiment=experiment,
                    prompt_condition=prompt_condition,
                    dimension="joint",
                    metric="accuracy",
                    value=values.get("joint_accuracy"),
                    n=_int(values.get("n")),
                    n_failures=_int(values.get("n_failures")),
                )

        # core group: experiments 1-3
        append_recognition("exp1", "correct_target", ("tone",), False, "core")
        append_recognition("exp2", "wrong_target", ("tone",), False, "core")
        append_recognition("exp3", "free", ("initial", "final", "tone"), True, "core")
        # pairs group: experiments 4/5/6 × three sub-experiments
        for experiment, dimension in (
            ("exp4", "initial"),
            ("exp5", "final"),
            ("exp6", "tone"),
        ):
            for prompt_condition in ("free", "correct_target", "wrong_target"):
                append_recognition(
                    experiment, prompt_condition, (dimension,), False, "pairs"
                )
        # parse failure rates for the two groups
        experiment_values = frame["experiment"].astype(str)
        for group_name, mask, experiment_label in (
            ("parse_core", experiment_values.isin(["exp1", "exp2", "exp3"]), "exp1-3"),
            ("parse_pairs", experiment_values.isin(["exp4", "exp5", "exp6"]), "exp4-6"),
        ):
            group_frame = frame[mask]
            total = len(group_frame)
            if total == 0:
                continue
            failures = int((~group_frame["parse_status"].eq("ok")).sum())
            self._append_ablation_metric(
                group=group_name,
                compared_with=compared_with,
                is_main=is_main,
                condition_id=f"*__{generation}",
                generation_condition=generation,
                experiment=experiment_label,
                prompt_condition="",
                dimension="all",
                metric="parse_failure_rate",
                value=failures / total,
                n=total,
                n_failures=failures,
            )

    # ------------------------------------------------------------------
    # Paired condition comparisons (McNemar + paired bootstrap)
    # ------------------------------------------------------------------
    def _family_correctness(self, frame: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Correctness maps for the comparison families (v3 ``COMPARISON_SPECS``)."""
        result: dict[str, dict[str, int]] = {}
        if frame.empty:
            return result
        ok = _bool_series(frame["parse_status"].eq("ok"))
        experiment = frame["experiment"].astype(str)
        prompt_condition = frame["prompt_condition"].astype(str)
        initial_ok = ok & frame["pred_initial"].notna()
        final_ok = ok & frame["pred_final"].notna()
        tone_ok = ok & frame["pred_tone"].notna()
        consistency_ok = ok & frame["pred_consistent"].notna()
        initial_correct = _eq_mask(frame["gold_initial"], frame["pred_initial"])
        final_correct = _eq_mask(frame["gold_final"], frame["pred_final"])
        tone_correct = _eq_mask(frame["gold_tone"], frame["pred_tone"])
        joint_correct = initial_correct & final_correct & tone_correct
        # consistency correctness is pure equality: the detect experiment carries both
        # consistent (gold=True) and inconsistent (gold=False) claims, so each side must
        # compare equal to its matching prediction (missing predictions are handled by
        # the per-row validity mask below).
        consistency_correct = _eq_mask(
            frame["consistent_gold"].astype("boolean"),
            frame["pred_consistent"].astype("boolean"),
        )
        dimensions: dict[str, tuple[pd.Series, pd.Series]] = {
            "initial": (initial_ok, initial_correct),
            "final": (final_ok, final_correct),
            "tone": (tone_ok, tone_correct),
            "joint": (ok, joint_correct),
            "consistency": (consistency_ok, consistency_correct),
        }

        def add(name: str, scope: pd.Series, valid: pd.Series, correct: pd.Series) -> None:
            mask = _bool_series(scope) & _bool_series(valid)
            if not mask.any():
                return
            item_ids = frame.loc[mask, "item_id"].astype(str).tolist()
            values = _bool_series(correct).loc[mask].astype(int).tolist()
            result.setdefault(name, {}).update(dict(zip(item_ids, values, strict=True)))

        for family, _metric_name, dimension, prompt_condition_value in COMPARISON_SPECS:
            experiment_name = family.split(".", 1)[0]
            scope = experiment.eq(experiment_name) & prompt_condition.eq(
                prompt_condition_value
            )
            valid, correct = dimensions[dimension]
            add(f"{family}|{dimension}", scope, valid, correct)
        return result

    @staticmethod
    def _meta_map(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
        if frame.empty:
            return {}
        item_ids = frame["item_id"].astype(str).tolist()
        clusters = frame["cluster_id"].astype(str).tolist()
        voices = frame["voice"].astype(str).tolist()
        return {
            item_id: (cluster, voice)
            for item_id, cluster, voice in zip(item_ids, clusters, voices, strict=True)
        }

    def _paired_compact_rows(
        self,
        families_a: Mapping[str, Mapping[str, int]],
        families_b: Mapping[str, Mapping[str, int]],
        meta: Mapping[str, tuple[str, str]],
        family_order: Sequence[str],
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[str],
        dict[str, tuple[list[int], list[int]]],
    ]:
        rows_a: list[dict[str, Any]] = []
        rows_b: list[dict[str, Any]] = []
        names: list[str] = []
        outcomes: dict[str, tuple[list[int], list[int]]] = {}
        for name in family_order:
            map_a = families_a.get(name) or {}
            map_b = families_b.get(name) or {}
            if not map_a or not map_b:
                continue
            common = sorted(set(map_a) & set(map_b))
            if not common:
                continue
            code = len(names)
            names.append(name)
            values_a: list[int] = []
            values_b: list[int] = []
            for item_id in common:
                cluster, voice = meta.get(item_id, ("", ""))
                value_a = int(map_a[item_id])
                value_b = int(map_b[item_id])
                values_a.append(value_a)
                values_b.append(value_b)
                rows_a.append(
                    {"item_id": item_id, "cluster_id": cluster, "voice": voice,
                     "_f": code, "_c": value_a}
                )
                rows_b.append(
                    {"item_id": item_id, "cluster_id": cluster, "voice": voice,
                     "_f": code, "_c": value_b}
                )
            outcomes[name] = (values_a, values_b)
        return rows_a, rows_b, names, outcomes

    def _comparison_metric_fn(
        self, names: Sequence[str]
    ) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, float | None]]:
        def metric_fn(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
            counts = [0] * len(names)
            sums = [0] * len(names)
            for row in rows:
                code = row["_f"]
                counts[code] += 1
                sums[code] += row["_c"]
            return {
                name: (sums[code] / counts[code]) if counts[code] else None
                for code, name in enumerate(names)
            }

        return metric_fn

    def _comparison_pairs(self) -> list[tuple[str, str, str]]:
        parsed: dict[str, dict[str, Any]] = {}
        for key in self.condition_keys:
            condition = _parse_generation_key(key)
            if condition is not None:
                parsed[key] = condition
        keys = sorted(parsed)
        pairs: list[tuple[str, str, str]] = []
        for index, key_a in enumerate(keys):
            for key_b in keys[index + 1:]:
                condition_a = parsed[key_a]
                condition_b = parsed[key_b]
                differing = [
                    field
                    for field in ("thinking", "audio_quality", "output_form")
                    if condition_a[field] != condition_b[field]
                ]
                if len(differing) == 1:
                    pairs.append((key_b, key_a, differing[0]))
        return pairs

    def _comparison_results(self) -> None:
        lines: list[str] = []
        if not self.comparisons_enabled:
            lines.append("_配置 report.mcnemar=false：跳过条件比较。_")
            self.md["comparisons"] = lines
            return
        pairs = self._comparison_pairs()
        if not pairs:
            lines.append("_没有只差一个生成维度的条件对（需要至少两个条件）；跳过条件比较。_")
            self.md["comparisons"] = lines
            return
        items = self.items
        if items.empty:
            lines.append("_items.parquet 为空：跳过条件比较。_")
            self.md["comparisons"] = lines
            return
        items_by_condition = {
            key: items[items["generation_condition"] == key] for key in self.condition_keys
        }
        family_order = [f"{exp}|{dim}" for exp, _metric, dim, _pc in COMPARISON_SPECS]
        for key_a, key_b, axis in pairs:
            frame_a = items_by_condition.get(key_a)
            frame_b = items_by_condition.get(key_b)
            if frame_a is None or frame_b is None or frame_a.empty or frame_b.empty:
                continue
            families_a = self._family_correctness(frame_a)
            families_b = self._family_correctness(frame_b)
            meta = self._meta_map(frame_a)
            rows_a, rows_b, names, outcomes = self._paired_compact_rows(
                families_a, families_b, meta, family_order
            )
            if not names:
                continue
            uncertainty = self._paired_bootstrap(rows_a, rows_b, names)
            for name in names:
                values_a, values_b = outcomes[name]
                n_pairs = len(values_a)
                if n_pairs == 0:
                    continue
                accuracy_a = sum(values_a) / n_pairs
                accuracy_b = sum(values_b) / n_pairs
                mcnemar = stats.mcnemar_test(values_a, values_b)
                experiment, metric_name, dimension, _prompt = self._comparison_spec(name)
                diff_info = uncertainty.get(name) or {}
                diff = diff_info.get("estimate")
                if diff is None:
                    diff = accuracy_a - accuracy_b
                self.comparison_rows.append(
                    {
                        "family": experiment,
                        "metric": metric_name,
                        "dimension": dimension,
                        "axis": axis,
                        "condition_a": key_a,
                        "condition_b": key_b,
                        "n_pairs": n_pairs,
                        "acc_a": _num(accuracy_a),
                        "acc_b": _num(accuracy_b),
                        "diff": _num(diff),
                        "diff_lo": _num(diff_info.get("lo")),
                        "diff_hi": _num(diff_info.get("hi")),
                        "mcnemar_b": _int(mcnemar.get("b")),
                        "mcnemar_c": _int(mcnemar.get("c")),
                        "p": _num(mcnemar.get("p_value")),
                        "p_holm": None,
                        "method": str(mcnemar.get("method", "")),
                    }
                )
        self._apply_multiple_comparison()
        self._sort_comparison_rows()
        if not self.comparison_rows:
            lines.append("_条件对存在，但相同 items 上没有可比较的正确性数据；跳过条件比较。_")
            self.md["comparisons"] = lines
            return
        lines.append("### 5.1 条件对比较（McNemar + 配对 bootstrap）")
        lines.append("")
        lines.append(
            "> 条件对 = 只差一个生成维度（thinking / audio_quality / output_form）的两个条件；"
            "只使用两侧都有解析成功结果的相同 items（按 item_id 对齐），配对 bootstrap 以 audio "
            f"为聚类单位、voice 分层，n_boot={self.n_boot}，seed={self.seed}；差异 = A − B。`p_holm` 为同一 "
            "family/指标内 Hol。"
        )
        lines.append("")
        lines.append(
            "| family | 指标 | 对比维度 | 条件 A | 条件 B | n | acc A | acc B | 差值 [95% CI] "
            "| b/c | p | p_holm |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in self.comparison_rows:
            metric_label = (
                f"{row['metric']} "
                f"({DIMENSION_TITLES.get(row['dimension'], row['dimension'])})"
            )
            lines.append(
                "| {} | {} | {} | `{}` | `{}` | {} | {} | {} | {} {} | {}/{} | {} | {} |".format(
                    row["family"],
                    metric_label,
                    AXIS_TITLES.get(row["axis"], row["axis"]),
                    row["condition_a"],
                    row["condition_b"],
                    row["n_pairs"],
                    _fmt(row["acc_a"]),
                    _fmt(row["acc_b"]),
                    _fmt(row["diff"]),
                    _fmt_ci(row["diff_lo"], row["diff_hi"]),
                    row["mcnemar_b"],
                    row["mcnemar_c"],
                    _fmt(row["p"], 4),
                    _fmt(row["p_holm"], 4),
                )
            )
        if "detect" not in {str(row["family"]) for row in self.comparison_rows}:
            lines.append("")
            lines.append(
                "> 说明：检测一致性 family 需要 `detect` 任务在两侧条件中都存在；"
                "v3 设计里检测实验只跑主条件（不参与消融），因此该 family 在消融条件"
                "对比较中自动跳过，检测指标见第 3 节。"
            )
        self.md["comparisons"] = lines

    def _paired_bootstrap(
        self,
        rows_a: Sequence[Mapping[str, Any]],
        rows_b: Sequence[Mapping[str, Any]],
        names: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        if not rows_a or not rows_b:
            return {}
        metric_fn = self._comparison_metric_fn(names)
        try:
            return stats.paired_bootstrap_difference(
                rows_a,
                rows_b,
                metric_fn,
                cluster_key="cluster_id",
                n_boot=self.n_boot,
                alpha=self.alpha,
                seed=self.seed,
            )
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            self._warn(f"配对 bootstrap 失败（{exc}）；差值置信区间置空")
            return {}

    @staticmethod
    def _comparison_spec(name: str) -> tuple[str, str, str, str]:
        family, dimension = name.split("|", 1)
        for spec_family, metric_name, spec_dimension, prompt_condition in COMPARISON_SPECS:
            if spec_family == family and spec_dimension == dimension:
                return spec_family, metric_name, spec_dimension, prompt_condition
        return family, f"{dimension}_accuracy", dimension, ""

    def _apply_multiple_comparison(self) -> None:
        if not self.comparison_rows:
            return
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(self.comparison_rows):
            grouped[(str(row["family"]), str(row["metric"]))].append(index)
        for indices in grouped.values():
            pvalues = []
            for index in indices:
                p_value = self.comparison_rows[index]["p"]
                pvalues.append(1.0 if p_value is None else float(p_value))
            if self.multiple_comparison == "bh":
                adjusted = stats.bh_correct(pvalues)
            elif self.multiple_comparison in ("none", ""):
                adjusted = list(pvalues)
            else:
                adjusted = stats.holm_correct(pvalues)
            for index, value in zip(indices, adjusted, strict=True):
                self.comparison_rows[index]["p_holm"] = _num(value)

    def _sort_comparison_rows(self) -> None:
        family_rank = {experiment: rank for rank, experiment in enumerate(EXPERIMENT_ORDER)}
        family_rank["detection"] = len(family_rank)
        self.comparison_rows.sort(
            key=lambda row: (
                family_rank.get(str(row["family"]).split(".", 1)[0], 99),
                str(row["metric"]),
                str(row["axis"]),
                str(row["condition_a"]),
                str(row["condition_b"]),
            )
        )

    # ------------------------------------------------------------------
    # Run cost & token usage (token accounting sec. 2026-10-28)
    # ------------------------------------------------------------------
    def _resolve_pricing(self) -> tuple[Pricing | None, str | None, list[str]]:
        """Resolve the pricing snapshot: manifest > manifest.config > config."""
        notes: list[str] = []
        candidates: list[tuple[str, Any]] = [
            ("manifest.pricing", self.manifest.get("pricing") if self.manifest else None),
        ]
        manifest_config = (
            self.manifest.get("config")
            if isinstance(self.manifest.get("config"), Mapping)
            else None
        )
        if manifest_config is not None:
            candidates.append(("manifest.config.pricing", manifest_config.get("pricing")))
        candidates.append(("config.pricing", self.cfg.get("pricing")))
        for source, raw in candidates:
            if not isinstance(raw, Mapping) or not raw:
                continue
            pricing = _pricing_from_mapping(raw)
            if pricing is None:
                continue
            if source == "config.pricing":
                notes.append(
                    "run_manifest 未携带 pricing，使用当前 config 的 pricing 快照"
                    "（可能不是该 run 运行时的历史价格）"
                )
            return pricing, source, notes
        notes.append("未找到 pricing（manifest / config 均无），无法计算成本；仅汇总 token 用量")
        return None, None, notes

    def _resolve_budget(self) -> float | None:
        values = [self.manifest.get("budget_cny") if self.manifest else None]
        values.append(self.cfg.get("generation.budget_cny"))
        for value in values:
            if value in (None, "", 0, 0.0):
                continue
            try:
                budget = float(value)
            except (TypeError, ValueError):
                continue
            if budget > 0:
                return budget
        return None

    def _generation_usage(self, pricing: Pricing) -> dict[str, Any] | None:
        """Stream ``generations.jsonl`` into token / latency / cost totals."""
        path = self.run_dir / GENERATIONS_NAME
        if not path.exists() or path.stat().st_size == 0:
            return None
        counters = {"n_lines": 0, "n_malformed": 0, "n_usage_records": 0}

        def _records():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (TypeError, ValueError):
                        counters["n_malformed"] += 1
                        continue
                    counters["n_lines"] += 1
                    if isinstance(record, Mapping) and record.get("usage"):
                        counters["n_usage_records"] += 1
                    yield record

        try:
            summary = summarize_generations(_records(), pricing)
        except OSError as exc:  # pragma: no cover - unreadable file
            self._usage_notes.append(f"读取 generations.jsonl 失败（{exc}）；无法汇总用量")
            return None
        if counters["n_lines"] == 0:
            return None
        summary.update(counters)
        return summary

    def _usage_from_manifest_counts(self) -> dict[str, Any] | None:
        """Fallback: use the runner's own accounting stored in ``manifest.counts``."""
        counts = self.manifest.get("counts") if self.manifest else None
        if not isinstance(counts, Mapping):
            return None
        relevant = (
            "n_calls",
            "n_ok",
            "n_error",
            "prompt_tokens",
            "input_tokens",
            "cached_tokens",
            "completion_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cost_cny",
        )
        if not any(counts.get(key) is not None for key in relevant):
            return None
        n_ok = _as_int_or_none(counts.get("n_ok"))
        n_errors = _as_int_or_none(counts.get("n_error"))
        n_calls = _as_int_or_none(counts.get("n_calls"))
        if n_calls is None and n_ok is not None and n_errors is not None:
            n_calls = n_ok + n_errors
        prompt_tokens = counts.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = counts.get("input_tokens")
        completion_tokens = counts.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = counts.get("output_tokens")
        overall = {
            "n_calls": n_calls or 0,
            "n_errors": n_errors or 0,
            "prompt_tokens": _as_int_or_none(prompt_tokens),
            "cached_tokens": _as_int_or_none(counts.get("cached_tokens")),
            "completion_tokens": _as_int_or_none(completion_tokens),
            "reasoning_tokens": _as_int_or_none(counts.get("reasoning_tokens")),
            "input_cost_cny": None,
            "output_cost_cny": None,
            "cost_cny": _as_float_or_none(counts.get("cost_cny")),
            "avg_latency_s": None,
        }
        return {
            "overall": overall,
            "by_condition": {},
            "by_generation": {},
            "n_lines": None,
            "n_malformed": 0,
            "n_usage_records": None,
        }

    @staticmethod
    def _totals_cost(
        totals: Mapping[str, Any], pricing: Pricing | None
    ) -> tuple[float | None, float | None, float | None]:
        """(input, output, total) CNY cost for aggregated token totals."""
        if pricing is None:
            return None, None, None
        prompt = _as_int_or_none(totals.get("prompt_tokens"))
        completion = _as_int_or_none(totals.get("completion_tokens"))
        if prompt is None or completion is None:
            return None, None, None
        cached = _as_int_or_none(totals.get("cached_tokens")) or 0
        input_cost = pricing.input_cost(prompt, cached)
        output_cost = pricing.output_cost(completion)
        return input_cost, output_cost, input_cost + output_cost

    def _usage_results(self) -> None:
        lines: list[str] = []
        self._usage_notes = []
        pricing, pricing_source, pricing_notes = self._resolve_pricing()
        notes = list(pricing_notes)
        budget = self._resolve_budget()
        token_pricing = pricing if pricing is not None else Pricing()
        computed = self._generation_usage(token_pricing)
        counts_usage = self._usage_from_manifest_counts()
        counts = (
            self.manifest.get("counts")
            if isinstance(self.manifest.get("counts"), Mapping)
            else {}
        )
        counts_cost = _as_float_or_none(counts.get("cost_cny")) if counts else None

        summary: dict[str, Any] | None = None
        source: str | None = None
        if computed is not None:
            if computed.get("n_usage_records", 0) == 0 and counts_usage is not None:
                notes.append(
                    "generations.jsonl 中 usage 字段全部缺失；用量改由 manifest counts 汇总"
                )
                summary, source = counts_usage, "manifest_counts"
            else:
                summary, source = computed, "generations.jsonl"
                if computed.get("n_usage_records", 0) == 0:
                    notes.append(
                        "generations.jsonl 中没有任何 usage 字段；只能报告调用数与耗时"
                    )
                if computed.get("n_malformed", 0):
                    notes.append(
                        f"generations.jsonl 有 {computed['n_malformed']} 行无法解析为 JSON，已跳过"
                    )
        elif counts_usage is not None:
            notes.append("未找到 generations.jsonl；用量改由 manifest counts 汇总")
            summary, source = counts_usage, "manifest_counts"

        if summary is None:
            notes.append(
                "无可用用量数据（generations.jsonl 缺失/为空，manifest counts 也没有 token 记录）"
            )
            self.usage_summary = {
                "source": None,
                "generations_path": None,
                "tokens_available": False,
                "n_calls": None,
                "n_ok": None,
                "n_errors": None,
                "prompt_tokens": None,
                "cached_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
                "avg_latency_s": None,
                "n_lines": None,
                "n_malformed": 0,
                "n_usage_records": None,
                "per_condition": {},
            }
            self.cost_summary = {
                "currency": None,
                "pricing_source": pricing_source,
                "pricing": pricing.as_dict() if pricing is not None else None,
                "input_cost_cny": None,
                "output_cost_cny": None,
                "total_cost_cny": None,
                "n_calls": None,
                "per_generation_condition": {},
                "budget_cny": budget,
                "budget_remaining_cny": None,
                "budget_used_pct": None,
                "budget_status": "unavailable",
                "status": "no_data",
                "notes": notes,
            }
            lines.append("### 6.1 总览")
            lines.append("")
            lines.append(
                "- 没有可用的用量数据：`generations.jsonl` 缺失/为空，"
                "`run_manifest.json` 的 `counts` 也没有 token/成本记录。"
            )
            lines.append(
                "- 成本核算需要 runner 记录的 `usage`（或 manifest counts）与 pricing 配置；"
                "本 run 无法报告花费。"
            )
            lines.extend(["", "### 6.4 降级说明", ""])
            lines.extend([f"- {note}" for note in notes])
            self.md["usage"] = lines
            return

        overall = dict(summary.get("overall") or {})
        n_calls = _as_int_or_none(overall.get("n_calls")) or 0
        n_errors = _as_int_or_none(overall.get("n_errors")) or 0
        n_ok = max(n_calls - n_errors, 0)
        tokens_available = False
        if source == "generations.jsonl":
            tokens_available = (summary.get("n_usage_records") or 0) > 0
        else:
            tokens_available = any(
                overall.get(key) is not None
                for key in ("prompt_tokens", "completion_tokens")
            )

        def token_value(totals: Mapping[str, Any], key: str) -> int | None:
            if not tokens_available:
                return None
            return _as_int_or_none(totals.get(key))

        def row_costs(
            totals: Mapping[str, Any],
        ) -> tuple[float | None, float | None, float | None]:
            input_cost, output_cost, total_cost = self._totals_cost(totals, pricing)
            if total_cost is not None:
                return input_cost, output_cost, total_cost
            if source == "manifest_counts":
                total_from_counts = _as_float_or_none(totals.get("cost_cny"))
                return None, None, total_from_counts
            return None, None, None

        def make_usage_row(
            scope: str,
            condition_id: str,
            generation_condition: str,
            totals: Mapping[str, Any],
        ) -> dict[str, Any]:
            input_cost, output_cost, total_cost = row_costs(totals)
            return {
                "scope": scope,
                "condition_id": condition_id,
                "generation_condition": generation_condition,
                "n_calls": _as_int_or_none(totals.get("n_calls")) or 0,
                "n_errors": _as_int_or_none(totals.get("n_errors")) or 0,
                "prompt_tokens": token_value(totals, "prompt_tokens"),
                "cached_tokens": token_value(totals, "cached_tokens"),
                "completion_tokens": token_value(totals, "completion_tokens"),
                "reasoning_tokens": token_value(totals, "reasoning_tokens"),
                "input_cost_cny": _round4(input_cost),
                "output_cost_cny": _round4(output_cost),
                "total_cost_cny": _round4(total_cost),
                "avg_latency_s": _round4(totals.get("avg_latency_s")),
            }

        overall_row = make_usage_row("overall", "", "", overall)
        self.usage_rows = [overall_row]
        overall_input = _as_float_or_none(overall_row["input_cost_cny"])
        overall_output = _as_float_or_none(overall_row["output_cost_cny"])
        overall_total = _as_float_or_none(overall_row["total_cost_cny"])
        counts_used_as_cost = False
        if overall_total is None and counts_cost is not None:
            overall_total = counts_cost
            counts_used_as_cost = True
            overall_row["total_cost_cny"] = _round4(counts_cost)
            notes.append(
                f"成本取自 run_manifest.counts.cost_cny={counts_cost:.6f} CNY"
                "（缺少 pricing 明细，无法拆分输入/输出成本）"
            )

        prompt_tokens = token_value(overall, "prompt_tokens")
        completion_tokens = token_value(overall, "completion_tokens")
        reasoning_tokens = token_value(overall, "reasoning_tokens")
        cached_tokens = token_value(overall, "cached_tokens")
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        avg_latency = _round4(overall.get("avg_latency_s"))

        per_condition: dict[str, Any] = {}
        for condition_id, totals in (summary.get("by_condition") or {}).items():
            condition_id = str(condition_id)
            generation = (
                generation_key(condition_id) if "__" in condition_id else condition_id
            )
            row = make_usage_row(
                "per_condition", condition_id, generation, dict(totals)
            )
            self.usage_rows.append(row)
            input_cost, output_cost, total_cost = row_costs(dict(totals))
            per_condition[condition_id] = {
                "generation_condition": generation,
                "n_calls": _as_int_or_none(totals.get("n_calls")) or 0,
                "n_errors": _as_int_or_none(totals.get("n_errors")) or 0,
                "prompt_tokens": token_value(totals, "prompt_tokens"),
                "cached_tokens": token_value(totals, "cached_tokens"),
                "completion_tokens": token_value(totals, "completion_tokens"),
                "reasoning_tokens": token_value(totals, "reasoning_tokens"),
                "input_cost_cny": _round4(input_cost),
                "output_cost_cny": _round4(output_cost),
                "total_cost_cny": _round4(total_cost),
                "avg_latency_s": _round4(totals.get("avg_latency_s")),
            }

        per_generation_cost: dict[str, Any] = {}
        generation_rows: list[dict[str, Any]] = []
        for generation, totals in (summary.get("by_generation") or {}).items():
            generation = str(generation)
            totals = dict(totals)
            input_cost, output_cost, total_cost = row_costs(totals)
            per_generation_cost[generation] = {
                "n_calls": _as_int_or_none(totals.get("n_calls")) or 0,
                "n_errors": _as_int_or_none(totals.get("n_errors")) or 0,
                "prompt_tokens": token_value(totals, "prompt_tokens"),
                "cached_tokens": token_value(totals, "cached_tokens"),
                "completion_tokens": token_value(totals, "completion_tokens"),
                "reasoning_tokens": token_value(totals, "reasoning_tokens"),
                "input_cost_cny": _round4(input_cost),
                "output_cost_cny": _round4(output_cost),
                "total_cost_cny": _round4(total_cost),
                "avg_latency_s": _round4(totals.get("avg_latency_s")),
            }
            generation_rows.append(
                {
                    "generation_condition": generation,
                    "n_calls": per_generation_cost[generation]["n_calls"],
                    "n_errors": per_generation_cost[generation]["n_errors"],
                    "prompt_tokens": per_generation_cost[generation]["prompt_tokens"],
                    "cached_tokens": per_generation_cost[generation]["cached_tokens"],
                    "completion_tokens": per_generation_cost[generation][
                        "completion_tokens"
                    ],
                    "reasoning_tokens": per_generation_cost[generation][
                        "reasoning_tokens"
                    ],
                    "input_cost_cny": per_generation_cost[generation]["input_cost_cny"],
                    "output_cost_cny": per_generation_cost[generation]["output_cost_cny"],
                    "total_cost_cny": per_generation_cost[generation]["total_cost_cny"],
                    "avg_latency_s": per_generation_cost[generation]["avg_latency_s"],
                }
            )

        # -- budget status -------------------------------------------------
        if budget is None:
            budget_info: dict[str, Any] = {
                "budget_cny": None,
                "cost_cny": _round4(overall_total),
                "within_budget": None,
                "used_fraction": None,
                "used_pct": None,
                "remaining_cny": None,
                "status": "no_budget",
            }
        elif overall_total is None:
            budget_info = {
                "budget_cny": budget,
                "cost_cny": None,
                "within_budget": None,
                "used_fraction": None,
                "used_pct": None,
                "remaining_cny": None,
                "status": "unavailable",
            }
        else:
            status = budget_status(overall_total, budget)
            within = status.get("within_budget")
            budget_info = {
                "budget_cny": budget,
                "cost_cny": _round4(overall_total),
                "within_budget": within,
                "used_fraction": _as_float_or_none(status.get("used_fraction")),
                "used_pct": _round4(
                    (overall_total / budget) * 100.0 if budget else None
                ),
                "remaining_cny": _round4(budget - overall_total),
                "status": "within_budget" if within else "over_budget",
            }
        if isinstance(counts, Mapping) and counts.get("n_skipped_budget"):
            notes.append(
                "该 run 因预算 guard 提前停止："
                f"n_skipped_budget={counts.get('n_skipped_budget')}"
            )
        if self.manifest.get("status") == "budget_exceeded":
            self._warn("run 状态为 budget_exceeded：预算 guard 提前停止了生成请求")

        # -- cross-checks against the runner's own accounting --------------
        counts_n_calls = _as_int_or_none(counts.get("n_calls")) if counts else None
        if source == "generations.jsonl" and counts_n_calls is not None:
            if counts_n_calls != n_calls:
                notes.append(
                    f"manifest counts.n_calls={counts_n_calls} 与 generations 行数 "
                    f"{n_calls} 不一致（可能存在未写入 generations 的调用）"
                )
        if (
            source == "generations.jsonl"
            and not counts_used_as_cost
            and counts_cost is not None
            and overall_total is not None
        ):
            tolerance = max(1e-4, 0.001 * max(abs(counts_cost), 1.0))
            if abs(overall_total - counts_cost) > tolerance:
                notes.append(
                    f"manifest counts.cost_cny={counts_cost:.6f} 与按 generations usage "
                    f"重算的 {overall_total:.6f} 不一致"
                )

        cost_status = "computed"
        if overall_total is None:
            cost_status = "no_pricing" if pricing is None else "no_usage"
        currency = None
        raw_pricing = self.manifest.get("pricing") if self.manifest else None
        if isinstance(raw_pricing, Mapping):
            currency = raw_pricing.get("currency")
        if currency is None and pricing is not None:
            currency = pricing.currency
        if currency is None and overall_total is not None:
            currency = "CNY"

        self.usage_summary = {
            "source": source,
            "generations_path": (
                str(self.run_dir / GENERATIONS_NAME)
                if source == "generations.jsonl"
                else None
            ),
            "tokens_available": tokens_available,
            "n_calls": n_calls,
            "n_ok": n_ok,
            "n_errors": n_errors,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "avg_latency_s": avg_latency,
            "n_lines": summary.get("n_lines"),
            "n_malformed": _as_int_or_none(summary.get("n_malformed")) or 0,
            "n_usage_records": summary.get("n_usage_records"),
            "per_condition": per_condition,
        }
        self.cost_summary = {
            "currency": currency,
            "pricing_source": pricing_source,
            "pricing": pricing.as_dict() if pricing is not None else None,
            "input_cost_cny": _round4(overall_input),
            "output_cost_cny": _round4(overall_output),
            "total_cost_cny": _round4(overall_total),
            "n_calls": n_calls,
            "per_generation_condition": per_generation_cost,
            "budget_cny": budget_info.get("budget_cny"),
            "budget_remaining_cny": budget_info.get("remaining_cny"),
            "budget_used_pct": budget_info.get("used_pct"),
            "budget_status": budget_info.get("status"),
            "budget": budget_info,
            "status": cost_status,
            "notes": notes,
        }

        # -- markdown ------------------------------------------------------
        lines.extend(["### 6.1 总览", ""])
        if source == "generations.jsonl":
            lines.append(
                "- 用量来源：`generations.jsonl`（"
                f"{_fmt_count(summary.get('n_lines'))} 行调用记录，"
                f"{_fmt_count(summary.get('n_usage_records'))} 行带 usage"
                + (
                    f"，跳过 {summary['n_malformed']} 行无法解析记录"
                    if summary.get("n_malformed")
                    else ""
                )
                + "）"
            )
        else:
            lines.append(
                "- 用量来源：`run_manifest.json` 的 `counts`（runner 记账总量；"
                "无逐条件明细）"
            )
        lines.append(f"- 调用：{_fmt_count(n_calls)}（成功 {_fmt_count(n_ok)} / 失败 {_fmt_count(n_errors)}）")
        if tokens_available:
            lines.append(
                f"- token：输入 {_fmt_count(prompt_tokens)}（缓存命中 {_fmt_count(cached_tokens)}）；输出 {_fmt_count(completion_tokens)}（reasoning {_fmt_count(reasoning_tokens)}）；合计 {_fmt_count(total_tokens)}"
            )
        else:
            lines.append("- token：不可用（generations.jsonl 未记录 usage）")
        if overall_total is not None:
            if overall_input is not None and overall_output is not None:
                lines.append(
                    f"- 成本：¥{overall_total:.4f}（输入 ¥{overall_input:.4f} + "
                    f"输出 ¥{overall_output:.4f}）"
                )
            else:
                lines.append(f"- 成本：¥{overall_total:.4f}（明细不可用，来自 manifest counts）")
        else:
            lines.append("- 成本：无法计算（缺少 pricing 或 usage，见 6.4 降级说明）")
        lines.append(
            f"- 平均延迟：{avg_latency:.3f} s" if avg_latency is not None
            else "- 平均延迟：—"
        )
        if counts and any(
            counts.get(key) is not None
            for key in ("n_calls", "cost_cny", "prompt_tokens", "completion_tokens")
        ):
            lines.append(
                "- runner 记账（manifest counts）：n_calls={}，cost_cny={}，"
                "prompt_tokens={}，completion_tokens={}".format(
                    _fmt_count(counts.get("n_calls")),
                    (
                        f"{float(counts['cost_cny']):.4f}"
                        if _as_float_or_none(counts.get("cost_cny")) is not None
                        else "—"
                    ),
                    _fmt_count(counts.get("prompt_tokens")),
                    _fmt_count(counts.get("completion_tokens")),
                )
            )
        lines.extend(["", "### 6.2 逐生成条件", ""])
        if generation_rows:
            lines.append(
                "| 生成条件 | 调用数 | 失败 | prompt tokens | cached tokens | "
                "completion tokens | reasoning tokens | 成本 CNY | 平均延迟 (s) |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in generation_rows:
                cost_text = (
                    f"{row['total_cost_cny']:.4f}"
                    if row["total_cost_cny"] is not None
                    else "—"
                )
                latency_text = (
                    f"{row['avg_latency_s']:.3f}"
                    if row["avg_latency_s"] is not None
                    else "—"
                )
                lines.append(
                    "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        row["generation_condition"],
                        _fmt_count(row["n_calls"]),
                        _fmt_count(row["n_errors"]),
                        _fmt_count(row["prompt_tokens"]),
                        _fmt_count(row["cached_tokens"]),
                        _fmt_count(row["completion_tokens"]),
                        _fmt_count(row["reasoning_tokens"]),
                        cost_text,
                        latency_text,
                    )
                )
        else:
            lines.append(
                "_没有逐生成条件用量数据（manifest counts 只提供总量，或该 run "
                "没有按条件记录 usage）。_"
            )
        lines.extend(["", "### 6.3 预算", ""])
        if budget is None:
            lines.append("- 未设置预算：`run_manifest.budget_cny` 为 null。")
        elif overall_total is None:
            lines.append(
                f"- 预算 ¥{budget:.2f}；成本无法计算，预算状态未知。"
            )
        else:
            pct = (overall_total / budget) * 100.0
            if budget_info.get("within_budget"):
                lines.append(
                    f"- 预算 ¥{budget:.2f}，实际 ¥{overall_total:.4f}，"
                    f"占比 {pct:.2f}%，状态：**在预算内**"
                    f"（剩余 ¥{max(budget - overall_total, 0.0):.4f}）。"
                )
            else:
                lines.append(
                    f"- 预算 ¥{budget:.2f}，实际 ¥{overall_total:.4f}，"
                    f"占比 {pct:.2f}%，状态：**超预算**"
                    f"（超出 ¥{overall_total - budget:.4f}）。"
                )
        lines.extend(["", "### 6.4 计费口径与降级说明", ""])
        if pricing is not None:
            lines.append(
                "| 计费项 | 单价 |"
            )
            lines.append("| --- | --- |")
            lines.append(f"| 币种 | {pricing.currency} |")
            lines.append(f"| input（未命中缓存） | ¥{pricing.input_per_mtok:g} / 1M tokens |")
            lines.append(
                f"| input（缓存命中） | ¥{pricing.cache_hit_input_per_mtok:g} / 1M tokens |"
            )
            lines.append(f"| output（含 reasoning） | ¥{pricing.output_per_mtok:g} / 1M tokens |")
            lines.append(
                f"| 音频输入 token | {pricing.audio_tokens_per_second:g} tokens / 秒 |"
            )
            lines.append("")
        lines.append(
            "- 成本公式：`(prompt_tokens - cached_tokens) * input_per_mtok / 1e6 + "
            "cached_tokens * cache_hit_input_per_mtok / 1e6 + completion_tokens * "
            "output_per_mtok / 1e6`；金额保留 4 位小数，`reasoning_tokens` 已包含在 "
            "`completion_tokens` 中，不重复计费。"
        )
        lines.append(f"- 价格快照来源：`{pricing_source or '—'}`。")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
        self.md["usage"] = lines
    # ------------------------------------------------------------------
    # Confusion matrices (main condition, free recognition = experiment 3)
    # ------------------------------------------------------------------
    def _confusion_results(self) -> None:
        lines: list[str] = []
        if not self.confusion_enabled:
            lines.append("_配置 report.confusion_matrices=false：跳过混淆矩阵。_")
            self.md["confusion"] = lines
            return
        rows = [
            row
            for row in self.mainrows
            if str(row.get("experiment", "")) == "exp3"
        ]
        if not rows:
            lines.append("_没有实验三 free 识别行；跳过混淆矩阵。_")
            self.md["confusion"] = lines
            return
        lines.append("### 7.1 主条件 free 识别混淆矩阵（实验三）")
        lines.append("")
        lines.append("> 混淆矩阵文件：`confusion/exp3_{维度}_overall.csv`，逐 voice 见 "
                     "`confusion/exp3_{维度}_voice_{X}.csv`。下表列出 misclassification 的 "
                     "top-10（不含对角线）。")
        lines.append("")
        for dimension in RECOGNITION_DIMS:
            gold_key = f"gold_{dimension}"
            pred_key = f"pred_{dimension}"
            gold = [row.get(gold_key) for row in rows]
            pred = [row.get(pred_key) for row in rows]
            matrix = metrics.confusion_matrix(gold, pred)
            self._write_confusion(matrix, f"exp3_{dimension}_overall.csv")
            top_count, top_rate = self._confusion_top_rows(matrix)
            self.confusion_tables[dimension] = {
                "matrix": matrix,
                "top_count": top_count,
                "top_rate": top_rate,
            }
            lines.append(f"#### {DIMENSION_TITLES[dimension]}")
            lines.append("")
            if top_count:
                lines.append("按次数 top-10：")
                lines.append("")
                lines.extend(
                    _md_table(
                        ["真实值", "预测值", "次数", "占该真实值比例"],
                        [
                            [entry["gold"], entry["pred"], entry["count"],
                             _fmt_rate(entry["rate"])]
                            for entry in top_count
                        ],
                    )
                )
            else:
                lines.append("_没有可报告的混淆（预测全部正确或没有可解析预测）。_")
            if top_rate:
                lines.append("")
                lines.append("按比例 top-10（至少出现 1 次）：")
                lines.append("")
                lines.extend(
                    _md_table(
                        ["真实值", "预测值", "次数", "占该真实值比例"],
                        [
                            [entry["gold"], entry["pred"], entry["count"],
                             _fmt_rate(entry["rate"])]
                            for entry in top_rate
                        ],
                    )
                )
            lines.append("")
            if self.per_voice:
                for voice in _sorted_strings(row.get("voice") for row in rows):
                    voice_rows = [row for row in rows if str(row.get("voice", "")) == voice]
                    if not voice_rows:
                        continue
                    voice_matrix = metrics.confusion_matrix(
                        [row.get(gold_key) for row in voice_rows],
                        [row.get(pred_key) for row in voice_rows],
                    )
                    self._write_confusion(
                        voice_matrix, f"exp3_{dimension}_voice_{voice}.csv"
                    )
        self.md["confusion"] = lines

    def _write_confusion(self, matrix: pd.DataFrame, name: str) -> None:
        path = self.out_dir / CONFUSION_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(path)
        self.artifacts[f"{CONFUSION_DIR}/{name}"] = str(path)

    @staticmethod
    def _confusion_top_rows(
        matrix: pd.DataFrame, limit: int = 10
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []
        if matrix.empty:
            return [], []
        for gold_label in matrix.index:
            support = int(matrix.loc[gold_label].sum())
            for pred_label in matrix.columns:
                if gold_label == pred_label:
                    continue
                count = int(matrix.loc[gold_label, pred_label])
                if count <= 0:
                    continue
                entries.append(
                    {
                        "gold": gold_label,
                        "pred": pred_label,
                        "count": count,
                        "rate": (count / support) if support else None,
                    }
                )
        by_count = sorted(
            entries,
            key=lambda entry: (-entry["count"], -(entry["rate"] or 0.0),
                               str(entry["gold"]), str(entry["pred"])),
        )[:limit]
        by_rate = sorted(
            [entry for entry in entries if entry["count"] > 0],
            key=lambda entry: (-(entry["rate"] or 0.0), -entry["count"],
                               str(entry["gold"]), str(entry["pred"])),
        )[:limit]
        return by_count, by_rate

    # ------------------------------------------------------------------
    # Appendix: data facts / minimal-pair counts / parse failures / metadata
    # ------------------------------------------------------------------
    def _audio_facts(self) -> dict[str, Any] | None:
        try:
            path = self.cfg.path("paths.manifest_dir") / AUDIO_MANIFEST_NAME
        except KeyError:
            return None
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - corrupt manifest edge case
            self._warn(f"读取 audio.parquet 失败（{exc}）；附录数据事实缺失")
            return None
        facts: dict[str, Any] = {"n_audio": int(len(frame))}
        if "voice" in frame.columns:
            facts["voices"] = {
                str(key): int(value)
                for key, value in frame["voice"].value_counts().sort_index().items()
            }
        if "tone" in frame.columns:
            facts["tones"] = {
                str(key): int(value)
                for key, value in frame["tone"].value_counts().sort_index().items()
            }
        if "sample_rate" in frame.columns:
            facts["sample_rates"] = _sorted_strings(frame["sample_rate"].tolist())
        if "channels" in frame.columns:
            facts["channels"] = _sorted_strings(frame["channels"].tolist())
        if "target" in frame.columns:
            facts["n_targets"] = int(frame["target"].nunique())
        return facts

    def _minimal_pair_counts(self) -> tuple[dict[str, dict[str, int]], str]:
        counts = {pair_type: dict(values) for pair_type, values in DESIGN_PAIR_COUNTS.items()}
        source = "docs/experiment-design.md 最小对计数表"
        try:
            path = self.cfg.path("paths.task_dir") / MINIMAL_PAIR_COUNTS_NAME
        except KeyError:
            return counts, source
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict) and loaded:
                    actual = {
                        str(pair_type): {
                            "n_pairs": int(values.get("n_pairs", 0)),
                            "n_instances": int(values.get("n_instances", 0)),
                        }
                        for pair_type, values in loaded.items()
                        if isinstance(values, dict)
                    }
                    if actual:
                        return actual, f"{path}（build-tasks 产物）"
            except (OSError, ValueError, TypeError) as exc:  # pragma: no cover
                self._warn(f"读取最小对计数失败（{exc}）；使用设计文档固定表")
        return counts, source

    def _detection_composition(self) -> dict[str, dict[str, int]]:
        composition: dict[str, dict[str, int]] = {}
        if self.detection.empty or "detection_subset" not in self.detection.columns:
            return composition
        frame = self.detection[self.detection["detection_role"].isin(["neg", "pos"])]
        for subset in _sorted_strings(frame["detection_subset"].tolist()):
            subset_frame = frame[frame["detection_subset"].astype(str) == subset]
            composition[subset] = {
                "n_neg": int((subset_frame["detection_role"] == "neg").sum()),
                "n_pos": int((subset_frame["detection_role"] == "pos").sum()),
            }
        return composition

    def _parse_failure_stats(self) -> dict[str, Any]:
        stats_dict: dict[str, Any] = {}
        if self.items.empty:
            return stats_dict
        frame = self.items
        statuses = frame["parse_status"].astype(str)
        stats_dict["by_status"] = {
            str(key): int(value) for key, value in statuses.value_counts().sort_index().items()
        }
        ok = int((statuses == "ok").sum())
        total = int(len(frame))
        stats_dict["n_total"] = total
        stats_dict["n_ok"] = ok
        stats_dict["n_failures"] = total - ok
        stats_dict["rate"] = (total - ok) / total if total else None
        failures = frame[statuses != "ok"]
        if not failures.empty:
            by_experiment = {
                str(key): int(value)
                for key, value in failures["experiment"].astype(str).value_counts().items()
            }
            stats_dict["by_experiment"] = dict(sorted(by_experiment.items()))
            examples: list[dict[str, Any]] = []
            failure_ids = failures["item_id"].astype(str).tolist()[:5]
            raw_lookup: dict[tuple[str, str], str] = {}
            source = self.items_full
            if (
                not source.empty
                and "raw_response" in source.columns
                and {"item_id", "condition_id"}.issubset(source.columns)
            ):
                for item_id, condition_id, response in zip(
                    source["item_id"].astype(str).tolist(),
                    source["condition_id"].astype(str).tolist(),
                    source["raw_response"].tolist(),
                    strict=True,
                ):
                    raw_lookup.setdefault((item_id, condition_id), str(response or ""))
            for item_id in failure_ids:
                matches = frame[frame["item_id"].astype(str) == item_id]
                if matches.empty:  # pragma: no cover - defensive
                    continue
                item_row = matches.iloc[0]
                raw_response = raw_lookup.get(
                    (item_id, str(item_row.get("condition_id", ""))), ""
                )
                examples.append(
                    {
                        "item_id": item_id,
                        "experiment": str(item_row.get("experiment", "")),
                        "condition_id": str(item_row.get("condition_id", "")),
                        "parse_status": str(item_row.get("parse_status", "")),
                        "raw_response": raw_response[:200],
                    }
                )
            stats_dict["examples"] = examples
        return stats_dict

    def _metadata(self) -> dict[str, Any]:
        manifest = self.manifest
        config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
        model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
        prompts = manifest.get("prompts") if isinstance(manifest.get("prompts"), dict) else {}
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        model_config = config.get("model") if isinstance(config.get("model"), dict) else {}
        return {
            "run_id": manifest.get("run_id", self.run_dir.name),
            "run_dir": str(self.run_dir),
            "run_status": manifest.get("status"),
            "created_at": manifest.get("created_at") or manifest.get("finished_at"),
            "git_commit": manifest.get("git_commit"),
            "config_path": manifest.get("config_path"),
            "config_sha256": manifest.get("config_sha256") or self.cfg.config_sha256(),
            "model": model.get("name") or model_config.get("model") or self.cfg.model_name,
            "model_provider": model.get("provider") or model_config.get("provider"),
            "model_mock": model.get("mock"),
            "temperature": model.get("temperature", model_config.get("temperature")),
            "top_p": model.get("top_p", model_config.get("top_p")),
            "max_output_tokens": model.get(
                "max_output_tokens", model_config.get("max_output_tokens")
            ),
            "prompt_version": prompts.get("version"),
            "prompt_notes": prompts.get("notes"),
            "dataset": {
                "hf_repo": dataset.get("hf_repo") or self.cfg.get("dataset.hf_repo"),
                "hf_revision": dataset.get("hf_revision") or self.cfg.get("dataset.hf_revision"),
                "config_name": dataset.get("config_name") or self.cfg.get("dataset.config_name"),
                "mapping_csv_sha256": dataset.get("mapping_csv_sha256"),
            },
            "plan": manifest.get("plan"),
            "counts": manifest.get("counts"),
            "conditions": manifest.get("conditions"),
            "item_subset_source": manifest.get("item_subset_source"),
        }

    def _pair_coverage(self) -> dict[str, int]:
        """How many distinct minimal pairs of each type the run actually covers."""
        if self.items.empty or "pair_type" not in self.items.columns:
            return {}
        frame = self.items
        if self.main_key:
            main_frame = frame[frame["generation_condition"] == self.main_key]
            if not main_frame.empty:
                frame = main_frame
        pair_type = frame["pair_type"]
        pair_id = frame["pair_id"]
        mask = pair_type.notna() & pair_id.notna()
        if not bool(mask.any()):
            return {}
        coverage: dict[str, int] = {}
        for name, group in frame[mask].groupby(pair_type[mask].astype(str)):
            coverage[str(name)] = int(group["pair_id"].nunique())
        return coverage

    def _appendix(self) -> None:
        audio = self._audio_facts()
        pair_counts, pair_source = self._minimal_pair_counts()
        detection_composition = self._detection_composition()
        parse_stats = self._parse_failure_stats()
        metadata = self._metadata()
        pair_coverage = self._pair_coverage()
        self.summary["data_facts"] = {
            "audio": audio,
            "minimal_pairs": pair_counts,
            "minimal_pair_source": pair_source,
            "minimal_pair_coverage": pair_coverage,
            "detection_composition": detection_composition,
            "parse_failures": parse_stats,
        }
        lines: list[str] = ["### 8.1 数据事实", ""]
        if audio:
            lines.append(
                "- 音频池：{} 条；voice 分布 {}；tone 分布 {}；采样率 {} Hz，声道 {}；"
                "canonical target {} 个。".format(
                    audio.get("n_audio"),
                    audio.get("voices"),
                    audio.get("tones"),
                    audio.get("sample_rates"),
                    audio.get("channels"),
                    audio.get("n_targets"),
                )
            )
        else:
            lines.append(
                "- 未找到 `data/manifests/audio.parquet`（或路径不可用），音频池汇总跳过。"
            )
        lines.append(
            f"- 最小对计数来源：{pair_source}。"
        )
        if pair_counts:
            lines.extend(
                [
                    "",
                    "| 最小对类型 | 设计对数 | 设计实例数 | run 覆盖对数 |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for pair_type, values in pair_counts.items():
                title = PAIR_TYPE_TITLES.get(pair_type, "")
                covered = pair_coverage.get(pair_type)
                lines.append(
                    f"| {pair_type} ({title}) | {values.get('n_pairs', 0)} | "
                    f"{values.get('n_instances', 0)} | "
                    f"{_fmt(covered) if covered is not None else '—'} |"
                )
            lines.append("")
            lines.append(
                "> `include_wide_nasal` 配置为 `{}`（若为 true，前后鼻会额外包含 "
                "ian/iang、uan/uang，计数与设计文档表不同）。".format(
                    self.cfg.get("tasks.minimal_pairs.include_wide_nasal", False)
                )
            )
        if detection_composition:
            lines.extend(["", "### 8.2 检测集构成", "", "| 子集 | neg | pos |", "| --- | --- | --- |"])
            for subset, values in detection_composition.items():
                lines.append(
                    f"| {subset} | {values.get('n_neg', 0)} | {values.get('n_pos', 0)} |"
                )
        if parse_stats:
            lines.extend(
                [
                    "",
                    "### 8.3 解析失败",
                    "",
                    "- 总行数 {}；解析成功 {}；失败 {}（失败率 {}）。".format(
                        parse_stats.get("n_total"),
                        parse_stats.get("n_ok"),
                        parse_stats.get("n_failures"),
                        _fmt_rate(parse_stats.get("rate")),
                    ),
                    "- 按 parse_status：{}。".format(parse_stats.get("by_status")),
                    "- 按实验：{}。".format(parse_stats.get("by_experiment", {})),
                ]
            )
            examples = parse_stats.get("examples") or []
            if examples:
                lines.extend(
                    [
                        "",
                        "失败样例（最多 5 条，原始响应截断 200 字符）：",
                        "",
                        "| item_id | 实验 | parse_status | 响应片段 |",
                        "| --- | --- | --- | --- |",
                    ]
                )
                for example in examples:
                    response = str(example.get("raw_response", "")).replace("\n", " ")
                    response = response.replace("|", "\\|")[:120]
                    lines.append(
                        f"| {example.get('item_id')} | {example.get('experiment')} | "
                        f"{example.get('parse_status')} | {response} |"
                    )
        lines.extend(
            [
                "",
                "### 8.4 设计/任务说明",
                "",
                f"- {DESIGN_REVISION_NOTE}",
                "- 检测指标均来自独立检测实验的 `detection_rows.parquet`（neg=一致样本，"
                "pos=不一致变体），不再由实验一/二拼装；实验一/二只报告声调识别。",
                "",
                "### 8.5 模型 / 参数 / 版本",
                "",
                "| 项目 | 值 |",
                "| --- | --- |",
                f"| run_id | {metadata.get('run_id')} |",
                f"| 状态 | {metadata.get('run_status')} |",
                f"| 创建时间 | {metadata.get('created_at')} |",
                f"| 模型 | {metadata.get('model')} |",
                f"| provider | {metadata.get('model_provider')} |",
                f"| mock | {metadata.get('model_mock')} |",
                f"| temperature / top_p / max_output_tokens | "
                f"{metadata.get('temperature')} / {metadata.get('top_p')} / "
                f"{metadata.get('max_output_tokens')} |",
                f"| prompt 版本 | {metadata.get('prompt_version')} |",
                f"| prompt 说明 | {metadata.get('prompt_notes')} |",
                f"| config sha256 | {metadata.get('config_sha256')} |",
                f"| 数据集 | {metadata.get('dataset', {}).get('hf_repo')} "
                f"@ {metadata.get('dataset', {}).get('hf_revision')} |",
                f"| mapping sha256 | {metadata.get('dataset', {}).get('mapping_csv_sha256')} |",
                f"| git commit | {metadata.get('git_commit')} |",
                f"| 生成条件 | {', '.join(self.condition_keys) if self.condition_keys else '—'} |",
            ]
        )
        self.md["appendix"] = lines

    # ------------------------------------------------------------------
    # Summary / rendering / writing
    # ------------------------------------------------------------------
    def _summary_section(self) -> list[str]:
        lines: list[str] = []
        n_items = int(len(self.items))
        ok = int(self.items["parse_status"].eq("ok").sum()) if n_items else 0
        lines.append(
            "- run_id：`{}`；主条件：`{}`（thinking={}, audio_quality={}, "
            "output_form={}）；items 行数 {}（解析成功 {}，失败 {}，失败率 {}）。".format(
                self.manifest.get("run_id", self.run_dir.name),
                self.main_key,
                self.main_condition.get("thinking"),
                self.main_condition.get("audio_quality"),
                self.main_condition.get("output_form"),
                n_items,
                ok,
                n_items - ok,
                _fmt_rate((n_items - ok) / n_items if n_items else None),
            )
        )
        lines.append("")
        rows: list[list[Any]] = []
        for experiment in EXPERIMENT_ORDER:
            entries = self.headline.get(experiment)
            if not isinstance(entries, Mapping):
                continue
            for prompt_condition, entry in entries.items():
                prompt_title = PROMPT_CONDITION_TITLES.get(
                    prompt_condition, prompt_condition
                )
                for dim in RECOGNITION_DIMS:
                    accuracy = entry.get(f"{dim}_accuracy")
                    if accuracy is None:
                        continue
                    rows.append(
                        [
                            experiment,
                            prompt_title,
                            DIMENSION_TITLES[dim],
                            "accuracy",
                            _fmt(accuracy),
                            _fmt(entry.get(f"{dim}_macro_f1")),
                            entry.get("n", 0),
                            entry.get("n_failures", 0),
                        ]
                    )
                if entry.get("joint_accuracy") is not None:
                    rows.append(
                        [
                            experiment,
                            prompt_title,
                            DIMENSION_TITLES["joint"],
                            "accuracy",
                            _fmt(entry.get("joint_accuracy")),
                            "—",
                            entry.get("n", 0),
                            entry.get("n_failures", 0),
                        ]
                    )
        if rows:
            lines.extend(
                _md_table(
                    ["实验", "提示条件", "维度", "指标", "值", "Macro-F1", "n", "解析失败"],
                    rows,
                )
            )
        else:
            lines.append("_主条件上没有可汇总的识别指标。_")
        if self.detection_headline:
            lines.extend(["", "**检测（main 子集，主条件）**", ""])
            detection_rows = []
            for subset in self.detection_subsets:
                entry = self.detection_headline.get(subset)
                if not entry:
                    continue
                detection_rows.append(
                    [
                        subset,
                        entry.get("n_neg", 0),
                        entry.get("n_pos", 0),
                        _fmt(entry.get("error_precision")),
                        _fmt(entry.get("error_recall")),
                        _fmt(entry.get("error_f1")),
                        _fmt(entry.get("fpr")),
                        _fmt(entry.get("fnr")),
                        _fmt(entry.get("balanced_accuracy")),
                    ]
                )
            if detection_rows:
                lines.extend(
                    _md_table(
                        ["子集", "neg", "pos", "EP", "ER", "Error-F1", "FPR", "FNR",
                         "平衡准确率"],
                        detection_rows,
                    )
                )
        if self.warnings:
            lines.extend(["", "**数据/降级提示**", ""])
            lines.extend([f"- {warning}" for warning in self.warnings])
        return lines

    def _header_section(self) -> list[str]:
        condition = self.main_condition
        lines = [
            "# Qwen3.8-Omni 普通话音素识别与一致性检测实验报告",
            "",
            f"- 生成时间：{self.generated_at}",
            f"- run 目录：`{self.run_dir}`",
            f"- 主条件：prompt 视图各自的 prompt_condition × thinking={condition.get('thinking')} "
            f"× audio_quality={condition.get('audio_quality')} × "
            f"output_form={condition.get('output_form')}，即 generation 维度 "
            f"`{self.main_key}`",
            "- 数据来源：`items.parquet` + `detection_rows.parquet`（run 内已有预测，报告不重算预测）",
            "- 统计口径：95% percentile bootstrap CI，以 `cluster_id`（audio）为重采样单位，"
            "`voice` 分层；条件比较用 McNemar + 配对 bootstrap；多重比较见 `comparisons.csv`",
            f"- 任务/数据版本：{DESIGN_REVISION_NOTE}",
        ]
        if self.item_subset_source:
            lines.append(
                f"- **子集 run**：`item_subset_source={self.item_subset_source}`；"
                "本报告的主结果/检测指标基于消融子集，不是全量数据口径。"
            )
        lines.append("")
        return lines

    def _write_csv(self, name: str, rows: Sequence[Mapping[str, Any]],
                   columns: Sequence[str]) -> None:
        path = self.out_dir / name
        frame = pd.DataFrame(list(rows), columns=list(columns))
        frame.to_csv(path, index=False)
        self.artifacts[name] = str(path)

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        path = self.out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
        self.artifacts[name] = str(path)

    def _render_report(self) -> None:
        sections: list[str] = []
        sections.extend(self._header_section())
        for key, title in (
            ("summary", "## 1. 摘要"),
            ("main", "## 2. 主结果（overall / 逐 voice / 逐最小对类型）"),
            ("detection", "## 3. 检测实验（一致性检测：target + target_tone）"),
            ("ablation", "## 4. 消融（生成条件 × 子集）"),
            ("comparisons", "## 5. 条件对统计比较"),
            ("usage", "## 6. 成本与用量"),
            ("confusion", "## 7. 混淆矩阵"),
            ("appendix", "## 8. 附录"),
        ):
            body = self.md.get(key) or []
            sections.append(title)
            sections.append("")
            sections.extend(body)
            sections.append("")
        sections.append("## 9. 工件清单")
        sections.append("")
        for name in sorted(self.artifacts):
            sections.append(f"- `{name}`")
        sections.append("")
        report_path = self.out_dir / REPORT_NAME
        report_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
        self.artifacts[REPORT_NAME] = str(report_path)

    def _build_summary(self) -> dict[str, Any]:
        n_items = int(len(self.items))
        ok = int(self.items["parse_status"].eq("ok").sum()) if n_items else 0
        summary: dict[str, Any] = {
            "run_id": self.manifest.get("run_id", self.run_dir.name),
            "run_dir": str(self.run_dir),
            "out_dir": str(self.out_dir),
            "generated_at": self.generated_at,
            "main_condition": self.main_condition,
            "main_generation_key": self.main_key,
            "generation_conditions": list(self.condition_keys),
            "n_items": n_items,
            "n_parse_ok": ok,
            "n_parse_failures": n_items - ok,
            "parse_failure_rate": _num((n_items - ok) / n_items if n_items else None),
            "bootstrap": {
                "n_boot": self.n_boot,
                "seed": self.seed,
                "alpha": self.alpha,
                "cluster_key": "cluster_id",
                "stratify_key": self.stratify_key,
            },
            "headline": self.headline,
            "detection": self.detection_headline,
            "detection_unbalanced_subsets": getattr(
                self, "detection_unbalanced_subsets", []
            ),
            "baselines": self.baseline_rows,
            "comparisons": self.comparison_rows,
            "n_comparisons": len(self.comparison_rows),
            "usage": self.usage_summary,
            "cost": self.cost_summary,
            "ablation_conditions": [
                key for key in self.condition_keys if key != self.main_key
            ],
            "ablation_subset_source": self.ablation_source or None,
            "data_facts": self.summary.get("data_facts"),
            "metadata": self._metadata(),
            "warnings": self.warnings,
            "artifacts": {},
        }
        return summary

    def run(self) -> dict[str, str]:
        self._load()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._select_main()
        self._main_results()
        self._baselines()
        self._detection_results()
        self._ablation_results()
        self._comparison_results()
        self._usage_results()
        self._confusion_results()
        self._appendix()
        self.md["summary"] = self._summary_section()
        self._write_csv(METRICS_NAME, self.metrics_rows, METRICS_COLUMNS)
        self._write_csv(DETECTION_NAME, self.detection_rows, DETECTION_COLUMNS)
        self._write_csv(ABLATION_NAME, self.ablation_rows, ABLATION_COLUMNS)
        self._write_csv(COMPARISONS_NAME, self.comparison_rows, COMPARISON_COLUMNS)
        self._write_csv(BASELINES_NAME, self.baseline_rows, BASELINE_COLUMNS)
        self._write_csv(USAGE_NAME, self.usage_rows, USAGE_COLUMNS)
        self._render_report()
        summary = self._build_summary()
        summary["artifacts"] = dict(self.artifacts)
        self.summary = summary
        self._write_json(SUMMARY_NAME, summary)
        log.info(
            "report written: %s (%d artifacts, main condition %s)",
            self.out_dir,
            len(self.artifacts),
            self.main_key,
        )
        return dict(self.artifacts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_report(
    cfg: Config,
    run_id_or_dir: str | Path,
    out_dir: str | Path | None = None,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    """Write the markdown report and metric tables for one scored run.

    Parameters
    ----------
    cfg:
        Loaded configuration (``Config.load(...)``); used for repo-relative paths
        and for the default bootstrap / reporting settings.
    run_id_or_dir:
        Either a run id below ``cfg``'s ``paths.results_dir``/``runs`` or the run
        directory itself.
    out_dir:
        Output directory; defaults to the run directory itself.
    n_boot:
        Override ``stats.n_boot`` (useful for quick smoke reports).
    seed:
        Override ``project.seed`` for the bootstrap draws.

    Returns
    -------
    dict[str, str]
        Mapping ``artifact name -> absolute path`` for every written file
        (``report.md``, ``metrics.csv``, ``detection.csv``, ``ablation.csv``,
        ``comparisons.csv``, ``baselines.csv``, ``summary.json`` and the
        ``confusion/*.csv`` matrices).
    """
    reporter = _Reporter(
        cfg,
        run_id_or_dir,
        out_dir=out_dir,
        n_boot=n_boot,
        seed=seed,
    )
    return reporter.run()
