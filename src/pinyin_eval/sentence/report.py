"""Report generation for the sentence-level pilot (design §9.3, §10).

:func:`build_report` turns one scored run directory (``rows_a.parquet`` /
``rows_b.parquet`` / ``rows_c.parquet`` plus the manifest / score / cost
side-car files written by the runner and scorer) into the design §9.3
deliverables and returns a JSON-serialisable summary.  Every number in the
report comes from :mod:`pinyin_eval.sentence.metrics` or
:mod:`pinyin_eval.sentence.metrics.acceptance_checks`; this module performs no
hand computation of headline statistics.

Artifacts written into the run directory
----------------------------------------
``report.md``
    Chinese markdown report: run metadata, task A/B/C tables, C1-vs-C2
    comparison, single-syllable aggregate comparison, cost / parse-failure
    tables, acceptance gates and the design §11 interpretation limits.
``metrics.csv``
    Long-format metric table (all tasks / both conditions / all subsets).
``comparisons.csv``
    Paired ``C1 - C2`` differences (bootstrap CI, McNemar / Holm where defined).
``per_speaker_a.csv``
    Task-A descriptive statistics per AISHELL-3 speaker.
``per_learner_b.csv``
    Task-B macro-F1 / error-F1 / FPR / FNR per OMPAL learner.
``per_learner_c.csv``
    Task-C Pearson / Spearman / MAE / RMSE per OMPAL learner.
``single_syllable_categories.csv``
    Design §7.4 per-category comparison (single-syllable vs in-run stable
    subset) with the single-syllable category distribution as weight.
``acceptance.json``
    Design §10.2 acceptance gates (parse success, alignment coverage, cost).
``report_summary.json``
    Compact machine-readable summary of the run (headline numbers, gate
    status, artifact paths, warnings).
``figures/c_pearson_scatter.svg``
    Task-C predicted vs gold small multiples (one panel per dimension).
``figures/a_accuracy_bar.svg``
    Task-A headline accuracy bars with 95 % CI (C1, C2 when available).
``figures/c1_vs_c2.svg``
    Paired ``C1 - C2`` effect sizes with 95 % CI across A/B/C.

All figures are hand-rolled, self-contained SVG (no matplotlib / scipy).

Frozen CSV schemas
------------------
``metrics.csv``
    ``task, condition, scope, subset, dimension, metric, estimate, ci_lo,
    ci_hi, n``
    * ``task``: ``A`` / ``B`` / ``C``;
    * ``condition``: ``C1`` / ``C2``;
    * ``scope``: ``headline`` (A), ``subset`` (A special subsets),
      ``n1_diagnostic`` (A sandhi diagnostic), ``dimension`` (B/C),
      ``confusion`` (B tp/fp/fn/tn counts), ``zero_initial_consonant`` (B),
      ``native_anchor`` (C), ``overall`` (B/C parse-free aggregate);
    * ``subset``: A special subset id (``stable`` / ``neutral`` / ``yibu`` /
      ``erhua``) or empty; ``dimension``: ``initial`` / ``final`` / ``tone`` /
      ``joint`` / ``mean_dimension`` / empty;
    * ``estimate`` / ``ci_lo`` / ``ci_hi``: bootstrap point estimate and
      percentile interval (``None`` when undefined); ``n``: denominator.

``comparisons.csv``
    ``task, metric, dimension, estimate, ci_lo, ci_hi, p_value, p_holm,
    n_pairs, direction, method``
    * ``direction`` is always ``C1 - C2``; ``estimate`` is the paired
      bootstrap difference (A headline metrics, B macro-F1, C Pearson / MAE);
    * ``p_value`` / ``p_holm`` are only filled for the McNemar rows (A
      ``sentence_exact``), otherwise ``None`` (CI instead of p, design §9.2).

``per_speaker_a.csv``
    ``speaker, n_items, joint_accuracy, initial_accuracy, final_accuracy,
    tone_accuracy``

``per_learner_b.csv``
    ``learner, n_items, macro_f1_tone, macro_f1_consonant, macro_f1_vowel,
    f1_error_tone, f1_error_consonant, f1_error_vowel, fpr_tone,
    fpr_consonant, fpr_vowel, fnr_tone, fnr_consonant, fnr_vowel``

``per_learner_c.csv``
    ``learner, n_items, pearson_accuracy, pearson_fluency, pearson_prosody,
    pearson_mean_dimension, spearman_mean_dimension, mae_accuracy,
    mae_fluency, mae_prosody, mae_mean_dimension, rmse_mean_dimension``

``single_syllable_categories.csv``
    ``dimension, category, single_n, single_acc, single_weight, sentence_n,
    sentence_acc, covered``
    * ``dimension``: ``initial`` / ``final`` / ``tone``;
    * ``category``: gold initial/final text (``""`` = zero initial) or tone
      ``1``-``5``;
    * ``single_*``: single-syllable exp3 free counts / accuracy / category
      weight (the distribution used for weighting);
    * ``sentence_*``: in-run AISHELL-3 stable-subset counts / accuracy;
    * ``covered``: the category exists on both sides and therefore enters the
      weighted aggregate (§6.2).

Payload structure (:func:`build_report` assembles it, :func:`render_markdown`
consumes it; every metric block is the verbatim return value of the
corresponding :mod:`pinyin_eval.sentence.metrics` function)
-----------------------------------------------------------------------
``run_id`` / ``generated_at`` / ``run_dir`` / ``main_condition``
    run identity (``main_condition`` is the design main condition ``C1``).
``manifest``
    subset of ``run_manifest.json`` (model, prompt_version, git_commit, seed,
    budget_cny, n_planned, prep_summary, conditions, ...).
``config``
    bootstrap settings + the sentence condition table + single-syllable source.
``available``
    ``{task: {condition: {available, n_items, reason}}}`` — partial runs mark
    a task/condition unavailable with the concrete reason.
``cost``
    runner cost summary (``per_task_condition`` / ``totals``).
``usage``
    ``{source, totals, by_task, by_task_condition}`` — cost + parse accounting
    used by the acceptance gates.
``score_summary`` / ``parse_failures``
    raw scorer summary plus ``{by_task, by_task_condition, rates_by_task,
    overall, n_failure_records, source}``.
``a`` / ``a_c2``, ``b`` / ``b_c2``, ``c`` / ``c_c2``
    :func:`task_a_metrics` / :func:`task_b_metrics` / :func:`task_c_metrics`
    for the ``C1`` / ``C2`` condition (``None`` when a condition is missing).
``compare``
    ``{"A": ..., "B": ..., "C": ...}`` = :func:`compare_conditions` for each
    task, or ``None`` when C1 and C2 cannot be paired.
``acceptance``
    :func:`acceptance_checks` output (design §10.2 gates).
``single_syllable``
    ``{available, weighted_available, weighted_reason, source, source_run_id,
    reference_source, reference_filter, single, sentence_all, sentence_stable,
    deltas, weighted, category_rows, caveats}`` — design §7.4 aggregate and
    category-weighted comparison against the existing single-syllable run
    (``items.parquet`` per-category reference + ``summary.json`` unweighted
    fallback; no per-item pairing).
``c_scatter``
    ``{dimension: [[gold, pred], ...]}`` — non-native task-C points used only
    by the SVG figure (not a metric block).
``warnings`` / ``artifacts``
    degradation notes and ``{relative path: relative path}`` artifact map.
    The returned dict also carries a compact ``summary`` block (written to
    ``report_summary.json``) and is what the CLI prints.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

from ..config import Config
from .metrics import (
    acceptance_checks,
    compare_conditions,
    task_a_metrics,
    task_b_metrics,
    task_c_metrics,
)
from .records import TASK_TITLES
from .runner import cost_summary as runner_cost_summary
from .runner import run_dir_for

log = logging.getLogger(__name__)

__all__ = [
    "ReportError",
    "build_report",
    "render_markdown",
    "write_comparisons_csv",
    "write_figures",
    "write_metrics_csv",
    "write_per_learner_b_csv",
    "write_per_learner_c_csv",
    "write_per_speaker_a_csv",
    "write_single_syllable_categories_csv",
]

# ---------------------------------------------------------------------------
# Artifact names / CSV schemas (frozen; documented in the module docstring)
# ---------------------------------------------------------------------------

REPORT_NAME = "report.md"
METRICS_NAME = "metrics.csv"
COMPARISONS_NAME = "comparisons.csv"
PER_SPEAKER_A_NAME = "per_speaker_a.csv"
PER_LEARNER_B_NAME = "per_learner_b.csv"
PER_LEARNER_C_NAME = "per_learner_c.csv"
SINGLE_SYLLABLE_CATEGORIES_NAME = "single_syllable_categories.csv"
ACCEPTANCE_NAME = "acceptance.json"
SUMMARY_NAME = "report_summary.json"
FIGURES_DIR = "figures"
FIG_C_SCATTER = "c_pearson_scatter.svg"
FIG_A_BAR = "a_accuracy_bar.svg"
FIG_C1_VS_C2 = "c1_vs_c2.svg"

ROWS_NAME = {task: f"rows_{task.lower()}.parquet" for task in ("A", "B", "C")}

METRICS_COLUMNS: tuple[str, ...] = (
    "task",
    "condition",
    "scope",
    "subset",
    "dimension",
    "metric",
    "estimate",
    "ci_lo",
    "ci_hi",
    "n",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "task",
    "metric",
    "dimension",
    "estimate",
    "ci_lo",
    "ci_hi",
    "p_value",
    "p_holm",
    "n_pairs",
    "direction",
    "method",
)

PER_SPEAKER_A_COLUMNS: tuple[str, ...] = (
    "speaker",
    "n_items",
    "joint_accuracy",
    "initial_accuracy",
    "final_accuracy",
    "tone_accuracy",
)

PER_LEARNER_B_COLUMNS: tuple[str, ...] = (
    "learner",
    "n_items",
    "macro_f1_tone",
    "macro_f1_consonant",
    "macro_f1_vowel",
    "f1_error_tone",
    "f1_error_consonant",
    "f1_error_vowel",
    "fpr_tone",
    "fpr_consonant",
    "fpr_vowel",
    "fnr_tone",
    "fnr_consonant",
    "fnr_vowel",
)

SINGLE_SYLLABLE_CATEGORY_COLUMNS: tuple[str, ...] = (
    "dimension",
    "category",
    "single_n",
    "single_acc",
    "single_weight",
    "sentence_n",
    "sentence_acc",
    "covered",
)

PER_LEARNER_C_COLUMNS: tuple[str, ...] = (
    "learner",
    "n_items",
    "pearson_accuracy",
    "pearson_fluency",
    "pearson_prosody",
    "pearson_mean_dimension",
    "spearman_mean_dimension",
    "mae_accuracy",
    "mae_fluency",
    "mae_prosody",
    "mae_mean_dimension",
    "rmse_mean_dimension",
)

# ---------------------------------------------------------------------------
# Labels / ordering used by the markdown report
# ---------------------------------------------------------------------------

CONDITIONS: tuple[str, ...] = ("C1", "C2")
TASKS: tuple[str, ...] = ("A", "B", "C")

TASK_SUBTITLES: dict[str, str] = {
    "A": "整句拼音转写（AISHELL-3，仅音频）",
    "B": "整句发音评测（OMPAL learner，音频 + 文本，词级标签）",
    "C": "整句朗读评分（OMPAL learner，音频 + 文本，句级人工均分）",
}

A_METRIC_ORDER: tuple[str, ...] = (
    "joint_accuracy",
    "initial_accuracy",
    "final_accuracy",
    "tone_accuracy",
    "sentence_exact_rate",
    "cer_tone",
    "cer_stripped",
    "insertion_rate",
    "deletion_rate",
    "alignment_coverage",
)

A_METRIC_LABELS: dict[str, str] = {
    "joint_accuracy": "字级拼音准确率（initial+final+tone 全对）",
    "initial_accuracy": "声母准确率",
    "final_accuracy": "韵母准确率",
    "tone_accuracy": "声调准确率",
    "sentence_exact_rate": "句级完全正确率",
    "cer_tone": "tone-sensitive CER（越低越好）",
    "cer_stripped": "tone-stripped CER（越低越好）",
    "insertion_rate": "插入率（越低越好）",
    "deletion_rate": "删除率（越低越好）",
    "alignment_coverage": "字对齐覆盖率",
}

A_DIMENSION_OF: dict[str, str] = {
    "joint_accuracy": "joint",
    "initial_accuracy": "initial",
    "final_accuracy": "final",
    "tone_accuracy": "tone",
}

A_SUBSET_ORDER: tuple[str, ...] = ("stable", "neutral", "yibu", "erhua")
A_SUBSET_LABELS: dict[str, str] = {
    "stable": "stable（排除轻声/一不/儿化）",
    "neutral": "轻声（neutral）",
    "yibu": "一/不（yibu）",
    "erhua": "儿化（erhua）",
}

A_N1_ROWS: tuple[tuple[str, str, str], ...] = (
    ("joint_accuracy", "n1_joint_accuracy", "字级拼音准确率"),
    ("initial_accuracy", "n1_initial_accuracy", "声母准确率"),
    ("final_accuracy", "n1_final_accuracy", "韵母准确率"),
    ("tone_accuracy", "n1_tone_accuracy", "声调准确率"),
    ("sentence_exact_rate", "n1_sentence_exact_rate", "句级完全正确率"),
    ("cer_tone", "n1_cer_tone", "tone-sensitive CER"),
)

B_DIM_ORDER: tuple[str, ...] = ("tone", "consonant", "vowel")
B_DIM_LABELS: dict[str, str] = {"tone": "声调", "consonant": "声母", "vowel": "韵母"}

C_DIM_ORDER: tuple[str, ...] = ("accuracy", "fluency", "prosody", "mean_dimension")
C_DIM_LABELS: dict[str, str] = {
    "accuracy": "accuracy（准确度）",
    "fluency": "fluency（流利度）",
    "prosody": "prosody（韵律）",
    "mean_dimension": "三维平均",
}

SINGLE_SYLLABLE_DIMS: tuple[tuple[str, str], ...] = (
    ("joint_accuracy", "联合（initial+final+tone）"),
    ("initial_accuracy", "声母"),
    ("final_accuracy", "韵母"),
    ("tone_accuracy", "声调"),
)

DEFAULT_SINGLE_SYLLABLE_SUMMARY = "results/runs/mini128-v1/summary.json"
DEFAULT_SINGLE_SYLLABLE_ITEMS = "results/runs/mini128-v1/items.parquet"
DEFAULT_SINGLE_SYLLABLE_RUN_ID = "mini128-v1"

#: generation condition of the single-syllable main result (design: C1 = think off)
SINGLE_SYLLABLE_MAIN_GENERATION_KEY = "free__think0__clean__json"

#: (dimension, per-category stats column on sentence rows, single-syllable gold
#: column, single-syllable correctness column)
CATEGORY_DIMENSIONS: tuple[tuple[str, str, str, str], ...] = (
    ("initial", "cat_initial_stats", "gold_initial", "initial_correct"),
    ("final", "cat_final_stats", "gold_final", "final_correct"),
    ("tone", "cat_tone_stats", "gold_tone", "tone_correct"),
)

CATEGORY_DIMENSION_LABELS: dict[str, str] = {
    "initial": "声母（initial）",
    "final": "韵母（final）",
    "tone": "声调（tone）",
}

SINGLE_SYLLABLE_CAVEATS: tuple[str, ...] = (
    "语料/域不同：单音节结果来自 TTS 合成单音节（prosodic-abx 子集），句内结果来自 "
    "AISHELL-3 真人朗读整句；两者不是同一批音频/说话人，不能做逐条配对。",
    "加权口径：§6.2 使用单音节 exp3 free（think off / clean / JSON 主条件）的类别分布"
    "作为权重；权重在两侧共同类别上重新归一化，权重覆盖率见 §6.2；只出现在一侧的类别"
    "不参与加权（明细见 §6.3 / `single_syllable_categories.csv`）。",
    "stable 子集排除了轻声、一/不、儿化，只对照共享的稳定音节维度。",
    "只报告“同一维度在单音节 vs 句内”的聚合差值，不宣称整句上下文导致了个体层面的变化。",
)

DESIGN_LIMITATIONS: tuple[str, ...] = (
    "AISHELL-3 的一/不 surface 声调标注不完整（一仅约 65.9% 应用变调）：主指标用 N0 "
    "原始 surface，N1 只作诊断；单个数字不能定论。",
    "儿化合并/拆分歧义会影响字级对齐；本文单列 erhua 子集，必要时需人工抽查。",
    "AISHELL-3 train/test speaker 有重叠：本实验是 zero-shot 评估抽样，不能声称 "
    "unseen speaker / held-out speaker。",
    "AISHELL-3 是朗读语料：结论不外推到自发语音、强口音或对话场景。",
    "OMPAL 发布数据只有 3 位评分者（论文写 4 位）：与论文的口径可比性有限，未做 4 人 ICC。",
    "OMPAL 无 consent/伦理声明；对外发布前需再核许可与合规（CC BY 4.0）。",
    "OMPAL 为法语 L1 learner 子集（本 run 的条数/learner 数见 §1、§3）：B/C 结论只限定在该 learner "
    "集合，不外推所有 L2 或其它母语背景。",
    "B/C 的输入文本为繁体、模型输出可能简繁混用：已用 OpenCC t2s 对齐，插入/删除单独报告。",
    "B 的零声母字 consonant 标签口径可能与主标签不一致：本文单列（§3.3），不混入主结论。",
    "无时间戳/强制对齐：只做逐字 / 编辑距离对齐，不做音素级时序分析。",
    "模型输出格式漂移会表现为解析失败：失败率单独报告（§7），不静默丢弃、也不进指标分母。",
    "A/B/C 使用不同语料与 ground truth，不能互相替代或横向比较；pilot 结论不是产品替代结论。",
)


class ReportError(RuntimeError):
    """Raised when a report cannot be produced at all (no scored rows / bad run dir)."""


# ---------------------------------------------------------------------------
# Small generic helpers
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True for ``None`` / NaN / pandas NA; empty strings are **not** missing."""
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
    """Convert a value to a finite ``float`` (``None`` for missing / NaN / inf)."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _int(value: Any, default: int = 0) -> int:
    """Coerce a counter / id field to ``int`` (missing -> ``default``)."""
    number = _num(value)
    return default if number is None else int(number)


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas scalars and NaN to plain JSON types."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [_jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a metric for markdown (``—`` when missing)."""
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def _fmt_int(value: Any) -> str:
    """Format a counter for markdown (``—`` when missing)."""
    if _is_missing(value):
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    """Format a ``[0, 1]`` rate as a percentage string."""
    number = _num(value)
    if number is None:
        return "—"
    return f"{number * 100:.{digits}f}%"


def _fmt_p(value: Any) -> str:
    """Format a p-value compactly."""
    number = _num(value)
    if number is None:
        return "—"
    if number < 0.001:
        return "<0.001"
    return f"{number:.3f}"


def _fmt_cny(value: Any, digits: int = 4) -> str:
    """Format a CNY amount."""
    number = _num(value)
    if number is None:
        return "—"
    return f"¥{number:.{digits}f}"


def _fmt_ci(entry: Mapping[str, Any] | None, digits: int = 4) -> str:
    """Format one bootstrap entry ``{lo, hi}`` as ``[lo, hi]``."""
    if not isinstance(entry, Mapping):
        return "—"
    lo = _fmt(entry.get("lo"), digits)
    hi = _fmt(entry.get("hi"), digits)
    if lo == "—" or hi == "—":
        return "—"
    return f"[{lo}, {hi}]"


def _entry(block: Any, name: str) -> Mapping[str, Any]:
    """Return ``block[name]`` when it is a mapping, else an empty mapping."""
    if isinstance(block, Mapping):
        sub = block.get(name)
        if isinstance(sub, Mapping):
            return sub
    return {}


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Render a markdown table (cells escaped for ``|``)."""
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for value in row:
            text = "" if value is None else str(value)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _json_load(path: Path, default: Any) -> Any:
    """Read one JSON file, returning ``default`` when absent / unreadable."""
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt side-car
        log.warning("cannot read %s: %s", path, exc)
        return default


def _jsonl_load(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts (``[]`` when absent)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except ValueError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError as exc:  # pragma: no cover - unreadable file
        log.warning("cannot read %s: %s", path, exc)
    return rows


def _write_json(path: Path, payload: Any) -> Path:
    """Write one JSON file (UTF-8, pretty printed, deterministic)."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    """Write a CSV with a frozen column order (missing keys become empty cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return path


def _now_iso() -> str:
    """Local ISO-8601 timestamp used as the report generation time."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _condition_of(row: Mapping[str, Any]) -> str:
    """Condition label of one scored row: ``C1`` / ``C2`` (``C1`` for legacy)."""
    raw = row.get("condition")
    if _is_missing(raw):
        return "C1"
    value = str(raw).strip().upper()
    return value if value in CONDITIONS else "C1"


def _parse_ok(row: Mapping[str, Any]) -> bool:
    """Whether one scored row is parseable (missing ``parse_status`` counts as ok)."""
    status = row.get("parse_status", "ok")
    if _is_missing(status):
        return False
    return str(status) == "ok"


def _unique_items(rows: Sequence[Mapping[str, Any]]) -> int:
    """Number of distinct ``item_id`` values in one row list (all rows when empty ids)."""
    items = {
        str(row.get("item_id"))
        for row in rows
        if not _is_missing(row.get("item_id"))
    }
    return len(items) if items else len(rows)


def _as_bool(value: Any) -> bool | None:
    """Coerce a stored correctness value to ``bool`` (``None`` when missing)."""
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def _category_key(value: Any) -> str | None:
    """Canonical category key: gold initial/final text, or tone ``"1"``-``"5"``."""
    if _is_missing(value):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(value)


def _parse_category_stats(raw: Any) -> dict[str, dict[str, int]] | None:
    """Parse one ``cat_*_stats`` row value (JSON string or mapping).

    Returns ``{}`` for the explicit "no counts" value (``"{}"``), ``None`` when
    the value is missing / not parseable (old scoring rows).
    """
    if raw is None or _is_missing(raw):
        return None
    payload: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except ValueError:
            return None
    if not isinstance(payload, Mapping):
        return None
    parsed: dict[str, dict[str, int]] = {}
    for category, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        parsed[str(category)] = {
            "n": _int(entry.get("n")),
            "correct": _int(entry.get("correct")),
        }
    return parsed


def _is_stable_row(row: Mapping[str, Any]) -> bool:
    """Design §7.4 stable subset: not neutral / 一不 / erhua."""
    value = row.get("stable")
    if value is not None and not _is_missing(value):
        return bool(value)
    flags = _entry(row, "flags")
    return not any(bool(flags.get(name)) for name in ("neutral", "yibu", "erhua"))


def _category_display(dimension: str, category: Any) -> str:
    """Human-readable category label for markdown."""
    text = "" if category is None else str(category)
    if dimension == "initial" and text == "":
        return "∅（零声母）"
    if dimension == "tone":
        return f"T{text}"
    return f"`{text}`"


def _aggregate_sentence_categories(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], str | None]:
    """Sum the task-A in-run ``cat_*_stats`` counters over stable-subset rows.

    Returns ``(categories, reason)`` where ``categories`` is
    ``{dimension: {category: {"n", "correct", "accuracy"}}}``; ``reason`` is a
    human-readable explanation when the categories are unavailable (old scoring
    rows without the JSON counter columns, empty stable subset, ...).
    """
    if not rows:
        return {}, "任务 A stable 子集没有可评分行"
    first = rows[0]
    missing_columns = [
        column
        for _dimension, column, _gold, _correct in CATEGORY_DIMENSIONS
        if column not in first
    ]
    if missing_columns:
        return {}, (
            "任务 A 评分行缺少 cat_*_stats 列（旧评分行 "
            f"missing: {', '.join(missing_columns)}）"
        )
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, column, _gold, _correct in CATEGORY_DIMENSIONS:
        counters: dict[str, list[int]] = {}
        parsed_any = False
        for row in rows:
            if not _parse_ok(row):
                continue
            parsed = _parse_category_stats(row.get(column))
            if parsed is None:
                continue
            parsed_any = True
            for category, entry in parsed.items():
                n_units = _int(entry.get("n"))
                if n_units <= 0:
                    continue
                bucket = counters.setdefault(str(category), [0, 0])
                bucket[0] += n_units
                bucket[1] += _int(entry.get("correct"))
        if not parsed_any or not counters:
            return {}, f"任务 A stable 子集的 {column} 没有类别计数（旧评分行或全部解析失败）"
        out[dimension] = {
            category: {
                "n": bucket[0],
                "correct": bucket[1],
                "accuracy": (bucket[1] / bucket[0]) if bucket[0] else None,
            }
            for category, bucket in counters.items()
        }
    return out, None


def _compare_categories(
    reference_categories: Mapping[str, Mapping[str, Mapping[str, Any]]],
    reference_weights: Mapping[str, Mapping[str, float]],
    sentence_categories: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Category-weighted aggregate comparison (design §7.4).

    Weights are the single-syllable category distribution; categories missing
    on either side are excluded and the weights are renormalised over the
    categories present on both sides.  Returns ``(weighted, category_rows)``:
    ``weighted`` is ``{dimension: {sentence, reference, delta, coverage,
    n_categories_*}}``, ``category_rows`` is the long per-category table.
    """
    weighted: dict[str, dict[str, Any]] = {}
    category_rows: list[dict[str, Any]] = []
    for dimension, _column, _gold, _correct in CATEGORY_DIMENSIONS:
        reference = dict(reference_categories.get(dimension) or {})
        weights = dict(reference_weights.get(dimension) or {})
        sentence = dict(sentence_categories.get(dimension) or {})
        total_weight = sum(float(weight) for weight in weights.values())
        common = sorted(
            category
            for category in weights
            if category in reference
            and category in sentence
            and reference[category].get("accuracy") is not None
            and sentence[category].get("accuracy") is not None
        )
        covered_weight = sum(float(weights[category]) for category in common)
        coverage = (covered_weight / total_weight) if total_weight else None
        entry: dict[str, Any] = {
            "sentence": None,
            "reference": None,
            "delta": None,
            "coverage": coverage,
            "n_categories_single": len(reference),
            "n_categories_sentence": len(sentence),
            "n_categories_covered": len(common),
        }
        if covered_weight > 0:
            sentence_accuracy = sum(
                float(weights[category]) * float(sentence[category]["accuracy"])
                for category in common
            )
            reference_accuracy = sum(
                float(weights[category]) * float(reference[category]["accuracy"])
                for category in common
            )
            sentence_weighted = sentence_accuracy / covered_weight
            reference_weighted = reference_accuracy / covered_weight
            entry["sentence"] = sentence_weighted
            entry["reference"] = reference_weighted
            entry["delta"] = sentence_weighted - reference_weighted
        weighted[dimension] = entry
        ordered_categories = sorted(
            set(reference) | set(sentence),
            key=lambda category: (-float(weights.get(category, 0.0)), category),
        )
        for category in ordered_categories:
            reference_entry = reference.get(category)
            sentence_entry = sentence.get(category)
            category_rows.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "single_n": reference_entry.get("n") if reference_entry else None,
                    "single_acc": (
                        reference_entry.get("accuracy") if reference_entry else None
                    ),
                    "single_weight": (
                        float(weights[category]) if category in weights else None
                    ),
                    "sentence_n": sentence_entry.get("n") if sentence_entry else None,
                    "sentence_acc": (
                        sentence_entry.get("accuracy") if sentence_entry else None
                    ),
                    "covered": bool(
                        category in reference
                        and category in sentence
                        and category in weights
                    ),
                }
            )
    return weighted, category_rows


# ---------------------------------------------------------------------------
# Run loader / payload assembly
# ---------------------------------------------------------------------------


class _ReportBuilder:
    """Load one scored run, compute the metric blocks and write all artifacts."""

    def __init__(
        self,
        cfg: Config,
        run_id: str,
        *,
        n_boot: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.run_id = str(run_id)
        self.run_dir = self._resolve_run_dir()
        configured_boot = cfg.get("sentence.stats.n_boot", cfg.get("stats.n_boot", 2000))
        self.n_boot = int(n_boot if n_boot is not None else (configured_boot or 2000))
        if self.n_boot < 1:
            raise ReportError(f"n_boot must be a positive integer, got {self.n_boot!r}")
        self.alpha = float(cfg.get("sentence.stats.alpha", cfg.get("stats.alpha", 0.05)) or 0.05)
        seed = cfg.get("sentence.seed", cfg.get("project.seed", 20261101))
        self.seed = int(seed if seed is not None else 20261101)
        self.generated_at = _now_iso()
        self.warnings: list[str] = []

        self.manifest: dict[str, Any] = {}
        self.score_summary: dict[str, Any] = {}
        self.cost: dict[str, Any] = {}
        self.cost_source: str = "none"
        self.parse_failure_records: list[dict[str, Any]] = []
        self.rows: dict[str, list[dict[str, Any]] | None] = {}
        self.splits: dict[str, dict[str, list[dict[str, Any]]]] = {
            task: {condition: [] for condition in CONDITIONS} for task in TASKS
        }

    # -- paths / warnings --------------------------------------------------
    def _resolve_run_dir(self) -> Path:
        raw = Path(str(self.run_id))
        if raw.is_absolute():
            candidate = raw
        elif raw.exists() and raw.is_dir():
            candidate = raw
        else:
            candidate = run_dir_for(self.cfg, self.run_id)
        if not candidate.exists():
            raise ReportError(
                f"run directory not found: {candidate}; "
                f"run `pinyin-eval sentence score --run-id {self.run_id}` first"
            )
        return candidate.resolve()

    def _warn(self, message: str) -> None:
        log.warning(message)
        self.warnings.append(message)

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        manifest = _json_load(self.run_dir / "run_manifest.json", {})
        self.manifest = manifest if isinstance(manifest, Mapping) else {}
        if not self.manifest:
            self._warn("run_manifest.json 缺失或不可读：元信息回退到 config 默认值")

        score_summary = _json_load(self.run_dir / "score_summary.json", {})
        self.score_summary = score_summary if isinstance(score_summary, Mapping) else {}
        if not self.score_summary:
            self._warn("score_summary.json 缺失：解析失败率改为从 rows/parse_failures 推断")

        self._load_cost()
        for task in TASKS:
            self.rows[task] = self._load_rows(task)
        if all(rows is None for rows in self.rows.values()):
            raise ReportError(
                f"{self.run_dir} 中没有可用的评分行（rows_a/b/c.parquet 均缺失）；"
                "先运行 `pinyin-eval sentence score`"
            )
        self.parse_failure_records = _jsonl_load(self.run_dir / "parse_failures.jsonl")
        for task in TASKS:
            rows = self.rows[task] or []
            for row in rows:
                condition = _condition_of(row)
                self.splits[task][condition].append(row)
            raw_conditions = {
                ""
                if _is_missing(row.get("condition"))
                else str(row.get("condition")).strip().upper()
                for row in rows
            }
            if rows and raw_conditions == {""}:
                self._warn(f"任务 {task} 的评分行没有 condition 字段：全部按 C1 处理")
            unknown = raw_conditions - {"", "C1", "C2"}
            if unknown:
                self._warn(f"任务 {task} 存在未知 condition {sorted(unknown)}：已忽略这些行")

    def _load_cost(self) -> None:
        raw = _json_load(self.run_dir / "cost_summary.json", None)
        if isinstance(raw, Mapping) and raw.get("per_task_condition") is not None:
            self.cost = dict(raw)
            self.cost_source = "cost_summary.json"
            return
        try:
            self.cost = runner_cost_summary(self.cfg, self.run_id)
            self.cost_source = "generations.jsonl（runner.cost_summary 重算）"
        except Exception as exc:  # pragma: no cover - defensive
            self._warn(f"成本汇总不可用（cost_summary.json 缺失且 generations.jsonl 无法重算）：{exc}")
            self.cost = {"per_task_condition": {}, "totals": {}}
            self.cost_source = "unavailable"

    def _load_rows(self, task: str) -> list[dict[str, Any]] | None:
        path = self.run_dir / ROWS_NAME[task]
        if not path.exists():
            csv_path = self.run_dir / f"rows_{task.lower()}.csv"
            if not csv_path.exists():
                self._warn(f"任务 {task} 评分行缺失（{ROWS_NAME[task]}）：该任务不可用")
                return None
            try:
                frame = pd.read_csv(csv_path)
            except (OSError, ValueError) as exc:
                raise ReportError(f"无法读取 {csv_path}: {exc}") from exc
        else:
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError) as exc:
                raise ReportError(f"无法读取 {path}: {exc}") from exc
        if frame.empty:
            self._warn(f"{path.name} 为空：任务 {task} 不可用")
            return None
        if "cluster_id" not in frame.columns:
            fallback = next(
                (column for column in ("speaker", "learner") if column in frame.columns),
                None,
            )
            if fallback is None:
                self._warn(f"任务 {task} 评分行缺少 cluster_id：bootstrap 退化为逐行重采样")
                frame = frame.copy()
                frame["cluster_id"] = [f"row-{index}" for index in range(len(frame))]
            else:
                self._warn(f"任务 {task} 评分行缺少 cluster_id：回退到 {fallback} 作为聚类单位")
                frame = frame.copy()
                frame["cluster_id"] = frame[fallback]
        if "condition" not in frame.columns:
            frame = frame.copy()
            frame["condition"] = "C1"
        return frame.to_dict(orient="records")

    # -- metric blocks -----------------------------------------------------
    def _metric_kwargs(self) -> dict[str, Any]:
        return {
            "n_boot": self.n_boot,
            "alpha": self.alpha,
            "seed": self.seed,
            "cluster_key": "cluster_id",
        }

    def _run_metrics(
        self, task: str, condition: str, metric_fn: Any
    ) -> dict[str, Any] | None:
        rows = self.splits[task][condition]
        if not rows:
            return None
        try:
            return metric_fn(rows, **self._metric_kwargs())
        except Exception as exc:  # noqa: BLE001 - report must survive partial data
            self._warn(f"任务 {task} {condition} 指标计算失败：{exc}")
            return None

    def _compare(self, task: str) -> dict[str, Any] | None:
        rows_c1 = self.splits[task]["C1"]
        rows_c2 = self.splits[task]["C2"]
        if not rows_c1 or not rows_c2:
            return None
        if task in ("A", "C"):
            ids_c1 = {str(row.get("item_id")) for row in rows_c1}
            ids_c2 = {str(row.get("item_id")) for row in rows_c2}
            common = ids_c1 & ids_c2
            if not common:
                self._warn(f"任务 {task}: C1/C2 没有共同可评分 item，跳过配对比较")
                return None
            if len(common) != len(ids_c1) or len(common) != len(ids_c2):
                self._warn(
                    f"任务 {task}: C1/C2 可评分 item 数为 {len(ids_c1)}/{len(ids_c2)}，"
                    f"配对比较使用共同交集 {len(common)} 条"
                )
            rows_c1 = [row for row in rows_c1 if str(row.get("item_id")) in common]
            rows_c2 = [row for row in rows_c2 if str(row.get("item_id")) in common]
        try:
            block = compare_conditions(rows_c1, rows_c2, task, **self._metric_kwargs())
        except Exception as exc:  # noqa: BLE001 - report must survive partial data
            self._warn(f"任务 {task} C1 vs C2 配对比较失败：{exc}")
            return None
        block["method"] = "paired_cluster_bootstrap"
        return block

    # -- availability ------------------------------------------------------
    def _build_available(self) -> dict[str, dict[str, dict[str, Any]]]:
        available: dict[str, dict[str, dict[str, Any]]] = {}
        for task in TASKS:
            available[task] = {}
            rows = self.rows[task]
            for condition in CONDITIONS:
                cond_rows = self.splits[task][condition]
                if rows is None:
                    reason = f"{ROWS_NAME[task]} 缺失"
                elif not cond_rows:
                    reason = f"{condition} 条件没有可评分行（解析失败或未生成）"
                else:
                    reason = None
                n_items = _unique_items(cond_rows) if task == "B" else len(cond_rows)
                available[task][condition] = {
                    "available": bool(cond_rows),
                    "n_items": n_items,
                    "reason": reason,
                }
        return available

    # -- parse failures ----------------------------------------------------
    def _build_parse_failures(self) -> dict[str, Any]:
        matrix: dict[str, Counter] = {}

        def bump(key: str, status: str) -> None:
            matrix.setdefault(key, Counter())[status] += 1

        for task in TASKS:
            rows = self.rows[task] or []
            if task == "B":
                seen: set[tuple[str, str]] = set()
                for row in rows:
                    if not _parse_ok(row):
                        continue
                    condition = _condition_of(row)
                    identity = (condition, str(row.get("item_id")))
                    if identity in seen:
                        continue
                    seen.add(identity)
                    bump(f"{task}|{condition}", "ok")
            else:
                for row in rows:
                    if _parse_ok(row):
                        bump(f"{task}|{_condition_of(row)}", "ok")

        failure_source = "none"
        for record in self.parse_failure_records:
            task = str(record.get("task") or "").upper()
            if task not in TASKS:
                continue
            condition = str(record.get("condition") or "").strip().upper() or "C1"
            status = str(record.get("parse_status") or "parse_error")
            bump(f"{task}|{condition}", status)
            failure_source = "parse_failures.jsonl"
        if not self.parse_failure_records:
            for task in ("A", "C"):
                for row in self.rows[task] or []:
                    status = row.get("parse_status", "ok")
                    if _is_missing(status) or str(status) == "ok":
                        continue
                    bump(f"{task}|{_condition_of(row)}", str(status))
                    failure_source = "rows_*.parquet"

        by_task_condition = {
            key: dict(sorted(counts.items(), key=lambda item: (item[0] != "ok", item[0])))
            for key, counts in sorted(matrix.items())
        }
        status_counts = (
            self.score_summary.get("status_counts")
            if isinstance(self.score_summary, Mapping)
            else None
        )
        by_task: dict[str, dict[str, int]] = {}
        if isinstance(status_counts, Mapping):
            for task in TASKS:
                raw = status_counts.get(task)
                if not isinstance(raw, Mapping):
                    continue
                cleaned = {str(status): _int(count) for status, count in raw.items() if _int(count) > 0}
                if cleaned:
                    by_task[task] = cleaned
        if not by_task:
            aggregated: dict[str, Counter] = {}
            for key, counts in by_task_condition.items():
                task = key.split("|", 1)[0]
                aggregated.setdefault(task, Counter()).update(counts)
            by_task = {task: dict(counts) for task, counts in sorted(aggregated.items())}

        rates_by_task: dict[str, dict[str, Any]] = {}
        for task in TASKS:
            counts = by_task.get(task) or {}
            total = sum(counts.values())
            ok = counts.get("ok", 0)
            rates_by_task[task] = {
                "n_calls": total,
                "n_ok": ok,
                "n_failures": total - ok,
                "rate": (ok / total) if total else None,
            }
        overall_calls = sum(entry["n_calls"] for entry in rates_by_task.values())
        overall_ok = sum(entry["n_ok"] for entry in rates_by_task.values())
        source = "score_summary.json" if by_task and status_counts else failure_source
        return {
            "by_task": by_task,
            "by_task_condition": by_task_condition,
            "rates_by_task": rates_by_task,
            "overall": {
                "n_calls": overall_calls,
                "n_ok": overall_ok,
                "n_failures": overall_calls - overall_ok,
                "rate": (overall_ok / overall_calls) if overall_calls else None,
            },
            "n_failure_records": len(self.parse_failure_records),
            "source": source,
        }

    # -- usage / cost ------------------------------------------------------
    def _build_usage(self, parse_block: Mapping[str, Any]) -> dict[str, Any]:
        cost_entries = _entry(self.cost, "per_task_condition")
        by_task_condition = _entry(parse_block, "by_task_condition")
        keys = sorted(set(cost_entries) | set(by_task_condition))
        by_condition: dict[str, dict[str, Any]] = {}
        for key in keys:
            task, _, condition = key.partition("|")
            if task not in TASKS:
                continue
            cost_entry = cost_entries.get(key) if isinstance(cost_entries, Mapping) else None
            cost_entry = cost_entry if isinstance(cost_entry, Mapping) else {}
            status_counts = by_task_condition.get(key)
            status_counts = status_counts if isinstance(status_counts, Mapping) else {}
            has_ok = "ok" in status_counts
            n_ok = _int(status_counts.get("ok")) if has_ok else None
            n_parse_failures = sum(
                _int(count) for status, count in status_counts.items() if status != "ok"
            )
            n_scored = (n_ok or 0) + n_parse_failures
            n_calls = max(_int(cost_entry.get("n_calls")), n_scored)
            n_errors_cost = _int(cost_entry.get("n_errors"))
            non_ok = (n_scored - (n_ok or 0)) if has_ok else None
            n_errors = (
                min(n_errors_cost, non_ok) if non_ok is not None else n_errors_cost
            )
            n_parse = (non_ok - n_errors) if non_ok is not None else None
            by_condition[key] = {
                "task": task,
                "condition": condition,
                "n_calls": n_calls,
                "n_scored_calls": n_scored,
                "n_ok": n_ok,
                "n_errors": n_errors,
                "n_parse_failures": n_parse,
                "n_retries": _int(cost_entry.get("n_retries")),
                "prompt_tokens": _num(cost_entry.get("prompt_tokens")),
                "completion_tokens": _num(cost_entry.get("completion_tokens")),
                "reasoning_tokens": _num(cost_entry.get("reasoning_tokens")),
                "avg_latency_s": _num(cost_entry.get("avg_latency_s")),
                "cost_cny": _num(cost_entry.get("cost_cny")),
                "finish_reasons": dict(cost_entry.get("finish_reasons") or {}),
            }

        by_task: dict[str, dict[str, Any]] = {}
        for task in TASKS:
            entries = [entry for entry in by_condition.values() if entry["task"] == task]
            n_ok_values = [entry["n_ok"] for entry in entries if entry["n_ok"] is not None]
            by_task[task] = {
                "n_calls": sum(entry["n_calls"] for entry in entries),
                "n_scored_calls": sum(entry["n_scored_calls"] for entry in entries),
                "n_ok": sum(n_ok_values) if len(n_ok_values) == len(entries) and entries else None,
                "n_errors": sum(entry["n_errors"] for entry in entries),
                "n_parse_failures": (
                    sum(entry["n_parse_failures"] for entry in entries)
                    if entries and all(entry["n_parse_failures"] is not None for entry in entries)
                    else None
                ),
                "n_retries": sum(entry["n_retries"] for entry in entries),
                "cost_cny": sum(entry["cost_cny"] or 0.0 for entry in entries),
            }

        cost_totals = _entry(self.cost, "totals")
        totals = {
            "n_calls": _int(cost_totals.get("n_calls")) or sum(
                entry["n_calls"] for entry in by_condition.values()
            ),
            "n_scored_calls": sum(entry["n_scored_calls"] for entry in by_condition.values()),
            "n_ok": sum(entry["n_ok"] for entry in by_task.values() if entry["n_ok"] is not None) or None,
            "n_errors": _int(cost_totals.get("n_errors")) or sum(
                entry["n_errors"] for entry in by_condition.values()
            ),
            "n_parse_failures": sum(
                entry["n_parse_failures"]
                for entry in by_condition.values()
                if entry["n_parse_failures"] is not None
            ),
            "n_retries": sum(entry["n_retries"] for entry in by_condition.values()),
            "prompt_tokens": _num(cost_totals.get("prompt_tokens")),
            "completion_tokens": _num(cost_totals.get("completion_tokens")),
            "reasoning_tokens": _num(cost_totals.get("reasoning_tokens")),
            "avg_latency_s": _num(cost_totals.get("avg_latency_s")),
            "cost_cny": (
                _num(cost_totals.get("cost_cny"))
                if _num(cost_totals.get("cost_cny")) is not None
                else sum(entry["cost_cny"] or 0.0 for entry in by_condition.values())
            ),
        }
        return {
            "source": self.cost_source,
            "totals": totals,
            "by_task": by_task,
            "by_task_condition": by_condition,
        }

    # -- single-syllable aggregate comparison (design §7.4) ----------------
    def _load_single_syllable_reference(
        self, summary_data: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Per-category single-syllable reference from ``items.parquet`` (design §7.4).

        Selects the main free-recognition condition of the single-syllable run
        (``experiment=exp3``, ``prompt_condition=free``, ``parse_status=ok`` and
        the summary's ``main_generation_key``) and computes, for initial / final
        / tone: per-category ``n`` / ``correct`` / accuracy plus the category
        distribution used as weights.  Returns ``None`` (with a §9.1 warning)
        when the parquet or a required column is unavailable.
        """
        configured = self.cfg.get("sentence.report.single_syllable_items")
        source = str(configured or DEFAULT_SINGLE_SYLLABLE_ITEMS)
        path = Path(source)
        if not path.is_absolute():
            path = self.cfg.repo_root / path
        relative = self._repo_relative(path)
        if not path.exists():
            self._warn(
                f"未找到单音节逐条结果 {relative}：§6.2 类别加权对照不可用"
                "（§6.1 可回退 summary.json 的无加权结果）"
            )
            return None
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            self._warn(f"无法读取单音节逐条结果 {relative}（{exc}）：§6.2 类别加权对照不可用")
            return None
        required = {"experiment", "prompt_condition", "parse_status"}
        required.update(
            column
            for _dimension, _stats, gold_column, correct_column in CATEGORY_DIMENSIONS
            for column in (gold_column, correct_column)
        )
        missing = sorted(required - set(frame.columns))
        if missing:
            self._warn(
                f"单音节逐条结果 {relative} 缺少列 {missing}：§6.2 类别加权对照不可用"
            )
            return None
        mask = (
            frame["experiment"].astype(str).eq("exp3")
            & frame["prompt_condition"].astype(str).eq("free")
            & frame["parse_status"].astype(str).eq("ok")
        )
        filter_parts = ["experiment=exp3", "prompt_condition=free", "parse_status=ok"]
        # The single-syllable run's main condition is ``C1`` = think off / clean /
        # JSON; its condition id also carries the prompt-condition prefix
        # (``free__think0__clean__json``).  ``summary.main_generation_key`` only
        # has the ``think0__clean__json`` part, so rebuild the id from
        # ``main_condition`` and fall back to the frozen constant.
        main_key = SINGLE_SYLLABLE_MAIN_GENERATION_KEY
        main_condition = summary_data.get("main_condition")
        if isinstance(main_condition, Mapping):
            thinking = bool(main_condition.get("thinking", False))
            quality = str(main_condition.get("audio_quality") or "clean")
            form = str(main_condition.get("output_form") or "json")
            main_key = f"free__think{1 if thinking else 0}__{quality}__{form}"
        condition_column = next(
            (
                column
                for column in ("condition_id", "generation_condition")
                if column in frame.columns
            ),
            None,
        )
        if condition_column is not None:
            mask = mask & frame[condition_column].astype(str).eq(main_key)
            filter_parts.append(f"{condition_column}={main_key}")
        selected = frame[mask]
        if selected.empty:
            self._warn(
                f"单音节逐条结果 {relative} 按主条件过滤后为空（{' & '.join(filter_parts)}）："
                "§6.2 类别加权对照不可用"
            )
            return None
        free_headline = _entry(_entry(_entry(summary_data, "headline"), "exp3"), "free")
        expected_n = _int(free_headline.get("n")) if free_headline else 0
        if expected_n and len(selected) != expected_n:
            self._warn(
                f"单音节参考按主条件过滤后有 {len(selected)} 行，与 summary.json "
                f"exp3.free n={expected_n} 不一致：§6 对照口径可能不一致"
            )
        categories: dict[str, dict[str, dict[str, Any]]] = {}
        weights: dict[str, dict[str, float]] = {}
        unweighted: dict[str, Any] = {"n": int(len(selected))}
        for dimension, _stats, gold_column, correct_column in CATEGORY_DIMENSIONS:
            counters: dict[str, list[int]] = {}
            gold_values = selected[gold_column].tolist()
            correct_values = selected[correct_column].tolist()
            for gold, correct in zip(gold_values, correct_values, strict=True):
                category = _category_key(gold)
                outcome = _as_bool(correct)
                if category is None or outcome is None:
                    continue
                bucket = counters.setdefault(category, [0, 0])
                bucket[0] += 1
                if outcome:
                    bucket[1] += 1
            total = sum(bucket[0] for bucket in counters.values())
            categories[dimension] = {
                category: {
                    "n": bucket[0],
                    "correct": bucket[1],
                    "accuracy": (bucket[1] / bucket[0]) if bucket[0] else None,
                }
                for category, bucket in counters.items()
            }
            weights[dimension] = {
                category: (bucket[0] / total if total else 0.0)
                for category, bucket in counters.items()
            }
            correct_total = sum(bucket[1] for bucket in counters.values())
            unweighted[f"{dimension}_accuracy"] = (
                (correct_total / total) if total else None
            )
        unweighted["joint_accuracy"] = None
        if "joint_correct" in selected.columns:
            outcomes = [
                _as_bool(value) for value in selected["joint_correct"].tolist()
            ]
            valid = [value for value in outcomes if value is not None]
            if valid:
                unweighted["joint_accuracy"] = sum(1 for value in valid if value) / len(valid)
        expected_joint = _num(free_headline.get("joint_accuracy")) if free_headline else None
        actual_joint = _num(unweighted.get("joint_accuracy"))
        if (
            expected_joint is not None
            and actual_joint is not None
            and abs(expected_joint - actual_joint) > 0.005
        ):
            self._warn(
                f"单音节参考 joint={actual_joint:.4f} 与 summary.json exp3.free "
                f"joint={expected_joint:.4f} 不一致：§6 对照口径可能不一致"
            )
        return {
            "source": relative,
            "filter": " & ".join(filter_parts),
            "unweighted": unweighted,
            "categories": categories,
            "weights": weights,
        }

    def _build_single_syllable(self, a_metrics: Mapping[str, Any] | None) -> dict[str, Any]:
        configured = self.cfg.get("sentence.report.single_syllable_summary")
        summary_source = str(configured or DEFAULT_SINGLE_SYLLABLE_SUMMARY)
        summary_run_id = str(
            self.cfg.get("sentence.report.single_syllable_run_id")
            or DEFAULT_SINGLE_SYLLABLE_RUN_ID
        )
        summary_path = Path(summary_source)
        if not summary_path.is_absolute():
            summary_path = self.cfg.repo_root / summary_path
        summary_relative = self._repo_relative(summary_path)
        summary_data = _json_load(summary_path, {}) if summary_path.exists() else {}
        free = _entry(_entry(_entry(summary_data, "headline"), "exp3"), "free")

        block: dict[str, Any] = {
            "available": False,
            "weighted_available": False,
            "weighted_reason": None,
            "source": summary_relative,
            "source_run_id": summary_run_id,
            "reference_source": None,
            "reference_filter": None,
            "single": {},
            "sentence_all": {},
            "sentence_stable": {},
            "deltas": {},
            "weighted": {},
            "category_rows": [],
            "caveats": list(SINGLE_SYLLABLE_CAVEATS),
        }

        subsets = _entry(a_metrics, "subsets")
        stable = _entry(subsets, "stable")
        sentence_headline = _entry(a_metrics, "headline")
        sentence_stable = {
            "n_items": _int(stable.get("n_items")),
            "n_parse_ok": _int(stable.get("n_parse_ok")),
            **{dim: _num(stable.get(dim)) for dim, _label in SINGLE_SYLLABLE_DIMS},
        }
        sentence_all = {
            "n_items": _int(_entry(a_metrics, "n_items")),
            **{dim: _num(sentence_headline.get(dim)) for dim, _label in SINGLE_SYLLABLE_DIMS},
        }

        reference = self._load_single_syllable_reference(summary_data)
        if reference is not None:
            single = dict(reference["unweighted"])
            if single.get("joint_accuracy") is None and free:
                single["joint_accuracy"] = _num(free.get("joint_accuracy"))
            for dimension, _label in SINGLE_SYLLABLE_DIMS:
                if single.get(dimension) is None and free:
                    single[dimension] = _num(free.get(dimension))
            source_used = str(reference["source"])
            reference_filter = str(reference["filter"])
        elif free:
            single = {
                "n": _int(free.get("n")),
                **{dim: _num(free.get(dim)) for dim, _label in SINGLE_SYLLABLE_DIMS},
            }
            source_used = summary_relative
            reference_filter = None
        else:
            if not summary_path.exists():
                self._warn(
                    f"未找到单音节汇总 {summary_relative} 与逐条结果：§6 聚合对照不可用"
                )
            else:
                self._warn(
                    f"单音节汇总 {summary_relative} 中没有 exp3.free 主结果：§6 聚合对照不可用"
                )
            return block
        if all(single.get(dim) is None for dim, _label in SINGLE_SYLLABLE_DIMS):
            self._warn(f"单音节参考 {source_used} 缺少对照维度：§6 聚合对照不可用")
            return block

        deltas: dict[str, float | None] = {}
        for dim, _label in SINGLE_SYLLABLE_DIMS:
            stable_value = _num(sentence_stable.get(dim))
            single_value = _num(single.get(dim))
            deltas[dim] = (
                stable_value - single_value
                if stable_value is not None and single_value is not None
                else None
            )
        block.update(
            {
                "available": True,
                "single": single,
                "sentence_all": sentence_all,
                "sentence_stable": sentence_stable,
                "deltas": deltas,
                "source": source_used,
                "reference_source": (
                    reference["source"] if reference is not None else None
                ),
                "reference_filter": reference_filter,
            }
        )

        stable_rows = [
            row for row in self.splits["A"]["C1"] if _is_stable_row(row)
        ]
        sentence_categories, category_reason = _aggregate_sentence_categories(stable_rows)
        weighted_reason: str | None = None
        if reference is None:
            weighted_reason = "单音节逐条结果 items.parquet 不可用（只有 summary.json 回退）"
        elif category_reason:
            weighted_reason = category_reason
        else:
            weighted, category_rows = _compare_categories(
                reference["categories"],
                reference["weights"],
                sentence_categories,
            )
            if any(
                entry.get("sentence") is not None
                for entry in weighted.values()
            ):
                block["weighted_available"] = True
                block["weighted"] = weighted
                block["category_rows"] = category_rows
            else:
                weighted_reason = "单音节与句内 stable 子集没有共同的类别"
        if not block["weighted_available"]:
            block["weighted_reason"] = weighted_reason
            self._warn(f"§6.2 类别加权对照不可用：{weighted_reason}")
        return block

    def _repo_relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.cfg.repo_root))
        except (ValueError, OSError):
            return str(path)

    # -- task-C scatter points (for the SVG figure only) -------------------
    @staticmethod
    def _build_c_scatter(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[list[float]]]:
        scatter: dict[str, list[list[float]]] = {}
        for dimension in ("accuracy", "fluency", "prosody"):
            points: list[list[float]] = []
            for row in rows:
                if bool(row.get("native")):
                    continue
                if not _parse_ok(row):
                    continue
                gold = _num(row.get(f"gold_{dimension}"))
                pred = _num(row.get(f"pred_{dimension}"))
                if gold is None or pred is None:
                    continue
                points.append([gold, pred])
            scatter[dimension] = points
        return scatter

    # -- payload -----------------------------------------------------------
    def _build_payload(self) -> dict[str, Any]:
        a = self._run_metrics("A", "C1", task_a_metrics)
        a_c2 = self._run_metrics("A", "C2", task_a_metrics)
        b = self._run_metrics("B", "C1", task_b_metrics)
        b_c2 = self._run_metrics("B", "C2", task_b_metrics)
        c = self._run_metrics("C", "C1", task_c_metrics)
        c_c2 = self._run_metrics("C", "C2", task_c_metrics)
        compare = {"A": self._compare("A"), "B": self._compare("B"), "C": self._compare("C")}

        native_prep, native_used = _native_counts(
            {
                "manifest": self.manifest,
                "c": c,
            }
        )
        if native_prep and not native_used:
            self._warn(
                f"C 任务评分行不含 native 锚点 items：prep 清单有 {native_prep} 条 "
                "`ompal_native_items.jsonl`，本 run 未纳入 C 生成/评分；§4.2 锚点为空"
            )

        parse_block = self._build_parse_failures()
        usage_block = self._build_usage(parse_block)
        usage_for_acceptance = {
            task: {
                "n_calls": _int(_entry(usage_block.get("by_task"), task).get("n_calls")),
                "n_errors": _int(_entry(usage_block.get("by_task"), task).get("n_errors")),
                "n_parse_failures": _int(
                    _entry(usage_block.get("by_task"), task).get("n_parse_failures")
                ),
                "cost_cny": _num(_entry(usage_block.get("by_task"), task).get("cost_cny")),
            }
            for task in TASKS
        }
        acceptance = acceptance_checks(a, b, c, usage_for_acceptance)

        manifest = {
            key: self.manifest.get(key)
            for key in (
                "run_id",
                "created_at",
                "model",
                "provider",
                "prompt_version",
                "prompt_fingerprints",
                "git_commit",
                "seed",
                "mock",
                "n_planned",
                "tasks",
                "conditions",
                "pricing",
                "budget_cny",
                "config_path",
                "config_sha256",
                "prep_summary",
                "items_manifests",
                "operator",
            )
            if self.manifest.get(key) is not None
        }
        if not manifest:
            model = self.cfg.get("model.model")
            manifest = {"run_id": self.run_id, "model": model}
        if manifest.get("run_id") in (None, ""):
            manifest["run_id"] = self.run_id

        config_block = {
            "n_boot": self.n_boot,
            "alpha": self.alpha,
            "seed": self.seed,
            "cluster_key": "cluster_id",
            "conditions": self.cfg.get("sentence.conditions") or {},
            "budget_cny": self.cfg.get("sentence.budget_cny"),
            "model": self.cfg.get("model.model"),
            "single_syllable_source": (
                self.cfg.get("sentence.report.single_syllable_summary")
                or DEFAULT_SINGLE_SYLLABLE_SUMMARY
            ),
        }

        payload: dict[str, Any] = {
            "run_id": str(manifest.get("run_id") or self.run_id),
            "generated_at": self.generated_at,
            "run_dir": str(self.run_dir),
            "main_condition": "C1",
            "manifest": manifest,
            "config": config_block,
            "available": self._build_available(),
            "cost": dict(self.cost),
            "usage": usage_block,
            "score_summary": dict(self.score_summary),
            "parse_failures": parse_block,
            "a": a,
            "a_c2": a_c2,
            "b": b,
            "b_c2": b_c2,
            "c": c,
            "c_c2": c_c2,
            "compare": compare,
            "acceptance": acceptance,
            "single_syllable": self._build_single_syllable(a),
            "c_scatter": self._build_c_scatter(self.splits["C"]["C1"]),
            "warnings": list(self.warnings),
        }
        return _jsonable(payload)

    # -- artifact writing / summary ---------------------------------------
    def _artifact_path(self, name: str) -> str:
        return name

    def _build_summary(
        self, payload: Mapping[str, Any], artifacts: Mapping[str, str]
    ) -> dict[str, Any]:
        a = payload.get("a")
        b = payload.get("b")
        c = payload.get("c")
        a_block: dict[str, Any] = {}
        if isinstance(a, Mapping):
            headline = _entry(a, "headline")
            a_block = {
                "n_items": _int(a.get("n_items")),
                "n_parse_ok": _int(a.get("n_parse_ok")),
                "joint_accuracy": _num(headline.get("joint_accuracy")),
                "tone_accuracy": _num(headline.get("tone_accuracy")),
                "sentence_exact_rate": _num(headline.get("sentence_exact_rate")),
                "cer_tone": _num(headline.get("cer_tone")),
            }
        b_block: dict[str, Any] = {"n_items": _int(b.get("n_items")) if isinstance(b, Mapping) else 0}
        if isinstance(b, Mapping):
            for dimension in B_DIM_ORDER:
                b_block[f"{dimension}_macro_f1"] = _num(_entry(b, dimension).get("macro_f1"))
        c_block: dict[str, Any] = {}
        if isinstance(c, Mapping):
            c_block["n_items"] = _int(c.get("n_items"))
            for dimension in C_DIM_ORDER:
                c_block[f"{dimension}_pearson"] = _num(_entry(c, dimension).get("pearson"))
            native_anchor = _entry(c, "native_anchor")
            c_block["native_anchor"] = {
                "n": _int(native_anchor.get("n")),
                "mean_gold": _num(native_anchor.get("mean_gold")),
                "mean_pred": _num(native_anchor.get("mean_pred")),
            }
        acceptance = _entry(payload, "acceptance")
        usage_totals = _entry(_entry(payload, "usage"), "totals")
        parse_overall = _entry(_entry(payload, "parse_failures"), "overall")
        parse_calls = _int(parse_overall.get("n_calls"))
        parse_failures = _int(parse_overall.get("n_failures"))
        summary = {
            "run_id": payload.get("run_id"),
            "run_dir": payload.get("run_dir"),
            "generated_at": payload.get("generated_at"),
            "main_condition": payload.get("main_condition"),
            "bootstrap": {
                "n_boot": self.n_boot,
                "alpha": self.alpha,
                "seed": self.seed,
                "cluster_key": "cluster_id",
            },
            "available": payload.get("available"),
            "headline": {"A": a_block, "B": b_block, "C": c_block},
            "acceptance": {
                "passed": acceptance.get("passed"),
                "parse_success": {
                    task: _entry(_entry(acceptance, "parse_success"), task).get("rate")
                    for task in TASKS
                },
                "alignment_coverage": {
                    task: _entry(_entry(acceptance, "alignment_coverage"), task).get("value")
                    for task in TASKS
                },
                "cost_cny": _entry(acceptance, "cost_cny").get("value"),
            },
            "usage_totals": usage_totals,
            "parse_success_rate": _num(parse_overall.get("rate")),
            "parse_failure_rate": (
                (parse_failures / parse_calls) if parse_calls else None
            ),
            "single_syllable_available": bool(
                _entry(payload, "single_syllable").get("available")
            ),
            "single_syllable_weighted_available": bool(
                _entry(payload, "single_syllable").get("weighted_available")
            ),
            "warnings": list(payload.get("warnings") or []),
            "artifacts": dict(artifacts),
        }
        return _jsonable(summary)

    def run(self) -> dict[str, Any]:
        self._load()
        payload = self._build_payload()
        artifacts: dict[str, str] = {}

        write_metrics_csv(payload, self.run_dir / METRICS_NAME)
        artifacts[METRICS_NAME] = self._artifact_path(METRICS_NAME)
        write_comparisons_csv(payload, self.run_dir / COMPARISONS_NAME)
        artifacts[COMPARISONS_NAME] = self._artifact_path(COMPARISONS_NAME)
        write_per_speaker_a_csv(payload, self.run_dir / PER_SPEAKER_A_NAME)
        artifacts[PER_SPEAKER_A_NAME] = self._artifact_path(PER_SPEAKER_A_NAME)
        write_per_learner_b_csv(payload, self.run_dir / PER_LEARNER_B_NAME)
        artifacts[PER_LEARNER_B_NAME] = self._artifact_path(PER_LEARNER_B_NAME)
        write_per_learner_c_csv(payload, self.run_dir / PER_LEARNER_C_NAME)
        artifacts[PER_LEARNER_C_NAME] = self._artifact_path(PER_LEARNER_C_NAME)
        write_single_syllable_categories_csv(
            payload, self.run_dir / SINGLE_SYLLABLE_CATEGORIES_NAME
        )
        artifacts[SINGLE_SYLLABLE_CATEGORIES_NAME] = self._artifact_path(
            SINGLE_SYLLABLE_CATEGORIES_NAME
        )

        _write_json(self.run_dir / ACCEPTANCE_NAME, payload.get("acceptance") or {})
        artifacts[ACCEPTANCE_NAME] = self._artifact_path(ACCEPTANCE_NAME)

        figures = write_figures(payload, self.run_dir / FIGURES_DIR)
        for name in sorted(figures):
            relative = str(Path(FIGURES_DIR) / name)
            artifacts[relative] = relative

        artifacts[SUMMARY_NAME] = self._artifact_path(SUMMARY_NAME)
        artifacts[REPORT_NAME] = self._artifact_path(REPORT_NAME)
        payload["artifacts"] = dict(artifacts)

        report_text = render_markdown(payload)
        (self.run_dir / REPORT_NAME).write_text(report_text, encoding="utf-8")

        summary = self._build_summary(payload, artifacts)
        payload["summary"] = summary
        _write_json(self.run_dir / SUMMARY_NAME, summary)
        log.info(
            "sentence report written: %s (%d artifacts)",
            self.run_dir,
            len(artifacts),
        )
        return payload


# ---------------------------------------------------------------------------
# Markdown rendering (pure function: payload -> text)
# ---------------------------------------------------------------------------


def render_markdown(payload: Mapping[str, Any]) -> str:
    """Render the complete Chinese markdown report from a payload dict.

    This function is pure and deterministic: it only reads ``payload`` (the
    structure assembled by :func:`build_report`) and formats it.  Missing /
    unavailable metric blocks are reported as ``不可用`` instead of raising.
    """
    lines: list[str] = []
    lines.extend(_md_title(payload))
    lines.extend(_md_metadata(payload))
    lines.extend(_md_task_a(payload))
    lines.extend(_md_task_b(payload))
    lines.extend(_md_task_c(payload))
    lines.extend(_md_compare(payload))
    lines.extend(_md_single_syllable(payload))
    lines.extend(_md_usage(payload))
    lines.extend(_md_acceptance(payload))
    lines.extend(_md_limitations(payload))
    lines.extend(_md_artifacts(payload))
    return "\n".join(lines).rstrip() + "\n"


def _model_line(payload: Mapping[str, Any]) -> str:
    manifest = _entry(payload, "manifest")
    config = _entry(payload, "config")
    model = manifest.get("model") or config.get("model") or "—"
    prompt_version = manifest.get("prompt_version") or "—"
    mock = manifest.get("mock")
    mock_text = "是（mock 客户端）" if mock else "否"
    return f"> 模型：`{model}` · prompt：`{prompt_version}` · mock：{mock_text}"


def _md_title(payload: Mapping[str, Any]) -> list[str]:
    run_id = payload.get("run_id") or "—"
    return [
        f"# Qwen3.8-Omni 整句拼音评测报告（run `{run_id}`）",
        "",
        "> 设计：`docs/sentence-experiment-design.md`（v0.1, 2026-11-01）· "
        f"生成时间：{payload.get('generated_at') or '—'}",
        _model_line(payload),
        "> 本报告由 `pinyin_eval.sentence.report` 自动生成；所有指标来自 "
        "`pinyin_eval.sentence.metrics`，未做手工计算。",
        "",
    ]


def _prep_entry(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    prep = payload.get("manifest")
    prep_summary = _entry(prep, "prep_summary")
    for key in (name, name.lower()):
        value = prep_summary.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _task_a_data_text(payload: Mapping[str, Any]) -> str:
    prep = _prep_entry(payload, "aishell3")
    items = _entry(prep, "items")
    n_items = items.get("n") or prep.get("n_items") or prep.get("n_utterances")
    if n_items is None:
        n_items = _entry(payload, "a").get("n_items")
    per_speaker = _entry(payload, "a").get("per_speaker")
    n_speakers = len(per_speaker) if isinstance(per_speaker, list) and per_speaker else None
    special = _entry(prep, "special_subset")
    parts = [
        "AISHELL-3 抽样 "
        + (f"{_fmt_int(n_items)} 句" if n_items is not None else "句数未知")
    ]
    if n_speakers is not None:
        parts.append(f"/ {_fmt_int(n_speakers)} speakers")
    if special.get("count") is not None:
        parts.append(f"· 特殊子集 {_fmt_int(special.get('count'))} 句")
    return " ".join(parts) + "（真人朗读，仅音频）"


def _native_counts(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """(prep 清单中的 native 条数, 本次 run 评分行里的 native 条数)."""
    prep = _prep_entry(payload, "ompal")
    prep_items = _entry(prep, "items")
    prep_native = _entry(prep_items, "C_native").get("n")
    if prep_native is None:
        cleaning = _entry(prep, "cleaning")
        prep_native = cleaning.get("raw_native")
    used_native = _entry(_entry(payload, "c"), "native_anchor").get("n")
    return (
        _int(prep_native) if prep_native is not None else None,
        _int(used_native) if used_native is not None else None,
    )


def _ompal_data_text(payload: Mapping[str, Any]) -> str:
    prep = _prep_entry(payload, "ompal")
    items = _entry(prep, "items")
    n_learner = (
        _entry(items, "B").get("n")
        or _entry(items, "C").get("n")
        or prep.get("n_learner")
        or prep.get("n_learners")
    )
    if n_learner is None:
        n_learner = _entry(payload, "b").get("n_items")
    n_sentences = _entry(prep, "sentence_inventory").get("n_sentences")
    per_learner = _entry(payload, "b").get("per_learner")
    n_learners = len(per_learner) if isinstance(per_learner, list) and per_learner else None
    prep_native, used_native = _native_counts(payload)
    extra: list[str] = []
    if n_learners is not None:
        extra.append(f"{_fmt_int(n_learners)} learners")
    if n_sentences is not None:
        extra.append(f"{_fmt_int(n_sentences)} 句")
    parts = [
        "OMPAL learner "
        + (f"{_fmt_int(n_learner)} 条" if n_learner is not None else "条数未知")
    ]
    if extra:
        parts.append("· " + " / ".join(extra))
    if prep_native:
        if used_native:
            parts.append(f"· native 锚点 {_fmt_int(prep_native)} 条（本次纳入 {_fmt_int(used_native)}）")
        else:
            parts.append(f"· native 锚点 {_fmt_int(prep_native)} 条（本 run 未纳入）")
    return " ".join(parts) + "（法语 L1，音频 + 文本）"


def _md_metadata(payload: Mapping[str, Any]) -> list[str]:
    manifest = _entry(payload, "manifest")
    config = _entry(payload, "config")
    usage = _entry(_entry(payload, "usage"), "totals")
    parse_overall = _entry(_entry(payload, "parse_failures"), "overall")
    acceptance = _entry(payload, "acceptance")
    budget = manifest.get("budget_cny")
    if budget is None:
        budget = config.get("budget_cny")
    seed = config.get("seed")
    alpha = config.get("alpha")
    n_boot = config.get("n_boot")
    git_commit = manifest.get("git_commit") or "—"
    conditions_text = (
        "C1 = thinking off / clean / JSON（主条件）；"
        "C2 = thinking on / clean / JSON（RQ-Thinking 对照）"
    )
    rows = [
        ("run_id", f"`{payload.get('run_id')}`"),
        ("模型", f"`{manifest.get('model') or config.get('model') or '—'}`"),
        ("prompt_version", f"`{manifest.get('prompt_version') or '—'}`"),
        ("运行创建时间", str(manifest.get("created_at") or "—")),
        ("报告生成时间", str(payload.get("generated_at") or "—")),
        ("git commit", f"`{git_commit}`"),
        ("生成条件", conditions_text),
        ("数据 A", _task_a_data_text(payload)),
        ("数据 B/C", _ompal_data_text(payload)),
        (
            "统计口径",
            f"cluster bootstrap {_fmt_int(n_boot)} 次 · `cluster_id` 聚类 · "
            f"alpha={_fmt(alpha, 3)} · seed={_fmt_int(seed)}",
        ),
        ("计划调用数", _fmt_int(manifest.get("n_planned"))),
        (
            "实际调用 / 评分调用",
            f"{_fmt_int(usage.get('n_calls'))} / {_fmt_int(usage.get('n_scored_calls'))}",
        ),
        (
            "解析成功率",
            f"{_fmt_pct(parse_overall.get('rate'))}"
            f"（ok {_fmt_int(parse_overall.get('n_ok'))}"
            f" / {_fmt_int(parse_overall.get('n_calls'))}）",
        ),
        ("实际成本", _fmt_cny(usage.get("cost_cny"))),
        ("预算上限", _fmt_cny(budget)),
        (
            "验收总结果",
            "通过" if acceptance.get("passed") else "未通过（见 §8）",
        ),
        ("运行目录", f"`{payload.get('run_dir') or '—'}`"),
    ]
    lines = ["## 1. 运行元信息", ""]
    lines.extend(_md_table(("项", "值"), rows))
    lines.append("")
    lines.append("### 1.1 数据可用性")
    lines.append("")
    available = _entry(payload, "available")
    avail_rows = []
    for task in TASKS:
        entry = _entry(available, task)
        cells = [f"任务 {task}：{TASK_TITLES.get(task, task)}"]
        reasons = []
        for condition in CONDITIONS:
            condition_entry = _entry(entry, condition)
            if condition_entry.get("available"):
                cells.append(f"可用（n={_fmt_int(condition_entry.get('n_items'))}）")
            else:
                cells.append("不可用")
                reason = condition_entry.get("reason")
                if reason:
                    reasons.append(f"{condition}: {reason}")
        cells.append("；".join(reasons) if reasons else "—")
        avail_rows.append(cells)
    lines.extend(_md_table(("任务", "C1", "C2", "说明"), avail_rows))
    lines.append("")
    warnings = payload.get("warnings") or []
    if warnings:
        lines.append("> 降级/警告（详见 §9.1）：" + "；".join(str(item) for item in warnings))
        lines.append("")
    return lines


def _md_task_a(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 2. 任务 A：整句拼音转写（AISHELL-3）", ""]
    a = payload.get("a")
    a_c2 = payload.get("a_c2")
    if not isinstance(a, Mapping):
        lines.append(
            "_任务 A 不可用：没有 C1 条件的可评分行（见 §1.1 与 `rows_a.parquet`）。_"
        )
        lines.append("")
        return lines
    lines.extend(
        [
            "评估口径：N0 orthographic normalization（设计 §6.1）；主指标为字级三要素"
            "（initial + final + tone）全对率；95% CI 为按 speaker 聚类的 bootstrap；"
            "插入不进分母、删除按错误计。",
            "",
            "### 2.1 主表（C1 主条件，C2 对照）",
            "",
        ]
    )
    headline = _entry(a, "headline")
    with_ci = _entry(a, "with_ci")
    headline_c2 = _entry(a_c2, "headline")
    with_ci_c2 = _entry(a_c2, "with_ci")
    rows = []
    for metric in A_METRIC_ORDER:
        rows.append(
            [
                A_METRIC_LABELS.get(metric, metric),
                _fmt(headline.get(metric)),
                _fmt_ci(_entry(with_ci, metric)),
                _fmt(headline_c2.get(metric)),
                _fmt_ci(_entry(with_ci_c2, metric)),
            ]
        )
    lines.extend(_md_table(("指标", "C1 估计", "C1 95% CI", "C2 估计", "C2 95% CI"), rows))
    lines.append("")
    lines.append(
        f"- C1 样本：n_items={_fmt_int(a.get('n_items'))} · "
        f"parse-ok={_fmt_int(a.get('n_parse_ok'))} · "
        f"解析失败={_fmt_int(a.get('n_parse_failures'))} · "
        f"无 gold 跳过={_fmt_int(a.get('n_skipped_no_gold'))}"
    )
    if not isinstance(a_c2, Mapping):
        lines.append("- C2 条件不可用：RQ-Thinking 对照见 §1.1。")
    lines.append("")

    lines.extend(["### 2.2 特殊子集（设计 §7.1）", ""])
    subsets = _entry(a, "subsets")
    subset_rows = []
    for subset in A_SUBSET_ORDER:
        block = _entry(subsets, subset)
        if not block:
            continue
        subset_rows.append(
            [
                A_SUBSET_LABELS.get(subset, subset),
                _fmt_int(block.get("n_items")),
                _fmt_int(block.get("n_parse_ok")),
                _fmt(block.get("joint_accuracy")),
                _fmt(block.get("initial_accuracy")),
                _fmt(block.get("final_accuracy")),
                _fmt(block.get("tone_accuracy")),
            ]
        )
    if subset_rows:
        lines.extend(
            _md_table(
                ("子集", "n_items", "n_parse_ok", "joint", "initial", "final", "tone"),
                subset_rows,
            )
        )
    else:
        lines.append("_特殊子集统计不可用。_")
    lines.append("")

    lines.extend(["### 2.3 N0 vs N1（一/不 sandhi 诊断，设计 §6.2）", ""])
    n1 = _entry(a, "n1")
    if n1:
        delta = _entry(n1, "delta_vs_n0")
        n1_rows = []
        for n0_name, n1_name, label in A_N1_ROWS:
            delta_value: Any = None
            if n0_name in ("joint_accuracy", "cer_tone"):
                delta_value = delta.get(n0_name)
            n1_rows.append(
                [
                    label,
                    _fmt(headline.get(n0_name)),
                    _fmt(n1.get(n1_name)),
                    _fmt(delta_value),
                ]
            )
        lines.extend(_md_table(("指标", "N0（主分析）", "N1（诊断）", "N1−N0"), n1_rows))
        lines.append("")
        lines.append(
            "- 主分析固定使用 N0；N1 只是诊断口径，两者不能混在同一个数字里。"
        )
    else:
        lines.append("_N1 诊断不可用。_")
    lines.append("")

    lines.extend(["### 2.4 逐 speaker 描述统计", ""])
    per_speaker = a.get("per_speaker")
    if isinstance(per_speaker, list) and per_speaker:
        speaker_rows = [
            [
                str(row.get("speaker") or "—"),
                _fmt_int(row.get("n_items")),
                _fmt(row.get("joint_accuracy")),
                _fmt(row.get("initial_accuracy")),
                _fmt(row.get("final_accuracy")),
                _fmt(row.get("tone_accuracy")),
            ]
            for row in per_speaker
            if isinstance(row, Mapping)
        ]
        lines.extend(
            _md_table(("speaker", "n_items", "joint", "initial", "final", "tone"), speaker_rows)
        )
        lines.append("")
        lines.append("- 逐 speaker 仅作描述统计；推断统计使用 §2.1 的按 speaker 聚类 CI。")
    else:
        lines.append("_逐 speaker 统计不可用。_")
    lines.append("")
    return lines


def _md_task_b(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 3. 任务 B：整句发音评测（OMPAL）", ""]
    b = payload.get("b")
    b_c2 = payload.get("b_c2")
    if not isinstance(b, Mapping):
        lines.append("_任务 B 不可用：没有 C1 条件的可评分行（见 §1.1 与 `rows_b.parquet`）。_")
        lines.append("")
        return lines
    lines.extend(
        [
            "评估口径（设计 §6.4/§7.2）：正类 = gold 标签 0（发音错误）；"
            "pred_error = 模型判 `*_ok=false`；macro-F1 为 error 类与 ok 类 F1 的均值；"
            "文本对齐失败的字按错误计（fn）。",
            "",
            "### 3.1 三维主指标（含 95% CI）",
            "",
        ]
    )
    rows = []
    with_ci = _entry(b, "with_ci")
    for dimension in B_DIM_ORDER:
        stats = _entry(b, dimension)
        if not stats:
            continue
        ci = _entry(_entry(with_ci, dimension), "macro_f1")
        c2_stats = _entry(b_c2, dimension)
        c2_macro = c2_stats.get("macro_f1") if isinstance(c2_stats, Mapping) else None
        rows.append(
            [
                B_DIM_LABELS.get(dimension, dimension),
                _fmt(stats.get("macro_f1")),
                _fmt_ci(ci),
                _fmt(c2_macro),
                _fmt(stats.get("f1_error")),
                _fmt(stats.get("fpr")),
                _fmt(stats.get("fnr")),
                _fmt(stats.get("accuracy")),
                _fmt_int(stats.get("n_units")),
                _fmt_int(stats.get("n_items")),
            ]
        )
    lines.extend(
        _md_table(
            (
                "维度",
                "macro-F1",
                "95% CI",
                "C2 macro-F1",
                "error-F1",
                "FPR",
                "FNR",
                "准确率",
                "n_units",
                "n_items",
            ),
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"- 文本对齐覆盖率：{_fmt(b.get('alignment_coverage'))} · "
        f"n_items={_fmt_int(b.get('n_items'))} · "
        f"解析失败={_fmt_int(b.get('n_parse_failures'))} · "
        f"无 gold 跳过={_fmt_int(b.get('n_skipped_no_gold'))}"
    )
    lines.append("")

    lines.extend(["### 3.2 混淆矩阵（正类 = 错误）", ""])
    confusion_rows = []
    for dimension in B_DIM_ORDER:
        stats = _entry(b, dimension)
        if not stats:
            continue
        confusion = _entry(stats, "confusion")
        confusion_rows.append(
            [
                B_DIM_LABELS.get(dimension, dimension),
                _fmt_int(confusion.get("tp")),
                _fmt_int(confusion.get("fp")),
                _fmt_int(confusion.get("fn")),
                _fmt_int(confusion.get("tn")),
                _fmt_int(stats.get("n_units")),
            ]
        )
    lines.extend(
        _md_table(("维度", "TP", "FP", "FN", "TN", "n_units"), confusion_rows)
    )
    lines.append("")

    lines.extend(["### 3.3 零声母 consonant 单列（设计 §4.2/§6.4）", ""])
    zero = _entry(b, "zero_initial_consonant")
    if zero:
        lines.extend(
            _md_table(
                ("口径", "n_items", "n_units", "macro-F1", "error-F1", "FPR", "FNR", "准确率"),
                [
                    [
                        "零声母 consonant",
                        _fmt_int(zero.get("n_items")),
                        _fmt_int(zero.get("n_units")),
                        _fmt(zero.get("macro_f1")),
                        _fmt(zero.get("f1_error")),
                        _fmt(zero.get("fpr")),
                        _fmt(zero.get("fnr")),
                        _fmt(zero.get("accuracy")),
                    ]
                ],
            )
        )
        lines.append("")
        lines.append("- 零声母标签口径可能与主标签不一致，因此单列，不混入 §3.1 的主结论。")
    else:
        lines.append("_零声母 consonant 统计不可用。_")
    lines.append("")

    lines.extend(["### 3.4 逐 learner（按 learner 聚类）", ""])
    per_learner = b.get("per_learner")
    if isinstance(per_learner, list) and per_learner:
        learner_rows = []
        for row in per_learner:
            if not isinstance(row, Mapping):
                continue
            macro = _entry(row, "macro_f1")
            fpr = _entry(row, "fpr")
            fnr = _entry(row, "fnr")
            learner_rows.append(
                [
                    str(row.get("learner") or "—"),
                    _fmt_int(row.get("n_items")),
                    _fmt(macro.get("tone")),
                    _fmt(macro.get("consonant")),
                    _fmt(macro.get("vowel")),
                    _fmt(fpr.get("tone")),
                    _fmt(fpr.get("consonant")),
                    _fmt(fpr.get("vowel")),
                    _fmt(fnr.get("tone")),
                    _fmt(fnr.get("consonant")),
                    _fmt(fnr.get("vowel")),
                ]
            )
        lines.extend(
            _md_table(
                (
                    "learner",
                    "n_items",
                    "macro-F1 声调",
                    "macro-F1 声母",
                    "macro-F1 韵母",
                    "FPR 声调",
                    "FPR 声母",
                    "FPR 韵母",
                    "FNR 声调",
                    "FNR 声母",
                    "FNR 韵母",
                ),
                learner_rows,
            )
        )
    else:
        lines.append("_逐 learner 统计不可用。_")
    lines.append("")
    return lines


def _md_task_c(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 4. 任务 C：整句朗读评分（OMPAL）", ""]
    c = payload.get("c")
    if not isinstance(c, Mapping):
        lines.append("_任务 C 不可用：没有 C1 条件的可评分行（见 §1.1 与 `rows_c.parquet`）。_")
        lines.append("")
        return lines
    lines.extend(
        [
            "评估口径（设计 §7.3）：目标是 OMPAL 3 位评分者均值；native 的常数 5 分不进入"
            "相关/误差计算，只作人类锚点；95% CI 为按 learner 聚类的 bootstrap。",
            "",
            "### 4.1 三维相关与误差",
            "",
        ]
    )
    with_ci = _entry(c, "with_ci")
    rows = []
    for dimension in C_DIM_ORDER:
        stats = _entry(c, dimension)
        if not stats:
            continue
        ci = _entry(with_ci, dimension)
        rows.append(
            [
                C_DIM_LABELS.get(dimension, dimension),
                _fmt_int(stats.get("n")),
                _fmt(stats.get("pearson")),
                _fmt_ci(_entry(ci, "pearson")),
                _fmt(stats.get("spearman")),
                _fmt(stats.get("mae")),
                _fmt_ci(_entry(ci, "mae")),
                _fmt(stats.get("rmse")),
            ]
        )
    lines.extend(
        _md_table(
            ("维度", "n", "Pearson", "95% CI", "Spearman", "MAE", "95% CI", "RMSE"), rows
        )
    )
    lines.append("")
    non_native_n = _entry(c, "mean_dimension").get("n")
    lines.append(
        f"- 评分可用行：n_items={_fmt_int(c.get('n_items'))} · "
        f"parse-ok={_fmt_int(c.get('n_parse_ok'))} · "
        f"解析失败={_fmt_int(c.get('n_parse_failures'))} · "
        f"对齐覆盖率={_fmt(c.get('alignment_coverage'))}；"
        f"上表相关/误差口径不含 native（有效配对 n={_fmt_int(non_native_n)}，"
        "native 单列于 §4.2）"
    )
    lines.append("")

    lines.extend(["### 4.2 native 人类满分锚点", ""])
    native = _entry(c, "native_anchor")
    if _int(native.get("n")) > 0:
        lines.append(
            f"- native 条数 n={_fmt_int(native.get('n'))}；"
            f"人工三维均分={_fmt(native.get('mean_gold'))}；"
            f"模型预测三维均分={_fmt(native.get('mean_pred'))}。"
        )
        lines.append(
            "- native 为 L1 人类语音、人工分全为 5：只作人类满分上限/模型偏差参考，"
            "不进入 task-C 的相关/MAE 计算（见上表 n）。"
        )
        per_dimension = _entry(native, "per_dimension")
        if per_dimension:
            lines.append("")
            lines.append("| 维度 | 人工均分 | 模型预测均分 |")
            lines.append("| --- | --- | --- |")
            for dimension in ("accuracy", "fluency", "prosody"):
                entry = _entry(per_dimension, dimension)
                if not entry:
                    continue
                lines.append(
                    f"| {C_DIM_LABELS.get(dimension, dimension)} | "
                    f"{_fmt(entry.get('mean_gold'))} | {_fmt(entry.get('mean_pred'))} |"
                )
    else:
        lines.append(
            "- 本次 run 的评分行里没有 native 锚点 items（见 §9.1 警告）；"
            "native 只作人类满分上限，不进入相关/误差计算。"
        )
    lines.append("")

    lines.extend(["### 4.3 逐 learner", ""])
    per_learner = c.get("per_learner")
    if isinstance(per_learner, list) and per_learner:
        learner_rows = []
        for row in per_learner:
            if not isinstance(row, Mapping):
                continue
            pearson = _entry(row, "pearson")
            mae = _entry(row, "mae")
            rmse = _entry(row, "rmse")
            learner_rows.append(
                [
                    str(row.get("learner") or "—"),
                    _fmt_int(row.get("n_items")),
                    _fmt(pearson.get("accuracy")),
                    _fmt(pearson.get("fluency")),
                    _fmt(pearson.get("prosody")),
                    _fmt(pearson.get("mean_dimension")),
                    _fmt(mae.get("accuracy")),
                    _fmt(mae.get("fluency")),
                    _fmt(mae.get("prosody")),
                    _fmt(mae.get("mean_dimension")),
                    _fmt(rmse.get("mean_dimension")),
                ]
            )
        lines.extend(
            _md_table(
                (
                    "learner",
                    "n_items",
                    "Pearson acc",
                    "Pearson flu",
                    "Pearson pro",
                    "Pearson mean",
                    "MAE acc",
                    "MAE flu",
                    "MAE pro",
                    "MAE mean",
                    "RMSE mean",
                ),
                learner_rows,
            )
        )
    else:
        lines.append("_逐 learner 统计不可用。_")
    lines.append("")

    lines.extend(["### 4.4 散点图", ""])
    if _has_artifact(payload, str(Path(FIGURES_DIR) / FIG_C_SCATTER)):
        lines.append("![任务 C：预测分 vs 人工分（三维小多图）](figures/c_pearson_scatter.svg)")
        lines.append("")
        lines.append(
            "- 每个点 = 一个非 native 评分项；虚线为 y=x；面板标题给出该维度的 Pearson r 与 n。"
        )
    else:
        lines.append("_散点图不可用：没有可绘制的非 native 配对评分。_")
    lines.append("")
    return lines


def _direction_phrase(entry: Mapping[str, Any] | None) -> str:
    """Describe one paired difference entry using only the values it carries."""
    if not isinstance(entry, Mapping):
        return "不可用"
    estimate = _num(entry.get("estimate"))
    lo = _num(entry.get("lo"))
    hi = _num(entry.get("hi"))
    if estimate is None:
        return "不可用"
    if lo is not None and hi is not None:
        if lo > 0:
            return f"C1 更高（点估计 {_fmt(estimate)}）"
        if hi < 0:
            return f"C2 更高（点估计 {_fmt(estimate)}）"
        return f"无显著差异（CI 跨 0，点估计 {_fmt(estimate)}）"
    return f"仅点估计 {_fmt(estimate)}（CI 不可用）"


def _md_compare(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 5. C1 vs C2（thinking off vs thinking on）", ""]
    compare = _entry(payload, "compare")
    lines.extend(
        [
            "方法（设计 §9.2）：同一 item 配对；差值方向恒为 **C1 − C2**；配对 bootstrap"
            "重采样同一 cluster；任务 A 的句级完全正确率额外做 McNemar 精确检验并按任务族"
            " Holm 校正；B/C 只报告差值 CI（无逐 item 的 F1 / 相关量可做 McNemar）。",
            "",
        ]
    )
    if not any(isinstance(compare.get(task), Mapping) for task in TASKS):
        lines.append("_C1/C2 配对比较不可用：缺少任一条件的可评分行（见 §1.1）。_")
        lines.append("")
        return lines

    lines.extend(["### 5.1 任务 A", ""])
    block_a = compare.get("A")
    if isinstance(block_a, Mapping):
        differences = _entry(block_a, "differences")
        rows = []
        for metric in A_METRIC_ORDER:
            entry = differences.get(metric)
            if not isinstance(entry, Mapping):
                continue
            rows.append(
                [
                    A_METRIC_LABELS.get(metric, metric),
                    _fmt(entry.get("estimate")),
                    _fmt_ci(entry),
                    _direction_phrase(entry),
                ]
            )
        lines.extend(_md_table(("指标", "Δ(C1−C2)", "95% CI", "方向"), rows))
        lines.append("")
        mcnemar = _entry(_entry(block_a, "mcnemar"), "sentence_exact")
        holm = _entry(block_a, "holm")
        if mcnemar:
            lines.append(
                f"- 句级完全正确率 McNemar：b={_fmt_int(mcnemar.get('b'))}（仅 C1 对）、"
                f"c={_fmt_int(mcnemar.get('c'))}（仅 C2 对）、"
                f"n_pairs={_fmt_int(mcnemar.get('n_pairs'))}、"
                f"p={_fmt_p(mcnemar.get('p_value'))}、"
                f"Holm 校正 p={_fmt_p(holm.get('sentence_exact'))}"
                f"（{mcnemar.get('method') or '—'}）"
            )
        counts = _entry(block_a, "per_item_joint_counts")
        if counts:
            lines.append(
                f"- 逐 item 字级联合全对计数：C1 全对 {_fmt_int(counts.get('joint_all_correct_c1'))}、"
                f"C2 全对 {_fmt_int(counts.get('joint_all_correct_c2'))}、"
                f"仅 C1 对 {_fmt_int(counts.get('c1_only'))}、仅 C2 对 {_fmt_int(counts.get('c2_only'))}、"
                f"两者都对 {_fmt_int(counts.get('both'))}、都不对 {_fmt_int(counts.get('neither'))}"
            )
        lines.append(f"- 配对 item 数：{_fmt_int(block_a.get('n_items_paired'))}")
    else:
        lines.append("_任务 A 的 C1/C2 配对比较不可用。_")
    lines.append("")

    lines.extend(["### 5.2 任务 B（Δmacro-F1）", ""])
    block_b = compare.get("B")
    if isinstance(block_b, Mapping):
        differences = _entry(_entry(block_b, "differences"), "macro_f1")
        rows = []
        for dimension in B_DIM_ORDER:
            entry = differences.get(dimension)
            if not isinstance(entry, Mapping):
                continue
            rows.append(
                [
                    B_DIM_LABELS.get(dimension, dimension),
                    _fmt(entry.get("estimate")),
                    _fmt_ci(entry),
                    _direction_phrase(entry),
                ]
            )
        lines.extend(_md_table(("维度", "Δmacro-F1", "95% CI", "方向"), rows))
        lines.append("")
        lines.append(f"- 配对 item 数：{_fmt_int(block_b.get('n_items_paired'))}")
    else:
        lines.append("_任务 B 的 C1/C2 配对比较不可用。_")
    lines.append("")

    lines.extend(["### 5.3 任务 C（ΔPearson / ΔMAE）", ""])
    block_c = compare.get("C")
    if isinstance(block_c, Mapping):
        differences = _entry(block_c, "differences")
        pearson_diffs = _entry(differences, "pearson")
        mae_diffs = _entry(differences, "mae")
        rows = []
        for dimension in C_DIM_ORDER:
            pearson_entry = pearson_diffs.get(dimension)
            mae_entry = mae_diffs.get(dimension)
            if not isinstance(pearson_entry, Mapping) and not isinstance(mae_entry, Mapping):
                continue
            rows.append(
                [
                    C_DIM_LABELS.get(dimension, dimension),
                    _fmt(pearson_entry.get("estimate") if isinstance(pearson_entry, Mapping) else None),
                    _fmt_ci(pearson_entry if isinstance(pearson_entry, Mapping) else None),
                    _fmt(mae_entry.get("estimate") if isinstance(mae_entry, Mapping) else None),
                    _fmt_ci(mae_entry if isinstance(mae_entry, Mapping) else None),
                ]
            )
        lines.extend(_md_table(("维度", "ΔPearson", "95% CI", "ΔMAE", "95% CI"), rows))
        lines.append("")
        non_native_n = _entry(payload.get("c"), "mean_dimension").get("n")
        native_n = _int(_entry(payload.get("c"), "native_anchor").get("n"))
        paired = _int(block_c.get("n_items_paired"))
        learner_paired = paired - native_n if native_n and paired else None
        if learner_paired is not None:
            lines.append(
                f"- 配对 item 数：{_fmt_int(paired)}"
                f"（learner {_fmt_int(learner_paired)} + native {native_n}）；"
                f"Δ 统计已剔除 native，learner 有效配对 n={_fmt_int(learner_paired)}"
            )
        elif non_native_n is not None:
            lines.append(
                f"- 配对 item 数：{_fmt_int(paired)}"
                f"（Δ 统计不含 native，C 有效配对 n={_fmt_int(non_native_n)}）"
            )
        else:
            lines.append(f"- 配对 item 数：{_fmt_int(block_c.get('n_items_paired'))}")
    else:
        lines.append("_任务 C 的 C1/C2 配对比较不可用。_")
    lines.append("")

    lines.extend(["### 5.4 结论方向", ""])
    bullets: list[str] = []
    if isinstance(block_a, Mapping):
        entry = _entry(_entry(block_a, "differences"), "joint_accuracy")
        bullets.append(f"- 任务 A 字级联合准确率：{_direction_phrase(entry)}。")
    if isinstance(block_b, Mapping):
        for dimension in B_DIM_ORDER:
            entry = _entry(_entry(block_b, "differences"), "macro_f1").get(dimension)
            bullets.append(
                f"- 任务 B {B_DIM_LABELS.get(dimension, dimension)} macro-F1："
                f"{_direction_phrase(entry)}。"
            )
    if isinstance(block_c, Mapping):
        for dimension in C_DIM_ORDER:
            entry = _entry(_entry(block_c, "differences"), "pearson").get(dimension)
            bullets.append(
                f"- 任务 C {C_DIM_LABELS.get(dimension, dimension)} Pearson r："
                f"{_direction_phrase(entry)}。"
            )
    if bullets:
        lines.extend(bullets)
    else:
        lines.append("_无可用结论方向。_")
    lines.append("")
    return lines


def _md_single_syllable(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 6. 与单音节结果的聚合对照（design §7.4）", ""]
    block = _entry(payload, "single_syllable")
    if not block.get("available"):
        lines.append(
            "_不可用：未找到单音节汇总/逐条结果 "
            f"`{block.get('source') or '—'}`。该对照需要已有的单音节实验 "
            "`summary.json`（无加权）与 `items.parquet`（类别加权）。_"
        )
        lines.append("")
        return lines
    single = _entry(block, "single")
    sentence_all = _entry(block, "sentence_all")
    sentence_stable = _entry(block, "sentence_stable")
    deltas = _entry(block, "deltas")
    lines.append(
        "§6.1 为不加权聚合对照；§6.2 按单音节 exp3 free 的类别分布加权"
        "（design §7.4）；§6.3 为逐类别明细（完整表见 "
        "`single_syllable_categories.csv`）。"
    )
    lines.append("")

    lines.extend(["### 6.1 不加权聚合对照", ""])
    rows = []
    for dimension, label in SINGLE_SYLLABLE_DIMS:
        rows.append(
            [
                label,
                _fmt(single.get(dimension)),
                _fmt(sentence_all.get(dimension)),
                _fmt(sentence_stable.get(dimension)),
                _fmt(deltas.get(dimension)),
            ]
        )
    lines.extend(
        _md_table(
            (
                "维度",
                "单音节 exp3 free",
                "句内 A 全部",
                "句内 A stable",
                "差值（stable − 单音节）",
            ),
            rows,
        )
    )
    lines.append("")
    lines.append(
        f"- 单音节来源：`{block.get('source')}`（run `{block.get('source_run_id')}`，"
        f"n={_fmt_int(single.get('n'))}）；"
        f"句内 stable 子集 n_items={_fmt_int(sentence_stable.get('n_items'))}"
        f"（parse-ok={_fmt_int(sentence_stable.get('n_parse_ok'))}）。"
    )
    reference_filter = block.get("reference_filter")
    if reference_filter:
        lines.append(f"- 单音节参考过滤条件：`{reference_filter}`。")
    lines.append(
        "- 差值 = 句内 stable − 单音节；负值表示句内聚合准确率低于单音节结果。"
    )
    lines.append("")

    lines.extend(["### 6.2 类别加权对照（design §7.4）", ""])
    if block.get("weighted_available"):
        weighted = _entry(block, "weighted")
        weighted_rows = []
        for dimension, _dimension_column, _gold, _correct in CATEGORY_DIMENSIONS:
            entry = _entry(weighted, dimension)
            weighted_rows.append(
                [
                    CATEGORY_DIMENSION_LABELS.get(dimension, dimension),
                    _fmt(entry.get("reference")),
                    _fmt(entry.get("sentence")),
                    _fmt(entry.get("delta")),
                    _fmt_pct(entry.get("coverage")),
                    f"{_fmt_int(entry.get('n_categories_covered'))}"
                    f" / {_fmt_int(entry.get('n_categories_single'))}",
                ]
            )
        lines.extend(
            _md_table(
                (
                    "维度",
                    "单音节加权准确率",
                    "句内 stable 加权准确率",
                    "差值（句内−单音节）",
                    "权重覆盖率",
                    "覆盖类别数",
                ),
                weighted_rows,
            )
        )
        lines.append("")
        lines.append(
            "- 加权方式：`Σ_c w_c · acc(c)`，`w_c` = 单音节 exp3 free 的类别占比；"
            "只在两侧都出现的类别上加权，并在覆盖类别上重新归一化"
            "（覆盖率 = 覆盖类别权重 / 单音节全部类别权重）。"
        )
    else:
        lines.append(
            "_不可用："
            f"{block.get('weighted_reason') or '缺少单音节逐条结果或 cat_*_stats 计数'}。"
            "见 §9.1 警告。_"
        )
    lines.append("")

    lines.extend(["### 6.3 逐类别明细（每维度前 10 类）", ""])
    category_rows = block.get("category_rows")
    if block.get("weighted_available") and isinstance(category_rows, list):
        for dimension, _dimension_column, _gold, _correct in CATEGORY_DIMENSIONS:
            dimension_rows = [
                row
                for row in category_rows
                if isinstance(row, Mapping) and row.get("dimension") == dimension
            ]
            if not dimension_rows:
                continue
            lines.append(
                f"**{CATEGORY_DIMENSION_LABELS.get(dimension, dimension)}**"
                f"（共 {len(dimension_rows)} 类，按单音节样本量排序）"
            )
            lines.append("")
            detail_rows = []
            for row in dimension_rows[:10]:
                detail_rows.append(
                    [
                        _category_display(dimension, row.get("category")),
                        _fmt_int(row.get("single_n")),
                        _fmt(row.get("single_acc")),
                        _fmt(row.get("single_weight")),
                        _fmt_int(row.get("sentence_n")),
                        _fmt(row.get("sentence_acc")),
                        "是" if row.get("covered") else "否",
                    ]
                )
            lines.extend(
                _md_table(
                    (
                        "类别",
                        "单音节 n",
                        "单音节 acc",
                        "权重",
                        "句内 n",
                        "句内 acc",
                        "覆盖",
                    ),
                    detail_rows,
                )
            )
            lines.append("")
        lines.append(
            f"- 完整逐类别表（全部 {len(category_rows)} 行）见 "
            "`single_syllable_categories.csv`。"
        )
    else:
        lines.append("_逐类别明细不可用（见 §6.2 与 §9.1）。_")
    lines.append("")

    caveats = block.get("caveats") or []
    if caveats:
        lines.extend(["### 6.4 解释边界", ""])
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    return lines


def _md_usage(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 7. 成本 / 用量 / 解析失败 / 重试", ""]
    usage = _entry(payload, "usage")
    totals = _entry(usage, "totals")
    parse_failures = _entry(payload, "parse_failures")
    overall = _entry(parse_failures, "overall")
    manifest = _entry(payload, "manifest")
    budget = manifest.get("budget_cny")
    rows = [
        ("评分调用数", _fmt_int(totals.get("n_scored_calls"))),
        ("已记录调用数（含重试前的错误记录）", _fmt_int(totals.get("n_calls"))),
        ("解析成功数", _fmt_int(overall.get("n_ok"))),
        ("解析失败数", _fmt_int(overall.get("n_failures"))),
        ("解析成功率", _fmt_pct(overall.get("rate"))),
        ("重试次数", _fmt_int(totals.get("n_retries"))),
        ("prompt tokens", _fmt_int(totals.get("prompt_tokens"))),
        ("completion tokens", _fmt_int(totals.get("completion_tokens"))),
        ("reasoning tokens", _fmt_int(totals.get("reasoning_tokens"))),
        ("平均延迟 (s)", _fmt(totals.get("avg_latency_s"), 3)),
        ("实际成本", _fmt_cny(totals.get("cost_cny"))),
        ("预算", _fmt_cny(budget)),
        ("成本来源", str(usage.get("source") or "—")),
        ("解析统计来源", str(parse_failures.get("source") or "—")),
    ]
    lines.extend(["### 7.1 总览", ""])
    lines.extend(_md_table(("项", "值"), rows))
    lines.append("")

    lines.extend(["### 7.2 按任务 × 条件", ""])
    by_condition = _entry(usage, "by_task_condition")
    condition_rows = []
    for key in sorted(by_condition):
        entry = _entry(by_condition, key)
        condition_rows.append(
            [
                f"`{key}`",
                _fmt_int(entry.get("n_calls")),
                _fmt_int(entry.get("n_ok")),
                _fmt_int(entry.get("n_parse_failures")),
                _fmt_int(entry.get("n_errors")),
                _fmt_int(entry.get("n_retries")),
                _fmt_int(entry.get("prompt_tokens")),
                _fmt_int(entry.get("completion_tokens")),
                _fmt_int(entry.get("reasoning_tokens")),
                _fmt(entry.get("avg_latency_s"), 3),
                _fmt(entry.get("cost_cny"), 4),
            ]
        )
    if condition_rows:
        lines.extend(
            _md_table(
                (
                    "任务 / 条件",
                    "n_calls",
                    "n_ok",
                    "解析失败",
                    "n_errors",
                    "重试",
                    "prompt",
                    "completion",
                    "reasoning",
                    "平均延迟 (s)",
                    "成本 (CNY)",
                ),
                condition_rows,
            )
        )
    else:
        lines.append("_没有任务 × 条件的调用记录。_")
    lines.append("")
    lines.append(
        "- 口径：`n_calls` 为 `generations.jsonl` 的记录数，包含重试前已失败的记录；"
        "`n_errors` 为最终未恢复的错误数；因此个别行会出现 "
        "`n_calls > n_ok + 解析失败 + n_errors`（差额为已被续跑恢复的尝试）。"
    )
    lines.append("")

    lines.extend(["### 7.3 解析失败分布（按状态）", ""])
    by_task_condition = _entry(parse_failures, "by_task_condition")
    failure_rows = []
    for key in sorted(by_task_condition):
        counts = _entry(by_task_condition, key)
        for status in sorted(counts):
            if status == "ok":
                continue
            failure_rows.append([f"`{key}`", str(status), _fmt_int(counts.get(status))])
    if failure_rows:
        lines.extend(_md_table(("任务 / 条件", "状态", "计数"), failure_rows))
        lines.append("")
        lines.append(
            "- 解析失败/空回复/API 错误都不进入指标分母；原始记录见 `parse_failures.jsonl`。"
        )
    else:
        lines.append("_没有解析失败记录。_")
    lines.append("")

    lines.extend(["### 7.4 finish_reason 分布", ""])
    finish_rows = []
    for key in sorted(by_condition):
        entry = _entry(by_condition, key)
        finish_reasons = _entry(entry, "finish_reasons")
        for reason in sorted(finish_reasons):
            finish_rows.append([f"`{key}`", str(reason), _fmt_int(finish_reasons.get(reason))])
    if finish_rows:
        lines.extend(_md_table(("任务 / 条件", "finish_reason", "计数"), finish_rows))
    else:
        lines.append("_没有可用的 finish_reason 记录。_")
    lines.append("")
    return lines


def _pass_text(value: Any) -> str:
    if value is True:
        return "✅ 通过"
    if value is False:
        return "❌ 未通过"
    return "—"


def _md_acceptance(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 8. 验收检查（design §10.2）", ""]
    acceptance = _entry(payload, "acceptance")
    if not acceptance:
        lines.append("_验收检查不可用。_")
        lines.append("")
        return lines
    rows = []
    parse_success = _entry(acceptance, "parse_success")
    for task in TASKS:
        entry = _entry(parse_success, task)
        rows.append(
            [
                f"解析成功率：任务 {task}",
                f"≥ {_fmt_pct(entry.get('threshold'))}",
                f"{_fmt_pct(entry.get('rate'))}"
                f"（{_fmt_int(entry.get('n_calls'))} 次调用）",
                _pass_text(entry.get("passed")),
            ]
        )
    alignment = _entry(acceptance, "alignment_coverage")
    for task in TASKS:
        entry = _entry(alignment, task)
        rows.append(
            [
                f"对齐覆盖率：任务 {task}",
                f"≥ {_fmt_pct(entry.get('threshold'))}",
                _fmt(entry.get("value")),
                _pass_text(entry.get("passed")),
            ]
        )
    cost = _entry(acceptance, "cost_cny")
    rows.append(
        [
            "总成本",
            f"≤ {_fmt_cny(cost.get('threshold'))}",
            _fmt_cny(cost.get("value")),
            _pass_text(cost.get("passed")),
        ]
    )
    lines.extend(_md_table(("检查项", "阈值", "实际值", "结果"), rows))
    lines.append("")
    lines.append(f"- **验收总结果：{_pass_text(acceptance.get('passed'))}**")
    lines.append("- 设计 §10.2 不设准确率硬阈值：pilot 的结论是“能否稳定测、效应方向/量级、下一步样本量”。")
    lines.append("")
    return lines


def _md_limitations(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 9. 局限与解释边界", ""]
    lines.append("以下条目来自设计 §11，按本次结果相关度排列；数字均来自本报告的指标块。")
    lines.append("")
    for item in DESIGN_LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")
    a = payload.get("a")
    if isinstance(a, Mapping):
        erhua = _entry(_entry(a, "subsets"), "erhua")
        if erhua:
            lines.append(
                f"- 本次运行 erhua 子集 n_items={_fmt_int(erhua.get('n_items'))}："
                "儿化样本少时字级准确率波动大，需结合 N1/人工抽查解读。"
            )
    c = payload.get("c")
    if isinstance(c, Mapping):
        native = _entry(c, "native_anchor")
        lines.append(
            f"- 本次运行 native 锚点 n={_fmt_int(native.get('n'))}：native 全为 5 分，"
            "只证明“人类满分上限”，不提供评分区分度信息。"
        )
    b = payload.get("b")
    if isinstance(b, Mapping):
        lines.append(
            "- 任务 B 的零声母 consonant 口径单独报告（§3.3），不参与主结论；"
            "若该子集标签口径有误，只影响 §3.3。"
        )
    warnings = payload.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("### 9.1 本次运行的降级与警告")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
    lines.append("")
    return lines


def _has_artifact(payload: Mapping[str, Any], name: str) -> bool:
    artifacts = payload.get("artifacts")
    return isinstance(artifacts, Mapping) and name in artifacts


def _md_artifacts(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 附录：工件清单", ""]
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        lines.append("_工件清单不可用。_")
        lines.append("")
        return lines
    lines.append("| 文件 | 路径 |")
    lines.append("| --- | --- |")
    for name in sorted(artifacts, key=str):
        lines.append(f"| `{name}` | `{artifacts[name]}` |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _metric_row(
    task: str,
    condition: str,
    scope: str,
    subset: str,
    dimension: str,
    metric: str,
    estimate: Any,
    ci_lo: Any = None,
    ci_hi: Any = None,
    n: Any = None,
) -> dict[str, Any]:
    """One long-format metrics.csv row."""
    return {
        "task": task,
        "condition": condition,
        "scope": scope,
        "subset": subset,
        "dimension": dimension,
        "metric": metric,
        "estimate": _num(estimate),
        "ci_lo": _num(ci_lo),
        "ci_hi": _num(ci_hi),
        "n": _int(n) if not _is_missing(n) else None,
    }


def _metrics_csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the long-format metric table from the payload (design §9.3)."""
    rows: list[dict[str, Any]] = []

    for condition, key in (("C1", "a"), ("C2", "a_c2")):
        block = payload.get(key)
        if not isinstance(block, Mapping):
            continue
        headline = _entry(block, "headline")
        with_ci = _entry(block, "with_ci")
        for metric in A_METRIC_ORDER:
            ci = _entry(with_ci, metric)
            rows.append(
                _metric_row(
                    "A",
                    condition,
                    "headline",
                    "",
                    A_DIMENSION_OF.get(metric, ""),
                    metric,
                    headline.get(metric),
                    ci.get("lo"),
                    ci.get("hi"),
                    block.get("n_parse_ok"),
                )
            )
        subsets = _entry(block, "subsets")
        for subset in A_SUBSET_ORDER:
            subset_block = _entry(subsets, subset)
            if not subset_block:
                continue
            for metric in A_METRIC_ORDER:
                rows.append(
                    _metric_row(
                        "A",
                        condition,
                        "subset",
                        subset,
                        A_DIMENSION_OF.get(metric, ""),
                        metric,
                        subset_block.get(metric),
                        None,
                        None,
                        subset_block.get("n_parse_ok"),
                    )
                )
        n1 = _entry(block, "n1")
        for metric in n1:
            if metric == "delta_vs_n0" or not isinstance(n1.get(metric), (int, float)):
                continue
            rows.append(
                _metric_row(
                    "A",
                    condition,
                    "n1_diagnostic",
                    "",
                    "",
                    metric,
                    n1.get(metric),
                    None,
                    None,
                    block.get("n_parse_ok"),
                )
            )

    for condition, key in (("C1", "b"), ("C2", "b_c2")):
        block = payload.get(key)
        if not isinstance(block, Mapping):
            continue
        rows.append(
            _metric_row(
                "B",
                condition,
                "overall",
                "",
                "",
                "alignment_coverage",
                block.get("alignment_coverage"),
                None,
                None,
                block.get("n_parse_ok"),
            )
        )
        with_ci = _entry(block, "with_ci")
        for dimension in B_DIM_ORDER:
            stats = _entry(block, dimension)
            if not stats:
                continue
            macro_ci = _entry(_entry(with_ci, dimension), "macro_f1")
            for metric in (
                "macro_f1",
                "f1_error",
                "f1_ok",
                "fpr",
                "fnr",
                "accuracy",
            ):
                rows.append(
                    _metric_row(
                        "B",
                        condition,
                        "dimension",
                        "",
                        dimension,
                        metric,
                        stats.get(metric),
                        macro_ci.get("lo") if metric == "macro_f1" else None,
                        macro_ci.get("hi") if metric == "macro_f1" else None,
                        stats.get("n_units"),
                    )
                )
            confusion = _entry(stats, "confusion")
            for count_name in ("tp", "fp", "fn", "tn"):
                rows.append(
                    _metric_row(
                        "B",
                        condition,
                        "confusion",
                        "",
                        dimension,
                        count_name,
                        confusion.get(count_name),
                        None,
                        None,
                        stats.get("n_units"),
                    )
                )
        zero = _entry(block, "zero_initial_consonant")
        for metric in ("macro_f1", "f1_error", "fpr", "fnr", "accuracy"):
            rows.append(
                _metric_row(
                    "B",
                    condition,
                    "zero_initial_consonant",
                    "",
                    "consonant",
                    metric,
                    zero.get(metric),
                    None,
                    None,
                    zero.get("n_units"),
                )
            )

    for condition, key in (("C1", "c"), ("C2", "c_c2")):
        block = payload.get(key)
        if not isinstance(block, Mapping):
            continue
        rows.append(
            _metric_row(
                "C",
                condition,
                "overall",
                "",
                "",
                "alignment_coverage",
                block.get("alignment_coverage"),
                None,
                None,
                block.get("n_parse_ok"),
            )
        )
        with_ci = _entry(block, "with_ci")
        for dimension in C_DIM_ORDER:
            stats = _entry(block, dimension)
            if not stats:
                continue
            ci = _entry(with_ci, dimension)
            for metric in ("pearson", "spearman", "mae", "rmse"):
                ci_entry = _entry(ci, metric) if metric in ("pearson", "mae") else {}
                rows.append(
                    _metric_row(
                        "C",
                        condition,
                        "dimension",
                        "",
                        dimension,
                        metric,
                        stats.get(metric),
                        ci_entry.get("lo"),
                        ci_entry.get("hi"),
                        stats.get("n"),
                    )
                )
        native = _entry(block, "native_anchor")
        rows.append(
            _metric_row(
                "C",
                condition,
                "native_anchor",
                "",
                "",
                "n",
                native.get("n"),
                None,
                None,
                native.get("n"),
            )
        )
        rows.append(
            _metric_row(
                "C",
                condition,
                "native_anchor",
                "",
                "",
                "mean_gold",
                native.get("mean_gold"),
                None,
                None,
                native.get("n"),
            )
        )
    return rows


def write_metrics_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``metrics.csv`` (long format, frozen column order)."""
    return _write_csv(Path(path), _metrics_csv_rows(payload), METRICS_COLUMNS)


def _comparison_row(
    task: str,
    metric: str,
    dimension: str,
    estimate: Any,
    ci_lo: Any,
    ci_hi: Any,
    p_value: Any = None,
    p_holm: Any = None,
    n_pairs: Any = None,
    direction: str = "C1 - C2",
    method: str = "paired_cluster_bootstrap",
) -> dict[str, Any]:
    """One comparisons.csv row."""
    return {
        "task": task,
        "metric": metric,
        "dimension": dimension,
        "estimate": _num(estimate),
        "ci_lo": _num(ci_lo),
        "ci_hi": _num(ci_hi),
        "p_value": _num(p_value),
        "p_holm": _num(p_holm),
        "n_pairs": _int(n_pairs) if not _is_missing(n_pairs) else None,
        "direction": direction,
        "method": method,
    }


def _comparisons_csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the paired C1-vs-C2 comparison table (design §9.2)."""
    rows: list[dict[str, Any]] = []
    compare = _entry(payload, "compare")
    for task in TASKS:
        block = compare.get(task)
        if not isinstance(block, Mapping):
            continue
        direction = str(block.get("direction") or "C1 - C2")
        method = str(block.get("method") or "paired_cluster_bootstrap")
        n_pairs = block.get("n_items_paired")
        differences = _entry(block, "differences")
        if task == "A":
            for metric in A_METRIC_ORDER:
                entry = differences.get(metric)
                if not isinstance(entry, Mapping):
                    continue
                rows.append(
                    _comparison_row(
                        "A",
                        "difference",
                        A_DIMENSION_OF.get(metric, ""),
                        entry.get("estimate"),
                        entry.get("lo"),
                        entry.get("hi"),
                        entry.get("p_value"),
                        None,
                        n_pairs,
                        direction,
                        method,
                    )
                )
            mcnemar = _entry(_entry(block, "mcnemar"), "sentence_exact")
            if mcnemar:
                rows.append(
                    _comparison_row(
                        "A",
                        "mcnemar",
                        "sentence_exact",
                        mcnemar.get("statistic"),
                        None,
                        None,
                        mcnemar.get("p_value"),
                        _entry(block, "holm").get("sentence_exact"),
                        mcnemar.get("n_pairs"),
                        direction,
                        str(mcnemar.get("method") or "exact_binomial"),
                    )
                )
        elif task == "B":
            macro = _entry(differences, "macro_f1")
            for dimension in B_DIM_ORDER:
                entry = macro.get(dimension)
                if not isinstance(entry, Mapping):
                    continue
                rows.append(
                    _comparison_row(
                        "B",
                        "macro_f1",
                        dimension,
                        entry.get("estimate"),
                        entry.get("lo"),
                        entry.get("hi"),
                        entry.get("p_value"),
                        None,
                        n_pairs,
                        direction,
                        method,
                    )
                )
        else:
            for metric in ("pearson", "mae"):
                per_dimension = _entry(differences, metric)
                for dimension in C_DIM_ORDER:
                    entry = per_dimension.get(dimension)
                    if not isinstance(entry, Mapping):
                        continue
                    rows.append(
                        _comparison_row(
                            "C",
                            metric,
                            dimension,
                            entry.get("estimate"),
                            entry.get("lo"),
                            entry.get("hi"),
                            entry.get("p_value"),
                            None,
                            n_pairs,
                            direction,
                            method,
                        )
                    )
    return rows


def write_comparisons_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``comparisons.csv`` (paired C1-vs-C2 differences, frozen columns)."""
    return _write_csv(Path(path), _comparisons_csv_rows(payload), COMPARISON_COLUMNS)


def write_per_speaker_a_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``per_speaker_a.csv`` (task-A descriptive statistics per speaker)."""
    rows: list[dict[str, Any]] = []
    a = payload.get("a")
    per_speaker = a.get("per_speaker") if isinstance(a, Mapping) else None
    if isinstance(per_speaker, list):
        for entry in per_speaker:
            if isinstance(entry, Mapping):
                rows.append({column: entry.get(column) for column in PER_SPEAKER_A_COLUMNS})
    return _write_csv(Path(path), rows, PER_SPEAKER_A_COLUMNS)


def write_per_learner_b_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``per_learner_b.csv`` (task-B macro-F1 / FPR / FNR per learner)."""
    rows: list[dict[str, Any]] = []
    b = payload.get("b")
    per_learner = b.get("per_learner") if isinstance(b, Mapping) else None
    if isinstance(per_learner, list):
        for entry in per_learner:
            if not isinstance(entry, Mapping):
                continue
            row: dict[str, Any] = {
                "learner": entry.get("learner"),
                "n_items": entry.get("n_items"),
            }
            for prefix in ("macro_f1", "f1_error", "fpr", "fnr"):
                values = _entry(entry, prefix)
                for dimension in B_DIM_ORDER:
                    row[f"{prefix}_{dimension}"] = values.get(dimension)
            rows.append(row)
    return _write_csv(Path(path), rows, PER_LEARNER_B_COLUMNS)


def write_per_learner_c_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``per_learner_c.csv`` (task-C correlations / errors per learner)."""
    rows: list[dict[str, Any]] = []
    c = payload.get("c")
    per_learner = c.get("per_learner") if isinstance(c, Mapping) else None
    if isinstance(per_learner, list):
        for entry in per_learner:
            if not isinstance(entry, Mapping):
                continue
            row: dict[str, Any] = {
                "learner": entry.get("learner"),
                "n_items": entry.get("n_items"),
            }
            pearson = _entry(entry, "pearson")
            spearman = _entry(entry, "spearman")
            mae = _entry(entry, "mae")
            rmse = _entry(entry, "rmse")
            for dimension in C_DIM_ORDER:
                row[f"pearson_{dimension}"] = pearson.get(dimension)
            row["spearman_mean_dimension"] = spearman.get("mean_dimension")
            for dimension in C_DIM_ORDER:
                row[f"mae_{dimension}"] = mae.get(dimension)
            row["rmse_mean_dimension"] = rmse.get("mean_dimension")
            rows.append(row)
    return _write_csv(Path(path), rows, PER_LEARNER_C_COLUMNS)


def write_single_syllable_categories_csv(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``single_syllable_categories.csv`` (design §7.4 per-category table)."""
    rows: list[dict[str, Any]] = []
    single_syllable = _entry(payload, "single_syllable")
    category_rows = single_syllable.get("category_rows")
    if isinstance(category_rows, list):
        for entry in category_rows:
            if not isinstance(entry, Mapping):
                continue
            rows.append(
                {column: entry.get(column) for column in SINGLE_SYLLABLE_CATEGORY_COLUMNS}
            )
    return _write_csv(Path(path), rows, SINGLE_SYLLABLE_CATEGORY_COLUMNS)


# ---------------------------------------------------------------------------
# Figures: hand-rolled, self-contained SVG (no matplotlib / scipy)
# ---------------------------------------------------------------------------

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG_C1_COLOR = "#4e79a7"
SVG_C2_COLOR = "#f28e2b"
SVG_AXIS_COLOR = "#444444"
SVG_GRID_COLOR = "#ececec"
SVG_TEXT_COLOR = "#222222"
SVG_MUTED_COLOR = "#666666"
FONT_FAMILY = "Helvetica, Arial, sans-serif"


def _svg_text(
    x: float,
    y: float,
    text: Any,
    *,
    size: int = 12,
    anchor: str = "start",
    fill: str = SVG_TEXT_COLOR,
    weight: str = "normal",
) -> str:
    """One SVG ``<text>`` element (content XML-escaped)."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT_FAMILY}" '
        f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
        f'font-weight="{weight}">{escape(str(text))}</text>'
    )


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = SVG_AXIS_COLOR,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    """One SVG ``<line>`` element."""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def _svg_circle(
    cx: float, cy: float, radius: float, fill: str, opacity: float = 0.55
) -> str:
    """One SVG ``<circle>`` element."""
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'fill-opacity="{opacity:.2f}" stroke="none"/>'
    )


def _svg_rect(
    x: float, y: float, width: float, height: float, fill: str, opacity: float = 0.85
) -> str:
    """One SVG ``<rect>`` element."""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" fill-opacity="{opacity:.2f}" stroke="none"/>'
    )


def _svg_document(width: int, height: int, title: str, body: Sequence[str]) -> str:
    """Wrap body elements into a standalone SVG 1.1 document."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="{SVG_NAMESPACE}" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def _svg_c_scatter(payload: Mapping[str, Any]) -> str | None:
    """Task-C predicted-vs-gold small multiples (one panel per dimension)."""
    scatter = payload.get("c_scatter")
    if not isinstance(scatter, Mapping):
        return None
    panels: list[tuple[str, list[Any]]] = []
    for dimension in ("accuracy", "fluency", "prosody"):
        points = scatter.get(dimension)
        if isinstance(points, list) and points:
            panels.append((dimension, points))
    if not panels:
        return None

    c_metrics = payload.get("c")
    panel_w, panel_h = 300, 280
    margin_left, margin_top = 50, 100
    gap = 36
    width = margin_left * 2 + len(panels) * panel_w + (len(panels) - 1) * gap
    height = margin_top + panel_h + 76
    body: list[str] = [
        _svg_text(width / 2, 36, "任务 C：预测分 vs 人工分（非 native，C1）", size=18,
                  anchor="middle", weight="bold"),
        _svg_text(
            width / 2,
            60,
            "每个点 = 一个评分项；虚线为 y=x；面板标题给出 Pearson r 与 n",
            size=12,
            anchor="middle",
            fill=SVG_MUTED_COLOR,
        ),
    ]
    for index, (dimension, points) in enumerate(panels):
        panel_x = margin_left + index * (panel_w + gap)
        panel_y = margin_top
        plot_left = panel_x + 46
        plot_top = panel_y + 40
        plot_w = panel_w - 46 - 18
        plot_h = panel_h - 40 - 34
        body.append(
            f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" width="{plot_w:.1f}" '
            f'height="{plot_h:.1f}" fill="#fbfbfb" stroke="#cccccc" stroke-width="1"/>'
        )
        for tick in (1, 2, 3, 4, 5):
            tick_x = plot_left + (tick - 1) / 4.0 * plot_w
            tick_y = plot_top + (1 - (tick - 1) / 4.0) * plot_h
            body.append(_svg_line(tick_x, plot_top, tick_x, plot_top + plot_h, stroke=SVG_GRID_COLOR))
            body.append(_svg_line(plot_left, tick_y, plot_left + plot_w, tick_y, stroke=SVG_GRID_COLOR))
            body.append(
                _svg_text(tick_x, plot_top + plot_h + 16, tick, size=11, anchor="middle",
                          fill=SVG_MUTED_COLOR)
            )
            body.append(
                _svg_text(plot_left - 8, tick_y + 4, tick, size=11, anchor="end",
                          fill=SVG_MUTED_COLOR)
            )
        body.append(
            _svg_line(
                plot_left,
                plot_top + plot_h,
                plot_left + plot_w,
                plot_top,
                stroke="#999999",
                width=1.2,
                dash="5,4",
            )
        )
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            gold = _num(point[0])
            pred = _num(point[1])
            if gold is None or pred is None:
                continue
            gold = min(max(gold, 1.0), 5.0)
            pred = min(max(pred, 1.0), 5.0)
            point_x = plot_left + (gold - 1.0) / 4.0 * plot_w
            point_y = plot_top + (1.0 - (pred - 1.0) / 4.0) * plot_h
            body.append(_svg_circle(point_x, point_y, 3.2, SVG_C1_COLOR, opacity=0.5))
        stats = _entry(c_metrics, dimension) if isinstance(c_metrics, Mapping) else {}
        body.append(
            _svg_text(
                plot_left + plot_w / 2,
                panel_y + 22,
                f"{C_DIM_LABELS.get(dimension, dimension)}  r={_fmt(stats.get('pearson'), 3)}"
                f"  n={len(points)}",
                size=13,
                anchor="middle",
                weight="bold",
            )
        )
    body.append(
        _svg_text(width / 2, height - 20, "人工分（gold）→", size=12, anchor="middle",
                  fill=SVG_MUTED_COLOR)
    )
    body.append(
        _svg_text(margin_left - 6, margin_top + 18, "预测分 ↑", size=11, anchor="middle",
                  fill=SVG_MUTED_COLOR)
    )
    return _svg_document(width, height, "任务 C：预测分 vs 人工分散点图", body)


def _svg_a_accuracy_bar(payload: Mapping[str, Any]) -> str | None:
    """Task-A headline accuracy bars (C1, C2 when available) with 95 % CI."""
    a = payload.get("a")
    if not isinstance(a, Mapping):
        return None
    series: list[tuple[str, Mapping[str, Any], str]] = [("C1", a, SVG_C1_COLOR)]
    a_c2 = payload.get("a_c2")
    if isinstance(a_c2, Mapping):
        series.append(("C2", a_c2, SVG_C2_COLOR))
    categories = (
        ("joint", "联合"),
        ("initial", "声母"),
        ("final", "韵母"),
        ("tone", "声调"),
    )
    values: dict[tuple[int, str], tuple[float | None, float | None, float | None]] = {}
    any_value = False
    for series_index, (_name, block, _color) in enumerate(series):
        headline = _entry(block, "headline")
        with_ci = _entry(block, "with_ci")
        for category, _label in categories:
            value = _num(headline.get(f"{category}_accuracy"))
            if value is not None:
                any_value = True
            ci = _entry(with_ci, f"{category}_accuracy")
            values[(series_index, category)] = (value, _num(ci.get("lo")), _num(ci.get("hi")))
    if not any_value:
        return None

    width, height = 760, 400
    margin_left, margin_top, margin_right, margin_bottom = 78, 86, 30, 74
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    body: list[str] = [
        _svg_text(width / 2, 36, "任务 A：字级准确率（C1 vs C2，95% CI）", size=18,
                  anchor="middle", weight="bold"),
        _svg_text(
            width / 2,
            58,
            f"C1 n={_fmt_int(a.get('n_parse_ok'))}；误差线为按 speaker 聚类的 bootstrap 95% CI",
            size=12,
            anchor="middle",
            fill=SVG_MUTED_COLOR,
        ),
    ]
    for tick in range(6):
        value = tick * 0.2
        grid_y = margin_top + plot_h * (1.0 - value)
        body.append(_svg_line(margin_left, grid_y, margin_left + plot_w, grid_y, stroke=SVG_GRID_COLOR))
        body.append(
            _svg_text(margin_left - 8, grid_y + 4, f"{value:.1f}", size=11, anchor="end",
                      fill=SVG_MUTED_COLOR)
        )
    group_w = plot_w / len(categories)
    bar_w = min(52.0, group_w / (len(series) + 1))
    for group_index, (category, label) in enumerate(categories):
        center = margin_left + group_w * (group_index + 0.5)
        offset = (len(series) * bar_w + (len(series) - 1) * 6) / 2.0
        for series_index, (_name, _block, color) in enumerate(series):
            value, lo, hi = values[(series_index, category)]
            bar_x = center - offset + series_index * (bar_w + 6)
            if value is not None:
                bar_h = plot_h * value
                bar_y = margin_top + plot_h - bar_h
                body.append(_svg_rect(bar_x, bar_y, bar_w, bar_h, color))
                body.append(
                    _svg_text(bar_x + bar_w / 2, bar_y - 6, f"{value:.4f}", size=10,
                              anchor="middle")
                )
                if lo is not None and hi is not None:
                    bar_center = bar_x + bar_w / 2
                    y_lo = margin_top + plot_h * (1.0 - lo)
                    y_hi = margin_top + plot_h * (1.0 - hi)
                    body.append(_svg_line(bar_center, y_hi, bar_center, y_lo, width=1.2,
                                          stroke="#333333"))
                    body.append(_svg_line(bar_center - 4, y_hi, bar_center + 4, y_hi,
                                          width=1.2, stroke="#333333"))
                    body.append(_svg_line(bar_center - 4, y_lo, bar_center + 4, y_lo,
                                          width=1.2, stroke="#333333"))
        body.append(
            _svg_text(center, margin_top + plot_h + 24, label, size=12, anchor="middle")
        )
    legend_x = margin_left
    legend_y = height - 20
    for series_index, (name, _block, color) in enumerate(series):
        legend_offset = series_index * 80
        body.append(_svg_rect(legend_x + legend_offset, legend_y - 10, 12, 12, color, opacity=1.0))
        body.append(
            _svg_text(legend_x + legend_offset + 18, legend_y, name, size=12,
                      fill=SVG_MUTED_COLOR)
        )
    return _svg_document(width, height, "任务 A：字级准确率柱状图", body)


def _svg_c1_vs_c2(payload: Mapping[str, Any]) -> str | None:
    """Paired ``C1 - C2`` effect sizes with 95 % CI across all three tasks."""
    compare = payload.get("compare")
    if not isinstance(compare, Mapping):
        return None
    entries: list[tuple[str, Mapping[str, Any]]] = []
    block_a = compare.get("A")
    if isinstance(block_a, Mapping):
        differences = _entry(block_a, "differences")
        for metric, label in (
            ("joint_accuracy", "A 联合准确率"),
            ("initial_accuracy", "A 声母准确率"),
            ("final_accuracy", "A 韵母准确率"),
            ("tone_accuracy", "A 声调准确率"),
        ):
            entry = differences.get(metric)
            if isinstance(entry, Mapping) and _num(entry.get("estimate")) is not None:
                entries.append((label, entry))
    block_b = compare.get("B")
    if isinstance(block_b, Mapping):
        macro = _entry(_entry(block_b, "differences"), "macro_f1")
        for dimension in B_DIM_ORDER:
            entry = macro.get(dimension)
            if isinstance(entry, Mapping) and _num(entry.get("estimate")) is not None:
                entries.append(
                    (f"B {B_DIM_LABELS.get(dimension, dimension)} macro-F1", entry)
                )
    block_c = compare.get("C")
    if isinstance(block_c, Mapping):
        pearson = _entry(_entry(block_c, "differences"), "pearson")
        for dimension in C_DIM_ORDER:
            entry = pearson.get(dimension)
            if isinstance(entry, Mapping) and _num(entry.get("estimate")) is not None:
                entries.append(
                    (f"C {C_DIM_LABELS.get(dimension, dimension)} Pearson", entry)
                )
    if not entries:
        return None

    bounds = [0.02]
    for _label, entry in entries:
        for key in ("estimate", "lo", "hi"):
            value = _num(entry.get(key))
            if value is not None:
                bounds.append(abs(value))
    domain = max(bounds) * 1.15
    row_h = 30
    width = 900
    height = 104 + len(entries) * row_h + 44
    margin_left, margin_top, margin_right = 250, 92, 40
    plot_w = width - margin_left - margin_right
    plot_h = len(entries) * row_h
    zero_x = margin_left + plot_w / 2.0

    def x_of(value: float) -> float:
        return margin_left + (value + domain) / (2.0 * domain) * plot_w

    body: list[str] = [
        _svg_text(width / 2, 34, "C1 vs C2：配对差值（C1 − C2，95% CI）", size=18,
                  anchor="middle", weight="bold"),
        _svg_text(
            width / 2,
            56,
            "0 线表示两条件无差异；C1 为正方向，C2 为负方向；误差线为配对共享聚类的 bootstrap CI",
            size=12,
            anchor="middle",
            fill=SVG_MUTED_COLOR,
        ),
    ]
    body.append(_svg_line(zero_x, margin_top - 8, zero_x, margin_top + plot_h + 8,
                          stroke="#888888", width=1.2))
    for step in (-domain / 2.0, domain / 2.0):
        body.append(_svg_line(x_of(step), margin_top - 8, x_of(step), margin_top + plot_h + 8,
                              stroke=SVG_GRID_COLOR))
        body.append(
            _svg_text(x_of(step), margin_top + plot_h + 26, f"{step:+.3f}", size=11,
                      anchor="middle", fill=SVG_MUTED_COLOR)
        )
    body.append(
        _svg_text(zero_x, margin_top + plot_h + 26, "0", size=11, anchor="middle",
                  fill=SVG_MUTED_COLOR)
    )
    for index, (label, entry) in enumerate(entries):
        row_center = margin_top + index * row_h + row_h / 2.0
        estimate = _num(entry.get("estimate"))
        lo = _num(entry.get("lo"))
        hi = _num(entry.get("hi"))
        body.append(
            _svg_text(margin_left - 10, row_center + 4, label, size=11, anchor="end")
        )
        if estimate is None:
            continue
        color = SVG_C1_COLOR if estimate >= 0 else SVG_C2_COLOR
        x_estimate = x_of(estimate)
        body.append(
            _svg_rect(min(zero_x, x_estimate), row_center - 7, abs(x_estimate - zero_x), 14, color)
        )
        if lo is not None and hi is not None:
            x_lo = x_of(lo)
            x_hi = x_of(hi)
            body.append(_svg_line(x_lo, row_center, x_hi, row_center, width=1.4,
                                  stroke="#333333"))
            body.append(_svg_line(x_lo, row_center - 4, x_lo, row_center + 4, width=1.2,
                                  stroke="#333333"))
            body.append(_svg_line(x_hi, row_center - 4, x_hi, row_center + 4, width=1.2,
                                  stroke="#333333"))
        label_x = x_estimate + 6 if estimate >= 0 else x_estimate - 6
        anchor = "start" if estimate >= 0 else "end"
        body.append(
            _svg_text(label_x, row_center + 4, f"{estimate:+.4f}", size=10, anchor=anchor)
        )
    return _svg_document(width, height, "C1 vs C2 配对差值图", body)


def write_figures(payload: Mapping[str, Any], figures_dir: str | Path) -> dict[str, str]:
    """Write every available SVG figure into ``figures_dir``.

    Returns ``{file name: absolute path}`` for the figures that were written; a
    figure whose data is unavailable is skipped instead of failing the report.
    """
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    builders = (
        (FIG_C_SCATTER, _svg_c_scatter),
        (FIG_A_BAR, _svg_a_accuracy_bar),
        (FIG_C1_VS_C2, _svg_c1_vs_c2),
    )
    for name, builder in builders:
        content = builder(payload)
        if content is None:
            continue
        path = directory / name
        path.write_text(content, encoding="utf-8")
        written[name] = str(path)
    return written


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_report(
    cfg: Config,
    run_id: str,
    *,
    n_boot: int | None = None,
) -> dict[str, Any]:
    """Write all sentence-level report artifacts for one scored run.

    Parameters
    ----------
    cfg:
        Loaded :class:`~pinyin_eval.config.Config`; used for the results
        directory, bootstrap settings and the single-syllable comparison source.
    run_id:
        Run id below ``sentence.paths.results_dir`` (or an explicit run
        directory path).
    n_boot:
        Override ``sentence.stats.n_boot`` (useful for fast smoke reports).

    Returns
    -------
    dict
        The report payload (the same JSON-serialisable structure that
        :func:`render_markdown` consumes, including the ``"summary"`` block and
        the artifact paths).  Artifacts are written into the run directory.
    """
    builder = _ReportBuilder(cfg, run_id, n_boot=n_boot)
    return builder.run()
