"""Offline tests for :mod:`pinyin_eval.sentence.report`.

Covers the pure markdown renderer, the frozen CSV writers, the hand-rolled SVG
figures and an end-to-end :func:`build_report` run over a synthetic (tmp_path)
run directory.  No network, no real audio, no API key.
"""

from __future__ import annotations

import copy
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from pinyin_eval.config import Config
from pinyin_eval.sentence import report

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "sentence.yaml"
SVG_NS = "{http://www.w3.org/2000/svg}"


# ---------------------------------------------------------------------------
# Synthetic payload
# ---------------------------------------------------------------------------


def _ci(value: float, delta: float = 0.02) -> dict[str, float]:
    return {"estimate": value, "lo": value - delta, "hi": value + delta}


def _a_metrics(*, bump: float = 0.0) -> dict:
    headline = {
        "joint_accuracy": 0.75 + bump,
        "initial_accuracy": 0.90 + bump,
        "final_accuracy": 0.85 + bump,
        "tone_accuracy": 0.80 + bump,
        "sentence_exact_rate": 0.25 + bump,
        "insertion_rate": 0.02,
        "deletion_rate": 0.03,
        "cer_tone": 0.18,
        "cer_stripped": 0.09,
        "alignment_coverage": 0.99,
    }
    subsets = {
        name: {"n_items": 4, "n_parse_ok": 4, **headline}
        for name in ("stable", "neutral", "yibu", "erhua")
    }
    n1 = {
        "n1_joint_accuracy": 0.76 + bump,
        "n1_initial_accuracy": 0.91 + bump,
        "n1_final_accuracy": 0.86 + bump,
        "n1_tone_accuracy": 0.81 + bump,
        "n1_sentence_exact_rate": 0.26 + bump,
        "n1_cer_tone": 0.17,
        "delta_vs_n0": {"joint_accuracy": 0.01, "cer_tone": -0.01},
    }
    return {
        "n_items": 4,
        "n_parse_ok": 4,
        "n_parse_failures": 0,
        "parse_failure_rate": 0.0,
        "n_skipped_no_gold": 0,
        "headline": headline,
        "with_ci": {name: _ci(value) for name, value in headline.items()},
        "subsets": subsets,
        "n1": n1,
        "per_speaker": [
            {
                "speaker": "SSB0005",
                "n_items": 2,
                "joint_accuracy": 0.8,
                "initial_accuracy": 0.9,
                "final_accuracy": 0.85,
                "tone_accuracy": 0.8,
            },
            {
                "speaker": "SSB0006",
                "n_items": 2,
                "joint_accuracy": 0.7,
                "initial_accuracy": 0.9,
                "final_accuracy": 0.85,
                "tone_accuracy": 0.8,
            },
        ],
    }


def _b_dimension(macro_f1: float, tp: int = 3, fp: int = 1, fn: int = 0, tn: int = 6) -> dict:
    return {
        "n_items": 2,
        "n_units": tp + fp + fn + tn,
        "macro_f1": macro_f1,
        "f1_error": macro_f1 - 0.05,
        "f1_ok": macro_f1 + 0.05,
        "fpr": 0.1,
        "fnr": 0.2,
        "accuracy": 0.9,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _b_metrics(*, bump: float = 0.0) -> dict:
    dimensions = {
        "tone": _b_dimension(0.60 + bump),
        "consonant": _b_dimension(0.65 + bump),
        "vowel": _b_dimension(0.70 + bump),
    }
    return {
        "n_items": 2,
        "n_parse_ok": 2,
        "n_skipped_no_gold": 0,
        "alignment_coverage": 0.98,
        **dimensions,
        "zero_initial_consonant": _b_dimension(0.55),
        "with_ci": {
            dimension: {"macro_f1": _ci(stats["macro_f1"])}
            for dimension, stats in dimensions.items()
        },
        "per_learner": [
            {
                "learner": "L01",
                "n_items": 1,
                "macro_f1": {"tone": 0.6, "consonant": 0.7, "vowel": 0.8},
                "f1_error": {"tone": 0.55, "consonant": 0.65, "vowel": 0.75},
                "fpr": {"tone": 0.1, "consonant": 0.2, "vowel": 0.3},
                "fnr": {"tone": 0.4, "consonant": 0.3, "vowel": 0.2},
            },
            {
                "learner": "L02",
                "n_items": 1,
                "macro_f1": {"tone": 0.65, "consonant": 0.75, "vowel": 0.85},
                "f1_error": {"tone": 0.6, "consonant": 0.7, "vowel": 0.8},
                "fpr": {"tone": 0.2, "consonant": 0.3, "vowel": 0.4},
                "fnr": {"tone": 0.3, "consonant": 0.2, "vowel": 0.1},
            },
        ],
    }


def _c_stats(n: int, pearson: float | None, mae: float) -> dict:
    return {
        "n": n,
        "pearson": pearson,
        "spearman": pearson,
        "mae": mae,
        "rmse": mae + 0.1,
    }


def _c_metrics(*, bump: float = 0.0) -> dict:
    dimensions = {
        "accuracy": _c_stats(3, 0.70 + bump, 0.40),
        "fluency": _c_stats(3, 0.55 + bump, 0.50),
        "prosody": _c_stats(3, 0.40 + bump, 0.60),
        "mean_dimension": _c_stats(3, 0.65 + bump, 0.45),
    }
    return {
        "n_items": 3,
        "n_parse_ok": 3,
        "n_parse_failures": 0,
        "parse_failure_rate": 0.0,
        "n_skipped_no_gold": 0,
        "alignment_coverage": 1.0,
        **dimensions,
        "native_anchor": {
            "n": 1,
            "mean_gold": 5.0,
            "mean_pred": 4.6,
            "per_dimension": {
                dimension: {"mean_gold": 5.0, "mean_pred": 4.6}
                for dimension in ("accuracy", "fluency", "prosody")
            },
        },
        "with_ci": {
            dimension: {
                "pearson": _ci(stats["pearson"]) if stats["pearson"] is not None else None,
                "mae": _ci(stats["mae"]),
            }
            for dimension, stats in dimensions.items()
        },
        "per_learner": [
            {
                "learner": "L01",
                "n_items": 2,
                "pearson": {"accuracy": 0.7, "fluency": 0.5, "prosody": 0.4, "mean_dimension": 0.6},
                "spearman": {"mean_dimension": 0.6},
                "mae": {"accuracy": 0.4, "fluency": 0.5, "prosody": 0.6, "mean_dimension": 0.5},
                "rmse": {"mean_dimension": 0.6},
            },
            {
                "learner": "L02",
                "n_items": 1,
                "pearson": {"accuracy": 0.6, "fluency": 0.4, "prosody": 0.3, "mean_dimension": 0.5},
                "spearman": {"mean_dimension": 0.5},
                "mae": {"accuracy": 0.5, "fluency": 0.6, "prosody": 0.7, "mean_dimension": 0.6},
                "rmse": {"mean_dimension": 0.7},
            },
        ],
    }


def _compare_a() -> dict:
    return {
        "task": "A",
        "direction": "C1 - C2",
        "method": "paired_cluster_bootstrap",
        "n_items_c1": 4,
        "n_items_c2": 4,
        "n_items_paired": 4,
        "differences": {
            metric: {"estimate": 0.05, "lo": 0.02, "hi": 0.08, "p_value": None}
            for metric in report.A_METRIC_ORDER
        },
        "mcnemar": {
            "sentence_exact": {
                "n_pairs": 4,
                "b": 2,
                "c": 0,
                "statistic": 0.0,
                "p_value": 0.5,
                "method": "exact_binomial",
            }
        },
        "holm": {"sentence_exact": 0.5},
        "per_item_joint_counts": {
            "n_items": 4,
            "joint_all_correct_c1": 3,
            "joint_all_correct_c2": 1,
            "both": 1,
            "c1_only": 2,
            "c2_only": 0,
            "neither": 1,
        },
    }


def _compare_b() -> dict:
    return {
        "task": "B",
        "direction": "C1 - C2",
        "method": "paired_cluster_bootstrap",
        "n_items_c1": 2,
        "n_items_c2": 2,
        "n_items_paired": 2,
        "differences": {
            "macro_f1": {
                dimension: {"estimate": 0.03, "lo": 0.01, "hi": 0.05, "p_value": None}
                for dimension in report.B_DIM_ORDER
            }
        },
        "mcnemar": {},
        "holm": {},
    }


def _compare_c() -> dict:
    return {
        "task": "C",
        "direction": "C1 - C2",
        "method": "paired_cluster_bootstrap",
        "n_items_c1": 3,
        "n_items_c2": 3,
        "n_items_paired": 3,
        "differences": {
            "pearson": {
                dimension: {"estimate": 0.04, "lo": 0.01, "hi": 0.07, "p_value": None}
                for dimension in report.C_DIM_ORDER
            },
            "mae": {
                dimension: {"estimate": -0.03, "lo": -0.05, "hi": -0.01, "p_value": None}
                for dimension in report.C_DIM_ORDER
            },
        },
        "mcnemar": {},
        "holm": {},
    }


def _usage() -> dict:
    def entry(task: str, condition: str, cost: float) -> dict:
        return {
            "task": task,
            "condition": condition,
            "n_calls": 4,
            "n_scored_calls": 4,
            "n_ok": 4,
            "n_errors": 0,
            "n_parse_failures": 0,
            "n_retries": 0,
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "reasoning_tokens": 200,
            "avg_latency_s": 1.2,
            "cost_cny": cost,
            "finish_reasons": {"stop": 4},
        }

    by_condition = {
        f"{task}|{condition}": entry(task, condition, 0.25)
        for task in report.TASKS
        for condition in report.CONDITIONS
    }
    return {
        "source": "cost_summary.json",
        "totals": {
            "n_calls": 24,
            "n_scored_calls": 24,
            "n_ok": 23,
            "n_errors": 0,
            "n_parse_failures": 1,
            "n_retries": 2,
            "prompt_tokens": 1200,
            "completion_tokens": 600,
            "reasoning_tokens": 200,
            "avg_latency_s": 1.2,
            "cost_cny": 1.5,
        },
        "by_task": {
            task: {
                "n_calls": 8,
                "n_scored_calls": 8,
                "n_ok": 8,
                "n_errors": 0,
                "n_parse_failures": 0,
                "n_retries": 0,
                "cost_cny": 0.5,
            }
            for task in report.TASKS
        },
        "by_task_condition": by_condition,
    }


def _parse_failures() -> dict:
    return {
        "by_task": {
            "A": {"ok": 8},
            "B": {"ok": 8},
            "C": {"ok": 7, "parse_error": 1},
        },
        "by_task_condition": {
            "A|C1": {"ok": 4},
            "A|C2": {"ok": 4},
            "B|C1": {"ok": 4},
            "B|C2": {"ok": 4},
            "C|C1": {"ok": 3, "parse_error": 1},
            "C|C2": {"ok": 4},
        },
        "rates_by_task": {},
        "overall": {"n_calls": 24, "n_ok": 23, "n_failures": 1, "rate": 23 / 24},
        "n_failure_records": 1,
        "source": "score_summary.json",
    }


def _acceptance() -> dict:
    return {
        "parse_success": {
            task: {
                "rate": 0.98,
                "n_calls": 8,
                "n_errors": 0,
                "n_parse_failures": 0,
                "threshold": 0.95,
                "passed": True,
            }
            for task in report.TASKS
        },
        "alignment_coverage": {
            task: {"value": 0.99, "threshold": 0.95, "passed": True}
            for task in report.TASKS
        },
        "cost_cny": {"value": 1.5, "threshold": 20.0, "passed": True},
        "passed": True,
    }


def _available() -> dict:
    return {
        task: {
            condition: {"available": True, "n_items": 4, "reason": None}
            for condition in report.CONDITIONS
        }
        for task in report.TASKS
    }


def _category_rows() -> list[dict]:
    return [
        {
            "dimension": "initial",
            "category": "n",
            "single_n": 400,
            "single_acc": 0.98,
            "single_weight": 0.4,
            "sentence_n": 20,
            "sentence_acc": 0.9,
            "covered": True,
        },
        {
            "dimension": "initial",
            "category": "",
            "single_n": 300,
            "single_acc": 0.92,
            "single_weight": 0.3,
            "sentence_n": 8,
            "sentence_acc": 0.875,
            "covered": True,
        },
        {
            "dimension": "initial",
            "category": "zh",
            "single_n": 100,
            "single_acc": 0.8,
            "single_weight": 0.1,
            "sentence_n": None,
            "sentence_acc": None,
            "covered": False,
        },
        {
            "dimension": "final",
            "category": "i",
            "single_n": 200,
            "single_acc": 0.99,
            "single_weight": 0.2,
            "sentence_n": 10,
            "sentence_acc": 0.9,
            "covered": True,
        },
        {
            "dimension": "tone",
            "category": "3",
            "single_n": 500,
            "single_acc": 0.88,
            "single_weight": 0.25,
            "sentence_n": 10,
            "sentence_acc": 0.7,
            "covered": True,
        },
    ]


def _single_syllable() -> dict:
    single = {
        "n": 2048,
        "joint_accuracy": 0.8384,
        "initial_accuracy": 0.9624,
        "final_accuracy": 0.9487,
        "tone_accuracy": 0.9116,
    }
    stable = {
        "n_items": 4,
        "n_parse_ok": 4,
        "joint_accuracy": 0.70,
        "initial_accuracy": 0.88,
        "final_accuracy": 0.84,
        "tone_accuracy": 0.79,
    }
    sentence_all = {
        "n_items": 4,
        "joint_accuracy": 0.75,
        "initial_accuracy": 0.90,
        "final_accuracy": 0.85,
        "tone_accuracy": 0.80,
    }
    deltas = {
        dimension: stable[dimension] - single[dimension]
        for dimension in ("joint_accuracy", "initial_accuracy", "final_accuracy", "tone_accuracy")
    }
    weighted = {
        "initial": {
            "sentence": 0.89,
            "reference": 0.95,
            "delta": -0.06,
            "coverage": 0.8,
            "n_categories_single": 22,
            "n_categories_sentence": 18,
            "n_categories_covered": 16,
        },
        "final": {
            "sentence": 0.87,
            "reference": 0.94,
            "delta": -0.07,
            "coverage": 1.0,
            "n_categories_single": 36,
            "n_categories_sentence": 30,
            "n_categories_covered": 30,
        },
        "tone": {
            "sentence": 0.76,
            "reference": 0.9,
            "delta": -0.14,
            "coverage": 0.75,
            "n_categories_single": 5,
            "n_categories_sentence": 4,
            "n_categories_covered": 3,
        },
    }
    return {
        "available": True,
        "weighted_available": True,
        "weighted_reason": None,
        "source": "results/runs/mini128-v1/items.parquet",
        "source_run_id": "mini128-v1",
        "reference_source": "results/runs/mini128-v1/items.parquet",
        "reference_filter": (
            "experiment=exp3 & prompt_condition=free & parse_status=ok & "
            "generation_condition=free__think0__clean__json"
        ),
        "single": single,
        "sentence_all": sentence_all,
        "sentence_stable": stable,
        "deltas": deltas,
        "weighted": weighted,
        "category_rows": _category_rows(),
        "caveats": [
            "语料不同（TTS 单音节 vs 真人朗读整句）",
            "加权口径：使用单音节 exp3 free 的类别分布作为权重；覆盖率见 §6.2",
        ],
    }


def synthetic_payload() -> dict:
    """A complete, hand-built payload matching :func:`build_report`'s output."""
    c_scatter = {
        "accuracy": [[4.0, 3.8], [3.2, 3.0], [2.5, 2.6]],
        "fluency": [[4.5, 4.2], [3.0, 2.8]],
        "prosody": [[4.0, 3.9]],
    }
    return {
        "run_id": "synthetic-run",
        "generated_at": "2026-11-01T21:00:00+08:00",
        "run_dir": "/tmp/results/sentence/synthetic-run",
        "main_condition": "C1",
        "manifest": {
            "run_id": "synthetic-run",
            "created_at": "2026-11-01T20:00:00+08:00",
            "model": "qwen3.8-omni-flash",
            "provider": "dashscope_openai",
            "prompt_version": "sentence-v1",
            "git_commit": "abc1234",
            "seed": 20261101,
            "mock": True,
            "n_planned": 24,
            "budget_cny": 20.0,
            "prep_summary": {
                "aishell3": {"n_items": 4, "n_speakers": 2},
                "ompal": {"n_learner": 2, "n_native_anchor": 1},
            },
        },
        "config": {
            "n_boot": 200,
            "alpha": 0.05,
            "seed": 20261101,
            "cluster_key": "cluster_id",
            "budget_cny": 20.0,
            "model": "qwen3.8-omni-flash",
        },
        "available": _available(),
        "cost": {"per_task_condition": {}, "totals": {}},
        "usage": _usage(),
        "score_summary": {"status_counts": {"A": {"ok": 8}, "B": {"ok": 8}, "C": {"ok": 7, "parse_error": 1}}},
        "parse_failures": _parse_failures(),
        "a": _a_metrics(),
        "a_c2": _a_metrics(bump=0.1),
        "b": _b_metrics(),
        "b_c2": _b_metrics(bump=0.01),
        "c": _c_metrics(),
        "c_c2": _c_metrics(bump=0.01),
        "compare": {"A": _compare_a(), "B": _compare_b(), "C": _compare_c()},
        "acceptance": _acceptance(),
        "single_syllable": _single_syllable(),
        "c_scatter": c_scatter,
        "warnings": ["示例降级：单音节汇总缺失"],
        "artifacts": {
            "report.md": "report.md",
            "metrics.csv": "metrics.csv",
            "comparisons.csv": "comparisons.csv",
            "figures/c_pearson_scatter.svg": "figures/c_pearson_scatter.svg",
            "figures/a_accuracy_bar.svg": "figures/a_accuracy_bar.svg",
            "figures/c1_vs_c2.svg": "figures/c1_vs_c2.svg",
        },
    }


@pytest.fixture()
def payload() -> dict:
    return synthetic_payload()


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_contains_all_sections(payload):
    text = report.render_markdown(payload)
    assert text.startswith("# Qwen3.8-Omni 整句拼音评测报告（run `synthetic-run`）")
    for heading in (
        "## 1. 运行元信息",
        "## 2. 任务 A",
        "## 3. 任务 B",
        "## 4. 任务 C",
        "## 5. C1 vs C2",
        "## 6. 与单音节结果的聚合对照",
        "## 7. 成本 / 用量 / 解析失败 / 重试",
        "## 8. 验收检查",
        "## 9. 局限与解释边界",
        "## 附录：工件清单",
    ):
        assert heading in text
    assert text.endswith("\n")


def test_render_markdown_contains_expected_values_and_tables(payload):
    text = report.render_markdown(payload)
    # task A headline (4 decimals) and its bootstrap CI
    assert "0.7500" in text
    assert "[0.7300, 0.7700]" in text
    # task A subsets / N1 / per speaker
    assert "stable（排除轻声/一不/儿化）" in text
    assert "N1−N0" in text
    assert "SSB0005" in text
    # task B macro-F1 / confusion / zero-initial consonant / per learner
    assert "macro-F1" in text
    assert "零声母 consonant" in text
    assert "L01" in text
    # task C correlations, native anchor and scatter figure
    assert "0.7000" in text
    assert "native 条数 n=1" in text
    assert "模型预测三维均分=4.6000" in text
    assert "| accuracy（准确度） | 5.0000 | 4.6000 |" in text
    assert "![任务 C：预测分 vs 人工分（三维小多图）](figures/c_pearson_scatter.svg)" in text
    # C1 vs C2 section: McNemar + direction
    assert "McNemar" in text
    assert "C1 更高" in text
    # single-syllable comparison: unweighted + category-weighted (design §7.4)
    assert "### 6.1 不加权聚合对照" in text
    assert "### 6.2 类别加权对照（design §7.4）" in text
    assert "### 6.3 逐类别明细（每维度前 10 类）" in text
    assert "### 6.4 解释边界" in text
    assert "单音节 exp3 free" in text
    assert "0.8384" in text
    assert "单音节加权准确率" in text
    assert "0.9500" in text  # weighted single-syllable initial accuracy
    assert "0.8900" in text  # weighted in-run initial accuracy
    assert "80.00%" in text  # weight coverage of the initial dimension
    assert "单音节参考过滤条件" in text
    assert "`single_syllable_categories.csv`" in text
    assert "未按类别分布加权" not in text
    # cost / parse failures / acceptance
    assert "cost_summary.json" in text
    assert "parse_error" in text
    assert "验收总结果" in text
    # the report must be substantial (all sections rendered)
    assert len(text.splitlines()) > 120


def test_render_markdown_is_deterministic_and_json_roundtrips(payload):
    first = report.render_markdown(payload)
    second = report.render_markdown(payload)
    roundtripped = json.loads(json.dumps(payload, ensure_ascii=False))
    third = report.render_markdown(roundtripped)
    assert first == second == third


def test_render_markdown_missing_c2_is_tolerated(payload):
    partial = copy.deepcopy(payload)
    partial["a_c2"] = None
    partial["b_c2"] = None
    partial["c_c2"] = None
    partial["compare"] = {"A": None, "B": None, "C": None}
    partial["available"]["A"]["C2"] = {
        "available": False,
        "n_items": 0,
        "reason": "C2 条件没有可评分行（解析失败或未生成）",
    }
    text = report.render_markdown(partial)
    assert "C2 条件不可用" in text
    assert "_C1/C2 配对比较不可用" in text
    assert "不可用" in text
    # C1 sections are still fully rendered
    assert "## 2. 任务 A" in text
    assert "0.7500" in text


def test_render_markdown_weighted_comparison_fallback(payload):
    partial = copy.deepcopy(payload)
    partial["single_syllable"]["weighted_available"] = False
    partial["single_syllable"]["weighted"] = {}
    partial["single_syllable"]["category_rows"] = []
    partial["single_syllable"]["weighted_reason"] = "任务 A 评分行缺少 cat_*_stats 列（旧评分行）"
    text = report.render_markdown(partial)
    assert "不可用" in text
    assert "cat_*_stats" in text
    assert "见 §9.1 警告" in text
    assert "### 6.1 不加权聚合对照" in text
    assert "0.8384" in text


def test_render_markdown_native_anchor_fallback(payload):
    partial = copy.deepcopy(payload)
    partial["c"]["native_anchor"] = {"n": 0, "mean_gold": None}
    text = report.render_markdown(partial)
    assert "没有 native 锚点 items" in text
    assert "模型预测三维均分" not in text


def test_render_markdown_task_unavailable():
    text = report.render_markdown({"run_id": "x", "a": None, "b": None, "c": None})
    assert "_任务 A 不可用" in text
    assert "_任务 B 不可用" in text
    assert "_任务 C 不可用" in text
    assert "_与单音节结果的聚合对照（design §7.4）_" not in text
    assert "不可用" in text


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_write_metrics_csv(payload, tmp_path):
    path = tmp_path / "metrics.csv"
    report.write_metrics_csv(payload, path)
    rows = _read_csv(path)
    assert tuple(rows[0].keys()) == report.METRICS_COLUMNS
    assert len(rows) > 40
    headline = [row for row in rows if row["task"] == "A" and row["scope"] == "headline"]
    assert headline
    joint = next(row for row in headline if row["metric"] == "joint_accuracy")
    assert joint["condition"] == "C1"
    assert float(joint["estimate"]) == pytest.approx(0.75)
    assert float(joint["ci_lo"]) == pytest.approx(0.73)
    assert float(joint["ci_hi"]) == pytest.approx(0.77)
    subset = [
        row
        for row in rows
        if row["task"] == "A"
        and row["condition"] == "C1"
        and row["scope"] == "subset"
        and row["subset"] == "stable"
    ]
    assert len(subset) == len(report.A_METRIC_ORDER)
    assert {row["task"] for row in rows} == {"A", "B", "C"}
    assert {row["condition"] for row in rows} == {"C1", "C2"}
    assert any(row["scope"] == "confusion" and row["metric"] == "tp" for row in rows)
    assert any(row["scope"] == "zero_initial_consonant" for row in rows)
    assert any(row["scope"] == "native_anchor" and row["metric"] == "mean_gold" for row in rows)


def test_write_comparisons_csv(payload, tmp_path):
    path = tmp_path / "comparisons.csv"
    report.write_comparisons_csv(payload, path)
    rows = _read_csv(path)
    assert tuple(rows[0].keys()) == report.COMPARISON_COLUMNS
    assert all(row["direction"] == "C1 - C2" for row in rows)
    mcnemar = [row for row in rows if row["task"] == "A" and row["metric"] == "mcnemar"]
    assert len(mcnemar) == 1
    assert float(mcnemar[0]["p_value"]) == pytest.approx(0.5)
    assert float(mcnemar[0]["p_holm"]) == pytest.approx(0.5)
    assert mcnemar[0]["method"] == "exact_binomial"
    assert any(row["task"] == "B" and row["metric"] == "macro_f1" for row in rows)
    assert any(row["task"] == "C" and row["metric"] == "mae" for row in rows)
    # the A headline differences include a bootstrap CI
    joint = next(
        row
        for row in rows
        if row["task"] == "A" and row["metric"] == "difference" and row["dimension"] == "joint"
    )
    assert float(joint["estimate"]) == pytest.approx(0.05)
    assert float(joint["ci_lo"]) == pytest.approx(0.02)
    assert joint["p_value"] == ""


def test_write_per_entity_csvs(payload, tmp_path):
    speaker_path = tmp_path / "per_speaker_a.csv"
    report.write_per_speaker_a_csv(payload, speaker_path)
    speakers = _read_csv(speaker_path)
    assert tuple(speakers[0].keys()) == report.PER_SPEAKER_A_COLUMNS
    assert [row["speaker"] for row in speakers] == ["SSB0005", "SSB0006"]

    learner_b_path = tmp_path / "per_learner_b.csv"
    report.write_per_learner_b_csv(payload, learner_b_path)
    learners_b = _read_csv(learner_b_path)
    assert tuple(learners_b[0].keys()) == report.PER_LEARNER_B_COLUMNS
    assert [row["learner"] for row in learners_b] == ["L01", "L02"]
    assert float(learners_b[0]["macro_f1_tone"]) == pytest.approx(0.6)
    assert float(learners_b[0]["fpr_vowel"]) == pytest.approx(0.3)

    learner_c_path = tmp_path / "per_learner_c.csv"
    report.write_per_learner_c_csv(payload, learner_c_path)
    learners_c = _read_csv(learner_c_path)
    assert tuple(learners_c[0].keys()) == report.PER_LEARNER_C_COLUMNS
    assert [row["learner"] for row in learners_c] == ["L01", "L02"]
    assert float(learners_c[0]["pearson_mean_dimension"]) == pytest.approx(0.6)
    assert float(learners_c[0]["rmse_mean_dimension"]) == pytest.approx(0.6)


def test_write_single_syllable_categories_csv(payload, tmp_path):
    path = tmp_path / "single_syllable_categories.csv"
    report.write_single_syllable_categories_csv(payload, path)
    rows = _read_csv(path)
    assert tuple(rows[0].keys()) == report.SINGLE_SYLLABLE_CATEGORY_COLUMNS
    assert len(rows) == len(payload["single_syllable"]["category_rows"])
    first = rows[0]
    assert first["dimension"] == "initial"
    assert first["category"] == "n"
    assert float(first["single_weight"]) == pytest.approx(0.4)
    assert first["covered"] == "True"
    zero_initial = [row for row in rows if row["dimension"] == "initial" and row["category"] == ""]
    assert len(zero_initial) == 1
    assert zero_initial[0]["sentence_acc"] == "0.875"
    missing = [row for row in rows if row["covered"] == "False"]
    assert missing and missing[0]["sentence_n"] == ""


def test_csv_writers_on_empty_payload(tmp_path):
    report.write_metrics_csv({}, tmp_path / "metrics.csv")
    assert _read_csv(tmp_path / "metrics.csv") == []
    header = (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == list(report.METRICS_COLUMNS)
    report.write_single_syllable_categories_csv({}, tmp_path / "categories.csv")
    assert _read_csv(tmp_path / "categories.csv") == []
    categories_header = (tmp_path / "categories.csv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert categories_header.split(",") == list(report.SINGLE_SYLLABLE_CATEGORY_COLUMNS)


# ---------------------------------------------------------------------------
# SVG figures
# ---------------------------------------------------------------------------


def _parse_svg(path: Path) -> ET.Element:
    tree = ET.parse(path)  # raises on invalid XML
    root = tree.getroot()
    assert root.tag == f"{SVG_NS}svg"
    return root


def test_write_figures_valid_xml_and_point_counts(payload, tmp_path):
    figures_dir = tmp_path / "figures"
    written = report.write_figures(payload, figures_dir)
    assert set(written) == {
        report.FIG_C_SCATTER,
        report.FIG_A_BAR,
        report.FIG_C1_VS_C2,
    }
    for name in written:
        assert (figures_dir / name).exists()
        _parse_svg(figures_dir / name)

    scatter_root = _parse_svg(figures_dir / report.FIG_C_SCATTER)
    circles = [element for element in scatter_root.iter() if element.tag == f"{SVG_NS}circle"]
    expected_points = sum(len(points) for points in payload["c_scatter"].values())
    assert len(circles) == expected_points
    assert len(circles) == 6

    bar_root = _parse_svg(figures_dir / report.FIG_A_BAR)
    rects = [element for element in bar_root.iter() if element.tag == f"{SVG_NS}rect"]
    # background + 4 categories x 2 conditions
    assert len(rects) >= 1 + 8

    delta_root = _parse_svg(figures_dir / report.FIG_C1_VS_C2)
    delta_rects = [element for element in delta_root.iter() if element.tag == f"{SVG_NS}rect"]
    # background + A(4) + B(3) + C(4) effect-size bars
    assert len(delta_rects) >= 1 + 11


def test_write_figures_skips_unavailable_data(tmp_path):
    written = report.write_figures(
        {"a": None, "a_c2": None, "b": None, "c": None, "compare": {}, "c_scatter": {}},
        tmp_path / "figures",
    )
    assert written == {}
    assert not (tmp_path / "figures" / report.FIG_C_SCATTER).exists()


def test_svg_scatter_respects_artifacts_gate(payload, tmp_path):
    text = report.render_markdown(payload)
    assert "figures/c_pearson_scatter.svg" in text
    payload_without_figure = copy.deepcopy(payload)
    payload_without_figure["artifacts"].pop("figures/c_pearson_scatter.svg")
    text_without = report.render_markdown(payload_without_figure)
    assert "散点图不可用" in text_without


# ---------------------------------------------------------------------------
# build_report end-to-end on a synthetic run directory
# ---------------------------------------------------------------------------


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_single_syllable_reference(tmp_path: Path) -> Path:
    """Fake ``results/runs/mini128-v1`` (items.parquet + summary.json)."""
    base = tmp_path / "results" / "runs" / "mini128-v1"
    base.mkdir(parents=True, exist_ok=True)
    main_condition_id = "free__think0__clean__json"
    rows = [
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": main_condition_id,
            "gold_initial": "n",
            "gold_final": "a",
            "gold_tone": 1,
            "initial_correct": True,
            "final_correct": True,
            "tone_correct": True,
            "joint_correct": True,
        },
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": main_condition_id,
            "gold_initial": "n",
            "gold_final": "a",
            "gold_tone": 2,
            "initial_correct": True,
            "final_correct": False,
            "tone_correct": False,
            "joint_correct": False,
        },
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": main_condition_id,
            "gold_initial": "h",
            "gold_final": "ai",
            "gold_tone": 3,
            "initial_correct": True,
            "final_correct": True,
            "tone_correct": False,
            "joint_correct": False,
        },
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": main_condition_id,
            "gold_initial": "h",
            "gold_final": "ai",
            "gold_tone": 4,
            "initial_correct": False,
            "final_correct": True,
            "tone_correct": True,
            "joint_correct": False,
        },
        # distractor rows from a non-main condition: the reference filter must drop them
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": "free__think1__clean__json",
            "gold_initial": "zh",
            "gold_final": "i",
            "gold_tone": 2,
            "initial_correct": True,
            "final_correct": True,
            "tone_correct": True,
            "joint_correct": True,
        },
        {
            "experiment": "exp3",
            "prompt_condition": "free",
            "parse_status": "ok",
            "condition_id": "free__think0__clean__text",
            "gold_initial": "zh",
            "gold_final": "i",
            "gold_tone": 2,
            "initial_correct": True,
            "final_correct": True,
            "tone_correct": True,
            "joint_correct": True,
        },
    ]
    pd.DataFrame(rows).to_parquet(base / "items.parquet", index=False)
    summary = {
        "run_id": "mini128-v1",
        "main_condition": {
            "thinking": False,
            "audio_quality": "clean",
            "output_form": "json",
        },
        # the real summary stores the bare generation key; the loader must rebuild
        # the ``free__...`` condition id from ``main_condition``.
        "main_generation_key": "think0__clean__json",
        "headline": {
            "exp3": {
                "free": {
                    "n": 4,
                    "joint_accuracy": 0.25,
                    "initial_accuracy": 0.75,
                    "final_accuracy": 0.75,
                    "tone_accuracy": 0.5,
                }
            }
        },
    }
    (base / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    return base


def _write_synthetic_run(tmp_path: Path) -> Path:
    _write_single_syllable_reference(tmp_path)
    run_dir = tmp_path / "results" / "sentence" / "synthetic-run"
    run_dir.mkdir(parents=True)

    rows_a = []
    for index, speaker in enumerate(("S1", "S2")):
        rows_a.append(
            {
                "item_id": f"A-C1-{index}",
                "task": "A",
                "condition": "C1",
                "cluster_id": speaker,
                "speaker": speaker,
                "parse_status": "ok",
                "n_gold": 2,
                "n_hyp": 2,
                "n_matched": 2,
                "n_deletions": 0,
                "n_insertions": 0,
                "n_joint_correct": 1,
                "n_initial_correct": 2,
                "n_final_correct": 2,
                "n_tone_correct": 1,
                "cer_tone_num": 1,
                "cer_tone_den": 2,
                "cer_stripped_num": 0,
                "cer_stripped_den": 2,
                "n_char_mismatches": 0,
                "n1_n_joint_correct": 1,
                "n1_n_initial_correct": 2,
                "n1_n_final_correct": 2,
                "n1_n_tone_correct": 1,
                "n1_cer_tone_num": 1,
                "n1_cer_tone_den": 2,
                "alignment_coverage": 1.0,
                "sentence_exact": index == 0,
                "n1_sentence_exact": index == 0,
                "flags": {"neutral": False},
                "stable": True,
                "duration_s": 1.0,
                "cat_initial_stats": json.dumps(
                    {
                        "n": {"n": 1, "correct": 1},
                        "h": {"n": 1, "correct": 0},
                    }
                    if index == 0
                    else {"n": {"n": 1, "correct": 1}},
                    sort_keys=True,
                ),
                "cat_final_stats": json.dumps(
                    {
                        "a": {"n": 1, "correct": 1},
                        "ai": {"n": 1, "correct": 1},
                    }
                    if index == 0
                    else {"ai": {"n": 1, "correct": 1}},
                    sort_keys=True,
                ),
                "cat_tone_stats": json.dumps(
                    {
                        "1": {"n": 1, "correct": 1},
                        "2": {"n": 1, "correct": 0},
                    }
                    if index == 0
                    else {"3": {"n": 1, "correct": 0}},
                    sort_keys=True,
                ),
            }
        )
    _write_parquet(rows_a, run_dir / "rows_a.parquet")

    rows_b = []
    for index, learner in enumerate(("L1", "L2")):
        for dimension in ("tone", "consonant", "vowel"):
            rows_b.append(
                {
                    "item_id": f"B-C1-{index}",
                    "task": "B",
                    "condition": "C1",
                    "cluster_id": learner,
                    "learner": learner,
                    "native": False,
                    "parse_status": "ok",
                    "dimension": dimension,
                    "n_units": 2,
                    "n_matched": 2,
                    "alignment_coverage": 1.0,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "tn": 1,
                    "n_missing": 0,
                    "n_inserted": 0,
                    "n_zero_initial_tp": 1 if dimension == "consonant" else 0,
                    "n_zero_initial_fp": 0,
                    "n_zero_initial_fn": 0,
                    "n_zero_initial_tn": 1 if dimension == "consonant" else 0,
                    "flags": {"neutral": False},
                    "stable": True,
                    "speaker": learner,
                    "duration_s": 1.0,
                }
            )
    _write_parquet(rows_b, run_dir / "rows_b.parquet")

    rows_c = []
    for index, learner in enumerate(("L1", "L2")):
        rows_c.append(
            {
                "item_id": f"C-C1-{index}",
                "task": "C",
                "condition": "C1",
                "cluster_id": learner,
                "learner": learner,
                "native": False,
                "parse_status": "ok",
                "speaker": learner,
                "flags": {"neutral": False},
                "stable": True,
                "duration_s": 1.0,
                "gold_accuracy": 4.0,
                "gold_fluency": 3.0,
                "gold_prosody": 3.5,
                "pred_accuracy": 3.5 + index * 0.5,
                "pred_fluency": 3.0,
                "pred_prosody": 3.0 + index * 0.5,
                "gold_mean": 3.5,
                "pred_mean": 3.2 + index * 0.4,
            }
        )
    _write_parquet(rows_c, run_dir / "rows_c.parquet")

    manifest = {
        "run_id": "synthetic-run",
        "created_at": "2026-11-01T20:00:00+08:00",
        "mock": True,
        "model": "qwen3.8-omni-flash",
        "provider": "dashscope_openai",
        "prompt_version": "sentence-v1",
        "git_commit": "abc1234",
        "seed": 20261101,
        "budget_cny": 20.0,
        "n_planned": 30,
        "conditions": {"C1": {"thinking": False, "max_tokens": 256}},
        "prep_summary": {"aishell3": {"n_items": 2, "n_speakers": 2}},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    score_summary = {
        "run_id": "synthetic-run",
        "generations": 6,
        "status_counts": {
            "A": {"ok": 2},
            "B": {"ok": 2},
            "C": {"ok": 2, "parse_error": 1},
        },
        "n_rows": {"A": 2, "B": 6, "C": 2},
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(score_summary, ensure_ascii=False), encoding="utf-8"
    )
    failure = {
        "task": "C",
        "item_id": "C-C1-9",
        "condition": "C1",
        "parse_status": "parse_error",
        "detail": "invalid JSON",
        "finish_reason": "stop",
        "raw": "oops",
    }
    (run_dir / "parse_failures.jsonl").write_text(
        json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cost = {
        "per_task_condition": {
            "A|C1": {
                "n_calls": 2,
                "n_errors": 0,
                "n_retries": 0,
                "prompt_tokens": 400,
                "completion_tokens": 200,
                "reasoning_tokens": 0,
                "cost_cny": 0.05,
                "latency_s": 2.0,
                "avg_latency_s": 1.0,
                "finish_reasons": {"stop": 2},
            },
            "B|C1": {
                "n_calls": 2,
                "n_errors": 0,
                "n_retries": 1,
                "prompt_tokens": 500,
                "completion_tokens": 250,
                "reasoning_tokens": 0,
                "cost_cny": 0.06,
                "latency_s": 2.0,
                "avg_latency_s": 1.0,
                "finish_reasons": {"stop": 2},
            },
            "C|C1": {
                "n_calls": 2,
                "n_errors": 0,
                "n_retries": 0,
                "prompt_tokens": 300,
                "completion_tokens": 100,
                "reasoning_tokens": 0,
                "cost_cny": 0.03,
                "latency_s": 2.0,
                "avg_latency_s": 1.0,
                "finish_reasons": {"stop": 2},
            },
        },
        "totals": {
            "n_calls": 6,
            "n_errors": 0,
            "prompt_tokens": 1200,
            "completion_tokens": 550,
            "reasoning_tokens": 0,
            "cost_cny": 0.14,
            "avg_latency_s": 1.0,
        },
    }
    (run_dir / "cost_summary.json").write_text(
        json.dumps(cost, ensure_ascii=False), encoding="utf-8"
    )
    generations = []
    for task in ("A", "B", "C"):
        for index in range(2):
            generations.append(
                {
                    "run_id": "synthetic-run",
                    "request_id": f"{task}|C1|{index}",
                    "item_id": f"{task}-C1-{index}",
                    "task": task,
                    "condition": "C1",
                    "thinking": False,
                    "max_tokens": 256,
                    "model": "mock-qwen3.8-omni-flash",
                    "prompt_version": "sentence-v1",
                    "prompt_sha256": "0" * 64,
                    "audio_path": "data/sentence/mock.wav",
                    "response_text": "{}",
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                    "latency_s": 0.5,
                    "error": None,
                    "retries": 0,
                    "timestamp": "2026-11-01T20:00:00+08:00",
                }
            )
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for record in generations:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return run_dir


@pytest.fixture()
def synthetic_run(tmp_path) -> tuple[Config, Path]:
    run_dir = _write_synthetic_run(tmp_path)
    cfg = Config.load(CONFIG_PATH, repo_root=tmp_path)
    return cfg, run_dir


def test_build_report_writes_all_artifacts(synthetic_run):
    cfg, run_dir = synthetic_run
    payload = report.build_report(cfg, "synthetic-run", n_boot=20)
    expected = {
        "report.md",
        "metrics.csv",
        "comparisons.csv",
        "per_speaker_a.csv",
        "per_learner_b.csv",
        "per_learner_c.csv",
        "single_syllable_categories.csv",
        "acceptance.json",
        "report_summary.json",
        "figures/c_pearson_scatter.svg",
        "figures/a_accuracy_bar.svg",
    }
    for name in expected:
        assert (run_dir / name).exists(), name
    # C2 rows are missing -> no paired comparison and no C1-vs-C2 figure
    assert not (run_dir / "figures" / "c1_vs_c2.svg").exists()

    summary = payload["summary"]
    for key in (
        "run_id",
        "run_dir",
        "generated_at",
        "bootstrap",
        "available",
        "headline",
        "acceptance",
        "usage_totals",
        "warnings",
        "artifacts",
    ):
        assert key in summary
    assert summary["run_id"] == "synthetic-run"
    assert summary["bootstrap"]["n_boot"] == 20
    assert summary["artifacts"]["report.md"] == "report.md"
    assert summary["artifacts"]["figures/c_pearson_scatter.svg"].endswith(
        "c_pearson_scatter.svg"
    )
    assert payload["run_id"] == "synthetic-run"
    assert payload["available"]["C"]["C2"]["available"] is False
    assert payload["compare"]["A"] is None
    assert payload["compare"]["C"] is None
    # metrics were really computed from the synthetic rows
    assert payload["a"]["headline"]["joint_accuracy"] == pytest.approx(0.5)
    assert payload["b"]["alignment_coverage"] == pytest.approx(1.0)
    assert payload["c"]["alignment_coverage"] == pytest.approx(1.0)
    # usage accounting separates ok / parse failures (A all ok, C one failure)
    assert payload["usage"]["by_task"]["A"]["n_parse_failures"] == 0
    assert payload["usage"]["by_task"]["C"]["n_parse_failures"] == 1
    assert payload["usage"]["totals"]["n_parse_failures"] == 1
    assert payload["summary"]["acceptance"]["parse_success"]["A"] == pytest.approx(1.0)
    assert payload["summary"]["acceptance"]["parse_success"]["C"] == pytest.approx(2 / 3)
    assert payload["summary"]["parse_success_rate"] == pytest.approx(6 / 7)
    assert payload["summary"]["parse_failure_rate"] == pytest.approx(1 / 7)
    # acceptance gates: C parse success 2/3 fails, so the overall gate fails
    assert payload["acceptance"]["parse_success"]["A"]["passed"] is True
    assert payload["acceptance"]["parse_success"]["C"]["passed"] is False
    assert payload["acceptance"]["passed"] is False
    # single-syllable reference (tmp_path/results/runs/mini128-v1) is available
    # and the category-weighted comparison is computed (design §7.4)
    single_syllable = payload["single_syllable"]
    assert single_syllable["available"] is True
    assert single_syllable["weighted_available"] is True
    assert single_syllable["reference_source"].endswith("items.parquet")
    assert "condition_id=free__think0__clean__json" in single_syllable["reference_filter"]
    assert single_syllable["single"]["n"] == 4  # distractor conditions filtered out
    assert single_syllable["single"]["joint_accuracy"] == pytest.approx(0.25)
    weighted_initial = single_syllable["weighted"]["initial"]
    assert weighted_initial["reference"] == pytest.approx(0.75)
    assert weighted_initial["sentence"] == pytest.approx(0.5)
    assert weighted_initial["delta"] == pytest.approx(-0.25)
    assert weighted_initial["coverage"] == pytest.approx(1.0)
    assert single_syllable["weighted"]["tone"]["coverage"] == pytest.approx(0.75)
    initial_categories = {
        row["category"]
        for row in single_syllable["category_rows"]
        if row["dimension"] == "initial"
    }
    assert initial_categories == {"n", "h"}  # distractor "zh" does not leak in
    assert len(single_syllable["category_rows"]) == 8
    category_csv = _read_csv(run_dir / "single_syllable_categories.csv")
    assert len(category_csv) == 8
    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "类别加权对照（design §7.4）" in report_md
    assert "75.00%" in report_md
    assert "-0.2500" in report_md
    assert payload["warnings"] == []

    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## 2. 任务 A" in report_text
    assert "## 8. 验收检查" in report_text
    assert "不可用" in report_text
    assert "0.5000" in report_text

    # report_summary.json is valid JSON and mirrors the returned summary
    written_summary = json.loads(
        (run_dir / "report_summary.json").read_text(encoding="utf-8")
    )
    assert written_summary["run_id"] == "synthetic-run"
    assert written_summary["artifacts"] == summary["artifacts"]
    acceptance = json.loads((run_dir / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["passed"] is False


def test_build_report_weighted_fallback_without_cat_columns(synthetic_run):
    cfg, run_dir = synthetic_run
    frame = pd.read_parquet(run_dir / "rows_a.parquet")
    category_columns = [column for column in frame.columns if column.startswith("cat_")]
    assert category_columns
    frame.drop(columns=category_columns).to_parquet(
        run_dir / "rows_a.parquet", index=False
    )
    payload = report.build_report(cfg, "synthetic-run", n_boot=10)
    single_syllable = payload["single_syllable"]
    assert single_syllable["available"] is True
    assert single_syllable["weighted_available"] is False
    assert "cat_*_stats" in str(single_syllable["weighted_reason"])
    assert any("cat_*_stats" in warning for warning in payload["warnings"])
    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "### 6.2 类别加权对照（design §7.4）" in report_md
    assert "不可用" in report_md
    # the CSV is still written (header only) so the artifact list is stable
    category_csv = _read_csv(run_dir / "single_syllable_categories.csv")
    assert category_csv == []


def test_build_report_recomputes_cost_from_generations(synthetic_run):
    cfg, run_dir = synthetic_run
    (run_dir / "cost_summary.json").unlink()
    payload = report.build_report(cfg, "synthetic-run", n_boot=10)
    assert "generations" in str(payload["usage"]["source"])
    assert payload["usage"]["totals"]["n_calls"] == 6
    assert payload["usage"]["totals"]["cost_cny"] is not None
    assert payload["usage"]["totals"]["cost_cny"] > 0


def test_build_report_missing_run_dir_raises(synthetic_run, tmp_path):
    cfg, _run_dir = synthetic_run
    with pytest.raises(report.ReportError, match="run directory not found"):
        report.build_report(cfg, "does-not-exist")


def test_build_report_without_rows_raises(synthetic_run):
    cfg, run_dir = synthetic_run
    for name in ("rows_a.parquet", "rows_b.parquet", "rows_c.parquet"):
        (run_dir / name).unlink()
    with pytest.raises(report.ReportError, match="没有可用的评分行"):
        report.build_report(cfg, "synthetic-run", n_boot=10)
