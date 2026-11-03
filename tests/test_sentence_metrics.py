"""Tests for :mod:`pinyin_eval.sentence.metrics` (design §7, §9, §10).

Hand-computed rows keep every expected number explicit: confusion counts ->
macro-F1, Pearson/Spearman/MAE/RMSE on fixed arrays, bootstrap determinism for a
fixed seed, paired C1-vs-C2 comparisons on synthetic rows, and the acceptance
gates.
"""

from __future__ import annotations

import json
import math

import pytest

from pinyin_eval.sentence.metrics import (
    acceptance_checks,
    compare_conditions,
    task_a_metrics,
    task_b_metrics,
    task_c_metrics,
)
from pinyin_eval.sentence.records import SentenceItem
from pinyin_eval.sentence.scoring import score_b

# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def a_row(
    *,
    item_id: str = "a1",
    speaker: str = "S1",
    condition: str = "C1",
    parse_status: str = "ok",
    n_gold: int = 2,
    n_joint: int = 2,
    n_initial: int = 2,
    n_final: int = 2,
    n_tone: int = 2,
    n_matched: int | None = None,
    n_insertions: int = 0,
    n_deletions: int = 0,
    cer_tone_num: int = 0,
    cer_stripped_num: int = 0,
    sentence_exact: bool = True,
    n1_joint: int | None = None,
    n1_initial: int | None = None,
    n1_final: int | None = None,
    n1_tone: int | None = None,
    n1_cer_tone_num: int | None = None,
    flags: dict[str, bool] | None = None,
    stable: bool | None = None,
) -> dict:
    flags = dict(flags or {})
    matched = n_matched if n_matched is not None else max(0, n_gold - n_deletions)
    row = {
        "item_id": item_id,
        "task": "A",
        "condition": condition,
        "cluster_id": speaker,
        "speaker": speaker,
        "parse_status": parse_status,
        "duration_s": 1.0,
        "flags": flags,
        "stable": (not any(flags.values())) if stable is None else stable,
        "n_gold": n_gold,
        "n_hyp": n_gold - n_deletions + n_insertions,
        "n_matched": matched,
        "alignment_coverage": (matched / n_gold) if n_gold else None,
        "n_deletions": n_deletions,
        "n_insertions": n_insertions,
        "n_joint_correct": n_joint,
        "n_initial_correct": n_initial,
        "n_final_correct": n_final,
        "n_tone_correct": n_tone,
        "cer_tone_num": cer_tone_num,
        "cer_tone_den": n_gold,
        "cer_stripped_num": cer_stripped_num,
        "cer_stripped_den": n_gold,
        "sentence_exact": sentence_exact,
        "n1_n_joint_correct": n_joint if n1_joint is None else n1_joint,
        "n1_n_initial_correct": n_initial if n1_initial is None else n1_initial,
        "n1_n_final_correct": n_final if n1_final is None else n1_final,
        "n1_n_tone_correct": n_tone if n1_tone is None else n1_tone,
        "n1_cer_tone_num": cer_tone_num if n1_cer_tone_num is None else n1_cer_tone_num,
        "n1_cer_tone_den": n_gold,
        "n1_sentence_exact": bool(sentence_exact),
    }
    if parse_status != "ok":
        for key in (
            "n_joint_correct",
            "n_initial_correct",
            "n_final_correct",
            "n_tone_correct",
            "cer_tone_num",
            "cer_stripped_num",
            "n_insertions",
            "n_deletions",
            "n1_n_joint_correct",
            "n1_n_initial_correct",
            "n1_n_final_correct",
            "n1_n_tone_correct",
            "n1_cer_tone_num",
        ):
            row[key] = 0
        row["n_matched"] = 0
        row["n_hyp"] = 0
        row["sentence_exact"] = False
        row["n1_sentence_exact"] = False
    return row


def b_row(
    *,
    item_id: str = "b1",
    learner: str = "L1",
    dimension: str = "tone",
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    tn: int = 0,
    n_missing: int = 0,
    n_inserted: int = 0,
    zero_tp: int = 0,
    zero_fp: int = 0,
    zero_fn: int = 0,
    zero_tn: int = 0,
) -> dict:
    n_units = tp + fp + fn + tn
    return {
        "item_id": item_id,
        "task": "B",
        "condition": "C1",
        "cluster_id": learner,
        "learner": learner,
        "speaker": learner,
        "parse_status": "ok",
        "dimension": dimension,
        "n_units": n_units,
        "n_matched": n_units - n_missing,
        "alignment_coverage": (n_units - n_missing) / n_units if n_units else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_missing": n_missing,
        "n_inserted": n_inserted,
        "n_zero_initial_tp": zero_tp if dimension == "consonant" else 0,
        "n_zero_initial_fp": zero_fp if dimension == "consonant" else 0,
        "n_zero_initial_fn": zero_fn if dimension == "consonant" else 0,
        "n_zero_initial_tn": zero_tn if dimension == "consonant" else 0,
    }


def c_row(
    *,
    item_id: str = "c1",
    learner: str = "L1",
    gold: tuple[float, float, float] = (4.0, 4.5, 3.5),
    pred: tuple[float, float, float] | None = (4.0, 4.0, 4.0),
    native: bool = False,
    parse_status: str = "ok",
) -> dict:
    return {
        "item_id": item_id,
        "task": "C",
        "condition": "C1",
        "cluster_id": learner,
        "learner": learner,
        "speaker": learner,
        "parse_status": parse_status,
        "native": native,
        "duration_s": 2.0,
        "gold_accuracy": gold[0],
        "gold_fluency": gold[1],
        "gold_prosody": gold[2],
        "pred_accuracy": pred[0] if pred else None,
        "pred_fluency": pred[1] if pred else None,
        "pred_prosody": pred[2] if pred else None,
        "gold_mean": sum(gold) / 3.0,
        "pred_mean": (sum(pred) / 3.0) if pred else None,
    }


# ---------------------------------------------------------------------------
# Task A
# ---------------------------------------------------------------------------


def _task_a_rows() -> list[dict]:
    return [
        a_row(item_id="a1", speaker="S1", n_joint=2, n_initial=2, n_final=2, n_tone=2,
              sentence_exact=True, flags={}),
        a_row(item_id="a2", speaker="S2", n_joint=1, n_initial=2, n_final=1, n_tone=1,
              cer_tone_num=1, sentence_exact=False, n1_joint=2, n1_initial=2,
              n1_final=2, n1_tone=2, n1_cer_tone_num=0, flags={"neutral": True}),
        a_row(item_id="a3", speaker="S2", parse_status="parse_error"),
    ]


def test_task_a_headline_known_values() -> None:
    result = task_a_metrics(_task_a_rows(), n_boot=16)
    headline = result["headline"]

    assert result["n_items"] == 3
    assert result["n_parse_ok"] == 2
    assert result["n_parse_failures"] == 1
    assert result["parse_failure_rate"] == pytest.approx(1.0 / 3.0)
    assert result["n_skipped_no_gold"] == 0

    # 3 of 4 gold syllables joint-correct; initial 4/4, final 3/4, tone 3/4.
    assert headline["joint_accuracy"] == pytest.approx(0.75)
    assert headline["initial_accuracy"] == pytest.approx(1.0)
    assert headline["final_accuracy"] == pytest.approx(0.75)
    assert headline["tone_accuracy"] == pytest.approx(0.75)
    assert headline["sentence_exact_rate"] == pytest.approx(0.5)
    assert headline["insertion_rate"] == pytest.approx(0.0)
    assert headline["deletion_rate"] == pytest.approx(0.0)
    assert headline["cer_tone"] == pytest.approx(0.25)
    assert headline["cer_stripped"] == pytest.approx(0.0)
    assert headline["alignment_coverage"] == pytest.approx(1.0)


def test_task_a_n1_block_and_delta() -> None:
    result = task_a_metrics(_task_a_rows(), n_boot=16)
    n1 = result["n1"]

    assert n1["n1_joint_accuracy"] == pytest.approx(1.0)
    assert n1["n1_initial_accuracy"] == pytest.approx(1.0)
    assert n1["n1_final_accuracy"] == pytest.approx(1.0)
    assert n1["n1_tone_accuracy"] == pytest.approx(1.0)
    assert n1["n1_cer_tone"] == pytest.approx(0.0)
    assert n1["delta_vs_n0"]["joint_accuracy"] == pytest.approx(0.25)
    assert n1["delta_vs_n0"]["cer_tone"] == pytest.approx(-0.25)


def test_task_a_subsets() -> None:
    result = task_a_metrics(_task_a_rows(), n_boot=16)
    subsets = result["subsets"]

    assert set(subsets) == {"stable", "neutral", "yibu", "erhua"}
    stable = subsets["stable"]
    # a3 is stable but parse-failed; a2 is not stable (neutral flag).
    assert stable["n_items"] == 2
    assert stable["n_parse_ok"] == 1
    assert stable["joint_accuracy"] == pytest.approx(1.0)

    neutral = subsets["neutral"]
    assert neutral["n_items"] == 1
    assert neutral["n_parse_ok"] == 1
    assert neutral["joint_accuracy"] == pytest.approx(0.5)

    erhua = subsets["erhua"]
    assert erhua["n_items"] == 0
    assert erhua["joint_accuracy"] is None


def test_task_a_per_speaker() -> None:
    per_speaker = task_a_metrics(_task_a_rows(), n_boot=16)["per_speaker"]
    assert [entry["speaker"] for entry in per_speaker] == ["S1", "S2"]
    by_speaker = {entry["speaker"]: entry for entry in per_speaker}
    assert by_speaker["S1"]["n_items"] == 1
    assert by_speaker["S1"]["joint_accuracy"] == pytest.approx(1.0)
    assert by_speaker["S2"]["n_items"] == 1
    assert by_speaker["S2"]["joint_accuracy"] == pytest.approx(0.5)
    assert by_speaker["S2"]["initial_accuracy"] == pytest.approx(1.0)


def test_task_a_bootstrap_ci_deterministic_for_fixed_seed() -> None:
    rows = _task_a_rows()
    first = task_a_metrics(rows, n_boot=64, seed=7)
    second = task_a_metrics(rows, n_boot=64, seed=7)
    assert first == second
    assert set(first["with_ci"]) == set(first["headline"])
    joint_ci = first["with_ci"]["joint_accuracy"]
    assert joint_ci["estimate"] == pytest.approx(first["headline"]["joint_accuracy"])
    assert joint_ci["lo"] <= joint_ci["estimate"] <= joint_ci["hi"]
    cer_ci = first["with_ci"]["cer_tone"]
    assert cer_ci["lo"] <= cer_ci["estimate"] <= cer_ci["hi"]


def test_task_a_parse_failures_do_not_enter_denominators() -> None:
    rows = _task_a_rows()
    baseline = task_a_metrics(rows, n_boot=8)["headline"]
    rows.append(a_row(item_id="a4", speaker="S3", parse_status="refused"))
    with_failure = task_a_metrics(rows, n_boot=8)
    assert with_failure["headline"] == baseline
    assert with_failure["n_parse_failures"] == 2


def test_task_a_empty_input() -> None:
    result = task_a_metrics([], n_boot=8)
    assert result["n_items"] == 0
    assert result["parse_failure_rate"] is None
    assert result["headline"]["joint_accuracy"] is None
    assert result["with_ci"]["joint_accuracy"]["estimate"] is None
    assert result["per_speaker"] == []


def test_task_a_no_gold_rows_are_counted() -> None:
    row = a_row(item_id="a1", n_gold=0, n_joint=0, n_initial=0, n_final=0, n_tone=0,
                n_matched=0, sentence_exact=False)
    result = task_a_metrics([row], n_boot=8)
    assert result["n_skipped_no_gold"] == 1
    assert result["headline"]["joint_accuracy"] is None


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------


def test_task_b_known_confusion_and_macro_f1() -> None:
    rows = [
        b_row(item_id="b1", dimension="tone", tp=1, fp=1, fn=1, tn=1),
        b_row(item_id="b1", dimension="consonant", tp=0, fp=0, fn=0, tn=4),
        b_row(item_id="b1", dimension="vowel", tp=0, fp=0, fn=0, tn=4),
    ]
    result = task_b_metrics(rows, n_boot=8)

    tone = result["tone"]
    assert tone["n_units"] == 4
    assert tone["n_items"] == 1
    assert tone["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert tone["f1_error"] == pytest.approx(0.5)
    assert tone["f1_ok"] == pytest.approx(0.5)
    assert tone["macro_f1"] == pytest.approx(0.5)
    assert tone["fpr"] == pytest.approx(0.5)
    assert tone["fnr"] == pytest.approx(0.5)
    assert tone["accuracy"] == pytest.approx(0.5)

    # All-correct dimensions: the error class has no positive support/predictions,
    # so its F1 is undefined (None) and is skipped in the macro average
    # (sklearn-on-observed-labels behaviour).
    consonant = result["consonant"]
    assert consonant["macro_f1"] == pytest.approx(1.0)
    assert consonant["f1_error"] is None
    assert consonant["confusion"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 4}
    assert result["alignment_coverage"] == pytest.approx(1.0)


def test_task_b_aggregates_confusion_across_rows() -> None:
    rows = [
        b_row(item_id="b1", dimension="tone", tp=1, fp=0, fn=0, tn=1),
        b_row(item_id="b2", dimension="tone", tp=1, fp=1, fn=1, tn=1),
        b_row(item_id="b1", dimension="consonant", tp=0, fp=0, fn=0, tn=1),
    ]
    tone = task_b_metrics(rows, n_boot=8)["tone"]
    assert tone["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert tone["f1_error"] == pytest.approx(2.0 / 3.0)
    assert tone["f1_ok"] == pytest.approx(2.0 / 3.0)
    assert tone["macro_f1"] == pytest.approx(2.0 / 3.0)
    assert tone["fpr"] == pytest.approx(1.0 / 3.0)
    assert tone["fnr"] == pytest.approx(1.0 / 3.0)
    assert tone["accuracy"] == pytest.approx(4.0 / 6.0)
    assert tone["n_items"] == 2


def test_task_b_zero_initial_consonant_block() -> None:
    rows = [
        b_row(item_id="b1", dimension="consonant", tp=0, fp=0, fn=0, tn=2,
              zero_tp=1, zero_fp=1, zero_fn=1, zero_tn=0),
        b_row(item_id="b1", dimension="tone", tp=0, fp=0, fn=0, tn=1),
    ]
    zero = task_b_metrics(rows, n_boot=8)["zero_initial_consonant"]
    assert zero["n_units"] == 3
    assert zero["n_items"] == 1
    assert zero["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
    assert zero["f1_error"] == pytest.approx(0.5)
    assert zero["f1_ok"] == pytest.approx(0.0)
    assert zero["macro_f1"] == pytest.approx(0.25)
    assert zero["fpr"] == pytest.approx(1.0)
    assert zero["fnr"] == pytest.approx(0.5)


def test_task_b_with_ci_deterministic_and_per_learner() -> None:
    rows = [
        b_row(item_id=f"b{index}", learner=learner, dimension=dimension, tp=tp, tn=tn,
              fp=fp, fn=fn)
        for index, (learner, (tp, fp, fn, tn)) in enumerate(
            (("L1", (2, 1, 1, 2)), ("L2", (1, 0, 0, 3)))
        )
        for dimension in ("tone", "consonant", "vowel")
    ]
    first = task_b_metrics(rows, n_boot=32, seed=5)
    second = task_b_metrics(rows, n_boot=32, seed=5)
    assert first == second
    assert set(first["with_ci"]) == {"tone", "consonant", "vowel"}
    ci = first["with_ci"]["tone"]["macro_f1"]
    assert ci["estimate"] == pytest.approx(first["tone"]["macro_f1"])
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]

    learners = {entry["learner"] for entry in first["per_learner"]}
    assert learners == {"L1", "L2"}
    l1 = next(entry for entry in first["per_learner"] if entry["learner"] == "L1")
    # L1 tone: tp/fp/fn/tn = 2,1,1,2 in two items -> the aggregate 2/3 macro-F1.
    assert l1["macro_f1"]["tone"] == pytest.approx(2.0 / 3.0)
    assert l1["f1_error"]["tone"] == pytest.approx(2.0 / 3.0)


def test_task_b_empty_input() -> None:
    result = task_b_metrics([], n_boot=8)
    assert result["n_items"] == 0
    assert result["tone"]["n_units"] == 0
    assert result["tone"]["macro_f1"] is None
    assert result["alignment_coverage"] is None
    assert result["per_learner"] == []


# ---------------------------------------------------------------------------
# Task C
# ---------------------------------------------------------------------------


def _task_c_rows() -> list[dict]:
    golds = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)]
    preds = [(2, 2, 2), (2, 2, 2), (4, 4, 4), (4, 4, 4)]
    return [
        c_row(item_id=f"c{index}", learner="L1", gold=gold, pred=pred)
        for index, (gold, pred) in enumerate(zip(golds, preds, strict=True))
    ]


def test_task_c_known_correlations_and_errors() -> None:
    result = task_c_metrics(_task_c_rows(), n_boot=8)

    expected_pearson = 4.0 / math.sqrt(20.0)  # cov 4, var gold 5, var pred 4
    expected_spearman = 4.0 / math.sqrt(20.0)  # tie-corrected average ranks: same
    for dimension in ("accuracy", "fluency", "prosody", "mean_dimension"):
        stats = result[dimension]
        assert stats["n"] == 4
        assert stats["pearson"] == pytest.approx(expected_pearson)
        assert stats["spearman"] == pytest.approx(expected_spearman)
        assert stats["mae"] == pytest.approx(0.5)
        assert stats["rmse"] == pytest.approx(math.sqrt(0.5))

    assert result["n_items"] == 4
    assert result["n_parse_ok"] == 4
    assert result["n_parse_failures"] == 0
    assert result["alignment_coverage"] == pytest.approx(1.0)
    assert result["native_anchor"] == {
        "n": 0,
        "mean_gold": None,
        "mean_pred": None,
        "per_dimension": {
            "accuracy": {"mean_gold": None, "mean_pred": None},
            "fluency": {"mean_gold": None, "mean_pred": None},
            "prosody": {"mean_gold": None, "mean_pred": None},
        },
    }


def test_task_c_native_items_are_excluded_and_anchored() -> None:
    rows = [
        c_row(item_id="c1", learner="L1", gold=(3, 3, 3), pred=(2, 2, 2)),
        c_row(item_id="c2", learner="L1", gold=(4, 4, 4), pred=(5, 5, 5)),
        c_row(item_id="n1", learner="NATIVE", gold=(5, 5, 5), pred=(4, 4, 4), native=True),
    ]
    result = task_c_metrics(rows, n_boot=8)

    accuracy = result["accuracy"]
    assert accuracy["n"] == 2  # native row excluded
    assert accuracy["pearson"] == pytest.approx(1.0)
    assert accuracy["mae"] == pytest.approx(1.0)
    anchor = result["native_anchor"]
    assert anchor["n"] == 1
    assert anchor["mean_gold"] == pytest.approx(5.0)  # backwards-compatible key
    assert anchor["mean_pred"] == pytest.approx(4.0)
    for dimension in ("accuracy", "fluency", "prosody"):
        assert anchor["per_dimension"][dimension]["mean_gold"] == pytest.approx(5.0)
        assert anchor["per_dimension"][dimension]["mean_pred"] == pytest.approx(4.0)


def test_task_c_native_anchor_per_dimension_and_partial_parse() -> None:
    rows = [
        c_row(item_id="n1", learner="NATIVE", gold=(5.0, 4.8, 4.6),
              pred=(4.5, 4.0, 4.1), native=True),
        c_row(item_id="n2", learner="NATIVE", gold=(5.0, 5.0, 5.0),
              pred=None, native=True, parse_status="parse_error"),
        c_row(item_id="c1", learner="L1", gold=(2, 2, 2), pred=(2, 2, 2)),
    ]
    anchor = task_c_metrics(rows, n_boot=8)["native_anchor"]

    assert anchor["n"] == 2  # both native rows counted, including the parse failure
    # mean_gold: both native rows have complete gold (5.0 and (5+4.8+4.6)/3 = 4.8).
    assert anchor["mean_gold"] == pytest.approx((5.0 + (5.0 + 4.8 + 4.6) / 3.0) / 2.0)
    # mean_pred: only the parsed native row contributes.
    assert anchor["mean_pred"] == pytest.approx((4.5 + 4.0 + 4.1) / 3.0)
    assert anchor["per_dimension"]["accuracy"]["mean_gold"] == pytest.approx(5.0)
    assert anchor["per_dimension"]["accuracy"]["mean_pred"] == pytest.approx(4.5)
    assert anchor["per_dimension"]["fluency"]["mean_gold"] == pytest.approx((4.8 + 5.0) / 2.0)
    assert anchor["per_dimension"]["fluency"]["mean_pred"] == pytest.approx(4.0)
    assert anchor["per_dimension"]["prosody"]["mean_gold"] == pytest.approx((4.6 + 5.0) / 2.0)
    assert anchor["per_dimension"]["prosody"]["mean_pred"] == pytest.approx(4.1)


def test_task_c_zero_variance_returns_none() -> None:
    rows = [
        c_row(item_id="c1", gold=(4, 4, 4), pred=(3, 3, 3)),
        c_row(item_id="c2", gold=(4, 4, 4), pred=(5, 5, 5)),
    ]
    accuracy = task_c_metrics(rows, n_boot=8)["accuracy"]
    assert accuracy["n"] == 2
    assert accuracy["pearson"] is None
    assert accuracy["spearman"] is None
    assert accuracy["mae"] == pytest.approx(1.0)


def test_task_c_parse_failures_excluded_from_denominator() -> None:
    rows = _task_c_rows() + [c_row(item_id="bad", pred=None, parse_status="parse_error")]
    result = task_c_metrics(rows, n_boot=8)
    assert result["n_items"] == 5
    assert result["n_parse_ok"] == 4
    assert result["n_parse_failures"] == 1
    assert result["parse_failure_rate"] == pytest.approx(0.2)
    assert result["accuracy"]["n"] == 4
    assert result["alignment_coverage"] == pytest.approx(1.0)


def test_task_c_with_ci_deterministic_and_per_learner() -> None:
    rows = _task_c_rows()
    for row in rows:
        row["learner"] = "L2" if row["item_id"] in {"c2", "c3"} else "L1"
        row["cluster_id"] = row["learner"]
    first = task_c_metrics(rows, n_boot=32, seed=9)
    second = task_c_metrics(rows, n_boot=32, seed=9)
    assert first == second
    ci = first["with_ci"]["accuracy"]
    assert ci["pearson"]["estimate"] == pytest.approx(first["accuracy"]["pearson"])
    assert ci["pearson"]["lo"] <= ci["pearson"]["estimate"] <= ci["pearson"]["hi"]
    assert ci["mae"]["estimate"] == pytest.approx(first["accuracy"]["mae"])

    learners = {entry["learner"] for entry in first["per_learner"]}
    assert learners == {"L1", "L2"}
    l1 = next(entry for entry in first["per_learner"] if entry["learner"] == "L1")
    assert l1["n_items"] == 2


def test_task_c_empty_input() -> None:
    result = task_c_metrics([], n_boot=8)
    assert result["n_items"] == 0
    assert result["accuracy"]["n"] == 0
    assert result["accuracy"]["pearson"] is None
    anchor = result["native_anchor"]
    assert anchor["n"] == 0
    assert anchor["mean_gold"] is None
    assert anchor["mean_pred"] is None
    assert anchor["per_dimension"]["accuracy"] == {"mean_gold": None, "mean_pred": None}


# ---------------------------------------------------------------------------
# compare_conditions
# ---------------------------------------------------------------------------


def _task_a_condition_rows(condition: str, exact: list[bool], joints: list[int]) -> list[dict]:
    rows = []
    for index, (is_exact, joint) in enumerate(zip(exact, joints, strict=True)):
        rows.append(
            a_row(
                item_id=f"a{index}",
                speaker=f"S{index % 2}",
                condition=condition,
                n_joint=joint,
                n_initial=2,
                n_final=2,
                n_tone=joint,
                sentence_exact=is_exact,
            )
        )
    return rows


def test_compare_conditions_a_known_values() -> None:
    rows_c1 = _task_a_condition_rows("C1", [True, True, False], [2, 2, 1])
    rows_c2 = _task_a_condition_rows("C2", [True, False, False], [2, 1, 1])
    rows_c1.append(a_row(item_id="a9", speaker="S9", condition="C1", parse_status="api_error"))
    rows_c2.append(a_row(item_id="a9", speaker="S9", condition="C2", parse_status="api_error"))

    result = compare_conditions(rows_c1, rows_c2, "A", n_boot=32, seed=11)

    assert result["task"] == "A"
    assert result["direction"] == "C1 - C2"
    assert result["n_items_c1"] == 4
    assert result["n_items_c2"] == 4
    assert result["n_items_paired"] == 3
    assert set(result["differences"]) == {
        "joint_accuracy", "initial_accuracy", "final_accuracy", "tone_accuracy",
        "sentence_exact_rate", "insertion_rate", "deletion_rate", "cer_tone",
        "cer_stripped", "alignment_coverage",
    }
    for entry in result["differences"].values():
        assert entry["p_value"] is None
    # C1 joint: 5/6, C2 joint: 4/6 -> difference 1/6.
    assert result["differences"]["joint_accuracy"]["estimate"] == pytest.approx(1.0 / 6.0)
    assert result["differences"]["sentence_exact_rate"]["estimate"] == pytest.approx(1.0 / 3.0)

    # sentence_exact pairs: (T,T), (T,F), (F,F) -> b=1, c=0.
    mcnemar = result["mcnemar"]["sentence_exact"]
    assert mcnemar["n_pairs"] == 3
    assert (mcnemar["b"], mcnemar["c"]) == (1, 0)
    assert mcnemar["p_value"] == pytest.approx(1.0)  # exact binomial with 1 discordant pair
    assert result["holm"]["sentence_exact"] == pytest.approx(1.0)

    counts = result["per_item_joint_counts"]
    assert counts["n_items"] == 3
    assert counts["joint_all_correct_c1"] == 2
    assert counts["joint_all_correct_c2"] == 1
    assert counts["both"] == 1
    assert counts["c1_only"] == 1
    assert counts["c2_only"] == 0
    assert counts["neither"] == 1


def test_compare_conditions_a_item_set_mismatch_raises() -> None:
    rows_c1 = _task_a_condition_rows("C1", [True, True, False], [2, 2, 1])
    rows_c2 = _task_a_condition_rows("C2", [True, True, False], [2, 2, 1])[:2]
    with pytest.raises(ValueError, match="item_id sets differ"):
        compare_conditions(rows_c1, rows_c2, "A", n_boot=8)


def test_compare_conditions_b_paired_macro_f1_and_missing_items() -> None:
    item_labels = [
        [{"text": "甲", "tone": 0, "consonant": 1, "vowel": 1, "zero_initial": False}],
        [{"text": "乙", "tone": 0, "consonant": 0, "vowel": 1, "zero_initial": False}],
        [{"text": "丙", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False}],
    ]
    item_words = [
        [{"text": "甲", "tone_ok": False, "consonant_ok": True, "vowel_ok": True}],
        [{"text": "乙", "tone_ok": True, "consonant_ok": False, "vowel_ok": True}],
        [{"text": "丙", "tone_ok": True, "consonant_ok": True, "vowel_ok": True}],
    ]

    def condition_rows(condition: str, worse: bool) -> list[dict]:
        rows: list[dict] = []
        for index, (labels, words) in enumerate(zip(item_labels, item_words, strict=True)):
            parsed_words = [dict(word) for word in words]
            if worse and index == 1:
                parsed_words[0]["consonant_ok"] = True  # missed error -> fn
            item = SentenceItem(
                item_id=f"b{index}",
                task="B",
                source="ompal",
                speaker=f"L{index}",
                audio_path="",
                duration_s=1.0,
                text="".join(unit["text"] for unit in labels),
                labels=[dict(unit) for unit in labels],
                learner=f"L{index}",
            )
            rows.extend(
                score_b(item, {"parse_status": "ok", "words": parsed_words}, condition)
            )
        return rows

    rows_c1 = condition_rows("C1", worse=False)
    rows_c2 = condition_rows("C2", worse=True)
    result = compare_conditions(rows_c1, rows_c2, "B", n_boot=32, seed=3)

    assert result["task"] == "B"
    assert result["n_items_paired"] == 3
    assert result["n_items_missing_in_c1"] == 0
    assert result["n_items_missing_in_c2"] == 0
    assert set(result["differences"]["macro_f1"]) == {"tone", "consonant", "vowel"}

    metrics_c1 = task_b_metrics(rows_c1, n_boot=4)
    metrics_c2 = task_b_metrics(rows_c2, n_boot=4)
    for dimension in ("tone", "consonant", "vowel"):
        expected = metrics_c1[dimension]["macro_f1"] - metrics_c2[dimension]["macro_f1"]
        assert result["differences"]["macro_f1"][dimension]["estimate"] == pytest.approx(expected)
        assert result["differences"]["macro_f1"][dimension]["p_value"] is None
    # C2 misses the consonant error on 乙 -> lower consonant macro-F1.
    assert result["differences"]["macro_f1"]["consonant"]["estimate"] > 0

    # A condition that lost an item on parse failure: intersection is used.
    trimmed = [row for row in rows_c2 if row["item_id"] != "b2"]
    missing = compare_conditions(rows_c1, trimmed, "B", n_boot=8, seed=3)
    assert missing["n_items_missing_in_c2"] == 1


def test_compare_conditions_c_known_values() -> None:
    def condition_rows(condition: str, offset: float) -> list[dict]:
        rows = []
        for index in range(4):
            row = c_row(
                item_id=f"c{index}",
                learner=f"L{index % 2}",
                gold=(float(index + 1),) * 3,
                pred=(float(index + 1) + offset,) * 3,
            )
            row["condition"] = condition
            rows.append(row)
        return rows

    rows_c1 = condition_rows("C1", offset=0.0)
    rows_c2 = condition_rows("C2", offset=0.5)
    result = compare_conditions(rows_c1, rows_c2, "C", n_boot=32, seed=2)

    assert result["task"] == "C"
    assert result["n_items_paired"] == 4
    assert set(result["differences"]) == {"pearson", "mae"}
    assert set(result["differences"]["pearson"]) == {
        "accuracy", "fluency", "prosody", "mean_dimension",
    }
    metrics_c1 = task_c_metrics(rows_c1, n_boot=4)
    metrics_c2 = task_c_metrics(rows_c2, n_boot=4)
    for dimension in ("accuracy", "fluency", "prosody", "mean_dimension"):
        expected_pearson = metrics_c1[dimension]["pearson"] - metrics_c2[dimension]["pearson"]
        expected_mae = metrics_c1[dimension]["mae"] - metrics_c2[dimension]["mae"]
        assert result["differences"]["pearson"][dimension]["estimate"] == pytest.approx(
            expected_pearson
        )
        assert result["differences"]["mae"][dimension]["estimate"] == pytest.approx(expected_mae)
    assert result["differences"]["pearson"]["accuracy"]["p_value"] is None
    # C1 is perfectly correlated with gold, C2 is shifted by a constant -> same r
    # but higher MAE; the MAE difference must be negative (C1 error is smaller).
    assert result["differences"]["mae"]["accuracy"]["estimate"] == pytest.approx(-0.5)


def test_compare_conditions_c_item_set_mismatch_raises() -> None:
    rows_c1 = [c_row(item_id="c1"), c_row(item_id="c2")]
    rows_c2 = [c_row(item_id="c1")]
    with pytest.raises(ValueError, match="item_id sets differ"):
        compare_conditions(rows_c1, rows_c2, "C", n_boot=8)


def test_compare_conditions_c_filters_native_rows_on_both_sides() -> None:
    def learner_rows(condition: str) -> list[dict]:
        rows = []
        for index in range(4):
            row = c_row(
                item_id=f"c{index}",
                learner=f"L{index % 2}",
                gold=(float(index + 1),) * 3,
                pred=(float(index + 1),) * 3,
            )
            row["condition"] = condition
            rows.append(row)
        return rows

    def native_row(condition: str, pred: tuple[float, float, float]) -> dict:
        row = c_row(
            item_id="n1",
            learner="NATIVE",
            gold=(5.0, 5.0, 5.0),
            pred=pred,
            native=True,
        )
        row["condition"] = condition
        return row

    # The learner rows are identical in both conditions; the native predictions
    # differ wildly and must be ignored on both sides of the paired comparison.
    rows_c1 = learner_rows("C1") + [native_row("C1", (1.0, 1.0, 1.0))]
    rows_c2 = learner_rows("C2") + [native_row("C2", (5.0, 5.0, 5.0))]

    result = compare_conditions(rows_c1, rows_c2, "C", n_boot=16, seed=1)

    assert result["n_items_paired"] == 5  # the native item is paired, then filtered
    for dimension in ("accuracy", "fluency", "prosody", "mean_dimension"):
        assert result["differences"]["pearson"][dimension]["estimate"] == pytest.approx(0.0)
        assert result["differences"]["mae"][dimension]["estimate"] == pytest.approx(0.0)


def test_compare_conditions_deterministic() -> None:
    rows_c1 = _task_a_condition_rows("C1", [True, False, True], [2, 1, 2])
    rows_c2 = _task_a_condition_rows("C2", [False, False, True], [1, 1, 2])
    first = compare_conditions(rows_c1, rows_c2, "A", n_boot=32, seed=4)
    second = compare_conditions(rows_c1, rows_c2, "A", n_boot=32, seed=4)
    assert first == second


def test_compare_conditions_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="unknown sentence task"):
        compare_conditions([], [], "Z", n_boot=8)


# ---------------------------------------------------------------------------
# acceptance_checks
# ---------------------------------------------------------------------------


def _metrics_with_coverage(a: float, b: float, c: float) -> tuple[dict, dict, dict]:
    return (
        {"headline": {"alignment_coverage": a}},
        {"alignment_coverage": b},
        {"alignment_coverage": c},
    )


def _usage(
    *, calls: int = 100, errors: int = 1, parse_failures: int = 1, cost: float = 1.0
) -> dict:
    return {
        task: {
            "n_calls": calls,
            "n_errors": errors,
            "n_parse_failures": parse_failures,
            "cost_cny": cost,
        }
        for task in ("A", "B", "C")
    }


def test_acceptance_checks_pass() -> None:
    a_metrics, b_metrics, c_metrics = _metrics_with_coverage(0.99, 0.98, 0.97)
    result = acceptance_checks(a_metrics, b_metrics, c_metrics, _usage())

    assert result["passed"] is True
    assert result["parse_success"]["A"]["rate"] == pytest.approx(0.98)
    assert result["parse_success"]["A"]["threshold"] == pytest.approx(0.95)
    assert result["parse_success"]["A"]["passed"] is True
    alignment_a = result["alignment_coverage"]["A"]
    assert alignment_a["value"] == pytest.approx(0.99)
    assert alignment_a["threshold"] == pytest.approx(0.95)
    assert alignment_a["passed"] is True
    assert result["cost_cny"]["value"] == pytest.approx(3.0)
    assert result["cost_cny"]["passed"] is True


def test_acceptance_checks_parse_success_failure() -> None:
    a_metrics, b_metrics, c_metrics = _metrics_with_coverage(1.0, 1.0, 1.0)
    usage = _usage(errors=10, parse_failures=0)  # 90% parse success
    result = acceptance_checks(a_metrics, b_metrics, c_metrics, usage)
    assert result["parse_success"]["A"]["rate"] == pytest.approx(0.9)
    assert result["parse_success"]["A"]["passed"] is False
    assert result["passed"] is False


def test_acceptance_checks_alignment_failure() -> None:
    a_metrics, b_metrics, c_metrics = _metrics_with_coverage(0.90, 1.0, 1.0)
    result = acceptance_checks(a_metrics, b_metrics, c_metrics, _usage())
    assert result["alignment_coverage"]["A"]["passed"] is False
    assert result["passed"] is False


def test_acceptance_checks_budget_failure() -> None:
    a_metrics, b_metrics, c_metrics = _metrics_with_coverage(1.0, 1.0, 1.0)
    usage = _usage(cost=8.0)  # 3 * 8 = 24 > 20
    result = acceptance_checks(a_metrics, b_metrics, c_metrics, usage)
    assert result["cost_cny"]["value"] == pytest.approx(24.0)
    assert result["cost_cny"]["passed"] is False
    assert result["passed"] is False


def test_acceptance_checks_handles_missing_inputs() -> None:
    result = acceptance_checks(None, None, None, None)
    assert result["passed"] is False
    assert result["parse_success"]["A"]["rate"] is None
    assert result["alignment_coverage"]["A"]["passed"] is False
    assert result["cost_cny"]["value"] is None


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------


def test_metric_blocks_are_json_serialisable() -> None:
    payload = {
        "a": task_a_metrics(_task_a_rows(), n_boot=8),
        "b": task_b_metrics([b_row(dimension="tone", tp=1, tn=1)], n_boot=8),
        "c": task_c_metrics(_task_c_rows(), n_boot=8),
    }
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert '"joint_accuracy"' in text
