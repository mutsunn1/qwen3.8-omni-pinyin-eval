"""Tests for :mod:`pinyin_eval.stats` (contract section 7).

Hand-computed cases for the exact binomial McNemar test, Holm and BH corrections;
cluster-bootstrap behaviour is checked through synthetic data whose resampling
distribution is known exactly (all bootstrap draws must lie in a small set of
attainable values).
"""

from __future__ import annotations

import json
import math

import pytest

from pinyin_eval import stats


def _accuracy(rows: list[dict]) -> dict:
    """Metric function: accuracy over rows whose ``correct`` value is known."""
    valid = [row for row in rows if row.get("correct") is not None]
    if not valid:
        return {"accuracy": None}
    return {"accuracy": sum(1 for row in valid if row["correct"]) / len(valid)}


def _accuracy_and_n(rows: list[dict]) -> dict:
    result = _accuracy(rows)
    result["n"] = float(len(rows))
    return result


def _mean_value(rows: list[dict]) -> dict:
    return {"mean": sum(row["value"] for row in rows) / len(rows)}


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def _accuracy_rows() -> list[dict]:
    return [
        {"cluster_id": "c1", "voice": "A", "item_id": "i1", "correct": True},
        {"cluster_id": "c1", "voice": "A", "item_id": "i2", "correct": False},
        {"cluster_id": "c2", "voice": "A", "item_id": "i3", "correct": True},
        {"cluster_id": "c2", "voice": "A", "item_id": "i4", "correct": True},
        {"cluster_id": "c3", "voice": "A", "item_id": "i5", "correct": False},
        {"cluster_id": "c3", "voice": "A", "item_id": "i6", "correct": True},
    ]


def test_bootstrap_ci_basic_and_reproducible() -> None:
    rows = _accuracy_rows()
    result = stats.bootstrap_ci(rows, _accuracy, n_boot=200, seed=7)

    assert list(result) == ["accuracy"]
    assert set(result["accuracy"]) == {"estimate", "lo", "hi"}
    assert result["accuracy"]["estimate"] == pytest.approx(4.0 / 6.0)
    assert result["accuracy"]["lo"] <= result["accuracy"]["estimate"]
    assert result["accuracy"]["estimate"] <= result["accuracy"]["hi"]
    assert 0.0 <= result["accuracy"]["lo"] <= result["accuracy"]["hi"] <= 1.0
    json.dumps(result)

    again = stats.bootstrap_ci(rows, _accuracy, n_boot=200, seed=7)
    assert again == result


def test_bootstrap_ci_resamples_clusters_with_duplicates() -> None:
    # Two clusters of unequal size: a draw that takes the big cluster twice must
    # contribute its rows twice (rows of a cluster are never split).
    rows = [
        {"cluster_id": "small", "voice": "A", "item_id": "s1"},
        {"cluster_id": "big", "voice": "A", "item_id": "b1"},
        {"cluster_id": "big", "voice": "A", "item_id": "b2"},
        {"cluster_id": "big", "voice": "A", "item_id": "b3"},
    ]
    result = stats.bootstrap_ci(
        rows, lambda sample: {"n": float(len(sample))}, n_boot=500, seed=0
    )
    assert result["n"]["estimate"] == pytest.approx(4.0)
    # both clusters drawn twice -> 2 * 1 or 2 * 3 rows; the 2.5% / 97.5% tails
    assert result["n"]["lo"] == pytest.approx(2.0)
    assert result["n"]["hi"] == pytest.approx(6.0)


def test_bootstrap_ci_respects_cluster_correlation() -> None:
    # Two clusters, four identical rows each.  Cluster bootstrap can only produce
    # the means {0.25, 0.5, 0.75}; row-level resampling would fill in e.g. 0.375.
    rows = []
    for cluster_id, value in (("c1", 0.25), ("c2", 0.75)):
        for index in range(4):
            rows.append(
                {"cluster_id": cluster_id, "voice": "A", "item_id": f"{cluster_id}-{index}", "value": value}
            )
    result = stats.bootstrap_ci(rows, _mean_value, n_boot=1000, seed=0)
    assert result["mean"]["estimate"] == pytest.approx(0.5)
    assert result["mean"]["lo"] == pytest.approx(0.25)
    assert result["mean"]["hi"] == pytest.approx(0.75)


def _stratified_rows() -> list[dict]:
    rows = []
    for cluster_id, voice in (("a1", "A"), ("a2", "A"), ("b1", "B"), ("b2", "B")):
        for index in range(4):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "voice": voice,
                    "item_id": f"{cluster_id}-{index}",
                }
            )
    return rows


def _share_b(rows: list[dict]) -> dict:
    return {"share_b": sum(1 for row in rows if row["voice"] == "B") / len(rows)}


def test_bootstrap_ci_stratification_keeps_voice_balance() -> None:
    rows = _stratified_rows()
    # Each stratum keeps its observed number of clusters (2), so the row share of
    # voice B is exactly 0.5 in every single resample.
    stratified = stats.bootstrap_ci(
        rows, _share_b, n_boot=200, seed=0, stratify_key="voice"
    )
    assert stratified["share_b"]["estimate"] == pytest.approx(0.5)
    assert stratified["share_b"]["lo"] == pytest.approx(0.5)
    assert stratified["share_b"]["hi"] == pytest.approx(0.5)

    # Without stratification the four clusters are resampled jointly: all-A or
    # all-B draws occur with probability 1/16 each, so the 2.5% / 97.5% interval
    # reaches both extremes (deterministically for this seed).
    unstratified = stats.bootstrap_ci(
        rows, _share_b, n_boot=1000, seed=0, stratify_key=None
    )
    assert unstratified["share_b"]["estimate"] == pytest.approx(0.5)
    assert unstratified["share_b"]["lo"] == pytest.approx(0.0)
    assert unstratified["share_b"]["hi"] == pytest.approx(1.0)


def test_bootstrap_ci_skips_undefined_resamples() -> None:
    rows = [
        {"cluster_id": "c1", "voice": "A", "item_id": "i1", "correct": True},
        {"cluster_id": "c1", "voice": "A", "item_id": "i2", "correct": True},
        {"cluster_id": "c2", "voice": "A", "item_id": "i3", "correct": None},
        {"cluster_id": "c2", "voice": "A", "item_id": "i4", "correct": None},
    ]
    result = stats.bootstrap_ci(rows, _accuracy, n_boot=200, seed=0)
    # resamples made only of the failed cluster have accuracy None -> skipped
    assert result["accuracy"]["estimate"] == pytest.approx(1.0)
    assert result["accuracy"]["lo"] == pytest.approx(1.0)
    assert result["accuracy"]["hi"] == pytest.approx(1.0)


def test_bootstrap_ci_all_undefined_and_empty_rows() -> None:
    rows = [{"cluster_id": "c1", "voice": "A", "item_id": "i1", "correct": None}]
    result = stats.bootstrap_ci(rows, _accuracy, n_boot=50, seed=0)
    assert result == {"accuracy": {"estimate": None, "lo": None, "hi": None}}

    empty = stats.bootstrap_ci([], _accuracy, n_boot=50, seed=0)
    assert empty == {"accuracy": {"estimate": None, "lo": None, "hi": None}}


def test_bootstrap_ci_validates_arguments_and_metric_output() -> None:
    rows = _accuracy_rows()
    with pytest.raises(ValueError, match="n_boot"):
        stats.bootstrap_ci(rows, _accuracy, n_boot=0)
    with pytest.raises(ValueError, match="alpha"):
        stats.bootstrap_ci(rows, _accuracy, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        stats.bootstrap_ci(rows, _accuracy, alpha=1.0)
    with pytest.raises(TypeError, match="metric_fn"):
        stats.bootstrap_ci(rows, lambda sample: ["not-a-mapping"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# paired_bootstrap_difference
# ---------------------------------------------------------------------------


def _paired_accuracy_rows() -> tuple[list[dict], list[dict]]:
    rows_a: list[dict] = []
    rows_b: list[dict] = []
    for cluster in range(4):
        for index in range(2):
            item_id = f"i{cluster}{index}"
            rows_a.append(
                {"item_id": item_id, "cluster_id": f"c{cluster}", "voice": "A", "correct": True}
            )
            rows_b.append(
                {
                    "item_id": item_id,
                    "cluster_id": f"c{cluster}",
                    "voice": "A",
                    "correct": index == 0,
                }
            )
    return rows_a, rows_b


def test_paired_bootstrap_difference_hand_computable() -> None:
    rows_a, rows_b = _paired_accuracy_rows()
    result = stats.paired_bootstrap_difference(rows_a, rows_b, _accuracy_and_n, n_boot=300)
    # A is 100% accurate, B is exactly 50% accurate in every cluster: the paired
    # difference is 0.5 in every resample.
    assert result["accuracy"]["estimate"] == pytest.approx(0.5)
    assert result["accuracy"]["lo"] == pytest.approx(0.5)
    assert result["accuracy"]["hi"] == pytest.approx(0.5)
    assert result["n"]["estimate"] == pytest.approx(0.0)
    assert result["n"]["lo"] == pytest.approx(0.0)
    assert result["n"]["hi"] == pytest.approx(0.0)
    json.dumps(result)

    again = stats.paired_bootstrap_difference(rows_a, rows_b, _accuracy, n_boot=300)
    assert again == stats.paired_bootstrap_difference(rows_a, rows_b, _accuracy, n_boot=300)
    assert list(again) == ["accuracy"]


def test_paired_bootstrap_difference_estimate_none_when_side_undefined() -> None:
    rows_a, _ = _paired_accuracy_rows()
    rows_b = [dict(row, correct=None) for row in rows_a]
    result = stats.paired_bootstrap_difference(rows_a, rows_b, _accuracy, n_boot=100)
    assert result["accuracy"] == {"estimate": None, "lo": None, "hi": None}


def test_paired_bootstrap_difference_validates_alignment() -> None:
    rows_a, rows_b = _paired_accuracy_rows()
    with pytest.raises(ValueError, match="lengths"):
        stats.paired_bootstrap_difference(rows_a[:-1], rows_b, _accuracy)

    swapped = [rows_b[1], rows_b[0], *rows_b[2:]]
    with pytest.raises(ValueError, match="not aligned"):
        stats.paired_bootstrap_difference(rows_a, swapped, _accuracy)


def test_paired_bootstrap_difference_empty_inputs() -> None:
    result = stats.paired_bootstrap_difference([], [], _accuracy)
    assert result == {"accuracy": {"estimate": None, "lo": None, "hi": None}}


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------


def test_mcnemar_exact_hand_computed() -> None:
    # b=2 (A right / B wrong), c=8, plus 3 concordant pairs => n_pairs=13
    correct_a = [True] * 2 + [False] * 8 + [True] * 3
    correct_b = [False] * 2 + [True] * 8 + [True] * 3
    result = stats.mcnemar_test(correct_a, correct_b)

    # p = 2 * (C(10,0) + C(10,1) + C(10,2)) / 2**10 = 112 / 1024
    assert result["n_pairs"] == 13
    assert result["b"] == 2
    assert result["c"] == 8
    assert result["statistic"] == pytest.approx(2.0)
    assert result["p_value"] == pytest.approx(112.0 / 1024.0)
    assert result["method"] == "exact_binomial"
    json.dumps(result)


def test_mcnemar_exact_symmetric_and_no_discordance() -> None:
    symmetric = stats.mcnemar_test(
        [True] * 3 + [False] * 3, [False] * 3 + [True] * 3
    )
    assert symmetric["p_value"] == pytest.approx(1.0)
    assert symmetric["statistic"] == pytest.approx(3.0)
    assert symmetric["method"] == "exact_binomial"

    agree = stats.mcnemar_test([True, True, False], [True, True, False])
    assert agree["b"] == 0
    assert agree["c"] == 0
    assert agree["p_value"] == pytest.approx(1.0)
    assert agree["statistic"] == pytest.approx(0.0)
    assert agree["method"] == "exact_binomial"


def test_mcnemar_large_sample_uses_continuity_corrected_chi2() -> None:
    b, c = 700, 400
    correct_a = [True] * b + [False] * c
    correct_b = [False] * b + [True] * c
    result = stats.mcnemar_test(correct_a, correct_b)

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    expected_p = math.erfc(math.sqrt(statistic / 2.0))
    assert result["n_pairs"] == 1100
    assert result["method"] == "chi2_continuity_corrected"
    assert result["statistic"] == pytest.approx(statistic)
    assert result["p_value"] == pytest.approx(expected_p, rel=1e-12)
    assert result["p_value"] < 1e-10


def test_mcnemar_exact_chi2_boundary_is_inclusive_for_exact_test() -> None:
    # b + c == 1000: still the exact test ("large" means > 1000)
    b, c = 600, 400
    correct_a = [True] * b + [False] * c
    correct_b = [False] * b + [True] * c
    result = stats.mcnemar_test(correct_a, correct_b)
    tail = sum(math.comb(1000, k) for k in range(401))
    expected_p = min(1.0, 2.0 * tail / float(2**1000))
    assert result["method"] == "exact_binomial"
    assert result["p_value"] == pytest.approx(expected_p, rel=1e-12)


def test_mcnemar_drops_missing_pairs_and_validates_length() -> None:
    correct_a = [True, False, None, True, False]
    correct_b = [False, True, True, True, None]
    result = stats.mcnemar_test(correct_a, correct_b)  # type: ignore[arg-type]
    assert result["n_pairs"] == 3
    assert (result["b"], result["c"]) == (1, 1)
    assert result["p_value"] == pytest.approx(1.0)

    empty = stats.mcnemar_test([], [])
    assert empty == {
        "n_pairs": 0,
        "b": 0,
        "c": 0,
        "statistic": 0.0,
        "p_value": 1.0,
        "method": "exact_binomial",
    }

    with pytest.raises(ValueError, match="lengths"):
        stats.mcnemar_test([True], [True, False])


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def test_holm_correct_hand_computed() -> None:
    # sorted raw: 0.005, 0.01, 0.03, 0.04 (m=4)
    # step-down: 4*0.005=0.02, 3*0.01=0.03, 2*0.03=0.06, 1*0.04=0.04
    # monotone max   : 0.02, 0.03, 0.06, 0.06
    result = stats.holm_correct([0.01, 0.04, 0.03, 0.005])
    assert result == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_correct_clips_and_keeps_monotonicity() -> None:
    assert stats.holm_correct([0.9, 0.9]) == pytest.approx([1.0, 1.0])
    assert stats.holm_correct([]) == []
    assert stats.holm_correct([0.04]) == pytest.approx([0.04])

    ordered = stats.holm_correct([0.001, 0.008, 0.039, 0.041, 0.042])
    assert ordered == sorted(ordered)
    for raw, adjusted in zip([0.001, 0.008, 0.039, 0.041, 0.042], ordered, strict=True):
        assert raw <= adjusted <= 1.0


def test_bh_correct_hand_computed() -> None:
    # sorted raw: 0.005, 0.01, 0.03, 0.04 (m=4)
    # step-up (unclipped): 0.02, 0.02, 0.04, 0.04
    result = stats.bh_correct([0.01, 0.04, 0.03, 0.005])
    assert result == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bh_correct_small_inputs_and_monotonicity() -> None:
    assert stats.bh_correct([0.9, 0.9]) == pytest.approx([0.9, 0.9])
    assert stats.bh_correct([]) == []
    assert stats.bh_correct([0.04]) == pytest.approx([0.04])

    ordered = stats.bh_correct([0.001, 0.008, 0.039, 0.041, 0.042])
    assert ordered == sorted(ordered)
    for raw, adjusted in zip([0.001, 0.008, 0.039, 0.041, 0.042], ordered, strict=True):
        assert raw <= adjusted <= 1.0


# ---------------------------------------------------------------------------
# paired_outcomes
# ---------------------------------------------------------------------------


def test_paired_outcomes_aligns_by_item_key_and_drops_failures() -> None:
    rows_a = [
        {"item_id": "i1", "is_correct": True},
        {"item_id": "i2", "is_correct": False},
        {"item_id": "i3", "is_correct": None},
        {"item_id": "i4", "is_correct": True, "parse_status": "parse_error"},
        {"item_id": "i5", "is_correct": True},
    ]
    rows_b = [
        {"item_id": "i5", "is_correct": True},
        {"item_id": "i4", "is_correct": True},
        {"item_id": "i3", "is_correct": True},
        {"item_id": "i2", "is_correct": True},
        {"item_id": "i1", "is_correct": False},
        {"item_id": "i7", "is_correct": True},  # not in A -> ignored
    ]
    outcomes_a, outcomes_b = stats.paired_outcomes(rows_a, rows_b)
    # A's order is preserved; i3 (None) and i4 (parse error) are dropped.
    assert outcomes_a == [True, False, True]
    assert outcomes_b == [False, True, True]
    assert all(isinstance(value, bool) for value in outcomes_a + outcomes_b)


def test_paired_outcomes_and_mcnemar_integration() -> None:
    rows_a = [
        {"item_id": f"i{index}", "is_correct": correct_a}
        for index, correct_a in enumerate([True, True, False, False])
    ]
    rows_b = [
        {"item_id": f"i{index}", "is_correct": correct_b}
        for index, correct_b in enumerate([False, True, False, True])
    ]
    outcomes_a, outcomes_b = stats.paired_outcomes(rows_a, rows_b)
    result = stats.mcnemar_test(outcomes_a, outcomes_b)
    assert result["n_pairs"] == 4
    assert (result["b"], result["c"]) == (1, 1)
    assert result["p_value"] == pytest.approx(1.0)


def test_paired_outcomes_empty_and_missing_correctness() -> None:
    assert stats.paired_outcomes([], []) == ([], [])
    rows_a = [{"item_id": "i1"}]
    rows_b = [{"item_id": "i1"}]
    assert stats.paired_outcomes(rows_a, rows_b) == ([], [])
