"""Tests for :mod:`pinyin_eval.metrics` (contract section 6).

All expectations are hand-computed on tiny synthetic rows; no fixtures under
``data/`` are read and no other project module is required.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pinyin_eval import metrics


def _row(**overrides: object) -> dict:
    """A complete, correct ``ItemResult.to_row()``-shaped row (all three dims right)."""
    row: dict = {
        "parse_status": "ok",
        "voice": "A",
        "cluster_id": "audio-1",
        "item_id": "item-1",
        "gold_initial": "zh",
        "gold_final": "ang",
        "gold_tone": 1,
        "pred_initial": "zh",
        "pred_final": "ang",
        "pred_tone": 1,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# classification_metrics
# ---------------------------------------------------------------------------


def test_classification_metrics_hand_computed() -> None:
    # gold: a a b b ; pred: a b b b  ->  accuracy 3/4
    # class a: tp=1 fp=0 fn=1 -> p=1, r=1/2, f1=2/3
    # class b: tp=2 fp=1 fn=0 -> p=2/3, r=1, f1=4/5
    result = metrics.classification_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"])

    assert result["n"] == 4
    assert result["n_failures"] == 0
    assert result["labels"] == ["a", "b"]
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["per_class"]["a"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert result["per_class"]["a"]["f1"] == pytest.approx(2.0 / 3.0)
    assert result["per_class"]["a"]["support"] == 2
    assert result["per_class"]["b"]["precision"] == pytest.approx(2.0 / 3.0)
    assert result["per_class"]["b"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["b"]["f1"] == pytest.approx(4.0 / 5.0)
    assert result["per_class"]["b"]["support"] == 2
    assert result["macro_f1"] == pytest.approx((2.0 / 3.0 + 4.0 / 5.0) / 2.0)


def test_classification_metrics_drops_failed_predictions() -> None:
    # gold: 1 2 1 ; pred: 1 None 2 -> n=2, failures=1, accuracy=1/2
    result = metrics.classification_metrics([1, 2, 1], [1, None, 2])

    assert result["n"] == 2
    assert result["n_failures"] == 1
    assert result["labels"] == [1, 2]
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["per_class"][1] == {
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2.0 / 3.0),
        "support": 2,
    }
    assert result["per_class"][2]["precision"] == pytest.approx(0.0)
    assert result["per_class"][2]["recall"] == pytest.approx(0.0)
    assert result["per_class"][2]["f1"] == pytest.approx(0.0)
    assert result["macro_f1"] == pytest.approx((2.0 / 3.0) / 2.0)


def test_classification_metrics_explicit_labels_match_sklearn_definition() -> None:
    # sklearn: labels are kept as given; absent labels score precision/recall/F1 = 0.
    result = metrics.classification_metrics(["a", "b"], ["a", "b"], labels=["c", "a", "b"])

    assert result["labels"] == ["c", "a", "b"]
    assert result["n"] == 2
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["per_class"]["c"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert result["per_class"]["a"]["f1"] == pytest.approx(1.0)
    assert result["per_class"]["b"]["f1"] == pytest.approx(1.0)
    assert result["macro_f1"] == pytest.approx(2.0 / 3.0)

    subset = metrics.classification_metrics(["a", "b"], ["a", "b"], labels=["a"])
    assert subset["labels"] == ["a"]
    assert subset["macro_f1"] == pytest.approx(1.0)


def test_classification_metrics_labels_subset_still_counts_all_errors() -> None:
    result = metrics.classification_metrics(["a", "a"], ["b", "b"], labels=["a"])
    # accuracy is computed over every scored pair, independent of ``labels``
    assert result["accuracy"] == pytest.approx(0.0)
    assert result["per_class"]["a"]["precision"] == pytest.approx(0.0)
    assert result["per_class"]["a"]["recall"] == pytest.approx(0.0)
    assert result["macro_f1"] == pytest.approx(0.0)


def test_classification_metrics_all_failures_and_empty() -> None:
    all_failed = metrics.classification_metrics(["a", "b"], [None, None])
    assert all_failed == {
        "n": 0,
        "n_failures": 2,
        "accuracy": None,
        "macro_f1": None,
        "labels": [],
        "per_class": {},
    }

    empty = metrics.classification_metrics([], [])
    assert empty == {
        "n": 0,
        "n_failures": 0,
        "accuracy": None,
        "macro_f1": None,
        "labels": [],
        "per_class": {},
    }


def test_classification_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        metrics.classification_metrics(["a"], ["a", "b"])


def test_classification_metrics_numpy_inputs_and_json_output() -> None:
    gold = np.array(["a", "b", "a"])
    pred = np.array(["a", "b", "a"], dtype=object)
    result = metrics.classification_metrics(gold, pred)
    # numpy labels are converted to plain Python, so strict JSON works
    assert result["labels"] == ["a", "b"]
    json.dumps(result)

    tones = metrics.classification_metrics(
        np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 4], dtype=np.int64)
    )
    assert tones["labels"] == [1, 2, 3, 4]
    assert tones["accuracy"] == pytest.approx(2.0 / 3.0)
    json.dumps(tones)


# ---------------------------------------------------------------------------
# recognition_metrics
# ---------------------------------------------------------------------------


def test_recognition_metrics_hand_computed() -> None:
    rows = [
        _row(item_id="i1"),
        _row(item_id="i2", pred_initial="z", pred_tone=2),
        _row(
            item_id="i3",
            parse_status="parse_error",
            pred_initial=None,
            pred_final=None,
            pred_tone=None,
        ),
        _row(item_id="i4", parse_status="refused", pred_initial=None, pred_final=None),
    ]
    result = metrics.recognition_metrics(rows)

    assert result["n"] == 2
    assert result["n_failures"] == 2
    assert result["initial"]["accuracy"] == pytest.approx(0.5)
    assert result["final"]["accuracy"] == pytest.approx(1.0)
    assert result["tone"]["accuracy"] == pytest.approx(0.5)
    assert result["joint_accuracy"] == pytest.approx(0.5)
    assert result["initial"]["n"] == 2
    assert result["initial"]["n_failures"] == 0
    json.dumps(result)


def test_recognition_metrics_ok_row_missing_prediction_counts_as_wrong_joint() -> None:
    rows = [_row(item_id="i1", pred_initial=None), _row(item_id="i2")]
    result = metrics.recognition_metrics(rows)
    assert result["n"] == 2
    assert result["n_failures"] == 0
    assert result["initial"]["n"] == 1
    assert result["initial"]["n_failures"] == 1
    assert result["joint_accuracy"] == pytest.approx(0.5)


def test_recognition_metrics_accepts_pandas_series_rows() -> None:
    dict_rows = [_row(item_id="i1"), _row(item_id="i2", pred_tone=4)]
    series_rows = [pd.Series(row) for row in dict_rows]

    from_dicts = metrics.recognition_metrics(dict_rows)
    from_series = metrics.recognition_metrics(series_rows)
    assert from_series == from_dicts
    assert from_series["tone"]["accuracy"] == pytest.approx(0.5)


def test_recognition_metrics_empty() -> None:
    result = metrics.recognition_metrics([])
    assert result["n"] == 0
    assert result["n_failures"] == 0
    assert result["joint_accuracy"] is None
    for dimension in ("initial", "final", "tone"):
        assert result[dimension]["n"] == 0
        assert result[dimension]["accuracy"] is None
        assert result[dimension]["macro_f1"] is None


# ---------------------------------------------------------------------------
# detection_metrics
# ---------------------------------------------------------------------------


def test_detection_metrics_hand_computed() -> None:
    # pred_consistent=True means "model judged consistent"; the positive class is
    # "judged inconsistent", i.e. pred_consistent=False.
    rows = [
        _row(detection_role="neg", pred_consistent=True, item_id="n1"),  # TN
        _row(detection_role="neg", pred_consistent=False, item_id="n2"),  # FP
        _row(detection_role="pos", pred_consistent=False, item_id="p1"),  # TP
        _row(detection_role="pos", pred_consistent=False, item_id="p2"),  # TP
        _row(detection_role="pos", pred_consistent=True, item_id="p3"),  # FN
        _row(detection_role="neg", pred_consistent=None, item_id="n3"),  # failure
        # parse status wins over a (stale) prediction value:
        _row(
            detection_role="pos",
            parse_status="empty",
            pred_consistent=True,
            item_id="p4",
        ),
    ]
    result = metrics.detection_metrics(rows)

    assert result["n_neg"] == 2
    assert result["n_pos"] == 3
    assert result["n_failures_neg"] == 1
    assert result["n_failures_pos"] == 1
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (2, 1, 1, 1)
    assert result["error_precision"] == pytest.approx(2.0 / 3.0)
    assert result["error_recall"] == pytest.approx(2.0 / 3.0)
    assert result["error_f1"] == pytest.approx(2.0 / 3.0)
    assert result["fpr"] == pytest.approx(0.5)
    assert result["fnr"] == pytest.approx(1.0 / 3.0)
    assert result["accuracy"] == pytest.approx(3.0 / 5.0)
    assert result["balanced_accuracy"] == pytest.approx((2.0 / 3.0 + 0.5) / 2.0)
    json.dumps(result)


def test_detection_metrics_pred_consistent_semantics_regression() -> None:
    """Regression: True = judged consistent, so positives are judged-inconsistent rows.

    The 4-row construction below has TP=1, FP=1, TN=1, FN=1 by the frozen
    definitions (TP: pos & pred_consistent False; FP: neg & pred_consistent False;
    TN: neg & pred_consistent True; FN: pos & pred_consistent True).
    """
    rows = [
        _row(detection_role="neg", pred_consistent=True),  # TN: neg, judged consistent
        _row(detection_role="neg", pred_consistent=False),  # FP: neg, judged inconsistent
        _row(detection_role="pos", pred_consistent=False),  # TP: pos, judged inconsistent
        _row(detection_role="pos", pred_consistent=True),  # FN: pos, judged consistent
    ]
    result = metrics.detection_metrics(rows)

    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["error_precision"] == pytest.approx(0.5)
    assert result["error_recall"] == pytest.approx(0.5)
    assert result["error_f1"] == pytest.approx(0.5)
    assert result["fpr"] == pytest.approx(0.5)
    assert result["fnr"] == pytest.approx(0.5)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)

    # If one of the two negative rows is unusable (parse failure), the same
    # construction gives FP=0 and therefore EP=1, FPR=0, FNR=0.5.
    rows_with_failure = [
        _row(detection_role="neg", pred_consistent=True),  # TN
        _row(detection_role="neg", pred_consistent=None),  # failure
        _row(detection_role="pos", pred_consistent=False),  # TP
        _row(detection_role="pos", pred_consistent=True),  # FN
    ]
    with_failure = metrics.detection_metrics(rows_with_failure)
    assert (
        with_failure["tp"],
        with_failure["fp"],
        with_failure["tn"],
        with_failure["fn"],
    ) == (1, 0, 1, 1)
    assert with_failure["n_failures_neg"] == 1
    assert with_failure["error_precision"] == pytest.approx(1.0)
    assert with_failure["error_recall"] == pytest.approx(0.5)
    assert with_failure["error_f1"] == pytest.approx(2.0 / 3.0)
    assert with_failure["fpr"] == pytest.approx(0.0)
    assert with_failure["fnr"] == pytest.approx(0.5)


def test_detection_metrics_numpy_bool_and_zero_denominators() -> None:
    # neg + judged consistent (pred_consistent=True) -> true negative
    rows = [_row(detection_role="neg", pred_consistent=np.bool_(True))]
    result = metrics.detection_metrics(rows)
    assert result["tn"] == 1
    # no positive example => precision/recall/F1, FNR and balanced accuracy undefined
    assert result["error_precision"] is None
    assert result["error_recall"] is None
    assert result["error_f1"] is None
    assert result["fnr"] is None
    assert result["balanced_accuracy"] is None
    assert result["fpr"] == pytest.approx(0.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_detection_metrics_all_failures() -> None:
    rows = [
        _row(detection_role="neg", pred_consistent=None),
        _row(detection_role="pos", parse_status="api_error", pred_consistent=True),
    ]
    result = metrics.detection_metrics(rows)
    assert (result["n_neg"], result["n_pos"]) == (0, 0)
    assert (result["n_failures_neg"], result["n_failures_pos"]) == (1, 1)
    assert result["tp"] == result["fp"] == result["tn"] == result["fn"] == 0
    for key in (
        "error_precision",
        "error_recall",
        "error_f1",
        "fpr",
        "fnr",
        "accuracy",
        "balanced_accuracy",
    ):
        assert result[key] is None


def test_detection_metrics_requires_detection_role() -> None:
    with pytest.raises(ValueError, match="detection_role"):
        metrics.detection_metrics([_row()])
    with pytest.raises(ValueError, match="detection_role"):
        metrics.detection_metrics([_row(detection_role="other", pred_consistent=True)])


# ---------------------------------------------------------------------------
# ab_metrics
# ---------------------------------------------------------------------------


def test_ab_metrics_hand_computed() -> None:
    rows = [
        _row(pred_choice="A", correct_choice="A"),
        _row(pred_choice="B", correct_choice="A"),
        _row(pred_choice=None, correct_choice="A"),  # parsed but no usable choice
        _row(parse_status="parse_error", pred_choice="A", correct_choice="A"),
    ]
    result = metrics.ab_metrics(rows)

    assert result["n"] == 3
    assert result["n_failures"] == 1
    assert result["accuracy"] == pytest.approx(1.0 / 3.0)
    assert result["choice_counts"] == {"A": 1, "B": 1}
    json.dumps(result)


def test_ab_metrics_empty() -> None:
    result = metrics.ab_metrics([])
    assert result == {
        "n": 0,
        "n_failures": 0,
        "accuracy": None,
        "choice_counts": {"A": 0, "B": 0},
    }


# ---------------------------------------------------------------------------
# confusion_matrix / confusion_rows
# ---------------------------------------------------------------------------


def test_confusion_matrix_rows_are_gold_columns_are_pred() -> None:
    gold = ["a", "a", "b", "b", "c"]
    pred = ["a", "b", "b", None, "c"]
    matrix = metrics.confusion_matrix(gold, pred)

    assert list(matrix.index) == ["a", "b", "c"]
    assert list(matrix.columns) == ["a", "b", "c"]
    assert matrix.index.name == "gold"
    assert matrix.columns.name == "pred"
    assert matrix.loc["a", "a"] == 1
    assert matrix.loc["a", "b"] == 1
    assert matrix.loc["b", "b"] == 1
    assert matrix.loc["b", "a"] == 0
    assert matrix.loc["c", "c"] == 1
    assert int(matrix.values.sum()) == 4


def test_confusion_matrix_explicit_labels_and_empty() -> None:
    matrix = metrics.confusion_matrix(["a", "b"], ["a", "b"], labels=["b", "a"])
    assert list(matrix.index) == ["b", "a"]
    assert matrix.loc["a", "a"] == 1
    assert matrix.loc["b", "b"] == 1
    assert matrix.loc["a", "b"] == 0

    empty = metrics.confusion_matrix([], [])
    assert empty.shape == (0, 0)

    with pytest.raises(ValueError, match="same length"):
        metrics.confusion_matrix(["a"], [])


def test_confusion_rows_hand_computed() -> None:
    rows = [
        _row(pred_initial="z"),
        _row(pred_initial="z"),
        _row(pred_initial="zh"),
        _row(parse_status="parse_error", pred_initial="z"),
        _row(pred_initial=None),
    ]
    result = metrics.confusion_rows(rows, dimension="initial")
    assert result == [
        {"gold": "zh", "pred": "z", "count": 2},
        {"gold": "zh", "pred": "zh", "count": 1},
    ]


def test_confusion_rows_tone_deterministic_order_and_validation() -> None:
    rows = [
        _row(pred_tone=2),
        _row(pred_tone=4),
        _row(pred_tone=2, item_id="i2"),
    ]
    result = metrics.confusion_rows(rows, dimension="tone")
    assert result == [
        {"gold": 1, "pred": 2, "count": 2},
        {"gold": 1, "pred": 4, "count": 1},
    ]
    assert metrics.confusion_rows([], dimension="final") == []
    with pytest.raises(ValueError, match="dimension"):
        metrics.confusion_rows(rows, dimension="bogus")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def test_majority_baseline() -> None:
    assert metrics.majority_baseline(["x", "x", "y"]) == {
        "label": "x",
        "accuracy": pytest.approx(2.0 / 3.0),
        "n": 3,
    }
    # tie -> deterministic sorted label, like sklearn's most_frequent
    assert metrics.majority_baseline(["b", "a"]) == {
        "label": "a",
        "accuracy": pytest.approx(0.5),
        "n": 2,
    }
    assert metrics.majority_baseline([1, 1, 2, 3]) == {
        "label": 1,
        "accuracy": pytest.approx(0.5),
        "n": 4,
    }
    assert metrics.majority_baseline([]) == {"label": None, "accuracy": None, "n": 0}
    assert metrics.majority_baseline([None, None]) == {
        "label": None,
        "accuracy": None,
        "n": 0,
    }


def test_random_and_detection_chance_baselines() -> None:
    assert metrics.random_baseline(4) == pytest.approx(0.25)
    assert metrics.random_baseline(2) == pytest.approx(0.5)
    assert metrics.random_baseline(1) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n_classes"):
        metrics.random_baseline(0)
    assert metrics.detection_chance_baseline() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# recognition_metric_pack
# ---------------------------------------------------------------------------


def test_recognition_metric_pack_by_voice_and_condition() -> None:
    groups = {
        "cond1": [
            _row(item_id="i1", voice="A"),
            _row(item_id="i2", voice="B"),
        ],
        "cond2": [
            _row(item_id="i3", voice="A", pred_initial="z"),
        ],
    }
    pack = metrics.recognition_metric_pack(groups)

    assert set(pack) == {"overall", "by_voice", "by_condition"}
    assert pack["overall"]["n"] == 3
    assert pack["overall"]["initial"]["accuracy"] == pytest.approx(2.0 / 3.0)
    assert sorted(pack["by_voice"]) == ["A", "B"]
    assert pack["by_voice"]["A"]["n"] == 2
    assert pack["by_voice"]["A"]["initial"]["accuracy"] == pytest.approx(0.5)
    assert pack["by_voice"]["B"]["initial"]["accuracy"] == pytest.approx(1.0)
    assert list(pack["by_condition"]) == ["cond1", "cond2"]
    assert pack["by_condition"]["cond1"]["initial"]["accuracy"] == pytest.approx(1.0)
    assert pack["by_condition"]["cond2"]["initial"]["accuracy"] == pytest.approx(0.0)
    json.dumps(pack)


def test_recognition_metric_pack_empty() -> None:
    pack = metrics.recognition_metric_pack({})
    assert pack["overall"]["n"] == 0
    assert pack["by_voice"] == {}
    assert pack["by_condition"] == {}
