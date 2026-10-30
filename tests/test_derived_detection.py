"""Tests for derived detection (free-recognition predictions as consistency judge)."""

from __future__ import annotations

import pandas as pd

from pinyin_eval.derived_detection import (
    compare_metrics,
    derive_rows,
    flat_table,
    split_target,
)
from pinyin_eval.normalize import MappingEntry

MAPPING = {
    "ai": MappingEntry("ai", "", "ai", 1, "test"),
    "zhang": MappingEntry("zhang", "zh", "ang", 2, "test"),
    "zang": MappingEntry("zang", "z", "ang", 1, "test"),
}


def detection_frame(rows):
    return pd.DataFrame(rows, columns=[
        "item_id", "audio_id", "detection_subset", "detection_role",
        "given_target", "given_tone", "parse_status", "pred_consistent",
    ])


def free_frame(rows):
    return pd.DataFrame(rows, columns=[
        "audio_id", "pred_initial", "pred_final", "pred_tone", "parse_status",
    ])


def test_split_target():
    assert split_target("zhang", MAPPING) == ("zh", "ang")
    assert split_target("ai", MAPPING) == ("", "ai")
    assert split_target("unknown", MAPPING) == (None, None)
    assert split_target(None, MAPPING) == (None, None)


def test_derive_rows_consistent_and_inconsistent():
    detection = detection_frame([
        # neg: claim matches the audio -> derived should be True
        ("d1", "a1", "main", "neg", "ai", 1, "ok", False),
        # pos: wrong initial claim -> derived should be False
        ("d2", "a1", "main", "pos", "zhang", 1, "ok", False),
        # pos: wrong tone claim -> derived False
        ("d3", "a1", "main", "pos", "ai", 2, "ok", False),
    ])
    free = free_frame([("a1", "", "ai", 1, "ok")])
    out = derive_rows(detection, free, MAPPING).set_index("item_id")
    assert out.loc["d1", "derived_consistent"] == True  # noqa: E712
    assert out.loc["d2", "derived_consistent"] == False  # noqa: E712
    assert out.loc["d3", "derived_consistent"] == False  # noqa: E712
    assert (out["claim_initial"] == pd.Series({"d1": "", "d2": "zh", "d3": ""})).all()


def test_derive_rows_free_failure_is_na():
    detection = detection_frame([
        ("d1", "a2", "main", "neg", "ai", 1, "ok", True),
    ])
    free = free_frame([("a2", None, None, None, "api_error")])
    out = derive_rows(detection, free, MAPPING)
    assert pd.isna(out.loc[0, "derived_consistent"])


def test_compare_metrics_model_vs_derived():
    detection = detection_frame([
        # one neg judged inconsistent by the model (FP), derived says consistent (TN)
        ("d1", "a1", "main", "neg", "ai", 1, "ok", False),
        # one pos judged inconsistent by both (TP)
        ("d2", "a1", "main", "pos", "zang", 1, "ok", False),
    ])
    free = free_frame([("a1", "", "ai", 1, "ok")])
    derived = derive_rows(detection, free, MAPPING)
    comparison = compare_metrics(derived)
    assert set(comparison) == {"main"}
    model = comparison["main"]["model"]
    naive = comparison["main"]["derived"]
    assert (model["tp"], model["fp"]) == (1, 1)
    assert (naive["tp"], naive["fp"]) == (1, 0)
    assert naive["error_f1"] > model["error_f1"]
    rows = flat_table(comparison)
    assert {row["judge"] for row in rows} == {"model", "derived"}


def test_compare_metrics_skips_missing_subsets():
    detection = detection_frame([
        ("d1", "a1", "main", "neg", "ai", 1, "ok", False),
    ])
    free = free_frame([("a1", "", "ai", 1, "ok")])
    derived = derive_rows(detection, free, MAPPING)
    assert set(compare_metrics(derived)) == {"main"}
    assert set(compare_metrics(derived, subsets=("main", "initial"))) == {"main"}


def test_derive_rows_requires_detection_parse_ok():
    detection = detection_frame([
        ("d1", "a1", "main", "neg", "ai", 1, "parse_error", None),
    ])
    free = free_frame([("a1", "", "ai", 1, "ok")])
    out = derive_rows(detection, free, MAPPING)
    assert pd.isna(out.loc[0, "derived_consistent"])
    comparison = compare_metrics(out)
    assert comparison["main"]["derived"]["n_neg"] == 0
    assert comparison["main"]["derived"]["error_f1"] is None
