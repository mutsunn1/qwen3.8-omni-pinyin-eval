"""Derived detection: judge consistency from free-recognition predictions.

The detection experiment asks the model to judge ``target + target_tone`` against
the audio.  Run ``mini3-v1`` showed the model is heavily biased towards
"inconsistent" (FPR 0.764), and prompt variants (v1/v2/v3) did not fix it.

A much better judge is already available for free: the *same audio* is also
transcribed by experiment 3 (free recognition: initial / final / tone).  We can
compare those predictions with the claimed target programmatically:

    derived_consistent = pred_initial == claim_initial
                     and pred_final  == claim_final
                     and pred_tone   == claim_tone

This module implements that comparison on top of ``items.parquet`` +
``detection_rows.parquet`` and reports model-judgment vs derived-detection
metrics side by side.  Used by ``pinyin-eval derive-detect`` and reported in
``docs/results-minimal-v1.md`` / ``docs/results-minimal-128.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .metrics import detection_metrics
from .normalize import MappingEntry

DEFAULT_FREE_CONDITION = "free__think0__clean__json"
DERIVED_SUBSETS = ("main", "initial", "final", "tone")


def split_target(
    target: str | None, mapping: Mapping[str, MappingEntry]
) -> tuple[str | None, str | None]:
    """Split a toneless claim into canonical (initial, final)."""
    if not isinstance(target, str):
        return None, None
    entry = mapping.get(target)
    return (entry.initial, entry.final) if entry else (None, None)


def derive_rows(
    detection_rows: pd.DataFrame,
    free_items: pd.DataFrame,
    mapping: Mapping[str, MappingEntry],
) -> pd.DataFrame:
    """Attach ``derived_consistent`` to detection rows using free predictions.

    ``free_items`` must contain the exp3 free-recognition items of the same run
    with columns ``audio_id, pred_initial, pred_final, pred_tone, parse_status``
    (extra columns are ignored).  Rows whose free prediction failed get a
    missing ``derived_consistent`` and are reported as failures by
    :func:`compare_metrics`.
    """
    free = (
        free_items[["audio_id", "pred_initial", "pred_final", "pred_tone", "parse_status"]]
        .drop_duplicates("audio_id")
        .rename(columns={
            "pred_initial": "pred_initial_free",
            "pred_final": "pred_final_free",
            "pred_tone": "pred_tone_free",
            "parse_status": "free_parse_status",
        })
    )
    out = detection_rows.merge(free, on="audio_id", how="left")
    out[["claim_initial", "claim_final"]] = [
        split_target(target, mapping) for target in out["given_target"].tolist()
    ]
    free_ok = out["free_parse_status"].fillna("").eq("ok")
    det_ok = out["parse_status"].fillna("").eq("ok")
    tone_match = out["pred_tone_free"].astype("Float64").eq(out["given_tone"].astype("Float64"))
    consistent = (
        out["pred_initial_free"].eq(out["claim_initial"])
        & out["pred_final_free"].eq(out["claim_final"])
        & tone_match
    )
    out["derived_consistent"] = pd.array(
        [bool(v) if ok else None for v, ok in zip(consistent, free_ok & det_ok, strict=True)],
        dtype="boolean",
    )
    return out


def _subset_metrics(df: pd.DataFrame, column: str) -> dict[str, Any]:
    rows = df[[column, "detection_role"]].dropna(subset=[column])
    rows = rows[rows["detection_role"].isin(["neg", "pos"])]
    records = [
        {"detection_role": role, "pred_consistent": bool(value)}
        for value, role in zip(rows[column], rows["detection_role"], strict=True)
    ]
    return detection_metrics(records)


def compare_metrics(
    derived: pd.DataFrame, subsets: tuple[str, ...] = DERIVED_SUBSETS
) -> dict[str, dict[str, Any]]:
    """Return ``{subset: {model: metrics, derived: metrics, n...}}``."""
    out: dict[str, dict[str, Any]] = {}
    available = set(derived["detection_subset"].dropna()) if "detection_subset" in derived else set()
    for subset in subsets:
        if subset not in available:
            continue
        df = derived[derived["detection_subset"] == subset]
        out[subset] = {
            "model": _subset_metrics(df, "pred_consistent"),
            "derived": _subset_metrics(df, "derived_consistent"),
        }
    return out


def flat_table(comparison: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten :func:`compare_metrics` into rows for CSV output."""
    rows: list[dict[str, Any]] = []
    for subset, judges in comparison.items():
        for judge, m in judges.items():
            rows.append(
                {
                    "subset": subset,
                    "judge": judge,
                    "n_neg": m["n_neg"],
                    "n_pos": m["n_pos"],
                    "error_precision": m["error_precision"],
                    "error_recall": m["error_recall"],
                    "error_f1": m["error_f1"],
                    "fpr": m["fpr"],
                    "fnr": m["fnr"],
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                }
            )
    return rows


__all__ = [
    "DEFAULT_FREE_CONDITION",
    "DERIVED_SUBSETS",
    "compare_metrics",
    "derive_rows",
    "flat_table",
    "split_target",
]
