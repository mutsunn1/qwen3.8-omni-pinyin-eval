"""Analyze the detection over-flagging probes (prompt variants v1/v2/v3).

Background: in run ``mini3-v1`` the model judged 76% of *consistent* detection
samples as inconsistent (FPR 0.764, error-F1 0.720).  Three prompt variants were
run on the same 512-audio subset (3,072 calls each, ~¥0.5 each):

    v1  explicit decision threshold (only clear mismatches count as inconsistent)
    v2  decompose into pinyin_match / tone_match, then aggregate
    v3  strict comparison wording

Result: none of them fixes the bias (paired on the 113 audios shared with the
baseline: FPR 0.805 -> 0.850/0.982/0.832).  This script also computes the
**derived detection** alternative: use the free-recognition predictions for the
same audio as the judge (compare predicted initial/final/tone with the claimed
target), which needs no extra API calls and yields F1 0.908 / FPR 0.193.

Usage::

    PYTHONPATH=src python scripts/analyze_detection_probe.py [RUN_ID]

``RUN_ID`` defaults to ``mini3-v1`` (the main minimal run).  Probe runs are
compared only when their ``detection_rows.parquet`` exists (they are local
artifacts, not committed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pinyin_eval.metrics import detection_metrics
from pinyin_eval.normalize import load_mapping

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
MAPPING = load_mapping(str(ROOT / "data/mapping/mandarin_tone_syn_target_mapping.csv"))


def load_main(run_id: str) -> pd.DataFrame:
    df = pd.read_parquet(RUNS / run_id / "detection_rows.parquet")
    if "detection_subset" in df.columns:
        df = df[df["detection_subset"] == "main"]
    return df


def metrics(df: pd.DataFrame, column: str = "pred_consistent") -> dict:
    rows = [
        {
            "detection_role": row.detection_role,
            "pred_consistent": None if pd.isna(getattr(row, column)) else bool(getattr(row, column)),
        }
        for row in df.itertuples()
    ]
    return detection_metrics(rows)


def derived_detection(run_id: str = "mini3-v1") -> dict:
    """Detection judged by the exp3 free-recognition predictions (C1 condition)."""
    items = pd.read_parquet(RUNS / run_id / "items.parquet")
    free = items[
        (items["experiment"] == "exp3")
        & (items["prompt_condition"] == "free")
        & (items["condition_id"] == "free__think0__clean__json")
    ][["audio_id", "pred_initial", "pred_final", "pred_tone", "parse_status"]].drop_duplicates("audio_id")
    det = load_main(run_id).merge(free, on="audio_id", how="left", suffixes=("", "_free"))
    det = det[det["parse_status_free"] == "ok"].copy()

    def split(target: str) -> tuple[str | None, str | None]:
        entry = MAPPING.get(target)
        return (entry.initial, entry.final) if entry else (None, None)

    det["claim_initial"] = det["given_target"].map(lambda t: split(t)[0])
    det["claim_final"] = det["given_target"].map(lambda t: split(t)[1])
    det["derived_consistent"] = (
        (det["pred_initial_free"] == det["claim_initial"])
        & (det["pred_final_free"] == det["claim_final"])
        & (det["pred_tone_free"].astype("Float64") == det["given_tone"].astype("Float64"))
    )
    return metrics(det, "derived_consistent")


def paired_probe_comparison() -> dict[str, dict]:
    common = set(load_main("mini3-v1")["audio_id"])
    for variant in ("probe-detect-v1", "probe-detect-v2", "probe-detect-v3"):
        common &= set(load_main(variant)["audio_id"])
    out: dict[str, dict] = {}
    for label, run in {
        "v0 (baseline)": "mini3-v1",
        "v1 threshold": "probe-detect-v1",
        "v2 decompose": "probe-detect-v2",
        "v3 strict": "probe-detect-v3",
    }.items():
        df = load_main(run)
        out[label] = metrics(df[df["audio_id"].isin(common)])
    return out


def main(run_id: str = "mini3-v1") -> None:
    print(f"== derived detection from free recognition (run {run_id}) ==")
    derived = derived_detection(run_id)
    judged = metrics(load_main("mini3-v1"))
    for name, m in (("model judgment", judged), ("derived (free recognition)", derived)):
        print(
            f"{name:28s} F1={m['error_f1']:.3f} FPR={m['fpr']:.3f} "
            f"recall={m['error_recall']:.3f} acc={m['accuracy']:.3f} (n={m['n_neg']})"
        )
    print("\n== prompt-variant probes, paired on shared audios ==")
    probes = paired_probe_comparison()
    for name, m in probes.items():
        print(
            f"{name:14s} F1={m['error_f1']:.3f} FPR={m['fpr']:.3f} "
            f"recall={m['error_recall']:.3f} acc={m['accuracy']:.3f} (n={m['n_neg']})"
        )
    out = {"run_id": run_id, "derived": derived, "model_judgment": judged, "probes": probes}
    target = ROOT / "results" / f"detection-probe-summary-{run_id}.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    print("\nNOTE: results/ is gitignored; this script is the reproducible entry point.")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "mini3-v1")
