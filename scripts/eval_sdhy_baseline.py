"""Evaluate the SDHY speaking-assessment baseline on our prosodic-abx subset.

Baseline (from `third_party/SDHY_Test_System/evaluater_speaking_0.1`, P0-v2):

* acoustic encoder: HF ``jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn``,
  pinned weights sha256 ``de031fd4b29e…`` (= ``models/p0-baseline.json``);
* M1 phone CTC head (``models/m1-ctc-head.npz``) -> initial prediction from the
  first decoded phone;
* X2 initial/final alignment heads (``models/x2-align-heads.npz``) -> final
  prediction from the best non-blank label over frames;
* tone: their F0 extractor + contour features, with thresholds calibrated on a
  **disjoint** 512-audio slice (they do not ship a 4-way tone classifier; this is
  an explicit adaptation for free recognition, see docs/baseline-sdhy.md).

The evaluation subset is the minimal-128 audio set (2,048 audios,
``data/manifests/tasks-min128/minimal_summary.json``) so the numbers are directly
comparable to run ``mini128-v1``.  Detection is derived from the baseline's free
predictions exactly like the Qwen "derived detection".

Run with the baseline venv (torch/transformers installed)::

    PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SDHY = ROOT / "third_party" / "SDHY_Test_System" / "evaluater_speaking_0.1"
MODEL_DIR = ROOT / "third_party" / "models" / "wav2vec2-large-xlsr-53-chinese-zh-cn"
WEIGHT_SHA = "de031fd4b29e0c0667e5346450fadfe1326c89936b888b59c4ede608db763ee4"

INITIALS = {
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s",
}
FINAL_MAP = {"ix": "i", "iy": "i", "iz": "i", "v": "ü", "van": "üan",
             "ve": "üe", "vn": "ün"}
FINALS = {
    "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia",
    "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "o", "ong", "ou",
    "u", "ua", "uai", "uan", "uang", "ueng", "ui", "un", "uo", "ü", "üan",
    "üe", "ün",
}
#: candidate thresholds for the adapted tone classifier (rise / dip-range /
#: voice-normalised height), selected on a disjoint calibration slice
RISE_GRID = (0.8, 1.2, 1.5, 2.0)
RANGE_GRID = (2.0, 2.5, 3.0, 3.5)
HEIGHT_GRID = (0.0, 0.5, 1.0)


def canonical_final(label: str) -> str | None:
    value = FINAL_MAP.get(label, label)
    return value if value in FINALS else None


def log(msg: str) -> None:
    print(f"[baseline] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", default="data/manifests/tasks-min128")
    parser.add_argument("--audio-manifest", default="data/manifests/audio.parquet")
    parser.add_argument("--output", default="results/baseline-sdhy-min128.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--calibration-size", type=int, default=512)
    args = parser.parse_args()

    sys.path.insert(0, str(SDHY))
    from app.audio import load_mono_16k
    from app.ctc import collapse_ctc
    from app.tone import contour_features, extract_f0
    from app.wav2vec2 import CtcHead, Wav2Vec2Encoder, zero_mean_unit_var
    from app.x2_aligner import X2AlignmentSpans, load_vocab

    from pinyin_eval.metrics import detection_metrics
    from pinyin_eval.normalize import load_mapping

    mapping = load_mapping(ROOT / "data/mapping/mandarin_tone_syn_target_mapping.csv")
    task_dir = ROOT / args.task_dir
    summary = json.loads((task_dir / "minimal_summary.json").read_text(encoding="utf-8"))
    eval_audios = list(summary["audio_ids"])
    if args.limit:
        eval_audios = eval_audios[: args.limit]
    eval_set = set(eval_audios)
    log(f"eval audios: {len(eval_audios)}")

    manifest = pd.read_parquet(ROOT / args.audio_manifest)
    manifest = manifest[manifest["audio_id"].isin(eval_set)].copy()
    assert len(manifest) == len(eval_audios), "audio manifest does not cover the subset"
    records = list(manifest.itertuples())

    # ---- model load (sha256 verified by Wav2Vec2Encoder before torch loads) ----
    encoder = Wav2Vec2Encoder(
        model_dir=MODEL_DIR,
        revision="sha256:" + WEIGHT_SHA,
        device="cpu",
    )
    m1 = CtcHead.load(SDHY / "models" / "m1-ctc-head.npz")
    x2 = X2AlignmentSpans(heads_path=SDHY / "models" / "x2-align-heads.npz")
    vocab = load_vocab(SDHY / "models" / "x2-align-heads.npz")
    blank = vocab["blank"]
    list(vocab["initial"])
    final_labels = list(vocab["final"])
    log(f"encoder loaded: {encoder.compute_mode}")

    # ---- features for the adapted tone classifier -------------------------
    def tone_contour(samples: np.ndarray) -> dict:
        track = extract_f0(samples, 16000)
        feats = contour_features(track, 0.0, len(samples) / 16000)
        if feats is None:
            return {"ok": False}
        return {
            "ok": True,
            "rise": feats.rise_st,
            "range": feats.range_st,
            "median_hz": feats.median_hz,
            "voiced_frames": feats.voiced_frames,
        }

    def tone_predict(feat: dict, voice_median: dict[str, float], voice: str,
                     rise_th: float, range_th: float, height_th: float) -> int | None:
        if not feat["ok"] or feat["voiced_frames"] < 4:
            return None
        rise = feat["rise"]
        if rise >= rise_th:
            return 2
        if rise <= -rise_th:
            return 4
        rel = 12.0 * float(np.log2(feat["median_hz"] / voice_median[voice]))
        if feat["range"] >= range_th and rel < height_th:
            return 3
        return 1 if rel >= height_th else 3

    # calibration slice: audios from the full manifest NOT in the eval set
    calib = manifest.iloc[0:0]
    pool = pd.read_parquet(ROOT / args.audio_manifest)
    pool = pool[~pool["audio_id"].isin(eval_set)]
    calib = pool.sample(n=min(args.calibration_size, len(pool)),
                        random_state=20261028).copy()
    log(f"calibration audios: {len(calib)}")

    calib_feats = []
    voice_hz: dict[str, list[float]] = {}
    for row in calib.itertuples():
        samples = load_mono_16k(row.wav_path)
        feats = tone_contour(samples)
        calib_feats.append((row.voice, int(row.tone), feats))
        voice_hz.setdefault(row.voice, []).extend(
            [x for x in extract_f0(samples, 16000).voiced]
        )
    voice_median = {v: float(np.median(hz)) for v, hz in voice_hz.items()}
    log(f"voice pitch medians: { {k: round(v, 1) for k, v in voice_median.items()} }")

    best = None
    for rise_th in RISE_GRID:
        for range_th in RANGE_GRID:
            for height_th in HEIGHT_GRID:
                correct = 0
                for voice, gold, feats in calib_feats:
                    pred = tone_predict(feats, voice_median, voice, rise_th, range_th, height_th)
                    correct += pred == gold
                if best is None or correct > best[0]:
                    best = (correct, rise_th, range_th, height_th)
    _, rise_th, range_th, height_th = best
    log(f"tone thresholds: rise={rise_th} range={range_th} height={height_th} "
        f"(calibration acc={best[0] / len(calib_feats):.3f})")

    # ---- evaluation --------------------------------------------------------
    preds: list[dict] = []
    for i, row in enumerate(records, 1):
        samples = load_mono_16k(row.wav_path)
        hidden = encoder.encode(zero_mean_unit_var(samples)).valid()
        hidden64 = hidden.astype(np.float64)

        # initial: M1 phone CTC greedy decode, first non-silence phone
        m1_logits = np.einsum("td,cd->tc", hidden64, m1.weight.astype(np.float64)) \
            + m1.bias.astype(np.float64)
        seq = [m1.labels[int(j)] for j in collapse_ctc(np.argmax(m1_logits, axis=1), m1.blank)]
        seq = [s for s in seq if s != "sil"]
        pred_initial = seq[0] if seq and seq[0] in INITIALS else ""

        # final: X2 head, best non-blank label over frames
        x2_logits = np.einsum("td,cd->tc", hidden64, x2._final_weight.astype(np.float64)) \
            + x2._final_bias.astype(np.float64)
        cols = [c for c in range(len(final_labels)) if c != blank]
        pred_final = canonical_final(final_labels[cols[int(np.argmax(x2_logits[:, cols].max(axis=0)))]])
        if pred_final is None:
            pred_final = ""

        feat = tone_contour(samples)
        pred_tone = tone_predict(feat, voice_median, row.voice, rise_th, range_th, height_th)

        preds.append({
            "audio_id": row.audio_id,
            "voice": row.voice,
            "gold_initial": row.initial,
            "gold_final": row.final,
            "gold_tone": int(row.tone),
            "pred_initial": pred_initial,
            "pred_final": pred_final,
            "pred_tone": pred_tone,
        })
        if i % 250 == 0:
            log(f"evaluated {i}/{len(records)}")

    frame = pd.DataFrame(preds)
    frame["initial_correct"] = frame["pred_initial"] == frame["gold_initial"]
    frame["final_correct"] = frame["pred_final"] == frame["gold_final"]
    frame["tone_correct"] = frame["pred_tone"] == frame["gold_tone"]
    frame["joint_correct"] = (
        frame["initial_correct"] & frame["final_correct"] & frame["tone_correct"]
    )
    metrics = {
        "n": int(len(frame)),
        "initial_accuracy": float(frame["initial_correct"].mean()),
        "final_accuracy": float(frame["final_correct"].mean()),
        "tone_accuracy": float(frame["tone_correct"].mean()),
        "joint_accuracy": float(frame["joint_correct"].mean()),
        "tone_null_predictions": int(frame["pred_tone"].isna().sum()),
    }

    # ---- derived detection on the same detection claims -------------------
    detection_tasks = pd.read_json(task_dir / "detection.jsonl", lines=True)
    detection_tasks = detection_tasks.set_index("item_id")
    detection_sets = pd.read_csv(task_dir / "detection_sets.csv")
    main = detection_sets[detection_sets["subset"] == "main"]
    main = main[main["audio_id"].isin(set(frame["audio_id"]))]
    lookup = frame.set_index("audio_id")
    detection_rows = []
    for row in main.itertuples():
        for role, item_id in (("neg", row.negative_item_id), ("pos", row.positive_item_id)):
            task = detection_tasks.loc[item_id]
            pred = lookup.loc[row.audio_id]
            claim = mapping.get(task["given_target"])
            claimed_initial = claim.initial if claim else None
            claimed_final = claim.final if claim else None
            consistent = (
                pred["pred_initial"] == claimed_initial
                and pred["pred_final"] == claimed_final
                and pred["pred_tone"] == int(task["given_tone"])
            )
            detection_rows.append({
                "detection_role": role,
                "pred_consistent": bool(consistent),
            })
    detection = detection_metrics(detection_rows)
    result = {
        "baseline": "SDHY evaluater_speaking_0.1 P0-v2 (adapted free recognition)",
        "encoder": {"model_id": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                    "weights_sha256": WEIGHT_SHA, "compute_mode": encoder.compute_mode},
        "subset": {"task_dir": args.task_dir, "n_audios": len(frame)},
        "tone_classifier": {"rise_th": rise_th, "range_th": range_th,
                            "height_th": height_th,
                            "calibration_acc": best[0] / len(calib_feats)},
        "metrics": metrics,
        "derived_detection": detection,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    frame.to_csv(out.with_suffix(".predictions.csv"), index=False)
    log(f"metrics: {json.dumps(metrics, ensure_ascii=False)}")
    log(f"derived detection: {json.dumps(detection, ensure_ascii=False)}")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
