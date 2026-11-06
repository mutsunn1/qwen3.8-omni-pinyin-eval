"""Evaluate SDHY P0-v2 in its **native** mode: known target + forced alignment.

The earlier SDHY baseline (`eval_sdhy_baseline.py`) used the frozen encoder / M1 /
X2 heads as a **free recognizer**, which is not what the product does.  The
native pipeline is target-aware:

    audio + target (initial, final, tone)
      -> Wav2Vec2 acoustic evidence
      -> X2 forced alignment spans
      -> M1.6 centroid margins (initial / final)
      -> F0 rule judge (tone)
      -> RuleCalibrator: GOOD / NEEDS_WORK / UNCERTAIN per dimension

For our detection task this is directly applicable: the claim
(``given_target + given_tone``) is the target, and the native verdict/margins
tell whether the audio is consistent with the claim.  We evaluate on the same
4,096 detection claims (2,048 audios x 1 consistent + 1 inconsistent) as the
Qwen/ISE detection baselines.

Outputs ``results/baseline-sdhy-native.json`` + per-claim CSV.

Run with the baseline venv::

    PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_native.py --limit 8
    PYTHONPATH=src third_party/.venv-baseline/bin/python scripts/eval_sdhy_native.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SDHY = ROOT / "third_party" / "SDHY_Test_System" / "evaluater_speaking_0.1"
MODEL_DIR = ROOT / "third_party" / "models" / "wav2vec2-large-xlsr-53-chinese-zh-cn"
WEIGHT_SHA = "de031fd4b29e0c0667e5346450fadfe1326c89936b888b59c4ede608db763ee4"

sys.path.insert(0, str(SDHY))
sys.path.insert(0, str(ROOT / "src"))
from pinyin_eval.normalize import load_mapping  # noqa: E402

MAPPING = load_mapping(ROOT / "data/mapping/mandarin_tone_syn_target_mapping.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_engine():
    from app.assessment_boundary import build_assessment
    from app.audio import load_mono_16k
    from app.centroid import CentroidComparator, NativeCentroids
    from app.content import default_inventory
    from app.wav2vec2 import CtcHead, Wav2Vec2Acoustic, Wav2Vec2Encoder
    from app.x2_aligner import X2AlignmentSpans

    encoder = Wav2Vec2Encoder(
        model_dir=MODEL_DIR,
        revision="sha256:" + WEIGHT_SHA,
        model_id="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        device="cpu",
    )
    spans = X2AlignmentSpans()
    acoustic = Wav2Vec2Acoustic(
        encoder=encoder,
        head=CtcHead.load(SDHY / "models" / "m1-ctc-head.npz"),
        reader=load_mono_16k,
        aligner_identity=f"x2-{spans.fingerprint()}",
    )
    comparator = CentroidComparator(
        centroids=NativeCentroids.load(SDHY / "models" / "native-centroids.npz"),
        span_provider=spans,
    )
    assessment = build_assessment(
        inventory=default_inventory(), acoustic=acoustic, comparator=comparator,
        explainer=None,
    )
    return assessment, encoder


def make_graph(initial: str, final: str, tone: int):
    from app.pronunciation import PronunciationGraph, SyllablePronunciation

    triple = (initial, final, str(tone))
    syllable = SyllablePronunciation(
        syllable_id="sy0", text="一", character_span=(0, 1),
        lexical=triple, surface=triple, variant_id="canonical",
    )
    return PronunciationGraph(
        graph_version="native-eval-v1", rule_version="none", syllables=(syllable,),
    )


TONE_PENALTY = 3.0


def claim_score(row) -> float:
    """Continuous inconsistency score from native margins/verdicts.

    initial/final: ``-margin`` (higher margin = more consistent);
    tone: +3 NEEDS_WORK / -3 GOOD / 0 UNCERTAIN.  The 3.0 scale is comparable to
    the centroid margin range (roughly -6..6) and was chosen before looking at
    the evaluation split; the threshold itself is calibrated on a held-out half.
    """
    parts: list[float] = []
    for key in ("margin_initial", "margin_final"):
        value = row.get(key) if isinstance(row, dict) else getattr(row, key, None)
        try:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                parts.append(-float(value))
        except (TypeError, ValueError):
            pass
    tone = row.get("verdict_tone") if isinstance(row, dict) else getattr(row, "verdict_tone", "")
    tone = str(tone or "").lower()
    if tone == "needs_work":
        parts.append(TONE_PENALTY)
    elif tone == "good":
        parts.append(-TONE_PENALTY)
    return max(parts) if parts else 0.0


def collect_claims(subset: set[str]) -> pd.DataFrame:
    detection = pd.read_json(ROOT / "data/manifests/tasks-min128/detection.jsonl", lines=True)
    detection = detection.set_index("item_id")
    sets = pd.read_csv(ROOT / "data/manifests/tasks-min128/detection_sets.csv")
    main = sets[sets["subset"] == "main"]
    main = main[main["audio_id"].isin(subset)]
    rows = []
    for row in main.itertuples():
        for role, item_id in (("neg", row.negative_item_id), ("pos", row.positive_item_id)):
            task = detection.loc[item_id]
            rows.append({
                "audio_id": task["audio_id"],
                "variant_type": task["variant_type"],
                "detection_role": role,
                "item_id": item_id,
                "given_target": task["given_target"],
                "given_tone": int(task["given_tone"]),
                "gold_initial": task["gold_initial"],
                "gold_final": task["gold_final"],
                "gold_tone": int(task["gold_tone"]),
            })
    frame = pd.DataFrame(rows)
    frame["label"] = (frame["detection_role"] == "pos").astype(int)
    return frame.sort_values(["audio_id", "detection_role"]).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> dict:
    def auc(labels, scores):
        labels = np.asarray(labels)
        scores = np.asarray(scores, dtype=float)
        pos, neg = scores[labels == 1], scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        return float(((pos[:, None] > neg[None, :]).sum()
                      + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))

    def f1(labels, pred):
        labels = np.asarray(labels)
        pred = np.asarray(pred)
        tp = int(((labels == 1) & (pred == 1)).sum())
        fp = int(((labels == 0) & (pred == 1)).sum())
        fn = int(((labels == 1) & (pred == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return (float(2 * p * r / (p + r)) if p + r else 0.0, float(p), float(r))

    ok = frame[frame["status"] == "completed"].copy()
    failures = int(len(frame) - len(ok))
    ok["inconsistency_score"] = ok.apply(claim_score, axis=1)
    needs_work = ok["verdict_inconsistent"].astype(bool)
    uncertain = ok["has_uncertain"].astype(bool)
    auc_margin = auc(ok["label"], ok["inconsistency_score"])
    result = {
        "n": int(len(ok)),
        "failures": failures,
        "n_pos": int(ok["label"].sum()),
        "verdict_rule": {
            "pred": "any dimension NEEDS_WORK -> inconsistent; otherwise consistent",
            "uncertain_as_consistent": {
                "error_f1": f1(ok.label, needs_work.astype(int))[0],
                "fpr": float(((needs_work) & (ok.label == 0)).sum()
                             / max(1, (ok.label == 0).sum())),
            },
            "uncertain_as_inconsistent": {
                "error_f1": f1(ok.label, (needs_work | uncertain).astype(int))[0],
                "fpr": float((((needs_work | uncertain)) & (ok.label == 0)).sum()
                             / max(1, (ok.label == 0).sum())),
            },
            "uncertain_rate": float(uncertain.mean()),
        },
        "margin_threshold": None,
        "auc": auc_margin,
    }
    if auc_margin is not None and not np.isnan(auc_margin):
        # calibrate threshold on even rows, evaluate odd rows (same protocol as ISE)
        ok = ok.reset_index(drop=True)
        # hash-split so both classes land in both halves regardless of row order
        ok["split"] = ok["item_id"].map(
            lambda item: "calib" if int(hashlib.sha256(str(item).encode()).hexdigest()[:8], 16) % 2 == 0 else "eval"
        )
        calib, eval_ = ok[ok.split == "calib"], ok[ok.split == "eval"]
        thresholds = np.unique(calib["inconsistency_score"])
        best = (0.0, -1.0)
        for t in thresholds:
            value = f1(calib["label"], (calib["inconsistency_score"] >= t).astype(int))[0]
            if value > best[1]:
                best = (float(t), float(value))
        pred = (eval_["inconsistency_score"] >= best[0]).astype(int)
        f, precision, recall = f1(eval_["label"], pred)
        result["margin_threshold"] = {
            "threshold": best[0],
            "calib_f1": best[1],
            "eval": {
                "n": int(len(eval_)),
                "error_f1": f,
                "precision": precision,
                "recall": recall,
                "fpr": float(((pred == 1) & (eval_["label"] == 0)).sum()
                             / max(1, (eval_["label"] == 0).sum())),
                "fnr": float(((pred == 0) & (eval_["label"] == 1)).sum()
                             / max(1, (eval_["label"] == 1).sum())),
                "accuracy": float((pred == eval_["label"]).mean()),
            },
        }
    # native localization: on positive claims, did the targeted dimension become NEEDS_WORK?
    loc = {}
    for variant, g in ok[ok["detection_role"] == "pos"].groupby("variant_type"):
        dim = {"initial": "initial", "final": "final", "tone": "tone"}.get(str(variant))
        if dim is None:
            continue
        loc[str(variant)] = {
            "n": int(len(g)),
            "target_dim_needs_work": float((g["dim_needs_work_" + dim]).mean()),
            "any_needs_work": float(g["verdict_inconsistent"].mean()),
        }
    result["localization_positive_claims"] = loc
    neg = ok[ok["detection_role"] == "neg"]
    result["negatives"] = {
        "n": int(len(neg)),
        "all_good": float((~neg["verdict_inconsistent"] & ~neg["has_uncertain"]).mean()),
        "any_needs_work": float(neg["verdict_inconsistent"].mean()),
        "uncertain": float(neg["has_uncertain"].mean()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit claims (smoke)")
    parser.add_argument("--out", default="results/baseline-sdhy-native.json")
    parser.add_argument("--reuse-only", action="store_true",
                        help="recompute metrics from the existing claims CSV (no assessment)")
    args = parser.parse_args()

    summary = json.loads((ROOT / "data/manifests/tasks-min128/minimal_summary.json")
                         .read_text(encoding="utf-8"))
    subset = set(summary["audio_ids"])
    out_path = ROOT / args.out
    if args.reuse_only:
        claims = pd.read_csv(out_path.with_suffix(".claims.csv"))
        metrics = summarize(claims)
        payload = {
            "baseline": "SDHY P0-v2 native target-aware assessment",
            "mode": "audio + claimed target -> X2 forced alignment + M1.6 centroid margins + F0 tone rules + RuleCalibrator",
            "reused_claims_csv": str(out_path.with_suffix(".claims.csv")),
            "subset": {"task_dir": "data/manifests/tasks-min128", "n_claims": int(len(claims)),
                       "n_audios": int(claims.audio_id.nunique())},
            "detection": metrics,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"wrote {out_path}")
        return 0
    claims = collect_claims(subset)
    if args.limit:
        claims = claims.head(args.limit)
    print(f"[sdhy-native] claims={len(claims)}", flush=True)

    assessment, encoder = build_engine()
    print(f"[sdhy-native] engine ready: {encoder.compute_mode}", flush=True)
    rows = []
    manifest = pd.read_parquet(ROOT / "data/manifests/audio.parquet").set_index("audio_id")
    started = time.time()
    for i, claim in enumerate(claims.itertuples(), 1):
        audio_path = ROOT / manifest.loc[claim.audio_id, "wav_path"]
        graph = make_graph(MAPPING[claim.given_target].initial,
                           MAPPING[claim.given_target].final, claim.given_tone)
        result = assessment.submit(
            audio_ref=str(audio_path),
            audio_sha256=_sha256(audio_path),
            sample_rate_hz=16000,
            graph=graph,
        )
        measured = result.to_measured()
        dims = {}
        if measured["syllables"]:
            for dim in measured["syllables"][0]["dimensions"]:
                dims[dim["dimension"]] = dim
        tone = dims.get("tone", {})
        tone_verdict = str(tone.get("verdict") or "").lower()
        score = claim_score({
            "margin_initial": dims.get("initial", {}).get("margin"),
            "margin_final": dims.get("final", {}).get("margin"),
            "verdict_tone": tone_verdict,
        })
        verdicts = [str(d.get("verdict") or "").lower() for d in dims.values()]
        verdict_inconsistent = any(v == "needs_work" for v in verdicts)
        has_good = bool(verdicts) and all(v == "good" for v in verdicts)
        has_uncertain = any(v == "uncertain" for v in verdicts)
        rows.append({
            **claim._asdict(),
            "status": result.status,
            "verdict": result.verdict,
            "verdict_inconsistent": verdict_inconsistent,
            "has_good": has_good,
            "has_uncertain": has_uncertain,
            "inconsistency_score": score,
            "margin_initial": dims.get("initial", {}).get("margin"),
            "margin_final": dims.get("final", {}).get("margin"),
            "verdict_initial": dims.get("initial", {}).get("verdict"),
            "verdict_final": dims.get("final", {}).get("verdict"),
            "verdict_tone": tone_verdict,
            "dim_needs_work_initial": str(dims.get("initial", {}).get("verdict") or "").lower() == "needs_work",
            "dim_needs_work_final": str(dims.get("final", {}).get("verdict") or "").lower() == "needs_work",
            "dim_needs_work_tone": tone_verdict == "needs_work",
            "reason_codes": ",".join(result.reason_codes),
            "model_version": result.model_version,
            "segmentation": result.segmentation,
        })
        if i % 100 == 0:
            print(f"[sdhy-native] {i}/{len(claims)}  elapsed={time.time()-started:.0f}s", flush=True)
    frame = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".claims.csv")
    frame.to_csv(csv_path, index=False)
    metrics = summarize(frame)
    payload = {
        "baseline": "SDHY P0-v2 native target-aware assessment",
        "mode": "audio + claimed target -> X2 forced alignment + M1.6 centroid margins + F0 tone rules + RuleCalibrator",
        "compute_mode": encoder.compute_mode,
        "encoder": {"model_id": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
                    "weights_sha256": WEIGHT_SHA},
        "subset": {"task_dir": "data/manifests/tasks-min128", "n_claims": int(len(frame)),
                   "n_audios": int(frame.audio_id.nunique())},
        "detection": metrics,
        "duration_s": round(time.time() - started, 1),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"wrote {out_path} / {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
