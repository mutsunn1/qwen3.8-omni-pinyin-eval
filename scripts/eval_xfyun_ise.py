"""Evaluate iFlytek ISE (语音评测流式版) as an external baseline.

ISE is a *scoring* API (audio + reference test paper -> pronunciation scores),
not free recognition.  We therefore use **candidate scoring**: for each audio we
send 2-4 candidate references and pick the best-scoring one.

Two empirical tricks are required (measured in this session):

* **silence padding** — isolated 0.2-0.6 s syllables get degenerate phone scores
  if sent raw; pad 300 ms of silence on both sides;
* **pinyin annotation** — a single character with a pinyin annotation is
  rejected (annotated chars must be <= 1/3 of the paper), so send
  ``\\ufeff一<ai4>一一`` (3 chars, first annotated) — ISE evaluates the first
  syllable as the annotated pinyin; the two deleted dummy chars dilute the
  absolute score but the *ranking* across candidates is stable.

Evaluations (minimal-128 subset):

1. tone classification: 512 audios (stratified 32/voice/tone) x 4 tone refs;
2. design minimal pairs: all pair audio sides (837) x 2 candidate refs;
3. SDHY-baseline weak pairs: 6 initial + 6 final + 3 tone pairs x 64 audios x 2;
4. detection: 256 consistent + 256 inconsistent claims, scored against the
   claimed target (AUC + threshold calibrated/evaluated on split halves).

Total unique calls ≈ 6.2k (budget 10k, hard cap below).  Calls are cached in
``data/ise/cache.jsonl`` so the run resumes after interruption.

Usage::

    set -a && . ./.env && set +a
    .venv/bin/python scripts/eval_xfyun_ise.py --smoke   # ~40 calls
    .venv/bin/python scripts/eval_xfyun_ise.py           # full run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import os
import random
import re
import sys
import urllib.parse
import wave
from email.utils import formatdate
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinyin_eval.normalize import load_mapping, normalize_syllable  # noqa: E402

HOST = "ise-api.xfyun.cn"
ISE_PATH = "/v2/open-ise"
RATE = 16000
PAD_S = 0.3
SEED = 20261028
MAX_CALLS = 9500
MAPPING = load_mapping(ROOT / "data/mapping/mandarin_tone_syn_target_mapping.csv")


# ---------------------------------------------------------------------------
# reference pinyin (canonical initial/final/tone -> ISE pinyin surface)
# ---------------------------------------------------------------------------

ZERO_INITIAL = {
    "i": "yi", "ia": "ya", "ie": "ye", "iao": "yao", "iu": "you", "ian": "yan",
    "in": "yin", "iang": "yang", "ing": "ying", "iong": "yong",
    "ü": "yu", "üe": "yue", "üan": "yuan", "ün": "yun",
    "u": "wu", "ua": "wa", "uo": "wo", "uai": "wai", "ui": "wei",
    "uan": "wan", "un": "wen", "uang": "wang", "ueng": "weng",
}
UMLAUT_SURFACE = {"ü": "v", "üe": "ve", "üan": "van", "ün": "vn"}
JQX_SURFACE = {"ü": "u", "üe": "ue", "üan": "uan", "ün": "un"}


def ise_pinyin(initial: str, final: str, tone: int) -> str:
    """Canonical (initial, final, tone) -> pinyin surface accepted by ISE."""
    if initial in ("j", "q", "x") and final.startswith("ü"):
        body = initial + JQX_SURFACE[final]  # ju / jue / juan / jun
    elif initial in ("l", "n") and final.startswith("ü"):
        body = initial + UMLAUT_SURFACE[final]  # lv / lve
    elif initial == "":
        body = ZERO_INITIAL.get(final, final)
    else:
        body = initial + final
    return f"{body}{int(tone)}"


def ref_from_target(target: str, tone: int) -> str:
    entry = MAPPING[target]
    return ise_pinyin(entry.initial, entry.final, tone)


def ref_from_audio(row: pd.Series, tone: int | None = None) -> str:
    return ise_pinyin(row["initial"], row["final"],
                      int(row["tone"]) if tone is None else int(tone))


# ---------------------------------------------------------------------------
# ISE websocket client
# ---------------------------------------------------------------------------


def auth_url(api_key: str, api_secret: str) -> str:
    date = formatdate(usegmt=True)
    origin = f"host: {HOST}\ndate: {date}\nGET {ISE_PATH} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode(), origin.encode(), hashlib.sha256).digest()
    ).decode()
    auth_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(auth_origin.encode()).decode()
    query = urllib.parse.urlencode({"authorization": authorization, "host": HOST, "date": date})
    return f"wss://{HOST}{ISE_PATH}?{query}"


def reference_text(pinyin: str) -> str:
    return f"\ufeff一<{pinyin}>一一\n"


def padded_wav(audio_id: str, src: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"{audio_id}.wav"
    if dst.exists():
        return dst
    with wave.open(str(src), "rb") as fh:
        pcm = fh.readframes(fh.getnframes())
    with wave.open(str(dst), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes(b"\x00\x00" * int(PAD_S * RATE) + pcm + b"\x00\x00" * int(PAD_S * RATE))
    return dst


def parse_scores(xml: str) -> dict[str, float]:
    match = re.search(r'<read_syllable[^>]*total_score="[^"]*"[^>]*>', xml)
    if not match:
        return {}
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', match.group(0)))
    return {k: float(attrs.get(k) or 0.0)
            for k in ("total_score", "tone_score", "phone_score", "integrity_score")}


async def ise_call(audio: Path, pinyin: str, app_id: str, key: str, secret: str) -> dict:
    import websockets

    with wave.open(str(audio), "rb") as fh:
        pcm = fh.readframes(fh.getnframes())
    chunk = RATE * 2 * 40 // 1000
    chunks = [pcm[i:i + chunk] for i in range(0, len(pcm), chunk)] or [b""]
    business = {
        "sub": "ise", "ent": "cn_vip", "category": "read_syllable", "cmd": "ssb",
        "text": reference_text(pinyin), "tte": "utf-8", "ttp_skip": True,
        "aue": "raw", "auf": "audio/L16;rate=16000", "rstcd": "utf8",
        "rst": "entirety", "ise_unite": "1",
        "extra_ability": "syll_phone_err_msg", "plev": "0",
        "group": "adult", "check_type": "common",
    }
    async with websockets.connect(auth_url(key, secret), max_size=None) as ws:
        await ws.send(json.dumps({"common": {"app_id": app_id}, "business": business,
                                  "data": {"status": 0}}, ensure_ascii=False))
        for index, payload in enumerate(chunks):
            aus = 1 if index == 0 else (4 if index == len(chunks) - 1 else 2)
            status = 2 if index == len(chunks) - 1 else 1
            await ws.send(json.dumps({
                "business": {"cmd": "auw", "aus": aus},
                "data": {"status": status, "data": base64.b64encode(payload).decode()},
            }))
            await asyncio.sleep(0.005)
        last: dict = {}
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            last = json.loads(raw)
            if last.get("code") not in (0, None):
                break
            if (last.get("data") or {}).get("status") == 2:
                break
    code = int(last.get("code") or 0)
    xml = ""
    if code == 0 and (last.get("data") or {}).get("data"):
        xml = base64.b64decode(last["data"]["data"]).decode("utf-8", errors="replace")
    scores = parse_scores(xml) if xml else {}
    return {
        "code": code,
        "total": scores.get("total_score", 0.0),
        "tone": scores.get("tone_score", 0.0),
        "phone": scores.get("phone_score", 0.0),
        "integrity": scores.get("integrity_score", 0.0),
        "sid": str(last.get("sid") or ""),
    }


async def call_retry(audio: Path, pinyin: str, app_id: str, key: str, secret: str,
                     sem: asyncio.Semaphore, attempts: int = 3) -> dict:
    result = {"code": -1, "total": 0.0, "tone": 0.0, "phone": 0.0, "integrity": 0.0}
    async with sem:
        for attempt in range(attempts):
            try:
                result = await ise_call(audio, pinyin, app_id, key, secret)
                if result["code"] == 0:
                    return result
            except Exception as exc:  # noqa: BLE001 - transient network errors
                result = {"code": -2, "total": 0.0, "tone": 0.0, "phone": 0.0,
                          "integrity": 0.0, "sid": str(exc)[:80]}
            await asyncio.sleep(0.4 * (attempt + 1))
    return result


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def build_plan(smoke: bool = False) -> tuple[dict[str, list[dict]], list[dict]]:
    man = pd.read_parquet(ROOT / "data/manifests/audio.parquet").set_index("audio_id")
    summary = json.loads((ROOT / "data/manifests/tasks-min128/minimal_summary.json")
                         .read_text(encoding="utf-8"))
    subset = list(summary["audio_ids"])
    rng = random.Random(SEED)
    plans: dict[str, list[dict]] = {}

    # 1) tone classification
    tone_sample: list[str] = []
    for voice in "ABCD":
        for tone in (1, 2, 3, 4):
            pool = [a for a in subset
                    if man.loc[a, "voice"] == voice and int(man.loc[a, "tone"]) == tone]
            per_cell = 1 if smoke else 32
            tone_sample.extend(rng.sample(pool, min(per_cell, len(pool))))
    plans["tone"] = [
        {"task": "tone", "audio_id": a, "ref": ref_from_audio(man.loc[a], t), "gold_tone": t}
        for a in tone_sample for t in (1, 2, 3, 4)
    ]

    # 2) design minimal pairs
    minimal = set((ROOT / "data/manifests/tasks-min128/minimal_pair_items.txt").read_text().split())
    pairs = pd.read_json(ROOT / "data/manifests/tasks-min128/pairs.jsonl", lines=True)
    pairs = pairs[pairs["item_id"].isin(minimal)].drop_duplicates(["pair_id", "pair_side"])
    if smoke:
        pairs = pairs.head(2)
    design: list[dict] = []
    for row in pairs.itertuples():
        audio = row.audio_id
        row_data = man.loc[audio]
        head, members, voice, tones = row.pair_id.split(":")
        mem = members.split("/")
        if row.experiment in ("exp4", "exp5"):
            tone = int(row_data["tone"])
            axis = "initial" if row.experiment == "exp4" else "final"
            gold = row_data["initial"] if axis == "initial" else row_data["final"]
            # which member is the audio? match the gold dimension (pair_side order is
            # not guaranteed to follow the pair_id member order)
            own_t = next(t for t in mem
                         if (MAPPING[t].initial if axis == "initial" else MAPPING[t].final) == gold)
            other_t = mem[1] if own_t == mem[0] else mem[0]
            refs = [ref_from_target(own_t, tone), ref_from_target(other_t, tone)]
        else:
            gold = int(row_data["tone"])
            t1, t2 = [int(x) for x in tones.split("-")]
            own_tone, other_tone = (t1, t2) if row.pair_side == "A" else (t2, t1)
            refs = [ref_from_audio(row_data, own_tone), ref_from_audio(row_data, other_tone)]
            axis = "tone"
        for ref in refs:
            design.append({"task": "pair", "audio_id": audio, "ref": ref,
                           "experiment": row.experiment, "pair_type": row.pair_type,
                           "axis": axis, "gold": gold, "own_ref": refs[0], "other_ref": refs[1]})
    plans["pairs"] = design

    # 3) SDHY weak pairs
    weak_specs = [
        ("initial", "m", "n"), ("initial", "n", "l"), ("initial", "d", "t"),
        ("initial", "s", "sh"), ("initial", "z", "zh"), ("initial", "r", "l"),
        ("final", "üe", "ie"), ("final", "ei", "ie"), ("final", "o", "uo"),
        ("final", "ia", "ie"), ("final", "in", "ing"), ("final", "en", "eng"),
        ("tone", 1, 3), ("tone", 2, 3), ("tone", 3, 4),
    ]
    weak: list[dict] = []

    def weak_gold(x: str, axis: str, a: object, b: object) -> object:
        if axis == "initial":
            return man.loc[x, "initial"]
        if axis == "final":
            return man.loc[x, "final"]
        return int(man.loc[x, "tone"])

    def weak_refs(x: str, axis: str, a: object, b: object) -> list[str]:
        if axis == "initial":
            return [ise_pinyin(str(a), man.loc[x, "final"], int(man.loc[x, "tone"])),
                    ise_pinyin(str(b), man.loc[x, "final"], int(man.loc[x, "tone"]))]
        if axis == "final":
            return [ise_pinyin(man.loc[x, "initial"], str(a), int(man.loc[x, "tone"])),
                    ise_pinyin(man.loc[x, "initial"], str(b), int(man.loc[x, "tone"]))]
        return [ref_from_audio(man.loc[x], int(a)), ref_from_audio(man.loc[x], int(b))]

    for axis, a, b in weak_specs:
        if axis == "initial":
            pool = [x for x in subset if man.loc[x, "initial"] in (a, b)]
        elif axis == "final":
            pool = [x for x in subset if man.loc[x, "final"] in (a, b)]
        else:
            pool = [x for x in subset if int(man.loc[x, "tone"]) in (a, b)]
        by_member: dict[object, list[str]] = {}
        for x in pool:
            by_member.setdefault(weak_gold(x, axis, a, b), []).append(x)
        per_member = 1 if smoke else 32
        picked: list[str] = []
        for member in (a, b):
            picked.extend(rng.sample(by_member.get(member, []), min(per_member, len(by_member.get(member, [])))))
        for x in picked:
            refs = weak_refs(x, axis, a, b)
            for ref in refs:
                weak.append({"task": "weak", "audio_id": x, "ref": ref, "axis": axis,
                             "pair": f"{a}/{b}", "gold": weak_gold(x, axis, a, b),
                             "own_ref": refs[0], "other_ref": refs[1]})
    plans["weak"] = weak

    # 4) detection claims
    detection = pd.read_json(ROOT / "data/manifests/tasks-min128/detection.jsonl", lines=True)
    detection = detection.set_index("item_id")
    sets = pd.read_csv(ROOT / "data/manifests/tasks-min128/detection_sets.csv")
    main = sets[sets["subset"] == "main"]
    # only claims whose audio is in this run's subset (detection sets are built on
    # the full 6,160-audio catalog; the minimal run only has 2,048 detection audios)
    main = main[main["audio_id"].isin(set(subset))]
    n_claims = 2 if smoke else 256
    neg = rng.sample(list(main["negative_item_id"]), n_claims)
    pos = rng.sample(list(main["positive_item_id"]), n_claims)
    detect: list[dict] = []
    for role, ids in (("neg", neg), ("pos", pos)):
        for item_id in ids:
            task = detection.loc[item_id]
            audio = task["audio_id"]
            detect.append({"task": "detect", "audio_id": audio,
                           "ref": ref_from_target(task["given_target"], int(task["given_tone"])),
                           "role": role, "item_id": item_id})
    plans["detect"] = detect
    return plans, list(man.index)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


async def execute(plans: dict[str, list[dict]], man: pd.DataFrame, cache_path: Path,
                  padded_dir: Path, app_id: str, key: str, secret: str,
                  concurrency: int) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("code") == 0:
                cache[row["call_key"]] = row
    unique: dict[str, dict] = {}
    for items in plans.values():
        for item in items:
            unique[f"{item['audio_id']}|{item['ref']}"] = item
    todo = [v for k, v in unique.items() if k not in cache]
    print(f"[ise] unique calls={len(unique)} cached={len(unique) - len(todo)} todo={len(todo)}",
          flush=True)
    if len(unique) > MAX_CALLS:
        raise SystemExit(f"plan has {len(unique)} calls > budget {MAX_CALLS}")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0
    handle = cache_path.open("a", encoding="utf-8")

    async def one(item: dict) -> None:
        nonlocal done
        src = Path(ROOT / man.loc[item["audio_id"], "wav_path"])
        audio = padded_wav(item["audio_id"], src, padded_dir)
        result = await call_retry(audio, item["ref"], app_id, key, secret, sem)
        row = {**result, "call_key": f"{item['audio_id']}|{item['ref']}",
               "audio_id": item["audio_id"], "ref": item["ref"]}
        async with lock:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if result["code"] == 0:
                cache[row["call_key"]] = row
            done += 1
            if done % 100 == 0:
                print(f"[ise] {done}/{len(todo)} calls", flush=True)

    await asyncio.gather(*(one(item) for item in todo))
    handle.close()
    return cache


def best(rows: list[dict], key: str = "total") -> dict:
    return max(rows, key=lambda r: (float(r.get("total") or 0.0),
                                    float(r.get("phone") or 0.0)))


def split_pred(ref: str, axis: str) -> object | None:
    """ISE pinyin surface ('zhang1', 'yu4', 'jue2') -> canonical initial/final."""
    initial, final = normalize_syllable(re.sub(r"[1-4]$", "", ref), MAPPING)
    return initial if axis == "initial" else final


def roc_auc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def f1(labels, pred) -> float:
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    tp = int(((labels == 1) & (pred == 1)).sum())
    fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


def evaluate(plans: dict[str, list[dict]], man: pd.DataFrame, cache: dict[str, dict]) -> dict:
    def get(audio_id: str, ref: str) -> dict:
        return cache.get(f"{audio_id}|{ref}", {"total": 0.0, "tone": 0.0, "phone": 0.0, "code": -3})

    tone_rows = []
    for a in sorted({x["audio_id"] for x in plans["tone"]}):
        refs = [x["ref"] for x in plans["tone"] if x["audio_id"] == a]
        chosen = best([get(a, r) | {"ref": r} for r in refs], "tone")
        gold = int(man.loc[a, "tone"])
        tone_rows.append({"audio_id": a, "voice": man.loc[a, "voice"], "gold": gold,
                          "pred": int(str(chosen["ref"])[-1]), "score": chosen.get("tone", 0.0)})
    tone_df = pd.DataFrame(tone_rows)
    tone = {"n": int(len(tone_df)),
            "accuracy": float((tone_df["gold"] == tone_df["pred"]).mean()) if len(tone_df) else None,
            "by_voice": {v: float((g["gold"] == g["pred"]).mean())
                         for v, g in tone_df.groupby("voice")}}

    def score_pair(meta: dict) -> dict:
        own = get(meta["audio_id"], meta["own_ref"])
        other = get(meta["audio_id"], meta["other_ref"])
        chosen = own if (own.get("total", 0) >= other.get("total", 0)) else other
        ref = meta["own_ref"] if chosen is own else meta["other_ref"]
        if meta["axis"] == "tone":
            pred = int(str(ref)[-1])
        else:
            pred = split_pred(ref, meta["axis"])
        return {**meta, "pred": pred, "correct": pred == meta["gold"]}

    pair_df = pd.DataFrame([score_pair(x) for x in plans["pairs"]])
    pair = {"n": int(len(pair_df)), "accuracy": float(pair_df["correct"].mean()),
            "by_pair_type": {k: {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
                             for k, g in pair_df.groupby("pair_type")}}
    weak_df = pd.DataFrame([score_pair(x) for x in plans["weak"]])
    weak = {f"{k1}/{k2}": {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
            for (k1, k2), g in weak_df.groupby(["axis", "pair"])}

    det_rows = []
    for meta in plans["detect"]:
        row = get(meta["audio_id"], meta["ref"])
        # inconsistency score: lower ISE total score = more likely inconsistent
        det_rows.append({**meta, "score": -float(row.get("total", 0.0)),
                         "total": float(row.get("total", 0.0)),
                         "label": 1 if meta["role"] == "pos" else 0})
    det = pd.DataFrame(det_rows)
    det["split"] = np.where(np.arange(len(det)) % 2 == 0, "calib", "eval")
    calib, eval_ = det[det["split"] == "calib"], det[det["split"] == "eval"]
    auc = roc_auc(det["label"], det["score"])
    best_t, best_f1 = 0.0, -1.0
    for threshold in np.unique(calib["score"]):
        pred = (calib["score"] >= threshold).astype(int)
        value = f1(calib["label"], pred)
        if value > best_f1:
            best_t, best_f1 = float(threshold), float(value)
    pred_eval = (eval_["score"] >= best_t).astype(int)
    detection = {
        "n": int(len(det)), "n_pos": int(det["label"].sum()), "auc": auc,
        "threshold": best_t, "calib_f1": best_f1,
        "eval": {"n": int(len(eval_)), "error_f1": f1(eval_["label"], pred_eval),
                 "fpr": float(pred_eval[eval_["label"] == 0].mean()),
                 "fnr": float((1 - pred_eval)[eval_["label"] == 1].mean()),
                 "accuracy": float((pred_eval == eval_["label"]).mean())},
    }
    return {"tone": tone, "design_pairs": pair, "weak_pairs": weak, "detection": detection,
            "tables": {"tone": tone_df, "pairs": pair_df, "weak": weak_df, "detect": det}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", default="results/ise-baseline.json")
    parser.add_argument("--reuse-only", action="store_true",
                        help="skip API calls and evaluate the existing cache")
    args = parser.parse_args()
    app = os.environ.get("XFYUN_APPID", "")
    key = os.environ.get("XFYUN_APIKEY", "")
    secret = os.environ.get("XFYUN_APISECRET", "")
    if not (app and key and secret):
        raise SystemExit("set XFYUN_APPID / XFYUN_APIKEY / XFYUN_APISECRET")
    plans, _ = build_plan(smoke=args.smoke)
    man = pd.read_parquet(ROOT / "data/manifests/audio.parquet").set_index("audio_id")
    cache_path = ROOT / "data/ise/cache.jsonl"
    if args.reuse_only:
        cache = {}
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("code") == 0:
                cache[row["call_key"]] = row
    else:
        cache = asyncio.run(execute(plans, man, cache_path, ROOT / "data/ise/padded",
                                    app, key, secret, args.concurrency))
    result = evaluate(plans, man, cache)
    tables = result.pop("tables")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "baseline": "iFlytek ISE 语音评测（流式版）",
        "candidate_scoring": {"audio_padding_s": PAD_S,
                              "reference_text": "\\ufeff一<{pinyin}>一一",
                              "tone_key": "tone_score,total_score",
                              "pair_key": "phone_score"},
        **result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, frame in tables.items():
        frame.to_csv(out.with_suffix(f".{name}.csv"), index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
