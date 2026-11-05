"""Tests for ``scripts/verify_sentence_d_run.py`` (independent task-D verifier).

A tiny synthetic task-D dataset (2 OMPAL utterances x 3 text sources) with
hand-computed expected values exercises the independent re-implementation:
gold alignment, reply parsing (design-correct ``error_type`` vocabulary),
text-error accounting (insertions count per design A.4), localization
macro-F1, sentence-score correlations, ASR text quality, injection integrity,
cost, acceptance and independent-vs-reported comparison.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_sentence_d_run.py"
_spec = importlib.util.spec_from_file_location("verify_sentence_d_run", SCRIPT)
assert _spec is not None and _spec.loader is not None
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)

MODEL = "qwen3.8-omni-flash"
RUN_ID = "verify-d-test"
SOURCES = ("D-gold", "D-asr", "D-inject-10")
SLUG = {"D-gold": "gold", "D-asr": "asr", "D-inject-10": "inject10"}

# u1: 他们都很忙 (都 tone error, 忙 vowel error)
U1 = "他们都很忙"
U1_LABELS = [
    {"text": "他", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "们", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "都", "tone": 0, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "很", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "忙", "tone": 1, "consonant": 1, "vowel": 0, "zero_initial": False},
]
# u2: 我家有四口人 (口 tone error)
U2 = "我家有四口人"
U2_LABELS = [
    {"text": "我", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": True},
    {"text": "家", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "有", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": True},
    {"text": "四", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "口", "tone": 0, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "人", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
]
UTTS = {
    "u1": {"gold": U1, "labels": U1_LABELS, "speaker": "L1",
           "scores": {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0}},
    "u2": {"gold": U2, "labels": U2_LABELS, "speaker": "L2",
           "scores": {"accuracy": 3.0, "fluency": 3.0, "prosody": 3.0}},
}
ASR_TEXT = {"u1": U1, "u2": "我家有四口入"}  # u2: 人 -> 入 substitution


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _item_id(source: str, utt: str) -> str:
    return f"D|{SLUG[source]}|{utt}"


def _gold_alignment_for(text: str, utt: str) -> tuple[list[dict], dict]:
    return verifier.gold_alignment(text, UTTS[utt]["gold"], UTTS[utt]["labels"])


def _make_d_items() -> list[dict]:
    items: list[dict] = []
    inject_ops = {
        "u1": [
            {"op": "sub", "position": 0, "gold": "他", "new": "她"},
            {"op": "ins", "position": 2, "gold": None, "new": "呀"},
        ],
        "u2": [{"op": "del", "position": 1, "gold": "家", "new": None}],
    }
    for utt, meta in UTTS.items():
        gold_chars = list(meta["gold"])
        asr_text = ASR_TEXT[utt]
        inject_chars, _problems = verifier.apply_injection_ops(
            gold_chars, inject_ops[utt]
        )
        texts = {
            "D-gold": meta["gold"],
            "D-asr": asr_text,
            "D-inject-10": "".join(inject_chars),
        }
        for source in SOURCES:
            text = texts[source]
            per_char, stats = _gold_alignment_for(text, utt)
            items.append(
                {
                    "item_id": _item_id(source, utt),
                    "task": "D",
                    "source": "ompal",
                    "speaker": meta["speaker"],
                    "audio_path": f"data/ompal/wav/{utt}.wav",
                    "duration_s": 1.0,
                    "text": text,
                    "chars": meta["gold"],
                    "pinyin_surface": [],
                    "labels": meta["labels"],
                    "scores": dict(meta["scores"]),
                    "native": False,
                    "learner": meta["speaker"],
                    "flags": {"native": False, "learner": True},
                    "meta": {
                        "text_source": source,
                        "source_slug": SLUG[source],
                        "utt_id": utt,
                        "learner_id": meta["speaker"],
                        "gold_text": meta["gold"],
                        "gold_alignment": per_char,
                        "gold_alignment_stats": stats,
                        "injection_ops": inject_ops[utt] if source == "D-inject-10" else [],
                        "asr_text_raw": ASR_TEXT[utt] if source == "D-asr" else None,
                    },
                }
            )
    return items


def _prediction_payload(item: dict) -> str:
    """Model reply: alignment follows the gold alignment, scores are predictions."""
    utt = item["meta"]["utt_id"]
    per_char = item["meta"]["gold_alignment"]
    text_chars = verifier.text_chars(item["text"])
    alignment = []
    for index, entry in enumerate(per_char):
        char = text_chars[index] if index < len(text_chars) else "?"
        if entry["aligned"]:
            text_error = bool(entry["text_error"])
            error_type = None if text_error else entry["error_type"]
        else:
            text_error = True
            error_type = None
        alignment.append(
            {
                "text": char,
                "aligned": bool(entry["aligned"]),
                "text_error": text_error,
                "error_type": error_type,
                "score": 4.0 if error_type in (None, "none") else 3.0,
            }
        )
    scores = (
        {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0}
        if utt == "u1"
        else {"accuracy": 2.0, "fluency": 2.0, "prosody": 2.0}
    )
    return json.dumps({"alignment": alignment, "scores": scores}, ensure_ascii=False)


def _make_generation_record(item: dict, *, error: str | None = None) -> dict:
    from pinyin_eval.sentence.prompts import (
        D_PROMPT_VERSION,
        TEMPLATE_D,
        prompt_sha256,
    )

    response = "" if error else _prediction_payload(item)
    return {
        "run_id": RUN_ID,
        "request_id": f"D|C1|{item['item_id']}",
        "item_id": item["item_id"],
        "task": "D",
        "condition": "C1",
        "thinking": False,
        "max_tokens": 1536,
        "model": MODEL,
        "prompt_version": D_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(
            "", TEMPLATE_D.format(text=item["text"]), D_PROMPT_VERSION
        ),
        "audio_path": item["audio_path"],
        "response_text": response,
        "reasoning_text": "",
        "finish_reason": None if error else "stop",
        "usage": None if error else {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 16},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "latency_s": 1.0,
        "error": error,
        "retries": 0,
        "timestamp": "2026-11-03T00:00:00+00:00",
        "http_request_id": "chatcmpl-test",
    }


def _build_fixture(tmp_path: Path) -> dict:
    manifest_dir = tmp_path / "manifests"
    data_dir = tmp_path / "d_asr"
    run_dir = tmp_path / "run"
    prompt_dir = tmp_path / "ompal"
    _ = prompt_dir

    b_rows = []
    c_rows = []
    for utt, meta in UTTS.items():
        b_rows.append(
            {
                "item_id": f"B|{utt}",
                "task": "B",
                "source": "ompal",
                "speaker": meta["speaker"],
                "audio_path": f"data/ompal/wav/{utt}.wav",
                "duration_s": 1.0,
                "text": meta["gold"],
                "chars": meta["gold"],
                "pinyin_surface": [],
                "labels": meta["labels"],
                "scores": {},
                "native": False,
                "learner": meta["speaker"],
                "flags": {"native": False, "learner": True},
                "meta": {"utt_id": utt, "learner_id": meta["speaker"]},
            }
        )
        c_rows.append(
            {
                "item_id": f"C|{utt}",
                "task": "C",
                "source": "ompal",
                "speaker": meta["speaker"],
                "audio_path": f"data/ompal/wav/{utt}.wav",
                "duration_s": 1.0,
                "text": meta["gold"],
                "chars": meta["gold"],
                "labels": [],
                "scores": dict(meta["scores"]),
                "native": False,
                "learner": meta["speaker"],
                "flags": {"native": False, "learner": True},
                "meta": {"utt_id": utt},
            }
        )
    _write_jsonl(manifest_dir / "ompal_b_items.jsonl", b_rows)
    _write_jsonl(manifest_dir / "ompal_c_items.jsonl", c_rows)

    d_items = _make_d_items()
    _write_jsonl(manifest_dir / "d_items.jsonl", d_items)
    _write_jsonl(
        manifest_dir / "d_injected_10.jsonl",
        [
            {
                "utt_id": item["meta"]["utt_id"],
                "text_source": "D-inject-10",
                "gold_text": item["meta"]["gold_text"],
                "injected_text": item["text"],
                "ops": item["meta"]["injection_ops"],
            }
            for item in d_items
            if item["meta"]["text_source"] == "D-inject-10"
        ],
    )
    injection_counts = {"sub": 0, "del": 0, "ins": 0}
    n_chars = 0
    for item in d_items:
        if item["meta"]["text_source"] == "D-inject-10":
            _gold, _stats = _gold_alignment_for(item["text"], item["meta"]["utt_id"])
            n_chars += len(item["meta"]["gold_text"])
            for op in item["meta"]["injection_ops"]:
                injection_counts[op["op"]] += 1
    n_perturbed = sum(injection_counts.values())
    _write_json(
        manifest_dir / "d_prep_summary.json",
        {
            "task": "D",
            "n_items": len(d_items),
            "n_utterances": len(UTTS),
            "per_source": {source: len(UTTS) for source in SOURCES},
            "n_asr_missing": 0,
            "injection": {
                "n_items": len(UTTS),
                **injection_counts,
                "n_chars": n_chars,
                "n_perturbed": n_perturbed,
                "ratio": n_perturbed / n_chars if n_chars else None,
            },
        },
    )

    asr_rows = [
        {
            "utt_id": utt,
            "speaker": UTTS[utt]["speaker"],
            "audio_path": f"data/ompal/wav/{utt}.wav",
            "raw_text": ASR_TEXT[utt],
            "text_norm": verifier.normalize_text(ASR_TEXT[utt]),
            "duration_s": 1.0,
            "sample_rate": 16000,
            "wall_s": 0.3,
            "rtf": 0.3,
            "device": "cpu",
            "ok": True,
            "error": None,
            "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "funasr_version": "1.4.21",
            "decode_params": {"batch_size_s": 1},
            "timestamp": "2026-11-03T00:00:00+00:00",
        }
        for utt in UTTS
    ]
    _write_jsonl(data_dir / "asr_outputs.jsonl", asr_rows)
    _write_json(
        data_dir / "d_asr_summary.json",
        {
            "task": "D-asr",
            "n_items": len(UTTS),
            "n_ok": len(UTTS),
            "n_fail": 0,
            "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            "funasr_version": "1.4.21",
            "decode_params": {"batch_size_s": 1},
            "text_quality": {
                "cer_micro": 1.0 / 11.0,
                "sdi": {"substitutions": 1, "deletions": 0, "insertions": 0},
                "normalization_rate": {"rate": 1.0},
            },
        },
    )

    records = []
    for item in d_items:
        records.append(_make_generation_record(item))
    # one transient error record that was retried to success (scorer keeps last)
    target = d_items[0]
    records.insert(0, _make_generation_record(target, error="APIConnectionError: boom"))
    _write_jsonl(run_dir / "generations.jsonl", records)

    from pinyin_eval.sentence.prompts import prompt_fingerprints

    try:
        import subprocess

        git_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover - defensive
        git_commit = ""
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": RUN_ID,
            "created_at": "2026-11-03T00:00:00+00:00",
            "model": MODEL,
            "provider": "dashscope_openai",
            "prompt_version": "sentence-d-v1",
            "prompt_fingerprints": prompt_fingerprints(),
            "config_sha256": "deadbeef",
            "seed": 20261103,
            "pricing": verifier.PRICING,
            "budget_cny": 10.0,
            "conditions": {
                "C1": {"thinking": False, "max_tokens": 256, "max_tokens_by_task": {}},
                "C2": {"thinking": True, "max_tokens": 1024, "max_tokens_by_task": {}},
            },
            "n_planned": len(d_items),
            "tasks": ["D"],
            "d_provenance": {
                "d_prep_summary": {
                    "file": str(manifest_dir / "d_prep_summary.json"),
                    "sha256": verifier.sha256_file(manifest_dir / "d_prep_summary.json"),
                },
                "injected_10": {
                    "file": str(manifest_dir / "d_injected_10.jsonl"),
                    "sha256": verifier.sha256_file(manifest_dir / "d_injected_10.jsonl"),
                },
                "asr": {
                    "outputs_file": str(data_dir / "asr_outputs.jsonl"),
                    "outputs_sha256": verifier.sha256_file(data_dir / "asr_outputs.jsonl"),
                    "summary_file": str(data_dir / "d_asr_summary.json"),
                    "summary_sha256": verifier.sha256_file(data_dir / "d_asr_summary.json"),
                    "model_id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    "funasr_version": "1.4.21",
                    "decode_params": {"batch_size_s": 1},
                    "n_ok": len(UTTS),
                },
                "injection": {"seed": 20261103, "ratio": 0.10,
                              "mix": {"sub": 0.6, "del": 0.2, "ins": 0.2}},
            },
            "items_manifests": {
                "D": {
                    "file": str(manifest_dir / "d_items.jsonl"),
                    "sha256": verifier.sha256_file(manifest_dir / "d_items.jsonl"),
                }
            },
            "git_commit": git_commit,
        },
    )
    return {
        "manifest_dir": manifest_dir,
        "data_dir": data_dir,
        "run_dir": run_dir,
        "items": d_items,
    }


# ---------------------------------------------------------------------------
# Normalisation / alignment
# ---------------------------------------------------------------------------


def test_normalize_text_design_rule() -> None:
    assert verifier.normalize_text(" 他們，都很忙。 ") == "他们都很忙"
    # letters/digits are kept (ASR may emit Latin tokens for mispronounced chars)
    assert verifier.normalize_text("我家有circle人") == "我家有circle人"
    assert verifier.normalize_text("ok 4口人") == "ok4口人"
    assert verifier.normalize_text("") == ""


def test_gold_alignment_semantics() -> None:
    per_char, stats = verifier.gold_alignment(U1, U1, U1_LABELS)
    assert stats["n_aligned"] == 5 and stats["n_deletions_gold"] == 0
    assert [entry["error_type"] for entry in per_char] == [
        "none", "none", "tone", "none", "vowel",
    ]
    # substitution: 都 -> 东 (aligned but text-correct False, text_error True)
    per_char, stats = verifier.gold_alignment("他们东很忙", U1, U1_LABELS)
    assert stats["n_deletions_gold"] == 0
    substitution = [entry for entry in per_char if entry["text_correct"] is False]
    assert substitution and substitution[0]["text_error"] is True
    assert substitution[0]["error_type"] is None
    # deletion: a gold char missing from the input text does not enter T-side metrics
    per_char, stats = verifier.gold_alignment("他们都忙", U1, U1_LABELS)
    assert stats["n_deletions_gold"] == 1
    # insertion: design A.4 says text_error = True (not None)
    per_char, stats = verifier.gold_alignment("他们都很忙啊", U1, U1_LABELS)
    assert stats["n_insertions_text"] == 1
    inserted = [entry for entry in per_char if not entry["aligned"]]
    assert len(inserted) == 1
    assert inserted[0]["text_error"] is True
    assert inserted[0]["text_correct"] is None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_d_reply_design_error_type_vocabulary() -> None:
    """Regression for the phase-A P0: English error_type must not become 'none'."""
    payload = json.dumps(
        {
            "alignment": [
                {"text": "他", "aligned": True, "text_error": False,
                 "error_type": "tone", "score": 3.0},
                {"text": "們", "aligned": True, "text_error": False,
                 "error_type": "consonant", "score": 3.0},
                {"text": "都", "aligned": True, "text_error": False,
                 "error_type": "vowel", "score": 3.0},
                {"text": "很", "aligned": True, "text_error": False,
                 "error_type": "null", "score": 4.0},
                {"text": "忙", "aligned": True, "text_error": False,
                 "error_type": "nonsense", "score": 4.0},
            ],
            "scores": {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0},
        },
        ensure_ascii=False,
    )
    parsed = verifier.parse_d_reply(payload)
    assert parsed["parse_status"] == "ok"
    assert [entry["error_type"] for entry in parsed["alignment"]] == [
        "tone", "consonant", "vowel", "none", None,
    ]
    # Chinese aliases also parse
    parsed_cn = verifier.parse_d_reply(payload.replace("tone", "声调"))
    assert parsed_cn["alignment"][0]["error_type"] == "tone"


def test_parse_d_reply_wrapper_and_failures() -> None:
    inner = {
        "alignment": [
            {"text": "他", "aligned": True, "text_error": False,
             "error_type": "none", "score": 4.0}
        ],
        "scores": {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0},
    }
    parsed = verifier.parse_d_reply(json.dumps([inner], ensure_ascii=False))
    assert parsed["parse_status"] == "ok"
    assert parsed["wrapper_unwrapped"] is True
    assert verifier.parse_d_reply("")["parse_status"] == "empty"
    assert verifier.parse_d_reply("这不是 JSON")["parse_status"] == "parse_error"
    assert verifier.parse_d_reply(
        json.dumps({"alignment": [{"text": "他"}]})
    )["parse_status"] == "parse_error"  # missing scores
    assert verifier.parse_d_reply(
        json.dumps({"alignment": [], "scores": {"accuracy": 4.0}})
    )["parse_status"] == "parse_error"


# ---------------------------------------------------------------------------
# Injection integrity
# ---------------------------------------------------------------------------


def test_apply_injection_ops() -> None:
    gold = list("他们都很忙")
    ops = [
        {"op": "sub", "position": 0, "gold": "他", "new": "她"},
        {"op": "ins", "position": 2, "gold": None, "new": "呀"},
        {"op": "del", "position": 4, "gold": "很", "new": None},
    ]
    chars, problems = verifier.apply_injection_ops(gold, ops)
    assert problems == []
    assert "".join(chars) == "她们呀都忙"  # sub, then insert at 2, then delete raw pos 4
    bad_chars, problems = verifier.apply_injection_ops(
        gold, [{"op": "sub", "position": 0, "gold": "错", "new": "她"}]
    )
    assert problems and "".join(bad_chars) == "他们都很忙"


# ---------------------------------------------------------------------------
# Scoring / aggregation
# ---------------------------------------------------------------------------


def test_score_d_item_known_counts() -> None:
    item = {
        "text": "他们都很忙啊",  # one insertion
        "scores": {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0},
        "meta": {"text_source": "D-inject-10", "utt_id": "u1"},
    }
    gold_per_char, gold_stats = verifier.gold_alignment(
        item["text"], U1, U1_LABELS
    )
    gold = {
        "per_char": gold_per_char,
        "stats": gold_stats,
        "input_chars": verifier.text_chars(item["text"]),
    }
    parsed = verifier.parse_d_reply(
        json.dumps(
            {
                "alignment": [
                    {"text": "他", "aligned": True, "text_error": False,
                     "error_type": "none", "score": 4.0},
                    {"text": "们", "aligned": True, "text_error": False,
                     "error_type": "none", "score": 4.0},
                    {"text": "都", "aligned": True, "text_error": False,
                     "error_type": "tone", "score": 3.0},
                    {"text": "很", "aligned": True, "text_error": False,
                     "error_type": "none", "score": 4.0},
                    {"text": "忙", "aligned": True, "text_error": False,
                     "error_type": "vowel", "score": 2.0},
                    {"text": "啊", "aligned": False, "text_error": True,
                     "error_type": None, "score": None},
                ],
                "scores": {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0},
            },
            ensure_ascii=False,
        )
    )
    row = verifier.score_d_item(item, parsed, gold)
    # 5 aligned chars + 1 inserted char
    assert row["align_tp"] == 5 and row["align_tn"] == 1
    assert row["align_fp"] == 0 and row["align_fn"] == 0
    # text_error: the insertion is a gold positive and the model says text_error=True
    assert row["terr_tp"] == 1
    assert row["terr_tn"] == 5
    # localization on the five text-correct chars: tone TP (都), vowel TP (忙)
    assert row["loc_tone_tp"] == 1 and row["loc_tone_tn"] == 4
    assert row["loc_vowel_tp"] == 1 and row["loc_vowel_tn"] == 4
    assert row["loc_consonant_fp"] == 0


def test_aggregate_d_known_values() -> None:
    items = [item for item in _make_d_items() if item["meta"]["text_source"] == "D-gold"]
    rows = []
    for item in items:
        gold_per_char, stats = _gold_alignment_for(item["text"], item["meta"]["utt_id"])
        gold = {
            "per_char": gold_per_char,
            "stats": stats,
            "input_chars": verifier.text_chars(item["text"]),
        }
        parsed = verifier.parse_d_reply(_prediction_payload(item))
        rows.append(verifier.score_d_item(item, parsed, gold))
    agg = verifier.aggregate_d(rows)
    assert agg["n_items"] == 2 and agg["n_parse_ok"] == 2
    assert agg["alignment"]["f1"] == pytest.approx(1.0)
    assert agg["localization"]["tone"]["macro_f1"] == pytest.approx(1.0)
    assert agg["localization"]["consonant"]["macro_f1"] == pytest.approx(1.0)
    assert agg["localization"]["vowel"]["macro_f1"] == pytest.approx(1.0)
    # predictions 4/2 vs gold 4/3 -> perfect rank correlation, MAE 0.5
    assert agg["scoring"]["accuracy"]["n"] == 2
    assert agg["scoring"]["accuracy"]["pearson"] == pytest.approx(1.0)
    assert agg["scoring"]["accuracy"]["mae"] == pytest.approx(0.5)
    assert agg["alignment_coverage"] == pytest.approx(1.0)


def test_recompute_asr_quality_known_values(tmp_path: Path) -> None:
    b_by_utt = {
        utt: {"chars": meta["gold"], "labels": meta["labels"], "speaker": meta["speaker"]}
        for utt, meta in UTTS.items()
    }
    asr_by_utt = {
        utt: {"raw_text": ASR_TEXT[utt]} for utt in UTTS
    }
    quality = verifier.recompute_asr_quality(b_by_utt, asr_by_utt)
    # u1 identical; u2 one substitution (人 -> 入)
    assert quality["sdi"] == {"sub": 1, "del": 0, "ins": 0}
    assert quality["cer_micro"] == pytest.approx(1.0 / 11.0)
    # OMPAL error chars: u1 都(tone) 忙(vowel); u2 口(tone) -> all still spelled
    assert quality["normalization_rate"] == pytest.approx(1.0)
    assert quality["normalization_n"]["error_chars"] == 3
    assert quality["per_learner_cer"]["n_learners"] == 2


# ---------------------------------------------------------------------------
# Data verification
# ---------------------------------------------------------------------------


def test_verify_d_data_checks(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    data = verifier.verify_d_data(
        fixture["manifest_dir"], fixture["data_dir"], tmp_path
    )
    checks = {check["item"]: check for check in data["checks"]}
    for item in (
        "D-gold text equals OMPAL reference text",
        "items carry OMPAL labels",
        "items carry OMPAL sentence scores",
        "no native items in D",
        "gold alignment recomputed == archived (aligned/text_correct/error_type)",
        "ASR raw text non-empty",
        "D-asr item text == normalise(archived raw ASR)",
        "injection ops reconstruct the injected text",
        "injected text differs from gold",
        "d_injected_10.jsonl ops reconstruct injected_text",
        "d_injected_10.jsonl covers every utterance",
        "ASR CER (micro) recomputed",
        "ASR S/D/I substitutions recomputed",
        "ASR S/D/I deletions recomputed",
        "ASR S/D/I insertions recomputed",
        "ASR normalization rate recomputed",
        "every utterance has an ASR row",
    ):
        assert checks.get(item) is not None, item
        assert checks[item]["ok"] is True, (item, checks[item])
    assert data["injection"]["ops"] == {"sub": 1, "del": 1, "ins": 1}
    assert data["injection"]["n_items"] == 2
    # corrupt one injected item text -> integrity check must fail
    items = [
        json.loads(line)
        for line in (fixture["manifest_dir"] / "d_items.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for item in items:
        if item["meta"]["text_source"] == "D-inject-10":
            item["text"] = item["text"] + "错"
            break
    _write_jsonl(fixture["manifest_dir"] / "d_items.jsonl", items)
    data = verifier.verify_d_data(fixture["manifest_dir"], fixture["data_dir"], tmp_path)
    checks = {check["item"]: check for check in data["checks"]}
    assert checks["injection ops reconstruct the injected text"]["ok"] is False


# ---------------------------------------------------------------------------
# Run verification / cost / acceptance
# ---------------------------------------------------------------------------


def test_verify_run_counts_metrics_cost(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    assert run["records"]["n_expected"] == 6
    assert run["records"]["n_present"] == 6
    assert run["records"]["n_rows"] == 7  # one retried error record
    assert run["records"]["duplicate_request_keys"] == 1
    for source in SOURCES:
        assert run["parse"][source]["n_parse_ok"] == 2
    gold = run["metrics"]["D-gold"]
    assert gold["alignment"]["f1"] == pytest.approx(1.0)
    assert gold["scoring"]["accuracy"]["pearson"] == pytest.approx(1.0)
    asr = run["metrics"]["D-asr"]
    assert asr["text_error"]["f1"] == pytest.approx(1.0)  # substitution caught
    # cost: 6 successful calls x (100 prompt * 0.8 + 10 completion * 2.7) /1e6
    expected_cost = 6 * (100 * 0.8 + 10 * 2.7) / 1_000_000.0
    assert run["cost"]["totals"]["cost_cny"] == pytest.approx(expected_cost)
    assert run["cost"]["totals"]["n_calls"] == 7
    assert run["envelope"]["condition_mismatch"] == 0
    assert run["envelope"]["prompt_sha_mismatch"] == 0
    checks = {check["item"]: check["ok"] for check in run["checks"]}
    for item in (
        "generation records for every planned request",
        "no unexpected request ids",
        "usage/finish_reason/timestamp recorded",
        "C1-only + D prompt version + max_tokens + thinking",
        "per-record prompt fingerprint",
        "model id recorded per call",
        "D total cost ≤ ¥10",
    ):
        assert checks.get(item) is True, (item, checks.get(item))


def test_acceptance_gates() -> None:
    run = {
        "parse": {
            source: {"parse_success_rate": 1.0, "n_items": 10, "n_parse_ok": 10}
            for source in SOURCES
        },
        "metrics": {
            source: {"alignment_coverage": 1.0} for source in SOURCES
        },
        "cost": {"totals": {"cost_cny": 5.0}},
    }
    complete = {"archived": True, "n_archived": 1758, "missing": []}
    acceptance = verifier.acceptance(run, None, asr_completeness=complete)
    assert acceptance["passed"] is True
    # low coverage
    run_low = {
        "parse": run["parse"],
        "metrics": {source: {"alignment_coverage": 0.90} for source in SOURCES},
        "cost": run["cost"],
    }
    assert verifier.acceptance(run_low, None, asr_completeness=complete)["passed"] is False
    # cost above design A.8 threshold
    run_cost = {
        "parse": run["parse"],
        "metrics": run["metrics"],
        "cost": {"totals": {"cost_cny": 11.0}},
    }
    assert verifier.acceptance(run_cost, None, asr_completeness=complete)["passed"] is False
    # incomplete ASR archive
    incomplete = {"archived": True, "n_archived": 1000, "missing": ["u1"]}
    assert verifier.acceptance(run, None, asr_completeness=incomplete)["passed"] is False


# ---------------------------------------------------------------------------
# Reported-vs-independent comparison
# ---------------------------------------------------------------------------


def _reported_run(tmp_path: Path) -> tuple[dict, Path]:
    fixture = _build_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    metrics = run["metrics"]
    acceptance = {
        "parse_success": {
            source: {"rate": 1.0, "n_items": 2, "n_parse_ok": 2, "threshold": 0.95,
                     "passed": True}
            for source in SOURCES
        },
        "alignment_coverage": {
            source: {"value": 1.0, "threshold": 0.95, "passed": True}
            for source in SOURCES
        },
        "cost_cny": {"value": run["cost"]["totals"]["cost_cny"],
                     "threshold": 10.0, "passed": True},
        "asr_archived": {"archived": True, "n_archived": 1758, "n_expected": 1758,
                         "passed": True},
        "passed": True,
    }
    run_for_compare = {
        "metrics": metrics,
        "comparisons": run["comparisons"],
        "acceptance": acceptance,
        "cost": run["cost"],
    }
    # d_metrics.csv
    rows = []
    for source, block in metrics.items():
        rows.append({"source": source, "metric": "alignment_f1",
                     "estimate": block["alignment"]["f1"]})
        rows.append({"source": source, "metric": "text_error_f1",
                     "estimate": block["text_error"]["f1"]})
        for dimension in verifier.D_DIMENSIONS:
            rows.append({"source": source, "metric": f"{dimension}_macro_f1",
                         "estimate": block["localization"][dimension]["macro_f1"]})
        for dimension in (*verifier.D_SCORE_DIMENSIONS, "mean_dimension"):
            rows.append({"source": source, "metric": f"{dimension}_mae",
                         "estimate": block["scoring"][dimension]["mae"]})
    path = fixture["run_dir"] / "d_metrics.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "metric", "estimate",
                                                "ci_lo", "ci_hi"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "ci_lo": "", "ci_hi": ""})
    _write_json(fixture["run_dir"] / "d_acceptance.json", acceptance)
    _write_json(
        fixture["run_dir"] / "d_report_summary.json",
        {
            "metrics": metrics,
            "acceptance": acceptance,
            "cost": run["cost"],
            "comparisons": run["comparisons"],
        },
    )
    return run_for_compare, path


def test_compare_reported_flags_mismatch(tmp_path: Path) -> None:
    run, metrics_path = _reported_run(tmp_path)
    comparison = verifier.compare_reported(metrics_path.parent, run)
    assert comparison["d_metrics_csv"]["available"] is True
    assert comparison["d_metrics_csv"]["n_compared"] > 0
    assert comparison["d_metrics_csv"]["n_mismatch"] == 0
    assert comparison["d_acceptance_json"]["n_mismatch"] == 0
    # corrupt one reported metric by 1pp -> discrepancy above the 0.1pp tolerance
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    header, rows = lines[0], lines[1:]
    corrupted = []
    changed = False
    for row in rows:
        cells = row.split(",")
        if not changed and len(cells) > 2 and cells[2] not in ("", "None"):
            cells[2] = repr(float(cells[2]) - 0.01)
            changed = True
        corrupted.append(",".join(cells))
    assert changed
    metrics_path.write_text("\n".join([header, *corrupted]) + "\n", encoding="utf-8")
    comparison = verifier.compare_reported(metrics_path.parent, run)
    assert comparison["d_metrics_csv"]["n_mismatch"] == 1
    assert comparison["discrepancies"]


def test_check_report_md_tables() -> None:
    metrics = {
        "D-gold": {
            "n_items": 2,
            "n_parse_ok": 2,
            "parse_failure_rate": 0.0,
            "alignment_coverage": 1.0,
            "alignment": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "localization": {
                dimension: {"macro_f1": 1.0, "f1_error": 1.0, "fpr": 0.0, "fnr": 0.0}
                for dimension in ("tone", "consonant", "vowel")
            },
            "scoring": {
                dimension: {"pearson": 0.5, "spearman": 0.4, "mae": 0.5, "rmse": 0.7}
                for dimension in ("accuracy", "fluency", "prosody", "mean_dimension")
            },
        }
    }
    text = "\n".join(
        [
            "| 文本条件 | items | parse-ok | 解析失败率 | 输入字覆盖率 |",
            "| --- | --- | --- | --- | --- |",
            "| D-gold | 2 | 2 | 0.00% | 100.00% |",
            "",
            "| 文本条件 | 对齐 precision | recall | F1 | F1 95% CI |",
            "| --- | --- | --- | --- | --- |",
            "| D-gold | 1.0000 | 1.0000 | 1.0000 | — |",
        ]
    )
    run = {"acceptance": {}}
    block = verifier.check_report_md(text, metrics, run, {})
    assert block["n_issue"] == 0, block["issues"]
    bad = text.replace("| D-gold | 2 | 2 |", "| D-gold | 3 | 2 |")
    block = verifier.check_report_md(bad, metrics, run, {})
    assert block["n_issue"] >= 1


def test_reproducibility_provenance(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    repro = verifier.verify_reproducibility(
        repo_root, fixture["run_dir"], fixture["manifest_dir"]
    )
    checks = {check["item"]: check for check in repro["checks"]}
    for item in (
        "run_manifest.json present",
        "prompt_version == sentence-d-v1",
        "D prompt fingerprint matches code",
        "d_items sha256 recorded",
        "d_provenance: d_prep_summary sha256",
        "d_provenance: d_injected_10 sha256",
        "d_provenance: ASR outputs sha256",
        "d_provenance: ASR summary sha256",
        "d_provenance: ASR model/version/decode recorded",
        "d_provenance: injection seed/ratio/mix recorded",
        "run seed == sentence.d.seed (20261103)",
        "git_commit == current HEAD",
    ):
        assert checks.get(item) is not None, item
        assert checks[item]["ok"] is True, (item, checks[item])
