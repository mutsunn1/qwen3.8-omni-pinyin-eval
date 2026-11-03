"""Offline tests for task D (ASR-text-conditioned alignment & scoring).

Covers text normalisation, gold alignment (match / substitution / insertion /
deletion), deterministic error injection, output parsing, per-item scoring,
aggregate metrics, source degradation comparisons, acceptance and the Markdown
renderer.  No network / no model download.
"""

from __future__ import annotations

import json
from pathlib import Path

from pinyin_eval.config import Config
from pinyin_eval.sentence.d_task import (
    D_PREP_SEED,
    TEXT_SOURCES,
    acceptance_d,
    build_d_items,
    build_substitution_pools,
    compare_d_sources,
    gold_alignment,
    inject_errors,
    map_hyp_entries,
    normalize_text,
    parse_task_d,
    render_d_markdown,
    score_d,
    task_d_metrics,
)
from pinyin_eval.sentence.records import SentenceItem, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

GOLD_TEXT = "他們都很忙"
GOLD_LABELS = [
    {"text": "他", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": True},
    {"text": "們", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "都", "tone": 0, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "很", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
    {"text": "忙", "tone": 1, "consonant": 1, "vowel": 0, "zero_initial": False},
]
GOLD_SCORES = {"accuracy": 4.33, "fluency": 4.67, "prosody": 4.0}


def _d_item(source: str = "D-gold", text: str | None = None, **overrides) -> SentenceItem:
    text = text if text is not None else GOLD_TEXT
    per_char, stats = gold_alignment(text, GOLD_TEXT, GOLD_LABELS)
    meta = {
        "text_source": source,
        "source_slug": {"D-gold": "gold", "D-asr": "asr", "D-inject-10": "inject10"}[source],
        "utt_id": "00200101",
        "gold_text": GOLD_TEXT,
        "gold_alignment": per_char,
        "gold_alignment_stats": stats,
        "injection_ops": [],
    }
    values = dict(
        item_id=f"D|{meta['source_slug']}|00200101",
        task="D",
        source="ompal",
        speaker="SPEAKER02001",
        audio_path="data/sentence/ompal/wav/SPEAKER02001/00200101.wav",
        duration_s=1.9,
        text=text,
        chars=GOLD_TEXT,
        labels=GOLD_LABELS,
        scores=dict(GOLD_SCORES),
        meta=meta,
    )
    values.update(overrides)
    return SentenceItem(**values)


def _perfect_parsed(text: str | None = None) -> dict:
    """A parse-ok reply with correct alignment/text_error/scores for GOLD_TEXT."""
    if text is None:
        text = GOLD_TEXT
    per_char, _ = gold_alignment(text, GOLD_TEXT, GOLD_LABELS)
    alignment = []
    for index, char in enumerate(normalize_text(text)):
        gold = per_char[index] if index < len(per_char) else {}
        error_type = gold.get("error_type")
        alignment.append(
            {
                "text": char,
                "aligned": bool(gold.get("aligned")),
                "text_error": False if gold.get("text_correct") else True,
                "error_type": error_type,
                "score": 4.0,
            }
        )
    return {
        "parse_status": "ok",
        "alignment": alignment,
        "scores": {"accuracy": 4.5, "fluency": 4.5, "prosody": 4.0},
        "detail": "",
    }


# ---------------------------------------------------------------------------
# normalisation / gold alignment
# ---------------------------------------------------------------------------


def test_normalize_text():
    # t2s is part of the design's alignment normalisation -> traditional becomes simplified
    assert normalize_text(" 他們，都很忙。 ") == "他们都很忙"
    assert normalize_text("ta1们 123") == "们"
    assert normalize_text("") == ""


def test_gold_alignment_perfect():
    per_char, stats = gold_alignment(GOLD_TEXT, GOLD_TEXT, GOLD_LABELS)
    assert len(per_char) == len(GOLD_TEXT)
    assert all(entry["aligned"] for entry in per_char)
    assert all(entry["text_correct"] for entry in per_char)
    assert stats["n_insertions_text"] == 0 and stats["n_deletions_gold"] == 0
    # 都 has tone label 0 -> tone error; 忙 has vowel label 0 -> vowel error
    assert per_char[2]["error_type"] == "tone"
    assert per_char[4]["error_type"] == "vowel"
    assert per_char[0]["error_type"] == "none"


def test_gold_alignment_substitution():
    # 們 -> 门 (substitution); the rest matches
    text = "他门都很忙"
    per_char, stats = gold_alignment(text, GOLD_TEXT, GOLD_LABELS)
    assert len(per_char) == len(normalize_text(text))
    assert per_char[0]["aligned"] and per_char[0]["text_correct"]
    assert per_char[1]["aligned"] and not per_char[1]["text_correct"]
    assert per_char[1]["error_type"] is None  # substitution -> no pronunciation label
    assert stats["n_insertions_text"] == 0 and stats["n_deletions_gold"] == 0


def test_gold_alignment_insertion_and_deletion():
    per_ins, stats_ins = gold_alignment("他门都很忙啊", GOLD_TEXT, GOLD_LABELS)
    assert stats_ins["n_insertions_text"] == 1 and stats_ins["n_deletions_gold"] == 0
    inserted = [entry for entry in per_ins if not entry["aligned"]]
    assert len(inserted) == 1 and inserted[0]["text_correct"] is None
    per_del, stats_del = gold_alignment("他们都很", GOLD_TEXT, GOLD_LABELS)
    assert stats_del["n_insertions_text"] == 0 and stats_del["n_deletions_gold"] == 1
    assert len(per_del) == 4 and all(entry["aligned"] for entry in per_del)


# ---------------------------------------------------------------------------
# injection
# ---------------------------------------------------------------------------


def test_injection_deterministic_and_rates():
    pool_texts = [GOLD_TEXT, "我家有四口人", "爸爸媽媽哥哥和我"]
    toned, toneless, insertion = build_substitution_pools(pool_texts)
    first = inject_errors(list(GOLD_TEXT), seed=7, toned_pool=toned,
                          toneless_pool=toneless, insertion_pool=insertion)
    second = inject_errors(list(GOLD_TEXT), seed=7, toned_pool=toned,
                           toneless_pool=toneless, insertion_pool=insertion)
    assert first == second
    chars, ops = first
    assert "".join(chars) != GOLD_TEXT
    kinds = [op["op"] for op in ops]
    assert kinds.count("sub") >= 1
    assert len(ops) >= 2
    # different seed -> different perturbation
    other, _ = inject_errors(list(GOLD_TEXT), seed=8, toned_pool=toned,
                             toneless_pool=toneless, insertion_pool=insertion)
    assert "".join(other) != "".join(chars)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_task_d_ok():
    parsed = _perfect_parsed()
    text = json.dumps(
        {"alignment": parsed["alignment"], "scores": parsed["scores"]}, ensure_ascii=False
    )
    result = parse_task_d(text)
    assert result["parse_status"] == "ok"
    assert len(result["alignment"]) == len(GOLD_TEXT)
    assert result["alignment"][0]["aligned"] is True
    assert result["scores"]["accuracy"] == 4.5


def test_parse_task_d_failures():
    assert parse_task_d("")["parse_status"] == "empty"
    assert parse_task_d("这不是 JSON")["parse_status"] == "parse_error"
    bad_scores = json.dumps({"alignment": [{"text": "他", "aligned": True}]})
    assert parse_task_d(bad_scores)["parse_status"] == "parse_error"


def test_map_hyp_entries_positional_and_aligned():
    hyp = [{"text": ch} for ch in GOLD_TEXT]
    mapped = map_hyp_entries(hyp, list(GOLD_TEXT))
    assert [entry["text"] for entry in mapped] == list(GOLD_TEXT)
    # shorter output -> char alignment
    mapped_short = map_hyp_entries([{"text": "他"}, {"text": "都"}], list(GOLD_TEXT))
    assert mapped_short[0]["text"] == "他"
    assert mapped_short[1] is None


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_score_d_perfect():
    item = _d_item()
    parsed = _perfect_parsed()
    row = score_d(item, parsed, "C1")
    assert row["parse_status"] == "ok"
    assert row["align_fn"] == 0 and row["align_fp"] == 0
    assert row["align_tp"] == len(GOLD_TEXT)
    assert row["terr_tp"] == 0 and row["terr_fp"] == 0
    assert row["output_coverage"] == 1.0
    assert row["n_localizable"] == len(GOLD_TEXT)
    # gold has one tone error (都) and one vowel error (忙); perfect model detects both
    assert row["loc_tone_tp"] == 1 and row["loc_tone_fn"] == 0
    assert row["loc_vowel_tp"] == 1 and row["loc_vowel_fn"] == 0
    assert row["loc_consonant_tp"] == 0 and row["loc_consonant_tn"] == 5
    assert row["pred_accuracy"] == 4.5


def test_score_d_parse_failure():
    item = _d_item()
    row = score_d(item, {"parse_status": "parse_error", "alignment": [], "scores": {}}, "C1")
    assert row["parse_status"] == "parse_error"
    assert row["align_tp"] == 0 and row["n_localizable"] == 0


# ---------------------------------------------------------------------------
# metrics / comparisons / acceptance
# ---------------------------------------------------------------------------


def _metrics_bundle():
    rows = []
    for item in (_d_item("D-gold"), _d_item("D-asr")):
        rows.append(score_d(item, _perfect_parsed(), "C1"))
    by_source = {"D-gold": rows[:1], "D-asr": rows[1:]}
    metrics = {
        source: task_d_metrics(rows, source, n_boot=20, seed=1)
        for source, rows in by_source.items()
    }
    return by_source, metrics


def test_task_d_metrics_known_values():
    _, metrics = _metrics_bundle()
    gold = metrics["D-gold"]
    assert gold["n_items"] == 1 and gold["n_parse_ok"] == 1
    assert gold["alignment"]["f1"] == 1.0
    assert gold["localization"]["tone"]["macro_f1"] is not None
    # perfect model: both classes defined? tone has one error and four ok
    assert gold["localization"]["tone"]["f1_error"] == 1.0


def test_compare_d_sources_and_acceptance():
    by_source, metrics = _metrics_bundle()
    comparisons = compare_d_sources(by_source, baseline="D-gold", n_boot=20, seed=1)
    assert comparisons["baseline"] == "D-gold"
    # single paired utterance -> bootstrap CI may be degenerate but must not crash
    cost = {"totals": {"cost_cny": 1.0}, "per_task_condition": {"D|gold": {"n_calls": 1}}}
    acceptance = acceptance_d(metrics, cost, asr_archived=True)
    assert acceptance["passed"] is True
    bad = acceptance_d(metrics, {"totals": {"cost_cny": 99.0}}, asr_archived=False)
    assert bad["passed"] is False


def test_render_d_markdown_sections():
    _, metrics = _metrics_bundle()
    comparisons = compare_d_sources(_metrics_bundle()[0], baseline="D-gold", n_boot=20, seed=1)
    payload = {
        "run_id": "sentence-d-test",
        "generated_at": "2026-11-03T00:00:00+00:00",
        "sources": list(metrics),
        "manifest": {"model": "qwen3.8-omni-flash", "budget_cny": 10},
        "asr": {"model_id": "paraformer-zh", "device": "cpu", "funasr_version": "1.4.21",
                "quality": {"cer_micro": 0.1, "cer_mean": 0.11, "normalization_rate": 0.5}},
        "metrics": metrics,
        "comparisons": comparisons,
        "cost": {"totals": {"cost_cny": 1.0, "n_calls": 2}},
        "acceptance": acceptance_d(metrics, {"totals": {"cost_cny": 1.0}}, True),
    }
    markdown = render_d_markdown(payload)
    for heading in ("## 1. 运行元信息", "## 4. 发音错误定位", "## 5. 整句评分相关性",
                    "## 8. 验收检查", "## 9. 局限与解释边界"):
        assert heading in markdown
    assert "D-gold" in markdown and "D-asr" in markdown
    assert "✅ 通过" in markdown


# ---------------------------------------------------------------------------
# data preparation (synthetic manifests)
# ---------------------------------------------------------------------------


def test_build_d_items_synthetic(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    asr_dir = tmp_path / "d_asr"
    asr_dir.mkdir()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")  # placeholder; build_d_items does not decode audio
    b_rows = []
    c_rows = []
    asr_rows = []
    for utt in ("00200101", "00200102"):
        text = GOLD_TEXT if utt == "00200101" else "我家有四口人"
        labels = [
            {"text": ch, "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False}
            for ch in text
        ]
        b_rows.append({
            "item_id": f"B|{utt}", "task": "B", "source": "ompal",
            "speaker": "SPEAKER02001", "audio_path": str(wav), "duration_s": 1.0,
            "text": text, "chars": text, "labels": labels, "scores": {},
            "meta": {"utt_id": utt, "learner_id": "SPEAKER02001", "sentence_id": "S00"},
        })
        c_rows.append({
            "item_id": f"C|{utt}", "task": "C", "source": "ompal",
            "speaker": "SPEAKER02001", "audio_path": str(wav), "duration_s": 1.0,
            "text": text, "chars": text, "scores": GOLD_SCORES,
            "meta": {"utt_id": utt},
        })
        asr_rows.append({"utt_id": utt, "ok": True, "raw_text": text, "text_norm": text})
    write_jsonl(manifest_dir / "ompal_b_items.jsonl", b_rows)
    write_jsonl(manifest_dir / "ompal_c_items.jsonl", c_rows)
    write_jsonl(asr_dir / "asr_outputs.jsonl", asr_rows)
    (asr_dir / "d_asr_summary.json").write_text(
        json.dumps({"model_id": "paraformer-zh", "device": "cpu", "n_ok": 2,
                    "quality": {"cer_micro": 0.0}}), encoding="utf-8")
    cfg = Config.load(
        str(ROOT / "configs" / "sentence.yaml"), repo_root=ROOT,
        overrides={"sentence": {"paths": {"manifest_dir": str(manifest_dir)},
                                "d": {"asr_outputs": str(asr_dir / "asr_outputs.jsonl"),
                                      "asr_summary": str(asr_dir / "d_asr_summary.json")}}},
    )
    items, summary = build_d_items(cfg)
    assert summary["n_utterances"] == 2
    assert summary["n_items"] == 2 * len(TEXT_SOURCES)
    assert all(item.task == "D" for item in items)
    assert {item.meta["text_source"] for item in items} == set(TEXT_SOURCES)
    gold_items = [item for item in items if item.meta["text_source"] == "D-gold"]
    assert all(item.meta["gold_alignment_stats"]["n_insertions_text"] == 0 for item in gold_items)
    assert all(item.scores == GOLD_SCORES for item in items)
    # determinism: rebuilding gives the same injected texts
    items2, _ = build_d_items(cfg)
    injected = {item.meta["utt_id"]: item.text for item in items if item.meta["text_source"] == "D-inject-10"}
    injected2 = {item.meta["utt_id"]: item.text for item in items2 if item.meta["text_source"] == "D-inject-10"}
    assert injected == injected2


# ---------------------------------------------------------------------------
# end-to-end mock pipeline (generate -> score -> report)
# ---------------------------------------------------------------------------


def _synthetic_cfg(tmp_path: Path, n_utts: int = 3) -> Config:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    asr_dir = tmp_path / "d_asr"
    asr_dir.mkdir()
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    texts = ["他們都很忙", "我家有四口人", "爸爸媽媽哥哥和我"]
    b_rows, c_rows, asr_rows = [], [], []
    for index in range(n_utts):
        utt = f"0020010{index}"
        text = texts[index % len(texts)]
        labels = [
            {"text": ch, "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False}
            for ch in text
        ]
        b_rows.append({
            "item_id": f"B|{utt}", "task": "B", "source": "ompal",
            "speaker": "SPEAKER02001", "audio_path": str(wav), "duration_s": 1.0,
            "text": text, "chars": text, "labels": labels, "scores": {},
            "meta": {"utt_id": utt, "learner_id": "SPEAKER02001", "sentence_id": "S00"},
        })
        c_rows.append({
            "item_id": f"C|{utt}", "task": "C", "source": "ompal",
            "speaker": "SPEAKER02001", "audio_path": str(wav), "duration_s": 1.0,
            "text": text, "chars": text, "scores": GOLD_SCORES, "meta": {"utt_id": utt},
        })
        asr_rows.append({"utt_id": utt, "ok": True, "raw_text": text, "text_norm": normalize_text(text)})
    write_jsonl(manifest_dir / "ompal_b_items.jsonl", b_rows)
    write_jsonl(manifest_dir / "ompal_c_items.jsonl", c_rows)
    write_jsonl(asr_dir / "asr_outputs.jsonl", asr_rows)
    (asr_dir / "d_asr_summary.json").write_text(
        json.dumps({"model_id": "paraformer-zh", "device": "cpu", "funasr_version": "1.4.21",
                    "n_ok": n_utts, "quality": {"cer_micro": 0.0, "normalization_rate": 0.0}}),
        encoding="utf-8")
    return Config.load(
        str(ROOT / "configs" / "sentence.yaml"), repo_root=ROOT,
        overrides={
            "sentence": {
                "paths": {"manifest_dir": str(manifest_dir), "results_dir": str(tmp_path / "results")},
                "d": {"asr_outputs": str(asr_dir / "asr_outputs.jsonl"),
                      "asr_summary": str(asr_dir / "d_asr_summary.json")},
            }
        },
    )


def test_mock_d_end_to_end(tmp_path):
    import pandas as pd

    from pinyin_eval.sentence.runner import build_plans, run_generation

    cfg = _synthetic_cfg(tmp_path)
    items, _summary = build_d_items(cfg)
    write_jsonl(cfg.path("sentence.paths.manifest_dir") / "d_items.jsonl", items)
    plans = build_plans(cfg, {"D": items}, tasks=["D"], condition_names=["C1"])
    assert len(plans) == len(items)
    summary = run_generation(cfg, plans, "mock-d", mock=True, concurrency=2)
    assert summary["n_called"] == len(plans) and summary["n_errors"] == 0
    run_dir = cfg.path("sentence.paths.results_dir") / "mock-d"
    generations = {}
    for line in (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        generations[f"{record['task']}|{record['condition']}|{record['item_id']}"] = record
    rows = []
    for item in items:
        record = generations[f"D|C1|{item.item_id}"]
        parsed = parse_task_d(record["response_text"])
        assert parsed["parse_status"] == "ok"
        rows.append(score_d(item, parsed, "C1"))
    pd.DataFrame(rows).to_parquet(run_dir / "rows_d.parquet", index=False)
    (run_dir / "d_prep_summary.json").write_text(
        json.dumps({"injection": {"n_items": 3, "sub": 2, "del": 1, "ins": 1}}), encoding="utf-8"
    )
    from pinyin_eval.sentence.d_task import build_d_report

    payload = build_d_report(cfg, "mock-d", n_boot=20)
    report = (run_dir / "d_report.md").read_text(encoding="utf-8")
    assert "## 8. 验收检查" in report and "D-inject-10" in report
    assert (run_dir / "d_metrics.csv").exists()
    assert payload["acceptance"]["passed"] is True
