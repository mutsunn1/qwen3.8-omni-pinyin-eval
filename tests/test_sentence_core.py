"""Offline tests for the sentence-level core modules (Lead-owned).

Covers: N0/N1 pinyin normalisation, erhua split, char alignment, prompt
rendering/fingerprints, output parsing, request plans, the sentence mock client
and an end-to-end mock generation run (no network, no API key).
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from pinyin_eval.config import Config
from pinyin_eval.sentence.normalize import (
    align_char_sequences,
    align_gold,
    apply_n1_sandhi,
    edit_distance,
    is_erhua_surface,
    normalize_pinyin_token,
    split_surface,
    t2s,
)
from pinyin_eval.sentence.parsing import (
    parse_task_a,
    parse_task_b,
    parse_task_c,
)
from pinyin_eval.sentence.prompts import build_prompt, prompt_fingerprints, prompt_sha256
from pinyin_eval.sentence.records import (
    GenerationRecord,
    SentenceItem,
    make_request_id,
    write_jsonl,
)
from pinyin_eval.sentence.runner import (
    build_manifest,
    build_plans,
    cost_summary,
    estimate_cost,
    load_items,
    run_generation,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wav(path: Path, seconds: float = 0.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(44100 * seconds)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(44100)
        fh.writeframes(b"\x00\x00" * frames)
    return path


def _item(task: str, **kwargs) -> SentenceItem:
    base = dict(
        item_id=f"{task}|t1",
        task=task,
        source="aishell3" if task == "A" else "ompal",
        speaker="SSB0005" if task == "A" else "SPEAKER01001",
        audio_path="audio.wav",
        duration_s=1.0,
        text="你好",
    )
    base.update(kwargs)
    return SentenceItem(**base)


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_normalize_pinyin_token_tone_and_umlaut():
    syl = normalize_pinyin_token("nv3")
    assert (syl.initial, syl.final, syl.tone) == ("n", "ü", 3)
    assert normalize_pinyin_token("nǚ").final == "ü"
    assert normalize_pinyin_token("lve4").final == "üe"
    assert normalize_pinyin_token("ng2").tone == 2  # interjection stays parsable
    marked = normalize_pinyin_token("nǐ")
    assert (marked.initial, marked.final, marked.tone) == ("n", "i", 3)


def test_split_surface_zero_initial_and_apical():
    assert split_surface("zhi") == ("zh", "i")
    assert split_surface("si") == ("s", "i")
    assert split_surface("wo") == ("", "uo")
    assert split_surface("yue") == ("", "üe")
    assert split_surface("jun") == ("j", "ün")
    assert split_surface("ang") == ("", "ang")


def test_erhua_detection_and_split():
    assert is_erhua_surface("dianr")
    assert is_erhua_surface("hair")
    assert not is_erhua_surface("er")
    assert not is_erhua_surface("r")
    gold = align_gold("点儿", ["dianr3"])
    assert gold is not None
    assert [s.char for s in gold] == ["点", "儿"]
    assert (gold[0].surface, gold[0].tone) == ("dian", 3)
    assert (gold[1].initial, gold[1].final, gold[1].tone) == ("", "er", 5)
    # 孩儿 merged into a single surface keeps the base tone
    gold2 = align_gold("男孩儿", ["nan2", "hair2"])
    assert gold2 is not None
    assert [s.char for s in gold2] == ["男", "孩", "儿"]
    assert gold2[1].tone == 2 and gold2[2].tone == 5


def test_align_gold_rejects_count_mismatch():
    assert align_gold("你好", ["ni3"]) is None
    assert align_gold("你", ["ni3", "hao3"]) is None


def test_n1_sandhi():
    gold = align_gold("一个", ["yi1", "ge4"])
    assert gold is not None
    n1 = apply_n1_sandhi(gold)
    assert (n1[0].tone, n1[1].tone) == (2, 4)
    gold2 = align_gold("一天", ["yi1", "tian1"])
    n2 = apply_n1_sandhi(gold2)
    assert n2[0].tone == 4
    gold3 = align_gold("不对", ["bu4", "dui4"])
    n3 = apply_n1_sandhi(gold3)
    assert n3[0].tone == 2


def test_char_alignment_operations():
    pairs, insertions, deletions = align_char_sequences(["你", "好"], ["你", "好"])
    assert pairs == [(0, 0), (1, 1)] and insertions == 0 and deletions == 0
    pairs, insertions, deletions = align_char_sequences(["你", "好"], ["你"])
    assert pairs == [(0, 0), (1, None)] and insertions == 0 and deletions == 1
    pairs, insertions, deletions = align_char_sequences(["你"], ["你", "好"])
    assert insertions == 1 and deletions == 0
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_t2s_available():
    # OpenCC is an optional dependency; when present it converts traditional text
    assert t2s("繁體字") == "繁体字"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_task_a_ok():
    text = json.dumps({"syllables": [
        {"char": "你", "initial": "n", "final": "i", "tone": 3},
        {"char": "我", "initial": "w", "final": "o", "tone": 3},
    ]}, ensure_ascii=False)
    parsed = parse_task_a(text)
    assert parsed["parse_status"] == "ok"
    assert parsed["syllables"][1] == {"char": "我", "initial": "", "final": "uo", "tone": 3}


def test_parse_task_a_accepts_array_and_fences():
    text = '```json\n[{"char":"你","initial":"n","final":"i","tone":3}]\n```'
    parsed = parse_task_a(text)
    assert parsed["parse_status"] == "ok"
    assert parsed["syllables"][0]["char"] == "你"


def test_parse_task_a_failures():
    assert parse_task_a("")["parse_status"] == "empty"
    assert parse_task_a("抱歉，我无法回答")["parse_status"] == "refused"
    assert parse_task_a("这不是 JSON")["parse_status"] == "parse_error"


def test_parse_task_array_wrapper():
    """The model occasionally wraps the object in a one-element array."""
    wrapped_b = ('[{"words":[{"text":"我","tone_ok":true,"consonant_ok":true,'
                 '"vowel_ok":true}]}]')
    parsed_b = parse_task_b(wrapped_b)
    assert parsed_b["parse_status"] == "ok"
    assert parsed_b["words"][0]["tone_ok"] is True
    wrapped_a = '[{"syllables":[{"char":"你","initial":"n","final":"i","tone":3}]}]'
    parsed_a = parse_task_a(wrapped_a)
    assert parsed_a["parse_status"] == "ok"
    assert parsed_a["syllables"][0]["char"] == "你"


def test_parse_task_b_and_c():
    b = json.dumps({"words": [
        {"text": "他", "tone_ok": True, "consonant_ok": True, "vowel_ok": True},
        {"text": "們", "tone_ok": 0, "consonant_ok": 1, "vowel_ok": False},
    ]}, ensure_ascii=False)
    parsed_b = parse_task_b(b)
    assert parsed_b["parse_status"] == "ok"
    assert parsed_b["words"][1]["tone_ok"] is False
    assert parsed_b["words"][1]["consonant_ok"] is True
    c = '{"accuracy": 4.3, "fluency": 4.7, "prosody": 4.0}'
    parsed_c = parse_task_c(c)
    assert parsed_c["scores"] == {"accuracy": 4.3, "fluency": 4.7, "prosody": 4.0}
    assert parse_task_c('{"accuracy": 4.3}')["parse_status"] == "parse_error"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompt_build_versions():
    item = _item("B", text="繁體句子")
    system, user = build_prompt("B", item)
    assert system == ""
    assert "繁體句子" in user
    assert "tone_ok" in user
    a_system, a_user = build_prompt("A", None)
    assert a_system == ""
    assert "syllables" in a_user
    assert prompt_sha256(system, user) == prompt_sha256(system, user)
    fingerprints = prompt_fingerprints()
    assert set(fingerprints) == {"A", "B", "C", "D"}
    # A sees no target text; changing the item must not change A's prompt
    assert build_prompt("A", item) == build_prompt("A", None)


# ---------------------------------------------------------------------------
# plans / runner
# ---------------------------------------------------------------------------


def _mock_config(tmp_path: Path) -> Config:
    return Config.load(
        str(ROOT / "configs" / "sentence.yaml"),
        repo_root=ROOT,
        overrides={
            "sentence": {
                "paths": {
                    "manifest_dir": str(tmp_path / "manifests"),
                    "results_dir": str(tmp_path / "results"),
                }
            }
        },
    )


def _write_manifests(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    audio = _wav(tmp_path / "wav" / "a1.wav")
    write_jsonl(manifest_dir / "aishell3_items.jsonl", [
        _item("A", item_id="A|t1", audio_path=str(audio), chars="你好",
              pinyin_surface=["ni3", "hao3"],
              flags={"neutral": False, "yibu": False, "erhua": False})
    ])
    write_jsonl(manifest_dir / "ompal_b_items.jsonl", [
        _item("B", item_id="B|t1", audio_path=str(audio), labels=[
            {"text": "你", "tone": 1, "consonant": 1, "vowel": 1, "zero_initial": False},
            {"text": "好", "tone": 0, "consonant": 1, "vowel": 0, "zero_initial": False},
        ])
    ])
    write_jsonl(manifest_dir / "ompal_c_items.jsonl", [
        _item("C", item_id="C|t1", audio_path=str(audio),
              scores={"accuracy": 4.0, "fluency": 4.5, "prosody": 4.0})
    ])


def test_build_plans_and_estimate(tmp_path):
    _write_manifests(tmp_path)
    cfg = _mock_config(tmp_path)
    items = load_items(cfg)
    assert {task: len(rows) for task, rows in items.items()} == {"A": 1, "B": 1, "C": 1}
    plans = build_plans(cfg, items)
    assert len(plans) == 6
    assert {plan.condition for plan in plans} == {"C1", "C2"}
    assert {plan.task for plan in plans} == {"A", "B", "C"}
    by_key = {(plan.task, plan.condition): plan for plan in plans}
    # design defaults with the pilot-calibrated per-task caps (configs/sentence.yaml)
    assert by_key[("C", "C2")].max_tokens == 1024
    assert by_key[("C", "C1")].max_tokens == 256
    assert by_key[("A", "C1")].max_tokens == 1536
    assert by_key[("B", "C1")].max_tokens == 768
    assert by_key[("A", "C2")].max_tokens == 1536
    estimate = estimate_cost(cfg, plans)
    assert estimate["totals"]["n_calls"] == 6
    assert estimate["totals"]["cost_cny"] > 0
    worst = estimate_cost(cfg, plans, worst_case=True)
    assert worst["totals"]["cost_cny"] >= estimate["totals"]["cost_cny"]


def test_manifest_records_reproducibility(tmp_path):
    _write_manifests(tmp_path)
    cfg = _mock_config(tmp_path)
    plans = build_plans(cfg, load_items(cfg))
    manifest = build_manifest(cfg, run_id="r1", plans=plans, budget_cny=20.0)
    assert manifest["model"] == "qwen3.8-omni-flash"
    assert manifest["prompt_version"] == "sentence-v1"
    assert len(manifest["config_sha256"]) == 64
    assert manifest["n_planned"] == 6
    assert manifest["pricing"]["input_per_mtok"] == 0.8


def test_mock_end_to_end_and_resume(tmp_path):
    _write_manifests(tmp_path)
    cfg = _mock_config(tmp_path)
    plans = build_plans(cfg, load_items(cfg))
    summary = run_generation(cfg, plans, "mock-run", mock=True, concurrency=2)
    assert summary["n_called"] == 6 and summary["n_errors"] == 0
    run_dir = tmp_path / "results" / "mock-run"
    rows = [json.loads(line) for line in
            (run_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert (run_dir / "run_manifest.json").exists()
    # resume: a second invocation must not call the API again
    again = run_generation(cfg, plans, "mock-run", mock=True, concurrency=2)
    assert again["n_called"] == 0 and again["n_already_completed"] == 6
    cost = cost_summary(cfg, "mock-run")
    assert cost["totals"]["n_calls"] == 6
    assert cost["totals"]["cost_cny"] > 0


def test_generation_record_roundtrip():
    record = GenerationRecord(
        run_id="r", request_id="A|C1|A|t1", item_id="A|t1", task="A",
        condition="C1", thinking=False, max_tokens=256, model="m",
        prompt_version="sentence-v1", prompt_sha256="x", audio_path="a.wav",
        response_text='{"syllables":[]}', usage={"prompt_tokens": 1},
    )
    assert GenerationRecord.from_dict(record.to_dict()) == record
    assert make_request_id("A", "C1", "A|t1") == "A|C1|A|t1"
