"""Tests for ``scripts/verify_sentence_run.py`` (the independent verifier).

The verifier is a stand-alone script (not part of the installed package), so it
is loaded by path.  The fixtures build a tiny synthetic run directory from a
hand-computed gold answer so the expected A / B / C metrics, costs, parse
accounting and acceptance gates are known exactly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_sentence_run.py"
_spec = importlib.util.spec_from_file_location("verify_sentence_run", SCRIPT)
assert _spec is not None and _spec.loader is not None
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)

MODEL = "qwen3.8-omni-flash"
RUN_ID = "test-run"
CONDITIONS = {
    "C1": {"thinking": False, "max_tokens": 256, "max_tokens_by_task": {}},
    "C2": {"thinking": True, "max_tokens": 1024, "max_tokens_by_task": {}},
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def _a_item(item_id: str, chars: str, pinyin: list[str], *,
            flags: dict | None = None, speaker: str = "S001") -> dict:
    return {
        "item_id": item_id,
        "task": "A",
        "source": "aishell3",
        "speaker": speaker,
        "audio_path": f"data/audio/{item_id.split('|')[-1]}.wav",
        "duration_s": 1.0,
        "text": chars,
        "chars": chars,
        "pinyin_surface": pinyin,
        "labels": [],
        "scores": {},
        "native": False,
        "learner": "",
        "flags": flags or {"neutral": False, "yibu": False, "erhua": False},
        "meta": {},
    }


def _b_item(item_id: str, text: str, labels: list[dict], *, learner: str = "L1") -> dict:
    return {
        "item_id": item_id,
        "task": "B",
        "source": "ompal",
        "speaker": learner,
        "audio_path": f"data/audio/{item_id.split('|')[-1]}.wav",
        "duration_s": 1.0,
        "text": text,
        "chars": text,
        "pinyin_surface": [],
        "labels": labels,
        "scores": {},
        "native": False,
        "learner": learner,
        "flags": {"native": False, "learner": True},
        "meta": {},
    }


def _c_item(item_id: str, text: str, scores: dict[str, float], *,
            native: bool = False, learner: str = "L1") -> dict:
    return {
        "item_id": item_id,
        "task": "C",
        "source": "ompal",
        "speaker": learner or "NATIVE",
        "audio_path": f"data/audio/{item_id.split('|')[-1]}.wav",
        "duration_s": 1.0,
        "text": text,
        "chars": text,
        "pinyin_surface": [],
        "labels": [],
        "scores": scores,
        "native": native,
        "learner": learner,
        "flags": {"native": native, "learner": not native},
        "meta": {},
    }


def _prompt_sha(task: str, item: dict) -> str:
    from pinyin_eval.sentence.prompts import (
        TEMPLATE_A,
        TEMPLATE_B,
        TEMPLATE_C,
        prompt_sha256,
    )

    if task == "A":
        return prompt_sha256("", TEMPLATE_A.format())
    if task == "B":
        return prompt_sha256("", TEMPLATE_B.format(text=item["text"]))
    return prompt_sha256("", TEMPLATE_C.format(text=item["text"]))


def _record(
    task: str,
    condition: str,
    item: dict,
    response: str,
    *,
    usage: dict | None = None,
    error: str | None = None,
    retries: int = 0,
) -> dict:
    item_id = item["item_id"]
    spec = CONDITIONS[condition]
    return {
        "run_id": RUN_ID,
        "request_id": f"{task}|{condition}|{item_id}",
        "item_id": item_id,
        "task": task,
        "condition": condition,
        "thinking": spec["thinking"],
        "max_tokens": spec["max_tokens"],
        "model": MODEL,
        "prompt_version": "sentence-v1",
        "prompt_sha256": _prompt_sha(task, item),
        "audio_path": item["audio_path"],
        "response_text": response,
        "reasoning_text": "",
        "finish_reason": None if error else "stop",
        "usage": None if error else (usage or {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 16},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }),
        "latency_s": 1.0,
        "error": error,
        "retries": retries,
        "timestamp": "2026-11-01T00:00:00+00:00",
        "http_request_id": "chatcmpl-test",
    }


def _syllables(*items: tuple[str, str, str, int]) -> str:
    payload = [
        {"char": char, "initial": initial, "final": final, "tone": tone}
        for char, initial, final, tone in items
    ]
    return json.dumps({"syllables": payload}, ensure_ascii=False)


def _words(*items: tuple[str, bool, bool, bool]) -> str:
    payload = [
        {"text": text, "tone_ok": tone, "consonant_ok": consonant, "vowel_ok": vowel}
        for text, tone, consonant, vowel in items
    ]
    return json.dumps({"words": payload}, ensure_ascii=False)


def _scores(accuracy: float, fluency: float, prosody: float) -> str:
    return json.dumps(
        {"accuracy": accuracy, "fluency": fluency, "prosody": prosody},
        ensure_ascii=False,
    )


#: gold answer for the synthetic fixture
EXPECTED = {
    "a_c1": {
        "joint_accuracy": 1.0, "initial_accuracy": 1.0, "final_accuracy": 1.0,
        "tone_accuracy": 1.0, "sentence_exact_rate": 1.0, "cer_tone": 0.0,
        "insertion_rate": 0.0, "deletion_rate": 0.0, "alignment_coverage": 1.0,
    },
    "a_c2": {
        "joint_accuracy": 0.6, "initial_accuracy": 1.0, "final_accuracy": 0.8,
        "tone_accuracy": 0.8, "sentence_exact_rate": 0.0, "cer_tone": 0.4,
        "insertion_rate": 0.0, "deletion_rate": 0.0, "alignment_coverage": 1.0,
    },
    "b_tone_c1": {"tp": 0, "fp": 0, "fn": 1, "tn": 2, "macro_f1": 0.4,
                  "fpr": 0.0, "fnr": 1.0},
    "b_consonant_c1": {"tp": 1, "fp": 0, "fn": 0, "tn": 2, "macro_f1": 1.0,
                       "fpr": 0.0, "fnr": 0.0},
    "b_vowel_c1": {"tp": 0, "fp": 1, "fn": 1, "tn": 1, "macro_f1": 0.25,
                   "fpr": 0.5, "fnr": 1.0},
    "b_vowel_c2": {"tp": 0, "fp": 1, "fn": 1, "tn": 1, "macro_f1": 0.25,
                   "fpr": 0.5, "fnr": 1.0},
    "c_pearson": 0.8660254037844387,
    "c_mae": 2.0 / 3.0,
    "cost": 16 * (100 * 0.8 + 10 * 2.7) / 1_000_000.0,
}


def _build_run_fixture(tmp_path: Path, *, corrupt_c2_l3: bool = False) -> dict:
    manifest_dir = tmp_path / "manifests"
    run_dir = tmp_path / "run"
    a_items = [
        _a_item("A|u1", "你好", ["ni3", "hao3"]),
        _a_item(
            "A|u2", "中国人", ["zhong1", "guo2", "ren2"],
            flags={"neutral": True, "yibu": False, "erhua": False},
        ),
    ]
    b_items = [
        _b_item(
            "B|b1", "他们",
            [
                {"text": "他", "tone": 1, "consonant": 1, "vowel": 1,
                 "zero_initial": False},
                {"text": "们", "tone": 0, "consonant": 1, "vowel": 0,
                 "zero_initial": False},
            ],
        ),
        _b_item(
            "B|b2", "我",
            [{"text": "我", "tone": 1, "consonant": 0, "vowel": 1,
              "zero_initial": True}],
        ),
    ]
    c_items = [
        _c_item("C|l1", "你好", {"accuracy": 4.0, "fluency": 4.0, "prosody": 4.0},
                learner="L1"),
        _c_item("C|l2", "你好", {"accuracy": 3.0, "fluency": 3.0, "prosody": 3.0},
                learner="L2"),
        _c_item("C|l3", "你好", {"accuracy": 5.0, "fluency": 5.0, "prosody": 5.0},
                learner="L3"),
        _c_item("C|n1", "你好", {"accuracy": 5.0, "fluency": 5.0, "prosody": 5.0},
                native=True, learner=""),
    ]
    native_items = [c_items[-1]]
    _write_jsonl(manifest_dir / "aishell3_items.jsonl", a_items)
    _write_jsonl(manifest_dir / "ompal_b_items.jsonl", b_items)
    _write_jsonl(manifest_dir / "ompal_c_items.jsonl", c_items[:-1])
    _write_jsonl(manifest_dir / "ompal_native_items.jsonl", native_items)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": RUN_ID,
            "model": MODEL,
            "conditions": CONDITIONS,
            "n_planned": (len(a_items) + len(b_items) + len(c_items)) * 2,
        },
    )
    records: list[dict] = []
    # task A
    for condition in verifier.CONDITIONS:
        if condition == "C1":
            records.append(_record("A", condition, a_items[0],
                                   _syllables(("你", "n", "i", 3), ("好", "h", "ao", 3))))
            records.append(_record("A", condition, a_items[1], _syllables(
                ("中", "zh", "ong", 1), ("国", "g", "uo", 2), ("人", "r", "en", 2))))
        else:
            records.append(_record("A", condition, a_items[0],
                                   _syllables(("你", "n", "i", 3), ("好", "h", "iao", 3))))
            records.append(_record("A", condition, a_items[1], _syllables(
                ("中", "zh", "ong", 1), ("国", "g", "uo", 2), ("人", "r", "en", 3))))
    # task B
    records.append(_record("B", "C1", b_items[0],
                           _words(("他", True, True, True), ("们", True, True, True))))
    records.append(_record("B", "C1", b_items[1], _words(("我", True, False, False))))
    records.append(_record("B", "C2", b_items[0], _words(("他", True, True, True))))
    records.append(_record("B", "C2", b_items[1], _words(("我", True, True, True))))
    # task C
    c1_pred = {4: 4.0, 3: 2.0, 5: 4.0}
    for item in c_items:
        gold = item["scores"]["accuracy"]
        records.append(_record(
            "C", "C1", item,
            _scores(*([4.8] * 3) if item["native"] else [c1_pred[int(gold)]] * 3),
        ))
    for item in c_items:
        gold = item["scores"]["accuracy"]
        if item["item_id"] == "C|l3" and corrupt_c2_l3:
            records.append(_record("C", "C2", item, "抱歉，无法评分。"))
        else:
            records.append(_record(
                "C", "C2", item,
                _scores(*([4.8] * 3) if item["native"] else [c1_pred[int(gold)]] * 3),
            ))
    _write_jsonl(run_dir / "generations.jsonl", records)
    return {
        "manifest_dir": manifest_dir,
        "run_dir": run_dir,
        "a_items": a_items,
        "b_items": b_items,
        "c_items": c_items,
    }


# ---------------------------------------------------------------------------
# Task A recomputation
# ---------------------------------------------------------------------------


def test_task_a_metrics_known_values(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    c1 = run["metrics"]["A"]["C1"]
    c2 = run["metrics"]["A"]["C2"]
    assert c1["n_items"] == 2
    assert c1["n_parse_ok"] == 2
    assert c1["n_gold"] == 5
    for metric, expected in EXPECTED["a_c1"].items():
        assert c1["headline"][metric] == pytest.approx(expected), metric
    for metric, expected in EXPECTED["a_c2"].items():
        assert c2["headline"][metric] == pytest.approx(expected), metric
    # special subsets: neutral = u2 only, stable = u1 only
    assert c1["subsets"]["neutral"]["n_items"] == 1
    assert c1["subsets"]["neutral"]["joint_accuracy"] == pytest.approx(1.0)
    assert c1["subsets"]["stable"]["n_items"] == 1
    assert c1["subsets"]["stable"]["joint_accuracy"] == pytest.approx(1.0)
    # N1 diagnostic has no 一/不 in this fixture -> identical to N0 joint accuracy
    assert c1["n1"]["n1_joint_accuracy"] == pytest.approx(1.0)
    assert run["parse"]["A"]["C1"]["n_ok"] == 2


# ---------------------------------------------------------------------------
# Task B confusion recomputation
# ---------------------------------------------------------------------------


def test_task_b_confusion_known_values(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    b_c1 = run["metrics"]["B"]["C1"]
    tone = b_c1["tone"]
    assert tone["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 2}
    for metric in ("macro_f1", "fpr", "fnr"):
        assert tone[metric] == pytest.approx(EXPECTED["b_tone_c1"][metric]), metric
    assert b_c1["consonant"]["confusion"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 2}
    assert b_c1["consonant"]["macro_f1"] == pytest.approx(1.0)
    assert b_c1["vowel"]["confusion"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 1}
    assert b_c1["vowel"]["macro_f1"] == pytest.approx(0.25)
    assert b_c1["vowel"]["fpr"] == pytest.approx(0.5)
    assert b_c1["alignment_coverage"] == pytest.approx(1.0)
    # zero-initial consonant sub-counts: only 我 (gold consonant 0) is zero-initial
    assert b_c1["zero_initial_consonant"]["confusion"] == {
        "tp": 1, "fp": 0, "fn": 0, "tn": 0
    }
    assert b_c1["tone"]["n_items"] == 2
    # C2: b1 lost 们 (deletion -> false negatives on every dimension), b2 is complete
    b_c2 = run["metrics"]["B"]["C2"]
    assert b_c2["tone"]["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 2}
    assert b_c2["consonant"]["confusion"] == {"tp": 0, "fp": 0, "fn": 2, "tn": 1}
    assert b_c2["vowel"]["confusion"] == {"tp": 0, "fp": 0, "fn": 1, "tn": 2}
    assert b_c2["alignment_coverage"] == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Task C correlation / MAE / native anchor
# ---------------------------------------------------------------------------


def test_task_c_pearson_mae_native(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    c1 = run["metrics"]["C"]["C1"]
    assert c1["n_items"] == 4
    for dimension in ("accuracy", "fluency", "prosody"):
        assert c1[dimension]["n"] == 3
        assert c1[dimension]["pearson"] == pytest.approx(EXPECTED["c_pearson"])
        assert c1[dimension]["mae"] == pytest.approx(EXPECTED["c_mae"])
        assert c1[dimension]["rmse"] == pytest.approx((2.0 / 3.0) ** 0.5)
        assert c1[dimension]["spearman"] == pytest.approx(EXPECTED["c_pearson"])
    assert c1["alignment_coverage"] == pytest.approx(1.0)
    native = c1["native_anchor"]
    assert native["n"] == 1
    assert native["mean_gold"] == pytest.approx(5.0)
    assert native["mean_pred"] == pytest.approx(4.8)


# ---------------------------------------------------------------------------
# Cost / parse accounting / acceptance
# ---------------------------------------------------------------------------


def test_cost_parse_success_and_acceptance(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    assert run["cost"]["totals"]["n_calls"] == 16
    assert run["cost"]["totals"]["cost_cny"] == pytest.approx(EXPECTED["cost"])
    acceptance = run["acceptance"]
    assert acceptance["passed"] is True
    for task in ("A", "B", "C"):
        assert acceptance["parse_success"][task]["rate"] == pytest.approx(1.0)
        assert acceptance["alignment_coverage"][task]["passed"] is True
    assert acceptance["cost_cny"]["passed"] is True
    assert run["records"]["n_expected"] == 16
    assert run["records"]["n_present"] == 16
    assert run["envelope"]["missing_usage"] == 0


def test_acceptance_fails_on_parse_error(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path, corrupt_c2_l3=True)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    acceptance = run["acceptance"]
    assert acceptance["passed"] is False
    # C: 7 of 8 calls parsed -> below the 95% gate
    assert acceptance["parse_success"]["C"]["rate"] == pytest.approx(7.0 / 8.0)
    assert acceptance["parse_success"]["C"]["passed"] is False
    assert run["parse"]["C"]["C2"]["status_counts"]["refused"] == 1
    # C2 metrics use only parse-ok rows: 2 learner pairs (4,4) / (3,2) -> r = 1
    c2 = run["metrics"]["C"]["C2"]
    assert c2["accuracy"]["n"] == 2
    assert c2["accuracy"]["pearson"] == pytest.approx(1.0)
    assert c2["accuracy"]["mae"] == pytest.approx(0.5)


def test_missing_generation_counts_as_failure(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    # drop one A|C1 record -> the planned-call gate must fail
    path = fixture["run_dir"] / "generations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["request_id"] != "A|C1|A|u1"]
    _write_jsonl(path, rows)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    assert run["parse"]["A"]["C1"]["n_missing"] == 1
    assert run["acceptance"]["parse_success"]["A"]["rate"] == pytest.approx(3.0 / 4.0)
    assert run["acceptance"]["parse_success"]["A"]["passed"] is False


# ---------------------------------------------------------------------------
# Comparison against reported artifacts
# ---------------------------------------------------------------------------


def _write_metrics_csv(path: Path, rows: list[tuple]) -> None:
    header = ["task", "condition", "scope", "subset", "dimension", "metric",
              "estimate", "ci_lo", "ci_hi", "n"]
    lines = [",".join(header)]
    for key, value in rows:
        lines.append(",".join([*key, repr(value), "", "", ""]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_compare_reported_metrics_csv(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    lookup = verifier._metric_lookup(run["metrics"])
    keys = [
        ("A", "C1", "headline", "", "joint", "joint_accuracy"),
        ("A", "C2", "headline", "", "joint", "joint_accuracy"),
        ("B", "C1", "dimension", "", "tone", "macro_f1"),
        ("C", "C1", "dimension", "", "accuracy", "pearson"),
        ("C", "C1", "native_anchor", "", "", "mean_gold"),
    ]
    for key in keys:
        assert key in lookup
    _write_metrics_csv(
        fixture["run_dir"] / "metrics.csv",
        [(key, lookup[key]) for key in keys],
    )
    comparison = verifier.compare_reported(fixture["run_dir"], run)
    assert comparison["metrics_csv"]["n_compared"] == len(keys)
    assert comparison["metrics_csv"]["n_mismatch"] == 0
    # corrupt one reported number by 1pp -> must be flagged
    _write_metrics_csv(
        fixture["run_dir"] / "metrics.csv",
        [
            (keys[0], lookup[keys[0]] - 0.01),
            (keys[1], lookup[keys[1]]),
            (keys[2], lookup[keys[2]]),
            (keys[3], lookup[keys[3]]),
            (keys[4], lookup[keys[4]]),
        ],
    )
    comparison = verifier.compare_reported(fixture["run_dir"], run)
    assert comparison["metrics_csv"]["n_mismatch"] == 1
    mismatch = comparison["metrics_csv"]["mismatches"][0]
    assert mismatch["key"] == "A|C1|headline||joint|joint_accuracy"
    assert mismatch["delta_pp"] == pytest.approx(1.0, abs=1e-6)


def test_compare_reported_report_summary_and_acceptance(tmp_path: Path) -> None:
    fixture = _build_run_fixture(tmp_path)
    run = verifier.verify_run(fixture["run_dir"], fixture["manifest_dir"])
    run_dir = fixture["run_dir"]
    summary = {
        "headline": {
            "A": {
                "n_items": 2, "n_parse_ok": 2,
                "joint_accuracy": 1.0, "tone_accuracy": 1.0,
                "sentence_exact_rate": 1.0, "cer_tone": 0.0,
            },
            "B": {
                "n_items": 2, "tone_macro_f1": 0.4,
                "consonant_macro_f1": 1.0, "vowel_macro_f1": 0.25,
            },
            "C": {
                "n_items": 4,
                "accuracy_pearson": EXPECTED["c_pearson"],
                "fluency_pearson": EXPECTED["c_pearson"],
                "prosody_pearson": EXPECTED["c_pearson"],
                "native_anchor": {"n": 1, "mean_gold": 5.0, "mean_pred": 4.8},
            },
        },
        "acceptance": {
            "passed": True,
            "parse_success": {"A": 1.0, "B": 1.0, "C": 1.0},
            "alignment_coverage": {"A": 1.0, "B": 1.0, "C": 1.0},
            "cost_cny": EXPECTED["cost"],
        },
        "usage_totals": {"cost_cny": EXPECTED["cost"]},
    }
    _write_json(run_dir / "report_summary.json", summary)
    _write_json(
        run_dir / "acceptance.json",
        {
            "passed": True,
            "parse_success": {
                task: {"rate": 1.0, "passed": True} for task in ("A", "B", "C")
            },
            "alignment_coverage": {
                task: {"value": 1.0, "passed": True} for task in ("A", "B", "C")
            },
            "cost_cny": {"value": EXPECTED["cost"], "passed": True},
        },
    )
    comparison = verifier.compare_reported(run_dir, run)
    assert comparison["report_summary_json"]["n_mismatch"] == 0, (
        comparison["report_summary_json"]["mismatches"]
    )
    assert comparison["acceptance_json"]["n_mismatch"] == 0
    # corrupt the reported headline tone accuracy -> flagged above the tolerance
    summary["headline"]["A"]["tone_accuracy"] = 0.5
    _write_json(run_dir / "report_summary.json", summary)
    comparison = verifier.compare_reported(run_dir, run)
    mismatches = comparison["report_summary_json"]["mismatches"]
    assert any(item["key"] == "headline.A.tone_accuracy" for item in mismatches)


# ---------------------------------------------------------------------------
# Data-prep verification helpers
# ---------------------------------------------------------------------------


def test_aishell_label_and_spk_parsers() -> None:
    lines, malformed = verifier._parse_aishell_labels(
        "# comment\n"
        "SSB0001|ni3 hao3 %|你好%\n"
        "SSB0002|wo3 ai4 $|我爱 $\n"
        "broken line\n"
    )
    assert len(lines) == 2
    assert lines[0]["tokens"] == ["ni3", "hao3"]
    assert lines[0]["chars"] == "你好"
    assert lines[1]["chars"] == "我爱"
    assert len(malformed) == 1
    spk = verifier._parse_spk_info(
        "# voice-file name; age group; gender; accent\n\nSSB0001\tB\tfemale\tnorth\n"
    )
    assert spk == {"SSB0001": {"age": "B", "gender": "female", "accent": "north"}}


def test_alignment_and_edit_distance() -> None:
    pairs, insertions, deletions = verifier.align_sequences(["a", "b", "c"], ["a", "c", "d"])
    assert pairs == [(0, 0), (1, 1), (2, 2)]  # b->c substitution, c->d substitution
    assert insertions == 0
    assert deletions == 0
    pairs, insertions, deletions = verifier.align_sequences(["a", "b", "c"], ["a", "c"])
    assert deletions == 1
    assert insertions == 0
    pairs, insertions, deletions = verifier.align_sequences(["a"], ["a", "b"])
    assert insertions == 1
    assert deletions == 0
    assert verifier.edit_distance(["a", "b"], ["a", "c"]) == 1
    assert verifier.edit_distance([], ["a"]) == 1


def test_zero_initial_recompute() -> None:
    items = [
        {
            "item_id": "B|1",
            "text": "你好",
            "labels": [{"zero_initial": False}, {"zero_initial": False}],
        }
    ]
    result = verifier._verify_zero_initials(items)
    assert result["mismatches"] == 0
    assert result["checked"] == 2
    items[0]["labels"][1]["zero_initial"] = True
    assert verifier._verify_zero_initials(items)["mismatches"] == 1


def test_audio_probe_and_check(tmp_path: Path) -> None:
    good = tmp_path / "good.wav"
    sf.write(str(good), np.zeros(1600, dtype="float32"), 16000, subtype="PCM_16")
    probe = verifier._probe_audio(good)
    assert probe["ok"] is True
    assert probe["samplerate"] == 16000
    assert probe["channels"] == 1
    assert probe["subtype"] == "PCM_16"
    items = [{"audio_path": "good.wav", "duration_s": 0.1}]
    checks: list[dict] = []
    audio = verifier._check_audio(
        checks, "X", items, tmp_path,
        expected_format={"samplerate": 16000, "channels": 1, "subtype": "PCM_16"},
        check_audio=True,
    )
    assert audio["ok"] == 1
    assert checks[-1]["ok"] is True
    # a wrong sample rate must be reported as a format violation
    bad = tmp_path / "bad.wav"
    sf.write(str(bad), np.zeros(4410, dtype="float32"), 44100, subtype="PCM_16")
    checks = []
    audio = verifier._check_audio(
        checks, "X", [{"audio_path": "bad.wav", "duration_s": 0.1}], tmp_path,
        expected_format={"samplerate": 16000, "channels": 1, "subtype": "PCM_16"},
        check_audio=True,
    )
    assert audio["format_violations"]
    assert checks[-1]["ok"] is False


def _make_ompal_data_tree(tmp_path: Path) -> Path:
    """Minimal data tree with the OMPAL source anomalies + manifests.

    Counts mirror the real corpus (1,768 raw learner rows, 10 removals ->
    1,758; 82 native rows with 3 corrupted slots + 4 text/slot mismatches) so the
    verifier's design-expected checks actually execute against known data.
    """
    data_dir = tmp_path / "data"
    repo_dir = data_dir / "ompal" / "repo"
    manifest_dir = data_dir / "manifests"

    def normal_learner(utt_id: str) -> dict:
        return {
            "text": "你好",
            "accuracy": 4.0,
            "fluency": 4.0,
            "prosody": 4.0,
            "words": [
                {"text": ["你"], "tone": 1, "phoneme_consonant": 1,
                 "phoneme_vowel": 1},
                {"text": ["好"], "tone": 1, "phoneme_consonant": 1,
                 "phoneme_vowel": 1},
            ],
        }

    def corrupted_learner(utt_id: str) -> dict:
        record = normal_learner(utt_id)
        record["text"] = "你"
        record["words"] = [
            {"text": ["你", "好"], "tone": 1, "phoneme_consonant": 1,
             "phoneme_vowel": 1},
        ]
        return record

    def mismatch_learner(utt_id: str) -> dict:
        record = normal_learner(utt_id)
        record["text"] = "你好吗"
        return record

    learner: dict[str, dict] = {}
    for index in range(1758):
        learner[f"002{index:05d}"] = normal_learner(f"002{index:05d}")
    corrupted_ids = [f"003{index:05d}" for index in range(9)]
    mismatch_id = "00400001"
    for utt_id in corrupted_ids:
        learner[utt_id] = corrupted_learner(utt_id)
    learner[mismatch_id] = mismatch_learner(mismatch_id)
    assert len(learner) == 1768

    native: dict[str, dict] = {}
    for index in range(78):
        utt_id = f"001{index:05d}"
        native[utt_id] = {
            "text": "你好",
            "accuracy": 5.0,
            "fluency": 5.0,
            "prosody": 5.0,
            "words": [
                {"text": ["你"], "tone": 1, "phoneme_consonant": 1,
                 "phoneme_vowel": 1},
                {"text": ["好"], "tone": 1, "phoneme_consonant": 1,
                 "phoneme_vowel": 1},
            ],
        }
    for offset in range(3):
        utt_id = f"0019000{offset}"
        record = json.loads(json.dumps(native[f"001{offset:05d}"]))
        record["text"] = "你"
        record["words"] = [
            {"text": ["你", "好"], "tone": 1, "phoneme_consonant": 1,
             "phoneme_vowel": 1}
        ]
        native[utt_id] = record
    pure_mismatch = json.loads(json.dumps(native["00100003"]))
    pure_mismatch["text"] = "你好吗"
    native["00190004"] = pure_mismatch
    assert len(native) == 82

    _write_json(repo_dir / "non-native_scores.json", learner)
    _write_json(repo_dir / "native_scores.json", native)

    excluded = set(corrupted_ids) | {mismatch_id}
    b_items: list[dict] = []
    c_items: list[dict] = []
    for index, (utt_id, record) in enumerate(
        sorted((key, value) for key, value in learner.items() if key not in excluded)
    ):
        learner_id = f"LK{index % 46:02d}"
        text = record["text"]
        labels = []
        for word in record["words"]:
            slot = word["text"][0]
            labels.append(
                {
                    "text": slot,
                    "tone": int(word["tone"]),
                    "consonant": int(word["phoneme_consonant"]),
                    "vowel": int(word["phoneme_vowel"]),
                    "zero_initial": False,
                }
            )
        item_id = f"B|{utt_id}"
        b_items.append(
            {
                "item_id": item_id, "task": "B", "source": "ompal",
                "speaker": learner_id, "audio_path": f"wav/{utt_id}.wav",
                "duration_s": 1.0, "text": text, "chars": text,
                "pinyin_surface": [], "labels": labels, "scores": {},
                "native": False, "learner": learner_id,
                "flags": {"native": False, "learner": True}, "meta": {},
            }
        )
        c_items.append(
            {
                "item_id": f"C|{utt_id}", "task": "C", "source": "ompal",
                "speaker": learner_id, "audio_path": f"wav/{utt_id}.wav",
                "duration_s": 1.0, "text": text, "chars": text,
                "pinyin_surface": [], "labels": [],
                "scores": {
                    "accuracy": record["accuracy"],
                    "fluency": record["fluency"],
                    "prosody": record["prosody"],
                },
                "native": False, "learner": learner_id,
                "flags": {"native": False, "learner": True}, "meta": {},
            }
        )
    native_items: list[dict] = []
    for utt_id, record in sorted(native.items()):
        native_items.append(
            {
                "item_id": f"C|{utt_id}", "task": "C", "source": "ompal",
                "speaker": "NATIVE", "audio_path": f"wav/{utt_id}.wav",
                "duration_s": 1.0, "text": record["text"],
                "chars": record["text"], "pinyin_surface": [], "labels": [],
                "scores": {"accuracy": 5.0, "fluency": 5.0, "prosody": 5.0},
                "native": True, "learner": "",
                "flags": {"native": True, "learner": False}, "meta": {},
            }
        )
    _write_jsonl(manifest_dir / "ompal_b_items.jsonl", b_items)
    _write_jsonl(manifest_dir / "ompal_c_items.jsonl", c_items)
    _write_jsonl(manifest_dir / "ompal_native_items.jsonl", native_items)
    _write_jsonl(manifest_dir / "aishell3_items.jsonl", [])
    _write_json(
        manifest_dir / "prep_summary.json",
        {
            "sources": {},
            "ompal": {
                "cleaning": {
                    "excluded_learner": len(excluded),
                    "exclusions": [
                        {"utt_id": utt_id, "reasons": ["corrupted_word_slot"]}
                        for utt_id in sorted(excluded)
                    ],
                }
            },
        },
    )
    audio_paths = {
        item["audio_path"] for item in (*b_items, *native_items)
    }
    _write_json(
        data_dir / "audio_sources.json",
        {path: f"https://example.invalid/{path}" for path in sorted(audio_paths)},
    )
    return manifest_dir


def test_verify_data_ompal_anomalies(tmp_path: Path) -> None:
    manifest_dir = _make_ompal_data_tree(tmp_path)
    data = verifier.verify_data(manifest_dir, tmp_path, check_audio=False)
    checks = {check["item"]: check["ok"] for check in data["checks"]}
    for item in (
        "OMPAL raw learner rows",
        "OMPAL raw native rows",
        "OMPAL learner corrupted word-slot rows",
        "OMPAL learner text/slot mismatch rows",
        "OMPAL learner exclusions (union)",
        "OMPAL cleaned learner rows",
        "OMPAL native corrupted word-slot rows",
        "OMPAL native text/slot mismatch rows",
        "OMPAL cleaning count matches prep",
        "B item count",
        "C item count",
        "native anchor item count",
        "B learners",
        "B/C same item ids",
        "B label texts == item chars",
        "B labels match source word labels",
        "B per-char labels in {0,1}",
        "B scores empty",
        "C scores in [1,5] and equal 3-rater means",
        "C learner items with scores",
        "B/C texts identical",
        "native items: text/5.0 scores/native flag",
        "native scores all 5.0",
        "audio_sources.json covers every manifest wav",
    ):
        assert checks.get(item) is True, (item, checks.get(item))
    assert len(data["ompal"]["excluded"]) == 10
    assert data["ompal"]["cleaned_learner"] == 1758
    assert data["ompal"]["zero_initial_recompute"]["mismatches"] == 0


def test_parser_unwraps_single_object_array() -> None:
    """The one-element array wrapper (observed on 4 task-B C2 replies) is accepted."""
    wrapped = json.dumps([
        {"words": [
            {"text": "我", "tone_ok": True, "consonant_ok": True, "vowel_ok": True}
        ]}
    ])
    lenient = verifier.classify_reply("B", wrapped)
    assert lenient["parse_status"] == "ok"
    assert lenient["words"][0]["text"] == "我"
    strict = verifier.classify_reply("B", wrapped, unwrap_wrapper=False)
    assert strict["parse_status"] == "parse_error"
    # a bare task-A array (design §4.1) must stay valid in both modes
    bare = json.dumps([{"char": "你", "initial": "n", "final": "i", "tone": 3}])
    for parsed in (
        verifier.classify_reply("A", bare),
        verifier.classify_reply("A", bare, unwrap_wrapper=False),
    ):
        assert parsed["parse_status"] == "ok"
        assert parsed["syllables"][0]["initial"] == "n"
    # a multi-element wrapper is not silently accepted
    two = json.dumps([{"words": []}, {"words": []}])
    assert verifier.classify_reply("B", two)["parse_status"] != "ok"
