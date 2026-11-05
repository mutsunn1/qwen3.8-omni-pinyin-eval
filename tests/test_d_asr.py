"""Offline unit tests for ``scripts/run_d_asr.py``.

Everything here runs against synthetic fixtures / ``tmp_path``: no network, no ASR model
download, no real corpus file is touched.  Run with::

    .venv/bin/python -m pytest tests/test_d_asr.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_d_asr as dasr  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Import hygiene: importing the module must never pull in funasr / torch
# ---------------------------------------------------------------------------


def test_import_does_not_load_model_stack() -> None:
    """A plain import must not import funasr/torch (and therefore must not download)."""
    code = (
        "import sys; sys.path.insert(0, 'scripts'); import run_d_asr; "
        "assert 'funasr' not in sys.modules, 'funasr imported eagerly'; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly'; "
        "print('clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


# ---------------------------------------------------------------------------
# 2. Text normalisation (NFKC -> strip punctuation/space -> OpenCC t2s)
# ---------------------------------------------------------------------------


def test_t2s_traditional_to_simplified() -> None:
    assert dasr.t2s("他們都很忙") == "他们都很忙"
    assert dasr.t2s("萬事如意") == "万事如意"
    assert dasr.t2s("") == ""
    # simplified text stays put
    assert dasr.t2s("他们都很忙") == "他们都很忙"


def test_normalize_text_punctuation_whitespace_and_t2s() -> None:
    assert dasr.normalize_text("他們都很忙") == "他们都很忙"
    assert dasr.normalize_text(" 他們，都 很忙。") == "他们都很忙"
    assert dasr.normalize_text("他们\t都\n很忙！") == "他们都很忙"
    # NFKC applies first: full-width comma becomes ASCII punctuation and is dropped
    assert dasr.normalize_text("你好，世界") == "你好世界"
    assert dasr.normalize_text("ＡＢＣ１２３") == "ABC123"
    # empty / missing input is tolerated
    assert dasr.normalize_text("") == ""
    assert dasr.normalize_text(None) == ""  # type: ignore[arg-type]


def test_strip_punctuation_and_space_keeps_letters_and_digits() -> None:
    assert dasr.strip_punctuation_and_space("他们man 123，。!?") == "他们man123"
    # symbols are not punctuation and are preserved
    assert dasr.strip_punctuation_and_space("¥100") == "¥100"


# ---------------------------------------------------------------------------
# 3. Character alignment / CER / S-D-I
# ---------------------------------------------------------------------------


def _ops_kinds(ops: list[tuple[str, int | None, int | None]]) -> list[str]:
    return [op for op, _gi, _hi in ops]


def test_align_chars_identical() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很忙")
    ops = dasr.align_chars(gold, hyp)
    assert _ops_kinds(ops) == ["match"] * 5
    assert ops == [("match", i, i) for i in range(5)]
    assert dasr.edit_distance(gold, hyp) == 0
    assert dasr.count_sdi(ops) == {"substitutions": 0, "deletions": 0, "insertions": 0}


def test_align_chars_substitution() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很萌")
    ops = dasr.align_chars(gold, hyp)
    assert _ops_kinds(ops)[-1] == "sub"
    assert ops[-1] == ("sub", 4, 4)
    assert dasr.count_sdi(ops) == {"substitutions": 1, "deletions": 0, "insertions": 0}
    assert dasr.edit_distance(gold, hyp) == 1
    assert dasr.char_cer(gold, hyp) == pytest.approx(0.2)


def test_align_chars_deletion() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都忙")
    ops = dasr.align_chars(gold, hyp)
    kinds = _ops_kinds(ops)
    assert kinds.count("del") == 1
    assert ("del", 3, None) in ops
    assert dasr.count_sdi(ops) == {"substitutions": 0, "deletions": 1, "insertions": 0}
    assert dasr.edit_distance(gold, hyp) == 1


def test_align_chars_insertion() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很忙啊")
    ops = dasr.align_chars(gold, hyp)
    assert ("ins", None, 5) in ops
    assert dasr.count_sdi(ops) == {"substitutions": 0, "deletions": 0, "insertions": 1}
    assert dasr.edit_distance(gold, hyp) == 1


def test_align_chars_mixed_script_counts_match_distance() -> None:
    gold = list("abcdef")
    hyp = list("abxdefg")
    ops = dasr.align_chars(gold, hyp)
    sdi = dasr.count_sdi(ops)
    assert sum(sdi.values()) == dasr.edit_distance(gold, hyp) == 2
    assert sdi == {"substitutions": 1, "deletions": 0, "insertions": 1}


def test_align_chars_empty_sequences() -> None:
    assert dasr.align_chars([], []) == []
    assert _ops_kinds(dasr.align_chars(list("ab"), [])) == ["del", "del"]
    assert _ops_kinds(dasr.align_chars([], list("ab"))) == ["ins", "ins"]
    assert dasr.edit_distance([], list("ab")) == 2
    assert dasr.edit_distance(list("ab"), []) == 2
    assert dasr.char_cer([], list("ab")) is None


def test_edit_distance_known_fixtures() -> None:
    assert dasr.edit_distance([], []) == 0
    assert dasr.edit_distance(list("kitten"), list("sitting")) == 3
    assert dasr.edit_distance(list("他们都很忙"), list("他们都很忙")) == 0
    assert dasr.edit_distance(list("他们都很忙"), list("他们都忙")) == 1


def test_count_sdi_from_handmade_script() -> None:
    ops = [
        ("match", 0, 0),
        ("sub", 1, 1),
        ("del", 2, None),
        ("ins", None, 2),
        ("match", 3, 3),
    ]
    assert dasr.count_sdi(ops) == {"substitutions": 1, "deletions": 1, "insertions": 1}


# ---------------------------------------------------------------------------
# 4. ASR normalization rate
# ---------------------------------------------------------------------------


def test_normalization_rate_all_kept() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很忙")
    flags = [False, False, False, False, True]  # 忙 is the OMPAL error char
    result = dasr.normalization_rate(gold, hyp, flags)
    assert result == {"n_ompal_error_chars": 1, "n_asr_same_char": 1, "rate": 1.0}


def test_normalization_rate_changed_character() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很萌")
    flags = [False, False, False, False, True]
    result = dasr.normalization_rate(gold, hyp, flags)
    assert result["n_ompal_error_chars"] == 1
    assert result["n_asr_same_char"] == 0
    assert result["rate"] == 0.0


def test_normalization_rate_deleted_character_counts_as_not_same() -> None:
    gold = list("他们都很忙")
    hyp = list("他们都很")
    flags = [False, False, False, False, True]
    result = dasr.normalization_rate(gold, hyp, flags)
    assert result == {"n_ompal_error_chars": 1, "n_asr_same_char": 0, "rate": 0.0}


def test_normalization_rate_partial() -> None:
    gold = list("abcd")
    hyp = list("axcd")
    flags = [False, True, True, False]  # b and c are OMPAL-error chars
    result = dasr.normalization_rate(gold, hyp, flags)
    assert result["n_ompal_error_chars"] == 2
    assert result["n_asr_same_char"] == 1  # c matched, b substituted
    assert result["rate"] == 0.5


def test_normalization_rate_no_error_chars() -> None:
    result = dasr.normalization_rate(list("ab"), list("ab"), [False, False])
    assert result == {"n_ompal_error_chars": 0, "n_asr_same_char": 0, "rate": None}


def test_normalization_rate_rejects_misaligned_flags() -> None:
    with pytest.raises(ValueError, match="error_flags"):
        dasr.normalization_rate(list("abc"), list("abc"), [False])


# ---------------------------------------------------------------------------
# 5. Percentile helper (per-learner CER distribution)
# ---------------------------------------------------------------------------


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert dasr._percentile(values, 0.0) == 1.0
    assert dasr._percentile(values, 0.5) == pytest.approx(2.5)
    assert dasr._percentile(values, 0.9) == pytest.approx(3.7)
    assert dasr._percentile(values, 1.0) == 4.0
    assert dasr._percentile([7.0], 0.9) == 7.0
    assert dasr._percentile([], 0.9) is None


# ---------------------------------------------------------------------------
# 6. Manifest loading / gold construction
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _labels(chars: str, **error_dims: str) -> list[dict]:
    """Labels for ``chars``; each name in ``error_dims`` is the comma list of error chars."""
    labels = []
    for char in chars:
        label = {"text": char, "tone": 1, "consonant": 1, "vowel": 1}
        for dim, error_chars in error_dims.items():
            if char in error_chars:
                label[dim] = 0
        labels.append(label)
    return labels


def test_load_items_derives_utt_id_and_skips_missing_audio(tmp_path: Path) -> None:
    import soundfile as sf

    wav = tmp_path / "ok.wav"
    sf.write(str(wav), [0.0] * 320, 16000)
    manifest = tmp_path / "items.jsonl"
    _write_manifest(
        manifest,
        [
            {"item_id": "B|00200101", "text": "他们", "audio_path": str(wav), "speaker": "S1"},
            {"item_id": "B|00200102", "text": "他们", "audio_path": str(tmp_path / "no.wav")},
        ],
    )
    items, missing = dasr.load_items(manifest)
    assert [item["utt_id"] for item in items] == ["00200101"]
    assert items[0]["speaker"] == "S1"
    assert missing == ["00200102"]
    limited, limited_missing = dasr.load_items(manifest, limit=1)
    assert [item["utt_id"] for item in limited] == ["00200101"]
    assert limited_missing == ["00200102"]  # missing audio is always reported


def test_utt_id_from_item_id() -> None:
    assert dasr.utt_id_from_item_id("B|00200101") == "00200101"
    assert dasr.utt_id_from_item_id("00200101") == "00200101"


def test_build_gold_normalises_and_flags() -> None:
    item = {
        "item_id": "B|1",
        "text": "他們都很忙",
        "labels": _labels("他們都很忙", tone="忙", consonant="很"),
    }
    gold, error_any, error_by_dim = dasr.build_gold(item)
    assert gold == list("他们都很忙")
    assert error_any == [False, False, False, True, True]
    assert error_by_dim["tone"] == [False, False, False, False, True]
    assert error_by_dim["consonant"] == [False, False, False, True, False]
    assert error_by_dim["vowel"] == [False] * 5


# ---------------------------------------------------------------------------
# 7. Row IO (resume)
# ---------------------------------------------------------------------------


def test_read_rows_and_append_row(tmp_path: Path) -> None:
    path = tmp_path / "asr_outputs.jsonl"
    dasr.append_row(path, {"utt_id": "00200101", "ok": True, "raw_text": "他们"})
    dasr.append_row(path, {"utt_id": "00200102", "ok": False, "raw_text": None})
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    rows, bad = dasr.read_rows(path)
    assert set(rows) == {"00200101", "00200102"}
    assert bad == 1
    dasr.write_rows_atomic(path, [rows["00200102"], rows["00200101"]])
    rows2, bad2 = dasr.read_rows(path)
    assert list(rows2) == ["00200102", "00200101"]
    assert bad2 == 0


# ---------------------------------------------------------------------------
# 8. ASR text-quality metrics on a known fixture
# ---------------------------------------------------------------------------


def _quality_fixture() -> tuple[list[dict], dict[str, dict]]:
    items = [
        {
            "item_id": "B|001",
            "utt_id": "001",
            "speaker": "SPEAKER_A",
            "text": "他們都很忙",
            "labels": _labels("他們都很忙", tone="忙"),
        },
        {
            "item_id": "B|002",
            "utt_id": "002",
            "speaker": "SPEAKER_B",
            "text": "我家有四口人",
            "labels": _labels("我家有四口人"),
        },
        {
            "item_id": "B|003",
            "utt_id": "003",
            "speaker": "SPEAKER_C",
            "text": "大家好",
            "labels": _labels("大家好"),
        },
    ]
    rows = {
        # 1 substitution: 忙 -> 萌 (an OMPAL-error char that the ASR text did not keep)
        "001": {"utt_id": "001", "ok": True, "raw_text": "他们都很萌"},
        # 1 deletion: 四 dropped; all gold chars are OMPAL-correct
        "002": {"utt_id": "002", "ok": True, "raw_text": "我家有口人"},
        # not transcribed -> excluded from the metrics
        "003": {"utt_id": "003", "ok": False, "raw_text": None},
    }
    return items, rows


def test_text_quality_metrics_known_fixture() -> None:
    items, rows = _quality_fixture()
    quality = dasr.text_quality_metrics(items, rows)
    assert quality["n_scored_utterances"] == 2
    assert quality["n_skipped_utterances"] == 1
    assert quality["n_gold_chars"] == 11  # 5 + 6 characters
    assert quality["n_hyp_chars"] == 10
    assert quality["cer_micro"] == pytest.approx(2 / 11)  # 1 substitution + 1 deletion
    assert quality["cer_mean_utterance"] == pytest.approx((0.2 + 1 / 6) / 2)
    assert quality["cer_median_utterance"] == pytest.approx((1 / 6 + 0.2) / 2)
    assert quality["sentence_exact_rate"] == 0.0
    assert quality["sdi"] == {
        "substitutions": 1,
        "deletions": 1,
        "insertions": 0,
        "total_errors": 2,
        "substitution_rate": pytest.approx(1 / 11),
        "deletion_rate": pytest.approx(1 / 11),
        "insertion_rate": 0.0,
    }
    norm = quality["normalization_rate"]
    assert norm["n_ompal_error_chars"] == 1
    assert norm["n_asr_same_char"] == 0
    assert norm["rate"] == 0.0
    assert norm["on_ompal_correct_chars"] == {
        "n_ompal_correct_chars": 10,
        "n_asr_same_char": 9,
        "rate": pytest.approx(0.9),
    }
    assert norm["by_dimension"]["tone"]["n_ompal_error_chars"] == 1
    assert norm["by_dimension"]["tone"]["rate"] == 0.0
    assert norm["by_dimension"]["consonant"]["rate"] is None
    learner = quality["per_learner_cer"]
    assert learner["n_learners"] == 2
    assert learner["min"] == pytest.approx(1 / 6)
    assert learner["median"] == pytest.approx((1 / 6 + 0.2) / 2)
    assert learner["p90"] == pytest.approx((1 / 6) + (0.2 - 1 / 6) * 0.9)
    assert learner["max"] == pytest.approx(0.2)
    assert learner["per_learner"] == {
        "SPEAKER_A": pytest.approx(0.2),
        "SPEAKER_B": pytest.approx(1 / 6),
    }


def test_text_quality_metrics_all_perfect() -> None:
    items = [
        {
            "item_id": "B|001",
            "utt_id": "001",
            "speaker": "SPEAKER_A",
            "text": "他們都很忙",
            "labels": _labels("他們都很忙"),
        }
    ]
    rows = {"001": {"utt_id": "001", "ok": True, "raw_text": "他们都很忙"}}
    quality = dasr.text_quality_metrics(items, rows)
    assert quality["cer_micro"] == 0.0
    assert quality["sentence_exact_rate"] == 1.0
    assert quality["sdi"]["total_errors"] == 0
    assert quality["normalization_rate"]["rate"] is None
    assert quality["normalization_rate"]["on_ompal_correct_chars"]["rate"] == 1.0
