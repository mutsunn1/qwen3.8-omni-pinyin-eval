"""Tests for :mod:`pinyin_eval.sentence.scoring` (design §6, §7).

Known-value fixtures only: every expected count below is derived by hand from the
gold/hypothesis pair in the test, so a change in the alignment or CER rules shows
up as an explicit number difference.
"""

from __future__ import annotations

import json

import pytest

from pinyin_eval.sentence.records import GenerationRecord, SentenceItem
from pinyin_eval.sentence.scoring import build_rows, score_a, score_b, score_c

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def make_item_a(
    *,
    item_id: str = "a1",
    chars: str = "你好",
    tokens: tuple[str, ...] = ("ni3", "hao3"),
    speaker: str = "S1",
    flags: dict[str, bool] | None = None,
    duration_s: float = 2.0,
) -> SentenceItem:
    return SentenceItem(
        item_id=item_id,
        task="A",
        source="aishell3",
        speaker=speaker,
        audio_path=f"audio/{item_id}.wav",
        duration_s=duration_s,
        text=chars,
        chars=chars,
        pinyin_surface=list(tokens),
        flags=dict(flags or {}),
    )


def syl(char: str, initial: str, final: str, tone: int) -> dict:
    return {"char": char, "initial": initial, "final": final, "tone": tone}


def parsed_a(*syllables: dict) -> dict:
    return {"parse_status": "ok", "syllables": list(syllables)}


def cat_stats(row: dict, key: str) -> dict:
    """Decode one JSON-string category-counter column of a task-A row."""
    value = row[key]
    assert isinstance(value, str)
    return json.loads(value)


def make_item_b(
    labels: list[dict],
    *,
    item_id: str = "b1",
    learner: str = "L1",
) -> SentenceItem:
    return SentenceItem(
        item_id=item_id,
        task="B",
        source="ompal",
        speaker=learner,
        audio_path=f"audio/{item_id}.wav",
        duration_s=3.0,
        text="".join(label["text"] for label in labels),
        labels=[dict(label) for label in labels],
        learner=learner,
    )


def label(
    text: str,
    *,
    tone: int = 1,
    consonant: int = 1,
    vowel: int = 1,
    zero_initial: bool = False,
) -> dict:
    return {
        "text": text,
        "tone": tone,
        "consonant": consonant,
        "vowel": vowel,
        "zero_initial": zero_initial,
    }


def word(
    text: str,
    *,
    tone_ok: bool = True,
    consonant_ok: bool = True,
    vowel_ok: bool = True,
) -> dict:
    return {
        "text": text,
        "tone_ok": tone_ok,
        "consonant_ok": consonant_ok,
        "vowel_ok": vowel_ok,
    }


def parsed_b(*words: dict) -> dict:
    return {"parse_status": "ok", "words": list(words)}


def make_item_c(
    *,
    item_id: str = "c1",
    gold: dict[str, float] | None = None,
    learner: str = "L1",
    native: bool = False,
) -> SentenceItem:
    return SentenceItem(
        item_id=item_id,
        task="C",
        source="ompal",
        speaker=learner,
        audio_path=f"audio/{item_id}.wav",
        duration_s=4.0,
        text="句子",
        scores=dict(gold or {"accuracy": 4.0, "fluency": 4.5, "prosody": 3.5}),
        learner=learner,
        native=native,
    )


def parsed_c(accuracy: float, fluency: float, prosody: float) -> dict:
    return {
        "parse_status": "ok",
        "scores": {"accuracy": accuracy, "fluency": fluency, "prosody": prosody},
    }


# ---------------------------------------------------------------------------
# Task A — basic matching
# ---------------------------------------------------------------------------


def test_score_a_exact_match() -> None:
    row = score_a(make_item_a(), parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 3)), "C1")

    assert row["task"] == "A"
    assert row["condition"] == "C1"
    assert row["cluster_id"] == "S1"
    assert row["speaker"] == "S1"
    assert row["parse_status"] == "ok"
    assert row["n_gold"] == 2
    assert row["n_hyp"] == 2
    assert row["n_matched"] == 2
    assert row["alignment_coverage"] == pytest.approx(1.0)
    assert row["n_deletions"] == 0
    assert row["n_insertions"] == 0
    assert row["n_joint_correct"] == 2
    assert row["n_initial_correct"] == 2
    assert row["n_final_correct"] == 2
    assert row["n_tone_correct"] == 2
    assert row["cer_tone_num"] == 0
    assert row["cer_tone_den"] == 2
    assert row["cer_stripped_num"] == 0
    assert row["cer_stripped_den"] == 2
    assert row["sentence_exact"] is True
    assert row["n1_sentence_exact"] is True
    assert row["stable"] is True
    assert row["flags"] == {"neutral": False, "yibu": False, "erhua": False}


def test_score_a_tone_sensitive_vs_stripped_cer() -> None:
    # Surfaces are identical; only the tone of 你 differs.
    row = score_a(make_item_a(), parsed_a(syl("你", "n", "i", 2), syl("好", "h", "ao", 3)))

    assert row["cer_tone_num"] == 1
    assert row["cer_stripped_num"] == 0
    assert row["n_tone_correct"] == 1
    assert row["n_joint_correct"] == 1
    assert row["n_initial_correct"] == 2
    assert row["n_final_correct"] == 2
    assert row["sentence_exact"] is False


def test_score_a_deletion_is_wrong_and_counted_in_cer() -> None:
    # Gold 你好, hypothesis only 你: one gold char unmatched.
    row = score_a(make_item_a(), parsed_a(syl("你", "n", "i", 3)))

    assert row["n_hyp"] == 1
    assert row["n_matched"] == 1
    assert row["n_deletions"] == 1
    assert row["n_insertions"] == 0
    assert row["alignment_coverage"] == pytest.approx(0.5)
    assert row["n_joint_correct"] == 1
    assert row["cer_tone_num"] == 1
    assert row["cer_stripped_num"] == 1
    assert row["sentence_exact"] is False


def test_score_a_insertion_not_in_denominator() -> None:
    # Gold 你好, hypothesis 你好吗: the extra 吗 is an insertion (design §6.3).
    row = score_a(
        make_item_a(),
        parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 3), syl("吗", "m", "a", 5)),
    )

    assert row["n_gold"] == 2
    assert row["n_hyp"] == 3
    assert row["n_matched"] == 2
    assert row["n_insertions"] == 1
    assert row["n_deletions"] == 0
    assert row["n_joint_correct"] == 2  # insertions never add correct gold units
    assert row["cer_tone_num"] == 1
    assert row["cer_stripped_num"] == 1
    assert row["sentence_exact"] is False  # insertion breaks sentence exactness


def test_score_a_alignment_survives_middle_deletion() -> None:
    # Gold 你好吗, hypothesis 你吗: the character-level alignment deletes 好.
    row = score_a(
        make_item_a(chars="你好吗", tokens=("ni3", "hao3", "ma5")),
        parsed_a(syl("你", "n", "i", 3), syl("吗", "m", "a", 5)),
    )

    assert row["n_gold"] == 3
    assert row["n_matched"] == 2
    assert row["n_deletions"] == 1
    assert row["n_joint_correct"] == 2
    assert row["cer_tone_num"] == 1


def test_score_a_character_substitution_compared_by_pinyin() -> None:
    # 哪 -> 那 is a homophone substitution: pinyin counts as correct, the
    # character mismatch is only a diagnostic (documented in the module docstring).
    row = score_a(
        make_item_a(chars="哪", tokens=("na3",)),
        parsed_a(syl("那", "n", "a", 3)),
    )

    assert row["n_joint_correct"] == 1
    assert row["n_char_mismatches"] == 1
    assert row["n_deletions"] == 0
    assert row["sentence_exact"] is True


def test_score_a_predicted_jqx_u_is_normalised() -> None:
    # Gold 居 is (j, ü, 1); the parser may hand back final "u" — scoring
    # re-canonicalises initial+final, so this must still match.
    row = score_a(
        make_item_a(chars="居", tokens=("ju1",)),
        parsed_a(syl("居", "j", "u", 1)),
    )
    assert row["n_joint_correct"] == 1
    assert row["sentence_exact"] is True


def test_score_a_predicted_y_surface_is_normalised() -> None:
    # Gold 一 is the zero-initial syllable ("" , "i", 1); a "y"+"i" prediction
    # spells the same canonical syllable.
    row = score_a(
        make_item_a(chars="一", tokens=("yi1",)),
        parsed_a(syl("一", "y", "i", 1)),
    )
    assert row["n_joint_correct"] == 1
    assert row["sentence_exact"] is True


def test_score_a_missing_tone_never_matches() -> None:
    row = score_a(
        make_item_a(chars="你", tokens=("ni3",)),
        parsed_a(syl("你", "n", "i", 0)),
    )
    assert row["n_initial_correct"] == 1
    assert row["n_final_correct"] == 1
    assert row["n_tone_correct"] == 0
    assert row["n_joint_correct"] == 0


# ---------------------------------------------------------------------------
# Task A — erhua
# ---------------------------------------------------------------------------


def test_score_a_erhua_gold_split_matches_split_prediction() -> None:
    # Gold surface "dianr3" over 点儿 is split into dian3 + er5 (design §6.1).
    row = score_a(
        make_item_a(chars="点儿", tokens=("dianr3",)),
        parsed_a(syl("点", "d", "ian", 3), syl("儿", "", "er", 5)),
    )

    assert row["n_gold"] == 2
    assert row["n_joint_correct"] == 2
    assert row["n_tone_correct"] == 2
    assert row["sentence_exact"] is True
    assert row["cer_tone_num"] == 0
    # The 儿 gold syllable is a zero-initial category, kept verbatim as "".
    assert cat_stats(row, "cat_initial_stats") == {
        "": {"n": 1, "correct": 1},
        "d": {"n": 1, "correct": 1},
    }
    assert cat_stats(row, "cat_final_stats") == {
        "er": {"n": 1, "correct": 1},
        "ian": {"n": 1, "correct": 1},
    }
    assert cat_stats(row, "cat_tone_stats") == {
        "3": {"n": 1, "correct": 1},
        "5": {"n": 1, "correct": 1},
    }


def test_score_a_erhua_merged_prediction_is_a_deletion() -> None:
    # The model merged 点儿 into one entry: one gold syllable has no partner.
    row = score_a(
        make_item_a(chars="点儿", tokens=("dianr3",)),
        parsed_a(syl("点儿", "d", "ianr", 3)),
    )

    assert row["n_gold"] == 2
    assert row["n_matched"] == 1
    assert row["n_deletions"] == 1
    assert row["n_joint_correct"] == 0
    assert row["alignment_coverage"] == pytest.approx(0.5)
    assert row["sentence_exact"] is False


# ---------------------------------------------------------------------------
# Task A — N1 sandhi diagnostics
# ---------------------------------------------------------------------------


def test_score_a_n1_yi_before_t4_recovers_textbook_tone() -> None:
    # Gold raw surface yi1 ge4; hypothesis gives the textbook yi2.
    row = score_a(
        make_item_a(chars="一个", tokens=("yi1", "ge4")),
        parsed_a(syl("一", "", "i", 2), syl("个", "g", "e", 4)),
    )

    assert row["n_joint_correct"] == 1  # N0: 个 only
    assert row["n_tone_correct"] == 1
    assert row["cer_tone_num"] == 1
    assert row["n1_n_joint_correct"] == 2  # N1: yi1 -> yi2 before T4
    assert row["n1_n_tone_correct"] == 2
    assert row["n1_cer_tone_num"] == 0
    assert row["n1_sentence_exact"] is True
    assert row["sentence_exact"] is False


def test_score_a_n1_bu_before_t4_recovers_textbook_tone() -> None:
    # Gold raw surface bu4 dui4; hypothesis gives the textbook bu2.
    row = score_a(
        make_item_a(chars="不对", tokens=("bu4", "dui4")),
        parsed_a(syl("不", "b", "u", 2), syl("对", "d", "ui", 4)),
    )

    assert row["n_joint_correct"] == 1
    assert row["n1_n_joint_correct"] == 2
    assert row["n1_n_tone_correct"] == 2
    assert row["n1_cer_tone_num"] == 0


def test_score_a_n1_yi_before_non_t4_becomes_t4() -> None:
    # 一 before a non-T4 syllable is textbook yi4; raw gold is yi1.
    row = score_a(
        make_item_a(chars="一天", tokens=("yi1", "tian1")),
        parsed_a(syl("一", "", "i", 4), syl("天", "t", "ian", 1)),
    )

    assert row["n_joint_correct"] == 1
    assert row["n1_n_joint_correct"] == 2
    assert row["n1_cer_tone_num"] == 0


# ---------------------------------------------------------------------------
# Task A — parse failures, flags, errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["parse_error", "refused", "empty", "api_error"])
def test_score_a_parse_failure_zero_row(status: str) -> None:
    row = score_a(
        make_item_a(),
        {"parse_status": status, "syllables": [syl("你", "n", "i", 3)]},
    )

    assert row["parse_status"] == status
    assert row["n_gold"] == 2  # kept for the failure-rate accounting
    assert row["n_hyp"] == 0
    assert row["n_matched"] == 0
    assert row["n_deletions"] == 0
    assert row["n_insertions"] == 0
    assert row["n_joint_correct"] == 0
    assert row["n1_n_joint_correct"] == 0
    assert row["cer_tone_num"] == 0
    assert row["cer_tone_den"] == 2
    assert row["sentence_exact"] is False
    assert row["n1_sentence_exact"] is False
    assert row["alignment_coverage"] == 0.0
    assert row["cat_initial_stats"] == "{}"
    assert row["cat_final_stats"] == "{}"
    assert row["cat_tone_stats"] == "{}"


def test_score_a_unalignable_gold_raises() -> None:
    item = make_item_a(chars="你好", tokens=("ni3",))  # 2 chars, 1 token
    with pytest.raises(ValueError, match="not alignable"):
        score_a(item, parsed_a())


def test_score_a_flags_and_stable() -> None:
    special = score_a(
        make_item_a(flags={"neutral": True, "yibu": False, "erhua": False}),
        parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 3)),
    )
    assert special["flags"] == {"neutral": True, "yibu": False, "erhua": False}
    assert special["stable"] is False

    plain = score_a(make_item_a(), parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 3)))
    assert plain["stable"] is True


def test_score_a_traditional_characters_match_after_t2s() -> None:
    pytest.importorskip("opencc")
    row = score_a(
        make_item_a(chars="他們", tokens=("ta1", "men5")),
        parsed_a(syl("他", "t", "a", 1), syl("们", "m", "en", 5)),
    )
    assert row["n_char_mismatches"] == 0
    assert row["n_joint_correct"] == 2
    assert row["sentence_exact"] is True


# ---------------------------------------------------------------------------
# Task A — per-category correctness counters (design §7.4)
# ---------------------------------------------------------------------------


def test_score_a_category_stats_known_values() -> None:
    # 你好 + one deliberate error: 好 has the wrong tone (hao4 vs gold hao3).
    row = score_a(
        make_item_a(),
        parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 4)),
    )

    expected_initial = {
        "h": {"n": 1, "correct": 1},
        "n": {"n": 1, "correct": 1},
    }
    expected_final = {
        "ao": {"n": 1, "correct": 1},
        "i": {"n": 1, "correct": 1},
    }
    expected_tone = {"3": {"n": 2, "correct": 1}}

    assert cat_stats(row, "cat_initial_stats") == expected_initial
    assert cat_stats(row, "cat_final_stats") == expected_final
    assert cat_stats(row, "cat_tone_stats") == expected_tone
    # Deterministic, flat JSON-string columns: sorted keys and default separators.
    assert row["cat_initial_stats"] == json.dumps(expected_initial, sort_keys=True)
    assert row["cat_final_stats"] == json.dumps(expected_final, sort_keys=True)
    assert row["cat_tone_stats"] == json.dumps(expected_tone, sort_keys=True)


def test_score_a_category_stats_initial_error_and_tone_key_strings() -> None:
    # 你 ni3 -> 你 li3: the initial category "n" misses, finals/tone are right.
    row = score_a(
        make_item_a(),
        parsed_a(syl("你", "l", "i", 3), syl("好", "h", "ao", 3)),
    )
    assert cat_stats(row, "cat_initial_stats") == {
        "h": {"n": 1, "correct": 1},
        "n": {"n": 1, "correct": 0},
    }
    assert cat_stats(row, "cat_final_stats")["i"] == {"n": 1, "correct": 1}
    assert cat_stats(row, "cat_tone_stats") == {"3": {"n": 2, "correct": 2}}
    # Tone categories are always JSON string keys.
    assert set(cat_stats(row, "cat_tone_stats")) == {"3"}


def test_score_a_category_stats_deletions_count_in_n_only() -> None:
    # Gold 你好, hypothesis only 你: the deleted 好 counts in n but not correct.
    row = score_a(make_item_a(), parsed_a(syl("你", "n", "i", 3)))

    assert cat_stats(row, "cat_initial_stats") == {
        "h": {"n": 1, "correct": 0},
        "n": {"n": 1, "correct": 1},
    }
    assert cat_stats(row, "cat_final_stats") == {
        "ao": {"n": 1, "correct": 0},
        "i": {"n": 1, "correct": 1},
    }
    assert cat_stats(row, "cat_tone_stats") == {"3": {"n": 2, "correct": 1}}
    # Denominator matches n_gold / the n_joint_correct denominator.
    for key in ("cat_initial_stats", "cat_final_stats", "cat_tone_stats"):
        assert sum(entry["n"] for entry in cat_stats(row, key).values()) == row["n_gold"]


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------


def test_score_b_known_confusion_matrix() -> None:
    item = make_item_b([
        label("甲", tone=0),  # wrong, detected            -> tp
        label("乙", tone=1),  # correct, wrongly flagged   -> fp
        label("丙", tone=0),  # wrong, missed              -> fn
        label("丁", tone=1),  # correct                    -> tn
    ])
    parsed = parsed_b(
        word("甲", tone_ok=False),
        word("乙", tone_ok=False),
        word("丙", tone_ok=True),
        word("丁", tone_ok=True),
    )
    rows = score_b(item, parsed, "C1")

    assert [row["dimension"] for row in rows] == ["tone", "consonant", "vowel"]
    tone = rows[0]
    assert tone["task"] == "B"
    assert tone["condition"] == "C1"
    assert tone["cluster_id"] == "L1"
    assert tone["n_units"] == 4
    assert tone["n_matched"] == 4
    assert tone["alignment_coverage"] == pytest.approx(1.0)
    assert (tone["tp"], tone["fp"], tone["fn"], tone["tn"]) == (1, 1, 1, 1)
    assert tone["n_missing"] == 0
    assert tone["n_inserted"] == 0
    # F1(positive=error) = 2*1/(2*1+1+1) = 0.5; F1(correct) = 0.5.
    # The consonant/vowel dimensions are all-correct -> F1 error class = 0.0.
    assert rows[1]["tp"] == 0
    assert rows[2]["fn"] == 0


def test_score_b_missing_gold_unit_counts_as_fn() -> None:
    item = make_item_b([label("甲", tone=0), label("乙", tone=0)])
    # 乙 is missing from the prediction -> no error claimed -> fn.
    rows = score_b(item, parsed_b(word("甲", tone_ok=False)), "C1")

    tone = rows[0]
    assert tone["n_units"] == 2
    assert tone["n_matched"] == 1
    assert tone["n_missing"] == 1
    assert tone["alignment_coverage"] == pytest.approx(0.5)
    assert (tone["tp"], tone["fp"], tone["fn"], tone["tn"]) == (1, 0, 1, 0)


def test_score_b_inserted_unit_not_in_denominator() -> None:
    item = make_item_b([label("甲", tone=1)])
    rows = score_b(
        item,
        parsed_b(word("甲", tone_ok=True), word("乙", tone_ok=False)),
        "C1",
    )
    tone = rows[0]
    assert tone["n_units"] == 1
    assert tone["n_matched"] == 1
    assert tone["n_inserted"] == 1
    assert (tone["tp"], tone["fp"], tone["fn"], tone["tn"]) == (0, 0, 0, 1)


def test_score_b_parse_failure_returns_no_rows() -> None:
    item = make_item_b([label("甲")])
    for status in ("parse_error", "refused", "empty", "api_error"):
        assert score_b(item, {"parse_status": status, "words": []}, "C1") == []


def test_score_b_alignment_by_text_after_t2s() -> None:
    pytest.importorskip("opencc")
    item = make_item_b([label("他", tone=1), label("們", tone=1)])
    rows = score_b(
        item,
        parsed_b(word("他", tone_ok=True), word("们", tone_ok=True)),
        "C1",
    )
    assert rows[0]["n_matched"] == 2
    assert rows[0]["alignment_coverage"] == pytest.approx(1.0)


def test_score_b_zero_initial_consonant_subcounts() -> None:
    item = make_item_b([
        label("安", consonant=0, zero_initial=True),  # wrong, detected  -> ztp
        label("爱", consonant=1, zero_initial=True),  # correct, flagged -> zfp
        label("啊", consonant=0, zero_initial=True),  # wrong, missed    -> zfn
        label("他", consonant=1, zero_initial=False),  # correct, ok     -> tn (not zero)
    ])
    parsed = parsed_b(
        word("安", consonant_ok=False),
        word("爱", consonant_ok=False),
        word("啊", consonant_ok=True),
        word("他", consonant_ok=True),
    )
    consonant = score_b(item, parsed, "C1")[1]

    assert consonant["dimension"] == "consonant"
    assert consonant["tp"] == 1
    assert consonant["fp"] == 1
    assert consonant["fn"] == 1
    assert consonant["tn"] == 1
    assert consonant["n_zero_initial_tp"] == 1
    assert consonant["n_zero_initial_fp"] == 1
    assert consonant["n_zero_initial_fn"] == 1
    assert consonant["n_zero_initial_tn"] == 0

    # Zero-initial sub-counts are only filled for the consonant dimension.
    for row in (score_b(item, parsed, "C1")[0], score_b(item, parsed, "C1")[2]):
        assert row["n_zero_initial_tp"] == 0
        assert row["n_zero_initial_fp"] == 0
        assert row["n_zero_initial_fn"] == 0
        assert row["n_zero_initial_tn"] == 0


def test_score_b_invalid_gold_label_counts_as_fn() -> None:
    item = make_item_b([label("甲", tone=1)])
    item.labels[0]["tone"] = None
    rows = score_b(item, parsed_b(word("甲", tone_ok=True)), "C1")
    assert rows[0]["fn"] == 1
    assert rows[0]["tp"] == 0
    assert rows[0]["tn"] == 0


# ---------------------------------------------------------------------------
# Task C
# ---------------------------------------------------------------------------


def test_score_c_row_values_and_means() -> None:
    item = make_item_c()
    row = score_c(item, parsed_c(3.0, 4.0, 4.0), "C1")

    assert row["task"] == "C"
    assert row["condition"] == "C1"
    assert row["cluster_id"] == "L1"
    assert row["learner"] == "L1"
    assert row["native"] is False
    assert row["gold_accuracy"] == 4.0
    assert row["gold_fluency"] == 4.5
    assert row["gold_prosody"] == 3.5
    assert row["pred_accuracy"] == 3.0
    assert row["pred_fluency"] == 4.0
    assert row["pred_prosody"] == 4.0
    assert row["gold_mean"] == pytest.approx(4.0)
    assert row["pred_mean"] == pytest.approx(11.0 / 3.0)


def test_score_c_parse_failure_keeps_gold() -> None:
    row = score_c(make_item_c(native=True), {"parse_status": "parse_error", "scores": {}}, "C1")

    assert row["parse_status"] == "parse_error"
    assert row["native"] is True
    assert row["gold_accuracy"] == 4.0
    assert row["pred_accuracy"] is None
    assert row["pred_fluency"] is None
    assert row["pred_prosody"] is None
    assert row["pred_mean"] is None


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------


def test_build_rows_wires_generations_to_scorers() -> None:
    item_a1 = make_item_a(item_id="a1")
    item_a2 = make_item_a(item_id="a2")  # no generation at all -> empty
    item_b = make_item_b([label("你")], item_id="b1")
    item_c = make_item_c(item_id="c1")

    generation_a = GenerationRecord(
        run_id="r1",
        request_id="A|C1|a1",
        item_id="a1",
        task="A",
        condition="C1",
        thinking=False,
        max_tokens=256,
        model="mock",
        prompt_version="sentence-v1",
        prompt_sha256="x",
        audio_path="a1.wav",
        response_text="fake",
    )
    generations = [
        generation_a,
        {"item_id": "b1", "task": "B", "condition": "C1", "response_text": "fake"},
        {"item_id": "c1", "task": "C", "condition": "C1", "error": "boom"},
        # Wrong condition: must be ignored.
        {"item_id": "a1", "task": "A", "condition": "C2", "response_text": "fake"},
    ]
    parsers = {
        "A": lambda _text: parsed_a(syl("你", "n", "i", 3), syl("好", "h", "ao", 3)),
        "B": lambda _text: parsed_b(word("你")),
        "C": lambda _text: parsed_c(4.0, 4.0, 4.0),
    }

    rows = build_rows([item_a1, item_a2, item_b, item_c], generations, "C1", parsers)

    assert set(rows) == {"A", "B", "C"}
    assert len(rows["A"]) == 2
    assert rows["A"][0]["parse_status"] == "ok"
    assert rows["A"][0]["condition"] == "C1"
    assert rows["A"][1]["parse_status"] == "empty"
    assert len(rows["B"]) == 3  # three dimension rows
    assert all(row["parse_status"] == "ok" for row in rows["B"])
    assert rows["C"][0]["parse_status"] == "api_error"
    assert rows["C"][0]["pred_accuracy"] is None


def test_build_rows_unknown_task_raises() -> None:
    item = make_item_a()
    item.task = "Z"
    with pytest.raises(ValueError, match="unknown sentence task"):
        build_rows([item], [], "C1", {"A": lambda _text: parsed_a()})
