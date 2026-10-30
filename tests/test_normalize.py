"""Tests for pinyin normalisation / target canonicalisation."""

from __future__ import annotations

import pytest

from pinyin_eval.normalize import (
    TONE_MARK_TABLE,
    canonical_target,
    load_mapping,
    normalize_final,
    normalize_initial,
    normalize_output_pair,
    normalize_syllable,
    normalize_tone,
    normalize_toneless_string,
    parse_target_with_tone,
    split_syllable,
    tone_from_marks,
)

MAPPING_CSV = "data/mapping/mandarin_tone_syn_target_mapping.csv"


@pytest.fixture(scope="module")
def mapping():
    return load_mapping(MAPPING_CSV)


# ---------------------------------------------------------------------------
# mapping / canonical targets
# ---------------------------------------------------------------------------


def test_mapping_row_count_and_columns(mapping):
    assert len(mapping) == 378
    entry = mapping["zhang"]
    assert (entry.initial, entry.final) == ("zh", "ang")
    assert mapping["ai"].initial == ""
    # j/q/x + u are stored as ü finals in the table (not hardcoded in code)
    assert mapping["ju"].final == "ü"
    assert mapping["jun"].final == "ün"
    assert mapping["lü"].final == "ü"


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("in", "yin"), ("ing", "ying"), ("iu", "you"), ("u", "wu"),
        ("ui", "wei"), ("un", "wen"), ("ün", "yun"),
        ("zhang", "zhang"), ("üe", "üe"), ("er", "er"),
    ],
)
def test_canonical_target_rules(mapping, raw, canonical):
    assert canonical_target(raw, mapping) == canonical


def test_canonical_target_unknown_raises(mapping):
    with pytest.raises(KeyError):
        canonical_target("not_a_syllable", mapping)


# ---------------------------------------------------------------------------
# initial / final normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("zh", "zh"), (" ZH ", "zh"), ("sh", "sh"), ("", ""), (None, ""),
        ("无", ""), ("none", ""), ("-", ""),
        ("\"zh\"", "zh"), ("(zh)", "zh"), ("1. zh", "zh"),
    ],
)
def test_normalize_initial(value, expected):
    assert normalize_initial(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ang", "ang"), (" ANG ", "ang"), ("iang", "iang"), ("ü", "ü"),
        ("v", "ü"), ("üe", "üe"), ("ian", "ian"), ("iou", "iu"),
        ("uei", "ui"), ("uen", "un"),
    ],
)
def test_normalize_final(value, expected):
    assert normalize_final(value) == expected


def test_normalize_output_pair_plain(mapping):
    assert normalize_output_pair("zh", "ang", mapping) == ("zh", "ang")
    assert normalize_output_pair("", "ai", mapping) == ("", "ai")


def test_normalize_output_pair_jqx_u_to_ü(mapping):
    assert normalize_output_pair("j", "u", mapping) == ("j", "ü")
    assert normalize_output_pair("q", "uan", mapping) == ("q", "üan")
    assert normalize_output_pair("x", "un", mapping) == ("x", "ün")
    # u stays u after non-j/q/x initials
    assert normalize_output_pair("d", "u", mapping) == ("d", "u")


def test_normalize_output_pair_yw_surface(mapping):
    # model may return y/w as the "initial": rewrite via mapping table
    assert normalize_output_pair("y", "in", mapping) == ("", "in")
    assert normalize_output_pair("y", "ing", mapping) == ("", "ing")
    assert normalize_output_pair("w", "u", mapping) == ("", "u")
    assert normalize_output_pair("w", "ei", mapping) == ("", "ui")


def test_normalize_output_pair_split_yw_surface(mapping):
    # model writes the y/w surface split into initial+final
    assert normalize_output_pair("y", "e", mapping) == ("", "ie")
    assert normalize_output_pair("y", "ie", mapping) == ("", "ie")
    assert normalize_output_pair("y", "u", mapping) == ("", "ü")
    assert normalize_output_pair("y", "uan", mapping) == ("", "üan")
    assert normalize_output_pair("w", "o", mapping) == ("", "uo")


def test_normalize_output_pair_apical_with_empty_final(mapping):
    # zhi/chi/shi/ri/zi/ci/si: the apical vowel is often put in the initial
    for initial in ("zh", "ch", "sh", "r", "z", "c", "s"):
        assert normalize_output_pair(initial, "", mapping) == (initial, "i")
        assert normalize_output_pair(initial, None, mapping) == (initial, "i")


def test_normalize_output_pair_consonant_only_keeps_initial(mapping):
    assert normalize_output_pair("q", "", mapping) == ("q", "")


def test_normalize_output_pair_full_syllable_in_one_field(mapping):
    assert normalize_output_pair("zhang", "", mapping) == ("zh", "ang")
    assert normalize_output_pair("", "zhang", mapping) == ("zh", "ang")
    assert normalize_output_pair("", "yin", mapping) == ("", "in")
    assert normalize_output_pair("", "you", mapping) == ("", "iu")
    assert normalize_output_pair("", "ang", mapping) == ("", "ang")


def test_normalize_syllable(mapping):
    assert normalize_syllable("zhang", mapping) == ("zh", "ang")
    assert normalize_syllable("yang", mapping) == ("", "iang")
    assert normalize_syllable("wu", mapping) == ("", "u")
    assert normalize_syllable("nü", mapping) == ("n", "ü")


def test_split_syllable():
    assert split_syllable("zhang") == ("zh", "ang")
    assert split_syllable("an") == ("", "an")
    assert split_syllable("chuan") == ("ch", "uan")
    assert split_syllable("") == ("", "")


# ---------------------------------------------------------------------------
# tone normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1), ("2", 2), (" 3 ", 3), ("4", 4), ("四", 4), ("一声", 1),
        ("tone=2", 2), ("声调：3", 3), ("5", None), ("0", None), (None, None),
        (True, None), ("ā", 1), ("à", 4),
    ],
)
def test_normalize_tone(value, expected):
    assert normalize_tone(value) == expected


def test_tone_from_marks():
    assert tone_from_marks("zhāng") == ("zhang", 1)
    assert tone_from_marks("zhǎng") == ("zhang", 3)
    assert tone_from_marks("lǜ") == ("lü", 4)
    assert tone_from_marks("ü") == ("ü", None)
    assert set(TONE_MARK_TABLE)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("zhang4", ("zhang", 4)),
        ("zhāng", ("zhang", 1)),
        ("zhang", ("zhang", None)),
        ("lǜ", ("lü", 4)),
        ("", ("", None)),
    ],
)
def test_parse_target_with_tone(value, expected):
    assert parse_target_with_tone(value) == expected


def test_normalize_toneless_string():
    assert normalize_toneless_string("zhāng") == "zhang"
    assert normalize_toneless_string("zhang4") == "zhang"
    assert normalize_toneless_string("ZHANG") == "zhang"
    assert normalize_toneless_string("lǜ") == "lü"
    assert normalize_toneless_string(" v ") == "ü"
