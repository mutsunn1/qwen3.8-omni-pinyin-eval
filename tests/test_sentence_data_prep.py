"""Offline unit tests for ``scripts/prepare_sentence_data.py``.

Everything here runs against synthetic fixtures / ``tmp_path``: no network and
no real corpus files are touched.  Run with::

    .venv/bin/python -m pytest tests/test_sentence_data_prep.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_sentence_data as prep  # noqa: E402

from pinyin_eval.sentence.normalize import normalize_pinyin_token  # noqa: E402
from pinyin_eval.sentence.records import SentenceItem, load_jsonl  # noqa: E402

#: A perfectly alignable five-character sentence used by many fixtures.
SENTENCE_CHARS = "我们都很忙"
SENTENCE_TOKENS = ("wo3", "men5", "dou1", "hen3", "mang2")

#: zero-initial spellings covering all 36 canonical finals (plus a tone digit later)
FINAL_TOKENS: dict[str, str] = {
    "a": "a", "ai": "ai", "an": "an", "ang": "ang", "ao": "ao",
    "e": "e", "ei": "ei", "en": "en", "eng": "eng", "er": "er",
    "i": "yi", "ia": "ya", "ian": "yan", "iang": "yang", "iao": "yao",
    "ie": "ye", "in": "yin", "ing": "ying", "iong": "yong", "iu": "you",
    "o": "o", "ong": "ong", "ou": "ou",
    "u": "wu", "ua": "wa", "uai": "wai", "uan": "wan", "uang": "wang",
    "ueng": "weng", "ui": "wei", "un": "wen", "uo": "wo",
    "ü": "yu", "üan": "yuan", "üe": "yue", "ün": "yun",
}

#: spellings covering all 21 canonical initials (plus a tone digit later)
INITIAL_TOKENS: dict[str, str] = {
    "zh": "zhi", "ch": "chi", "sh": "shi", "b": "ba", "p": "pa", "m": "ma",
    "f": "fa", "d": "da", "t": "ta", "n": "na", "l": "la", "g": "ga",
    "k": "ka", "h": "ha", "j": "ji", "q": "qi", "x": "xi", "r": "ri",
    "z": "zi", "c": "ci", "s": "si",
}


def _record(utt_id: str, chars: str, tokens: tuple[str, ...]) -> prep.AishellRecord:
    return prep.make_aishell_record(utt_id, chars, list(tokens))


def _coverage_records(prefix: str) -> list[prep.AishellRecord]:
    """57 one-character records covering every initial, final and tone 1-4 once."""
    records: list[prep.AishellRecord] = []
    finals = sorted(prep.COVERAGE_FINALS)
    for index, final_code in enumerate(finals):
        token = f"{FINAL_TOKENS[final_code]}{(index % 4) + 1}"
        syllable = normalize_pinyin_token(token)
        assert syllable.initial == "" and syllable.final == final_code, token
        records.append(_record(f"{prefix}{index:04d}", "字", (token,)))
    for index, initial in enumerate(prep.COVERAGE_INITIALS):
        token = f"{INITIAL_TOKENS[initial]}{((index + len(finals)) % 4) + 1}"
        syllable = normalize_pinyin_token(token)
        assert syllable.initial == initial, token
        records.append(_record(f"{prefix}{len(finals) + index:04d}", "字", (token,)))
    return records


def _without_final(records: list[prep.AishellRecord], final_code: str) -> list[prep.AishellRecord]:
    return [record for record in records if not any(syllable.final == final_code for syllable in record.gold)]


def _speaker_info(speaker: str, *, gender: str = "female", accent: str = "north", age: str = "B") -> prep.SpkInfo:
    return prep.SpkInfo(speaker=speaker, age=age, gender=gender, accent=accent)


#: category -> count table that satisfies the real QUOTAS exactly (30 speakers)
CATEGORY_COUNTS: dict[tuple[str, str, str], int] = {
    ("female", "north", "B"): 17,
    ("female", "north", "C"): 1,
    ("male", "north", "C"): 3,
    ("male", "north", "D"): 1,
    ("male", "south", "B"): 7,
    ("male", "south", "C"): 1,
}


def _synthetic_aishell(
    root: Path,
    *,
    utts_per_speaker: int = 10,
) -> dict[str, str]:
    """Write synthetic ``label_train-set.txt`` / ``spk-info.txt``.

    Returns ``{speaker: stratum}`` for the 30 speakers that exactly satisfy
    ``prep.QUOTAS``.
    """
    aishell_dir = root / prep.AISHELL_DIR
    aishell_dir.mkdir(parents=True, exist_ok=True)
    label_lines = ["# synthetic fixture"]
    spk_lines = ["# synthetic fixture"]
    speakers: dict[str, str] = {}
    speaker_index = 0
    for category in sorted(CATEGORY_COUNTS):
        gender, accent, age = category
        for _ in range(CATEGORY_COUNTS[category]):
            speaker_index += 1
            speaker = f"SSB{speaker_index:04d}"
            speakers[speaker] = f"{gender}/{accent}/{age}"
            spk_lines.append(f"{speaker}\t{age}\t{gender}\t{accent}")
            for utt_index in range(utts_per_speaker):
                utt_id = f"{speaker}{utt_index:04d}"
                label_lines.append(f"{utt_id}|{' '.join(SENTENCE_TOKENS)}|{SENTENCE_CHARS}")
    (aishell_dir / "label_train-set.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    (aishell_dir / "spk-info.txt").write_text("\n".join(spk_lines) + "\n", encoding="utf-8")
    return speakers


def _ompal_record(text: str, words_chars: str, *, scores: tuple[float, float, float] = (4.0, 4.0, 4.0)) -> dict:
    return {
        "accuracy": scores[0],
        "fluency": scores[1],
        "prosody": scores[2],
        "text": text,
        "words": [
            {"text": [char], "tone": 1, "phoneme_consonant": 1, "phoneme_vowel": 1}
            for char in words_chars
        ],
    }


def _synthetic_ompal(root: Path) -> dict[str, object]:
    """Write synthetic OMPAL score files (one corrupted learner row included)."""
    repo_dir = root / prep.OMPAL_REPO_DIR
    repo_dir.mkdir(parents=True, exist_ok=True)
    learner = {
        "00200101": _ompal_record("他們都很忙", "他們都很忙", scores=(4.0, 4.5, 4.0)),
        "00200102": _ompal_record("我家有四口人", "我家有四口人", scores=(3.5, 4.0, 3.5)),
        # corrupted: text has four characters, word slots only cover three
        "00200103": _ompal_record("爸爸媽媽哥哥和我", "爸爸媽媽哥哥"),
    }
    native = {
        "00100101": _ompal_record("他們都很忙", "他們都很忙", scores=(5.0, 5.0, 5.0)),
    }
    detail = {
        utt_id: {
            "accuracy": [int(record["accuracy"])] * 3,
            "fluency": [int(record["fluency"])] * 3,
            "prosody": [int(record["prosody"])] * 3,
            "text": record["text"],
            "words": [
                {
                    "text": list(word["text"]),
                    "tone": [word["tone"]] * 3,
                    "phoneme_consonant": [word["phoneme_consonant"]] * 3,
                    "phoneme_vowel": [word["phoneme_vowel"]] * 3,
                }
                for word in record["words"]
            ],
        }
        for utt_id, record in learner.items()
    }
    (repo_dir / "non-native_scores.json").write_text(json.dumps(learner, ensure_ascii=False), encoding="utf-8")
    (repo_dir / "native_scores.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    (repo_dir / "non-native_scores-detail.json").write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
    (repo_dir / "README.md").write_text("# synthetic OMPAL fixture\n", encoding="utf-8")
    return {"learner": learner, "native": native, "detail": detail}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_aishell_labels_strips_prosody_marks_and_skips_malformed() -> None:
    text = "\n".join(
        [
            "# comment line",
            "",
            "SSB00010001|wo3 men5 % hen3 mang2 $|我们%很忙$",
            "this line has no pipes",
            "SSB00010002||   ",
        ]
    )
    lines, malformed = prep.parse_aishell_labels(text)
    assert [line.utt_id for line in lines] == ["SSB00010001"]
    assert lines[0].tokens == ("wo3", "men5", "hen3", "mang2")
    assert lines[0].chars == "我们很忙"
    assert len(malformed) == 2


def test_parse_aishell_labels_rejects_duplicate_ids() -> None:
    text = "SSB00010001|wo3 men5|我们\nSSB00010001|wo3 men5|我们\n"
    lines, malformed = prep.parse_aishell_labels(text)
    assert len(lines) == 1
    assert malformed and "duplicate" in malformed[0]["error"]


def test_parse_spk_info_strips_trailing_whitespace() -> None:
    text = "\n".join(
        [
            "# header",
            "SSB0001\tB\tfemale\tnorth ",
            "SSB0002\tC\tmale\tsouth",
            "not-enough-fields",
        ]
    )
    infos, malformed = prep.parse_spk_info(text)
    assert infos["SSB0001"] == prep.SpkInfo("SSB0001", "B", "female", "north")
    assert infos["SSB0002"].accent == "south"
    assert len(malformed) == 1


def test_build_aishell_records_filters_length_and_unalignable() -> None:
    lines = [
        prep.AishellLine("SSB00010001", SENTENCE_TOKENS, SENTENCE_CHARS),
        prep.AishellLine("SSB00010002", ("wo3", "men5"), "我们"),  # too short
        prep.AishellLine("SSB00010003", ("wo3",) * 31, "我" * 31),  # too long
        prep.AishellLine("SSB00010004", ("ni3",), "你好"),  # unalignable
    ]
    filtered = prep.build_aishell_records(lines)
    assert [record.utt_id for record in filtered["eligible"]] == ["SSB00010001"]
    assert filtered["length_filtered"] == 2
    assert [entry["utt_id"] for entry in filtered["unalignable"]] == ["SSB00010004"]
    assert filtered["unalignable"][0]["reason"]


def test_strip_aishell_marks_nfkc() -> None:
    assert prep.strip_aishell_marks("持起%红缨枪$") == "持起红缨枪"
    # NFKC folds full-width characters
    assert prep.strip_aishell_marks("ＡＢ％") == "AB"


# ---------------------------------------------------------------------------
# Stratified speaker selection
# ---------------------------------------------------------------------------


def _assert_quotas(achieved: dict[str, dict[str, int]]) -> None:
    for dimension, targets in prep.QUOTAS.items():
        assert achieved[dimension] == targets


def test_solve_strata_counts_exact() -> None:
    solution, slack = prep.solve_strata_counts(CATEGORY_COUNTS)
    assert slack == 0
    assert solution is not None
    assert sum(solution.values()) == prep.AISHELL_TARGET_SPEAKERS
    used: dict[str, Counter[str]] = {"gender": Counter(), "accent": Counter(), "age": Counter()}
    for (gender, accent, age), count in solution.items():
        used["gender"][gender] += count
        used["accent"][accent] += count
        used["age"][age] += count
    _assert_quotas({dimension: dict(counter) for dimension, counter in used.items()})


def test_solve_strata_counts_custom_quotas() -> None:
    quotas = {
        "gender": {"female": 1, "male": 1},
        "accent": {"north": 1, "south": 1},
        "age": {"B": 1, "D": 1},
    }
    sizes = {("female", "north", "B"): 5, ("male", "south", "D"): 5}
    solution, slack = prep.solve_strata_counts(sizes, quotas)
    assert slack == 0 and solution == {("female", "north", "B"): 1, ("male", "south", "D"): 1}


def test_solve_strata_counts_infeasible_returns_none() -> None:
    assert prep.solve_strata_counts({("female", "north", "B"): 4}) is None


def test_select_speakers_hits_quotas_and_is_deterministic() -> None:
    pool: dict[str, list[prep.AishellRecord]] = {}
    spk_info: dict[str, prep.SpkInfo] = {}
    speaker_index = 0
    for category in sorted(CATEGORY_COUNTS):
        gender, accent, age = category
        for _ in range(CATEGORY_COUNTS[category]):
            speaker_index += 1
            speaker = f"SSB{speaker_index:04d}"
            spk_info[speaker] = _speaker_info(speaker, gender=gender, accent=accent, age=age)
            pool[speaker] = [
                _record(f"{speaker}{utt_index:04d}", SENTENCE_CHARS, SENTENCE_TOKENS)
                for utt_index in range(12)
            ]
    # an out-of-quota speaker must be reported, not selected
    spk_info["SSB0999"] = _speaker_info("SSB0999", accent="others")
    pool["SSB0999"] = [_record(f"SSB0999{index:04d}", SENTENCE_CHARS, SENTENCE_TOKENS) for index in range(12)]

    selection = prep.select_speakers(pool, spk_info, seed=prep.SEED)
    assert len(selection["speakers"]) == prep.AISHELL_TARGET_SPEAKERS
    _assert_quotas(selection["achieved"])
    assert selection["slack"] == 0
    assert "SSB0999" in {entry["speaker"] for entry in selection["exclusions"]}
    again = prep.select_speakers(pool, spk_info, seed=prep.SEED)
    assert again["speakers"] == selection["speakers"]


def test_candidate_speakers_reports_short_pools() -> None:
    pool = {"SSB0001": [_record(f"SSB0001000{index}", SENTENCE_CHARS, SENTENCE_TOKENS) for index in range(3)]}
    spk_info = {"SSB0001": _speaker_info("SSB0001")}
    candidates, exclusions = prep.candidate_speakers(pool, spk_info)
    assert candidates == []
    assert exclusions and "eligible utterances" in exclusions[0]["reason"]


def test_sample_speaker_utterances_is_deterministic() -> None:
    pool = {
        speaker: [
            _record(f"{speaker}{index:04d}", SENTENCE_CHARS, SENTENCE_TOKENS) for index in range(12)
        ]
        for speaker in ("SSB0001", "SSB0002")
    }
    first = prep.sample_speaker_utterances(pool, ["SSB0001", "SSB0002"], seed=prep.SEED)
    second = prep.sample_speaker_utterances(pool, ["SSB0001", "SSB0002"], seed=prep.SEED)
    assert [record.utt_id for record in first] == [record.utt_id for record in second]
    assert len(first) == 20
    assert Counter(record.speaker for record in first) == Counter({"SSB0001": 10, "SSB0002": 10})


def test_speaker_from_utt_and_ompal_paths() -> None:
    assert prep.speaker_from_utt("00200101") == "SPEAKER02001"
    assert prep.speaker_from_utt("00100101") == "SPEAKER01001"
    with pytest.raises(ValueError):
        prep.speaker_from_utt("42")


# ---------------------------------------------------------------------------
# Coverage / special subset
# ---------------------------------------------------------------------------


def test_coverage_fixture_is_complete() -> None:
    coverage = prep.coverage_stats(_coverage_records("ZZZ0001"))
    assert coverage.complete, coverage.as_dict()
    assert len(coverage.initials) == len(prep.COVERAGE_INITIALS)
    assert len(coverage.finals) == len(prep.COVERAGE_FINALS)
    assert set(coverage.tones) == set(prep.COVERAGE_TONES)


def test_coverage_stats_detects_missing_categories() -> None:
    records = [
        _record("SSB00010001", "翁", ("weng1",)),
        _record("SSB00010002", "大", ("da4",)),
    ]
    coverage = prep.coverage_stats(records)
    assert "ueng" not in coverage.missing_finals
    assert coverage.missing_tones == [2, 3]
    assert "b" in coverage.missing_initials
    assert coverage.complete is False


def test_apply_coverage_supplements_same_speaker_swap() -> None:
    base = _coverage_records("ZZZ0001")
    duplicate = _coverage_records("ZZZ0002")
    records = _without_final(base, "ueng") + _without_final(duplicate, "ueng")
    pool = {"ZZZ0001": base, "ZZZ0002": duplicate}
    assert "ueng" in prep.coverage_stats(records).missing_finals

    supplemented, supplements = prep.apply_coverage_supplements(records, pool, {}, per_speaker=10)
    coverage = prep.coverage_stats(supplemented)
    assert coverage.complete
    assert len(supplemented) == len(records)
    assert len(supplements) == 1
    assert supplements[0]["kind"] == "same_speaker_swap"
    assert supplements[0]["value"] == "ueng"


def test_apply_coverage_supplements_speaker_swap() -> None:
    base = _without_final(_coverage_records("ZZZ0001"), "ueng")
    victim = _without_final(_coverage_records("SSB0010"), "ueng")
    # candidate speaker inside the same stratum, never selected
    candidate_ueng = _record("SSB00110001", "翁", ("weng1",))
    candidate_other = _record("SSB00110002", "大", ("da4",))
    records = base + victim
    spk_info = {
        "ZZZ0001": _speaker_info("ZZZ0001"),
        "SSB0010": _speaker_info("SSB0010"),
        "SSB0011": _speaker_info("SSB0011"),
    }
    pool = {
        "ZZZ0001": base,
        "SSB0010": victim,
        "SSB0011": [candidate_ueng, candidate_other],
    }
    supplemented, supplements = prep.apply_coverage_supplements(
        records, pool, spk_info, per_speaker=2
    )
    assert prep.coverage_stats(supplemented).complete
    assert len(supplements) == 1
    entry = supplements[0]
    assert entry["kind"] == "speaker_swap"
    assert entry["removed_speaker"] == "SSB0010"
    assert entry["added_speaker"] == "SSB0011"
    assert entry["forced_utt"] == "SSB00110001"
    assert {record.speaker for record in supplemented} == {"ZZZ0001", "SSB0011"}


def test_ensure_special_subset_swaps_in_special_item() -> None:
    base = _coverage_records("ZZZ0001")  # complete, no neutral / 一不 / erhua
    non_special = _record("SSB00200001", "字", ("ba1",))
    special = _record("SSB00200002", "的", ("de5",))  # neutral tone 5
    records = base + [non_special]
    pool = {"ZZZ0001": base, "SSB0020": [non_special, special]}
    assert not any(record.flags.special for record in base)

    supplemented, supplements = prep.ensure_special_subset(records, pool, min_special=1)
    assert sum(1 for record in supplemented if record.flags.special) == 1
    assert supplements and supplements[0]["reason"] == "special_subset"
    assert supplements[0]["added_utt"] == "SSB00200002"
    assert prep.coverage_stats(supplemented).complete


def test_make_aishell_record_rejects_unalignable() -> None:
    with pytest.raises(ValueError):
        prep.make_aishell_record("SSB00010001", "你好", ["ni3"])


# ---------------------------------------------------------------------------
# OMPAL cleaning / labels
# ---------------------------------------------------------------------------


def _ompal_word(char: str, *, tone: int = 1, consonant: int = 1, vowel: int = 1) -> dict:
    return {"text": [char], "tone": tone, "phoneme_consonant": consonant, "phoneme_vowel": vowel}


def test_detect_ompal_anomalies_and_classify() -> None:
    records = {
        "00200101": {"text": "他們都很忙", "words": [_ompal_word(char) for char in "他們都很忙"]},
        "00200102": {"text": "孩子們", "words": [_ompal_word("孩"), {"text": ["還", "也"], "tone": 1, "phoneme_consonant": 1, "phoneme_vowel": 1}, _ompal_word("們")]},
        "00200103": {"text": "所以這一天", "words": [_ompal_word(char) for char in "所以"]},
        "00200104": {"text": "你好", "words": [{"text": [], "tone": 1, "phoneme_consonant": 1, "phoneme_vowel": 1}, _ompal_word("好")]},
    }
    anomalies = prep.detect_ompal_anomalies(records)
    assert anomalies["multi_char_word_slot_rows"] == ["00200102"]
    assert [entry["utt_id"] for entry in anomalies["text_join_mismatch_rows"]] == [
        "00200102",
        "00200103",
        "00200104",
    ]
    assert anomalies["empty_word_slot_rows"] == ["00200104"]

    cleaned = prep.classify_learner_records(records)
    assert cleaned["kept"] == ["00200101"]
    reasons = {entry["utt_id"]: entry["reasons"] for entry in cleaned["exclusions"]}
    assert reasons["00200102"] == ["corrupted_word_slot", "text_slot_mismatch"]
    assert reasons["00200103"] == ["text_slot_mismatch"]
    assert reasons["00200104"] == ["corrupted_word_slot", "text_slot_mismatch"]


def test_words_to_labels_maps_every_dimension() -> None:
    words = [
        {"text": ["他"], "tone": 1, "phoneme_consonant": 1, "phoneme_vowel": 0},
        {"text": ["們"], "tone": 0, "phoneme_consonant": 1, "phoneme_vowel": 1},
    ]
    labels = prep.words_to_labels(words, [True, False])
    assert labels[0] == {"text": "他", "tone": 1, "consonant": 1, "vowel": 0, "zero_initial": True}
    assert labels[1]["tone"] == 0 and labels[1]["zero_initial"] is False
    with pytest.raises(ValueError):
        prep.words_to_labels([{"text": ["還", "也"], "tone": 1, "phoneme_consonant": 1, "phoneme_vowel": 1}], [False])


def test_compute_zero_initials_uses_context() -> None:
    pytest.importorskip("pypinyin")
    flags = prep.compute_zero_initials("我爱中国")
    assert flags == [True, True, False, False]


def test_check_aggregated_scores_validates_means_and_majority() -> None:
    aggregated = {
        "00200101": {
            "accuracy": 4.33,
            "fluency": 4.67,
            "prosody": 4.0,
            "text": "他們都很忙",
            "words": [_ompal_word(char, tone=1) for char in "他們都很忙"],
        }
    }
    detail = {
        "00200101": {
            "accuracy": [4, 5, 4],
            "fluency": [5, 4, 5],
            "prosody": [4, 4, 4],
            "text": "他們都很忙",
            "words": [
                {
                    "text": [char],
                    "tone": [1, 1, 1],
                    "phoneme_consonant": [1, 1, 1],
                    "phoneme_vowel": [1, 1, 1],
                }
                for char in "他們都很忙"
            ],
        }
    }
    validation = prep.check_aggregated_scores(aggregated, detail)
    assert validation["aggregated_is_mean_of_raters"] is True
    assert validation["aggregated_is_majority_vote"] is True
    assert validation["n_direct_id_score_mean_mismatches"] == 0

    broken = json.loads(json.dumps(detail))
    broken["00200101"]["accuracy"] = [4, 4, 4]  # mean 4.0 != 4.33
    validation = prep.check_aggregated_scores(aggregated, broken)
    assert validation["aggregated_is_mean_of_raters"] is False


def test_build_sentence_inventory_is_stable() -> None:
    learner = {"1": {"text": "乙"}, "2": {"text": "甲"}}
    native = {"3": {"text": "乙"}, "4": {"text": "丙"}}
    inventory = prep.build_sentence_inventory(learner, native)
    assert inventory == {"丙": "S00", "乙": "S01", "甲": "S02"}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def test_probe_audio_reads_duration_and_reports_missing(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / "tone.wav"
    soundfile.write(str(path), [0.0] * 1600, 16000, subtype="PCM_16")
    probe = prep.probe_audio(path)
    assert probe["ok"] is True
    assert probe["samplerate"] == 16000
    assert probe["channels"] == 1
    assert probe["subtype"] == "PCM_16"
    assert probe["duration_s"] == pytest.approx(0.1)
    missing = prep.probe_audio(tmp_path / "missing.wav")
    assert missing["ok"] is False and "missing" in missing["error"]


def test_audio_section_reports_failures_and_format_violations() -> None:
    reports = [
        {"audio_path": "a.wav", "probe": {"ok": True, "samplerate": 44100, "channels": 1, "subtype": "PCM_16", "frames": 44100, "duration_s": 1.0}},
        {"audio_path": "b.wav", "probe": {"ok": True, "samplerate": 16000, "channels": 1, "subtype": "PCM_16", "frames": 16000, "duration_s": 1.0}},
        {"audio_path": "c.wav", "probe": {"ok": False, "error": "missing file"}},
    ]
    section = prep._audio_section(reports, prep.AISHELL_EXPECTED_FORMAT, label="synthetic")
    assert section["checked"] == 3
    assert section["ok"] == 2
    assert len(section["failures"]) == 1
    assert len(section["format_violations"]) == 1
    assert section["total_duration_s"] == pytest.approx(2.0)


def test_duration_stats_and_sha256(tmp_path: Path) -> None:
    stats = prep.duration_stats([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert stats["total_s"] == pytest.approx(6.0)
    assert stats["median_s"] == pytest.approx(2.0)
    assert prep.duration_stats([])["n"] == 0

    path = tmp_path / "fixture.txt"
    path.write_text("abc", encoding="utf-8")
    assert prep.sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_majority_label() -> None:
    assert prep.majority_label([0, 0, 1]) == 0
    assert prep.majority_label([0, 1, 1]) == 1
    assert prep.majority_label([1, 0]) == 0


# ---------------------------------------------------------------------------
# End-to-end offline build (synthetic corpus + tmp_path)
# ---------------------------------------------------------------------------

EXPECTED_A_COUNT = prep.AISHELL_TARGET_SPEAKERS * 10


def _prepare_synthetic_root(tmp_path: Path) -> Path:
    _synthetic_aishell(tmp_path)
    _synthetic_ompal(tmp_path)
    return tmp_path


def test_build_end_to_end_offline(tmp_path: Path) -> None:
    root = _prepare_synthetic_root(tmp_path)
    result = prep.build(root, seed=prep.SEED)
    counts = result["summary"]["counts"]
    assert counts["A"] == EXPECTED_A_COUNT
    assert counts["B"] == 2  # 3 learner rows, 1 corrupted -> excluded
    assert counts["C"] == 2
    assert counts["C_native"] == 1

    summary = result["summary"]
    assert summary["seed"] == prep.SEED
    assert summary["aishell3"]["stratification"]["achieved"] == {
        "gender": {"female": 18, "male": 12},
        "accent": {"north": 22, "south": 8},
        "age": {"B": 24, "C": 5, "D": 1},
    }
    assert summary["ompal"]["cleaning"]["excluded_learner"] == 1
    assert summary["ompal"]["cleaning"]["exclusions"][0]["utt_id"] == "00200103"
    assert summary["ompal"]["validation"]["native_all_scores_5"] is True
    assert summary["ompal"]["labels"]["zero_initial_failures"] == 0
    # AISHELL wavs do not exist in the synthetic root -> durations 0 and failures reported
    assert summary["aishell3"]["items"]["n"] == EXPECTED_A_COUNT
    assert len(summary["aishell3"]["audio"]["failures"]) == EXPECTED_A_COUNT
    # 2 learner + 1 native OMPAL items have no synthetic wav files either
    assert len(summary["ompal"]["audio"]["failures"]) == 3

    paths = prep.write_outputs(root, result)
    manifest_dir = root / prep.MANIFEST_DIR
    for name in ("aishell3_items.jsonl", "ompal_b_items.jsonl", "ompal_c_items.jsonl", "ompal_native_items.jsonl", "prep_summary.json", "README.md"):
        assert (manifest_dir / name).exists(), name
        assert paths[name]

    a_rows = load_jsonl(manifest_dir / "aishell3_items.jsonl")
    assert len(a_rows) == EXPECTED_A_COUNT
    first = SentenceItem.from_dict(a_rows[0])
    assert first.task == "A" and first.item_id.startswith("A|")
    assert first.source == "aishell3"
    assert first.duration_s == 0.0
    assert first.chars == SENTENCE_CHARS
    assert first.pinyin_surface == list(SENTENCE_TOKENS)
    assert set(first.flags) == {"neutral", "yibu", "erhua"}
    assert first.flags["neutral"] is True  # men5
    assert set(first.meta) >= {"utt_id", "gender", "accent", "age", "in_special", "coverage_supplement", "stratum"}
    assert first.meta["in_special"] is True

    b_rows = load_jsonl(manifest_dir / "ompal_b_items.jsonl")
    assert len(b_rows) == 2
    b_item = SentenceItem.from_dict(b_rows[0])
    assert b_item.task == "B" and b_item.item_id.startswith("B|")
    assert b_item.speaker == prep.speaker_from_utt(b_item.meta["utt_id"])
    assert b_item.learner == b_item.speaker
    assert len(b_item.labels) == len(b_item.chars)
    for label in b_item.labels:
        assert set(label) == {"text", "tone", "consonant", "vowel", "zero_initial"}
        assert label["zero_initial"] in (True, False)
    assert b_item.flags == {"native": False, "learner": True}

    c_rows = load_jsonl(manifest_dir / "ompal_c_items.jsonl")
    c_item = SentenceItem.from_dict(c_rows[0])
    assert c_item.task == "C"
    assert set(c_item.scores) == {"accuracy", "fluency", "prosody"}

    native_rows = load_jsonl(manifest_dir / "ompal_native_items.jsonl")
    native_item = SentenceItem.from_dict(native_rows[0])
    assert native_item.task == "C" and native_item.native is True
    assert native_item.scores == {"accuracy": 5.0, "fluency": 5.0, "prosody": 5.0}


def test_build_is_deterministic(tmp_path: Path) -> None:
    root = _prepare_synthetic_root(tmp_path)
    first = prep.build(root, seed=prep.SEED)
    second = prep.build(root, seed=prep.SEED)
    for key in ("aishell3", "ompal_b", "ompal_c", "ompal_native"):
        assert [item.item_id for item in first["items"][key]] == [
            item.item_id for item in second["items"][key]
        ]
    assert first["summary"]["aishell3"]["stratification"]["achieved"] == second["summary"]["aishell3"]["stratification"]["achieved"]


def test_build_requires_ompal_metadata(tmp_path: Path) -> None:
    _synthetic_aishell(tmp_path)
    with pytest.raises(FileNotFoundError):
        prep.build(tmp_path)
