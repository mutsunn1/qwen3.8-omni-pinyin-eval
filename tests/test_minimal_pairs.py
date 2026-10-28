"""Tests for ``pinyin_eval.data.minimal_pairs``.

Hard acceptance criterion: ``count_table`` must reproduce the design document's
counts exactly (docs/experiment-design.md「最小对计数」):

    sh_zh 36/576, aspiration 67/1072, nasal_lateral 21/336,
    front_back_nasal_ptr 40/640, i_ü 5/80, ian_üan 3/48, tone 2268/9072.

The canonical target table is read from the real mapping CSV (``utf-8-sig``);
audio records are synthetic, so no WAV files are needed.  The fixture mirrors
the dataset shape: 378 canonical targets x voices A-D x tones 1-4, plus a
second raw spelling for the seven merged targets (385 raw spellings, 6160
records, exactly like ``mandarin_tone_syn``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from pinyin_eval.data.minimal_pairs import (
    FRONT_BACK_NASAL_FINALS,
    I_U_FINAL_PAIR,
    IAN_UAN_FINAL_PAIR,
    PAIR_TYPE_DIMENSION,
    TONE_PAIRS,
    WIDE_NASAL_FINALS,
    MappingEntry,
    MinimalPairInstance,
    build_all_pairs,
    build_instances,
    build_pairs,
    count_table,
    instance_sort_key,
    instances_by_type,
    iter_pair_items,
    load_mapping,
    pair_sort_key,
    pairs_by_type,
)
from pinyin_eval.records import PAIR_TYPES, AudioRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = REPO_ROOT / "data" / "mapping" / "mandarin_tone_syn_target_mapping.csv"

VOICES: tuple[str, ...] = ("A", "B", "C", "D")
TONES: tuple[int, ...] = (1, 2, 3, 4)

#: the seven merged targets and their alternative raw spelling in the dataset.
MERGED_ALTERNATE_SPELLINGS: dict[str, str] = {
    "wei": "ui",
    "wen": "un",
    "wu": "u",
    "yin": "in",
    "ying": "ing",
    "you": "iu",
    "yun": "ün",
}

EXPECTED_PAIR_COUNTS: dict[str, int] = {
    "sh_zh": 36,
    "aspiration": 67,
    "nasal_lateral": 21,
    "front_back_nasal_ptr": 40,
    "i_ü": 5,
    "ian_üan": 3,
    "tone": 2268,
}
EXPECTED_INSTANCE_COUNTS: dict[str, int] = {
    "sh_zh": 576,
    "aspiration": 1072,
    "nasal_lateral": 336,
    "front_back_nasal_ptr": 640,
    "i_ü": 80,
    "ian_üan": 48,
    "tone": 9072,
}
TOTAL_PAIRS = sum(EXPECTED_PAIR_COUNTS.values())  # 2440
TOTAL_INSTANCES = sum(EXPECTED_INSTANCE_COUNTS.values())  # 11824


# ---------------------------------------------------------------------------
# Synthetic fixtures (mirroring the dataset shape)
# ---------------------------------------------------------------------------


def _record(
    *,
    target: str,
    initial: str,
    final: str,
    voice: str,
    tone: int,
    raw_target: str,
    row_index: int,
) -> AudioRecord:
    audio_id = f"ZH_TTS_{voice}_{raw_target}{tone}"
    return AudioRecord(
        audio_id=audio_id,
        row_index=row_index,
        speaker=f"ZH_TTS_{voice}",
        voice=voice,
        raw_target=raw_target,
        target=target,
        initial=initial,
        final=final,
        tone=tone,
        wav_path=f"data/audio/ZH_TTS_{voice}/{audio_id}.wav",
        duration_s=0.5,
        sample_rate=16000,
        channels=1,
    )


def _build_synthetic_records(mapping: dict[str, MappingEntry]) -> list[AudioRecord]:
    records: list[AudioRecord] = []
    for target in sorted(mapping):
        entry = mapping[target]
        spellings = [target]
        alternate = MERGED_ALTERNATE_SPELLINGS.get(target)
        if alternate is not None:
            spellings.append(alternate)
        for voice in VOICES:
            for tone in TONES:
                for raw_target in spellings:
                    records.append(
                        _record(
                            target=target,
                            initial=entry.initial,
                            final=entry.final,
                            voice=voice,
                            tone=tone,
                            raw_target=raw_target,
                            row_index=len(records),
                        )
                    )
    return records


def _instance_identity(instance: MinimalPairInstance) -> tuple:
    """(pair identity, voice, common tone) - the de-duplication key of an instance."""
    return (*instance.pair.identity, instance.voice, instance.tone)


@pytest.fixture(scope="session")
def mapping() -> dict[str, MappingEntry]:
    return load_mapping(MAPPING_PATH)


@pytest.fixture(scope="session")
def targets(mapping: dict[str, MappingEntry]) -> list[str]:
    return sorted(mapping)


@pytest.fixture(scope="session")
def audio_records(mapping: dict[str, MappingEntry]) -> list[AudioRecord]:
    return _build_synthetic_records(mapping)


@pytest.fixture(scope="session")
def pairs(targets: list[str], mapping: dict[str, MappingEntry]) -> list:
    return build_pairs(targets, mapping)


@pytest.fixture(scope="session")
def instances(pairs: list, audio_records: list[AudioRecord]) -> list[MinimalPairInstance]:
    return build_instances(pairs, audio_records)


# ---------------------------------------------------------------------------
# Mapping CSV
# ---------------------------------------------------------------------------


def test_load_mapping_reads_bom_csv(mapping: dict[str, MappingEntry]) -> None:
    assert len(mapping) == 378
    # the UTF-8 BOM must not leak into the first column value.
    assert list(mapping)[0] == "ai"
    assert mapping["ai"] == MappingEntry(target="ai", initial="", final="ai", count=16, source="final")
    assert mapping["zhang"] == MappingEntry(
        target="zhang", initial="zh", final="ang", count=16, source="regular"
    )
    # j/q/x + u are stored as (initial, "ü") by the mapping table itself.
    assert (mapping["ju"].initial, mapping["ju"].final) == ("j", "ü")
    assert (mapping["qu"].initial, mapping["qu"].final) == ("q", "ü")
    # zero initials are "" and keep the underlying final (e.g. yin -> in).
    assert (mapping["yin"].initial, mapping["yin"].final) == ("", "in")
    assert (mapping["an"].initial, mapping["an"].final) == ("", "an")
    assert sum(entry.count for entry in mapping.values()) == 6160
    merged = sorted(target for target, entry in mapping.items() if entry.count == 32)
    assert merged == sorted(MERGED_ALTERNATE_SPELLINGS)


def test_load_mapping_rejects_malformed_csv(tmp_path: Path) -> None:
    missing_column = tmp_path / "missing_column.csv"
    missing_column.write_text("target,initial\nai,\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="missing mapping columns"):
        load_mapping(missing_column)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "target,initial,final,count,source\nai,,ai,16,final\nai,,ai,16,final\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError, match="duplicate target"):
        load_mapping(duplicate)

    empty = tmp_path / "empty.csv"
    empty.write_text("target,initial,final,count,source\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="empty mapping table"):
        load_mapping(empty)


# ---------------------------------------------------------------------------
# Acceptance: design-document counts
# ---------------------------------------------------------------------------


def test_fixture_mirrors_dataset_shape(
    audio_records: list[AudioRecord], mapping: dict[str, MappingEntry]
) -> None:
    assert len(audio_records) == 6160
    assert len({record.raw_target for record in audio_records}) == 385
    assert len({record.target for record in audio_records}) == 378
    for target, entry in mapping.items():
        expected_raw = {target}
        if target in MERGED_ALTERNATE_SPELLINGS:
            expected_raw.add(MERGED_ALTERNATE_SPELLINGS[target])
        target_records = [record for record in audio_records if record.target == target]
        assert {record.raw_target for record in target_records} == expected_raw
        assert all(record.initial == entry.initial for record in target_records)
        assert all(record.final == entry.final for record in target_records)


def test_build_pairs_matches_design_pair_counts(pairs: list) -> None:
    table = count_table(pairs, [])
    assert {pair_type: table[pair_type]["n_pairs"] for pair_type in PAIR_TYPES} == (
        EXPECTED_PAIR_COUNTS
    )
    assert len(pairs) == TOTAL_PAIRS
    assert len({pair.pair_id for pair in pairs}) == len(pairs)
    assert pairs == sorted(pairs, key=pair_sort_key)


def test_build_instances_matches_design_instance_counts(
    pairs: list, instances: list[MinimalPairInstance]
) -> None:
    table = count_table(pairs, instances)
    assert {pair_type: table[pair_type]["n_instances"] for pair_type in PAIR_TYPES} == (
        EXPECTED_INSTANCE_COUNTS
    )
    assert len(instances) == TOTAL_INSTANCES
    # one instance == one two-audio pairing; ids are unique (de-duplicated).
    assert len({instance.pair_id for instance in instances}) == len(instances)
    assert {instance.voice for instance in instances} == set(VOICES)
    assert instances == sorted(instances, key=instance_sort_key)


def test_count_table_shape_and_order(pairs: list, instances: list) -> None:
    table = count_table(pairs, instances)
    expected = {
        pair_type: {
            "n_pairs": EXPECTED_PAIR_COUNTS[pair_type],
            "n_instances": EXPECTED_INSTANCE_COUNTS[pair_type],
        }
        for pair_type in PAIR_TYPES
    }
    assert table == expected
    assert list(table) == list(PAIR_TYPES)
    assert sum(row["n_pairs"] for row in table.values()) == TOTAL_PAIRS
    assert sum(row["n_instances"] for row in table.values()) == TOTAL_INSTANCES


# ---------------------------------------------------------------------------
# Pairing rules
# ---------------------------------------------------------------------------


def test_initial_contrast_pair_types(pairs: list) -> None:
    grouped = pairs_by_type(pairs)
    contract = {
        "sh_zh": (("z", "zh"), ("c", "ch"), ("s", "sh")),
        "aspiration": (("b", "p"), ("d", "t"), ("g", "k"), ("j", "q")),
        "nasal_lateral": (("n", "l"),),
    }
    for pair_type, contrasts in contract.items():
        group = grouped[pair_type]
        assert group
        assert all((pair.initial_a, pair.initial_b) in contrasts for pair in group)
        assert all(pair.final_a == pair.final_b for pair in group)
        assert all(pair.tone_a is None and pair.tone_b is None for pair in group)
        assert all(PAIR_TYPE_DIMENSION[pair.pair_type] == "initial" for pair in group)


def test_sh_zh_excludes_i_group(pairs: list, mapping: dict[str, MappingEntry]) -> None:
    sh_zh = pairs_by_type(pairs)["sh_zh"]
    assert len(sh_zh) == 36
    assert all(pair.final_a != "i" and pair.final_b != "i" for pair in sh_zh)
    pair_targets = {(pair.target_a, pair.target_b) for pair in sh_zh}
    for target_a, target_b in (("zi", "zhi"), ("ci", "chi"), ("si", "shi")):
        assert (target_a, target_b) not in pair_targets
        assert target_a in mapping
        assert target_b in mapping
    assert ("zai", "zhai") in pair_targets
    assert ("sang", "shang") in pair_targets
    # Independently recount from the mapping table: 39 candidate pairs without
    # the -i exclusion, exactly 3 of which belong to the -i group.
    by_final: dict[str, dict[str, set[str]]] = {}
    for target, entry in mapping.items():
        by_final.setdefault(entry.final, {}).setdefault(entry.initial, set()).add(target)
    naive = sum(
        len(groups.get(initial_a, ())) * len(groups.get(initial_b, ()))
        for groups in by_final.values()
        for initial_a, initial_b in (("z", "zh"), ("c", "ch"), ("s", "sh"))
    )
    assert naive == 39
    assert naive - 3 == len(sh_zh)


def test_aspiration_and_nasal_lateral_target_pairs(pairs: list) -> None:
    grouped = pairs_by_type(pairs)
    aspiration_targets = {(pair.target_a, pair.target_b) for pair in grouped["aspiration"]}
    assert ("bai", "pai") in aspiration_targets
    assert ("dan", "tan") in aspiration_targets
    assert ("gai", "kai") in aspiration_targets
    assert ("ji", "qi") in aspiration_targets
    assert len(grouped["aspiration"]) == 67

    nasal_targets = {(pair.target_a, pair.target_b) for pair in grouped["nasal_lateral"]}
    assert ("nai", "lai") in nasal_targets
    assert ("nong", "long") in nasal_targets
    assert len(grouped["nasal_lateral"]) == 21


def test_front_back_nasal_default_scope(pairs: list) -> None:
    front_back = pairs_by_type(pairs)["front_back_nasal_ptr"]
    assert len(front_back) == 40
    assert {(pair.final_a, pair.final_b) for pair in front_back} == set(FRONT_BACK_NASAL_FINALS)
    assert all(pair.initial_a == pair.initial_b for pair in front_back)
    target_pairs = {(pair.target_a, pair.target_b) for pair in front_back}
    assert ("an", "ang") in target_pairs  # zero initial
    assert ("yin", "ying") in target_pairs  # zero initial, merged spellings
    assert ("ban", "bang") in target_pairs
    assert ("lin", "ling") in target_pairs
    assert not any(
        {pair.final_a, pair.final_b} & {"ian", "iang", "uan", "uang"} for pair in front_back
    )


def test_include_wide_nasal_adds_twelve_pairs(
    pairs: list,
    targets: list[str],
    mapping: dict[str, MappingEntry],
    audio_records: list[AudioRecord],
) -> None:
    wide_pairs = build_pairs(targets, mapping, include_wide_nasal=True)
    wide_table = count_table(wide_pairs, [])
    assert wide_table["front_back_nasal_ptr"]["n_pairs"] == 52
    assert len(wide_pairs) == TOTAL_PAIRS + 12

    wide_grouped = pairs_by_type(wide_pairs)
    default_grouped = pairs_by_type(pairs)
    default_ids = {pair.pair_id for pair in default_grouped["front_back_nasal_ptr"]}
    extra = [
        pair for pair in wide_grouped["front_back_nasal_ptr"] if pair.pair_id not in default_ids
    ]
    assert len(extra) == 12
    assert {(pair.final_a, pair.final_b) for pair in extra} == set(WIDE_NASAL_FINALS)
    # only the front/back nasal family changes.
    for pair_type in PAIR_TYPES:
        if pair_type == "front_back_nasal_ptr":
            continue
        assert wide_grouped[pair_type] == default_grouped[pair_type]

    # instances scale accordingly: 640 + 12 * 16 = 832.
    wide_instances = build_instances(wide_pairs, audio_records)
    assert count_table(wide_pairs, wide_instances)["front_back_nasal_ptr"]["n_instances"] == 832
    assert len(wide_instances) == TOTAL_INSTANCES + 12 * len(VOICES) * len(TONES)
    extra_target_pairs = {(pair.target_a, pair.target_b) for pair in extra}
    extra_instance_count = sum(
        1
        for instance in wide_instances
        if instance.pair.pair_type == "front_back_nasal_ptr"
        and (instance.pair.target_a, instance.pair.target_b) in extra_target_pairs
    )
    assert extra_instance_count == 12 * len(VOICES) * len(TONES)


def test_final_contrast_i_u_and_ian_uan(pairs: list) -> None:
    grouped = pairs_by_type(pairs)
    i_u = grouped["i_ü"]
    assert len(i_u) == 5
    assert {(pair.target_a, pair.target_b) for pair in i_u} == {
        ("ji", "ju"),
        ("qi", "qu"),
        ("xi", "xu"),
        ("li", "lü"),
        ("ni", "nü"),
    }
    assert all((pair.final_a, pair.final_b) == I_U_FINAL_PAIR for pair in i_u)
    assert all(pair.initial_a == pair.initial_b for pair in i_u)
    assert all(pair.dimension == "final" for pair in i_u)

    ian_uan = grouped["ian_üan"]
    assert len(ian_uan) == 3
    assert {(pair.target_a, pair.target_b) for pair in ian_uan} == {
        ("jian", "juan"),
        ("qian", "quan"),
        ("xian", "xuan"),
    }
    assert all((pair.final_a, pair.final_b) == IAN_UAN_FINAL_PAIR for pair in ian_uan)
    assert all(pair.initial_a == pair.initial_b for pair in ian_uan)
    assert all(pair.dimension == "final" for pair in ian_uan)


def test_tone_pairs_per_target(pairs: list) -> None:
    tone_pairs = pairs_by_type(pairs)["tone"]
    assert len(tone_pairs) == 2268
    by_target: dict[str, list] = {}
    for pair in tone_pairs:
        by_target.setdefault(pair.target_a, []).append(pair)
    assert len(by_target) == 378
    assert all(len(group) == 6 for group in by_target.values())
    assert {(pair.tone_a, pair.tone_b) for pair in tone_pairs} == set(TONE_PAIRS)
    for pair in tone_pairs:
        assert pair.target_a == pair.target_b
        assert pair.initial_a == pair.initial_b
        assert pair.final_a == pair.final_b
        assert pair.tone_a is not None and pair.tone_b is not None
        assert pair.tone_a < pair.tone_b
        assert pair.dimension == "tone"


def test_pair_level_ids_and_dimensions(pairs: list) -> None:
    for pair in pairs:
        assert pair.dimension == PAIR_TYPE_DIMENSION[pair.pair_type]
        assert pair.targets == (pair.target_a, pair.target_b)
        if pair.pair_type == "tone":
            assert pair.pair_id == f"tone:{pair.target_a}:{pair.tone_a}-{pair.tone_b}"
        else:
            assert pair.pair_id == f"{pair.pair_type}:{pair.target_a}/{pair.target_b}"


# ---------------------------------------------------------------------------
# Instance semantics
# ---------------------------------------------------------------------------


def test_instance_sides_are_consistent(instances: list[MinimalPairInstance]) -> None:
    for instance in instances:
        pair = instance.pair
        assert instance.voice in VOICES
        assert instance.audio_a.target == pair.target_a
        assert instance.audio_b.target == pair.target_b
        assert instance.audio_a.voice == instance.audio_b.voice == instance.voice
        assert instance.audio_a.audio_id != instance.audio_b.audio_id
        assert instance.side_audio("A") == instance.audio_a
        assert instance.side_audio("B") == instance.audio_b
        sides = {
            side: (audio.audio_id, other.audio_id)
            for side, audio, other in instance.iter_sides()
        }
        assert sides == {
            "A": (instance.audio_a.audio_id, instance.audio_b.audio_id),
            "B": (instance.audio_b.audio_id, instance.audio_a.audio_id),
        }
        if pair.pair_type == "tone":
            assert instance.tone is None
            assert instance.tone_a == pair.tone_a
            assert instance.tone_b == pair.tone_b
            assert instance.audio_a.tone == pair.tone_a
            assert instance.audio_b.tone == pair.tone_b
        else:
            assert instance.tone in TONES
            assert instance.tone_a == instance.tone_b == instance.tone
            assert instance.audio_a.tone == instance.audio_b.tone == instance.tone
    with pytest.raises(ValueError, match="side must be"):
        instances[0].side_audio("C")


def test_segmental_instances_are_complete_cross_products(
    instances: list[MinimalPairInstance],
) -> None:
    grouped = instances_by_type(instances)
    for pair_type in PAIR_TYPES:
        if pair_type == "tone":
            continue
        group = grouped[pair_type]
        expected = EXPECTED_PAIR_COUNTS[pair_type] * len(VOICES) * len(TONES)
        assert len(group) == expected
        keys = {
            (instance.pair.identity, instance.voice, instance.tone) for instance in group
        }
        assert len(keys) == len(group)


def test_tone_instances_are_complete_per_voice(instances: list[MinimalPairInstance]) -> None:
    group = instances_by_type(instances)["tone"]
    assert len(group) == EXPECTED_PAIR_COUNTS["tone"] * len(VOICES)
    keys = {(instance.pair.identity, instance.voice) for instance in group}
    assert len(keys) == len(group)
    assert all(instance.tone is None for instance in group)


# ---------------------------------------------------------------------------
# Determinism / de-duplication
# ---------------------------------------------------------------------------


def test_pairs_output_order_is_deterministic(
    pairs: list, targets: list[str], mapping: dict[str, MappingEntry]
) -> None:
    assert build_pairs(list(reversed(targets)), mapping) == pairs
    assert build_pairs(targets + targets[:20], mapping) == pairs
    assert pairs == sorted(pairs, key=pair_sort_key)
    for group in pairs_by_type(pairs).values():
        assert group == sorted(group, key=pair_sort_key)


def test_instances_output_is_deterministic_and_deduped(
    pairs: list, instances: list[MinimalPairInstance], audio_records: list[AudioRecord]
) -> None:
    shuffled_records = list(audio_records)
    random.Random(0).shuffle(shuffled_records)
    shuffled_pairs = list(pairs)
    random.Random(1).shuffle(shuffled_pairs)

    assert build_instances(shuffled_pairs, shuffled_records) == instances
    # duplicated pairs and records must not duplicate instances.
    assert build_instances(pairs + pairs[:30], audio_records + audio_records[:30]) == instances
    assert instances == sorted(instances, key=instance_sort_key)
    assert len({_instance_identity(instance) for instance in instances}) == len(instances)


# ---------------------------------------------------------------------------
# Missing audio / merged spellings
# ---------------------------------------------------------------------------


def test_missing_single_audio_drops_instances(
    pairs: list, instances: list[MinimalPairInstance], audio_records: list[AudioRecord]
) -> None:
    missing_audio_id = "ZH_TTS_B_zhai3"
    reduced_records = [record for record in audio_records if record.audio_id != missing_audio_id]
    assert len(reduced_records) == len(audio_records) - 1

    reduced = build_instances(pairs, reduced_records)
    # one sh_zh instance (zai/zhai B tone3) + three zhai tone instances (1,3)(2,3)(3,4).
    assert len(reduced) == len(instances) - 1 - 3

    keys = {_instance_identity(instance) for instance in reduced}
    assert ("sh_zh", "zai", "zhai", None, None, "B", 3) not in keys
    assert ("sh_zh", "zai", "zhai", None, None, "B", 2) in keys
    zhai_tone_keys = {
        key
        for key in keys
        if key[0] == "tone" and key[1] == "zhai" and key[5] == "B"
    }
    assert {key[3:5] for key in zhai_tone_keys} == {(1, 2), (1, 4), (2, 4)}
    # the target-level pair itself is unaffected by missing audio.
    assert len(pairs) == TOTAL_PAIRS


def test_missing_whole_target_drops_instances(
    pairs: list, audio_records: list[AudioRecord]
) -> None:
    reduced_records = [record for record in audio_records if record.target != "zhai"]
    assert len(reduced_records) == 6160 - 16
    reduced = build_instances(pairs, reduced_records)
    table = count_table(pairs, reduced)
    assert table["sh_zh"]["n_instances"] == EXPECTED_INSTANCE_COUNTS["sh_zh"] - 16
    assert table["tone"]["n_instances"] == EXPECTED_INSTANCE_COUNTS["tone"] - 6 * 4
    assert len(reduced) == TOTAL_INSTANCES - 16 - 24
    assert not any("zhai" in (instance.pair.target_a, instance.pair.target_b) for instance in reduced)


def test_merged_targets_prefer_canonical_raw_spelling(instances: list[MinimalPairInstance]) -> None:
    # each (target, voice, tone) resolves to exactly one chosen audio.
    chosen: dict[tuple[str, str, int], AudioRecord] = {}
    for instance in instances:
        for audio in (instance.audio_a, instance.audio_b):
            key = (audio.target, audio.voice, audio.tone)
            assert chosen.setdefault(key, audio) == audio
    assert len(chosen) == 378 * 16

    for target in MERGED_ALTERNATE_SPELLINGS:
        for voice in VOICES:
            for tone in TONES:
                audio = chosen[(target, voice, tone)]
                assert audio.raw_target == target
                assert audio.audio_id == f"ZH_TTS_{voice}_{target}{tone}"
    for (target, _voice, _tone), audio in chosen.items():
        if target not in MERGED_ALTERNATE_SPELLINGS:
            assert audio.raw_target == target


def test_merged_targets_fall_back_to_alternate_spelling(
    pairs: list, audio_records: list[AudioRecord]
) -> None:
    records = [
        record
        for record in audio_records
        if not (record.target == "yin" and record.raw_target == "yin")
    ]
    fallback = build_instances(pairs, records)
    # the alternate spelling covers every canonical slot, so no instance is lost.
    assert len(fallback) == TOTAL_INSTANCES
    yin_audios = [
        audio
        for instance in fallback
        for audio in (instance.audio_a, instance.audio_b)
        if audio.target == "yin"
    ]
    assert len(yin_audios) == 4 * 4 + 48  # front/back nasal (16) + tone pairs (2 * 24)
    assert all(audio.raw_target == "in" for audio in yin_audios)


# ---------------------------------------------------------------------------
# Input flexibility / edge cases
# ---------------------------------------------------------------------------


def test_build_all_pairs_matches_build_pairs(
    pairs: list,
    targets: list[str],
    mapping: dict[str, MappingEntry],
    audio_records: list[AudioRecord],
) -> None:
    assert build_all_pairs(audio_records, mapping) == pairs
    shuffled_records = list(audio_records)
    random.Random(11).shuffle(shuffled_records)
    assert build_all_pairs(shuffled_records, mapping) == pairs
    expected_wide = build_pairs(targets, mapping, include_wide_nasal=True)
    assert build_all_pairs(audio_records, mapping, include_wide_nasal=True) == expected_wide


def test_build_pairs_accepts_tuple_mapping(
    pairs: list, targets: list[str], mapping: dict[str, MappingEntry]
) -> None:
    tuple_mapping = {
        target: (entry.initial, entry.final) for target, entry in mapping.items()
    }
    assert build_pairs(targets, tuple_mapping) == pairs


def test_build_pairs_accepts_duck_typed_mapping_entries(
    pairs: list, targets: list[str], mapping: dict[str, MappingEntry]
) -> None:
    @dataclass(frozen=True, slots=True)
    class ForeignEntry:
        initial: str
        final: str

    foreign = {
        target: ForeignEntry(entry.initial, entry.final) for target, entry in mapping.items()
    }
    assert build_pairs(targets, foreign) == pairs


def test_build_pairs_rejects_unknown_target(mapping: dict[str, MappingEntry]) -> None:
    with pytest.raises(ValueError, match="missing from mapping"):
        build_pairs(["zhang", "zzz"], mapping)


def test_empty_inputs(mapping: dict[str, MappingEntry]) -> None:
    empty_table = {pair_type: {"n_pairs": 0, "n_instances": 0} for pair_type in PAIR_TYPES}
    empty_grouped = {pair_type: [] for pair_type in PAIR_TYPES}
    assert build_pairs([], mapping) == []
    assert build_pairs([], {}) == []
    assert build_all_pairs([], mapping) == []
    assert build_instances([], []) == []
    assert count_table([], []) == empty_table
    assert pairs_by_type([]) == empty_grouped
    assert instances_by_type([]) == empty_grouped
    assert list(iter_pair_items([])) == []


# ---------------------------------------------------------------------------
# Helpers for tasks.py
# ---------------------------------------------------------------------------


def test_grouping_helpers(pairs: list, instances: list[MinimalPairInstance]) -> None:
    pair_groups = pairs_by_type(pairs)
    assert list(pair_groups) == list(PAIR_TYPES)
    assert {pair_type: len(group) for pair_type, group in pair_groups.items()} == (
        EXPECTED_PAIR_COUNTS
    )
    assert all(
        pair.pair_type == pair_type
        for pair_type, group in pair_groups.items()
        for pair in group
    )

    instance_groups = instances_by_type(instances)
    assert list(instance_groups) == list(PAIR_TYPES)
    assert {pair_type: len(group) for pair_type, group in instance_groups.items()} == (
        EXPECTED_INSTANCE_COUNTS
    )
    for group in instance_groups.values():
        assert group == sorted(group, key=instance_sort_key)


def test_iter_pair_items_flattens_instances(instances: list[MinimalPairInstance]) -> None:
    items = list(iter_pair_items(instances))
    assert len(items) == 2 * len(instances)

    expected: list[tuple[str, str, str]] = []
    for instance in sorted(instances, key=instance_sort_key):
        expected.append((instance.pair_id, "A", instance.audio_a.audio_id))
        expected.append((instance.pair_id, "B", instance.audio_b.audio_id))
    assert [(item.pair_id, item.side, item.audio_id) for item in items] == expected

    for index, item in enumerate(items):
        assert item.side == ("A" if index % 2 == 0 else "B")
        assert item.voice in VOICES
        assert item.pair_type == item.pair.pair_type
        assert item.dimension == PAIR_TYPE_DIMENSION[item.pair_type]
        if item.side == "A":
            assert item.audio.target == item.pair.target_a
            assert item.other_audio.target == item.pair.target_b
        else:
            assert item.audio.target == item.pair.target_b
            assert item.other_audio.target == item.pair.target_a
