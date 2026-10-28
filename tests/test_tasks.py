"""Tests for task construction: experiments 1-6, detection sets, ablation subsets."""

from __future__ import annotations

import pytest

from pinyin_eval import tasks as T
from pinyin_eval.data.minimal_pairs import build_all_pairs, build_instances, count_table
from pinyin_eval.normalize import load_mapping
from pinyin_eval.records import TaskItem

from .conftest import make_synthetic_records


@pytest.fixture(scope="module")
def full_records(mapping):
    return make_synthetic_records(mapping)


@pytest.fixture(scope="module")
def core_tasks(full_records, repo_cfg, mapping):
    return T.build_core_tasks(full_records, repo_cfg, mapping)


@pytest.fixture(scope="module")
def pair_tasks(full_records, repo_cfg, mapping):
    return T.build_pair_tasks(full_records, repo_cfg, mapping)


# ---------------------------------------------------------------------------
# Core tasks (experiments 1-3)
# ---------------------------------------------------------------------------


def test_core_task_counts(core_tasks, mapping):
    n_audio = 378 * 4 * 4
    assert len(core_tasks) == n_audio * 7 - 2 * 16 * 2  # er/weng miss 2 initial variants
    # every item has prompts and a stable cluster
    for task in core_tasks[:100]:
        assert task.prompt_texts and set(task.prompt_texts) == {"json", "text"}
        assert task.prompt_version
        assert task.cluster_id == task.audio_id


def test_exp1_view_gives_correct_toneless_target(core_tasks):
    exp1 = [t for t in core_tasks if t.experiment == "exp1"]
    assert exp1
    for task in exp1[:50]:
        assert task.prompt_condition == "correct_target"
        assert task.consistent_gold is True
        assert task.given_target is not None and task.given_tone is None
        # given target must be the canonical spelling of the audio's syllable
        assert task.given_target.endswith(task.gold_final) or task.given_target in {
            "yin", "ying", "you", "wu", "wei", "wen", "yun",
        }


def test_exp2_initial_variants_change_only_initial(core_tasks, mapping):
    exp2 = [t for t in core_tasks if t.experiment == "exp2"]
    assert len(exp2) > 0
    for task in exp2:
        if task.variant_type != "initial":
            continue
        given = mapping[task.given_target]
        assert given.initial != task.gold_initial
        assert given.final == task.gold_final
        assert task.consistent_gold is False
        assert task.given_tone is None


def test_exp2_final_variants_change_only_final(core_tasks, mapping):
    for task in core_tasks:
        if task.experiment != "exp2" or task.variant_type != "final":
            continue
        given = mapping[task.given_target]
        assert given.initial == task.gold_initial
        assert given.final != task.gold_final
        assert task.consistent_gold is False


def test_exp2_tone_variants_carry_wrong_tone(core_tasks):
    tone_variants = [
        t for t in core_tasks if t.experiment == "exp2" and t.variant_type == "tone"
    ]
    assert tone_variants
    for task in tone_variants:
        assert task.given_tone is not None and task.given_tone != task.gold_tone
        assert task.given_target == _target_of(task)
        assert task.consistent_gold is False


def _target_of(task: TaskItem) -> str:
    # canonical target reconstructed from gold segments via the mapping table
    return _TARGET_BY_IF[(task.gold_initial, task.gold_final)]


@pytest.fixture(scope="module", autouse=True)
def _target_lookup():
    global _TARGET_BY_IF
    mapping = load_mapping("data/mapping/mandarin_tone_syn_target_mapping.csv")
    _TARGET_BY_IF = {(e.initial, e.final): t for t, e in mapping.items()}


def test_exp3_free_has_no_target(core_tasks):
    exp3 = [t for t in core_tasks if t.experiment == "exp3"]
    assert exp3
    for task in exp3[:50]:
        assert task.prompt_condition == "free"
        assert task.given_target is None and task.given_tone is None
        assert task.consistent_gold is None


def test_detection_sets(core_tasks, repo_cfg):
    detection = T.build_detection_sets(core_tasks, repo_cfg)
    assert set(detection["subset"]) == {"main", "initial", "final", "tone"}
    assert set(detection["variant_type"]) == {"initial", "final", "tone"}
    sizes = detection.groupby("subset").size().to_dict()
    # every audio has a main pair; er/weng have no initial alternative
    assert sizes["main"] == 6048
    assert sizes["initial"] == 6048 - 32
    assert sizes["final"] == 6048
    assert sizes["tone"] == 6048
    item_ids = {t.item_id for t in core_tasks}
    assert set(detection["negative_item_id"]) <= item_ids
    assert set(detection["positive_item_id"]) <= item_ids
    # main positives mix error types ~2:2:1
    main_counts = detection[detection["subset"] == "main"]["variant_type"].value_counts()
    assert abs(main_counts["initial"] / main_counts["final"] - 1.0) < 0.15
    assert abs((main_counts["initial"] + main_counts["final"]) / main_counts["tone"] - 4.0) < 0.4


def test_detection_sets_deterministic(core_tasks, repo_cfg):
    a = T.build_detection_sets(core_tasks, repo_cfg)
    b = T.build_detection_sets(core_tasks, repo_cfg)
    assert a.equals(b)


# ---------------------------------------------------------------------------
# Pair tasks (experiments 4-6)
# ---------------------------------------------------------------------------


def test_pair_task_counts_and_uniqueness(pair_tasks, full_records, mapping):
    pairs = build_all_pairs(full_records, mapping)
    instances = build_instances(pairs, full_records)
    table = count_table(pairs, instances)
    assert table["sh_zh"] == {"n_pairs": 36, "n_instances": 576}
    assert table["tone"] == {"n_pairs": 2268, "n_instances": 9072}
    n_audio_items = sum(v["n_instances"] for v in table.values()) * 2
    assert len(pair_tasks) == n_audio_items * 4  # 4 prompt conditions
    assert len({t.item_id for t in pair_tasks}) == len(pair_tasks)


def test_pair_task_experiment_mapping(pair_tasks):
    by_exp = {}
    for task in pair_tasks:
        by_exp.setdefault(task.experiment, set()).add(task.pair_type)
    assert by_exp["exp4"] == {"sh_zh", "aspiration", "nasal_lateral"}
    assert by_exp["exp5"] == {"front_back_nasal_ptr", "i_ü", "ian_üan"}
    assert by_exp["exp6"] == {"tone"}


def test_pair_wrong_target_semantics(pair_tasks):
    for task in pair_tasks:
        if task.prompt_condition != "wrong_target":
            continue
        assert task.consistent_gold is False
        assert task.given_target is not None
        if task.experiment == "exp6":
            assert task.variant_type == "tone"
            assert task.given_tone is not None and task.given_tone != task.gold_tone
            assert task.given_target == _target_of(task)
        else:
            assert task.variant_type in ("initial", "final")
            assert task.given_tone is None
            assert task.given_target != _target_of(task)


def test_pair_ab_choice(pair_tasks):
    ab = [t for t in pair_tasks if t.prompt_condition == "ab_choice"]
    assert ab
    for task in ab[:200]:
        assert task.alternatives and len(task.alternatives) == 2
        assert task.correct_choice in ("A", "B")
        own = f"{_target_of(task)}{task.gold_tone}"
        assert own in task.alternatives
        assert task.alternatives[{"A": 0, "B": 1}[task.correct_choice]] == own


def test_pair_correct_target(pair_tasks):
    for task in pair_tasks:
        if task.prompt_condition != "correct_target":
            continue
        assert task.consistent_gold is True
        assert task.given_target == _target_of(task)
        assert task.given_tone is None


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_ablation_subsets_deterministic(core_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(core_tasks, repo_cfg)
    a = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_sets=detection)
    b = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_sets=detection)
    assert a["core"] == b["core"]
    assert a["pairs"] == b["pairs"]


def test_ablation_core_has_256_items_per_prompt_condition(core_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(core_tasks, repo_cfg)
    selection = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_sets=detection)
    selected = set(selection["core"])
    assert len(selected) == 3 * 256
    by_prompt = {"free": set(), "correct_target": set(), "wrong_target": set()}
    for task in core_tasks:
        if task.item_id in selected:
            by_prompt[task.prompt_condition].add(task.item_id)
    assert {k: len(v) for k, v in by_prompt.items()} == {
        "free": 256, "correct_target": 256, "wrong_target": 256,
    }
    cells = {(t.voice, t.gold_tone) for t in core_tasks if t.item_id in selected}
    assert cells == {(v, tone) for v in "ABCD" for tone in (1, 2, 3, 4)}
    # every selected audio keeps a complete detection pair -> 256 balanced pairs
    by_id = {t.item_id: t for t in core_tasks}
    audios_with_neg = {by_id[i].audio_id for i in selected if i in by_id and by_id[i].experiment == "exp1"}
    audios_with_pos = {by_id[i].audio_id for i in selected if i in by_id and by_id[i].experiment == "exp2"}
    assert audios_with_neg == audios_with_pos and len(audios_with_neg) == 256


def test_ablation_pairs_cap(core_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(core_tasks, repo_cfg)
    selection = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_sets=detection)
    summary = selection["pair_instance_summary"]
    assert summary["i_ü"]["available"] == 80 and summary["i_ü"]["selected"] == 80
    assert summary["ian_üan"]["available"] == 48 and summary["ian_üan"]["selected"] == 48
    assert summary["sh_zh"]["selected"] == 256
    assert summary["tone"]["selected"] == 256


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_task_jsonl_roundtrip(core_tasks, tmp_path):
    path = tmp_path / "core.jsonl"
    T.write_tasks(core_tasks[:20], path)
    loaded = T.load_tasks(path)
    assert len(loaded) == 20
    assert [t.to_dict() for t in loaded] == [t.to_dict() for t in core_tasks[:20]]


def test_summarize_tasks(core_tasks):
    summary = T.summarize_tasks(core_tasks)
    assert summary["n_items"] == len(core_tasks)
    assert summary["by_experiment"]["exp2"] > summary["by_experiment"]["exp1"]
    assert summary["exp2_variant_types"]["tone"] == 6048


def test_prompt_texts_embedded(core_tasks):
    for task in core_tasks[:20]:
        assert "json" in task.prompt_texts and "text" in task.prompt_texts
        assert task.prompt_texts["json"] != task.prompt_texts["text"]
