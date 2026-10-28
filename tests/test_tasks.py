"""Tests for task construction under the 2026-10-28 design update.

Covers experiments 1-3 (tone / tone / free), the new detection experiment, the
minimal-pair experiments 4-6 and the ablation subsets.
"""

from __future__ import annotations

import pytest

from pinyin_eval import tasks as T
from pinyin_eval.data.minimal_pairs import build_all_pairs, build_instances, count_table
from pinyin_eval.records import TaskItem

from .conftest import make_synthetic_records

_MAPPING_BY_IF: dict[tuple[str, str], str] = {}


@pytest.fixture(scope="module", autouse=True)
def _target_lookup(mapping):
    _MAPPING_BY_IF.clear()
    _MAPPING_BY_IF.update({(e.initial, e.final): t for t, e in mapping.items()})
    yield


@pytest.fixture(scope="module")
def full_records(mapping):
    return make_synthetic_records(mapping)


@pytest.fixture(scope="module")
def core_tasks(full_records, repo_cfg, mapping):
    return T.build_core_tasks(full_records, repo_cfg, mapping)


@pytest.fixture(scope="module")
def detection_tasks(full_records, repo_cfg, mapping):
    return T.build_detection_tasks(full_records, repo_cfg, mapping)


@pytest.fixture(scope="module")
def pair_tasks(full_records, repo_cfg, mapping):
    return T.build_pair_tasks(full_records, repo_cfg, mapping)


def _target_of(task: TaskItem) -> str:
    return _MAPPING_BY_IF[(task.gold_initial, task.gold_final)]


def _n_audio() -> int:
    return 378 * 4 * 4


# ---------------------------------------------------------------------------
# Core tasks (experiments 1-3)
# ---------------------------------------------------------------------------


def test_core_task_counts_and_types(core_tasks):
    # 7 tasks per audio (free + correct + 5 wrong variants) minus 2 missing
    # initial variants for er/weng (32 audios)
    assert len(core_tasks) == _n_audio() * 7 - 2 * 16 * 2
    by_exp = {}
    for task in core_tasks:
        by_exp.setdefault(task.experiment, set()).add(task.task_type)
        assert task.prompt_texts and set(task.prompt_texts) == {"json", "text"}
        assert task.system_prompt
        assert task.prompt_version
        assert task.cluster_id == task.audio_id
    assert by_exp == {"exp1": {"tone"}, "exp2": {"tone"}, "exp3": {"free"}}


def test_exp1_gives_correct_toneless_target_for_tone_only(core_tasks):
    exp1 = [t for t in core_tasks if t.experiment == "exp1"]
    assert len(exp1) == _n_audio()
    for task in exp1[:300]:
        assert task.given_target == _target_of(task)
        assert task.given_tone is None
        assert task.consistent_gold is True
        assert task.task_type == "tone"


def test_exp2_five_segmental_toneless_variants(core_tasks, mapping):
    exp2 = [t for t in core_tasks if t.experiment == "exp2"]
    per_audio: dict[str, list[TaskItem]] = {}
    for task in exp2:
        per_audio.setdefault(task.audio_id, []).append(task)
    # er/weng have only 3 variants (2 final + 1 extra final), all others 5
    assert set(len(v) for v in per_audio.values()) == {3, 5}
    assert len(exp2) == _n_audio() * 5 - 2 * 16 * 2
    for tasks in per_audio.values():
        for task in tasks:
            assert task.task_type == "tone"
            assert task.given_tone is None  # true tone is never revealed
            assert task.consistent_gold is False
            assert task.variant_type in ("initial", "final")
            given = mapping[task.given_target]
            if task.variant_type == "initial":
                assert given.initial != tasks[0].gold_initial
                assert given.final == tasks[0].gold_final
            else:
                assert given.initial == tasks[0].gold_initial
                assert given.final != tasks[0].gold_final


def test_exp3_free_has_no_target(core_tasks):
    exp3 = [t for t in core_tasks if t.experiment == "exp3"]
    assert len(exp3) == _n_audio()
    for task in exp3[:300]:
        assert task.task_type == "free"
        assert task.given_target is None and task.given_tone is None
        assert task.consistent_gold is None


# ---------------------------------------------------------------------------
# Detection experiment
# ---------------------------------------------------------------------------


def test_detection_task_structure(detection_tasks):
    per_audio: dict[str, list[TaskItem]] = {}
    for task in detection_tasks:
        assert task.experiment == "detect"
        assert task.prompt_condition == "detect"
        assert task.task_type == "detect"
        assert task.given_target is not None
        assert task.given_tone in (1, 2, 3, 4)
        per_audio.setdefault(task.audio_id, []).append(task)
    # 6 tasks per audio except er/weng (no same-final alternative -> 4)
    assert set(len(v) for v in per_audio.values()) == {4, 6}
    assert len(detection_tasks) == _n_audio() * 6 - 2 * 16 * 2


def test_detection_consistent_and_inconsistent_claims(detection_tasks):
    n_consistent = 0
    for task in detection_tasks:
        if task.consistent_gold is True:
            n_consistent += 1
            assert task.variant_type is None
            assert task.given_target == _target_of(task)
            assert task.given_tone == task.gold_tone
        else:
            assert task.consistent_gold is False
            if task.variant_type in ("initial", "final"):
                assert task.given_tone == task.gold_tone  # only segments differ
                assert task.given_target != _target_of(task)
            elif task.variant_type == "tone":
                assert task.given_target == _target_of(task)
                assert task.given_tone != task.gold_tone
            else:
                raise AssertionError(task.variant_type)
    assert n_consistent == _n_audio()


def test_detection_sets(detection_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    assert set(detection["subset"]) == {"main", "initial", "final", "tone"}
    sizes = detection.groupby("subset").size().to_dict()
    assert sizes["main"] == _n_audio()
    assert sizes["initial"] == _n_audio() - 32
    assert sizes["final"] == _n_audio()
    assert sizes["tone"] == _n_audio()
    item_ids = {t.item_id for t in detection_tasks}
    assert set(detection["negative_item_id"]) <= item_ids
    assert set(detection["positive_item_id"]) <= item_ids
    main_counts = detection[detection["subset"] == "main"]["variant_type"].value_counts()
    assert abs(main_counts["initial"] / main_counts["final"] - 1.0) < 0.15
    assert abs((main_counts["initial"] + main_counts["final"]) / main_counts["tone"] - 4.0) < 0.4
    neg_ids = {t.item_id for t in detection_tasks if t.consistent_gold is True}
    assert set(detection["negative_item_id"]) == neg_ids


def test_detection_sets_deterministic(detection_tasks, repo_cfg):
    assert T.build_detection_sets(detection_tasks, repo_cfg).equals(
        T.build_detection_sets(detection_tasks, repo_cfg)
    )


# ---------------------------------------------------------------------------
# Pair tasks (experiments 4-6)
# ---------------------------------------------------------------------------


def test_pair_task_counts_and_uniqueness(pair_tasks, full_records, mapping):
    pairs = build_all_pairs(full_records, mapping)
    instances = build_instances(pairs, full_records)
    table = count_table(pairs, instances)
    assert table["sh_zh"] == {"n_pairs": 36, "n_instances": 576}
    assert table["tone"] == {"n_pairs": 2268, "n_instances": 9072}
    by_exp = {}
    for task in pair_tasks:
        by_exp.setdefault(task.experiment, set()).add(task.task_type)
    assert by_exp["exp4"] == {"free", "tone", "ab"}
    assert by_exp["exp5"] == {"free", "tone", "ab"}
    assert by_exp["exp6"] == {"free", "tone", "ab"}
    assert len({t.item_id for t in pair_tasks}) == len(pair_tasks)
    # exp6: 3 prompt conditions (wrong target is not expressible), others 4
    prompt_conditions = {}
    for task in pair_tasks:
        prompt_conditions.setdefault(task.experiment, set()).add(task.prompt_condition)
    assert prompt_conditions["exp6"] == {"free", "correct_target", "ab_choice"}
    assert prompt_conditions["exp4"] == {"free", "correct_target", "wrong_target", "ab_choice"}
    assert prompt_conditions["exp5"] == {"free", "correct_target", "wrong_target", "ab_choice"}


def test_pair_target_conditions(pair_tasks):
    for task in pair_tasks:
        if task.prompt_condition == "correct_target":
            assert task.task_type == "tone"
            assert task.consistent_gold is True
            assert task.given_target == _target_of(task)
            assert task.given_tone is None
        elif task.prompt_condition == "wrong_target":
            assert task.experiment in ("exp4", "exp5")
            assert task.task_type == "tone"
            assert task.consistent_gold is False
            assert task.given_target != _target_of(task)
            assert task.given_tone is None


def test_pair_ab_choice(pair_tasks):
    ab = [t for t in pair_tasks if t.prompt_condition == "ab_choice"]
    assert ab
    for task in ab[:500]:
        assert task.task_type == "ab"
        assert task.alternatives and len(task.alternatives) == 2
        own = f"{_target_of(task)}{task.gold_tone}"
        assert own in task.alternatives
        assert task.alternatives[{"A": 0, "B": 1}[task.correct_choice]] == own


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_ablation_subsets(core_tasks, detection_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    selection = T.sample_ablation(
        repo_cfg, core_tasks, pair_tasks, detection_tasks, detection_sets=detection,
    )
    assert len(selection["core"]) == 3 * 256
    assert len(selection["pairs"]) > 0
    by_id = {t.item_id: t for t in detection_tasks}
    selected = [by_id[i] for i in selection["detect"]]
    assert len({t.audio_id for t in selected}) == 256
    # 6 tasks per audio, except er/weng which only have 4 detection tasks
    assert len(selected) == sum(
        1 for t in detection_tasks if t.audio_id in {x.audio_id for x in selected}
    )
    assert 256 * 6 - 32 * 2 <= len(selected) <= 256 * 6
    selected_core = {t.item_id: t for t in core_tasks if t.item_id in set(selection["core"])}
    counts = {}
    for task in selected_core.values():
        counts[task.prompt_condition] = counts.get(task.prompt_condition, 0) + 1
    assert counts == {"free": 256, "correct_target": 256, "wrong_target": 256}
    cells = {(t.voice, t.gold_tone) for t in selected_core.values()}
    assert cells == {(v, tone) for v in "ABCD" for tone in (1, 2, 3, 4)}


def test_ablation_deterministic(core_tasks, detection_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    a = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_tasks,
                          detection_sets=detection)
    b = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_tasks,
                          detection_sets=detection)
    assert a["core"] == b["core"] and a["detect"] == b["detect"] and a["pairs"] == b["pairs"]


def test_ablation_pairs_cap(core_tasks, detection_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    selection = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_tasks,
                                  detection_sets=detection)
    summary = selection["pair_instance_summary"]
    assert summary["i_ü"]["selected"] == 80
    assert summary["ian_üan"]["selected"] == 48
    assert summary["sh_zh"]["selected"] == 256
    assert summary["tone"]["selected"] == 256


# ---------------------------------------------------------------------------
# Serialisation / summary
# ---------------------------------------------------------------------------


def test_task_jsonl_roundtrip(core_tasks, tmp_path):
    path = tmp_path / "core.jsonl"
    T.write_tasks(core_tasks[:20], path)
    loaded = T.load_tasks(path)
    assert [t.to_dict() for t in loaded] == [t.to_dict() for t in core_tasks[:20]]


def test_summarize_tasks(core_tasks, detection_tasks):
    summary = T.summarize_tasks(core_tasks)
    assert summary["n_items"] == len(core_tasks)
    assert summary["by_experiment"]["exp2"] > summary["by_experiment"]["exp1"]
    det_summary = T.summarize_tasks(detection_tasks)
    assert det_summary["by_experiment"]["detect"] == len(detection_tasks)
