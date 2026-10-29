"""Tests for task construction under the v3 design (2026-10-28 20:37).

Covers experiments 1-3 (tone / tone / free), the detection experiment, the
experiments 4-6 sub-experiments (free / correct info / wrong info, no A/B) and
the ablation / minimal subsets.
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
    assert len(core_tasks) == _n_audio() * 7 - 2 * 16 * 2
    by_exp = {}
    for task in core_tasks:
        by_exp.setdefault(task.experiment, set()).add(task.task_type)
        assert task.prompt_texts and set(task.prompt_texts) == {"json", "text"}
        assert task.system_prompt
        assert task.prompt_version
        assert task.cluster_id == task.audio_id
    assert by_exp == {"exp1": {"tone"}, "exp2": {"tone"}, "exp3": {"free"}}


def test_exp1_correct_toneless_target(core_tasks):
    exp1 = [t for t in core_tasks if t.experiment == "exp1"]
    assert len(exp1) == _n_audio()
    for task in exp1[:300]:
        assert task.given_target == _target_of(task)
        assert task.given_tone is None
        assert task.consistent_gold is True


def test_exp2_five_segmental_toneless_variants(core_tasks, mapping):
    exp2 = [t for t in core_tasks if t.experiment == "exp2"]
    per_audio: dict[str, list[TaskItem]] = {}
    for task in exp2:
        per_audio.setdefault(task.audio_id, []).append(task)
    assert set(len(v) for v in per_audio.values()) == {3, 5}
    for tasks in per_audio.values():
        for task in tasks:
            assert task.task_type == "tone"
            assert task.given_tone is None
            assert task.consistent_gold is False
            assert task.variant_type in ("initial", "final")
            given = mapping[task.given_target]
            if task.variant_type == "initial":
                assert given.initial != tasks[0].gold_initial
                assert given.final == tasks[0].gold_final
            else:
                assert given.initial == tasks[0].gold_initial
                assert given.final != tasks[0].gold_final


def test_exp3_free(core_tasks):
    exp3 = [t for t in core_tasks if t.experiment == "exp3"]
    assert len(exp3) == _n_audio()
    for task in exp3[:300]:
        assert task.task_type == "free"
        assert task.given_target is None and task.given_tone is None


# ---------------------------------------------------------------------------
# Detection experiment
# ---------------------------------------------------------------------------


def test_detection_task_structure(detection_tasks):
    per_audio: dict[str, list[TaskItem]] = {}
    for task in detection_tasks:
        assert task.experiment == "detect"
        assert task.prompt_condition == "detect"
        assert task.task_type == "detect"
        assert task.given_target is not None and task.given_tone in (1, 2, 3, 4)
        per_audio.setdefault(task.audio_id, []).append(task)
    assert set(len(v) for v in per_audio.values()) == {4, 6}
    assert len(detection_tasks) == _n_audio() * 6 - 2 * 16 * 2


def test_detection_claims(detection_tasks):
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
                assert task.given_tone == task.gold_tone
                assert task.given_target != _target_of(task)
            elif task.variant_type == "tone":
                assert task.given_target == _target_of(task)
                assert task.given_tone != task.gold_tone
    assert n_consistent == _n_audio()


def test_detection_sets(detection_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    assert set(detection["subset"]) == {"main", "initial", "final", "tone"}
    sizes = detection.groupby("subset").size().to_dict()
    assert sizes["main"] == _n_audio()
    assert sizes["initial"] == _n_audio() - 32
    assert sizes["final"] == _n_audio()
    assert sizes["tone"] == _n_audio()
    neg_ids = {t.item_id for t in detection_tasks if t.consistent_gold is True}
    assert set(detection["negative_item_id"]) == neg_ids


def test_detection_sets_deterministic(detection_tasks, repo_cfg):
    assert T.build_detection_sets(detection_tasks, repo_cfg).equals(
        T.build_detection_sets(detection_tasks, repo_cfg)
    )


# ---------------------------------------------------------------------------
# Pair tasks: experiments 4-6 x three sub-experiments (v3)
# ---------------------------------------------------------------------------


def test_pair_task_counts_and_uniqueness(pair_tasks, full_records, mapping):
    pairs = build_all_pairs(full_records, mapping)
    instances = build_instances(pairs, full_records)
    table = count_table(pairs, instances)
    assert table["sh_zh"] == {"n_pairs": 36, "n_instances": 576}
    assert table["tone"] == {"n_pairs": 2268, "n_instances": 9072}
    # 3 sub-experiments per pair-instance side
    assert len(pair_tasks) == (1984 + 768 + 9072) * 2 * 3
    by_exp = {}
    for task in pair_tasks:
        by_exp.setdefault(task.experiment, set()).add(task.task_type)
    assert by_exp == {"exp4": {"initial"}, "exp5": {"final"}, "exp6": {"tone"}}
    prompt_conditions = {}
    for task in pair_tasks:
        prompt_conditions.setdefault(task.experiment, set()).add(task.prompt_condition)
    assert all(v == {"free", "correct_target", "wrong_target"}
               for v in prompt_conditions.values())
    assert len({t.item_id for t in pair_tasks}) == len(pair_tasks)


def test_pair_free_sub_experiment_has_no_target(pair_tasks):
    free = [t for t in pair_tasks if t.prompt_condition == "free"]
    assert len(free) == (1984 + 768 + 9072) * 2
    for task in free[:300]:
        assert task.given_target is None
        assert task.given_initial is None and task.given_final is None
        assert task.given_tone is None


def test_pair_correct_info(pair_tasks):
    info = [t for t in pair_tasks if t.prompt_condition == "correct_target"]
    assert info
    for task in info[:500]:
        assert task.consistent_gold is True
        assert task.given_initial == task.gold_initial
        assert task.given_final == task.gold_final
        assert task.given_tone == task.gold_tone


def test_pair_wrong_info_changes_only_dimension(pair_tasks):
    seen = set()
    for task in pair_tasks:
        if task.prompt_condition != "wrong_target":
            continue
        seen.add(task.experiment)
        assert task.consistent_gold is False
        if task.experiment == "exp4":
            assert task.given_initial != task.gold_initial
            assert task.given_final == task.gold_final
            assert task.given_tone == task.gold_tone
        elif task.experiment == "exp5":
            assert task.given_initial == task.gold_initial
            assert task.given_final != task.gold_final
            assert task.given_tone == task.gold_tone
        else:
            assert task.given_initial == task.gold_initial
            assert task.given_final == task.gold_final
            assert task.given_tone != task.gold_tone
    assert seen == {"exp4", "exp5", "exp6"}


# ---------------------------------------------------------------------------
# Ablation (v3: detection does NOT participate)
# ---------------------------------------------------------------------------


def test_ablation_subsets_v3(core_tasks, detection_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    selection = T.sample_ablation(
        repo_cfg, core_tasks, pair_tasks, detection_tasks, detection_sets=detection,
    )
    assert len(selection["core"]) == 3 * 256
    assert "detect" not in selection  # v3: detection experiment is not ablated
    assert len(selection["pairs"]) > 0
    # core: 256 items per prompt condition, all voice|tone cells covered
    selected_core = {t.item_id: t for t in core_tasks if t.item_id in set(selection["core"])}
    counts = {}
    for task in selected_core.values():
        counts[task.prompt_condition] = counts.get(task.prompt_condition, 0) + 1
    assert counts == {"free": 256, "correct_target": 256, "wrong_target": 256}
    cells = {(t.voice, t.gold_tone) for t in selected_core.values()}
    assert cells == {(v, tone) for v in "ABCD" for tone in (1, 2, 3, 4)}
    # pairs: 7 pair types x selected instances x both sides x 3 sub-experiments
    pair_items = [t for t in pair_tasks if t.item_id in set(selection["pairs"])]
    assert len(pair_items) % 3 == 0


def test_ablation_deterministic(core_tasks, detection_tasks, pair_tasks, repo_cfg):
    detection = T.build_detection_sets(detection_tasks, repo_cfg)
    a = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_tasks,
                          detection_sets=detection)
    b = T.sample_ablation(repo_cfg, core_tasks, pair_tasks, detection_tasks,
                          detection_sets=detection)
    assert a["core"] == b["core"] and a["pairs"] == b["pairs"]


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
# Minimal (¥70) design subset
# ---------------------------------------------------------------------------


def test_minimal_subset_builder(core_tasks, detection_tasks, pair_tasks, full_records,
                                repo_cfg):
    cfg = repo_cfg.with_overrides({"minimal": {"n_targets": 40,
                                              "pair_instances_per_type": 2}})
    subset = T.build_minimal_subset(cfg, core_tasks, detection_tasks, pair_tasks,
                                    full_records)
    summary = subset["summary"]
    assert summary["n_targets"] == 40
    assert summary["n_audio"] == 40 * 16
    assert summary["n_core_tasks"] > 0
    assert summary["n_detection_tasks"] > 0
    assert summary["n_pair_tasks"] > 0
    assert summary["n_all_items"] == (
        summary["n_core_tasks"] + summary["n_detection_tasks"] + summary["n_pair_tasks"]
    )
    # v3: ablation subset excludes the detection experiment
    assert summary["n_no_detection_items"] == summary["n_core_tasks"] + summary["n_pair_tasks"]
    assert set(subset["no_detection_item_ids"]).isdisjoint(
        set(t.item_id for t in detection_tasks)
    )
    # pair instances capped; every pair type represented; 3 sub-experiments each
    assert set(subset["pair_instances"]) == set(T.PAIR_TYPE_TO_EXPERIMENT)
    assert all(v["selected"] <= 2 for v in subset["pair_instances"].values())
    pair_items = [t for t in pair_tasks if t.item_id in set(subset["pair_item_ids"])]
    assert len(pair_items) % 3 == 0
    # 40 targets round-robin over finals -> all 36 finals covered
    from pinyin_eval.normalize import load_mapping

    mapping = load_mapping("data/mapping/mandarin_tone_syn_target_mapping.csv")
    assert len({mapping[t].final for t in subset["targets"]}) == 36
    # deterministic
    again = T.build_minimal_subset(cfg, core_tasks, detection_tasks, pair_tasks,
                                   full_records)
    assert again["all_item_ids"] == subset["all_item_ids"]


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
    assert summary["by_experiment"]["exp2"] > summary["by_experiment"]["exp1"]
    det_summary = T.summarize_tasks(detection_tasks)
    assert det_summary["by_experiment"]["detect"] == len(detection_tasks)
