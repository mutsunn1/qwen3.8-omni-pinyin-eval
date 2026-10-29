"""End-to-end (offline) tests: task build -> mock generation -> scoring -> metrics.

No API key, no real audio: the mock client answers from the gold labels in the
request metadata.  Updated for the 2026-10-28 design (tone-only exp1/2 +
dedicated detection experiment).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pinyin_eval import metrics, scoring
from pinyin_eval import tasks as T
from pinyin_eval.config import Config
from pinyin_eval.runner import load_generations, read_manifest, run_generation

from .conftest import CONFIG_PATH, MAPPING_CSV


@pytest.fixture()
def small_cfg(tmp_path):
    overrides = {
        "paths": {
            "raw_parquet": str(tmp_path / "raw" / "data.parquet"),
            "mapping_csv": str(MAPPING_CSV),
            "audio_dir": str(tmp_path / "audio"),
            "noisy_dir": str(tmp_path / "noisy"),
            "manifest_dir": str(tmp_path / "manifests"),
            "task_dir": str(tmp_path / "manifests" / "tasks"),
            "results_dir": str(tmp_path / "results"),
        },
        "model": {"concurrency": 2, "qps": None},
        "mock": {"mode": "perfect", "seed": 7},
    }
    return Config.load(CONFIG_PATH, repo_root=tmp_path, overrides=overrides)


def _tasks(small_cfg, tiny_records, mapping):
    core = T.build_core_tasks(tiny_records, small_cfg, mapping)
    detect = T.build_detection_tasks(tiny_records, small_cfg, mapping)
    pairs = T.build_pair_tasks(tiny_records, small_cfg, mapping)
    return core, detect, pairs


def _audio_subset(tasks, n_audios=2):
    keep = sorted({t.audio_id for t in tasks})[:n_audios]
    keep_set = set(keep)
    return keep_set, {t.item_id for t in tasks if t.audio_id in keep_set}


def test_runner_dry_run(small_cfg, tiny_records, mapping):
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    result = run_generation(
        small_cfg, core + pairs + detect, conditions=[small_cfg.main_condition],
        run_id="dry", dry_run=True,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "dry_run_plan.jsonl").exists()
    assert result["n_requests"] > 0


def test_mock_run_scoring_and_detection(small_cfg, tiny_records, mapping):
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    all_tasks = core + detect + pairs
    keep, subset = _audio_subset(all_tasks, n_audios=2)
    result = run_generation(
        small_cfg, all_tasks, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="mock-perfect", mock=True,
    )
    assert result["n_error"] == 0
    assert result["n_ok"] == result["n_requests"]

    out = scoring.score_run(small_cfg, "mock-perfect", tasks=all_tasks)
    items = out["items"]
    assert items["parse_status"].eq("ok").all()
    assert set(items["task_type"]) == {"free", "tone", "detect", "ab"}

    # perfect mock -> perfect free recognition (experiment 3 view)
    free = items[items["task_type"] == "free"].to_dict("records")
    rec = metrics.recognition_metrics(free)
    assert rec["tone"]["accuracy"] == 1.0
    assert rec["initial"]["accuracy"] == 1.0
    assert rec["joint_accuracy"] == 1.0

    # experiments 1/2: tone-only accuracy
    tone = items[items["task_type"] == "tone"].to_dict("records")
    assert tone
    tone_metrics = metrics.classification_metrics(
        [int(r["gold_tone"]) for r in tone], [int(r["pred_tone"]) for r in tone],
    )
    assert tone_metrics["accuracy"] == 1.0

    # detection experiment: perfect consistency judgement
    assert len(out["detection"]) > 0
    main = out["detection"][
        (out["detection"]["detection_subset"] == "main")
        & (out["detection"]["generation_condition"] == "think1__clean__json")
    ].to_dict("records")
    det = metrics.detection_metrics(main)
    assert det["tp"] > 0 and det["tn"] > 0
    assert det["fp"] == 0 and det["fn"] == 0
    assert det["error_precision"] == 1.0
    assert det["error_recall"] == 1.0
    assert det["fpr"] == 0.0 and det["fnr"] == 0.0

    # A/B tasks parsed and correct
    ab = items[items["task_type"] == "ab"].to_dict("records")
    assert ab and metrics.ab_metrics(ab)["accuracy"] == 1.0

    # correctness columns are scoped to the right task types
    assert items.loc[items["task_type"] == "free", "initial_correct"].notna().all()
    assert items.loc[items["task_type"] != "free", "initial_correct"].isna().all()
    assert items.loc[items["task_type"].isin(["free", "tone"]), "tone_correct"].notna().all()
    assert items.loc[items["task_type"] == "detect", "consistent_correct"].notna().all()
    assert items.loc[items["task_type"] != "detect", "consistent_correct"].isna().all()
    assert items.loc[items["task_type"] == "ab", "choice_correct"].notna().all()


def test_limit_is_seeded_per_condition_sample(small_cfg, tiny_records, mapping):
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    all_tasks = core + detect + pairs
    conditions = [small_cfg.main_condition]
    first = run_generation(
        small_cfg, all_tasks, conditions=conditions, run_id="limit-a",
        dry_run=True, limit=3,
    )
    second = run_generation(
        small_cfg, all_tasks, conditions=conditions, run_id="limit-a",
        dry_run=True, limit=3,
    )
    assert first["n_requests"] == second["n_requests"]
    # at most 3 requests per condition id and the plan is deterministic
    plan_a = Path(first["run_dir"]) / "dry_run_plan.jsonl"
    plan_b = Path(second["run_dir"]) / "dry_run_plan.jsonl"
    assert plan_a.read_text() == plan_b.read_text()
    lines = [json.loads(line) for line in plan_a.read_text().splitlines() if line.strip()]
    per_condition: dict[str, int] = {}
    for row in lines:
        per_condition[row["condition_id"]] = per_condition.get(row["condition_id"], 0) + 1
    assert all(count <= 3 for count in per_condition.values())
    # a pilot sample should not be confined to a single prompt view
    assert len({row["prompt_condition"] for row in lines}) >= 2


def test_resume_is_idempotent(small_cfg, tiny_records, mapping):
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    all_tasks = core + detect + pairs
    keep, subset = _audio_subset(all_tasks, n_audios=1)
    conditions = [small_cfg.main_condition, {"thinking": False, "audio_quality": "clean",
                                             "output_form": "json"}]
    first = run_generation(
        small_cfg, all_tasks, conditions=conditions, item_subset=subset,
        run_id="resume", mock=True,
    )
    n_first = first["n_ok"]
    second = run_generation(
        small_cfg, all_tasks, conditions=conditions, item_subset=subset,
        run_id="resume", mock=True,
    )
    assert second["n_skipped_resume"] == n_first
    assert second["n_ok"] == 0
    generations = load_generations(Path(second["run_dir"]))
    assert len(generations) == n_first


def test_request_dedupe_across_experiment_views(small_cfg, tiny_records, mapping):
    """exp3 free and exp4/5/6 free share one API call for the same audio."""
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    all_tasks = core + detect + pairs
    keep, subset = _audio_subset(all_tasks, n_audios=2)
    result = run_generation(
        small_cfg, all_tasks, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="dedupe", mock=True,
    )
    generations = load_generations(Path(result["run_dir"]))
    out = scoring.score_run(small_cfg, "dedupe", tasks=all_tasks)
    free_items = out["items"][out["items"]["task_type"] == "free"]
    free_calls = [g for g in generations if g.prompt_condition == "free"]
    assert len(free_items) > len(free_calls)


def test_error_records_are_kept(small_cfg, tiny_records, mapping):
    core, _detect, _pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=1)
    cfg = small_cfg.with_overrides({"mock": {"mode": "perfect", "fail_rate": 1.0}})
    result = run_generation(
        cfg, core, conditions=[cfg.main_condition], item_subset=subset,
        run_id="failing", mock=True,
    )
    assert result["n_error"] == result["n_requests"]
    out = scoring.score_run(cfg, "failing", tasks=core)
    assert out["items"]["parse_status"].eq("api_error").all()


def test_manifest_records_reproducibility_fields(small_cfg, tiny_records, mapping):
    core, _detect, _pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=1)
    run_generation(
        small_cfg, core, conditions=[small_cfg.main_condition], item_subset=subset,
        run_id="manifest", mock=True, task_sources=["core.jsonl"],
    )
    manifest = read_manifest(small_cfg.path("paths.results_dir") / "runs" / "manifest")
    assert manifest["dataset"]["hf_revision"]
    assert manifest["dataset"]["mapping_csv_sha256"]
    assert manifest["prompts"]["version"]
    assert manifest["model"]["name"] == small_cfg.model_name
    assert manifest["config"]["generation"]["main_condition"]["thinking"] is True
    assert manifest["counts"]["n_planned"] > 0


def test_item_subset_accepts_comma_separated_files(small_cfg, tiny_records, mapping):
    core, _detect, _pairs = _tasks(small_cfg, tiny_records, mapping)
    tdir = small_cfg.path("paths.task_dir")
    tdir.mkdir(parents=True, exist_ok=True)
    ids_a = [t.item_id for t in core[:3]]
    ids_b = [t.item_id for t in core[3:5]]
    (tdir / "a.txt").write_text("\n".join(ids_a))
    (tdir / "b.txt").write_text("\n".join(ids_b))
    manifest = {"item_subset_source": "manifests/tasks/a.txt,manifests/tasks/b.txt"}
    subset = scoring.load_run_item_subset(small_cfg, manifest)
    assert subset == set(ids_a) | set(ids_b)


def test_scoring_filters_to_planned_calls(small_cfg, tiny_records, mapping):
    core, detect, pairs = _tasks(small_cfg, tiny_records, mapping)
    all_tasks = core + detect + pairs
    keep, subset = _audio_subset(all_tasks, n_audios=1)
    run_generation(
        small_cfg, all_tasks, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="filter", mock=True,
    )
    out = scoring.score_run(small_cfg, "filter", tasks=all_tasks)
    n_possible = len(all_tasks) * len([small_cfg.main_condition])
    assert 0 < len(out["items"]) < n_possible
