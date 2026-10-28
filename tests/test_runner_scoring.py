"""End-to-end (offline) tests: task build -> mock generation -> scoring -> metrics.

No API key, no real audio: the mock client answers from the gold labels in the
request metadata.
"""

from __future__ import annotations

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
    pairs = T.build_pair_tasks(tiny_records, small_cfg, mapping)
    return core, pairs


def _audio_subset(tasks, n_audios=2):
    keep = sorted({t.audio_id for t in tasks})[:n_audios]
    return set(keep), {t.item_id for t in tasks if t.audio_id in set(keep)}


def test_runner_dry_run(small_cfg, tiny_records, mapping):
    core, pairs = _tasks(small_cfg, tiny_records, mapping)
    result = run_generation(
        small_cfg, core + pairs, conditions=[small_cfg.main_condition],
        run_id="dry", dry_run=True,
    )
    run_dir = Path(result["run_dir"])
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "dry_run_plan.jsonl").exists()
    assert result["n_requests"] > 0


def test_mock_run_scoring_and_detection(small_cfg, tiny_records, mapping):
    core, pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=2)
    result = run_generation(
        small_cfg, core + pairs, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="mock-perfect", mock=True,
    )
    assert result["n_error"] == 0
    assert result["n_ok"] == result["n_requests"]

    out = scoring.score_run(small_cfg, "mock-perfect", tasks=core + pairs)
    items = out["items"]
    assert items["parse_status"].eq("ok").all()
    # perfect mock -> perfect recognition metrics
    free = items[(
        (items["prompt_condition"] == "free") & (items["experiment"] == "exp3")
    )].to_dict("records")
    rec = metrics.recognition_metrics(free)
    assert rec["tone"]["accuracy"] == 1.0
    assert rec["initial"]["accuracy"] == 1.0
    assert rec["final"]["accuracy"] == 1.0
    assert rec["joint_accuracy"] == 1.0

    # perfect mock -> perfect error detection (pred_consistent semantics)
    detection = out["detection"]
    assert len(detection) > 0
    main = detection[
        (detection["detection_subset"] == "main")
        & (detection["generation_condition"] == "think1__clean__json")
    ].to_dict("records")
    det = metrics.detection_metrics(main)
    assert det["tp"] > 0 and det["tn"] > 0
    assert det["fp"] == 0 and det["fn"] == 0
    assert det["error_precision"] == 1.0
    assert det["error_recall"] == 1.0
    assert det["fpr"] == 0.0 and det["fnr"] == 0.0

    # A/B tasks parsed and correct
    ab = items[items["prompt_condition"] == "ab_choice"].to_dict("records")
    if ab:
        assert metrics.ab_metrics(ab)["accuracy"] == 1.0

    # correctness columns are nullable booleans and scoped to the right task type
    assert set(items["initial_correct"].dropna().unique()) <= {True}
    assert items[items["prompt_condition"] != "free"]["initial_correct"].isna().all()


def test_resume_is_idempotent(small_cfg, tiny_records, mapping):
    core, pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=1)
    conditions = [small_cfg.main_condition, {"thinking": False, "audio_quality": "clean",
                                             "output_form": "json"}]
    first = run_generation(
        small_cfg, core + pairs, conditions=conditions, item_subset=subset,
        run_id="resume", mock=True,
    )
    n_first = first["n_ok"]
    second = run_generation(
        small_cfg, core + pairs, conditions=conditions, item_subset=subset,
        run_id="resume", mock=True,
    )
    assert second["n_skipped_resume"] == n_first
    assert second["n_ok"] == 0
    generations = load_generations(Path(second["run_dir"]))
    assert len(generations) == n_first


def test_request_dedupe_across_experiment_views(small_cfg, tiny_records, mapping):
    """exp3 free and exp4/5/6 free share one API call for the same audio."""
    core, pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=2)
    result = run_generation(
        small_cfg, core + pairs, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="dedupe", mock=True,
    )
    generations = load_generations(Path(result["run_dir"]))
    out = scoring.score_run(small_cfg, "dedupe", tasks=core + pairs)
    free_items = out["items"][out["items"]["prompt_condition"] == "free"]
    # more free item rows than free API calls => dedupe happened
    free_calls = [g for g in generations if g.prompt_condition == "free"]
    assert len(free_items) > len(free_calls)


def test_error_records_are_kept(small_cfg, tiny_records, mapping):
    core, _pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=1)
    cfg = small_cfg.with_overrides({"mock": {"mode": "perfect", "fail_rate": 1.0}})
    result = run_generation(
        cfg, core, conditions=[cfg.main_condition], item_subset=subset,
        run_id="failing", mock=True,
    )
    assert result["n_error"] == result["n_requests"]
    out = scoring.score_run(cfg, "failing", tasks=core)
    assert out["items"]["parse_status"].eq("api_error").all()
    assert out["summary"]["parse_status"].get("api_error", 0) > 0


def test_manifest_records_reproducibility_fields(small_cfg, tiny_records, mapping):
    core, _pairs = _tasks(small_cfg, tiny_records, mapping)
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


def test_scoring_filters_to_planned_calls(small_cfg, tiny_records, mapping):
    core, pairs = _tasks(small_cfg, tiny_records, mapping)
    keep, subset = _audio_subset(core, n_audios=1)
    run_generation(
        small_cfg, core + pairs, conditions=[small_cfg.main_condition],
        item_subset=subset, run_id="filter", mock=True,
    )
    out = scoring.score_run(small_cfg, "filter", tasks=core + pairs)
    # only planned cells are scored, not the whole task catalog
    n_possible = len(core + pairs) * len([small_cfg.main_condition])
    assert 0 < len(out["items"]) < n_possible
