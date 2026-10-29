"""CLI tests on a tiny synthetic dataset (offline, mock client)."""

from __future__ import annotations

import json

import pytest
import yaml

from pinyin_eval.cli import build_parser, main


@pytest.fixture()
def cli_args(tiny_parquet_cfg, tmp_path):
    """Write the merged tiny config to disk so the CLI sees the tmp paths."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(tiny_parquet_cfg.as_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return ["--config", str(config_path), "--root", str(tiny_parquet_cfg.repo_root)]


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["check"])
    assert args.command == "check"
    args = parser.parse_args(["generate", "--tasks", "core", "--mock", "--limit", "5"])
    assert args.tasks == "core" and args.mock and args.limit == 5


def test_check_json(cli_args):
    assert main(cli_args + ["check", "--json"]) == 0


def test_cli_pipeline(cli_args, tiny_parquet_cfg):
    assert main(cli_args + ["prepare-data", "--skip-download"]) == 0
    assert main(cli_args + ["build-tasks"]) == 0
    task_dir = tiny_parquet_cfg.path("paths.task_dir")
    for name in ("core.jsonl", "detection.jsonl", "pairs.jsonl", "detection_sets.csv",
                 "ablation_core_items.txt", "ablation_pairs_items.txt"):
        assert (task_dir / name).exists(), name
    assert not (task_dir / "ablation_detection_items.txt").exists()  # v3: no detection ablation

    assert main(cli_args + [
        "generate", "--tasks", "all", "--mock", "--run-id", "cli-run",
        "--conditions", "main",
    ]) == 0
    assert main(cli_args + ["score", "--run-id", "cli-run"]) == 0
    run_dir = tiny_parquet_cfg.path("paths.results_dir") / "runs" / "cli-run"
    assert (run_dir / "items.parquet").exists()
    assert (run_dir / "detection_rows.parquet").exists()
    assert main(cli_args + ["cost", "--run-id", "cli-run"]) == 0
    summary = json.loads((run_dir / "score_summary.json").read_text())
    assert summary["n_items"] > 0
    assert summary["parse_status"].get("ok", 0) == summary["n_items"]
    assert summary["per_experiment"].get("detect", {}).get("n", 0) > 0


def test_estimate_and_dry_run(cli_args, tiny_parquet_cfg, capsys):
    assert main(cli_args + ["prepare-data", "--skip-download"]) == 0
    assert main(cli_args + ["build-tasks", "--skip-pairs"]) == 0
    capsys.readouterr()  # drop build-tasks output
    assert main(cli_args + ["estimate", "--tasks", "core", "--profile", "main"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert payload["cost_estimate"]["n_requests"] == payload["n_requests"] > 0
    assert payload["cost_estimate"]["conservative_cost_cny"] > 0
    assert main(cli_args + [
        "generate", "--tasks", "core", "--dry-run", "--run-id", "dry",
    ]) == 0
    run_dir = tiny_parquet_cfg.path("paths.results_dir") / "runs" / "dry"
    plan = run_dir / "dry_run_plan.jsonl"
    assert plan.exists() and plan.stat().st_size > 0
    lines = plan.read_text().strip().splitlines()
    assert len(lines) > 0


def test_minimal_build(cli_args, tiny_parquet_cfg):
    assert main(cli_args + ["prepare-data", "--skip-download"]) == 0
    assert main(cli_args + ["build-tasks", "--minimal"]) == 0
    task_dir = tiny_parquet_cfg.path("paths.task_dir")
    for name in ("minimal_items.txt", "minimal_no_detection_items.txt",
                 "minimal_summary.json", "minimal_core_items.txt",
                 "minimal_detection_items.txt", "minimal_pair_items.txt"):
        assert (task_dir / name).exists(), name
    summary = json.loads((task_dir / "minimal_summary.json").read_text())["summary"]
    assert summary["n_all_items"] > 0
    assert summary["n_all_items"] == (
        summary["n_core_tasks"] + summary["n_detection_tasks"] + summary["n_pair_tasks"]
    )
    assert summary["n_no_detection_items"] == (
        summary["n_core_tasks"] + summary["n_pair_tasks"]
    )


def test_generate_without_tasks_fails(cli_args, tiny_parquet_cfg):
    # no build-tasks yet in this tmp dir -> friendly failure exit code
    assert main(cli_args + ["generate", "--tasks", "core", "--mock"]) == 1


def test_report_after_pipeline(cli_args, tiny_parquet_cfg):
    pytest.importorskip("pinyin_eval.report")
    assert main(cli_args + ["prepare-data", "--skip-download"]) == 0
    assert main(cli_args + ["build-tasks"]) == 0
    assert main(cli_args + [
        "generate", "--tasks", "all", "--mock", "--run-id", "cli-report",
    ]) == 0
    assert main(cli_args + ["score", "--run-id", "cli-report"]) == 0
    assert main(cli_args + ["report", "--run-id", "cli-report"]) == 0
    run_dir = tiny_parquet_cfg.path("paths.results_dir") / "runs" / "cli-report"
    assert (run_dir / "report.md").exists()
