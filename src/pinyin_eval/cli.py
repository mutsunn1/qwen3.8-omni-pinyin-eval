"""Command line interface.

Typical workflow::

    uv run pinyin-eval prepare-data          # download + export audio + manifest
    uv run pinyin-eval build-tasks           # task catalogs, detection sets, ablation subsets
    uv run pinyin-eval estimate              # count API calls before spending money
    uv run pinyin-eval generate --tasks core --conditions main
    uv run pinyin-eval score --run-id <id>
    uv run pinyin-eval report --run-id <id>

`all-mock` runs the whole pipeline offline on a small subsample (no API key).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Config, ConfigError, default_config_path, load_dotenv

log = logging.getLogger("pinyin_eval")

#: design-document minimal-pair counts (docs/experiment-design.md, table 最小对计数)
EXPECTED_PAIR_COUNTS: dict[str, dict[str, int]] = {
    "sh_zh": {"n_pairs": 36, "n_instances": 576},
    "aspiration": {"n_pairs": 67, "n_instances": 1072},
    "nasal_lateral": {"n_pairs": 21, "n_instances": 336},
    "front_back_nasal_ptr": {"n_pairs": 40, "n_instances": 640},
    "i_ü": {"n_pairs": 5, "n_instances": 80},
    "ian_üan": {"n_pairs": 3, "n_instances": 48},
    "tone": {"n_pairs": 2268, "n_instances": 9072},
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinyin-eval",
        description="Qwen3.8-Omni Mandarin pinyin recognition & consistency benchmark",
    )
    parser.add_argument("--config", default=default_config_path(),
                        help="YAML config path (default: $PINYIN_EVAL_CONFIG or %(default)s)")
    parser.add_argument("--root", default=None, help="repo root (default: current directory)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare-data -------------------------------------------------------
    p = sub.add_parser("prepare-data", help="download the dataset, export audio, build the audio manifest")
    p.add_argument("--force", action="store_true", help="re-download / rebuild everything")
    p.add_argument("--skip-download", action="store_true", help="use the existing parquet file")
    p.set_defaults(func=cmd_prepare_data)

    # build-tasks --------------------------------------------------------
    p = sub.add_parser("build-tasks", help="build task catalogs, detection sets and ablation subsets")
    p.add_argument("--skip-pairs", action="store_true", help="only build experiments 1-3 tasks")
    p.add_argument("--validate", action="store_true", help="validate minimal-pair counts against the design doc")
    p.add_argument("--minimal", action="store_true",
                   help="also build the fixed task subset for the minimal (¥70) design")
    p.add_argument("--include-wide-nasal", action="store_true",
                   help="also pair ian/iang and uan/uang (52 pairs; the design doc table counts 40)")
    p.set_defaults(func=cmd_build_tasks)

    # validate-minimal-pairs --------------------------------------------
    p = sub.add_parser("validate-minimal-pairs", help="print the minimal-pair count table and validate it")
    p.add_argument("--include-wide-nasal", action="store_true")
    p.set_defaults(func=cmd_validate_pairs)

    # estimate -----------------------------------------------------------
    p = sub.add_parser("estimate", help="count API calls / audio seconds for a run plan")
    p.add_argument("--tasks", default="all", choices=["core", "pairs", "detect", "all"])
    p.add_argument("--profile", default="main", choices=["main", "ablation", "minimal", "all"])
    p.add_argument("--conditions", default="main", choices=["main", "all", "minimal"])
    p.add_argument("--item-ids", default=None, help="explicit item-id list file")
    p.add_argument("--condition", action="append", default=None, metavar="THINK:QUALITY:FORM")
    p.set_defaults(func=cmd_estimate)

    # generate -----------------------------------------------------------
    p = sub.add_parser("generate", help="run model generations with resume support")
    p.add_argument("--tasks", default="all", choices=["core", "pairs", "detect", "all"])
    p.add_argument("--profile", default="main", choices=["main", "ablation", "minimal"],
                   help="main = full task catalog; ablation = 256-item subsets; minimal = ¥70 subset")
    p.add_argument("--item-ids", default=None, help="explicit item-id list file (overrides --profile)")
    p.add_argument("--conditions", default="main", choices=["main", "all", "minimal"],
                   help="main = config main_condition; all = 2x2x2 grid; minimal = config minimal.conditions")
    p.add_argument("--condition", action="append", default=None,
                   metavar="THINK:QUALITY:FORM",
                   help="extra/override generation condition, e.g. think0:clean:json")
    p.add_argument("--limit", type=int, default=None,
                   help="pilot mode: at most N requests per generation condition (seeded random sample)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--dry-run", action="store_true", help="write the request plan, do not call the API")
    p.add_argument("--mock", action="store_true", help="use the deterministic mock client")
    p.add_argument("--no-resume", action="store_true", help="ignore existing generations of the same run id")
    p.add_argument("--max-cost", type=float, default=None,
                   help="hard CNY budget: stop issuing new API calls when reached")
    p.set_defaults(func=cmd_generate)

    # score --------------------------------------------------------------
    p = sub.add_parser("score", help="parse generations into item results and detection rows")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_score)

    # cost ---------------------------------------------------------------
    p = sub.add_parser("cost", help="compute actual token usage and CNY cost of a run")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_cost)

    # report -------------------------------------------------------------
    p = sub.add_parser("report", help="write the markdown report and metric tables")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--n-boot", type=int, default=None)
    p.set_defaults(func=cmd_report)

    # all-mock -----------------------------------------------------------
    p = sub.add_parser("all-mock", help="offline end-to-end run on a small subsample (mock client)")
    p.add_argument("--audios", type=int, default=40, help="how many audios to include")
    p.add_argument("--conditions", default="main", choices=["main", "all"])
    p.add_argument("--run-id", default=None)
    p.set_defaults(func=cmd_all_mock)

    # check --------------------------------------------------------------
    p = sub.add_parser("check", help="print environment / dataset / model setup status")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_check)

    # smoke --------------------------------------------------------------
    p = sub.add_parser("smoke", help="one real API call to verify credentials and model id")
    p.add_argument("--audio", default=None, help="wav path (default: first clean audio)")
    p.set_defaults(func=cmd_smoke)

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(args: argparse.Namespace) -> Config:
    try:
        cfg = Config.load(args.config, repo_root=args.root)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    return cfg


def _task_dir(cfg: Config) -> Path:
    return cfg.path("paths.task_dir")


def _load_task_lists(cfg: Config, which: str):
    from .tasks import load_tasks

    tdir = _task_dir(cfg)
    lists = []
    if which in ("core", "all"):
        path = tdir / "core.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing; run `pinyin-eval build-tasks` first")
        lists.extend(load_tasks(path))
    if which in ("pairs", "all"):
        path = tdir / "pairs.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing; run `pinyin-eval build-tasks` first")
        lists.extend(load_tasks(path))
    if which in ("detect", "all"):
        path = tdir / "detection.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing; run `pinyin-eval build-tasks` first")
        lists.extend(load_tasks(path))
    return lists


def _repo_relative(cfg: Config, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(cfg.repo_root))
    except ValueError:
        return str(path)


def _task_sources(cfg: Config, which: str) -> list[str]:
    sources: list[str] = []
    tdir = _task_dir(cfg)
    if which in ("core", "all"):
        sources.append(_repo_relative(cfg, tdir / "core.jsonl"))
    if which in ("pairs", "all"):
        sources.append(_repo_relative(cfg, tdir / "pairs.jsonl"))
    if which in ("detect", "all"):
        sources.append(_repo_relative(cfg, tdir / "detection.jsonl"))
    return sources


def _load_minimal_subsets(cfg: Config, task_ids: set[str]) -> tuple[set[str], dict[str, set[str]]]:
    """Load the minimal design item subsets (v3: detection only under main)."""
    from .tasks import load_item_ids

    tdir = _task_dir(cfg)
    all_path = tdir / "minimal_items.txt"
    nodet_path = tdir / "minimal_no_detection_items.txt"
    if not all_path.exists() or not nodet_path.exists():
        raise SystemExit(
            f"minimal subsets missing ({all_path.name}/{nodet_path.name}); run "
            "`pinyin-eval --config configs/minimal.yaml build-tasks --minimal`"
        )
    all_ids = set(load_item_ids(all_path)) & task_ids
    nodet_ids = set(load_item_ids(nodet_path)) & task_ids
    by_generation: dict[str, set[str]] = {}
    for cond in cfg.get("minimal.conditions") or []:
        role = str(cond.get("role", ""))
        key = (
            f"{'think1' if cond['thinking'] else 'think0'}"
            f"__{cond['audio_quality']}__{cond['output_form']}"
        )
        by_generation[key] = all_ids if role == "main" else nodet_ids
    return all_ids, by_generation


def _conditions_from_args(cfg: Config, args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if args.conditions == "all":
        conditions.extend(cfg.all_conditions())
    elif args.conditions == "minimal":
        for cond in cfg.get("minimal.conditions") or []:
            conditions.append({
                "thinking": bool(cond["thinking"]),
                "audio_quality": str(cond["audio_quality"]),
                "output_form": str(cond["output_form"]),
            })
        if not conditions:
            raise SystemExit("config has no minimal.conditions (use configs/minimal.yaml)")
    else:
        conditions.append(cfg.main_condition)
    for spec in getattr(args, "condition", None) or []:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--condition must look like think0:clean:json, got {spec!r}")
        thinking_raw, quality, form = parts
        thinking = thinking_raw.lower() in ("1", "true", "on", "yes", "think1")
        cond = {"thinking": thinking, "audio_quality": quality, "output_form": form}
        if cond not in conditions:
            conditions.append(cond)
    # validate
    for cond in conditions:
        if cond["audio_quality"] not in ("clean", "noise_20db"):
            raise SystemExit(f"bad audio_quality {cond['audio_quality']!r}")
        if cond["output_form"] not in ("json", "text"):
            raise SystemExit(f"bad output_form {cond['output_form']!r}")
    return conditions


def _load_item_subset(cfg: Config, args: argparse.Namespace, tasks: Sequence[Any]) -> tuple[set[str] | None, str | None]:
    from .tasks import load_item_ids

    if getattr(args, "item_ids", None):
        path = Path(args.item_ids)
        ids = set(load_item_ids(path))
        return ids, str(path)
    profile = getattr(args, "profile", "main")
    if profile == "minimal":
        path = _task_dir(cfg) / "minimal_items.txt"
        if not path.exists():
            raise SystemExit(
                f"{path} missing; run `pinyin-eval --config configs/minimal.yaml build-tasks --minimal`"
            )
        return set(load_item_ids(path)), _repo_relative(cfg, path)
    if profile != "ablation":
        return None, None
    task_ids = {t.item_id for t in tasks}
    selected: set[str] = set()
    sources: list[str] = []
    for name in ("ablation_core_items.txt", "ablation_detection_items.txt",
                 "ablation_pairs_items.txt"):
        path = _task_dir(cfg) / name
        if path.exists():
            ids = set(load_item_ids(path)) & task_ids
            selected.update(ids)
            sources.append(str(path))
    if not selected:
        raise SystemExit("no ablation subset files found; run `pinyin-eval build-tasks` first")
    return selected, ",".join(sources)


def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_prepare_data(args: argparse.Namespace) -> int:
    from .data.dataset import prepare_data

    cfg = load_config(args)
    df = prepare_data(cfg, force=args.force, skip_download=args.skip_download)
    from .data.dataset import dataset_summary

    print(json.dumps(dataset_summary(df), ensure_ascii=False, indent=2))
    print(f"\naudio manifest: {cfg.path('paths.manifest_dir') / 'audio.parquet'}")
    return 0


def cmd_build_tasks(args: argparse.Namespace) -> int:

    from .data.dataset import load_audio_records
    from .data.minimal_pairs import build_all_pairs, build_instances, count_table
    from .normalize import load_mapping
    from .tasks import (
        build_core_tasks,
        build_detection_sets,
        build_detection_tasks,
        build_pair_tasks,
        sample_ablation,
        summarize_tasks,
        write_item_ids,
        write_tasks,
    )

    cfg = load_config(args)
    records = load_audio_records(cfg)
    mapping = load_mapping(cfg.path("paths.mapping_csv"))
    tdir = _task_dir(cfg)
    tdir.mkdir(parents=True, exist_ok=True)

    core = build_core_tasks(records, cfg, mapping)
    write_tasks(core, tdir / "core.jsonl")
    detect = build_detection_tasks(records, cfg, mapping)
    write_tasks(detect, tdir / "detection.jsonl")
    detection = build_detection_sets(detect, cfg)
    detection.to_csv(tdir / "detection_sets.csv", index=False)
    print("core tasks:", json.dumps(summarize_tasks(core), ensure_ascii=False, indent=1))
    print("detection tasks:", json.dumps(summarize_tasks(detect), ensure_ascii=False, indent=1))
    print("detection sets:", detection.groupby("subset").size().to_dict())

    pairs = []
    if not args.skip_pairs:
        wide = bool(args.include_wide_nasal or cfg.get("tasks.minimal_pairs.include_wide_nasal", False))
        pair_objs = build_all_pairs(records, mapping, include_wide_nasal=wide)
        instances = build_instances(pair_objs, records)
        table = count_table(pair_objs, instances)
        with (tdir / "minimal_pair_counts.json").open("w", encoding="utf-8") as fh:
            json.dump(table, fh, ensure_ascii=False, indent=2)
        pairs = build_pair_tasks(records, cfg, mapping)
        write_tasks(pairs, tdir / "pairs.jsonl")
        print("minimal pairs:", json.dumps(table, ensure_ascii=False, indent=1))
        print("pair tasks:", json.dumps(summarize_tasks(pairs), ensure_ascii=False, indent=1))
        if args.validate:
            problems = {k: (table.get(k), v) for k, v in EXPECTED_PAIR_COUNTS.items() if table.get(k) != v}
            if problems:
                print("VALIDATION FAILED:", json.dumps(problems, ensure_ascii=False))
                return 1
            print("minimal-pair counts match the design document ✓")

    selection = sample_ablation(cfg, core, pairs, detect, detection_sets=detection)
    write_item_ids(selection["core"], tdir / "ablation_core_items.txt")
    if pairs:
        write_item_ids(selection["pairs"], tdir / "ablation_pairs_items.txt")
        print("ablation pair instances:", json.dumps(selection["pair_instance_summary"], ensure_ascii=False))
    print(f"ablation items: core={len(selection['core'])} pairs={len(selection['pairs'])}")
    print("v3 ablation: detection experiment does not participate in the ablation grid")

    if getattr(args, "minimal", False):
        from .tasks import build_minimal_subset

        minimal = build_minimal_subset(cfg, core, detect, pairs, records)
        write_item_ids(minimal["core_item_ids"], tdir / "minimal_core_items.txt")
        write_item_ids(minimal["detection_item_ids"], tdir / "minimal_detection_items.txt")
        write_item_ids(minimal["pair_item_ids"], tdir / "minimal_pair_items.txt")
        write_item_ids(minimal["all_item_ids"], tdir / "minimal_items.txt")
        write_item_ids(minimal["no_detection_item_ids"],
                       tdir / "minimal_no_detection_items.txt")
        with (tdir / "minimal_summary.json").open("w", encoding="utf-8") as fh:
            json.dump(minimal, fh, ensure_ascii=False, indent=2)
        print("minimal subset:", json.dumps(minimal["summary"], ensure_ascii=False, indent=1))
        print("minimal files: minimal_items.txt / minimal_no_detection_items.txt / "
              "minimal_core_items.txt / minimal_detection_items.txt / "
              "minimal_pair_items.txt / minimal_summary.json")
    print("task dir:", tdir)
    return 0


def cmd_validate_pairs(args: argparse.Namespace) -> int:
    from .data.dataset import load_audio_records
    from .data.minimal_pairs import build_all_pairs, build_instances, count_table
    from .normalize import load_mapping

    cfg = load_config(args)
    records = load_audio_records(cfg)
    mapping = load_mapping(cfg.path("paths.mapping_csv"))
    wide = bool(args.include_wide_nasal or cfg.get("tasks.minimal_pairs.include_wide_nasal", False))
    pairs = build_all_pairs(records, mapping, include_wide_nasal=wide)
    instances = build_instances(pairs, records)
    table = count_table(pairs, instances)
    print(json.dumps(table, ensure_ascii=False, indent=2))
    if wide:
        print("\ninclude_wide_nasal=True: 与设计文档计数表不可直接比较（设计表按 40 对口径）")
        return 0
    problems = {k: (table.get(k), v) for k, v in EXPECTED_PAIR_COUNTS.items() if table.get(k) != v}
    if problems:
        print("\nVALIDATION FAILED:", json.dumps(problems, ensure_ascii=False))
        return 1
    print("\n计数与设计文档一致 ✓")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from .runner import (
        build_request_plan,
        estimate_plan_cost,
        filter_plans_by_generation,
        plan_summary,
    )

    cfg = load_config(args)
    tasks = _load_task_lists(cfg, args.tasks)
    conditions = _conditions_from_args(cfg, args)
    subset, subset_source = _load_item_subset(cfg, args, tasks)
    gen_subsets = None
    if getattr(args, "profile", "main") == "minimal":
        subset, gen_subsets = _load_minimal_subsets(cfg, {t.item_id for t in tasks})
        tdir = _task_dir(cfg)
        subset_source = ",".join(
            _repo_relative(cfg, tdir / name)
            for name in ("minimal_items.txt", "minimal_no_detection_items.txt")
        )
    plans = build_request_plan(cfg, tasks, conditions, item_subset=subset)
    plans = filter_plans_by_generation(plans, gen_subsets)
    summary = plan_summary(plans)
    cost = estimate_plan_cost(cfg, plans)
    payload = {
        "tasks": args.tasks,
        "profile": args.profile,
        "item_subset_source": subset_source,
        "conditions": conditions,
        **summary,
        "cost_estimate": cost,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from .runner import run_generation

    cfg = load_config(args)
    tasks = _load_task_lists(cfg, args.tasks)
    conditions = _conditions_from_args(cfg, args)
    subset, subset_source = _load_item_subset(cfg, args, tasks)
    gen_subsets = None
    if getattr(args, "profile", "main") == "minimal":
        subset, gen_subsets = _load_minimal_subsets(cfg, {t.item_id for t in tasks})
        tdir = _task_dir(cfg)
        subset_source = ",".join(
            _repo_relative(cfg, tdir / name)
            for name in ("minimal_items.txt", "minimal_no_detection_items.txt")
        )
    task_sources = _task_sources(cfg, args.tasks)
    result = run_generation(
        cfg,
        tasks,
        conditions=conditions,
        item_subset=subset,
        item_subset_source=subset_source,
        run_id=args.run_id,
        tag=args.tag,
        limit=args.limit,
        dry_run=args.dry_run,
        mock=args.mock,
        task_sources=task_sources,
        max_cost_cny=args.max_cost,
        item_subset_by_generation=gen_subsets,
        extra_manifest={"cli": {"tasks": args.tasks, "profile": args.profile,
                                "conditions_arg": args.conditions}},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from .scoring import score_run

    cfg = load_config(args)
    out = score_run(cfg, args.run_id)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    from .pricing import Pricing, budget_status, summarize_generations
    from .runner import load_generations, read_manifest

    cfg = load_config(args)
    run_dir = cfg.path("paths.results_dir") / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    manifest = read_manifest(run_dir)
    pricing = Pricing.from_config(cfg)
    summary = summarize_generations(load_generations(run_dir), pricing)
    budget = budget_status(summary["overall"]["cost_cny"], manifest.get("budget_cny"))
    payload = {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "budget": budget,
        "overall": summary["overall"],
        "by_condition": summary["by_condition"],
        "by_generation": summary["by_generation"],
        "pricing": summary["pricing"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    try:
        from .report import write_report
    except ImportError as exc:  # pragma: no cover - depends on module availability
        raise SystemExit(f"report module not available yet: {exc}") from exc
    artifacts = write_report(cfg, args.run_id, out_dir=args.out_dir, n_boot=args.n_boot)
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    return 0


def cmd_all_mock(args: argparse.Namespace) -> int:
    from .data.dataset import load_audio_records
    from .runner import run_generation
    from .scoring import score_run
    from .tasks import load_tasks

    cfg = load_config(args)
    tdir = _task_dir(cfg)
    if not all((tdir / name).exists() for name in ("core.jsonl", "pairs.jsonl", "detection.jsonl")):
        print("task catalogs missing; running build-tasks first ...")
        if cmd_build_tasks(args) != 0:
            return 1
    records = load_audio_records(cfg)
    keep = {r.audio_id for r in records[: args.audios]}
    core = load_tasks(tdir / "core.jsonl")
    pairs = load_tasks(tdir / "pairs.jsonl")
    detect = load_tasks(tdir / "detection.jsonl")
    all_tasks = core + pairs + detect
    subset = {t.item_id for t in all_tasks if t.audio_id in keep}
    conditions = cfg.all_conditions() if args.conditions == "all" else [cfg.main_condition]
    run_id = args.run_id or None
    result = run_generation(
        cfg, all_tasks, conditions=conditions, item_subset=subset,
        run_id=run_id, mock=True, tag="mock",
        task_sources=_task_sources(cfg, "all"),
    )
    run_id = result["run_id"]
    scored = score_run(cfg, run_id, tasks=all_tasks)
    print(json.dumps(scored["summary"], ensure_ascii=False, indent=2))
    try:
        from .report import write_report

        artifacts = write_report(cfg, run_id)
        print(json.dumps(artifacts, ensure_ascii=False, indent=2))
    except ImportError:
        print("report module not available; skipped report generation")
    print(f"\nrun dir: {result['run_dir']}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from .client import check_setup
    from .data.dataset import load_audio_records
    from .tasks import load_tasks

    cfg = load_config(args)
    report: dict[str, Any] = {"config": {
        "path": str(cfg.source_path),
        "sha256": cfg.config_sha256(),
        "repo_root": str(cfg.repo_root),
    }}
    env_file = cfg.repo_root / ".env"
    report["environment"] = {
        "python": sys.version.split()[0],
        "env_file": str(env_file),
        "env_file_exists": env_file.exists(),
        "config_env_var": os.environ.get("PINYIN_EVAL_CONFIG"),
        "dotenv_loaded_keys": sorted(
            key for key in ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "PINYIN_EVAL_CONFIG")
            if os.environ.get(key)
        ),
    }
    report["model"] = check_setup(cfg)
    report["dataset"] = {
        "raw_parquet": str(cfg.path("paths.raw_parquet")),
        "raw_parquet_exists": cfg.path("paths.raw_parquet").exists(),
        "manifest_exists": (cfg.path("paths.manifest_dir") / "audio.parquet").exists(),
        "mapping_csv": str(cfg.path("paths.mapping_csv")),
        "mapping_csv_exists": cfg.path("paths.mapping_csv").exists(),
    }
    tdir = _task_dir(cfg)
    counts: dict[str, Any] = {}
    for name in ("core.jsonl", "pairs.jsonl", "detection.jsonl", "detection_sets.csv",
                 "ablation_core_items.txt", "ablation_pairs_items.txt",
                 "minimal_items.txt", "minimal_no_detection_items.txt",
                 "minimal_summary.json"):
        path = tdir / name
        counts[name] = path.stat().st_size if path.exists() else None
    report["tasks"] = counts
    try:
        report["n_audio"] = len(load_audio_records(cfg))
        report["n_core_tasks"] = len(load_tasks(tdir / "core.jsonl"))
        report["n_pair_tasks"] = len(load_tasks(tdir / "pairs.jsonl"))
        report["n_detection_tasks"] = len(load_tasks(tdir / "detection.jsonl"))
    except Exception as exc:  # noqa: BLE001 - status command should not crash
        report["data_error"] = str(exc)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["dataset"]["raw_parquet_exists"] else 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["model"]["live_calls_possible"]:
        print("\n注意: 未配置 DASHSCOPE_API_KEY，只能使用 --mock 运行。")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from .client import smoke_call
    from .data.dataset import load_audio_records

    cfg = load_config(args)
    if args.audio:
        audio_path = args.audio
    else:
        records = load_audio_records(cfg)
        audio_path = str(cfg.repo_root / records[0].wav_path)
    result = smoke_call(cfg, audio_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    # .env in the current directory is loaded before parsing so that
    # PINYIN_EVAL_CONFIG / DASHSCOPE_* defaults are available everywhere.
    load_dotenv(Path.cwd() / ".env")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except SystemExit as exc:  # argparse --help / early exits keep their code
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        log.exception("command failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
