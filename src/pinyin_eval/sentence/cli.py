"""CLI for the sentence-level pilot, dispatched by ``pinyin-eval sentence``.

Subcommands mirror the pipeline:

    sentence prepare    # build item manifests + download audio (delegates to scripts/)
    sentence check      # environment / data / plan sanity check
    sentence estimate   # planned calls and expected/worst-case cost
    sentence smoke      # one real call per task, inspect raw JSON and usage
    sentence generate   # run C1/C2 generations with resume + hard budget
    sentence score      # parse generations into per-item scoring rows
    sentence cost       # real usage/cost table of a run
    sentence report     # metrics report (exported by the report module)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ..client import ChatRequest, create_client
from ..config import Config
from ..pricing import extract_usage
from .records import CONDITIONS, TASKS, load_jsonl, make_request_id
from .runner import (
    build_plans,
    cost_summary,
    estimate_cost,
    load_items,
    new_run_id,
    run_dir_for,
    run_generation,
    sentence_manifest_dir,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pinyin-eval sentence",
        description="Qwen3.8-Omni whole-utterance pinyin / pronunciation experiment",
    )
    parser.add_argument("--config", default="configs/sentence.yaml", help="YAML config path")
    parser.add_argument("--root", default=None, help="repo root (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build item manifests and download audio")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--max-audio", type=int, default=None, help="debug: limit downloaded audio")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("check", help="print item / audio / model / plan status")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("estimate", help="planned API calls and expected / worst-case cost")
    _add_plan_args(p)
    p.add_argument("--worst-case", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("smoke", help="one real call per task to verify output format")
    p.add_argument("--task", default="A", choices=[*TASKS, "D"])
    p.add_argument("--condition", default="C1", choices=list(CONDITIONS))
    p.add_argument("--item-index", type=int, default=0)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("generate", help="run C1/C2 generations with resume + hard budget")
    _add_plan_args(p)
    p.add_argument("--run-id", default=None)
    p.add_argument("--mock", action="store_true", help="offline mock client (no API key)")
    p.add_argument("--limit", type=int, default=None, help="hard cap on calls this invocation")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--budget-cny", type=float, default=None)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("d-prepare", help="build task-D items (gold / ASR / injected text)")
    p.add_argument("--require-asr", action="store_true", default=True,
                   help="fail when ASR outputs are missing (default: on)")
    p.add_argument("--allow-missing-asr", dest="require_asr", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_d_prepare)

    p = sub.add_parser("d-score", help="parse + score a task-D run (alignment / localization / scoring)")
    p.add_argument("--run-id", required=True)
    p.set_defaults(func=cmd_d_score)

    p = sub.add_parser("d-report", help="task-D report (d_report.md + metrics CSVs + acceptance)")
    p.add_argument("--run-id", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_d_report)

    p = sub.add_parser("score", help="parse generations into per-item scoring rows")
    p.add_argument("--run-id", required=True)
    p.add_argument("--tasks", default="A,B,C")
    p.add_argument("--only-generated", action="store_true",
                   help="skip items without a generation (partial / pilot runs)")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("cost", help="real usage / cost table of a run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("report", help="metrics + tables + report.md for a scored run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    return parser


def _add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tasks", default="A,B,C", help="comma separated: A,B,C")
    parser.add_argument("--conditions", default="C1,C2", help="comma separated: C1,C2")
    parser.add_argument("--max-items-per-task", type=int, default=None)
    parser.add_argument("--limit-per-task", type=int, default=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(args: argparse.Namespace) -> Config:
    from ..config import load_dotenv

    root = Path(args.root) if args.root else Path.cwd()
    load_dotenv(root / ".env")
    return Config.load(args.config, repo_root=root)


def _parse_tasks(value: str) -> list[str]:
    tasks = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = [task for task in tasks if task not in (*TASKS, "D")]
    if unknown:
        raise SystemExit(f"unknown task(s): {unknown}; expected {TASKS}")
    return tasks


def _parse_conditions(value: str) -> list[str]:
    names = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}; expected {CONDITIONS}")
    return names


def _items_limited(
    cfg: Config, tasks: list[str], max_items_per_task: int | None = None
) -> dict[str, list]:
    items = load_items(cfg, tasks)
    if max_items_per_task:
        items = {task: rows[:max_items_per_task] for task, rows in items.items()}
    return items


def _plans_from_args(args: argparse.Namespace, cfg: Config) -> list:
    tasks = _parse_tasks(args.tasks)
    conditions = _parse_conditions(args.conditions)
    items = _items_limited(cfg, tasks, getattr(args, "max_items_per_task", None))
    plans = build_plans(cfg, items, tasks=tasks, condition_names=conditions)
    limit = getattr(args, "limit_per_task", None)
    if limit:
        kept = []
        per_task: Counter = Counter()
        for plan in plans:
            if per_task[plan.task] >= limit:
                continue
            per_task[plan.task] += 1
            kept.append(plan)
        plans = kept
    return plans


def _run_generations(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Index ``generations.jsonl`` by (task, condition, item_id)."""
    out: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(run_dir / "generations.jsonl"):
        key = f"{row.get('task')}|{row.get('condition')}|{row.get('item_id')}"
        out[key] = row
    return out


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_prepare(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    script = cfg.repo_root / "scripts" / "prepare_sentence_data.py"
    if not script.exists():
        print(f"prepare script missing: {script}", file=sys.stderr)
        return 1
    command = [sys.executable, str(script), "--root", str(cfg.repo_root)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.skip_download:
        command += ["--skip-download"]
    if args.max_audio is not None:
        command += ["--max-audio", str(args.max_audio)]
    if args.force:
        command += ["--force"]
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=str(cfg.repo_root))


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    report: dict[str, Any] = {"config": str(cfg.source_path), "model": cfg.model_name}
    try:
        items = load_items(cfg)
    except FileNotFoundError as exc:
        print(f"item manifests missing: {exc}; run `sentence prepare` first", file=sys.stderr)
        return 1
    tasks: dict[str, Any] = {}
    total_planned = 0
    check_tasks = list(TASKS)
    if (sentence_manifest_dir(cfg) / "d_items.jsonl").exists():
        check_tasks.append("D")
        items.update(load_items(cfg, ["D"]))
    for task in check_tasks:
        rows = items.get(task, [])
        missing_audio = sum(
            1 for item in rows if not (cfg.repo_root / item.audio_path).exists()
        )
        durations = [item.duration_s for item in rows if item.duration_s]
        tasks[task] = {
            "n_items": len(rows),
            "n_missing_audio": missing_audio,
            "duration_s_total": round(sum(durations), 1),
            "duration_s_median": round(sorted(durations)[len(durations) // 2], 2)
            if durations else None,
        }
        total_planned += len(rows) * (1 if task == "D" else len(CONDITIONS))
    report["tasks"] = tasks
    report["planned_calls"] = total_planned
    report["conditions"] = {
        name: {"thinking": name == "C2", **{
            "max_tokens": int((cfg.get(f"sentence.conditions.{name}") or {}).get(
                "max_tokens", 1024 if name == "C2" else 256))}}
        for name in CONDITIONS
    }
    report["prep_summary_present"] = (sentence_manifest_dir(cfg) / "prep_summary.json").exists()
    from .prompts import PROMPT_VERSION, prompt_fingerprints

    report["prompt_version"] = PROMPT_VERSION
    report["prompt_versions"] = {
        task: ("sentence-d-v1" if task == "D" else PROMPT_VERSION) for task in check_tasks
    }
    report["prompt_fingerprints"] = prompt_fingerprints()
    report["budget_cny"] = cfg.get("sentence.budget_cny")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    plans = _plans_from_args(args, cfg)
    expected = estimate_cost(cfg, plans, worst_case=False)
    worst = estimate_cost(cfg, plans, worst_case=True)
    payload = {
        "n_plans": len(plans),
        "expected": expected,
        "worst_case": worst,
        "budget_cny": cfg.get("sentence.budget_cny"),
        "tasks": _parse_tasks(args.tasks),
        "conditions": _parse_conditions(args.conditions),
    }
    out_dir = run_dir_for(cfg, "plan")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "estimate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"planned calls: {payload['n_plans']}")
    print(f"{'task|cond':12s} {'calls':>6s} {'audio_s':>9s} {'in_tok':>9s} {'out_tok':>9s} {'¥expected':>10s}")
    for key, entry in expected["per_task_condition"].items():
        print(f"{key:12s} {entry['n_calls']:6d} {entry['audio_seconds']:9.1f} "
              f"{entry['input_tokens']:9d} {entry['output_tokens']:9d} {entry['cost_cny']:10.4f}")
    print(f"{'TOTAL':12s} {expected['totals']['n_calls']:6d} "
          f"{expected['totals']['audio_seconds']:9.1f} {expected['totals']['input_tokens']:9d} "
          f"{expected['totals']['output_tokens']:9d} {expected['totals']['cost_cny']:10.4f}")
    print(f"worst-case cost: ¥{worst['totals']['cost_cny']:.4f} "
          f"(budget ¥{payload['budget_cny']})")
    print(f"estimate written: {out_dir / 'estimate.json'}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    task = args.task.upper()
    items = load_items(cfg, [task])[task]
    if not items:
        print("no items", file=sys.stderr)
        return 1
    item = items[min(args.item_index, len(items) - 1)]
    from .prompts import build_prompt, prompt_sha256, prompt_version_for_task

    system_prompt, user_prompt = build_prompt(task, item)
    def _spec(name: str) -> dict:
        cond = cfg.get(f"sentence.conditions.{name}") or {}
        return {
            "thinking": bool(cond.get("thinking", name == "C2")),
            "max_tokens": int((cond.get("max_tokens_by_task") or {}).get(
                task, cond.get("max_tokens", 1024 if name == "C2" else 256))),
        }

    spec = _spec(args.condition)
    client = create_client(cfg)
    request = ChatRequest(
        request_id=make_request_id(task, args.condition, item.item_id),
        audio_path=str(cfg.repo_root / item.audio_path),
        prompt_text=user_prompt,
        output_form="json",
        thinking=spec["thinking"],
        system_prompt=system_prompt,
        max_tokens=spec["max_tokens"],
    )
    response = client.generate(request)
    from .parsing import parse_generation

    parsed = parse_generation(task, {
        "response_text": response.text,
        "error": response.error,
    })
    usage = extract_usage(response.usage)
    payload = {
        "task": task,
        "condition": args.condition,
        "item_id": item.item_id,
        "prompt_version": prompt_version_for_task(task),
        "prompt_sha256": prompt_sha256(system_prompt, user_prompt,
                                       prompt_version_for_task(task)),
        "audio_path": item.audio_path,
        "duration_s": item.duration_s,
        "ok": response.ok,
        "error": response.error,
        "latency_s": round(response.latency_s, 2),
        "finish_reason": response.finish_reason,
        "usage": usage,
        "text": response.text,
        "reasoning_chars": len(response.reasoning or ""),
        "reasoning_head": (response.reasoning or "")[:300],
        "parsed": parsed,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if response.ok else 1


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    tasks = _parse_tasks(args.tasks)
    if "D" in tasks:
        conditions = _parse_conditions(args.conditions)
        if "C2" in conditions:
            print("[generate] task D is C1-only (design A.6); dropping C2", file=sys.stderr)
        args.conditions = "C1"
        if args.budget_cny is None:
            args.budget_cny = cfg.get("sentence.d.budget_cny", 10.0)
    plans = _plans_from_args(args, cfg)
    run_id = args.run_id or new_run_id()
    budget = args.budget_cny if args.budget_cny is not None else cfg.get("sentence.budget_cny")
    concurrency = args.concurrency or cfg.get("sentence.concurrency")
    summary = run_generation(
        cfg,
        plans,
        run_id,
        mock=args.mock,
        limit=args.limit,
        concurrency=int(concurrency) if concurrency else None,
        budget_cny=float(budget) if budget is not None else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["n_errors"] == 0 else 2


def cmd_cost(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    payload = cost_summary(cfg, args.run_id)
    run_dir = run_dir_for(cfg, args.run_id)
    import csv

    usage_rows = []
    for key, entry in payload["per_task_condition"].items():
        task, condition = key.split("|")
        row = {"task": task, "condition": condition, **{k: v for k, v in entry.items()
                                                       if k != "finish_reasons"}}
        for finish, count in entry["finish_reasons"].items():
            row[f"finish_{finish}"] = count
        usage_rows.append(row)
    if usage_rows:
        columns = sorted({key for row in usage_rows for key in row})
        with (run_dir / "usage.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(usage_rows)
    (run_dir / "cost_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"{'task|cond':12s} {'calls':>6s} {'errors':>6s} {'in_tok':>9s} {'out_tok':>9s} "
          f"{'reason':>8s} {'¥cost':>9s}")
    for key, entry in payload["per_task_condition"].items():
        print(f"{key:12s} {entry['n_calls']:6d} {entry['n_errors']:6d} "
              f"{entry['prompt_tokens']:9d} {entry['completion_tokens']:9d} "
              f"{entry['reasoning_tokens']:8d} {entry['cost_cny']:9.4f}")
    print(f"{'TOTAL':12s} {payload['totals']['n_calls']:6d} {payload['totals']['n_errors']:6d} "
          f"{payload['totals']['prompt_tokens']:9d} {payload['totals']['completion_tokens']:9d} "
          f"{payload['totals']['reasoning_tokens']:8d} {payload['totals']['cost_cny']:9.4f}")
    print(f"usage.csv written: {run_dir / 'usage.csv'}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    tasks = _parse_tasks(args.tasks)
    try:
        from . import scoring
    except ImportError as exc:  # pragma: no cover - module still landing
        print(f"scoring module unavailable: {exc}", file=sys.stderr)
        return 1
    run_dir = run_dir_for(cfg, args.run_id)
    if not (run_dir / "generations.jsonl").exists():
        print(f"no generations for run {args.run_id} in {run_dir}", file=sys.stderr)
        return 1
    from .parsing import parse_generation

    items_by_task = load_items(cfg, tasks)
    generations = _run_generations(run_dir)
    parsers = {"A": scoring.score_a, "B": scoring.score_b, "C": scoring.score_c}
    all_rows: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    failures: list[dict[str, Any]] = []
    status_counts: dict[str, Counter] = {task: Counter() for task in TASKS}
    for task in tasks:
        for item in items_by_task.get(task, []):
            for condition in CONDITIONS:
                key = f"{task}|{condition}|{item.item_id}"
                generation = generations.get(key)
                if generation is None:
                    if getattr(args, "only_generated", False):
                        continue
                    status = "api_error"
                    parsed = {"parse_status": "api_error", "detail": "missing generation"}
                    raw = ""
                    finish_reason = None
                else:
                    parsed = parse_generation(task, generation)
                    status = parsed["parse_status"]
                    raw = generation.get("response_text") or ""
                    finish_reason = generation.get("finish_reason")
                status_counts[task][status] += 1
                if status != "ok":
                    failures.append({
                        "task": task,
                        "item_id": item.item_id,
                        "condition": condition,
                        "parse_status": status,
                        "detail": parsed.get("detail", "")[:300],
                        "finish_reason": finish_reason,
                        "raw": raw[:500],
                    })
                # A/C always emit a scored row (zero counts on parse failure) so
                # that parse-failure rates are visible in the metric dicts; B
                # has no per-item row without verdicts by design.
                if task == "B":
                    if status == "ok":
                        all_rows["B"].extend(parsers["B"](item, parsed, condition))
                else:
                    all_rows[task].append(parsers[task](item, parsed, condition))
    import pandas as pd

    written: dict[str, str] = {}
    for task in ("A", "B", "C"):
        rows = all_rows[task]
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        if "condition" not in frame.columns:
            frame["condition"] = ""
        parquet_path = run_dir / f"rows_{task.lower()}.parquet"
        csv_path = run_dir / f"rows_{task.lower()}.csv"
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        written[task] = str(parquet_path)
    with (run_dir / "parse_failures.jsonl").open("w", encoding="utf-8") as fh:
        for row in failures:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "run_id": args.run_id,
        "generations": len(generations),
        "status_counts": {task: dict(counter) for task, counter in status_counts.items()},
        "n_rows": {task: len(rows) for task, rows in all_rows.items() if rows},
        "files": written,
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    try:
        from .report import build_report
    except ImportError as exc:  # pragma: no cover - module still landing
        print(f"report module unavailable: {exc}", file=sys.stderr)
        return 1
    payload = build_report(cfg, args.run_id)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload.get("summary", payload), ensure_ascii=False, indent=2))
    return 0


def cmd_d_prepare(args: argparse.Namespace) -> int:
    from . import d_task
    from .records import write_jsonl

    cfg = _cfg(args)
    items, summary = d_task.build_d_items(
        cfg, limit=args.limit, require_asr=bool(getattr(args, "require_asr", True))
    )
    manifest_dir = sentence_manifest_dir(cfg)
    write_jsonl(manifest_dir / "d_items.jsonl", items)
    injected = [
        {
            "utt_id": item.meta.get("utt_id"),
            "text_source": "D-inject-10",
            "gold_text": item.meta.get("gold_text"),
            "injected_text": item.text,
            "ops": item.meta.get("injection_ops") or [],
        }
        for item in items
        if item.meta.get("text_source") == "D-inject-10"
    ]
    write_jsonl(manifest_dir / "d_injected_10.jsonl", injected)
    (manifest_dir / "d_prep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_d_score(args: argparse.Namespace) -> int:
    from . import d_task

    cfg = _cfg(args)
    run_dir = run_dir_for(cfg, args.run_id)
    if not (run_dir / "generations.jsonl").exists():
        print(f"no generations for run {args.run_id} in {run_dir}", file=sys.stderr)
        return 1
    items = load_items(cfg, ["D"]).get("D", [])
    if not items:
        print("no task-D items; run `sentence d-prepare` first", file=sys.stderr)
        return 1
    generations = _run_generations(run_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    status_counts: Counter = Counter()
    for item in items:
        key = f"D|C1|{item.item_id}"
        generation = generations.get(key)
        missing_generation = generation is None
        if missing_generation:
            parsed = {"parse_status": "api_error", "detail": "missing generation"}
            parsed["alignment"] = []
            parsed["scores"] = {}
            raw = ""
            finish_reason = None
        elif generation.get("error"):
            # a final failed API call (retries exhausted) is an api_error, not an
            # empty reply — keep it out of the "empty response" bucket
            parsed = {
                "parse_status": "api_error",
                "detail": str(generation.get("error"))[:300],
                "alignment": [],
                "scores": {},
            }
            raw = generation.get("response_text") or ""
            finish_reason = generation.get("finish_reason")
        else:
            parsed = d_task.parse_task_d(generation.get("response_text") or "")
            raw = generation.get("response_text") or ""
            finish_reason = generation.get("finish_reason")
        status = parsed["parse_status"]
        status_counts[status] += 1
        if status != "ok":
            failures.append({
                "item_id": item.item_id,
                "source": item.meta.get("text_source"),
                "parse_status": status,
                "missing_generation": missing_generation,
                "detail": parsed.get("detail", "")[:300],
                "finish_reason": finish_reason,
                "raw": raw[:500],
            })
        rows.append(d_task.score_d(item, parsed, "C1"))
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / "rows_d.parquet", index=False)
    frame.to_csv(run_dir / "rows_d.csv", index=False, encoding="utf-8")
    with (run_dir / "d_parse_failures.jsonl").open("w", encoding="utf-8") as fh:
        for failure in failures:
            fh.write(json.dumps(failure, ensure_ascii=False) + "\n")
    prep_summary = sentence_manifest_dir(cfg) / "d_prep_summary.json"
    if prep_summary.exists():
        (run_dir / "d_prep_summary.json").write_text(
            prep_summary.read_text(encoding="utf-8"), encoding="utf-8"
        )
    summary = {
        "run_id": args.run_id,
        "generations": len(generations),
        "n_items": len(items),
        "status_counts": dict(status_counts),
        "n_parse_failures": sum(1 for f in failures if f["parse_status"] not in ("api_error",)),
        "n_api_errors": sum(1 for f in failures if f["parse_status"] == "api_error"),
        "n_missing_generations": sum(1 for f in failures if f.get("missing_generation")),
        "rows": len(rows),
    }
    (run_dir / "d_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_d_report(args: argparse.Namespace) -> int:
    from . import d_task

    cfg = _cfg(args)
    payload = d_task.build_d_report(cfg, args.run_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if (payload.get("acceptance") or {}).get("passed") is not False else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
