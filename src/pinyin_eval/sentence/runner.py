"""Generation runner for the sentence-level experiment (design §5, §9, §12).

One :class:`RequestPlan` = one item x one task x one condition.  The runner

* resumes from ``generations.jsonl`` (append-only, de-duplicated by
  ``request_id``; failed calls are retried on the next invocation);
* caps the cost with ``sentence.budget_cny`` (stops scheduling new requests);
* records token usage / finish reason / retries / latency per call;
* writes a reproducible ``run_manifest.json`` before the first API call.

The actual HTTP client is :class:`pinyin_eval.client.DashScopeClient`; the mock
client is used by ``--mock`` for offline end-to-end tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..client import BaseClient, ChatRequest, create_client
from ..config import Config
from ..pricing import Pricing, extract_usage
from .prompts import (
    D_PROMPT_VERSION,
    PROMPT_VERSION,
    build_prompt,
    prompt_fingerprints,
    prompt_sha256,
    prompt_version_for_task,
)
from .records import (
    GenerationRecord,
    RequestPlan,
    SentenceItem,
    append_jsonl,
    load_jsonl,
    make_request_id,
)

__all__ = [
    "build_plans",
    "build_manifest",
    "cost_summary",
    "estimate_cost",
    "load_items",
    "new_run_id",
    "run_dir_for",
    "run_generation",
]


# ---------------------------------------------------------------------------
# Paths / items
# ---------------------------------------------------------------------------


def results_dir(cfg: Config) -> Path:
    return cfg.path("sentence.paths.results_dir", "results/sentence")


def run_dir_for(cfg: Config, run_id: str) -> Path:
    return results_dir(cfg) / run_id


def new_run_id(prefix: str = "sentence") -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


#: manifest file per task (produced by ``scripts/prepare_sentence_data.py``)
ITEM_MANIFESTS: dict[str, str] = {
    "A": "aishell3_items.jsonl",
    "B": "ompal_b_items.jsonl",
    "C": "ompal_c_items.jsonl",
    "D": "d_items.jsonl",
}

#: native (L1) utterances used only as task-C human-ceiling anchors (design §3.2)
NATIVE_MANIFEST = "ompal_native_items.jsonl"


def sentence_manifest_dir(cfg: Config) -> Path:
    return cfg.path("sentence.paths.manifest_dir", "data/sentence/manifests")


def load_items(cfg: Config, tasks: Iterable[str] = ("A", "B", "C")) -> dict[str, list[SentenceItem]]:
    """Load the prepared item manifests, one list per task."""
    base = sentence_manifest_dir(cfg)
    include_native = bool(cfg.get("sentence.include_native_anchor", True))
    out: dict[str, list[SentenceItem]] = {}
    for task in tasks:
        task = task.upper()
        path = base / ITEM_MANIFESTS[task]
        rows = load_jsonl(path)
        items = [SentenceItem.from_dict(row) for row in rows]
        if task == "C" and include_native and (base / NATIVE_MANIFEST).exists():
            native_rows = load_jsonl(base / NATIVE_MANIFEST)
            items.extend(SentenceItem.from_dict(row) for row in native_rows)
        out[task] = items
    return out


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


def conditions(cfg: Config) -> dict[str, dict[str, Any]]:
    """Condition table (design §5): ``C1`` = think off, ``C2`` = think on.

    ``max_tokens_by_task`` allows the pilot calibration of the output cap per
    task (task A's per-character JSON is much longer than task C's 3 numbers);
    the design's flat 256/1024 values remain the default and are overridden
    only with recorded evidence in ``run_manifest.json``.
    """
    configured = cfg.get("sentence.conditions") or {}
    out: dict[str, dict[str, Any]] = {}
    for name in ("C1", "C2"):
        spec = configured.get(name) or {}
        out[name] = {
            "thinking": bool(spec.get("thinking", name == "C2")),
            "max_tokens": int(spec.get("max_tokens", 1024 if name == "C2" else 256)),
            "max_tokens_by_task": dict(spec.get("max_tokens_by_task") or {}),
            "output_form": "json",
        }
    return out


def build_plans(
    cfg: Config,
    items: dict[str, list[SentenceItem]],
    tasks: Iterable[str] = ("A", "B", "C"),
    condition_names: Iterable[str] = ("C1", "C2"),
) -> list[RequestPlan]:
    """Build the full request plan list in a deterministic order."""
    table = conditions(cfg)
    plans: list[RequestPlan] = []
    for task in tasks:
        task = task.upper()
        for condition in condition_names:
            spec = table[condition]
            max_tokens = int(spec["max_tokens_by_task"].get(task, spec["max_tokens"]))
            task_prompt_version = prompt_version_for_task(task)
            for item in items.get(task, []):
                system_prompt, user_prompt = build_prompt(task, item)
                plans.append(
                    RequestPlan(
                        request_id=make_request_id(task, condition, item.item_id),
                        item_id=item.item_id,
                        task=task,
                        condition=condition,
                        audio_path=item.audio_path,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        prompt_version=task_prompt_version,
                        prompt_sha256=prompt_sha256(
                            system_prompt, user_prompt, task_prompt_version
                        ),
                        thinking=spec["thinking"],
                        max_tokens=max_tokens,
                        output_form="json",
                        meta={
                            "duration_s": float(item.duration_s or 0.0),
                            "speaker": item.speaker,
                            "prompt_chars": len(system_prompt) + len(user_prompt),
                            # gold for the deterministic mock client (never sent)
                            "sentence_task": task,
                            "gold_chars": item.chars,
                            "gold_pinyin": list(item.pinyin_surface),
                            "gold_labels": item.labels,
                            "gold_scores": item.scores,
                            "gold_text": item.text,
                            "gold_alignment": item.meta.get("gold_alignment") if item.meta else None,
                            "text_source": (item.meta or {}).get("text_source"),
                        },
                    )
                )
    return plans


# ---------------------------------------------------------------------------
# Manifest / fingerprints
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None


def build_manifest(
    cfg: Config,
    *,
    run_id: str,
    plans: list[RequestPlan],
    mock: bool = False,
    budget_cny: float | None = None,
) -> dict[str, Any]:
    """Reproducibility manifest (design §12)."""
    repo_root = cfg.repo_root
    prep_path = sentence_manifest_dir(cfg) / "prep_summary.json"
    prep_summary: dict[str, Any] = {}
    if prep_path.exists():
        try:
            prep_summary = json.loads(prep_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - defensive
            prep_summary = {}
    manifests = {
        task: {
            "file": str(sentence_manifest_dir(cfg) / name),
            "sha256": _sha256_file(sentence_manifest_dir(cfg) / name),
        }
        for task, name in ITEM_MANIFESTS.items()
    }
    pricing = Pricing.from_config(cfg)
    tasks = sorted({plan.task for plan in plans})
    prompt_version = D_PROMPT_VERSION if tasks == ["D"] else PROMPT_VERSION
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mock": bool(mock),
        "model": cfg.model_name,
        "provider": str(cfg.get("model.provider", "dashscope_openai")),
        "base_url": (os.environ.get(str(cfg.get("model.base_url_env", "DASHSCOPE_BASE_URL")))
                     or str(cfg.get("model.base_url_default", ""))),
        "prompt_version": prompt_version,
        "prompt_fingerprints": prompt_fingerprints(),
        "config_path": str(cfg.source_path) if cfg.source_path else None,
        "config_sha256": cfg.config_sha256(),
        "seed": cfg.seed(),
        "pricing": pricing.as_dict(),
        "budget_cny": budget_cny,
        "conditions": conditions(cfg),
        "n_planned": len(plans),
        "tasks": tasks,
        "items_manifests": manifests,
        "prep_summary": {
            key: prep_summary.get(key)
            for key in ("seed", "created_at", "aishell3", "ompal", "deviations", "sources")
            if key in prep_summary
        },
        "git_commit": _git_commit(repo_root),
        "operator": str(cfg.get("project.operator", "")),
        "environment": {
            "python": str(os.sys.version.split()[0]),
            "proxy_required": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")),
        },
    }


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(
    cfg: Config,
    plans: list[RequestPlan],
    *,
    worst_case: bool = False,
) -> dict[str, Any]:
    """Pre-run cost estimate from prompt length, audio duration and output caps."""
    pricing = Pricing.from_config(cfg)
    assumed = cfg.get("sentence.estimate.assumed_output_tokens") or {
        "C1": 300, "C2": 900, "C": 30,
    }
    table = conditions(cfg)
    per: dict[str, dict[str, Any]] = {}
    for plan in plans:
        key = f"{plan.task}|{plan.condition}"
        entry = per.setdefault(key, {
            "n_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0,
            "audio_seconds": 0.0,
        })
        duration = float(plan.meta.get("duration_s", 0.0))
        prompt_tokens = pricing.estimate_prompt_tokens(
            plan.system_prompt, plan.user_prompt, duration, mode="conservative"
        )
        if worst_case:
            output_tokens = plan.max_tokens
        else:
            default_assumed = assumed.get(plan.task) or assumed.get(plan.condition) or 300
            output_tokens = int(default_assumed)
            if plan.task == "C":
                output_tokens = min(output_tokens, int(assumed.get("C", 30)))
        entry["n_calls"] += 1
        entry["input_tokens"] += prompt_tokens
        entry["output_tokens"] += output_tokens
        entry["audio_seconds"] += duration
        entry["cost_cny"] += pricing.call_cost(prompt_tokens, output_tokens)["total_cost_cny"]
    totals = {
        "n_calls": sum(v["n_calls"] for v in per.values()),
        "input_tokens": sum(v["input_tokens"] for v in per.values()),
        "output_tokens": sum(v["output_tokens"] for v in per.values()),
        "audio_seconds": round(sum(v["audio_seconds"] for v in per.values()), 1),
        "cost_cny": round(sum(v["cost_cny"] for v in per.values()), 4),
    }
    for entry in per.values():
        entry["cost_cny"] = round(entry["cost_cny"], 4)
    return {"per_task_condition": per, "totals": totals,
            "mode": "worst_case" if worst_case else "expected", "table": table}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class _Budget:
    """Thread-safe cost accumulator with a hard cap."""

    def __init__(self, budget_cny: float | None, spent: float = 0.0):
        self.budget = budget_cny
        self.spent = spent
        self.lock = threading.Lock()
        self.exceeded = False

    def add(self, cost: float) -> None:
        with self.lock:
            self.spent += cost

    def allow(self) -> bool:
        with self.lock:
            if self.budget is None:
                return True
            if self.spent >= self.budget:
                self.exceeded = True
            return not self.exceeded


def _record_cost(record: GenerationRecord, pricing: Pricing) -> float:
    usage = extract_usage(record.usage)
    return pricing.call_cost(
        usage["prompt_tokens"], usage["completion_tokens"], usage["cached_tokens"]
    )["total_cost_cny"]


def run_generation(
    cfg: Config,
    plans: list[RequestPlan],
    run_id: str,
    *,
    mock: bool = False,
    limit: int | None = None,
    concurrency: int | None = None,
    budget_cny: float | None = None,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Execute (or resume) all planned requests and return a run summary."""
    pricing = Pricing.from_config(cfg)
    run_dir = run_dir_for(cfg, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    generations_path = run_dir / "generations.jsonl"

    manifest_path = run_dir / "run_manifest.json"
    existing = load_jsonl(generations_path)
    if not manifest_path.exists():
        manifest = build_manifest(cfg, run_id=run_id, plans=plans, mock=mock,
                                  budget_cny=budget_cny)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    completed: set[str] = set()
    spent = 0.0
    for row in existing:
        try:
            record = GenerationRecord.from_dict(row)
        except TypeError:  # pragma: no cover - defensive
            continue
        spent += _record_cost(record, pricing)
        if record.error is None and record.response_text.strip():
            completed.add(record.request_id)
    pending = [plan for plan in plans if plan.request_id not in completed]
    if limit is not None:
        pending = pending[:limit]

    budget = _Budget(budget_cny, spent=spent)
    client: BaseClient
    if mock:
        from .mock import SentenceMockClient

        client = SentenceMockClient(cfg)
    else:
        client = create_client(cfg, mock=False)
    workers = int(concurrency or cfg.get("sentence.concurrency", 4))
    write_lock = threading.Lock()
    counters = {"called": 0, "ok": 0, "errors": 0, "retries": 0, "skipped_budget": 0}
    started = time.time()

    def execute(plan: RequestPlan) -> GenerationRecord | None:
        if not budget.allow():
            counters["skipped_budget"] += 1
            return None
        request = ChatRequest(
            request_id=plan.request_id,
            audio_path=str(cfg.repo_root / plan.audio_path)
            if not Path(plan.audio_path).is_absolute() else plan.audio_path,
            prompt_text=plan.user_prompt,
            output_form=plan.output_form,
            thinking=plan.thinking,
            system_prompt=plan.system_prompt,
            max_tokens=plan.max_tokens,
            meta=plan.meta,
        )
        response = client.generate(request)
        record = GenerationRecord(
            run_id=run_id,
            request_id=plan.request_id,
            item_id=plan.item_id,
            task=plan.task,
            condition=plan.condition,
            thinking=plan.thinking,
            max_tokens=plan.max_tokens,
            model=response.model or cfg.model_name,
            prompt_version=plan.prompt_version,
            prompt_sha256=plan.prompt_sha256,
            audio_path=plan.audio_path,
            response_text=response.text,
            reasoning_text=response.reasoning,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_s=round(response.latency_s, 3),
            error=response.error,
            retries=response.retries,
            timestamp=response.timestamp,
            http_request_id=response.request_log_id,
        )
        cost = _record_cost(record, pricing)
        budget.add(cost)
        with write_lock:
            append_jsonl(generations_path, [record])
            counters["called"] += 1
            counters["ok"] += 1 if record.ok else 0
            counters["errors"] += 0 if record.ok else 1
            counters["retries"] += record.retries
            if counters["called"] % progress_every == 0 or counters["called"] == len(pending):
                print(
                    f"[generate] {counters['called']}/{len(pending)} calls "
                    f"ok={counters['ok']} err={counters['errors']} "
                    f"cost=¥{budget.spent:.4f} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
        return record

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(execute, plan) for plan in pending]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - never lose a run to one call
                    print(f"[generate] worker error: {exc}", flush=True)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_planned": len(plans),
        "n_already_completed": len(completed),
        "n_pending": len(pending),
        "n_called": counters["called"],
        "n_ok": counters["ok"],
        "n_errors": counters["errors"],
        "n_skipped_budget": counters["skipped_budget"],
        "budget_cny": budget_cny,
        "spent_cny": round(budget.spent, 6),
        "budget_exceeded": budget.exceeded,
        "elapsed_s": round(time.time() - started, 1),
        "mock": mock,
        "concurrency": workers,
    }
    (run_dir / "generate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------------
# Usage / cost accounting
# ---------------------------------------------------------------------------


def cost_summary(cfg: Config, run_id: str) -> dict[str, Any]:
    """Aggregate the recorded usage of a run (real numbers, not estimates)."""
    pricing = Pricing.from_config(cfg)
    run_dir = run_dir_for(cfg, run_id)
    rows = load_jsonl(run_dir / "generations.jsonl")
    per: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = GenerationRecord.from_dict(row)
        key = f"{record.task}|{record.condition}"
        entry = per.setdefault(key, {
            "n_calls": 0, "n_errors": 0, "n_retries": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "cost_cny": 0.0, "latency_s": 0.0, "finish_reasons": {},
        })
        usage = extract_usage(record.usage)
        cost = _record_cost(record, pricing)
        entry["n_calls"] += 1
        entry["n_errors"] += 0 if record.ok else 1
        entry["n_retries"] += record.retries
        entry["prompt_tokens"] += usage["prompt_tokens"]
        entry["completion_tokens"] += usage["completion_tokens"]
        entry["reasoning_tokens"] += usage["reasoning_tokens"]
        entry["cost_cny"] += cost
        entry["latency_s"] += record.latency_s
        finish = str(record.finish_reason)
        entry["finish_reasons"][finish] = entry["finish_reasons"].get(finish, 0) + 1
    totals = {
        "n_calls": sum(v["n_calls"] for v in per.values()),
        "n_errors": sum(v["n_errors"] for v in per.values()),
        "prompt_tokens": sum(v["prompt_tokens"] for v in per.values()),
        "completion_tokens": sum(v["completion_tokens"] for v in per.values()),
        "reasoning_tokens": sum(v["reasoning_tokens"] for v in per.values()),
        "cost_cny": round(sum(v["cost_cny"] for v in per.values()), 6),
        "avg_latency_s": None,
    }
    for entry in per.values():
        entry["cost_cny"] = round(entry["cost_cny"], 6)
        entry["avg_latency_s"] = (
            round(entry["latency_s"] / entry["n_calls"], 3) if entry["n_calls"] else None
        )
    if totals["n_calls"]:
        totals["avg_latency_s"] = round(
            sum(v["latency_s"] for v in per.values()) / totals["n_calls"], 3
        )
    return {"per_task_condition": per, "totals": totals}
