"""Generation runner: expand task items x conditions, de-duplicate requests,
resume from existing JSONL, call the model with bounded concurrency, and write a
run manifest for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import BaseClient, ChatRequest, ChatResponse, check_setup, create_client
from .config import Config
from .data.dataset import ensure_noisy_for
from .pricing import Pricing, budget_status, extract_usage
from .prompts import PROMPT_VERSION, describe_prompts
from .records import GenerationRecord, TaskItem
from .tasks import (
    condition_id_for,
    media_rel_for,
    request_id_for_task,
)

log = logging.getLogger(__name__)

MANIFEST_NAME = "run_manifest.json"
GENERATIONS_NAME = "generations.jsonl"
PLAN_NAME = "dry_run_plan.jsonl"
PLANNED_CALLS_NAME = "planned_calls.jsonl"


class RunnerError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Request planning
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RequestPlan:
    request_id: str
    condition_id: str
    thinking: bool
    audio_quality: str
    output_form: str
    prompt_condition: str
    experiment: str
    item_id: str
    audio_id: str
    media_rel: str
    media_path: str
    prompt_text: str
    task_type: str
    system_prompt: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "condition_id": self.condition_id,
            "thinking": self.thinking,
            "audio_quality": self.audio_quality,
            "output_form": self.output_form,
            "prompt_condition": self.prompt_condition,
            "experiment": self.experiment,
            "item_id": self.item_id,
            "audio_id": self.audio_id,
            "media_path": self.media_rel,
            "prompt_sha": hashlib.sha256(self.prompt_text.encode()).hexdigest()[:16],
        }


def _task_meta(task: TaskItem) -> dict[str, Any]:
    return {
        "task_type": task.task_type,
        "gold_initial": task.gold_initial,
        "gold_final": task.gold_final,
        "gold_tone": task.gold_tone,
        "given_target": task.given_target,
        "given_tone": task.given_tone,
        "consistent_gold": task.consistent_gold,
        "correct_choice": task.correct_choice,
        "alternatives": task.alternatives,
    }


def build_request_plan(
    cfg: Config,
    tasks: Sequence[TaskItem],
    conditions: Sequence[dict[str, Any]],
    *,
    item_subset: set[str] | None = None,
) -> list[RequestPlan]:
    """Expand and de-duplicate all requests implied by tasks x conditions."""
    plans: dict[tuple[str, str], RequestPlan] = {}
    for task in tasks:
        if item_subset is not None and task.item_id not in item_subset:
            continue
        if not task.prompt_texts:
            raise RunnerError(f"task {task.item_id} has no prompt_texts; rebuild tasks")
        for cond in conditions:
            thinking = bool(cond["thinking"])
            quality = str(cond["audio_quality"])
            form = str(cond["output_form"])
            cid = condition_id_for(
                task.prompt_condition, thinking=thinking,
                audio_quality=quality, output_form=form,
            )
            rid = request_id_for_task(
                cfg, task, output_form=form, audio_quality=quality
            )
            key = (rid, cid)
            if key in plans:
                continue
            media_rel = media_rel_for(cfg, task, quality)
            media_path = str((cfg.repo_root / media_rel).resolve()) if not Path(media_rel).is_absolute() else media_rel
            plans[key] = RequestPlan(
                request_id=rid,
                condition_id=cid,
                thinking=thinking,
                audio_quality=quality,
                output_form=form,
                prompt_condition=task.prompt_condition,
                experiment=task.experiment,
                item_id=task.item_id,
                audio_id=task.audio_id,
                media_rel=media_rel,
                media_path=media_path,
                prompt_text=task.prompt_texts[form],
                task_type=task.task_type,
                system_prompt=task.system_prompt,
                meta=_task_meta(task),
            )
    return sorted(plans.values(), key=lambda p: (p.condition_id, p.experiment, p.item_id, p.request_id))


def filter_plans_by_generation(
    plans: Sequence[RequestPlan],
    subset_by_generation: Mapping[str, set[str]] | None,
) -> list[RequestPlan]:
    """Keep only plans whose item is allowed for their generation condition.

    ``subset_by_generation`` maps a generation key (``think0__clean__json``) to
    the set of allowed item ids.  Conditions missing from the mapping are not
    filtered.  Used by the minimal design where the detection experiment runs
    only under the main condition (v3: detection does not participate in
    ablation).
    """
    if not subset_by_generation:
        return list(plans)
    kept: list[RequestPlan] = []
    for plan in plans:
        generation = plan.condition_id.split("__", 1)[1] if "__" in plan.condition_id else ""
        allowed = subset_by_generation.get(generation)
        if allowed is None or plan.item_id in allowed:
            kept.append(plan)
    return kept


def plan_summary(plans: Sequence[RequestPlan]) -> dict[str, Any]:
    per_condition: dict[str, int] = {}
    for plan in plans:
        per_condition[plan.condition_id] = per_condition.get(plan.condition_id, 0) + 1
    return {
        "n_requests": len(plans),
        "n_unique_prompts": len({p.prompt_text for p in plans}),
        "n_unique_audio": len({p.audio_id for p in plans}),
        "per_condition": dict(sorted(per_condition.items())),
    }


# ---------------------------------------------------------------------------
# Run ids / manifest
# ---------------------------------------------------------------------------


def _model_slug(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in model).strip("-")


def new_run_id(cfg: Config, *, tag: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    return f"{stamp}-{_model_slug(cfg.model_name)}{suffix}"


def run_dir_for(cfg: Config, run_id: str) -> Path:
    return cfg.path("paths.results_dir") / "runs" / run_id


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    cfg: Config,
    *,
    run_id: str,
    plans: Sequence[RequestPlan],
    task_sources: Sequence[str],
    item_subset_source: str | None,
    conditions: Sequence[dict[str, Any]],
    mock: bool,
    extra: dict[str, Any] | None = None,
    budget_cny: float | None = None,
) -> dict[str, Any]:
    mapping_path = cfg.path("paths.mapping_csv")
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "running",
        "git_commit": _git_commit(cfg.repo_root),
        "config_path": str(cfg.source_path) if cfg.source_path else None,
        "config_sha256": cfg.config_sha256(),
        "config": cfg.as_dict(),
        "dataset": {
            "hf_repo": cfg["dataset.hf_repo"],
            "hf_revision": cfg["dataset.hf_revision"],
            "config_name": cfg["dataset.config_name"],
            "mapping_csv_sha256": _sha256_file(mapping_path) if mapping_path.exists() else None,
        },
        "pricing": Pricing.from_config(cfg).as_dict(),
        "budget_cny": budget_cny,
        "model": {
            "name": cfg.model_name,
            "provider": cfg.get("model.provider"),
            "mock": mock,
            "temperature": cfg.get("model.temperature"),
            "top_p": cfg.get("model.top_p"),
            "max_output_tokens": cfg.get("model.max_output_tokens"),
            "thinking_param": cfg.get("model.thinking_param"),
            "thinking_on": cfg.get("model.thinking_on_reasoning_effort"),
            "thinking_off": cfg.get("model.thinking_off_reasoning_effort"),
        },
        "prompts": {
            "version": PROMPT_VERSION,
            "notes": describe_prompts()["notes"],
            "system_prompt": describe_prompts()["system_prompt"],
            "system_prompt_sha256": __import__("hashlib").sha256(
                describe_prompts()["system_prompt"].encode()
            ).hexdigest()[:16],
        },
        "task_sources": list(task_sources),
        "item_subset_source": item_subset_source,
        "conditions": [dict(c) for c in conditions],
        "plan": plan_summary(plans),
        "setup": check_setup(cfg),
        "counts": {},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_NAME
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        raise RunnerError(f"run manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Existing generations / resume
# ---------------------------------------------------------------------------


def load_generations(run_dir: str | Path) -> list[GenerationRecord]:
    path = Path(run_dir) / GENERATIONS_NAME
    records: list[GenerationRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(GenerationRecord.from_dict(json.loads(line)))
    return records


def _completed_keys(records: Iterable[GenerationRecord]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for record in records:
        if record.error is None and record.response_text.strip():
            keys.add((record.request_id, record.condition_id))
    return keys


# ---------------------------------------------------------------------------
# QPS limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    def __init__(self, qps: float | None):
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_generation(
    cfg: Config,
    tasks: Sequence[TaskItem],
    *,
    conditions: Sequence[dict[str, Any]] | None = None,
    item_subset: set[str] | None = None,
    run_id: str | None = None,
    tag: str = "",
    limit: int | None = None,
    dry_run: bool = False,
    mock: bool = False,
    client: BaseClient | None = None,
    task_sources: Sequence[str] = (),
    item_subset_source: str | None = None,
    extra_manifest: dict[str, Any] | None = None,
    max_cost_cny: float | None = None,
    item_subset_by_generation: Mapping[str, set[str]] | None = None,
    min_interval_s: float | None = None,
) -> dict[str, Any]:
    if conditions is None:
        conditions = [cfg.main_condition]
    plans = build_request_plan(cfg, tasks, conditions, item_subset=item_subset)
    plans = filter_plans_by_generation(plans, item_subset_by_generation)
    if limit is not None:
        # Keep at most N requests *per condition*, drawn as a deterministic
        # random sample: pilot runs with --limit must cover the whole condition
        # grid, all prompt views and the different audio/pair families, not just
        # the first N items of the catalog.
        import hashlib as _hashlib
        import random as _random

        per_condition: dict[str, list[RequestPlan]] = {}
        for plan in plans:
            per_condition.setdefault(plan.condition_id, []).append(plan)
        selected: list[RequestPlan] = []
        for condition_id, group in sorted(per_condition.items()):
            k = min(int(limit), len(group))
            digest = _hashlib.sha256(f"{cfg.seed()}:limit:{condition_id}".encode()).digest()
            rng = _random.Random(int.from_bytes(digest[:8], "little"))
            idx = sorted(rng.sample(range(len(group)), k)) if k < len(group) else range(len(group))
            selected.extend(group[i] for i in idx)
        plans = sorted(
            selected,
            key=lambda p: (p.condition_id, p.experiment, p.item_id, p.request_id),
        )
    summary = plan_summary(plans)
    log.info("run plan: %s", summary)

    pricing = Pricing.from_config(cfg)
    budget_cny = max_cost_cny if max_cost_cny is not None else cfg.get("generation.budget_cny")
    budget_cny = float(budget_cny) if budget_cny not in (None, "", 0) else None

    run_id = run_id or new_run_id(cfg, tag=tag or ("mock" if mock or dry_run else ""))
    run_dir = run_dir_for(cfg, run_id)
    manifest = build_manifest(
        cfg, run_id=run_id, plans=plans, task_sources=task_sources,
        item_subset_source=item_subset_source, conditions=conditions,
        mock=mock or dry_run, extra=extra_manifest, budget_cny=budget_cny,
    )
    write_manifest(run_dir, manifest)

    if dry_run:
        path = run_dir / PLAN_NAME
        with path.open("w", encoding="utf-8") as fh:
            for plan in plans:
                fh.write(json.dumps(plan.as_dict(), ensure_ascii=False) + "\n")
        manifest["status"] = "dry_run"
        manifest["counts"] = {"n_planned": len(plans)}
        manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_manifest(run_dir, manifest)
        return {"run_id": run_id, "run_dir": str(run_dir), "dry_run": True, **summary}

    # media variants (noise) must exist before the worker threads start.
    # Mock runs never read the audio, so they skip media preparation entirely.
    if not mock:
        noise_items: dict[str, str] = {}
        for plan in plans:
            if plan.audio_quality == "noise_20db":
                noise_items.setdefault(plan.audio_id, plan.media_rel)
        if noise_items:
            # media_rel points at the noisy destination; the clean source is
            # derived from the task manifest instead
            clean_by_audio = {}
            for task in tasks:
                clean_by_audio.setdefault(task.audio_id, task.wav_path)
            ensure_noisy_for(cfg, [(aid, clean_by_audio[aid]) for aid in sorted(noise_items)])
        missing = [p.media_path for p in plans if not Path(p.media_path).exists()]
        if missing:
            raise RunnerError(f"{len(missing)} media files missing, e.g. {missing[0]}")

    planned_path = run_dir / PLANNED_CALLS_NAME
    if not planned_path.exists() or not resume_ok(planned_path):
        with planned_path.open("w", encoding="utf-8") as fh:
            for plan in plans:
                fh.write(json.dumps(
                    {"request_id": plan.request_id, "condition_id": plan.condition_id},
                    ensure_ascii=False,
                ) + "\n")

    existing = load_generations(run_dir)
    done = _completed_keys(existing)
    todo = [p for p in plans if (p.request_id, p.condition_id) not in done]
    log.info("resume: %d/%d requests already done, %d to run",
             len(plans) - len(todo), len(plans), len(todo))

    if client is None:
        client = create_client(cfg, mock=mock or dry_run)
    concurrency = max(1, int(cfg.get("model.concurrency", 4)))
    limiter = RateLimiter(cfg.get("model.qps"))

    generations_path = run_dir / GENERATIONS_NAME
    write_lock = threading.Lock()
    counters = {"ok": 0, "error": 0}
    started = time.perf_counter()

    def _execute(plan: RequestPlan) -> ChatResponse:
        limiter.acquire()
        request = ChatRequest(
            request_id=plan.request_id,
            audio_path=plan.media_path,
            prompt_text=plan.prompt_text,
            output_form=plan.output_form,
            thinking=plan.thinking,
            system_prompt=plan.system_prompt,
            meta=plan.meta,
        )
        return client.generate(request)

    def _record(plan: RequestPlan, response: ChatResponse) -> GenerationRecord:
        return GenerationRecord(
            run_id=run_id,
            request_id=plan.request_id,
            item_id=plan.item_id,
            experiment=plan.experiment,
            prompt_condition=plan.prompt_condition,
            condition_id=plan.condition_id,
            thinking=plan.thinking,
            audio_quality=plan.audio_quality,
            output_form=plan.output_form,
            model=client.model,
            prompt_version=PROMPT_VERSION,
            response_text=response.text,
            reasoning_text=response.reasoning,
            finish_reason=response.finish_reason,
            usage=response.usage,
            latency_s=round(response.latency_s, 4),
            error=response.error,
            retries=response.retries,
            timestamp=response.timestamp,
            media_path=plan.media_rel,
        )

    counters.update({
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_cny": 0.0,
        "n_skipped_budget": 0,
    })
    budget_stop = False
    todo_iter = iter(todo)
    executed = 0
    # throttle staggering of request *starts* without reducing parallelism:
    # wait until there are (at most) k empty slots, then start one next request;
    # this spreads bursts across time while keeping `concurrency` in flight.
    min_interval = float(min_interval_s) if min_interval_s and min_interval_s > 0 else 0.0
    next_start = time.monotonic()
    if min_interval:
        log.info("staggering request starts by %.2fs (concurrency=%d)",
                 min_interval, concurrency)

    with generations_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            inflight: dict[Any, RequestPlan] = {}

            def _fill(wait_first: bool = False) -> None:
                """Submit up to ``concurrency`` more requests (unless budget hit)."""
                nonlocal next_start
                first = True
                while len(inflight) < concurrency and not budget_stop:
                    if min_interval and (wait_first or not first):
                        now = time.monotonic()
                        delay = next_start - now
                        if delay > 0:
                            time.sleep(delay)
                    next_start = time.monotonic() + min_interval
                    try:
                        plan = next(todo_iter)
                    except StopIteration:
                        return
                    inflight[pool.submit(_execute, plan)] = plan
                    first = False

            _fill()
            while inflight:
                done, _pending = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    plan = inflight.pop(future)
                    executed += 1
                    try:
                        response = future.result()
                    except Exception as exc:  # noqa: BLE001 - client should not raise
                        response = ChatResponse(
                            request_id=plan.request_id, error=f"worker crash: {exc}",
                        )
                    record = _record(plan, response)
                    usage = extract_usage(record.usage)
                    call_input_cost = pricing.input_cost(
                        usage["prompt_tokens"], usage["cached_tokens"]
                    )
                    call_output_cost = pricing.output_cost(usage["completion_tokens"])
                    with write_lock:
                        out_fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        out_fh.flush()
                        counters["ok" if record.error is None else "error"] += 1
                        counters["prompt_tokens"] += usage["prompt_tokens"]
                        counters["cached_tokens"] += usage["cached_tokens"]
                        counters["completion_tokens"] += usage["completion_tokens"]
                        counters["reasoning_tokens"] += usage["reasoning_tokens"]
                        counters["cost_cny"] += call_input_cost + call_output_cost
                        total = counters["ok"] + counters["error"]
                        if total % 100 == 0:
                            log.info(
                                "progress %d/%d (ok=%d error=%d cost=%.2f%s)",
                                total, len(todo), counters["ok"], counters["error"],
                                counters["cost_cny"],
                                f"/{budget_cny:.2f}" if budget_cny else "",
                            )
                    if (
                        budget_cny is not None
                        and not budget_stop
                        and counters["cost_cny"] >= budget_cny
                    ):
                        budget_stop = True
                        log.warning(
                            "cost budget reached: %.4f >= %.2f CNY; stopping new requests",
                            counters["cost_cny"], budget_cny,
                        )
                _fill(wait_first=min_interval > 0)

    counters["n_skipped_budget"] = max(0, len(todo) - executed)

    elapsed = time.perf_counter() - started
    all_done = counters["n_skipped_budget"] == 0
    manifest["status"] = "completed" if all_done else "budget_exceeded"
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    budget = budget_status(counters["cost_cny"], budget_cny)
    manifest["counts"] = {
        "n_planned": len(plans),
        "n_skipped_resume": len(plans) - len(todo),
        "n_skipped_budget": counters["n_skipped_budget"],
        "n_ok": counters["ok"],
        "n_error": counters["error"],
        "elapsed_s": round(elapsed, 1),
        "n_calls": counters["ok"] + counters["error"],
        "prompt_tokens": counters["prompt_tokens"],
        "cached_tokens": counters["cached_tokens"],
        "completion_tokens": counters["completion_tokens"],
        "reasoning_tokens": counters["reasoning_tokens"],
        "cost_cny": round(counters["cost_cny"], 6),
        "budget": budget,
    }
    write_manifest(run_dir, manifest)
    log.info("run %s finished in %.1fs: %s", run_id, elapsed, manifest["counts"])
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dry_run": False,
        **summary,
        **manifest["counts"],
    }


# ---------------------------------------------------------------------------
# Pre-run cost estimation
# ---------------------------------------------------------------------------


def load_audio_durations(cfg: Config) -> dict[str, float]:
    """audio_id -> duration seconds, from the prepared audio manifest (best effort)."""
    path = cfg.path("paths.manifest_dir") / "audio.parquet"
    if not path.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(path, columns=["audio_id", "duration_s"])
        return {str(a): float(d) for a, d in zip(df["audio_id"], df["duration_s"], strict=False)}
    except Exception:  # noqa: BLE001 - estimation must never break the CLI
        return {}


def estimate_plan_cost(cfg: Config, plans: Sequence[RequestPlan]) -> dict[str, Any]:
    """Estimate token usage and CNY cost of a plan before spending money.

    Two estimates are reported:

    * ``expected``     — expected prompt chars/token ratio (0.7) and expected
      thinking output tokens (``pricing.assumed_output_tokens_on``);
    * ``conservative`` — 1 token per prompt character and high thinking output
      tokens (``pricing.assumed_output_tokens_on_high``).

    Audio input tokens use the official rule ``ceil(duration_s) * 7``.  These are
    *estimates*; actual cost always comes from the API ``usage`` (see
    :mod:`pinyin_eval.pricing`).
    """
    pricing = Pricing.from_config(cfg)
    durations = load_audio_durations(cfg)
    fallback_duration = 0.65
    # model.max_output_tokens caps completion tokens (incl. reasoning), which
    # also caps the worst-case output cost per call.
    max_output_tokens = int(cfg.get("model.max_output_tokens", 256) or 256)

    by_condition: dict[str, dict[str, Any]] = {}
    totals: dict[str, float] = {
        "n_requests": 0,
        "expected_cost_cny": 0.0,
        "conservative_cost_cny": 0.0,
        "worst_case_cost_cny": 0.0,
        "prompt_tokens_expected": 0,
        "prompt_tokens_conservative": 0,
        "output_tokens_expected": 0,
        "output_tokens_conservative": 0,
    }
    for plan in plans:
        duration = durations.get(plan.audio_id, fallback_duration)
        system_prompt = plan.system_prompt or ""
        prompt_cons = pricing.estimate_prompt_tokens(
            system_prompt, plan.prompt_text, duration, mode="conservative"
        )
        prompt_exp = pricing.estimate_prompt_tokens(
            system_prompt, plan.prompt_text, duration, mode="expected"
        )
        output_exp = min(
            pricing.assumed_output_tokens(plan.thinking, high=False), max_output_tokens
        )
        output_cons = min(
            pricing.assumed_output_tokens(plan.thinking, high=True), max_output_tokens
        )
        cost_exp = pricing.input_cost(prompt_exp) + pricing.output_cost(output_exp)
        cost_cons = pricing.input_cost(prompt_cons) + pricing.output_cost(output_cons)
        cost_worst = pricing.input_cost(prompt_cons) + pricing.output_cost(max_output_tokens)

        totals["n_requests"] += 1
        totals["expected_cost_cny"] += cost_exp
        totals["conservative_cost_cny"] += cost_cons
        totals["prompt_tokens_expected"] += prompt_exp
        totals["prompt_tokens_conservative"] += prompt_cons
        totals["output_tokens_expected"] += output_exp
        totals["output_tokens_conservative"] += output_cons
        totals["worst_case_cost_cny"] += cost_worst

        cond = by_condition.setdefault(plan.condition_id, {
            "condition_id": plan.condition_id,
            "prompt_condition": plan.prompt_condition,
            "thinking": plan.thinking,
            "audio_quality": plan.audio_quality,
            "output_form": plan.output_form,
            "n_requests": 0,
            "expected_cost_cny": 0.0,
            "conservative_cost_cny": 0.0,
            "worst_case_cost_cny": 0.0,
        })
        cond["n_requests"] += 1
        cond["expected_cost_cny"] += cost_exp
        cond["conservative_cost_cny"] += cost_cons
        cond["worst_case_cost_cny"] += cost_worst

    for cond in by_condition.values():
        cond["expected_cost_cny"] = round(cond["expected_cost_cny"], 6)
        cond["conservative_cost_cny"] = round(cond["conservative_cost_cny"], 6)
        cond["worst_case_cost_cny"] = round(cond["worst_case_cost_cny"], 6)

    n = max(1, int(totals["n_requests"]))
    budget_cny = cfg.get("generation.budget_cny")
    budget_cny = float(budget_cny) if budget_cny not in (None, "", 0) else None
    return {
        "n_requests": int(totals["n_requests"]),
        "expected_cost_cny": round(totals["expected_cost_cny"], 6),
        "conservative_cost_cny": round(totals["conservative_cost_cny"], 6),
        "worst_case_cost_cny": round(totals["worst_case_cost_cny"], 6),
        "expected_cost_per_call_cny": round(totals["expected_cost_cny"] / n, 8),
        "conservative_cost_per_call_cny": round(totals["conservative_cost_cny"] / n, 8),
        "worst_case_cost_per_call_cny": round(totals["worst_case_cost_cny"] / n, 8),
        "max_output_tokens_cap": max_output_tokens,
        "prompt_tokens_expected": int(totals["prompt_tokens_expected"]),
        "prompt_tokens_conservative": int(totals["prompt_tokens_conservative"]),
        "output_tokens_expected": int(totals["output_tokens_expected"]),
        "output_tokens_conservative": int(totals["output_tokens_conservative"]),
        "by_condition": {k: v for k, v in sorted(by_condition.items())},
        "budget": {
            "budget_cny": budget_cny,
            "within_conservative_budget": (
                None if budget_cny is None else totals["conservative_cost_cny"] <= budget_cny
            ),
            "conservative_used_fraction": (
                None if budget_cny is None
                else round(totals["conservative_cost_cny"] / budget_cny, 4)
            ),
            "within_worst_case_budget": (
                None if budget_cny is None else totals["worst_case_cost_cny"] <= budget_cny
            ),
        },
        "pricing": pricing.as_dict(),
    }


# ---------------------------------------------------------------------------
# Scoring input assembly
# ---------------------------------------------------------------------------


def generations_frame(records: Sequence[GenerationRecord]) -> Any:
    """Convert generation records to a pandas DataFrame (deduplicated, latest win)."""
    import pandas as pd

    rows = []
    for record in records:
        row = record.to_dict()
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["request_id", "condition_id"], keep="last")
    return df


def resume_ok(planned_path: Path) -> bool:
    """True when an existing planned-calls file matches a non-empty run."""
    return planned_path.exists() and planned_path.stat().st_size > 0


__all__ = [
    "GENERATIONS_NAME",
    "estimate_plan_cost",
    "filter_plans_by_generation",
    "load_audio_durations",
    "PLANNED_CALLS_NAME",
    "MANIFEST_NAME",
    "PLAN_NAME",
    "RateLimiter",
    "RequestPlan",
    "RunnerError",
    "build_manifest",
    "build_request_plan",
    "generations_frame",
    "load_generations",
    "new_run_id",
    "plan_summary",
    "read_manifest",
    "run_dir_for",
    "run_generation",
    "write_manifest",
]
