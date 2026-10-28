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
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .client import BaseClient, ChatRequest, ChatResponse, check_setup, create_client
from .config import Config
from .data.dataset import ensure_noisy_for
from .prompts import PROMPT_VERSION, describe_prompts, task_kind
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
    task_kind: str
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
        "task_kind": task_kind(task.prompt_condition),
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
                task_kind=task_kind(task.prompt_condition),
                meta=_task_meta(task),
            )
    return sorted(plans.values(), key=lambda p: (p.condition_id, p.experiment, p.item_id, p.request_id))


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
        "prompts": {"version": PROMPT_VERSION, "notes": describe_prompts()["notes"]},
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
) -> dict[str, Any]:
    if conditions is None:
        conditions = [cfg.main_condition]
    plans = build_request_plan(cfg, tasks, conditions, item_subset=item_subset)
    if limit is not None:
        # keep the first N requests *per condition*, so --limit is useful for
        # pilot runs that need to cover the whole condition grid cheaply
        per_condition: dict[str, list[RequestPlan]] = {}
        for plan in plans:
            per_condition.setdefault(plan.condition_id, []).append(plan)
        plans = sorted(
            [p for group in per_condition.values() for p in group[: int(limit)]],
            key=lambda p: (p.condition_id, p.experiment, p.item_id, p.request_id),
        )
    summary = plan_summary(plans)
    log.info("run plan: %s", summary)

    run_id = run_id or new_run_id(cfg, tag=tag or ("mock" if mock or dry_run else ""))
    run_dir = run_dir_for(cfg, run_id)
    manifest = build_manifest(
        cfg, run_id=run_id, plans=plans, task_sources=task_sources,
        item_subset_source=item_subset_source, conditions=conditions,
        mock=mock or dry_run, extra=extra_manifest,
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

    with generations_path.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_execute, plan): plan for plan in todo}
            for future in as_completed(futures):
                plan = futures[future]
                try:
                    response = future.result()
                except Exception as exc:  # noqa: BLE001 - client should not raise, but be safe
                    response = ChatResponse(
                        request_id=plan.request_id, error=f"worker crash: {exc}",
                    )
                record = _record(plan, response)
                with write_lock:
                    out_fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    out_fh.flush()
                    counters["ok" if record.error is None else "error"] += 1
                    total = counters["ok"] + counters["error"]
                    if total % 100 == 0:
                        log.info("progress %d/%d (ok=%d error=%d)",
                                 total, len(todo), counters["ok"], counters["error"])

    elapsed = time.perf_counter() - started
    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["counts"] = {
        "n_planned": len(plans),
        "n_skipped_resume": len(plans) - len(todo),
        "n_ok": counters["ok"],
        "n_error": counters["error"],
        "elapsed_s": round(elapsed, 1),
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
