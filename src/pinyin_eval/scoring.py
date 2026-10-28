"""Scoring: join task items with generation records, parse predictions, build the
item-level result table and the detection-analysis rows.

Outputs per run (``results/runs/<run_id>/``):

* ``items.parquet`` — one row per (task item, generation condition) with parsed &
  normalised predictions and gold labels (``ItemResult`` schema);
* ``detection_rows.parquet`` — negatives from exp1 + positives from exp2, tagged
  with ``detection_role`` / ``detection_subset``;
* ``score_summary.json`` — parse-status counts and coverage per condition.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Config
from .normalize import load_mapping
from .parsing import OutputNormalizer, ParsedPrediction, parse_response
from .records import ItemResult, TaskItem
from .runner import PLANNED_CALLS_NAME, load_generations, read_manifest
from .tasks import load_item_ids, load_tasks, request_id_for_task

log = logging.getLogger(__name__)

ITEMS_NAME = "items.parquet"
DETECTION_ROWS_NAME = "detection_rows.parquet"
SUMMARY_NAME = "score_summary.json"
PARSE_FAILURES_NAME = "parse_failures.jsonl"


class ScoringError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Loading run inputs
# ---------------------------------------------------------------------------


def _resolve_source(cfg: Config, source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else cfg.repo_root / path


def load_run_tasks(cfg: Config, manifest: Mapping[str, Any], tasks: Sequence[TaskItem] | None) -> list[TaskItem]:
    if tasks is not None:
        return list(tasks)
    sources = manifest.get("task_sources") or []
    if not sources:
        raise ScoringError("run manifest has no task_sources and no tasks were provided")
    all_tasks: list[TaskItem] = []
    for source in sources:
        path = _resolve_source(cfg, str(source))
        all_tasks.extend(load_tasks(path))
    return all_tasks


def load_run_item_subset(cfg: Config, manifest: Mapping[str, Any]) -> set[str] | None:
    """Load the item-id subset of a run (may reference several comma-separated files)."""
    source = manifest.get("item_subset_source")
    if not source:
        return None
    items: set[str] = set()
    found = False
    for part in str(source).split(","):
        part = part.strip()
        if not part:
            continue
        path = _resolve_source(cfg, part)
        if not path.exists():
            log.warning("item subset file missing: %s", path)
            continue
        items.update(load_item_ids(path))
        found = True
    return items if found else None


# ---------------------------------------------------------------------------
# Build item-level results
# ---------------------------------------------------------------------------


def _make_result(
    task: TaskItem,
    condition: Mapping[str, Any],
    condition_id: str,
    generation: Any | None,
    parsed: ParsedPrediction | None,
    *,
    run_id: str,
    error: str | None = None,
) -> ItemResult:
    status = "ok"
    raw_response = ""
    raw_reasoning = ""
    latency = 0.0
    gen_error: str | None = error
    if generation is not None:
        raw_response = generation.response_text or ""
        raw_reasoning = generation.reasoning_text or ""
        latency = float(generation.latency_s or 0.0)
        gen_error = generation.error or error
        if gen_error:
            status = "api_error"
        elif not raw_response.strip():
            status = "empty"
        elif parsed is None:
            status = "parse_error"
        else:
            status = parsed.status
    else:
        status = "api_error" if not error else "api_error"

    pred = parsed or ParsedPrediction(status="parse_error")
    return ItemResult(
        run_id=run_id,
        item_id=task.item_id,
        experiment=task.experiment,
        prompt_condition=task.prompt_condition,
        condition_id=condition_id,
        task_type=task.task_type,
        thinking=bool(condition["thinking"]),
        audio_quality=str(condition["audio_quality"]),
        output_form=str(condition["output_form"]),
        audio_id=task.audio_id,
        cluster_id=task.cluster_id or task.audio_id,
        voice=task.voice,
        pair_id=task.pair_id,
        pair_type=task.pair_type,
        pair_side=task.pair_side,
        variant_type=task.variant_type,
        gold_initial=task.gold_initial,
        gold_final=task.gold_final,
        gold_tone=int(task.gold_tone),
        given_target=task.given_target,
        given_tone=task.given_tone,
        consistent_gold=task.consistent_gold,
        alternatives=task.alternatives,
        correct_choice=task.correct_choice,
        pred_consistent=pred.consistent if status == "ok" else None,
        pred_initial=pred.initial if status == "ok" else None,
        pred_final=pred.final if status == "ok" else None,
        pred_tone=pred.tone if status == "ok" else None,
        pred_choice=pred.choice if status == "ok" else None,
        parse_status=status,
        raw_response=raw_response,
        raw_reasoning=raw_reasoning,
        latency_s=latency,
        error=gen_error,
    )


def _load_planned_calls(run_dir: Path | None) -> set[tuple[str, str]] | None:
    if run_dir is None:
        return None
    path = Path(run_dir) / PLANNED_CALLS_NAME
    if not path.exists() or path.stat().st_size == 0:
        return None
    out: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.add((obj["request_id"], obj["condition_id"]))
    return out


def build_item_results(
    cfg: Config,
    tasks: Sequence[TaskItem],
    manifest: Mapping[str, Any],
    generations: Sequence[Any],
    *,
    run_id: str,
    run_dir: Path | None = None,
) -> pd.DataFrame:
    mapping = load_mapping(cfg.path("paths.mapping_csv"))
    normalizer = OutputNormalizer(mapping)
    conditions = list(manifest.get("conditions") or [])
    if not conditions:
        raise ScoringError("run manifest has no conditions")
    subset = load_run_item_subset(cfg, manifest)

    gen_index: dict[tuple[str, str], Any] = {}
    for record in generations:
        gen_index[(record.request_id, record.condition_id)] = record
    planned = _load_planned_calls(run_dir)

    rows: list[dict[str, Any]] = []
    missing_calls = 0
    for task in tasks:
        if subset is not None and task.item_id not in subset:
            continue
        for condition in conditions:
            thinking = bool(condition["thinking"])
            quality = str(condition["audio_quality"])
            form = str(condition["output_form"])
            condition_id = f"{task.prompt_condition}__{'think1' if thinking else 'think0'}__{quality}__{form}"
            request_id = request_id_for_task(
                cfg, task, output_form=form, audio_quality=quality
            )
            if planned is not None and (request_id, condition_id) not in planned:
                continue
            generation = gen_index.get((request_id, condition_id))
            parsed: ParsedPrediction | None = None
            if generation is not None and not generation.error and generation.response_text.strip():
                parsed = parse_response(
                    generation.response_text,
                    task_type=task.task_type,
                    output_form=form,
                    normalizer=normalizer,
                )
            if generation is None:
                missing_calls += 1
            result = _make_result(
                task, condition, condition_id, generation, parsed, run_id=run_id,
            )
            rows.append(result.to_row())
    df = pd.DataFrame(rows)
    log.info("item results: %d rows (%d task x condition cells without generation)",
             len(df), missing_calls)
    return df


# ---------------------------------------------------------------------------
# Correctness helpers
# ---------------------------------------------------------------------------


def attach_correctness(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-dimension correctness columns (NA when the task does not ask).

    Task types map to the design's output schemas:

    * ``free``   -> initial / final / tone / joint correctness (experiment 3, 4-6 free);
    * ``tone``   -> tone correctness only (experiments 1/2 and target conditions);
    * ``detect`` -> consistency correctness only (detection experiment);
    * ``ab``     -> A/B choice correctness (experiments 4-6).
    """
    if df.empty:
        return df
    out = df.copy()
    ok = out["parse_status"].eq("ok").tolist()
    gold_tone = out["gold_tone"].tolist()
    gold_initial = out["gold_initial"].tolist()
    gold_final = out["gold_final"].tolist()
    pred_initial = out["pred_initial"].tolist()
    pred_final = out["pred_final"].tolist()
    pred_tone = out["pred_tone"].tolist()
    consistent_gold = out["consistent_gold"].tolist()
    pred_consistent = out["pred_consistent"].tolist()
    correct_choice = out["correct_choice"].tolist()
    pred_choice = out["pred_choice"].tolist()

    def _missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    initial_correct: list[Any] = []
    final_correct: list[Any] = []
    tone_correct: list[Any] = []
    joint_correct: list[Any] = []
    consistent_correct: list[Any] = []
    choice_correct: list[Any] = []
    for i in range(len(out)):
        valid = ok[i]
        i_ok = valid and not _missing(pred_initial[i]) and gold_initial[i] == pred_initial[i]
        f_ok = valid and not _missing(pred_final[i]) and gold_final[i] == pred_final[i]
        t_ok = valid and not _missing(pred_tone[i]) and int(gold_tone[i]) == int(pred_tone[i])
        initial_correct.append(i_ok if valid else None)
        final_correct.append(f_ok if valid else None)
        tone_correct.append(t_ok if valid else None)
        joint_correct.append((i_ok and f_ok and t_ok) if valid else None)
        c_ok = (
            valid
            and consistent_gold[i] is not None
            and not _missing(pred_consistent[i])
            and bool(consistent_gold[i]) == bool(pred_consistent[i])
        )
        consistent_correct.append(c_ok if valid else None)
        ch_ok = (
            valid
            and correct_choice[i] is not None
            and not _missing(pred_choice[i])
            and str(correct_choice[i]) == str(pred_choice[i])
        )
        choice_correct.append(ch_ok if valid else None)

    out["initial_correct"] = pd.array(initial_correct, dtype="boolean")
    out["final_correct"] = pd.array(final_correct, dtype="boolean")
    out["tone_correct"] = pd.array(tone_correct, dtype="boolean")
    out["joint_correct"] = pd.array(joint_correct, dtype="boolean")
    out["consistent_correct"] = pd.array(consistent_correct, dtype="boolean")
    out["choice_correct"] = pd.array(choice_correct, dtype="boolean")

    tt = out["task_type"].astype(str) if "task_type" in out.columns else ""
    is_free = tt.eq("free") if hasattr(tt, "eq") else (tt == "free")
    is_tone = tt.eq("tone") if hasattr(tt, "eq") else (tt == "tone")
    is_detect = tt.eq("detect") if hasattr(tt, "eq") else (tt == "detect")
    is_ab = tt.eq("ab") if hasattr(tt, "eq") else (tt == "ab")

    # initial/final/joint exist only for free recognition
    for col in ("initial_correct", "final_correct", "joint_correct"):
        out.loc[~is_free, col] = pd.NA
    # tone is measured for free recognition and for the tone-only tasks
    out.loc[~(is_free | is_tone), "tone_correct"] = pd.NA
    out["consistency_tone_correct"] = pd.NA  # kept for schema compatibility
    out.loc[~is_detect, "consistent_correct"] = pd.NA
    out.loc[~is_ab, "choice_correct"] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Detection rows
# ---------------------------------------------------------------------------


def generation_key(condition_id: str) -> str:
    """``correct_target__think1__clean__json`` -> ``think1__clean__json``."""
    parts = condition_id.split("__", 1)
    return parts[1] if len(parts) == 2 else condition_id


_generation_key = generation_key  # backwards-compatible alias


_PROMPT_PREFIX = {"neg": "correct_target", "pos": "wrong_target"}


def build_detection_rows(
    cfg: Config,
    items_df: pd.DataFrame,
    *,
    detection_sets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pair consistent (negative) and inconsistent (positive) detection claims.

    Both claims belong to the dedicated detection experiment (prompt condition
    ``detect``) and are replayed under the same generation condition
    (thinking / audio quality / output form).
    """
    if items_df.empty:
        return pd.DataFrame()
    if detection_sets is None:
        detection_path = cfg.path("paths.task_dir") / "detection_sets.csv"
        if not detection_path.exists():
            log.warning("detection_sets.csv missing; run build-tasks first")
            return pd.DataFrame()
        detection_sets = pd.read_csv(detection_path)

    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in items_df.to_dict("records"):
        index[(row["item_id"], _generation_key(row["condition_id"]))] = row
    generation_keys = sorted({_generation_key(c) for c in items_df["condition_id"].unique()})

    rows: list[dict[str, Any]] = []
    for detection in detection_sets.to_dict("records"):
        subset = detection["subset"]
        for gkey in generation_keys:
            neg = index.get((detection["negative_item_id"], gkey))
            pos = index.get((detection["positive_item_id"], gkey))
            if neg is None and pos is None:
                continue
            for role, row in (("neg", neg), ("pos", pos)):
                if row is None:
                    continue
                record = dict(row)
                record["detection_role"] = role
                record["detection_subset"] = subset
                record["generation_condition"] = gkey
                record["variant_type"] = detection.get("variant_type") if role == "pos" else None
                rows.append(record)
    df = pd.DataFrame(rows)
    log.info("detection rows: %d (%d generation conditions)", len(df), len(generation_keys))
    return df


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------


def score_run(
    cfg: Config,
    run_dir: str | Path,
    *,
    tasks: Sequence[TaskItem] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = cfg.path("paths.results_dir") / "runs" / run_dir
    manifest = read_manifest(run_dir)
    generations = load_generations(run_dir)
    task_list = load_run_tasks(cfg, manifest, tasks)
    run_id = str(manifest.get("run_id") or run_dir.name)

    items = build_item_results(
        cfg, task_list, manifest, generations, run_id=run_id, run_dir=run_dir
    )
    items = attach_correctness(items)
    detection = build_detection_rows(cfg, items)
    if detection.empty and not items.empty:
        # fallback for fresh checkouts / tests: rebuild detection sets from the
        # task catalog in memory when data/manifests/tasks/detection_sets.csv is absent
        from .tasks import build_detection_sets

        has_detect = "detect" in set(items["experiment"].unique())
        if has_detect:
            detection_sets = build_detection_sets(task_list, cfg)
            detection = build_detection_rows(cfg, items, detection_sets=detection_sets)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "n_items": int(len(items)),
        "n_tasks": len(task_list),
        "n_generations": len(generations),
        "n_unique_requests": len({(g.request_id, g.condition_id) for g in generations}),
        "parse_status": {k: int(v) for k, v in items["parse_status"].value_counts().items()},
        "per_condition": {},
        "per_experiment": {},
        "detection_rows": int(len(detection)),
    }
    for condition_id, group in items.groupby("condition_id"):
        summary["per_condition"][condition_id] = {
            "n": int(len(group)),
            "ok": int((group["parse_status"] == "ok").sum()),
            "failures": int((group["parse_status"] != "ok").sum()),
        }
    for experiment, group in items.groupby("experiment"):
        summary["per_experiment"][experiment] = {
            "n": int(len(group)),
            "ok": int((group["parse_status"] == "ok").sum()),
        }

    if write:
        run_dir.mkdir(parents=True, exist_ok=True)
        items_path = run_dir / ITEMS_NAME
        items.to_parquet(items_path, index=False)
        if not detection.empty:
            detection.to_parquet(run_dir / DETECTION_ROWS_NAME, index=False)
        with (run_dir / SUMMARY_NAME).open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        failures = items[items["parse_status"] != "ok"].head(200)
        with (run_dir / PARSE_FAILURES_NAME).open("w", encoding="utf-8") as fh:
            for row in failures.to_dict("records"):
                fh.write(json.dumps({
                    "item_id": row["item_id"],
                    "condition_id": row["condition_id"],
                    "experiment": row["experiment"],
                    "parse_status": row["parse_status"],
                    "response": (row.get("raw_response") or "")[:500],
                    "error": row.get("error"),
                }, ensure_ascii=False) + "\n")
    log.info("scored %d items -> %s", len(items), run_dir)
    return {"items": items, "detection": detection, "summary": summary}


def load_run_items(cfg: Config, run_id_or_dir: str | Path) -> pd.DataFrame:
    path = Path(run_id_or_dir)
    if not path.is_absolute() or not path.exists():
        path = cfg.path("paths.results_dir") / "runs" / str(run_id_or_dir)
    items_path = path / ITEMS_NAME
    if not items_path.exists():
        raise ScoringError(f"items.parquet not found: {items_path}; run `score` first")
    return pd.read_parquet(items_path)


__all__ = [
    "DETECTION_ROWS_NAME",
    "generation_key",
    "ITEMS_NAME",
    "PARSE_FAILURES_NAME",
    "SUMMARY_NAME",
    "ScoringError",
    "attach_correctness",
    "build_detection_rows",
    "build_item_results",
    "load_run_items",
    "load_run_tasks",
    "score_run",
]
