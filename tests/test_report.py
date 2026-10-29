"""Tests for :mod:`pinyin_eval.report` (contract section 8).

The tests build a small synthetic scored run in a temporary directory (items /
detection rows / run manifest) that follows the **2026-10-28 design update**
(experiments 1/2 are tone-only, a dedicated ``detect`` experiment provides the
consistency judgement, experiments 4-6 are A/B-with-free views, experiment 6 has
no wrong-target prompt) and check that

* every expected artifact is created and the returned paths exist;
* the numbers in ``metrics.csv`` / ``detection.csv`` / ``baselines.csv`` equal
  what calling :mod:`pinyin_eval.metrics` directly on the same rows returns;
* detection metrics come from the dedicated ``detect`` experiment rows;
* ablation reports the three groups (core / detection / pairs) and the
  detection balance flags;
* comparisons use the new families (exp1/2 tone, exp3 4 metrics, detect
  consistency, exp4/5/6 dimension + A/B);
* runs with missing pieces (no detection rows, one generation condition, no
  manifest, ...) degrade gracefully instead of crashing;
* the CSV column order is stable;
* bootstrap results are reproducible for a fixed seed.

No file under ``data/`` or ``results/`` is written; everything lives under
``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pinyin_eval import metrics
from pinyin_eval.config import Config
from pinyin_eval.report import (
    ABLATION_COLUMNS,
    BASELINE_COLUMNS,
    COMPARISON_COLUMNS,
    DETECTION_COLUMNS,
    METRICS_COLUMNS,
    USAGE_COLUMNS,
    write_report,
)

from .conftest import CONFIG_PATH, MAPPING_CSV

# ---------------------------------------------------------------------------
# Synthetic dataset description (2026-10-28 v3 design)
# ---------------------------------------------------------------------------

VOICES = ("A", "B")
#: (target, initial, final, tone)
AUDIO_TABLE: tuple[tuple[str, str, str, int], ...] = (
    ("zhang1", "zh", "ang", 1),
    ("zang1", "z", "ang", 1),
    ("chan1", "ch", "an", 1),
    ("chang1", "ch", "ang", 1),
    ("zhang2", "zh", "ang", 2),
)
AUDIO_INDEX = {
    f"ZH_TTS_{voice}_{target}": index
    for voice in VOICES
    for index, (target, _initial, _final, _tone) in enumerate(AUDIO_TABLE)
}
#: (experiment, pair_type, side A target, side B target)
PAIR_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("exp4", "sh_zh", "zhang1", "zang1"),
    ("exp5", "front_back_nasal_ptr", "chan1", "chang1"),
    ("exp6", "tone", "zhang1", "zhang2"),
)
#: v3: experiments 4/5/6 have three sub-experiments each (3 prompt conditions)
PAIR_PROMPT_CONDITIONS: tuple[str, str, str] = ("free", "correct_target", "wrong_target")
#: output dimension of each minimal-pair experiment in v3
PAIR_TASK_TYPES: dict[str, str] = {"exp4": "initial", "exp5": "final", "exp6": "tone"}
#: detection variants per audio: 2 changed initials, 2 changed finals, 1 tone
DETECT_VARIANTS: tuple[tuple[str, int], ...] = (
    ("initial", 0),
    ("initial", 1),
    ("final", 0),
    ("final", 1),
    ("tone", 0),
)

MAIN_GENERATION_KEY = "think1__clean__json"
CONDITIONS: tuple[dict[str, object], ...] = (
    {"thinking": True, "audio_quality": "clean", "output_form": "json"},
    {"thinking": False, "audio_quality": "clean", "output_form": "json"},
    {"thinking": True, "audio_quality": "noise_20db", "output_form": "json"},
    {"thinking": True, "audio_quality": "clean", "output_form": "text"},
)
SALT = {
    "think1__clean__json": 0,
    "think0__clean__json": 1,
    "think1__noise_20db__json": 2,
    "think1__clean__text": 3,
}
SUBSET_TARGETS = ("zhang1", "zang1")


def _generation_key(condition: dict[str, object]) -> str:
    thinking = "think1" if condition["thinking"] else "think0"
    return f"{thinking}__{condition['audio_quality']}__{condition['output_form']}"


def _audio_id(voice: str, target: str) -> str:
    """``target`` already carries the tone digit (e.g. ``zhang1``)."""
    return f"ZH_TTS_{voice}_{target}"


def _audio_entry(target: str) -> tuple[str, str, int]:
    for name, initial, final, tone in AUDIO_TABLE:
        if name == target:
            return initial, final, tone
    raise KeyError(target)


def _base_row(
    *,
    item_id: str,
    experiment: str,
    prompt_condition: str,
    task_type: str,
    voice: str,
    target: str,
    **extra: object,
) -> dict[str, object]:
    initial, final, tone = _audio_entry(target)
    audio_id = _audio_id(voice, target)
    row: dict[str, object] = {
        "item_id": item_id,
        "experiment": experiment,
        "prompt_condition": prompt_condition,
        "task_type": task_type,
        "audio_id": audio_id,
        "cluster_id": audio_id,
        "voice": voice,
        "gold_initial": initial,
        "gold_final": final,
        "gold_tone": tone,
        "given_target": None,
        "given_initial": None,
        "given_final": None,
        "given_tone": None,
        "consistent_gold": None,
        "variant_type": None,
        "variant_index": None,
        "pair_id": None,
        "pair_type": None,
        "pair_side": None,
        "alternatives": None,
        "correct_choice": None,
    }
    row.update(extra)
    return row


def _pair_given_fields(
    experiment: str, prompt_condition: str, target: str
) -> dict[str, object]:
    """v3 target-information fields for experiments 4-6 (report ignores them)."""
    initial, final, tone = _audio_entry(target)
    fields: dict[str, object] = {
        "given_target": target,
        "given_initial": initial,
        "given_final": final,
        "given_tone": tone,
    }
    if prompt_condition == "wrong_target":
        dimension = PAIR_TASK_TYPES[experiment]
        if dimension == "initial":
            fields["given_initial"] = _alternate(str(initial))
        elif dimension == "final":
            fields["given_final"] = _alternate(str(final))
        else:
            fields["given_tone"] = (int(tone) % 4) + 1
    return fields


def _base_items() -> list[dict[str, object]]:
    """One row per task item, independent of the generation condition."""
    items: list[dict[str, object]] = []
    for voice in VOICES:
        for target, _initial, _final, _tone in AUDIO_TABLE:
            audio_id = _audio_id(voice, target)
            # experiment 1: correct information -> tone answer only (v3)
            items.append(
                _base_row(
                    item_id=f"exp1|correct_target|{audio_id}",
                    experiment="exp1",
                    prompt_condition="correct_target",
                    task_type="tone",
                    voice=voice,
                    target=target,
                    given_target=target,
                    consistent_gold=True,
                )
            )
            # experiment 2: five wrong toneless targets -> tone answer only
            for variant_type, variant_index in (
                ("initial", 0),
                ("initial", 1),
                ("final", 0),
                ("final", 1),
                ("final", 2),
            ):
                items.append(
                    _base_row(
                        item_id=(
                            f"exp2|wrong_target|{audio_id}|"
                            f"{variant_type}{variant_index}"
                        ),
                        experiment="exp2",
                        prompt_condition="wrong_target",
                        task_type="tone",
                        voice=voice,
                        target=target,
                        given_target=_audio_entry(target)[0] + "x",
                        consistent_gold=False,
                        variant_type=variant_type,
                        variant_index=variant_index,
                    )
                )
            # experiment 3: free detection / recognition
            items.append(
                _base_row(
                    item_id=f"exp3|free|{audio_id}",
                    experiment="exp3",
                    prompt_condition="free",
                    task_type="free",
                    voice=voice,
                    target=target,
                )
            )
            # dedicated detection experiment: 1 consistent claim + 5 variants
            items.append(
                _base_row(
                    item_id=f"detect|detect|{audio_id}",
                    experiment="detect",
                    prompt_condition="detect",
                    task_type="detect",
                    voice=voice,
                    target=target,
                    given_target=target,
                    given_tone=_tone,
                    consistent_gold=True,
                )
            )
            for variant_type, variant_index in DETECT_VARIANTS:
                items.append(
                    _base_row(
                        item_id=(
                            f"detect|detect|{audio_id}|"
                            f"{variant_type}{variant_index}"
                        ),
                        experiment="detect",
                        prompt_condition="detect",
                        task_type="detect",
                        voice=voice,
                        target=target,
                        given_target=_audio_entry(target)[0] + "x",
                        given_tone=_tone if variant_type != "tone" else (_tone % 4) + 1,
                        consistent_gold=False,
                        variant_type=variant_type,
                        variant_index=variant_index,
                    )
                )
        # minimal pairs (experiments 4-6, three sub-experiments each in v3)
        for experiment, pair_type, side_a, side_b in PAIR_SPECS:
            tone = _audio_entry(side_a)[2]
            pair_id = f"{experiment}:{side_a}/{side_b}:{voice}:{tone}"
            for side, target in (("A", side_a), ("B", side_b)):
                audio_id = _audio_id(voice, target)
                for prompt_condition in PAIR_PROMPT_CONDITIONS:
                    items.append(
                        _base_row(
                            item_id=(
                                f"{experiment}|{prompt_condition}|"
                                f"{audio_id}|{pair_id}"
                            ),
                            experiment=experiment,
                            prompt_condition=prompt_condition,
                            task_type=PAIR_TASK_TYPES[experiment],
                            voice=voice,
                            target=target,
                            pair_id=pair_id,
                            pair_type=pair_type,
                            pair_side=side,
                            **_pair_given_fields(experiment, prompt_condition, target),
                        )
                    )
    return items


def _alternate(value: str, pool: tuple[str, ...] = ("m", "f", "s")) -> str:
    for candidate in pool:
        if candidate != value:
            return candidate
    return value + "x"


def _with_predictions(base: dict[str, object], salt: int) -> dict[str, object]:
    """Attach deterministic predictions; correctness depends on ``salt``/index."""
    row = dict(base)
    index = AUDIO_INDEX[str(base["audio_id"])]
    gold_initial = str(base["gold_initial"])
    gold_final = str(base["gold_final"])
    gold_tone = int(base["gold_tone"])
    task_type = str(base["task_type"])
    consistent_gold = bool(base.get("consistent_gold"))
    row.update(
        {
            "pred_initial": None,
            "pred_final": None,
            "pred_tone": None,
            "pred_consistent": None,
            "pred_choice": None,
            "parse_status": "ok",
            "error": None,
            "raw_response": "",
            "latency_s": 0.0,
        }
    )
    if task_type == "tone":
        tone_ok = (index + salt) % 2 == 0
        row["pred_tone"] = gold_tone if tone_ok else (gold_tone % 4) + 1
    elif task_type == "free":
        row["pred_initial"] = (
            gold_initial if (index + salt) % 3 != 0 else _alternate(gold_initial)
        )
        row["pred_final"] = (
            gold_final if (index + salt) % 4 != 0 else _alternate(gold_final)
        )
        row["pred_tone"] = (
            gold_tone if (index + salt) % 5 != 0 else (gold_tone % 4) + 1
        )
    elif task_type == "initial":
        row["pred_initial"] = (
            gold_initial if (index + salt) % 2 == 0 else _alternate(gold_initial)
        )
    elif task_type == "final":
        row["pred_final"] = (
            gold_final if (index + salt) % 2 == 0 else _alternate(gold_final)
        )
    elif task_type == "detect":
        answered = (index + salt) % 4 != 0
        row["pred_consistent"] = consistent_gold if answered else not consistent_gold
    return row


def _subset_audios() -> set[str]:
    return {_audio_id(voice, target) for voice in VOICES for target in SUBSET_TARGETS}


def _ablation_ids() -> dict[str, set[str]]:
    """Item ids of the two v3 ablation subsets (core / pairs)."""
    groups: dict[str, set[str]] = {"core": set(), "pairs": set()}
    subset_audios = _subset_audios()
    for base in _base_items():
        item_id = str(base["item_id"])
        experiment = str(base["experiment"])
        if experiment in ("exp1", "exp2", "exp3"):
            if str(base["audio_id"]) in subset_audios:
                groups["core"].add(item_id)
        elif experiment in ("exp4", "exp5", "exp6"):
            groups["pairs"].add(item_id)
    return groups


def _ablation_item_ids() -> set[str]:
    groups = _ablation_ids()
    return groups["core"] | groups["pairs"]


def _build_items(ablation_ids: set[str], *, full_grid: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        generation = _generation_key(dict(condition))
        is_main = generation == MAIN_GENERATION_KEY
        salt = SALT[generation]
        for base in _base_items():
            item_id = str(base["item_id"])
            # detection experiment and non-ablation tasks are only planned on the
            # main condition in the v3 minimal/ablation design; ``full_grid`` keeps
            # every task in every condition (useful for the comparison tests).
            if not full_grid and not is_main and item_id not in ablation_ids:
                continue
            row = _with_predictions(base, salt)
            row.update(
                {
                    "condition_id": f"{base['prompt_condition']}__{generation}",
                    "thinking": condition["thinking"],
                    "audio_quality": condition["audio_quality"],
                    "output_form": condition["output_form"],
                }
            )
            if is_main and item_id == "exp3|free|ZH_TTS_B_zhang2":
                row.update({"parse_status": "parse_error", "raw_response": "{bad json"})
                for column in (
                    "pred_initial",
                    "pred_final",
                    "pred_tone",
                    "pred_consistent",
                    "pred_choice",
                ):
                    row[column] = None
            if not is_main and item_id == "exp1|correct_target|ZH_TTS_A_zhang1":
                row.update(
                    {
                        "parse_status": "refused",
                        "pred_tone": None,
                        "raw_response": "拒绝回答",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _build_detection(items: pd.DataFrame) -> pd.DataFrame:
    """Derive detection rows from the dedicated ``detect`` experiment items."""
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in items.to_dict("records"):
        generation = str(row["condition_id"]).split("__", 1)[1]
        lookup[(generation, str(row["item_id"]))] = row
    generation_conditions = sorted(
        {str(row["condition_id"]).split("__", 1)[1] for row in items.to_dict("records")}
    )
    # main subset uses one of the five inconsistent variants; typed subsets use
    # their own variant type
    subset_variant = {
        "main": "tone0",
        "initial": "initial0",
        "final": "final0",
        "tone": "tone0",
    }
    rows: list[dict[str, object]] = []
    for generation in generation_conditions:
        salt = SALT.get(generation, 0)
        for voice in VOICES:
            for target, _initial, _final, _tone in AUDIO_TABLE:
                audio_id = _audio_id(voice, target)
                index = AUDIO_INDEX[audio_id]
                negative_id = f"detect|detect|{audio_id}"
                if (generation, negative_id) not in lookup:
                    continue
                for subset, variant_suffix in subset_variant.items():
                    positive_id = f"detect|detect|{audio_id}|{variant_suffix}"
                    if (generation, positive_id) not in lookup:
                        continue
                    negative = dict(lookup[(generation, negative_id)])
                    negative.update(
                        {
                            "detection_role": "neg",
                            "detection_subset": subset,
                            "generation_condition": generation,
                            "pred_consistent": (index + salt) % 4 != 0,
                            "parse_status": "ok",
                        }
                    )
                    positive = dict(lookup[(generation, positive_id)])
                    positive.update(
                        {
                            "detection_role": "pos",
                            "detection_subset": subset,
                            "generation_condition": generation,
                            "pred_consistent": (index + salt) % 5 == 0,
                            "parse_status": "ok",
                        }
                    )
                    rows.append(negative)
                    rows.append(positive)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run construction helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> Config:
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
        "report": {
            "per_voice": True,
            "confusion_matrices": True,
            "mcnemar": True,
            "multiple_comparison": "holm",
        },
    }
    return Config.load(CONFIG_PATH, repo_root=tmp_path, overrides=overrides)


def _write_manifest(run_dir: Path, run_id: str, conditions: list[dict[str, object]]) -> None:
    manifest = {
        "run_id": run_id,
        "created_at": "2026-10-29T19:00:00+08:00",
        "status": "completed",
        "git_commit": None,
        "config_path": str(CONFIG_PATH),
        "config_sha256": "deadbeef",
        "config": {
            "generation": {
                "main_condition": {
                    "thinking": True,
                    "audio_quality": "clean",
                    "output_form": "json",
                }
            },
            "model": {"model": "qwen3.8-omni-flash"},
        },
        "model": {
            "name": "qwen3.8-omni-flash",
            "provider": "dashscope_openai",
            "mock": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_output_tokens": 256,
        },
        "prompts": {
            "version": "p3",
            "notes": "2026-10-28 v3 设计：取消 A/B；实验四~六三个子实验；检测不参与消融",
        },
        "dataset": {
            "hf_repo": "HaitongSUN/prosodic-abx",
            "hf_revision": "d607806619aba678030dac10ea086fb19347205a",
            "config_name": "mandarin_tone_syn",
            "mapping_csv_sha256": "cafebabe",
        },
        "conditions": conditions,
        "item_subset_source": None,
        "task_sources": [],
        "plan": {"n_requests": 0, "n_unique_audio": len(AUDIO_TABLE) * len(VOICES)},
        "counts": {"n_ok": 0, "n_error": 0},
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _write_task_dir(cfg: Config) -> None:
    task_dir = cfg.path("paths.task_dir")
    task_dir.mkdir(parents=True, exist_ok=True)
    groups = _ablation_ids()
    for name, ids in (
        ("ablation_core_items.txt", groups["core"]),
        ("ablation_pairs_items.txt", groups["pairs"]),
    ):
        (task_dir / name).write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    # v3: detection does not take part in ablation.  A stale detection list must be
    # ignored by the report even if an old workspace still contains it.
    detect_ids = [
        str(base["item_id"])
        for base in _base_items()
        if str(base["experiment"]) == "detect"
    ]
    (task_dir / "ablation_detection_items.txt").write_text(
        "\n".join(sorted(detect_ids)) + "\n", encoding="utf-8"
    )
    counts = {
        pair_type: {"n_pairs": 1, "n_instances": 2}
        for pair_type in ("sh_zh", "front_back_nasal_ptr", "tone")
    }
    with (task_dir / "minimal_pair_counts.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=False)
    audio_manifest = pd.DataFrame(
        {
            "audio_id": [
                _audio_id(voice, target) for voice in VOICES for target, *_ in AUDIO_TABLE
            ],
            "voice": [voice for voice in VOICES for _ in AUDIO_TABLE],
            "tone": [tone for _voice in VOICES for _t, _i, _f, tone in AUDIO_TABLE],
            "target": [target for _voice in VOICES for target, *_ in AUDIO_TABLE],
            "sample_rate": 16000,
            "channels": 1,
        }
    )
    manifest_dir = cfg.path("paths.manifest_dir")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    audio_manifest.to_parquet(manifest_dir / "audio.parquet", index=False)


def _build_run(
    tmp_path: Path,
    *,
    run_id: str = "synth-v3",
    write_detection: bool = True,
    write_manifest: bool = True,
    write_ablation: bool = True,
    conditions: tuple[dict[str, object], ...] = CONDITIONS,
    full_grid: bool = False,
) -> tuple[Config, Path, pd.DataFrame]:
    cfg = _make_cfg(tmp_path)
    run_dir = cfg.path("paths.results_dir") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    items = _build_items(_ablation_item_ids(), full_grid=full_grid)
    allowed = {_generation_key(dict(condition)) for condition in conditions}
    items = items[
        items["condition_id"].map(lambda cid: str(cid).split("__", 1)[1]).isin(allowed)
    ].reset_index(drop=True)
    items.to_parquet(run_dir / "items.parquet", index=False)
    if write_detection:
        detection = _build_detection(items)
        if not detection.empty:
            detection.to_parquet(run_dir / "detection_rows.parquet", index=False)
    if write_manifest:
        _write_manifest(run_dir, run_id, [dict(condition) for condition in conditions])
    if write_ablation:
        _write_task_dir(cfg)
    return cfg, run_dir, items


# ---------------------------------------------------------------------------
# Usage / cost fixtures (token accounting, task-4)
# ---------------------------------------------------------------------------

USAGE_PRICING: dict[str, object] = {
    "currency": "CNY",
    "input_per_mtok": 0.8,
    "cache_hit_input_per_mtok": 0.1,
    "output_per_mtok": 2.7,
    "audio_tokens_per_second": 7,
}
#: two costly calls (with cache hits + reasoning), one failed call and one
#: call whose usage has no detail fields; hand-computed costs live below.
USAGE_GENERATIONS: list[dict[str, object]] = [
    {
        "condition_id": "free__think1__clean__json",
        "usage": {
            "prompt_tokens": 500_000,
            "completion_tokens": 50_000,
            "total_tokens": 550_000,
            "prompt_tokens_details": {"cached_tokens": 100_000},
            "completion_tokens_details": {"reasoning_tokens": 20_000},
        },
        "latency_s": 2.0,
        "error": None,
    },
    {
        "condition_id": "free__think1__clean__json",
        "usage": {
            "prompt_tokens": 500_000,
            "completion_tokens": 50_000,
            "total_tokens": 550_000,
            "prompt_tokens_details": {"cached_tokens": 100_000},
            "completion_tokens_details": {"reasoning_tokens": 20_000},
        },
        "latency_s": 2.0,
        "error": None,
    },
    {
        "condition_id": "free__think1__clean__json",
        "usage": None,
        "latency_s": 0.5,
        "error": "timeout",
    },
    {
        "condition_id": "wrong_target__think0__clean__text",
        "usage": {"prompt_tokens": 250_000, "completion_tokens": 10_000},
        "latency_s": 1.5,
        "error": None,
    },
]
USAGE_COUNTS: dict[str, object] = {
    "n_planned": 4,
    "n_calls": 4,
    "n_ok": 3,
    "n_error": 1,
    "prompt_tokens": 1_250_000,
    "cached_tokens": 200_000,
    "completion_tokens": 110_000,
    "reasoning_tokens": 40_000,
    "cost_cny": 1.157,
    "elapsed_s": 6.0,
}


def _write_generations(
    run_dir: Path, records: list[dict[str, object]]
) -> None:
    with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_usage_run(
    tmp_path: Path,
    *,
    run_id: str = "usage-p2",
    budget_cny: float | None = 70.0,
    include_pricing: bool = True,
    include_generations: bool = True,
    counts: dict[str, object] | None = None,
) -> tuple[Config, Path]:
    """A synthetic run you can attach explicit usage / cost fields to."""
    cfg, run_dir, _items = _build_run(tmp_path, run_id=run_id)
    if include_generations:
        _write_generations(run_dir, USAGE_GENERATIONS)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if include_pricing:
        manifest["pricing"] = dict(USAGE_PRICING)
    manifest["budget_cny"] = budget_cny
    manifest["counts"] = dict(USAGE_COUNTS if counts is None else counts)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cfg, run_dir


def _read_csv(run_dir: Path, name: str, out_dir: Path | None = None) -> pd.DataFrame:
    base = out_dir if out_dir is not None else run_dir
    return pd.read_csv(base / name)


def _metric_value(
    table: pd.DataFrame,
    *,
    experiment: str,
    dimension: str,
    metric: str,
    scope: str = "overall",
    **filters: object,
) -> float:
    mask = (
        table["scope"].eq(scope)
        & table["experiment"].eq(experiment)
        & table["dimension"].eq(dimension)
        & table["metric"].eq(metric)
    )
    for column, value in filters.items():
        mask &= table[column].eq(value)
    matched = table[mask]
    assert len(matched) == 1, (
        f"expected exactly one row for {experiment}/{dimension}/{metric}, "
        f"got {matched}"
    )
    return float(matched["value"].iloc[0])


def _main_rows(
    items: pd.DataFrame,
    experiment: str,
    prompt_condition: str | None = None,
    voice: str | None = None,
    pair_type: str | None = None,
) -> list[dict]:
    generation = items["condition_id"].map(lambda cid: str(cid).split("__", 1)[1])
    mask = generation.eq(MAIN_GENERATION_KEY) & items["experiment"].eq(experiment)
    if prompt_condition is not None:
        mask &= items["prompt_condition"].eq(prompt_condition)
    if voice is not None:
        mask &= items["voice"].eq(voice)
    if pair_type is not None:
        mask &= items["pair_type"].eq(pair_type)
    return items[mask].to_dict("records")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_run(tmp_path: Path) -> tuple[Config, Path, pd.DataFrame]:
    return _build_run(tmp_path)


@pytest.fixture()
def report_outputs(synthetic_run: tuple[Config, Path, pd.DataFrame], tmp_path: Path):
    cfg, run_dir, items = synthetic_run
    out_dir = tmp_path / "out"
    artifacts = write_report(cfg, run_dir, out_dir=out_dir, n_boot=40, seed=7)
    return cfg, run_dir, items, out_dir, artifacts


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------


def test_write_report_creates_all_artifacts(report_outputs) -> None:
    _cfg, _run_dir, _items, out_dir, artifacts = report_outputs
    expected = {
        "report.md",
        "metrics.csv",
        "detection.csv",
        "ablation.csv",
        "comparisons.csv",
        "baselines.csv",
        "usage.csv",
        "summary.json",
        "confusion/exp3_initial_overall.csv",
        "confusion/exp3_final_overall.csv",
        "confusion/exp3_tone_overall.csv",
        "confusion/exp3_tone_voice_A.csv",
    }
    assert expected.issubset(set(artifacts))
    for name, path in artifacts.items():
        assert Path(path).exists(), f"artifact {name} missing at {path}"
        assert Path(path).read_text(encoding="utf-8").strip() or name.endswith(".json")
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "实验报告" in report
    assert MAIN_GENERATION_KEY in report
    assert "摘要" in report
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "synth-v3"
    assert summary["main_generation_key"] == MAIN_GENERATION_KEY
    assert summary["n_items"] > 0
    assert summary["n_items"] == summary["n_parse_ok"] + summary["n_parse_failures"]
    confusion = pd.read_csv(out_dir / "confusion" / "exp3_tone_overall.csv", index_col=0)
    assert list(confusion.columns)
    assert confusion.to_numpy().sum() > 0


def test_report_contains_new_design_sections(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, _artifacts = report_outputs
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    # experiment x metric matrix from the v3 design
    assert "实验 × 指标矩阵" in report
    assert "检测实验（target + target_tone）" in report
    assert "2026-10-28 v3 设计更新" in report
    assert "取消 A/B" in report
    # detection semantics are spelled out (positive class = inconsistent)
    assert "正类 = “不一致”" in report
    assert "pred_consistent" in report
    assert '{"consistent": true|false}' in report
    # exp1/exp2 are tone-only; exp4/5/6 only report their own dimension
    table = _read_csv(run_dir, "metrics.csv", out_dir)
    for experiment in ("exp1", "exp2"):
        dims = set(table[table["experiment"].eq(experiment)]["dimension"])
        assert dims == {"tone"}, dims
        assert "consistency" not in dims
    assert set(table[table["experiment"].eq("exp4")]["dimension"]) == {"initial"}
    assert set(table[table["experiment"].eq("exp5")]["dimension"]) == {"final"}
    assert set(table[table["experiment"].eq("exp6")]["dimension"]) == {"tone"}
    # experiments 4/5/6 report all three v3 prompt conditions
    for experiment in ("exp4", "exp5", "exp6"):
        conditions_seen = set(
            table[table["experiment"].eq(experiment)]["prompt_condition"]
        )
        assert {"free", "correct_target", "wrong_target"} <= conditions_seen
    # no A/B task anywhere
    assert "ab" not in set(table["dimension"])
    assert "ab" not in set(table["prompt_condition"])
    # the dedicated detect experiment is reported by detection.csv, not metrics.csv
    assert "detect" not in set(table["experiment"])
    detection = _read_csv(run_dir, "detection.csv", out_dir)
    assert set(detection["subset"]) >= {"main", "initial", "final", "tone"}


# ---------------------------------------------------------------------------
# Consistency with direct metrics.py calls
# ---------------------------------------------------------------------------


def test_metrics_match_direct_metrics_calls(report_outputs) -> None:
    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "metrics.csv", out_dir)

    # experiment 3 free recognition (overall)
    exp3_rows = _main_rows(items, "exp3", "free")
    direct = metrics.recognition_metrics(exp3_rows)
    assert direct["n"] > 0
    for dimension in ("initial", "final", "tone"):
        assert _metric_value(
            table, experiment="exp3", dimension=dimension, metric="accuracy"
        ) == pytest.approx(direct[dimension]["accuracy"])
        assert _metric_value(
            table, experiment="exp3", dimension=dimension, metric="macro_f1"
        ) == pytest.approx(direct[dimension]["macro_f1"])
    assert _metric_value(
        table, experiment="exp3", dimension="joint", metric="accuracy"
    ) == pytest.approx(direct["joint_accuracy"])
    row = table[
        table["scope"].eq("overall")
        & table["experiment"].eq("exp3")
        & table["dimension"].eq("tone")
        & table["metric"].eq("accuracy")
    ].iloc[0]
    assert int(row["n"]) == direct["n"]
    assert int(row["n_failures"]) == direct["n_failures"]

    # experiments 1/2 are tone-only
    for experiment in ("exp1", "exp2"):
        rows = _main_rows(items, experiment)
        direct_tone = metrics.recognition_metrics(rows)["tone"]
        assert _metric_value(
            table, experiment=experiment, dimension="tone", metric="accuracy"
        ) == pytest.approx(direct_tone["accuracy"])
        assert _metric_value(
            table, experiment=experiment, dimension="tone", metric="macro_f1"
        ) == pytest.approx(direct_tone["macro_f1"])

    # experiments 4/5/6: three v3 sub-experiments each, only their own dimension
    for experiment, dimension in (
        ("exp4", "initial"),
        ("exp5", "final"),
        ("exp6", "tone"),
    ):
        for prompt_condition in ("free", "correct_target", "wrong_target"):
            direct = metrics.recognition_metrics(
                _main_rows(items, experiment, prompt_condition)
            )[dimension]
            assert _metric_value(
                table,
                experiment=experiment,
                dimension=dimension,
                metric="accuracy",
                prompt_condition=prompt_condition,
            ) == pytest.approx(direct["accuracy"])
            assert _metric_value(
                table,
                experiment=experiment,
                dimension=dimension,
                metric="macro_f1",
                prompt_condition=prompt_condition,
            ) == pytest.approx(direct["macro_f1"])
        dims = set(table[table["experiment"].eq(experiment)]["dimension"])
        assert dims == {dimension}

    # per-voice split (exp4 free initial)
    direct4_voice = metrics.recognition_metrics(
        _main_rows(items, "exp4", "free", voice="A")
    )["initial"]
    assert _metric_value(
        table,
        scope="voice",
        experiment="exp4",
        dimension="initial",
        metric="accuracy",
        prompt_condition="free",
        voice="A",
    ) == pytest.approx(direct4_voice["accuracy"])

    # per pair_type split (exp6 free tone has a single pair type)
    direct6 = metrics.recognition_metrics(
        _main_rows(items, "exp6", "free", pair_type="tone")
    )["tone"]
    assert _metric_value(
        table,
        scope="pair_type",
        experiment="exp6",
        dimension="tone",
        metric="accuracy",
        prompt_condition="free",
        pair_type="tone",
    ) == pytest.approx(direct6["accuracy"])


def test_detection_matches_direct_metrics(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "detection.csv", out_dir)
    detection = pd.read_parquet(run_dir / "detection_rows.parquet")
    detection = detection[detection["generation_condition"].eq(MAIN_GENERATION_KEY)]
    for subset in ("main", "initial", "final", "tone"):
        rows = detection[detection["detection_subset"].eq(subset)].to_dict("records")
        direct = metrics.detection_metrics(rows)
        for metric_name in (
            "error_precision",
            "error_recall",
            "error_f1",
            "fpr",
            "fnr",
            "accuracy",
            "balanced_accuracy",
        ):
            mask = (
                table["scope"].eq("overall")
                & table["subset"].eq(subset)
                & table["metric"].eq(metric_name)
            )
            matched = table[mask]
            assert len(matched) == 1
            expected = direct[metric_name]
            if expected is None:
                assert pd.isna(matched["value"].iloc[0])
            else:
                assert float(matched["value"].iloc[0]) == pytest.approx(expected)
        overall = table[table["scope"].eq("overall") & table["subset"].eq(subset)].iloc[0]
        assert int(overall["tp"]) == direct["tp"]
        assert int(overall["fp"]) == direct["fp"]
        assert int(overall["tn"]) == direct["tn"]
        assert int(overall["fn"]) == direct["fn"]
        assert int(overall["n_neg"]) == direct["n_neg"]
        assert int(overall["n_pos"]) == direct["n_pos"]

    # baselines: balanced detection chance + tone 4-class chance + A/B 50%
    baselines = _read_csv(run_dir, "baselines.csv", out_dir)
    chance = baselines[
        baselines["scope"].eq("detection")
        & baselines["subset"].eq("main")
        & baselines["baseline"].eq("random")
    ]
    assert len(chance) == 1
    assert float(chance["value"].iloc[0]) == pytest.approx(0.5)
    for experiment in ("exp1", "exp2", "exp3", "exp6"):
        tone_random = baselines[
            baselines["scope"].eq("overall")
            & baselines["experiment"].eq(experiment)
            & baselines["dimension"].eq("tone")
            & baselines["baseline"].eq("random")
        ]
        assert len(tone_random) == 1, experiment
        assert float(tone_random["value"].iloc[0]) == pytest.approx(0.25)
    # v3: no A/B task -> no A/B baseline rows at all
    assert "ab" not in set(baselines["dimension"])


def test_comparisons_use_paired_items(report_outputs) -> None:
    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "comparisons.csv", out_dir)
    assert not table.empty
    assert table["axis"].isin(["thinking", "audio_quality", "output_form"]).all()
    assert set(table["axis"]) == {"thinking", "audio_quality", "output_form"}
    assert table["n_pairs"].gt(0).all()
    for _index, row in table.iterrows():
        assert float(row["diff"]) == pytest.approx(
            float(row["acc_a"]) - float(row["acc_b"]), abs=1e-9
        )
        assert 0.0 <= float(row["p"]) <= 1.0
        if not pd.isna(row["p_holm"]):
            assert 0.0 <= float(row["p_holm"]) <= 1.0
    # v3 families: exp1/2 tone, exp3 four metrics, exp4/5/6 three sub-experiments
    # each (free / correct / wrong); no A/B and no exp1/2 consistency.
    families = set(table["family"])
    expected_pairs = {
        "exp1",
        "exp2",
        "exp3",
    } | {
        f"{experiment}.{prompt_condition}"
        for experiment in ("exp4", "exp5", "exp6")
        for prompt_condition in ("free", "correct_target", "wrong_target")
    }
    assert expected_pairs <= families
    metrics_by_family = {
        family: set(group["metric"]) for family, group in table.groupby("family")
    }
    assert metrics_by_family["exp1"] == {"tone_accuracy"}
    assert metrics_by_family["exp2"] == {"tone_accuracy"}
    assert metrics_by_family["exp3"] == {
        "initial_accuracy",
        "final_accuracy",
        "tone_accuracy",
        "joint_accuracy",
    }
    for experiment, metric_name in (
        ("exp4", "initial_accuracy"),
        ("exp5", "final_accuracy"),
        ("exp6", "tone_accuracy"),
    ):
        for prompt_condition in ("free", "correct_target", "wrong_target"):
            assert metrics_by_family[f"{experiment}.{prompt_condition}"] == {
                metric_name
            }
    # no A/B family/metric anywhere
    assert not any("ab_accuracy" in group for group in metrics_by_family.values())
    assert "ab" not in set(table["dimension"])
    # detection experiment is not part of the ablation-condition comparison
    # (in the v3 minimal/ablation design detect tasks only run on the main
    # condition), so the detect family may legitimately be absent.
    assert "detect" not in families or "detect" in metrics_by_family


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_missing_detection_and_single_condition(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(
        tmp_path,
        run_id="partial-p2",
        write_detection=False,
        write_manifest=True,
        write_ablation=False,
        conditions=(CONDITIONS[0],),
    )
    artifacts = write_report(
        cfg, run_dir, out_dir=tmp_path / "partial_out", n_boot=20, seed=3
    )
    for name in (
        "report.md",
        "metrics.csv",
        "detection.csv",
        "ablation.csv",
        "comparisons.csv",
        "baselines.csv",
        "summary.json",
    ):
        assert name in artifacts
    detection = _read_csv(run_dir, "detection.csv", tmp_path / "partial_out")
    assert list(detection.columns) == list(DETECTION_COLUMNS)
    assert detection.empty
    ablation = _read_csv(run_dir, "ablation.csv", tmp_path / "partial_out")
    assert list(ablation.columns) == list(ABLATION_COLUMNS)
    assert ablation.empty
    comparisons = _read_csv(run_dir, "comparisons.csv", tmp_path / "partial_out")
    assert list(comparisons.columns) == list(COMPARISON_COLUMNS)
    assert comparisons.empty
    report = (tmp_path / "partial_out" / "report.md").read_text(encoding="utf-8")
    assert "跳过检测分析" in report
    summary = json.loads(
        (tmp_path / "partial_out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["warnings"]
    metrics_table = _read_csv(run_dir, "metrics.csv", tmp_path / "partial_out")
    assert not metrics_table.empty


def test_no_manifest_no_detection_small_run(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    run_dir = cfg.path("paths.results_dir") / "runs" / "tiny-p2"
    run_dir.mkdir(parents=True)
    items = _build_items(_ablation_item_ids())
    tiny = items[
        items["condition_id"].eq(f"free__{MAIN_GENERATION_KEY}")
        & items["experiment"].eq("exp3")
    ].head(2)
    tiny.to_parquet(run_dir / "items.parquet", index=False)
    artifacts = write_report(cfg, run_dir, out_dir=tmp_path / "tiny_out", n_boot=10, seed=1)
    assert "report.md" in artifacts
    report = (tmp_path / "tiny_out" / "report.md").read_text(encoding="utf-8")
    assert "主条件" in report
    summary = json.loads(
        (tmp_path / "tiny_out" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["n_items"] == 2
    assert summary["warnings"]


def test_run_id_resolution_and_default_out_dir(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="resolve-p2")
    artifacts = write_report(cfg, "resolve-p2", n_boot=15, seed=2)
    assert "report.md" in artifacts
    assert (run_dir / "report.md").exists()
    assert (run_dir / "summary.json").exists()
    assert Path(artifacts["report.md"]).resolve() == (run_dir / "report.md").resolve()


# ---------------------------------------------------------------------------
# CSV stability / reproducibility
# ---------------------------------------------------------------------------


def test_csv_columns_are_stable(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, _artifacts = report_outputs
    expected_metrics = [
        "scope",
        "experiment",
        "prompt_condition",
        "condition_id",
        "generation_condition",
        "voice",
        "pair_type",
        "subset",
        "dimension",
        "metric",
        "value",
        "ci_lo",
        "ci_hi",
        "n",
        "n_failures",
    ]
    expected_detection = [
        "scope",
        "generation_condition",
        "voice",
        "subset",
        "dimension",
        "metric",
        "value",
        "ci_lo",
        "ci_hi",
        "n",
        "n_failures",
        "tp",
        "fp",
        "tn",
        "fn",
        "n_neg",
        "n_pos",
    ]
    expected_ablation = [
        "group",
        "compared_with",
        "is_main",
        "condition_id",
        "generation_condition",
        "experiment",
        "prompt_condition",
        "dimension",
        "metric",
        "value",
        "n",
        "n_failures",
    ]
    expected_comparisons = [
        "family",
        "metric",
        "dimension",
        "axis",
        "condition_a",
        "condition_b",
        "n_pairs",
        "acc_a",
        "acc_b",
        "diff",
        "diff_lo",
        "diff_hi",
        "mcnemar_b",
        "mcnemar_c",
        "p",
        "p_holm",
        "method",
    ]
    expected_baselines = [
        "scope",
        "experiment",
        "subset",
        "dimension",
        "baseline",
        "value",
        "label",
        "n",
    ]
    expected_usage = [
        "scope",
        "condition_id",
        "generation_condition",
        "n_calls",
        "n_errors",
        "prompt_tokens",
        "cached_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "input_cost_cny",
        "output_cost_cny",
        "total_cost_cny",
        "avg_latency_s",
    ]
    assert _read_csv(run_dir, "metrics.csv", out_dir).columns.tolist() == expected_metrics
    assert _read_csv(run_dir, "detection.csv", out_dir).columns.tolist() == expected_detection
    assert _read_csv(run_dir, "ablation.csv", out_dir).columns.tolist() == expected_ablation
    assert _read_csv(run_dir, "comparisons.csv", out_dir).columns.tolist() == expected_comparisons
    assert _read_csv(run_dir, "baselines.csv", out_dir).columns.tolist() == expected_baselines
    assert _read_csv(run_dir, "usage.csv", out_dir).columns.tolist() == expected_usage
    assert list(METRICS_COLUMNS) == expected_metrics
    assert list(DETECTION_COLUMNS) == expected_detection
    assert list(ABLATION_COLUMNS) == expected_ablation
    assert list(COMPARISON_COLUMNS) == expected_comparisons
    assert list(BASELINE_COLUMNS) == expected_baselines
    assert list(USAGE_COLUMNS) == expected_usage


def test_seed_makes_bootstrap_reproducible(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="repro-p2")
    first_dir = tmp_path / "repro_a"
    second_dir = tmp_path / "repro_b"
    write_report(cfg, run_dir, out_dir=first_dir, n_boot=25, seed=123)
    write_report(cfg, run_dir, out_dir=second_dir, n_boot=25, seed=123)
    for name in (
        "metrics.csv",
        "detection.csv",
        "comparisons.csv",
        "baselines.csv",
        "ablation.csv",
        "usage.csv",
    ):
        assert (first_dir / name).read_text(encoding="utf-8") == (
            second_dir / name
        ).read_text(encoding="utf-8"), f"{name} is not reproducible"


def test_ablation_has_two_v3_groups(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "ablation.csv", out_dir)
    assert not table.empty
    assert set(table["group"]) == {"core", "pairs", "parse_core", "parse_pairs"}
    assert set(table["compared_with"]) >= {
        "think0__clean__json",
        "think1__noise_20db__json",
        "think1__clean__text",
    }
    # main and ablation rows are measured on the same item subset
    for compared_with, group in table.groupby("compared_with"):
        assert group["is_main"].any(), compared_with
        main_rows = group[group["is_main"]]
        assert (main_rows["generation_condition"] == MAIN_GENERATION_KEY).all()
    # core group: experiments 1-3 only (tone-only experiments 1/2 included)
    core = table[table["group"].eq("core")]
    assert {"exp1", "exp2", "exp3"} <= set(core["experiment"])
    # pairs group: exp4/5/6 × the three v3 sub-experiments / prompt conditions
    pairs = table[table["group"].eq("pairs")]
    assert {"exp4", "exp5", "exp6"} <= set(pairs["experiment"])
    assert {"free", "correct_target", "wrong_target"} <= set(
        pairs["prompt_condition"]
    )
    assert set(pairs["dimension"]) == {"initial", "final", "tone"}
    # v3: the detection experiment does not take part in ablation -- a stale
    # ablation_detection_items.txt must not leak into ablation.csv
    assert "detection" not in set(table["group"])
    assert "detect" not in set(table["experiment"])
    assert (table["metric"] == "parse_failure_rate").any()


def test_ablation_ignores_stale_detection_subset_file(tmp_path: Path) -> None:
    """A leftover ablation_detection_items.txt must not create a detection group."""
    cfg, run_dir, _items = _build_run(tmp_path, run_id="stale-detect-v3")
    task_dir = cfg.path("paths.task_dir")
    stale = task_dir / "ablation_detection_items.txt"
    assert stale.exists()  # written by the fixture on purpose
    out_dir = tmp_path / "stale_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)
    table = _read_csv(run_dir, "ablation.csv", out_dir)
    assert "detection" not in set(table["group"])


# ---------------------------------------------------------------------------
# Statistics plumbing
# ---------------------------------------------------------------------------


def test_bootstrap_ci_matches_direct_stats_call(report_outputs) -> None:
    """The CI columns must equal a direct ``stats.bootstrap_ci`` on identical rows."""
    from pinyin_eval import stats

    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "metrics.csv", out_dir)

    generation = items["condition_id"].map(lambda cid: str(cid).split("__", 1)[1])
    main = items[generation.eq(MAIN_GENERATION_KEY)]
    analysis_mask = (
        main["experiment"].isin(["exp1", "exp2", "exp3"])
        & main["prompt_condition"].isin(["free", "correct_target", "wrong_target"])
    ) | (
        main["experiment"].isin(["exp4", "exp5", "exp6"])
        & main["prompt_condition"].isin(
            ["free", "correct_target", "wrong_target"]
        )
    )
    analysis = main[analysis_mask]
    exp3_rows = analysis[
        analysis["experiment"].eq("exp3") & analysis["prompt_condition"].eq("free")
    ].to_dict("records")

    def metric_fn(rows):
        values = metrics.recognition_metrics(rows)
        return {
            "tone_accuracy": values["tone"]["accuracy"],
            "tone_macro_f1": values["tone"]["macro_f1"],
            "joint_accuracy": values["joint_accuracy"],
        }

    direct = stats.bootstrap_ci(
        exp3_rows,
        metric_fn,
        cluster_key="cluster_id",
        n_boot=40,
        alpha=0.05,
        seed=7,
        stratify_key="voice",
    )
    for metric_name, keys in (
        ("tone_accuracy", ("tone", "accuracy")),
        ("tone_macro_f1", ("tone", "macro_f1")),
        ("joint_accuracy", ("joint", "accuracy")),
    ):
        mask = (
            table["scope"].eq("overall")
            & table["experiment"].eq("exp3")
            & table["dimension"].eq(keys[0])
            & table["metric"].eq(keys[1])
        )
        row = table[mask].iloc[0]
        assert float(row["ci_lo"]) == pytest.approx(direct[metric_name]["lo"], abs=1e-9)
        assert float(row["ci_hi"]) == pytest.approx(direct[metric_name]["hi"], abs=1e-9)
        assert float(row["value"]) == pytest.approx(
            direct[metric_name]["estimate"], abs=1e-9
        )

    # detection CI: main subset
    detection = pd.read_parquet(run_dir / "detection_rows.parquet")
    detection = detection[detection["generation_condition"].eq(MAIN_GENERATION_KEY)]
    subset_rows = detection[detection["detection_subset"].eq("main")].to_dict("records")
    detection_table = _read_csv(run_dir, "detection.csv", out_dir)

    def detection_fn(rows):
        values = metrics.detection_metrics(rows)
        return {
            name: values[name]
            for name in (
                "error_precision",
                "error_recall",
                "error_f1",
                "fpr",
                "fnr",
                "accuracy",
                "balanced_accuracy",
            )
        }

    direct_detection = stats.bootstrap_ci(
        subset_rows,
        detection_fn,
        cluster_key="cluster_id",
        n_boot=40,
        alpha=0.05,
        seed=7,
        stratify_key="voice",
    )
    for metric_name in ("error_f1", "fpr", "balanced_accuracy"):
        mask = (
            detection_table["scope"].eq("overall")
            & detection_table["subset"].eq("main")
            & detection_table["metric"].eq(metric_name)
        )
        row = detection_table[mask].iloc[0]
        assert float(row["ci_lo"]) == pytest.approx(
            direct_detection[metric_name]["lo"], abs=1e-9
        )
        assert float(row["ci_hi"]) == pytest.approx(
            direct_detection[metric_name]["hi"], abs=1e-9
        )


def test_comparisons_consistency_correct_for_both_gold_values(tmp_path: Path) -> None:
    """Regression: detect claims with gold True/False must both count as correct.

    Perfect-mock-style data: every detect task answers ``consistent_gold`` itself
    (so both the consistent negatives and the inconsistent positives are correct),
    and experiments 1/2 answer the true tone.  ``comparisons.csv`` must report
    1.0 accuracy for the ``detect``/``exp1``/``exp2`` families -- an old formula
    ``gold & pred & (gold == pred)`` scored every gold=False claim as wrong.
    """
    cfg, run_dir, items = _build_run(tmp_path, run_id="consistency-v3", full_grid=True)
    perfect = items.copy()
    detect_mask = perfect["experiment"].eq("detect")
    perfect.loc[detect_mask, "pred_consistent"] = perfect.loc[
        detect_mask, "consistent_gold"
    ].astype(bool)
    for experiment in ("exp1", "exp2"):
        mask = perfect["experiment"].eq(experiment)
        perfect.loc[mask, "pred_tone"] = perfect.loc[mask, "gold_tone"]
    perfect.loc[:, "parse_status"] = "ok"
    perfect.to_parquet(run_dir / "items.parquet", index=False)
    out_dir = tmp_path / "consistency_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=2)
    table = _read_csv(run_dir, "comparisons.csv", out_dir)
    for family, metric_name in (
        ("detect", "consistency_accuracy"),
        ("exp1", "tone_accuracy"),
        ("exp2", "tone_accuracy"),
    ):
        rows = table[table["family"].eq(family) & table["metric"].eq(metric_name)]
        assert not rows.empty, family
        for value in rows["acc_a"]:
            assert float(value) == pytest.approx(1.0)
        for value in rows["acc_b"]:
            assert float(value) == pytest.approx(1.0)
        for value in rows["diff"]:
            assert float(value) == pytest.approx(0.0, abs=1e-9)
    detect_rows = table[table["family"].eq("detect")]
    assert set(detect_rows["metric"]) == {"consistency_accuracy"}


def test_detection_notes_unbalanced_typed_subsets(tmp_path: Path) -> None:
    """Typed detection subsets that are not counterbalanced are flagged in the report."""
    cfg, run_dir, _items = _build_run(tmp_path, run_id="imbalance-p2")
    detection = pd.read_parquet(run_dir / "detection_rows.parquet")
    positives = detection[
        detection["detection_role"].eq("pos")
        & detection["detection_subset"].isin(["initial", "final", "tone"])
    ]
    keep = positives.groupby("detection_subset", group_keys=False).head(3)
    balanced_main = detection[detection["detection_subset"].eq("main")]
    pruned = pd.concat(
        [detection.drop(positives.index), keep, balanced_main]
    ).reset_index(drop=True)
    pruned.to_parquet(run_dir / "detection_rows.parquet", index=False)
    out_dir = tmp_path / "imbalance_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=4)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "不配平" in report


# ---------------------------------------------------------------------------
# Optional pieces of the report
# ---------------------------------------------------------------------------


def test_appendix_and_confusion_sections(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="appendix-p2")
    out_dir = tmp_path / "appendix_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=12, seed=5)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "最小对计数" in report
    assert "解析失败" in report
    assert "模型 / 参数 / 版本" in report
    assert "混淆矩阵" in report
    assert "prompt 版本" in report
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    facts = summary["data_facts"]
    assert facts is not None
    assert facts["audio"] is not None
    assert facts["audio"]["n_audio"] == len(AUDIO_TABLE) * len(VOICES)
    assert "tone" in facts["minimal_pairs"]
    assert facts["detection_composition"]


# ---------------------------------------------------------------------------
# Cost & token usage (task-4)
# ---------------------------------------------------------------------------


def test_usage_cost_hand_computed(tmp_path: Path) -> None:
    """Token totals and CNY cost (including the cache-hit discount) are exact."""
    cfg, run_dir = _build_usage_run(tmp_path)
    out_dir = tmp_path / "usage_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)

    usage = pd.read_csv(out_dir / "usage.csv")
    assert usage.columns.tolist() == list(USAGE_COLUMNS)
    assert set(usage["scope"]) == {"overall", "per_condition"}
    overall = usage[usage["scope"].eq("overall")].iloc[0]
    # 2 calls * (prompt 500k, cached 100k, completion 50k) + 1 failed call
    # + 1 call (prompt 250k, completion 10k)
    assert int(overall["n_calls"]) == 4
    assert int(overall["n_errors"]) == 1
    assert int(overall["prompt_tokens"]) == 1_250_000
    assert int(overall["cached_tokens"]) == 200_000
    assert int(overall["completion_tokens"]) == 110_000
    assert int(overall["reasoning_tokens"]) == 40_000
    # input: ((1.25M - 0.2M) * 0.8 + 0.2M * 0.1) / 1M = 0.86 CNY
    assert float(overall["input_cost_cny"]) == pytest.approx(0.86)
    # output: 110_000 * 2.7 / 1M = 0.297 CNY
    assert float(overall["output_cost_cny"]) == pytest.approx(0.297)
    assert float(overall["total_cost_cny"]) == pytest.approx(1.157)
    # latencies: (2 + 2 + 0.5 + 1.5) / 4 calls = 1.5 s
    assert float(overall["avg_latency_s"]) == pytest.approx(1.5)

    per_condition = usage[usage["scope"].eq("per_condition")].set_index("condition_id")
    assert set(per_condition.index) == {
        "free__think1__clean__json",
        "wrong_target__think0__clean__text",
    }
    free = per_condition.loc["free__think1__clean__json"]
    assert free["generation_condition"] == "think1__clean__json"
    assert int(free["n_calls"]) == 3
    assert int(free["n_errors"]) == 1
    assert int(free["prompt_tokens"]) == 1_000_000
    assert int(free["cached_tokens"]) == 200_000
    assert float(free["input_cost_cny"]) == pytest.approx(0.66)
    assert float(free["output_cost_cny"]) == pytest.approx(0.27)
    assert float(free["total_cost_cny"]) == pytest.approx(0.93)
    ab = per_condition.loc["wrong_target__think0__clean__text"]
    assert ab["generation_condition"] == "think0__clean__text"
    assert int(ab["n_calls"]) == 1
    assert float(ab["input_cost_cny"]) == pytest.approx(0.2)
    assert float(ab["output_cost_cny"]) == pytest.approx(0.027)
    assert float(ab["total_cost_cny"]) == pytest.approx(0.227)

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["usage"]["source"] == "generations.jsonl"
    assert summary["usage"]["n_calls"] == 4
    assert summary["usage"]["n_ok"] == 3
    assert summary["usage"]["cached_tokens"] == 200_000
    assert summary["usage"]["n_usage_records"] == 3
    cost = summary["cost"]
    assert cost["total_cost_cny"] == pytest.approx(1.157)
    assert cost["budget_status"] == "within_budget"
    assert cost["budget_used_pct"] == pytest.approx(
        round(1.157 / 70.0 * 100.0, 4), abs=1e-3
    )
    assert cost["budget_remaining_cny"] == pytest.approx(70.0 - 1.157)
    assert set(cost["per_generation_condition"]) == {
        "think1__clean__json",
        "think0__clean__text",
    }
    assert cost["per_generation_condition"]["think1__clean__json"][
        "total_cost_cny"
    ] == pytest.approx(0.93)
    assert cost["per_generation_condition"]["think0__clean__text"][
        "total_cost_cny"
    ] == pytest.approx(0.227)

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## 6. 成本与用量" in report
    assert "¥1.1570" in report
    assert "在预算内" in report
    assert "缓存命中 200,000" in report
    assert "逐生成条件" in report


def test_usage_budget_overrun(tmp_path: Path) -> None:
    cfg, run_dir = _build_usage_run(tmp_path, run_id="overbudget-p2", budget_cny=1.0)
    out_dir = tmp_path / "overbudget_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    cost = summary["cost"]
    assert cost["budget_status"] == "over_budget"
    assert cost["budget_used_pct"] == pytest.approx(1.157 / 1.0 * 100.0, abs=1e-6)
    assert cost["budget_remaining_cny"] == pytest.approx(1.0 - 1.157)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "超预算" in report
    assert "超出 ¥0.1570" in report


def test_usage_degrades_without_pricing(tmp_path: Path) -> None:
    """Tokens are still reported but the cost is marked unavailable."""
    cfg, run_dir = _build_usage_run(
        tmp_path, run_id="degrade-pricing", include_pricing=False, counts={}
    )
    cfg = cfg.with_overrides({"pricing": None})
    out_dir = tmp_path / "degrade_pricing_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)
    usage = pd.read_csv(out_dir / "usage.csv")
    assert usage.columns.tolist() == list(USAGE_COLUMNS)
    overall = usage[usage["scope"].eq("overall")].iloc[0]
    assert int(overall["n_calls"]) == 4
    assert int(overall["prompt_tokens"]) == 1_250_000
    assert pd.isna(overall["total_cost_cny"])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["cost"]["total_cost_cny"] is None
    assert summary["cost"]["status"] == "no_pricing"
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "无法计算成本" in report
    assert any("pricing" in note for note in summary["cost"]["notes"])


def test_usage_degrades_without_usage_data(tmp_path: Path) -> None:
    """No generations.jsonl and no token counts -> graceful empty usage report."""
    cfg, run_dir = _build_usage_run(
        tmp_path,
        run_id="degrade-usage",
        include_generations=False,
        counts={},
    )
    out_dir = tmp_path / "degrade_usage_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)
    usage = pd.read_csv(out_dir / "usage.csv")
    assert usage.columns.tolist() == list(USAGE_COLUMNS)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["usage"]["source"] is None
    assert summary["cost"]["status"] == "no_data"
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "没有可用的用量数据" in report


def test_usage_falls_back_to_manifest_counts(tmp_path: Path) -> None:
    """Without generations.jsonl the runner's manifest counts are used."""
    cfg, run_dir = _build_usage_run(
        tmp_path, run_id="counts-p2", include_generations=False
    )
    out_dir = tmp_path / "counts_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=3)
    usage = pd.read_csv(out_dir / "usage.csv")
    overall = usage[usage["scope"].eq("overall")].iloc[0]
    assert int(overall["n_calls"]) == 4
    assert int(overall["n_errors"]) == 1
    assert int(overall["prompt_tokens"]) == 1_250_000
    # counts carries cached tokens too, so the cache discount is applied
    assert float(overall["input_cost_cny"]) == pytest.approx(0.86)
    assert float(overall["total_cost_cny"]) == pytest.approx(1.157)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["usage"]["source"] == "manifest_counts"
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "manifest counts" in report


def test_usage_section_present_in_existing_run_report(report_outputs) -> None:
    """Every report (even without generations.jsonl) has the usage section + CSV."""
    _cfg, run_dir, _items, out_dir, artifacts = report_outputs
    assert "usage.csv" in artifacts
    usage = _read_csv(run_dir, "usage.csv", out_dir)
    assert usage.columns.tolist() == list(USAGE_COLUMNS)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "## 6. 成本与用量" in report
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "usage" in summary
    assert "cost" in summary
