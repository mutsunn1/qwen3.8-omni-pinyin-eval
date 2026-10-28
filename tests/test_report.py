"""Tests for :mod:`pinyin_eval.report` (contract section 8).

The tests build a small synthetic scored run in a temporary directory (items /
detection rows / run manifest), run :func:`pinyin_eval.report.write_report` on it
and check that

* every expected artifact is created and the returned paths exist;
* the numbers in ``metrics.csv`` / ``detection.csv`` / ``baselines.csv`` equal
  what calling :mod:`pinyin_eval.metrics` directly on the same rows returns;
* runs with missing pieces (no detection rows, one generation condition, no
  manifest, ...) degrade gracefully instead of crashing;
* the CSV column order is stable;
* bootstrap results are reproducible for a fixed seed.

No file under ``data/`` or ``results/`` is read or written; everything lives
under ``tmp_path``.
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
    write_report,
)

from .conftest import CONFIG_PATH, MAPPING_CSV

# ---------------------------------------------------------------------------
# Synthetic dataset description
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
        "audio_id": audio_id,
        "cluster_id": audio_id,
        "voice": voice,
        "gold_initial": initial,
        "gold_final": final,
        "gold_tone": tone,
        "given_target": None,
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


def _base_items() -> list[dict[str, object]]:
    """One row per task item, independent of the generation condition."""
    items: list[dict[str, object]] = []
    for voice in VOICES:
        for target, _initial, _final, _tone in AUDIO_TABLE:
            audio_id = _audio_id(voice, target)
            items.append(
                _base_row(
                    item_id=f"exp1|correct_target|{audio_id}",
                    experiment="exp1",
                    prompt_condition="correct_target",
                    voice=voice,
                    target=target,
                    given_target=target,
                    consistent_gold=True,
                )
            )
            for variant in ("initial", "final", "tone"):
                items.append(
                    _base_row(
                        item_id=f"exp2|wrong_target|{audio_id}|{variant}0",
                        experiment="exp2",
                        prompt_condition="wrong_target",
                        voice=voice,
                        target=target,
                        given_target=target,
                        consistent_gold=False,
                        variant_type=variant,
                        variant_index=0,
                    )
                )
            items.append(
                _base_row(
                    item_id=f"exp3|free|{audio_id}",
                    experiment="exp3",
                    prompt_condition="free",
                    voice=voice,
                    target=target,
                )
            )
        for experiment, pair_type, side_a, side_b in PAIR_SPECS:
            tone = _audio_entry(side_a)[2]
            pair_id = f"{experiment}:{side_a}/{side_b}:{voice}:{tone}"
            for side, target in (("A", side_a), ("B", side_b)):
                audio_id = _audio_id(voice, target)
                for prompt_condition in ("free", "ab_choice"):
                    alternatives = (
                        [f"{_audio_entry(side_a)[0]}{tone}", f"{_audio_entry(side_b)[0]}{tone}"]
                        if prompt_condition == "ab_choice"
                        else None
                    )
                    correct_choice = ("A" if tone % 2 == 1 else "B") if alternatives else None
                    items.append(
                        _base_row(
                            item_id=f"{experiment}|{prompt_condition}|{audio_id}|{pair_id}",
                            experiment=experiment,
                            prompt_condition=prompt_condition,
                            voice=voice,
                            target=target,
                            pair_id=pair_id,
                            pair_type=pair_type,
                            pair_side=side,
                            alternatives=alternatives,
                            correct_choice=correct_choice,
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
    experiment = str(base["experiment"])
    prompt_condition = str(base["prompt_condition"])
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
    if experiment == "exp1":
        tone_ok = (index + salt) % 2 == 0
        row["pred_tone"] = gold_tone if tone_ok else (gold_tone % 4) + 1
        row["pred_consistent"] = (index + salt) % 3 != 0
    elif experiment == "exp2":
        tone_ok = (index + salt) % 2 == 1
        row["pred_tone"] = gold_tone if tone_ok else (gold_tone % 4) + 1
        # correct detection of an inconsistent claim => pred_consistent == False
        row["pred_consistent"] = False if (index + salt) % 4 != 0 else True
    elif experiment == "exp3":
        row["pred_initial"] = (
            gold_initial if (index + salt) % 3 != 0 else _alternate(gold_initial)
        )
        row["pred_final"] = (
            gold_final if (index + salt) % 4 != 0 else _alternate(gold_final)
        )
        row["pred_tone"] = gold_tone if (index + salt) % 5 != 0 else (gold_tone % 4) + 1
    elif experiment in ("exp4", "exp5", "exp6") and prompt_condition == "free":
        dimension = {"exp4": "initial", "exp5": "final", "exp6": "tone"}[experiment]
        ok = (index + salt) % 2 == 0
        if dimension == "initial":
            row["pred_initial"] = gold_initial if ok else _alternate(gold_initial)
        elif dimension == "final":
            row["pred_final"] = gold_final if ok else _alternate(gold_final)
        else:
            row["pred_tone"] = gold_tone if ok else (gold_tone % 4) + 1
    elif prompt_condition == "ab_choice":
        correct_choice = str(base["correct_choice"])
        choice_ok = (index + salt) % 2 == 0
        if choice_ok:
            row["pred_choice"] = correct_choice
        else:
            row["pred_choice"] = "B" if correct_choice == "A" else "A"
    return row


def _ablation_item_ids() -> set[str]:
    subset_audios = {
        _audio_id(voice, target) for voice in VOICES for target in ("zhang1", "zang1")
    }
    ids: set[str] = set()
    for base in _base_items():
        item_id = str(base["item_id"])
        if str(base["experiment"]) in ("exp1", "exp2", "exp3"):
            if str(base["audio_id"]) in subset_audios:
                ids.add(item_id)
        else:
            ids.add(item_id)
    return ids


def _build_items(ablation_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        generation = _generation_key(dict(condition))
        is_main = generation == MAIN_GENERATION_KEY
        salt = SALT[generation]
        for base in _base_items():
            item_id = str(base["item_id"])
            if not is_main and item_id not in ablation_ids:
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
                row.update(
                    {"parse_status": "parse_error", "raw_response": "{bad json"}
                )
                for column in (
                    "pred_initial",
                    "pred_final",
                    "pred_tone",
                    "pred_consistent",
                    "pred_choice",
                ):
                    row[column] = None
            if not is_main and item_id == "exp1|correct_target|ZH_TTS_A_zhang1":
                row.update({"parse_status": "refused", "pred_consistent": None,
                            "pred_tone": None, "raw_response": "拒绝回答"})
            rows.append(row)
    return pd.DataFrame(rows)


def _subset_audios() -> set[str]:
    return {
        _audio_id(voice, target) for voice in VOICES for target in ("zhang1", "zang1")
    }


def _build_detection(items: pd.DataFrame) -> pd.DataFrame:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in items.to_dict("records"):
        lookup[(str(row["condition_id"]).split("__", 1)[1], str(row["item_id"]))] = row
    generation_conditions = [MAIN_GENERATION_KEY] + [
        _generation_key(dict(condition))
        for condition in CONDITIONS
        if _generation_key(dict(condition)) != MAIN_GENERATION_KEY
    ]
    subset_audios = _subset_audios()
    rows: list[dict[str, object]] = []
    for generation in generation_conditions:
        is_main = generation == MAIN_GENERATION_KEY
        salt = SALT[generation]
        for voice in VOICES:
            for target, _initial, _final, _tone in AUDIO_TABLE:
                audio_id = _audio_id(voice, target)
                if not is_main and audio_id not in subset_audios:
                    continue
                index = AUDIO_INDEX[audio_id]
                negative_id = f"exp1|correct_target|{audio_id}"
                positive_ids = {
                    "main": f"exp2|wrong_target|{audio_id}|tone0",
                    "initial": f"exp2|wrong_target|{audio_id}|initial0",
                    "final": f"exp2|wrong_target|{audio_id}|final0",
                    "tone": f"exp2|wrong_target|{audio_id}|tone0",
                }
                for subset, positive_id in positive_ids.items():
                    negative = dict(lookup[(generation, negative_id)])
                    negative.update(
                        {
                            "detection_role": "neg",
                            "detection_subset": subset,
                            "generation_condition": generation,
                            "pred_consistent": (index + salt) % 3 != 0,
                            "parse_status": "ok",
                        }
                    )
                    positive = dict(lookup[(generation, positive_id)])
                    positive.update(
                        {
                            "detection_role": "pos",
                            "detection_subset": subset,
                            "generation_condition": generation,
                            "pred_consistent": not ((index + salt) % 4 != 0),
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
        "report": {"per_voice": True, "confusion_matrices": True, "mcnemar": True,
                   "multiple_comparison": "holm"},
    }
    return Config.load(CONFIG_PATH, repo_root=tmp_path, overrides=overrides)


def _write_manifest(run_dir: Path, run_id: str, conditions: list[dict[str, object]]) -> None:
    manifest = {
        "run_id": run_id,
        "created_at": "2026-10-28T15:00:00+08:00",
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
        "prompts": {"version": "p1", "notes": "synthetic test"},
        "dataset": {
            "hf_repo": "HaitongSUN/prosodic-abx",
            "hf_revision": "d607806619aba678030dac10ea086fb19347205a",
            "config_name": "mandarin_tone_syn",
            "mapping_csv_sha256": "cafebabe",
        },
        "conditions": conditions,
        "conditions_arg": None,
        "item_subset_source": None,
        "task_sources": [],
        "plan": {"n_requests": 0, "n_unique_audio": len(AUDIO_TABLE) * len(VOICES)},
        "counts": {"n_ok": 0, "n_error": 0},
    }
    with (run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def _write_task_dir(cfg: Config, items: pd.DataFrame) -> None:
    task_dir = cfg.path("paths.task_dir")
    task_dir.mkdir(parents=True, exist_ok=True)
    ablation_ids = _ablation_item_ids()
    core_ids = sorted(
        item_id
        for item_id in ablation_ids
        if item_id.startswith(("exp1|", "exp2|", "exp3|"))
    )
    pair_ids = sorted(
        item_id
        for item_id in ablation_ids
        if item_id.startswith(("exp4|", "exp5|", "exp6|"))
    )
    (task_dir / "ablation_core_items.txt").write_text(
        "\n".join(core_ids) + "\n", encoding="utf-8"
    )
    (task_dir / "ablation_pairs_items.txt").write_text(
        "\n".join(pair_ids) + "\n", encoding="utf-8"
    )
    counts = {
        pair_type: {"n_pairs": 1, "n_instances": 2} for pair_type in
        ("sh_zh", "front_back_nasal_ptr", "tone")
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
    run_id: str = "synth-001",
    write_detection: bool = True,
    write_manifest: bool = True,
    write_ablation: bool = True,
    conditions: tuple[dict[str, object], ...] = CONDITIONS,
) -> tuple[Config, Path, pd.DataFrame]:
    cfg = _make_cfg(tmp_path)
    run_dir = cfg.path("paths.results_dir") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    items = _build_items(_ablation_item_ids())
    # keep only configured conditions
    allowed = {_generation_key(dict(condition)) for condition in conditions}
    items = items[
        items["condition_id"].map(lambda cid: str(cid).split("__", 1)[1]).isin(allowed)
    ].reset_index(drop=True)
    items.to_parquet(run_dir / "items.parquet", index=False)
    if write_detection:
        detection = _build_detection(items)
        detection.to_parquet(run_dir / "detection_rows.parquet", index=False)
    if write_manifest:
        _write_manifest(run_dir, run_id, [dict(condition) for condition in conditions])
    if write_ablation:
        _write_task_dir(cfg, items)
    return cfg, run_dir, items


def _read_csv(run_dir: Path, name: str, out_dir: Path | None = None) -> pd.DataFrame:
    base = out_dir if out_dir is not None else run_dir
    return pd.read_csv(base / name)


def _metric_value(
    table: pd.DataFrame, *, experiment: str, dimension: str, metric: str,
    scope: str = "overall", **filters: object,
) -> float:
    mask = (
        table["scope"].eq(scope)
        & table["experiment"].eq(experiment)
        & table["dimension"].eq(dimension)
        & table["metric"].eq(metric)
    )
    for column, value in filters.items():
        mask &= table[column].eq(value)
    rows = table[mask]
    assert len(rows) == 1, f"expected exactly one row for {experiment}/{dimension}/{metric}, got {rows}"
    return float(rows["value"].iloc[0])


def _main_rows(items: pd.DataFrame, experiment: str, prompt_condition: str | None = None,
               voice: str | None = None, pair_type: str | None = None) -> list[dict]:
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
def report_outputs(
    synthetic_run: tuple[Config, Path, pd.DataFrame], tmp_path: Path
):
    cfg, run_dir, items = synthetic_run
    out_dir = tmp_path / "out"
    artifacts = write_report(cfg, run_dir, out_dir=out_dir, n_boot=40, seed=7)
    return cfg, run_dir, items, out_dir, artifacts


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------


def test_write_report_creates_all_artifacts(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, artifacts = report_outputs
    expected = {
        "report.md",
        "metrics.csv",
        "detection.csv",
        "ablation.csv",
        "comparisons.csv",
        "baselines.csv",
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
    assert summary["run_id"] == "synth-001"
    assert summary["main_generation_key"] == MAIN_GENERATION_KEY
    assert summary["n_items"] > 0
    assert summary["n_items"] == summary["n_parse_ok"] + summary["n_parse_failures"]
    # confusion matrices are readable and square-ish (gold x pred)
    confusion = pd.read_csv(out_dir / "confusion" / "exp3_tone_overall.csv", index_col=0)
    assert list(confusion.columns)
    assert confusion.to_numpy().sum() > 0


# ---------------------------------------------------------------------------
# Consistency with direct metrics.py calls
# ---------------------------------------------------------------------------


def test_metrics_match_direct_metrics_calls(report_outputs) -> None:
    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "metrics.csv", out_dir)

    # exp3 free recognition (overall)
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

    # exp1 / exp2 tone accuracy
    for experiment in ("exp1", "exp2"):
        rows = _main_rows(items, experiment)
        direct = metrics.recognition_metrics(rows)["tone"]
        assert _metric_value(
            table, experiment=experiment, dimension="tone", metric="accuracy"
        ) == pytest.approx(direct["accuracy"])

    # exp4 free initial accuracy and per-voice split
    exp4_rows = _main_rows(items, "exp4", "free")
    direct4 = metrics.recognition_metrics(exp4_rows)["initial"]
    assert _metric_value(
        table, experiment="exp4", dimension="initial", metric="accuracy"
    ) == pytest.approx(direct4["accuracy"])
    direct4_voice = metrics.recognition_metrics(
        _main_rows(items, "exp4", "free", voice="A")
    )["initial"]
    assert _metric_value(
        table, scope="voice", experiment="exp4", dimension="initial",
        metric="accuracy", voice="A",
    ) == pytest.approx(direct4_voice["accuracy"])

    # pair_type split (experiment 6 has a single pair type)
    direct6 = metrics.recognition_metrics(
        _main_rows(items, "exp6", "free", pair_type="tone")
    )["tone"]
    assert _metric_value(
        table, scope="pair_type", experiment="exp6", dimension="tone",
        metric="accuracy", pair_type="tone",
    ) == pytest.approx(direct6["accuracy"])

    # A/B accuracy
    for experiment in ("exp4", "exp5", "exp6"):
        ab_rows = _main_rows(items, experiment, "ab_choice")
        direct_ab = metrics.ab_metrics(ab_rows)
        assert _metric_value(
            table, experiment=experiment, dimension="ab", metric="accuracy"
        ) == pytest.approx(direct_ab["accuracy"])


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

    # baselines: detection chance 50% and majority-class accuracy
    baselines = _read_csv(run_dir, "baselines.csv", out_dir)
    chance = baselines[
        baselines["scope"].eq("detection")
        & baselines["subset"].eq("main")
        & baselines["baseline"].eq("random")
    ]
    assert len(chance) == 1
    assert float(chance["value"].iloc[0]) == pytest.approx(0.5)
    tone_random = baselines[
        baselines["scope"].eq("overall")
        & baselines["experiment"].eq("exp3")
        & baselines["dimension"].eq("tone")
        & baselines["baseline"].eq("random")
    ]
    assert len(tone_random) == 1
    assert float(tone_random["value"].iloc[0]) == pytest.approx(0.25)
    ab_random = baselines[
        baselines["experiment"].eq("exp4")
        & baselines["dimension"].eq("ab")
        & baselines["baseline"].eq("random")
    ]
    assert len(ab_random) == 1
    assert float(ab_random["value"].iloc[0]) == pytest.approx(0.5)


def test_comparisons_use_paired_items(report_outputs) -> None:
    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "comparisons.csv", out_dir)
    assert not table.empty
    assert table["axis"].isin(["thinking", "audio_quality", "output_form"]).all()
    assert set(table["axis"]) == {"thinking", "audio_quality", "output_form"}
    assert table["n_pairs"].gt(0).all()
    # diff equals acc_a - acc_b (paired bootstrap estimate uses the same rows)
    for _index, row in table.iterrows():
        assert float(row["diff"]) == pytest.approx(
            float(row["acc_a"]) - float(row["acc_b"]), abs=1e-9
        )
        assert 0.0 <= float(row["p"]) <= 1.0
        if not pd.isna(row["p_holm"]):
            assert 0.0 <= float(row["p_holm"]) <= 1.0
    # there must be a comparison between the main condition and each ablation axis
    thinking_pair = table[
        (table["condition_a"] == MAIN_GENERATION_KEY)
        & (table["condition_b"] == "think0__clean__json")
    ]
    assert not thinking_pair.empty
    # McNemar counts on a concrete exp3 joint comparison are consistent with the
    # raw correctness of the paired items.
    row = table[
        (table["family"] == "exp3")
        & (table["metric"] == "joint_accuracy")
        & (table["axis"] == "thinking")
    ].iloc[0]
    main_frame = items[
        items["condition_id"].eq(f"free__{MAIN_GENERATION_KEY}")
        & items["experiment"].eq("exp3")
    ]
    other_frame = items[
        items["condition_id"].eq(f"free__{row['condition_b']}")
        & items["experiment"].eq("exp3")
    ]
    other_ids = set(other_frame["item_id"]) & set(main_frame["item_id"])
    assert int(row["n_pairs"]) == len(other_ids)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_missing_detection_and_single_condition(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(
        tmp_path,
        run_id="partial-001",
        write_detection=False,
        write_manifest=True,
        write_ablation=False,
        conditions=(CONDITIONS[0],),
    )
    artifacts = write_report(cfg, run_dir, out_dir=tmp_path / "partial_out", n_boot=20, seed=3)
    for name in ("report.md", "metrics.csv", "detection.csv", "ablation.csv",
                 "comparisons.csv", "baselines.csv", "summary.json"):
        assert name in artifacts
    # headers exist even when the sections are empty
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
    # main results still exist
    metrics_table = _read_csv(run_dir, "metrics.csv", tmp_path / "partial_out")
    assert not metrics_table.empty


def test_no_manifest_no_detection_small_run(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    run_dir = cfg.path("paths.results_dir") / "runs" / "tiny-001"
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
    summary = json.loads((tmp_path / "tiny_out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_items"] == 2
    assert summary["warnings"]


def test_run_id_resolution_and_default_out_dir(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="resolve-001")
    # run id (not a path) resolves below <results_dir>/runs; default out dir = run dir
    artifacts = write_report(cfg, "resolve-001", n_boot=15, seed=2)
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
        "scope", "experiment", "prompt_condition", "condition_id",
        "generation_condition", "voice", "pair_type", "subset", "dimension",
        "metric", "value", "ci_lo", "ci_hi", "n", "n_failures",
    ]
    expected_detection = [
        "scope", "generation_condition", "voice", "subset", "dimension", "metric",
        "value", "ci_lo", "ci_hi", "n", "n_failures", "tp", "fp", "tn", "fn",
        "n_neg", "n_pos",
    ]
    expected_ablation = [
        "group", "compared_with", "is_main", "condition_id", "generation_condition",
        "experiment", "prompt_condition", "dimension", "metric", "value", "n",
        "n_failures",
    ]
    expected_comparisons = [
        "family", "metric", "dimension", "axis", "condition_a", "condition_b",
        "n_pairs", "acc_a", "acc_b", "diff", "diff_lo", "diff_hi", "mcnemar_b",
        "mcnemar_c", "p", "p_holm", "method",
    ]
    expected_baselines = [
        "scope", "experiment", "subset", "dimension", "baseline", "value", "label", "n",
    ]
    assert _read_csv(run_dir, "metrics.csv", out_dir).columns.tolist() == expected_metrics
    assert _read_csv(run_dir, "detection.csv", out_dir).columns.tolist() == expected_detection
    assert _read_csv(run_dir, "ablation.csv", out_dir).columns.tolist() == expected_ablation
    assert _read_csv(run_dir, "comparisons.csv", out_dir).columns.tolist() == expected_comparisons
    assert _read_csv(run_dir, "baselines.csv", out_dir).columns.tolist() == expected_baselines
    # the constants exported by the module must agree with the documented order
    assert list(METRICS_COLUMNS) == expected_metrics
    assert list(DETECTION_COLUMNS) == expected_detection
    assert list(ABLATION_COLUMNS) == expected_ablation
    assert list(COMPARISON_COLUMNS) == expected_comparisons
    assert list(BASELINE_COLUMNS) == expected_baselines


def test_seed_makes_bootstrap_reproducible(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="repro-001")
    first_dir = tmp_path / "repro_a"
    second_dir = tmp_path / "repro_b"
    write_report(cfg, run_dir, out_dir=first_dir, n_boot=25, seed=123)
    write_report(cfg, run_dir, out_dir=second_dir, n_boot=25, seed=123)
    for name in ("metrics.csv", "detection.csv", "comparisons.csv", "baselines.csv",
                 "ablation.csv"):
        assert (first_dir / name).read_text(encoding="utf-8") == (
            second_dir / name
        ).read_text(encoding="utf-8"), f"{name} is not reproducible"


def test_ablation_table_contains_main_and_condition_values(report_outputs) -> None:
    _cfg, run_dir, _items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "ablation.csv", out_dir)
    assert not table.empty
    assert set(table["compared_with"]) >= {
        "think0__clean__json",
        "think1__noise_20db__json",
        "think1__clean__text",
    }
    # every comparison has the main condition measured on the same subset
    for compared_with, group in table.groupby("compared_with"):
        assert group["is_main"].any(), compared_with
        main_rows = group[group["is_main"]]
        assert (main_rows["generation_condition"] == MAIN_GENERATION_KEY).all()
    # free-recognition and detection metrics are present
    groups = set(table["group"])
    assert {"core", "pairs", "parse_core", "parse_pairs", "detection"} <= groups
    assert (table["metric"] == "parse_failure_rate").any()
    # ablation detection only reports the balanced ``main`` subset
    detection_rows = table[table["group"].eq("detection")]
    assert not detection_rows.empty
    assert detection_rows["metric"].str.startswith("main.").all()


# ---------------------------------------------------------------------------
# Optional pieces of the report
# ---------------------------------------------------------------------------


def test_appendix_and_confusion_sections(tmp_path: Path) -> None:
    cfg, run_dir, _items = _build_run(tmp_path, run_id="appendix-001")
    out_dir = tmp_path / "appendix_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=12, seed=5)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "最小对计数" in report
    assert "解析失败" in report
    assert "模型 / 参数 / 版本" in report
    assert "混淆矩阵" in report
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    facts = summary["data_facts"]
    assert facts is not None
    assert facts["audio"] is not None
    assert facts["audio"]["n_audio"] == len(AUDIO_TABLE) * len(VOICES)
    assert "tone" in facts["minimal_pairs"]
    assert facts["detection_composition"]


def test_bootstrap_ci_matches_direct_stats_call(report_outputs) -> None:
    """The CI columns must equal a direct ``stats.bootstrap_ci`` on identical rows."""
    from pinyin_eval import stats

    _cfg, run_dir, items, out_dir, _artifacts = report_outputs
    table = _read_csv(run_dir, "metrics.csv", out_dir)

    generation = items["condition_id"].map(lambda cid: str(cid).split("__", 1)[1])
    main = items[generation.eq(MAIN_GENERATION_KEY)]
    analysis_mask = main["experiment"].isin(["exp1", "exp2", "exp3"]) | (
        main["experiment"].isin(["exp4", "exp5", "exp6"])
        & main["prompt_condition"].isin(["free", "ab_choice"])
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
        return {name: values[name] for name in (
            "error_precision", "error_recall", "error_f1", "fpr", "fnr", "accuracy",
            "balanced_accuracy",
        )}

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


def test_comparisons_consistency_correct_for_gold_false(tmp_path: Path) -> None:
    """Regression: ``consistent_gold=False`` rows must count as correct on a False prediction.

    Perfect-mock-style data: exp1 (correct target, gold consistent) always answered
    "consistent", exp2 (wrong target, gold inconsistent) always answered
    "inconsistent".  Both must score 1.0 in ``comparisons.csv`` -- the old formula
    ``gold & pred & (gold == pred)`` scored every exp2 row as incorrect.
    """
    cfg, run_dir, items = _build_run(tmp_path, run_id="consistency-001")
    perfect = items.copy()
    perfect.loc[perfect["experiment"].eq("exp1"), "pred_consistent"] = True
    perfect.loc[perfect["experiment"].eq("exp2"), "pred_consistent"] = False
    for experiment in ("exp1", "exp2"):
        mask = perfect["experiment"].eq(experiment)
        perfect.loc[mask, "pred_tone"] = perfect.loc[mask, "gold_tone"]
    perfect.loc[:, "parse_status"] = "ok"
    perfect.loc[perfect["pred_consistent"].isna(), "pred_consistent"] = True
    perfect.to_parquet(run_dir / "items.parquet", index=False)
    out_dir = tmp_path / "consistency_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=2)
    table = pd.read_csv(out_dir / "comparisons.csv")
    for family in ("exp1", "exp2"):
        rows = table[
            table["family"].eq(family) & table["metric"].eq("consistency_accuracy")
        ]
        assert not rows.empty, family
        for value in rows["acc_a"]:
            assert float(value) == pytest.approx(1.0)
        for value in rows["acc_b"]:
            assert float(value) == pytest.approx(1.0)
        for value in rows["diff"]:
            assert float(value) == pytest.approx(0.0, abs=1e-9)
    # tone families are perfect as well (sanity: no confusion introduced elsewhere)
    tone_rows = table[
        table["family"].isin(["exp1", "exp2"]) & table["metric"].eq("tone_accuracy")
    ]
    assert not tone_rows.empty
    assert tone_rows["acc_a"].astype(float).eq(1.0).all()
    assert tone_rows["acc_b"].astype(float).eq(1.0).all()


def test_detection_notes_unbalanced_typed_subsets(tmp_path: Path) -> None:
    """Typed detection subsets that are not counterbalanced are flagged in the report."""
    cfg, run_dir, _items = _build_run(tmp_path, run_id="imbalance-001")
    detection = pd.read_parquet(run_dir / "detection_rows.parquet")
    positives = detection[
        detection["detection_role"].eq("pos")
        & detection["detection_subset"].isin(["initial", "final", "tone"])
    ]
    # keep only one third of the typed positives -> imbalanced typed subsets
    keep = positives.groupby("detection_subset", group_keys=False).head(3)
    balanced_main = detection[detection["detection_subset"].eq("main")]
    prune = detection.drop(positives.index) if len(positives) else detection
    pruned = pd.concat([prune, keep, balanced_main]).reset_index(drop=True)
    pruned.to_parquet(run_dir / "detection_rows.parquet", index=False)
    out_dir = tmp_path / "imbalance_out"
    write_report(cfg, run_dir, out_dir=out_dir, n_boot=10, seed=4)
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "不配平" in report
