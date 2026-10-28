"""Task construction: experiment 1-6 prompt-condition units + detection sets.

Terminology (see docs/contracts.md §3):

* **core tasks** cover all 6160 audios with the three prompt conditions that
  define experiments 1-3 (correct target -> exp1 view, wrong target -> exp2
  view, free -> exp3 view).  Experiment 2 adds five wrong-target variants per
  audio (2 initial / 2 final / 1 tone).
* **pair tasks** cover minimal-pair instances (experiments 4-6) with the fourth
  prompt condition ``ab_choice``.
* **detection sets** are analysis views over exp1 negatives + exp2 positives;
  they never trigger extra API calls.

All task construction is deterministic given the global seed.  Prompt texts are
attached to every item (``prompt_texts``) so request fingerprints can be
recomputed at scoring time and duplicate prompts are only sent once.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .data.minimal_pairs import build_all_pairs, build_instances, iter_pair_items
from .normalize import MappingEntry, load_mapping
from .prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt_texts,
    request_id_for,
)
from .records import (
    AudioRecord,
    TaskItem,
    make_item_id,
)

log = logging.getLogger(__name__)

#: pair_type -> experiment (design section 三: exp4 initial, exp5 final, exp6 tone)
PAIR_TYPE_TO_EXPERIMENT: dict[str, str] = {
    "sh_zh": "exp4",
    "aspiration": "exp4",
    "nasal_lateral": "exp4",
    "front_back_nasal_ptr": "exp5",
    "i_ü": "exp5",
    "ian_üan": "exp5",
    "tone": "exp6",
}

#: export order used in summaries
EXPERIMENT_ORDER = ("exp1", "exp2", "exp3", "detect", "exp4", "exp5", "exp6")


class TaskError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Deterministic RNG helpers
# ---------------------------------------------------------------------------


def _rng(seed: int, *parts: object) -> np.random.Generator:
    payload = "\x00".join(str(p) for p in (seed, *parts))
    digest = hashlib.sha256(payload.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _balanced_choice(seed: int, key: str) -> int:
    """Deterministic balanced bit (0/1) used to swap A/B candidate positions."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return digest[0] & 1


def _tone_digit_string(target: str, tone: int) -> str:
    return f"{target}{int(tone)}"


# ---------------------------------------------------------------------------
# Target pools for experiment 2 error injection
# ---------------------------------------------------------------------------


def _build_pools(mapping: dict[str, MappingEntry]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """same-final pool (for initial swaps) and same-initial pool (for final swaps)."""
    by_final: dict[str, list[str]] = defaultdict(list)
    by_initial: dict[str, list[str]] = defaultdict(list)
    for target in sorted(mapping):
        entry = mapping[target]
        by_final[entry.final].append(target)
        by_initial[entry.initial].append(target)
    return dict(by_final), dict(by_initial)


def _initial_swap_pool(target: str, mapping: dict[str, MappingEntry],
                       by_final: dict[str, list[str]]) -> list[str]:
    entry = mapping[target]
    return [t for t in by_final.get(entry.final, []) if mapping[t].initial != entry.initial]


def _final_swap_pool(target: str, mapping: dict[str, MappingEntry],
                     by_initial: dict[str, list[str]]) -> list[str]:
    entry = mapping[target]
    return [t for t in by_initial.get(entry.initial, []) if mapping[t].final != entry.final]


def _sample_without_replacement(pool: Sequence[str], k: int, rng: np.random.Generator) -> list[str]:
    pool = sorted(pool)
    if not pool or k <= 0:
        return []
    k = min(k, len(pool))
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Core tasks (experiments 1-3 views)
# ---------------------------------------------------------------------------


def _with_prompts(task: TaskItem) -> TaskItem:
    """Attach the shared system prompt + user prompts and stable metadata."""
    task.prompt_texts = build_prompt_texts(task)
    task.system_prompt = SYSTEM_PROMPT
    task.prompt_version = PROMPT_VERSION
    if not task.cluster_id:
        task.cluster_id = task.audio_id
    if not task.stratum:
        task.stratum = f"{task.voice}|{task.gold_tone}"
    return task


def _base_fields(record: AudioRecord) -> dict[str, Any]:
    return dict(
        audio_id=record.audio_id,
        wav_path=record.wav_path,
        voice=record.voice,
        gold_initial=record.initial,
        gold_final=record.final,
        gold_tone=record.tone,
        cluster_id=record.audio_id,
        stratum=f"{record.voice}|{record.tone}",
    )


# ---------------------------------------------------------------------------
# Core tasks: experiments 1-3 views (tone / tone / free recognition)
# ---------------------------------------------------------------------------


def build_core_tasks(
    records: Sequence[AudioRecord],
    cfg: Config,
    mapping: dict[str, MappingEntry] | None = None,
) -> list[TaskItem]:
    """Shared task catalog for experiments 1-3.

    Design update (2026-10-28):
    * experiment 1: correct toneless target -> tone only;
    * experiment 2: wrong toneless target -> tone only (five segmental variants:
      2 initial changes, 2 final changes, 1 extra initial-or-final change; no
      tone change — tone errors are covered by the detection experiment);
    * experiment 3: free recognition (initial/final/tone).
    """
    if mapping is None:
        mapping = load_mapping(cfg.path("paths.mapping_csv"))
    seed = cfg.seed()
    by_final, by_initial = _build_pools(mapping)
    exp2_cfg = cfg.get("tasks.exp2", {}) or {}
    n_init = int(exp2_cfg.get("n_initial_variants", 2))
    n_final = int(exp2_cfg.get("n_final_variants", 2))
    n_extra = int(exp2_cfg.get("n_extra_segmental_variants", 1))

    tasks: list[TaskItem] = []
    skipped_initial = 0
    skipped_final = 0
    for record in sorted(records, key=lambda r: r.audio_id):
        base = _base_fields(record)

        # experiment 3 view: free recognition
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id("exp3", "free", record.audio_id),
            experiment="exp3",
            prompt_condition="free",
            task_type="free",
            **base,
        )))

        # experiment 1 view: correct toneless target -> tone only
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id("exp1", "correct_target", record.audio_id),
            experiment="exp1",
            prompt_condition="correct_target",
            task_type="tone",
            given_target=record.target,
            consistent_gold=True,
            **base,
        )))

        # experiment 2 view: five wrong toneless segmental targets -> tone only
        rng = _rng(seed, "exp2", record.audio_id)
        init_pool = _initial_swap_pool(record.target, mapping, by_final)
        final_pool = _final_swap_pool(record.target, mapping, by_initial)
        init_choices = _sample_without_replacement(init_pool, n_init, rng)
        final_choices = _sample_without_replacement(final_pool, n_final, rng)
        if len(init_choices) < n_init:
            skipped_initial += n_init - len(init_choices)
        if len(final_choices) < n_final:
            skipped_final += n_final - len(final_choices)

        # 5th (extra) variant: one more initial or final change, chosen by hash so
        # every variant still changes only one segmental part (design: 只改声母或韵母)
        extra_choices: list[tuple[str, str]] = []
        if n_extra > 0:
            used_init = set(init_choices)
            used_final = set(final_choices)
            remaining_init = [t for t in init_pool if t not in used_init]
            remaining_final = [t for t in final_pool if t not in used_final]
            prefer_initial = _balanced_choice(seed, f"exp2-extra:{record.audio_id}") == 0
            order = [("initial", remaining_init), ("final", remaining_final)]
            if not prefer_initial:
                order.reverse()
            for variant_type, pool in order:
                if pool:
                    picked = _sample_without_replacement(pool, n_extra, rng)
                    extra_choices = [(variant_type, t) for t in picked]
                    break

        for idx, swapped_target in enumerate(init_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("exp2", "wrong_target", f"{record.audio_id}|initial{idx}"),
                experiment="exp2",
                prompt_condition="wrong_target",
                task_type="tone",
                given_target=swapped_target,
                consistent_gold=False,
                variant_type="initial",
                variant_index=idx,
                **base,
            )))
        for idx, swapped_final in enumerate(final_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("exp2", "wrong_target", f"{record.audio_id}|final{idx}"),
                experiment="exp2",
                prompt_condition="wrong_target",
                task_type="tone",
                given_target=swapped_final,
                consistent_gold=False,
                variant_type="final",
                variant_index=idx,
                **base,
            )))
        for idx, (variant_type, swapped_target) in enumerate(extra_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("exp2", "wrong_target", f"{record.audio_id}|extra{idx}"),
                experiment="exp2",
                prompt_condition="wrong_target",
                task_type="tone",
                given_target=swapped_target,
                consistent_gold=False,
                variant_type=variant_type,
                variant_index=idx + 2,
                **base,
            )))

    log.info(
        "core tasks: %d items; exp2 variant shortfall: initial=%d final=%d extra=%d",
        len(tasks), skipped_initial, skipped_final,
        sum(1 for t in tasks if t.experiment == "exp2" and t.variant_type == "extra"),
    )
    return tasks


# ---------------------------------------------------------------------------
# Detection experiment (new in the 2026-10-28 design update)
# ---------------------------------------------------------------------------


def build_detection_tasks(
    records: Sequence[AudioRecord],
    cfg: Config,
    mapping: dict[str, MappingEntry] | None = None,
) -> list[TaskItem]:
    """Detection experiment: target + target_tone -> consistency only.

    Per audio: 1 consistent sample (original ``target`` + ``label`` tone) and 5
    inconsistent variants: 2 initial changes, 2 final changes, 1 tone change.
    Negative/positive sets for the metrics are built by
    :func:`build_detection_sets`.
    """
    if mapping is None:
        mapping = load_mapping(cfg.path("paths.mapping_csv"))
    seed = cfg.seed()
    by_final, by_initial = _build_pools(mapping)
    det_cfg = cfg.get("tasks.detection", {}) or {}
    n_init = int(det_cfg.get("n_initial_variants", 2))
    n_final = int(det_cfg.get("n_final_variants", 2))
    n_tone = int(det_cfg.get("n_tone_variants", 1))

    tasks: list[TaskItem] = []
    for record in sorted(records, key=lambda r: r.audio_id):
        base = _base_fields(record)

        # consistent negative sample: original target + true tone
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id("detect", "detect", record.audio_id),
            experiment="detect",
            prompt_condition="detect",
            task_type="detect",
            given_target=record.target,
            given_tone=record.tone,
            consistent_gold=True,
            **base,
        )))

        rng = _rng(seed, "detect", record.audio_id)
        init_pool = _initial_swap_pool(record.target, mapping, by_final)
        final_pool = _final_swap_pool(record.target, mapping, by_initial)
        init_choices = _sample_without_replacement(init_pool, n_init, rng)
        final_choices = _sample_without_replacement(final_pool, n_final, rng)
        wrong_tone_pool = [t for t in (1, 2, 3, 4) if t != record.tone]
        tone_choices = _sample_without_replacement([str(t) for t in wrong_tone_pool], n_tone, rng)

        for idx, swapped_target in enumerate(init_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("detect", "detect", f"{record.audio_id}|initial{idx}"),
                experiment="detect",
                prompt_condition="detect",
                task_type="detect",
                given_target=swapped_target,
                given_tone=record.tone,  # only segments are wrong
                consistent_gold=False,
                variant_type="initial",
                variant_index=idx,
                **base,
            )))
        for idx, swapped_final in enumerate(final_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("detect", "detect", f"{record.audio_id}|final{idx}"),
                experiment="detect",
                prompt_condition="detect",
                task_type="detect",
                given_target=swapped_final,
                given_tone=record.tone,
                consistent_gold=False,
                variant_type="final",
                variant_index=idx,
                **base,
            )))
        for idx, wrong_tone in enumerate(tone_choices):
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id("detect", "detect", f"{record.audio_id}|tone{idx}"),
                experiment="detect",
                prompt_condition="detect",
                task_type="detect",
                given_target=record.target,
                given_tone=int(wrong_tone),
                consistent_gold=False,
                variant_type="tone",
                variant_index=idx,
                **base,
            )))
    log.info("detection tasks: %d items", len(tasks))
    return tasks


# ---------------------------------------------------------------------------
# Pair tasks (experiments 4-6 views)
# ---------------------------------------------------------------------------


def build_pair_tasks(
    records: Sequence[AudioRecord],
    cfg: Config,
    mapping: dict[str, MappingEntry] | None = None,
) -> list[TaskItem]:
    if mapping is None:
        mapping = load_mapping(cfg.path("paths.mapping_csv"))
    seed = cfg.seed()
    include_wide_nasal = bool(cfg.get("tasks.minimal_pairs.include_wide_nasal", False))
    pairs = build_all_pairs(records, mapping, include_wide_nasal=include_wide_nasal)
    instances = build_instances(pairs, records)
    log.info("minimal pairs: %d pairs -> %d instances", len(pairs), len(instances))

    tasks: list[TaskItem] = []
    skipped_exp6_wrong = 0
    for item in iter_pair_items(instances):
        experiment = PAIR_TYPE_TO_EXPERIMENT.get(item.pair_type)
        if experiment is None:
            raise TaskError(f"unknown pair_type {item.pair_type!r}")
        audio: AudioRecord = item.audio
        other: AudioRecord = item.other_audio
        base = dict(
            audio_id=audio.audio_id,
            wav_path=audio.wav_path,
            voice=audio.voice,
            gold_initial=audio.initial,
            gold_final=audio.final,
            gold_tone=audio.tone,
            cluster_id=audio.audio_id,
            pair_id=item.pair_id,
            pair_type=item.pair_type,
            pair_side=item.side,
            stratum=f"{audio.voice}|{audio.tone}",
        )
        pair_suffix = f"{audio.audio_id}|{item.pair_id}"

        # free recognition (same prompt as experiment 3 -> request-level dedupe)
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id(experiment, "free", pair_suffix),
            experiment=experiment,
            prompt_condition="free",
            task_type="free",
            **base,
        )))

        # correct target: tone-only prompt with the audio's own toneless target
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id(experiment, "correct_target", pair_suffix),
            experiment=experiment,
            prompt_condition="correct_target",
            task_type="tone",
            given_target=audio.target,
            consistent_gold=True,
            **base,
        )))

        # wrong target: tone-only prompt with the minimal-pair partner's target.
        # For experiment 6 the partner has the same toneless target (tone pairs
        # differ only in tone), so a "wrong target" is not expressible in the
        # toneless-target prompt; those items are skipped (documented in
        # docs/decisions.md) and tone discrimination is measured via A/B.
        if audio.target == other.target:
            skipped_exp6_wrong += 1
        else:
            tasks.append(_with_prompts(TaskItem(
                item_id=make_item_id(experiment, "wrong_target", pair_suffix),
                experiment=experiment,
                prompt_condition="wrong_target",
                task_type="tone",
                given_target=other.target,
                given_tone=None,
                consistent_gold=False,
                variant_type="initial" if experiment == "exp4" else "final",
                **base,
            )))

        # A/B discrimination (balanced candidate order)
        own_surface = _tone_digit_string(audio.target, audio.tone)
        other_surface = _tone_digit_string(other.target, other.tone)
        swap = _balanced_choice(seed, f"{item.pair_id}:{audio.audio_id}")
        if swap:
            alternatives = [other_surface, own_surface]
            correct_choice = "B"
        else:
            alternatives = [own_surface, other_surface]
            correct_choice = "A"
        tasks.append(_with_prompts(TaskItem(
            item_id=make_item_id(experiment, "ab_choice", pair_suffix),
            experiment=experiment,
            prompt_condition="ab_choice",
            task_type="ab",
            alternatives=alternatives,
            correct_choice=correct_choice,
            **base,
        )))

    log.info("pair tasks: %d items (skipped exp6 wrong-target: %d)", len(tasks), skipped_exp6_wrong)
    return tasks


# ---------------------------------------------------------------------------
# Detection set assembly
# ---------------------------------------------------------------------------


def build_detection_sets(detection_tasks: Sequence[TaskItem], cfg: Config) -> pd.DataFrame:
    """Pair consistent (negative) with inconsistent (positive) detection claims.

    Returns a long dataframe with one row per (subset, audio):
    ``subset`` in {main, initial, final, tone}; ``negative_item_id`` is the
    consistent sample (original target+tone) and ``positive_item_id`` is an
    inconsistent variant.  ``main`` samples one of the five variants per audio
    with weights 2/2/1 (initial/final/tone); each per-type subset samples one
    variant of that type.  ``er``/``weng`` have no same-final alternative, so the
    ``initial`` subset is slightly smaller.
    """
    seed = cfg.seed()
    weights = cfg.get("tasks.detection.strata_weights", {"initial": 2, "final": 2, "tone": 1})
    subsets = list(cfg.get("tasks.detection.subsets", ["main", "initial", "final", "tone"]))

    negatives: dict[str, TaskItem] = {}
    positives: dict[str, dict[str, list[TaskItem]]] = defaultdict(lambda: defaultdict(list))
    for task in detection_tasks:
        if task.experiment != "detect":
            continue
        if task.consistent_gold is True:
            negatives[task.audio_id] = task
        elif task.variant_type:
            positives[task.audio_id][task.variant_type].append(task)

    rows: list[dict[str, Any]] = []
    for audio_id in sorted(negatives):
        neg = negatives[audio_id]
        available = {k: v for k, v in positives.get(audio_id, {}).items() if v}
        if not available:
            log.warning("no inconsistent detection variants for audio %s", audio_id)
            continue
        for subset in subsets:
            if subset == "main":
                types = sorted(available)
                if len(types) > 1:
                    w = np.array([float(weights.get(t, 0)) for t in types], dtype=float)
                    if w.sum() <= 0:
                        w = np.ones(len(types), dtype=float)
                    rng = _rng(seed, "detection-main", audio_id)
                    pick = types[int(rng.choice(len(types), p=w / w.sum()))]
                else:
                    pick = types[0]
                candidates = available[pick]
                rng = _rng(seed, "detection-main-variant", audio_id)
                pos = candidates[int(rng.integers(len(candidates)))]
            elif subset in available and available[subset]:
                candidates = available[subset]
                rng = _rng(seed, "detection-subset", audio_id, subset)
                pos = candidates[int(rng.integers(len(candidates)))]
            else:
                continue
            rows.append({
                "subset": subset,
                "audio_id": audio_id,
                "cluster_id": neg.cluster_id,
                "voice": neg.voice,
                "tone": neg.gold_tone,
                "negative_item_id": neg.item_id,
                "positive_item_id": pos.item_id,
                "variant_type": pos.variant_type,
                "negative_given_target": neg.given_target,
                "negative_given_tone": neg.given_tone,
                "positive_given_target": pos.given_target,
                "positive_given_tone": pos.given_tone,
            })

    df = pd.DataFrame(rows)
    log.info("detection rows: %s", df.groupby("subset").size().to_dict() if len(df) else {})
    return df


# ---------------------------------------------------------------------------
# Ablation subsets
# ---------------------------------------------------------------------------


def sample_ablation(
    cfg: Config,
    core_tasks: Sequence[TaskItem],
    pair_tasks: Sequence[TaskItem],
    detection_tasks: Sequence[TaskItem],
    detection_sets: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the fixed item-id subsets used by the ablation grid.

    * core (experiments 1-3): 256 items per prompt condition, stratified by
      voice|tone (16 per cell) -> 3 x 256 = 768 items.
    * detection: 256 audios stratified by voice|tone, keeping **all six**
      detection tasks per audio (1 consistent + 5 inconsistent) so that both the
      balanced ``main`` set and the typed subsets stay computable -> 1536 items.
    * pairs (experiments 4-6): up to ``cap`` pair instances per ``pair_type``,
      keeping both sides and all prompt conditions of those instances.
    """
    seed = cfg.seed()
    per_condition = int(cfg.get("tasks.ablation.items_per_condition", 256))
    strata_per_cell = int(cfg.get("tasks.ablation.strata_per_voice_tone", 16))
    pair_cap = int(cfg.get("tasks.minimal_pairs.ablation_per_type_cap", 256))
    tone_cap = int(cfg.get("tasks.minimal_pairs.tone_ablation_pairs", 256))

    # --- core (experiments 1-3 views): 256 items per prompt condition ---
    core_selected: list[str] = []
    by_prompt: dict[str, list[TaskItem]] = defaultdict(list)
    for task in core_tasks:
        by_prompt[task.prompt_condition].append(task)
    for prompt_condition in sorted(by_prompt):
        tasks_for_prompt = by_prompt[prompt_condition]
        by_stratum: dict[str, list[TaskItem]] = defaultdict(list)
        for task in tasks_for_prompt:
            by_stratum[task.stratum].append(task)
        chosen: list[str] = []
        for stratum in sorted(by_stratum):
            pool = sorted(by_stratum[stratum], key=lambda t: t.item_id)
            rng = _rng(seed, "ablation-core", prompt_condition, stratum)
            k = min(strata_per_cell, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            chosen.extend(pool[int(i)].item_id for i in sorted(idx))
        if chosen and len(chosen) != per_condition:
            log.warning(
                "ablation core[%s] selected %d items (target %d)",
                prompt_condition, len(chosen), per_condition,
            )
        core_selected.extend(chosen)

    # --- detection: 256 audios x all six detection tasks ---
    detection_selected: list[str] = []
    if detection_tasks:
        by_stratum_det: dict[str, list[str]] = defaultdict(list)
        for task in detection_tasks:
            if task.consistent_gold is True:
                by_stratum_det[task.stratum].append(task.audio_id)
        selected_audios: set[str] = set()
        for stratum in sorted(by_stratum_det):
            audio_ids = sorted(set(by_stratum_det[stratum]))
            rng = _rng(seed, "ablation-detection", stratum)
            k = min(strata_per_cell, len(audio_ids))
            idx = rng.choice(len(audio_ids), size=k, replace=False)
            selected_audios.update(audio_ids[int(i)] for i in sorted(idx))
        detection_selected = [
            t.item_id for t in detection_tasks if t.audio_id in selected_audios
        ]
        if len(selected_audios) != per_condition:
            log.warning(
                "ablation detection selected %d audios (target %d)",
                len(selected_audios), per_condition,
            )

    # --- minimal pairs: cap pair instances per pair_type ---
    by_type_instances: dict[str, dict[str, str]] = defaultdict(dict)
    for task in pair_tasks:
        assert task.pair_id and task.pair_type
        by_type_instances[task.pair_type][task.pair_id] = task.voice

    selected_pair_ids: set[str] = set()
    selected_summary: dict[str, dict[str, int]] = {}
    for pair_type in sorted(by_type_instances):
        instance_ids = sorted(by_type_instances[pair_type])
        cap = tone_cap if pair_type == "tone" else pair_cap
        rng = _rng(seed, "ablation-pairs", pair_type)
        k = min(cap, len(instance_ids))
        idx = rng.choice(len(instance_ids), size=k, replace=False)
        chosen = {instance_ids[int(i)] for i in sorted(idx)}
        selected_pair_ids.update(chosen)
        selected_summary[pair_type] = {"available": len(instance_ids), "selected": len(chosen)}

    pair_ids = [t.item_id for t in pair_tasks if t.pair_id in selected_pair_ids]
    log.info(
        "ablation selection: core=%d, detection=%d, pairs=%d items",
        len(core_selected), len(detection_selected), len(pair_ids),
    )
    log.info("ablation pair instances: %s", selected_summary)
    return {
        "core": core_selected,
        "detect": detection_selected,
        "pairs": pair_ids,
        "pair_instance_summary": selected_summary,
    }


# ---------------------------------------------------------------------------
# Request identity helpers (shared by runner and scoring)
# ---------------------------------------------------------------------------


def condition_id_for(prompt_condition: str, *, thinking: bool, audio_quality: str,
                     output_form: str) -> str:
    """Generation-grid condition id, e.g. ``free__think1__clean__json``."""
    think = "think1" if thinking else "think0"
    return f"{prompt_condition}__{think}__{audio_quality}__{output_form}"


def media_rel_for(cfg: Config, task: TaskItem, audio_quality: str) -> str:
    """Repo-relative media path for a task under an audio-quality condition."""
    if audio_quality == "clean":
        return task.wav_path
    if audio_quality == "noise_20db":
        from .data.dataset import noisy_path

        path = noisy_path(cfg, task.audio_id)
        try:
            return str(path.resolve().relative_to(cfg.repo_root))
        except ValueError:
            return str(path)
    raise ValueError(f"unknown audio_quality: {audio_quality!r}")


def request_id_for_task(cfg: Config, task: TaskItem, *, output_form: str,
                        audio_quality: str) -> str:
    """Recompute the API-call fingerprint used by the runner's dedupe."""
    if not task.prompt_texts:
        raise TaskError(f"task {task.item_id} has no prompt_texts; rebuild tasks")
    return request_id_for(
        media_relpath=media_rel_for(cfg, task, audio_quality),
        output_form=output_form,
        prompt_text=task.prompt_texts[output_form],
        system_prompt=task.system_prompt or SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def write_tasks(tasks: Iterable[TaskItem], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    log.info("wrote %d tasks to %s", n, path)
    return n


def load_tasks(path: str | Path) -> list[TaskItem]:
    path = Path(path)
    if not path.exists():
        raise TaskError(f"task file not found: {path}")
    tasks: list[TaskItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(TaskItem.from_dict(json.loads(line)))
    return tasks


def write_item_ids(item_ids: Sequence[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for item_id in item_ids:
            fh.write(item_id + "\n")


def load_item_ids(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Summaries / estimation
# ---------------------------------------------------------------------------


def summarize_tasks(tasks: Sequence[TaskItem]) -> dict[str, Any]:
    by_exp = Counter(t.experiment for t in tasks)
    by_cond = Counter((t.experiment, t.prompt_condition) for t in tasks)
    by_variant = Counter(t.variant_type for t in tasks if t.variant_type)
    by_pair_type = Counter(t.pair_type for t in tasks if t.pair_type)
    return {
        "n_items": len(tasks),
        "by_experiment": {e: by_exp.get(e, 0) for e in EXPERIMENT_ORDER},
        "by_experiment_prompt": {
            f"{e}|{p}": by_cond.get((e, p), 0) for e in EXPERIMENT_ORDER
            for p in sorted({c for (x, c) in by_cond if x == e})
        },
        "exp2_variant_types": dict(by_variant),
        "pair_types": dict(by_pair_type),
    }


def count_unique_requests(
    tasks: Sequence[TaskItem],
    cfg: Config,
    *,
    conditions: Sequence[dict[str, Any]] | None = None,
    item_subset: set[str] | None = None,
) -> dict[str, Any]:
    """Estimate how many API calls a run needs (after request dedupe)."""
    if conditions is None:
        conditions = [cfg.main_condition]
    from .data.dataset import noisy_path

    noisy_rel = str(noisy_path(cfg, "placeholder").relative_to(cfg.repo_root))
    noisy_prefix = noisy_rel.rsplit("placeholder", 1)[0]
    request_ids: set[tuple[str, str, str]] = set()
    per_condition: dict[str, int] = {}
    for task in tasks:
        if item_subset is not None and task.item_id not in item_subset:
            continue
        assert task.prompt_texts
        for cond in conditions:
            media_rel = task.wav_path if cond["audio_quality"] == "clean" else (
                noisy_prefix + f"{task.audio_id}__snr20db.wav"
            )
            prompt_text = task.prompt_texts[cond["output_form"]]
            rid = request_id_for(
                media_relpath=media_rel,
                output_form=cond["output_form"],
                prompt_text=prompt_text,
            )
            # thinking does not change the request text but changes the call
            key = (rid, str(cond["thinking"]), cond["output_form"])
            if key in request_ids:
                continue
            request_ids.add(key)
            cid = f"{cond['thinking']}|{cond['audio_quality']}|{cond['output_form']}"
            per_condition[cid] = per_condition.get(cid, 0) + 1
    return {
        "n_requests": len(request_ids),
        "per_condition": dict(sorted(per_condition.items())),
        "n_conditions": len(conditions),
    }


__all__ = [
    "EXPERIMENT_ORDER",
    "condition_id_for",
    "media_rel_for",
    "request_id_for_task",
    "PAIR_TYPE_TO_EXPERIMENT",
    "TaskError",
    "build_core_tasks",
    "build_detection_sets",
    "build_pair_tasks",
    "count_unique_requests",
    "load_item_ids",
    "load_tasks",
    "sample_ablation",
    "summarize_tasks",
    "write_item_ids",
    "write_tasks",
]
