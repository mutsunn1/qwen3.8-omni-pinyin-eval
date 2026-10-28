"""Frozen data structures shared by all modules.

These dataclasses / schemas are the contract between data preparation, task
building, generation, parsing, scoring and reporting.  See ``docs/contracts.md``.

Design principles
-----------------
* Every record is JSON-serialisable and flat enough to be stored as JSONL
  (generation records) or Parquet (audio manifest, item results).
* Item identifiers are deterministic so that runs are resumable and paired
  comparisons are stable.
* Gold labels always describe the *audio* (never the given target text).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

# ---------------------------------------------------------------------------
# Enumerations (plain tuples to keep JSON / parquet friendly)
# ---------------------------------------------------------------------------

EXPERIMENTS: tuple[str, ...] = ("exp1", "exp2", "exp3", "detect", "exp4", "exp5", "exp6")
EXPERIMENT_TITLES: dict[str, str] = {
    "exp1": "exp1_correct_target_tone",
    "exp2": "exp2_wrong_target_tone",
    "exp3": "exp3_free_recognition",
    "detect": "detect_consistency",
    "exp4": "exp4_initial_minimal_pairs",
    "exp5": "exp5_final_minimal_pairs",
    "exp6": "exp6_tone_minimal_pairs",
}

#: task types (output schema) sent to the model
TASK_TYPES: tuple[str, ...] = ("free", "tone", "detect", "ab")

PROMPT_CONDITIONS: tuple[str, ...] = ("free", "correct_target", "wrong_target", "detect", "ab_choice")
PROMPT_CONDITION_TITLES: dict[str, str] = {
    "free": "自由识别",
    "correct_target": "目标条件（正确目标）",
    "wrong_target": "目标条件（错误目标）",
    "detect": "检测一致性",
    "ab_choice": "A/B 辨别",
}

#: prompt conditions allowed per experiment (design section 四)
EXPERIMENT_PROMPT_CONDITIONS: dict[str, tuple[str, ...]] = {
    "exp1": ("free", "correct_target", "wrong_target"),
    "exp2": ("free", "correct_target", "wrong_target"),
    "exp3": ("free", "correct_target", "wrong_target"),
    "detect": ("detect",),
    "exp4": ("free", "correct_target", "wrong_target", "ab_choice"),
    "exp5": ("free", "correct_target", "wrong_target", "ab_choice"),
    "exp6": ("free", "correct_target", "wrong_target", "ab_choice"),
}

OUTPUT_FORMS: tuple[str, ...] = ("json", "text")
AUDIO_QUALITIES: tuple[str, ...] = ("clean", "noise_20db")
THINKING_LEVELS: tuple[bool, ...] = (True, False)

#: minimal-pair families (design section 三, table 最小对立组)
PAIR_TYPES: tuple[str, ...] = (
    "sh_zh",          # 平翘舌 z/zh c/ch s/sh  (excluding -i group)
    "aspiration",     # 送气 b/p d/t g/k j/q
    "nasal_lateral",  # n/l
    "front_back_nasal_ptr",  # 前后鼻音 an/ang en/eng in/ing (design count = 40)
    "i_ü",            # i/ü
    "ian_üan",        # ian/üan
    "tone",           # 声调最小对
)

#: experiment -> metric family used by the design's 实验 × 指标 matrix
EXPERIMENT_METRIC_DIMENSION: dict[str, str] = {
    "exp1": "tone",
    "exp2": "tone",
    "exp3": "joint",
    "detect": "detection",
    "exp4": "initial",
    "exp5": "final",
    "exp6": "tone",
}

PARSE_STATUSES: tuple[str, ...] = ("ok", "parse_error", "refused", "empty", "api_error")


def slugify_condition(
    *,
    thinking: bool,
    audio_quality: str,
    output_form: str,
) -> str:
    """Canonical condition slug, e.g. ``think1_noise20_text``."""
    t = "think1" if thinking else "think0"
    aq = audio_quality
    of = output_form
    return f"{t}__{aq}__{of}"


# ---------------------------------------------------------------------------
# Audio manifest record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AudioRecord:
    """One audio clip of ``mandarin_tone_syn`` after canonicalisation."""

    audio_id: str
    row_index: int
    speaker: str       # dataset ``speaker`` == TTS voice id, e.g. ZH_TTS_A
    voice: str         # short voice id A/B/C/D
    raw_target: str    # dataset ``target`` (385 spellings)
    target: str        # canonical syllable (378 spellings)
    initial: str       # from mapping CSV; "" = zero initial
    final: str         # from mapping CSV
    tone: int          # dataset ``label`` in 1..4
    wav_path: str      # repo-relative path, e.g. data/audio/ZH_TTS_A/xxx.wav
    duration_s: float
    sample_rate: int
    channels: int

    def __post_init__(self) -> None:
        if self.tone not in (1, 2, 3, 4):
            raise ValueError(f"tone must be 1..4, got {self.tone!r} ({self.audio_id})")

    @property
    def stratum(self) -> str:
        """Voice x tone stratum used for ablation sampling."""
        return f"{self.voice}|{self.tone}"


# ---------------------------------------------------------------------------
# Task item
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaskItem:
    """A single (audio, prompt) unit that will be sent to the model once.

    The same item is replayed across every generation condition
    (thinking / audio quality / output form) and both media variants.
    """

    item_id: str
    experiment: str
    prompt_condition: str
    audio_id: str
    wav_path: str
    voice: str
    gold_initial: str
    gold_final: str
    gold_tone: int
    cluster_id: str = ""
    pair_id: str | None = None
    pair_type: str | None = None
    pair_side: str | None = None
    variant_type: str | None = None          # exp2: initial | final | tone
    variant_index: int | None = None         # 0/1 for init & final variants, 0 for tone
    given_target: str | None = None          # what the prompt claims (canonical spelling)
    given_tone: int | None = None            # set only when the claim carries a tone digit
    consistent_gold: bool | None = None      # target conditions only
    alternatives: list[str] | None = None    # ab_choice only, order ["A","B"]
    correct_choice: str | None = None        # ab_choice only: "A" | "B"
    stratum: str = ""                        # voice|tone, for stratified sampling
    task_type: str = ""                      # free | tone | detect | ab (output schema)
    prompt_texts: dict[str, str] | None = None   # {"json":..., "text":...} built by tasks.py
    system_prompt: str = ""                  # shared system prompt (part of request id)
    prompt_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.cluster_id:
            d["cluster_id"] = self.audio_id
        if not self.stratum:
            d["stratum"] = f"{self.voice}|{self.gold_tone}"
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskItem:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def display_target(self) -> str:
        """Canonical pinyin with tone digit, e.g. ``zhang1`` (for A/B + prompts)."""
        base = self.given_target if self.given_target is not None else self.gold_initial + self.gold_final
        return f"{base}{self.given_tone}" if self.given_tone else base


# ---------------------------------------------------------------------------
# Generation record (one model call)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenerationRecord:
    run_id: str
    request_id: str
    item_id: str
    experiment: str
    prompt_condition: str
    condition_id: str
    thinking: bool
    audio_quality: str
    output_form: str
    model: str
    prompt_version: str
    response_text: str = ""
    reasoning_text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_s: float = 0.0
    error: str | None = None
    retries: int = 0
    timestamp: str = ""
    media_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def call_key(self) -> tuple[str, str, str]:
        return (self.run_id, self.request_id, self.condition_id)


# ---------------------------------------------------------------------------
# Scoring row (one item x condition, with parsed prediction)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ItemResult:
    """Item-level scored record written to ``items.parquet``."""

    run_id: str
    item_id: str
    experiment: str
    prompt_condition: str
    condition_id: str
    thinking: bool
    audio_quality: str
    output_form: str
    audio_id: str
    cluster_id: str
    voice: str
    pair_id: str | None
    pair_type: str | None
    pair_side: str | None
    variant_type: str | None
    gold_initial: str
    gold_final: str
    gold_tone: int
    given_target: str | None
    given_tone: int | None
    consistent_gold: bool | None
    alternatives: list[str] | None
    correct_choice: str | None
    # predictions after prompt parsing + normalisation; None = not available
    pred_consistent: bool | None
    pred_initial: str | None
    pred_final: str | None
    pred_tone: int | None
    pred_choice: str | None
    parse_status: str
    raw_response: str = ""
    raw_reasoning: str = ""
    latency_s: float = 0.0
    error: str | None = None
    # filled when assembling a detection analysis (negatives = consistent claims,
    # positives = inconsistent claims of the detection experiment); None otherwise.
    detection_role: str | None = None
    detection_subset: str | None = None
    # output schema of the task: free | tone | detect | ab
    task_type: str = ""

    PARSE_STATUSES: ClassVar[tuple[str, ...]] = PARSE_STATUSES

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, d: Any) -> ItemResult:
        known = {f.name for f in fields(cls)}
        if hasattr(d, "keys"):
            d = {k: d[k] for k in d.keys()}
        row = {}
        for k, v in d.items():
            if k not in known:
                continue
            if v is None or (isinstance(v, float) and isinstance(v, float) and v != v):  # NaN
                row[k] = None
            else:
                row[k] = v
        # list-typed alternatives may arrive as numpy arrays
        alt = row.get("alternatives")
        if alt is not None and not isinstance(alt, list):
            row["alternatives"] = [str(x) for x in alt]
        return cls(**row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_item_id(experiment: str, prompt_condition: str, suffix: str) -> str:
    return f"{experiment}|{prompt_condition}|{suffix}"


def make_pair_id(pair_type: str, target_a: str, target_b: str, voice: str, tone: str) -> str:
    """Deterministic minimal-pair id (order independent)."""
    a, b = sorted((target_a, target_b))
    return f"{pair_type}:{a}/{b}:{voice}:{tone}"


__all__ = [
    "AUDIO_QUALITIES",
    "EXPERIMENTS",
    "TASK_TYPES",
    "EXPERIMENT_METRIC_DIMENSION",
    "EXPERIMENT_PROMPT_CONDITIONS",
    "EXPERIMENT_TITLES",
    "OUTPUT_FORMS",
    "PAIR_TYPES",
    "PARSE_STATUSES",
    "PROMPT_CONDITIONS",
    "PROMPT_CONDITION_TITLES",
    "THINKING_LEVELS",
    "AudioRecord",
    "GenerationRecord",
    "ItemResult",
    "TaskItem",
    "make_item_id",
    "make_pair_id",
    "slugify_condition",
]
