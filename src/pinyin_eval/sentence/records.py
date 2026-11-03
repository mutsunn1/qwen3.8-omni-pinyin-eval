"""Frozen data structures for the sentence-level experiment (design §4, §12).

Pipeline (all JSONL/Parquet friendly):

``items`` -> ``RequestPlan`` -> ``GenerationRecord`` -> scored rows -> report

The three tasks use different corpora and gold labels but share one item /
generation / result envelope so that the runner and reporting code stay uniform.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = [
    "CONDITIONS",
    "PARSE_STATUSES",
    "TASKS",
    "TASK_TITLES",
    "GenerationRecord",
    "RequestPlan",
    "SentenceItem",
    "append_jsonl",
    "load_jsonl",
    "make_request_id",
    "write_jsonl",
]

#: task ids (design §4)
TASKS: tuple[str, ...] = ("A", "B", "C")
TASK_TITLES: dict[str, str] = {
    "A": "整句拼音转写",
    "B": "整句发音评测",
    "C": "整句朗读评分",
}

#: generation conditions (design §5): C1 = think off, C2 = think on
CONDITIONS: tuple[str, ...] = ("C1", "C2")

#: parse statuses; only ``ok`` rows enter the metric denominators
PARSE_STATUSES: tuple[str, ...] = ("ok", "parse_error", "refused", "empty", "api_error")

DEFAULT_MAX_TOKENS: dict[str, int] = {"C1": 256, "C2": 1024}


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SentenceItem:
    """One evaluation item of task A, B or C.

    Fields shared by all tasks: ``item_id`` / ``task`` / ``source`` /
    ``speaker`` (cluster id) / ``audio_path`` / ``duration_s`` / ``text``.

    A-specific gold: ``chars`` (gold characters) + ``pinyin_surface``
    (raw AISHELL-3 syllables, ``%``/``$`` prosody marks already stripped).

    B-specific gold: ``labels`` — position-ordered units
    ``[{"text": "他", "tone": 0|1, "consonant": 0|1, "vowel": 0|1,
    "zero_initial": bool}]``; label ``0`` means "wrong" (positive class).

    C-specific gold: ``scores`` — ``{"accuracy": 4.3, "fluency": 4.7,
    "prosody": 4.0}`` (mean of the 3 released raters).

    ``flags`` holds the AISHELL-3 special-subset flags
    (``neutral`` / ``yibu`` / ``erhua``); OMPAL items carry ``native`` and
    ``learner`` instead.
    """

    item_id: str
    task: str
    source: str
    speaker: str
    audio_path: str
    duration_s: float
    text: str
    chars: str = ""
    pinyin_surface: list[str] = field(default_factory=list)
    labels: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    native: bool = False
    learner: str = ""
    flags: dict[str, bool] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SentenceItem:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def cluster_id(self) -> str:
        """Bootstrap cluster: AISHELL speaker id or OMPAL learner id."""
        return self.speaker


# ---------------------------------------------------------------------------
# Request / generation records
# ---------------------------------------------------------------------------


def make_request_id(task: str, condition: str, item_id: str) -> str:
    """Deterministic request id used for resume / de-duplication."""
    return f"{task}|{condition}|{item_id}"


@dataclass(slots=True)
class RequestPlan:
    """One planned API call (one item x one task x one condition)."""

    request_id: str
    item_id: str
    task: str
    condition: str
    audio_path: str
    system_prompt: str
    user_prompt: str
    prompt_version: str
    prompt_sha256: str
    thinking: bool
    max_tokens: int
    output_form: str = "json"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationRecord:
    """One model call (mirrors the design §5/§12 bookkeeping requirements)."""

    run_id: str
    request_id: str
    item_id: str
    task: str
    condition: str
    thinking: bool
    max_tokens: int
    model: str
    prompt_version: str
    prompt_sha256: str
    audio_path: str
    response_text: str = ""
    reasoning_text: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_s: float = 0.0
    error: str | None = None
    retries: int = 0
    timestamp: str = ""
    http_request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenerationRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.response_text.strip())


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows: list[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: str | Path, rows: list[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.touch()
        return p
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    opened = False
    if opened:  # pragma: no cover - never reached, keeps ruff happy about branch
        pass
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
