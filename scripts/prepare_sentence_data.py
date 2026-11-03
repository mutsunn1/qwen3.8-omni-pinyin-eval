#!/usr/bin/env python
"""Prepare the sentence-level pinyin-evaluation data (design §3, §6).

Builds the frozen data artifacts used by tasks A (AISHELL-3 pinyin
transcription), B (OMPAL learner pronunciation labels) and C (OMPAL learner /
native sentence scores):

outputs (under ``data/sentence/manifests/``):
  * ``aishell3_items.jsonl``     task A, 300 utterances / 30 speakers
  * ``ompal_b_items.jsonl``      task B, cleaned learner utterances
  * ``ompal_c_items.jsonl``      task C, cleaned learner utterances
  * ``ompal_native_items.jsonl`` task C, 82 native anchor utterances
  * ``prep_summary.json``        counts, strata, coverage, exclusions, hashes
  * ``README.md``                rebuild instructions + documented deviations

audio (kept in original format, never resampled):
  * ``data/sentence/aishell3/wav/{speaker}/{utt}.wav``  44.1 kHz / 16-bit / mono
  * ``data/sentence/ompal/wav/{SPEAKER}/{utt}.wav``     16 kHz / 16-bit / mono

Stages (all resumable, deterministic for a fixed seed):

    .venv/bin/python scripts/prepare_sentence_data.py all       # metadata + build + download + rebuild
    .venv/bin/python scripts/prepare_sentence_data.py build     # offline: parse / sample / write manifests
    .venv/bin/python scripts/prepare_sentence_data.py download  # fetch missing audio, then rebuild durations
    .venv/bin/python scripts/prepare_sentence_data.py metadata  # fetch missing source JSON (network)

Network: ``urllib`` honours ``HTTPS_PROXY`` / ``HTTP_PROXY`` (and the macOS
system proxy when the variables are unset).  AISHELL-3 wavs are fetched from
the complete mirror first and fall back to the partial one; every attempt is
logged to ``data/sentence/download_log.jsonl`` together with the URL that
finally served the file.

Documented deviations from ``docs/sentence-experiment-design.md`` §3:
  * AISHELL-3 source is the **train split only**.  The HF mirrors ship
    ``train/label_train-set.txt`` (63,262 lines, 174 speakers) but the older
    ``AISHELL/AISHELL-3`` repo hosts only 9,802 of those wavs, so the primary
    wav mirror is ``shenyunhang/AISHELL-3`` (verified complete) with
    ``AISHELL/AISHELL-3`` as fallback.
  * ``align_gold`` rejects more than the two known lines (18 in total); every
    rejected utterance id + reason is recorded in the summary.
  * OMPAL cleaning: the design predicted 12 corrupted word-slot rows + 2
    text/slot mismatches -> 1,754.  The published learner file actually has
    9 corrupted word-slot rows + 1 text/slot mismatch (``00204117``) = 10
    removals -> **1,758** learner utterances; the design's 12+2 describes the
    corpus-wide anomaly count (12 includes 3 native rows and the second text
    mismatch ``00100217`` is native).  See ``prep_summary.json["deviations"]``.
  * ``non-native_scores-detail.json`` re-numbers 656 learner utterances to
    speaker ids 047-066; the aggregated ``non-native_scores.json`` matches the
    released wav tree exactly and is therefore the authoritative source.
    3-rater means / majority votes are still validated against the detail file.
  * Native OMPAL rows keep their 4 anomalous rows (anchor only, all 5.0).
  * ``sentence_id`` meta is an ordinal over the 84 distinct reference texts
    found in the released OMPAL score files (the corpus' internal sentence
    numbering is not recoverable from the release).

Everything is deterministic: sampling uses ``random.Random(SEED)`` over sorted
inputs, manifests are ordered by speaker/utterance id and written with the
frozen :class:`pinyin_eval.sentence.records.SentenceItem` serialiser.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import statistics
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # allow `python scripts/prepare_sentence_data.py`
    sys.path.insert(0, str(_SRC))

from pinyin_eval.normalize import INITIALS  # noqa: E402
from pinyin_eval.sentence.normalize import (  # noqa: E402
    GoldSyllable,
    SpecialFlags,
    align_gold,
    erhua_flags,
    normalize_pinyin_token,
)
from pinyin_eval.sentence.records import SentenceItem, load_jsonl, write_jsonl  # noqa: E402

__all__ = [
    "AISHELL_MIRRORS",
    "AISHELL_TARGET_ITEMS",
    "COVERAGE_FINALS",
    "COVERAGE_INITIALS",
    "COVERAGE_TONES",
    "OMPAL_REPO",
    "OMPAL_RAW_BASE",
    "QUOTAS",
    "SEED",
    "AishellRecord",
    "Coverage",
    "apply_coverage_supplements",
    "build",
    "build_aishell_records",
    "build_aishell_records_section",
    "build_ompal_items",
    "classify_learner_records",
    "compute_zero_initials",
    "coverage_stats",
    "detect_ompal_anomalies",
    "download_audio",
    "ensure_special_subset",
    "main",
    "make_aishell_record",
    "parse_aishell_labels",
    "parse_spk_info",
    "probe_audio",
    "sample_speaker_utterances",
    "select_speakers",
    "sha256_file",
    "solve_strata_counts",
    "speaker_from_utt",
    "words_to_labels",
    "write_outputs",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: preparation script version (bump when the manifest schema changes)
PREP_VERSION = "1.0.0"

#: fixed seed for every sampling decision (design §12)
SEED = 20261101

#: AISHELL-3 (design §3.1)
AISHELL_TARGET_ITEMS = 300
AISHELL_TARGET_SPEAKERS = 30
AISHELL_UTTS_PER_SPEAKER = 10
AISHELL_MIN_CHARS = 5
AISHELL_MAX_CHARS = 30
AISHELL_SPECIAL_MIN = 50

#: AISHELL-3 wav mirrors: complete mirror first, partial official HF repo second.
AISHELL_MIRRORS: tuple[tuple[str, str], ...] = (
    (
        "shenyunhang/AISHELL-3",
        "https://huggingface.co/datasets/shenyunhang/AISHELL-3/resolve/main",
    ),
    (
        "AISHELL/AISHELL-3",
        "https://huggingface.co/datasets/AISHELL/AISHELL-3/resolve/main",
    ),
)

#: expected AISHELL-3 wav format (design §3.1 / §5)
AISHELL_EXPECTED_FORMAT = {"samplerate": 44100, "channels": 1, "subtype": "PCM_16"}
#: observed OMPAL release format (not enforced by the design; recorded + verified uniform)
OMPAL_EXPECTED_FORMAT = {"samplerate": 16000, "channels": 1, "subtype": "PCM_16"}

#: OMPAL source (design §3.2 / §12)
OMPAL_REPO = "phantomhsieh/OMPAL-corpus"
OMPAL_COMMIT = "c3aaa9e892e4b1c2a25b637401f52f6e35b53e74"
OMPAL_RAW_BASE = f"https://raw.githubusercontent.com/{OMPAL_REPO}/main"
OMPAL_REQUIRED_FILES = (
    "non-native_scores.json",
    "non-native_scores-detail.json",
    "native_scores.json",
    "README.md",
)
OMPAL_SCORE_DIMS = ("accuracy", "fluency", "prosody")
OMPAL_WORD_DIMS = ("tone", "phoneme_consonant", "phoneme_vowel")

#: requested speaker quotas for the AISHELL-3 sample ("约" = approximate allowed)
QUOTAS: dict[str, dict[str, int]] = {
    "gender": {"female": 18, "male": 12},
    "accent": {"north": 22, "south": 8},
    "age": {"B": 24, "C": 5, "D": 1},
}

#: canonical category inventories used by the coverage check
COVERAGE_INITIALS: tuple[str, ...] = tuple(INITIALS)
COVERAGE_FINALS: tuple[str, ...] = (
    "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
    "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu",
    "o", "ong", "ou", "u", "ua", "uai", "uan", "uang", "ueng", "ui",
    "un", "uo", "ü", "üan", "üe", "ün",
)
COVERAGE_TONES: tuple[int, ...] = (1, 2, 3, 4)

#: repository-relative roots
DATA_ROOT = Path("data/sentence")
AISHELL_DIR = DATA_ROOT / "aishell3"
OMPAL_DIR = DATA_ROOT / "ompal"
OMPAL_REPO_DIR = OMPAL_DIR / "repo"
MANIFEST_DIR = DATA_ROOT / "manifests"
DOWNLOAD_LOG = DATA_ROOT / "download_log.jsonl"
AUDIO_SOURCES = DATA_ROOT / "audio_sources.json"

_PINYIN_MARKS = "%$"
_LOG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Current UTC timestamp (ISO-8601)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    """sha256 hex digest of a file (streamed)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_stats(durations: Sequence[float]) -> dict[str, float | int]:
    """Descriptive duration statistics (seconds)."""
    values = [float(d) for d in durations]
    if not values:
        return {"n": 0, "total_s": 0.0, "median_s": 0.0, "min_s": 0.0, "max_s": 0.0, "mean_s": 0.0}
    return {
        "n": len(values),
        "total_s": round(sum(values), 3),
        "median_s": round(statistics.median(values), 3),
        "min_s": round(min(values), 3),
        "max_s": round(max(values), 3),
        "mean_s": round(statistics.fmean(values), 3),
    }


def speaker_from_utt(utt_id: str) -> str:
    """OMPAL utterance id -> released speaker directory id.

    ``00200101`` -> ``SPEAKER02001`` (learner), ``00100101`` -> ``SPEAKER01001``
    (native).  Verified against the released ``wav/`` tree.
    """
    if len(utt_id) != 8 or not utt_id.isdigit():
        raise ValueError(f"unexpected OMPAL utterance id: {utt_id!r}")
    return f"SPEAKER{utt_id[1:3]}{utt_id[3:6]}"


def majority_label(values: Sequence[Any]) -> int:
    """Majority vote of 0/1 labels (ties resolve to 1, matching the release)."""
    ints = [int(v) for v in values]
    return 1 if sum(ints) * 2 > len(ints) else 0


def _repo_rel(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:  # synthetic test roots live outside the repository
        return resolved.as_posix()


# ---------------------------------------------------------------------------
# AISHELL-3 parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpkInfo:
    """AISHELL-3 speaker meta row."""

    speaker: str
    age: str
    gender: str
    accent: str

    @property
    def stratum(self) -> str:
        return f"{self.gender}/{self.accent}/{self.age}"


@dataclass(frozen=True, slots=True)
class AishellLine:
    """One validated ``label_train-set.txt`` line (marks already stripped)."""

    utt_id: str
    tokens: tuple[str, ...]
    chars: str


@dataclass(slots=True)
class AishellRecord:
    """An AISHELL-3 utterance that ``align_gold`` could bind char-by-char."""

    utt_id: str
    chars: str
    tokens: tuple[str, ...]
    gold: tuple[GoldSyllable, ...]

    @property
    def speaker(self) -> str:
        return self.utt_id[:7]

    @property
    def flags(self) -> SpecialFlags:
        return erhua_flags(self.chars, list(self.tokens))


def strip_aishell_marks(text: str) -> str:
    """NFKC-normalise, then remove AISHELL-3 prosody marks (``%`` / ``$``)."""
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in cleaned if ch not in _PINYIN_MARKS)
    return cleaned.strip()


def parse_aishell_labels(text: str) -> tuple[list[AishellLine], list[dict[str, Any]]]:
    """Parse ``label_train-set.txt``.

    Format per data line: ``UTT|pinyin with % and $ marks|汉字 with % and $``.
    Returns ``(lines, malformed)``; ``malformed`` records ``{"line_no", "error"}``.
    Comments (``#``) and blank lines are skipped.
    """
    parsed: list[AishellLine] = []
    malformed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("|")
        if len(parts) != 3:
            malformed.append({"line_no": line_no, "error": f"expected 3 fields, got {len(parts)}"})
            continue
        utt_id, pinyin, chars = parts
        chars_clean = strip_aishell_marks(chars)
        chars_clean = re.sub(r"\s+", "", chars_clean)
        pinyin_clean = strip_aishell_marks(pinyin)
        tokens = tuple(tok for tok in pinyin_clean.split() if tok)
        if not utt_id or not tokens or not chars_clean:
            malformed.append({"line_no": line_no, "error": "empty utt_id/pinyin/chars field"})
            continue
        if utt_id in seen:
            malformed.append({"line_no": line_no, "error": f"duplicate utterance id {utt_id}"})
            continue
        seen.add(utt_id)
        parsed.append(AishellLine(utt_id=utt_id, tokens=tokens, chars=chars_clean))
    return parsed, malformed


def parse_spk_info(text: str) -> tuple[dict[str, SpkInfo], list[dict[str, Any]]]:
    """Parse ``spk-info.txt`` (tab separated, ``#`` comments, trailing spaces tolerated)."""
    infos: dict[str, SpkInfo] = {}
    malformed: list[dict[str, Any]] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = [field.strip() for field in stripped.split("\t")]
        if len(fields) != 4:
            malformed.append({"line_no": line_no, "error": f"expected 4 fields, got {len(fields)}"})
            continue
        speaker, age, gender, accent = fields
        infos[speaker] = SpkInfo(speaker=speaker, age=age, gender=gender, accent=accent)
    return infos, malformed


def make_aishell_record(utt_id: str, chars: str, tokens: Sequence[str]) -> AishellRecord:
    """Build an :class:`AishellRecord`, raising when ``align_gold`` rejects it."""
    gold = align_gold(chars, list(tokens))
    if gold is None:
        raise ValueError(f"pinyin cannot be aligned to characters: {utt_id}")
    return AishellRecord(
        utt_id=utt_id, chars=chars, tokens=tuple(tokens), gold=tuple(gold)
    )


def build_aishell_records(
    lines: Iterable[AishellLine],
    *,
    min_chars: int = AISHELL_MIN_CHARS,
    max_chars: int = AISHELL_MAX_CHARS,
) -> dict[str, Any]:
    """Filter AISHELL-3 lines to the eligibility frame.

    Eligibility = 5-30 characters (NFKC, marks stripped) **and** alignable.
    Returns a dict with ``eligible`` (:class:`AishellRecord` list), the
    unalignable ids (with reason), and the length-filtered count.
    """
    eligible: list[AishellRecord] = []
    unalignable: list[dict[str, Any]] = []
    length_filtered = 0
    for line in lines:
        gold = align_gold(line.chars, list(line.tokens))
        if gold is None:
            unalignable.append(
                {
                    "utt_id": line.utt_id,
                    "n_chars": len(line.chars),
                    "n_tokens": len(line.tokens),
                    "reason": "align_gold: token/character counts cannot be reconciled",
                }
            )
            continue
        if not (min_chars <= len(line.chars) <= max_chars):
            length_filtered += 1
            continue
        eligible.append(
            AishellRecord(
                utt_id=line.utt_id,
                chars=line.chars,
                tokens=line.tokens,
                gold=tuple(gold),
            )
        )
    return {
        "eligible": eligible,
        "unalignable": unalignable,
        "length_filtered": length_filtered,
    }


def group_by_speaker(records: Iterable[AishellRecord]) -> dict[str, list[AishellRecord]]:
    """Speaker -> records sorted by utterance id."""
    pool: dict[str, list[AishellRecord]] = defaultdict(list)
    for record in records:
        pool[record.speaker].append(record)
    for speaker in pool:
        pool[speaker].sort(key=lambda record: record.utt_id)
    return dict(pool)


# ---------------------------------------------------------------------------
# AISHELL-3 stratified speaker selection
# ---------------------------------------------------------------------------


def _compatible(info: SpkInfo) -> bool:
    return (
        info.gender in QUOTAS["gender"]
        and info.accent in QUOTAS["accent"]
        and info.age in QUOTAS["age"]
    )


def candidate_speakers(
    pool: dict[str, list[AishellRecord]],
    spk_info: dict[str, SpkInfo],
    *,
    min_utts: int = AISHELL_UTTS_PER_SPEAKER,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Speakers with at least ``min_utts`` eligible utterances.

    Speakers outside the requested gender/accent/age values are returned as
    exclusions (``compatible=False``) so the summary can document them.
    """
    candidates: list[str] = []
    exclusions: list[dict[str, Any]] = []
    for speaker in sorted(pool):
        n_utts = len(pool[speaker])
        info = spk_info.get(speaker)
        if info is None:
            exclusions.append({"speaker": speaker, "reason": "missing spk-info row"})
            continue
        if n_utts < min_utts:
            exclusions.append(
                {"speaker": speaker, "reason": f"only {n_utts} eligible utterances (< {min_utts})"}
            )
            continue
        if not _compatible(info):
            exclusions.append(
                {
                    "speaker": speaker,
                    "reason": f"attribute outside quotas ({info.stratum})",
                }
            )
            continue
        candidates.append(speaker)
    return candidates, exclusions


def _search_counts(
    categories: list[tuple[str, str, str]],
    sizes: dict[tuple[str, str, str], int],
    targets: dict[str, dict[str, int]],
    slack: int,
    total_target: int,
    idx: int,
    used: dict[str, dict[str, int]],
    solution: dict[tuple[str, str, str], int],
) -> bool:
    """Depth-first search for per-stratum speaker counts within ``slack``."""
    for dimension, target_map in targets.items():
        for key, target in target_map.items():
            if used[dimension][key] > target + slack:
                return False
    if idx == len(categories):
        if sum(used["gender"].values()) != total_target:
            return False
        for dimension, target_map in targets.items():
            for key, target in target_map.items():
                if abs(used[dimension][key] - target) > slack:
                    return False
        return True
    category = categories[idx]
    gender, accent, age = category
    remaining_total = total_target - sum(used["gender"].values())
    budget = min(sizes[category], remaining_total)
    for count in range(budget, -1, -1):
        solution[category] = count
        used["gender"][gender] += count
        used["accent"][accent] += count
        used["age"][age] += count
        if _search_counts(categories, sizes, targets, slack, total_target, idx + 1, used, solution):
            return True
        used["gender"][gender] -= count
        used["accent"][accent] -= count
        used["age"][age] -= count
    solution.pop(category, None)
    return False


def solve_strata_counts(
    category_sizes: dict[tuple[str, str, str], int],
    quotas: dict[str, dict[str, int]] = QUOTAS,
    *,
    max_slack: int = 3,
) -> tuple[dict[tuple[str, str, str], int], int] | None:
    """Deterministic exact/approximate count solution for the speaker quotas.

    ``category_sizes`` maps ``(gender, accent, age)`` to the number of
    available speakers.  Tries ``slack=0, 1, ... max_slack`` and returns the
    first feasible assignment ``(counts, slack)`` (``None`` when infeasible).
    """
    categories = sorted(category_sizes)
    if not categories:
        return None
    targets = {name: dict(quota) for name, quota in quotas.items()}
    total_target = sum(quotas["gender"].values())
    for slack in range(max_slack + 1):
        solution: dict[tuple[str, str, str], int] = {}
        used = {name: dict.fromkeys(target, 0) for name, target in targets.items()}
        if _search_counts(categories, category_sizes, targets, slack, total_target, 0, used, solution):
            return dict(solution), slack
    return None


def select_speakers(
    pool: dict[str, list[AishellRecord]],
    spk_info: dict[str, SpkInfo],
    *,
    seed: int = SEED,
    quotas: dict[str, dict[str, int]] = QUOTAS,
) -> dict[str, Any]:
    """Stratified deterministic speaker selection (design §3.1).

    Returns ``{"speakers", "counts", "achieved", "requested", "slack",
    "candidates", "exclusions"}``; raises ``RuntimeError`` when no assignment
    exists even with slack (callers should treat that as a hard data error).
    """
    candidates, exclusions = candidate_speakers(pool, spk_info)
    by_category: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for speaker in candidates:
        info = spk_info[speaker]
        by_category[(info.gender, info.accent, info.age)].append(speaker)

    solved = solve_strata_counts({category: len(items) for category, items in by_category.items()}, quotas)
    if solved is None:
        raise RuntimeError("speaker quotas are infeasible for the available AISHELL-3 speakers")
    counts, slack = solved

    rng = random.Random(seed)
    selected: list[str] = []
    for category in sorted(by_category):
        n_pick = counts.get(category, 0)
        if n_pick:
            selected.extend(sorted(rng.sample(sorted(by_category[category]), n_pick)))
    selected.sort()

    achieved = {
        "gender": Counter(spk_info[s].gender for s in selected),
        "accent": Counter(spk_info[s].accent for s in selected),
        "age": Counter(spk_info[s].age for s in selected),
    }
    return {
        "speakers": selected,
        "counts": {f"{category[0]}/{category[1]}/{category[2]}": n for category, n in sorted(counts.items())},
        "achieved": {key: dict(sorted(value.items())) for key, value in achieved.items()},
        "requested": {key: dict(value) for key, value in quotas.items()},
        "slack": slack,
        "candidates": candidates,
        "exclusions": exclusions,
        "by_category": {f"{c[0]}/{c[1]}/{c[2]}": sorted(v) for c, v in sorted(by_category.items())},
    }


def sample_speaker_utterances(
    pool: dict[str, list[AishellRecord]],
    speakers: Sequence[str],
    *,
    seed: int = SEED,
    per_speaker: int = AISHELL_UTTS_PER_SPEAKER,
) -> list[AishellRecord]:
    """Sample ``per_speaker`` utterances from each selected speaker (deterministic)."""
    rng = random.Random(seed)
    sampled: list[AishellRecord] = []
    for speaker in sorted(speakers):
        pool_records = pool[speaker]
        if len(pool_records) < per_speaker:
            raise ValueError(f"speaker {speaker} has only {len(pool_records)} eligible utterances")
        sampled.extend(sorted(rng.sample(pool_records, per_speaker), key=lambda record: record.utt_id))
    sampled.sort(key=lambda record: (record.speaker, record.utt_id))
    return sampled


# ---------------------------------------------------------------------------
# AISHELL-3 coverage / special-subset checks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Coverage:
    """Category coverage over a set of gold-bound utterances."""

    initials: Counter[str]
    finals: Counter[str]
    tones: Counter[int]

    @property
    def missing_initials(self) -> list[str]:
        return [x for x in COVERAGE_INITIALS if self.initials[x] == 0]

    @property
    def missing_finals(self) -> list[str]:
        return [x for x in COVERAGE_FINALS if self.finals[x] == 0]

    @property
    def missing_tones(self) -> list[int]:
        return [x for x in COVERAGE_TONES if self.tones[x] == 0]

    @property
    def complete(self) -> bool:
        return not (self.missing_initials or self.missing_finals or self.missing_tones)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_initials_present": len([x for x in COVERAGE_INITIALS if self.initials[x]]),
            "n_finals_present": len([x for x in COVERAGE_FINALS if self.finals[x]]),
            "n_tones_present": len([x for x in COVERAGE_TONES if self.tones[x]]),
            "initials": dict(sorted(self.initials.items())),
            "finals": dict(sorted(self.finals.items())),
            "tones": {str(tone): self.tones[tone] for tone in COVERAGE_TONES},
            "missing_initials": self.missing_initials,
            "missing_finals": self.missing_finals,
            "missing_tones": self.missing_tones,
            "complete": self.complete,
        }


def coverage_stats(records: Iterable[AishellRecord]) -> Coverage:
    """Count initial / final / tone(1-4) categories over gold syllables."""
    initials: Counter[str] = Counter()
    finals: Counter[str] = Counter()
    tones: Counter[int] = Counter()
    for record in records:
        for syllable in record.gold:
            if syllable.initial:
                initials[syllable.initial] += 1
            finals[syllable.final] += 1
            if syllable.tone in COVERAGE_TONES:
                tones[syllable.tone] += 1
    return Coverage(initials=initials, finals=finals, tones=tones)


def occurrence_categories(record: AishellRecord) -> set[tuple[str, Any]]:
    """Coverage categories (initial/final/tone) touched by one record."""
    categories: set[tuple[str, Any]] = set()
    for syllable in record.gold:
        if syllable.initial:
            categories.add(("initial", syllable.initial))
        categories.add(("final", syllable.final))
        if syllable.tone in COVERAGE_TONES:
            categories.add(("tone", syllable.tone))
    return categories


def _missing_categories(coverage: Coverage) -> list[tuple[str, Any]]:
    missing: list[tuple[str, Any]] = [("initial", value) for value in coverage.missing_initials]
    missing += [("final", value) for value in coverage.missing_finals]
    missing += [("tone", value) for value in coverage.missing_tones]
    return missing


def _rebuild_victim_items(
    pool: dict[str, list[AishellRecord]],
    speaker: str,
    forced: AishellRecord,
    per_speaker: int,
) -> list[AishellRecord]:
    """Ten-item replacement for a swapped speaker that includes ``forced``."""
    others = [record for record in sorted(pool[speaker], key=lambda r: r.utt_id) if record.utt_id != forced.utt_id]
    return [forced, *others[: per_speaker - 1]]


def apply_coverage_supplements(
    records: list[AishellRecord],
    pool: dict[str, list[AishellRecord]],
    spk_info: dict[str, SpkInfo],
    *,
    per_speaker: int = AISHELL_UTTS_PER_SPEAKER,
    max_iterations: int = 64,
) -> tuple[list[AishellRecord], list[dict[str, Any]]]:
    """Swap in utterances until every coverage category appears.

    For a missing category the script first tries to swap a *redundant* item of
    an already selected speaker (keeping 10 utterances per speaker), then — if
    no selected speaker's pool contains the category — swaps a whole speaker
    inside the same stratum (keeping the quota table intact).  Every swap is
    recorded in the returned supplement list and the added item's
    ``meta.coverage_supplement`` flag is set at manifest build time.
    """
    records = list(records)
    supplements: list[dict[str, Any]] = []
    for _iteration in range(max_iterations):
        coverage = coverage_stats(records)
        missing = _missing_categories(coverage)
        if not missing:
            break
        progress = False
        for category, value in missing:
            selected_speakers = sorted({record.speaker for record in records})
            candidate: tuple[str, AishellRecord] | None = None
            for speaker in selected_speakers:
                for record in pool[speaker]:
                    if record.utt_id in {r.utt_id for r in records}:
                        continue
                    if (category, value) in occurrence_categories(record):
                        candidate = (speaker, record)
                        break
                if candidate is not None:
                    break
            if candidate is not None:
                speaker, added = candidate
                for drop_index, dropped in enumerate(records):
                    if dropped.speaker != speaker:
                        continue
                    trial = [record for index, record in enumerate(records) if index != drop_index]
                    if coverage_stats([*trial, added]).complete:
                        records = trial + [added]
                        records.sort(key=lambda record: (record.speaker, record.utt_id))
                        supplements.append(
                            {
                                "kind": "same_speaker_swap",
                                "category": category,
                                "value": value,
                                "speaker": speaker,
                                "removed_utt": dropped.utt_id,
                                "added_utt": added.utt_id,
                            }
                        )
                        progress = True
                        break
            if progress:
                break
            # Fall back to replacing a whole speaker inside the same stratum.
            strata: dict[str, list[str]] = defaultdict(list)
            for speaker in selected_speakers:
                strata[spk_info[speaker].stratum].append(speaker)
            swapped = False
            for stratum in sorted(strata):
                for candidate_speaker in sorted(
                    speaker for speaker in pool if spk_info[speaker].stratum == stratum
                ):
                    if candidate_speaker in selected_speakers:
                        continue
                    forced = next(
                        (
                            record
                            for record in pool[candidate_speaker]
                            if (category, value) in occurrence_categories(record)
                        ),
                        None,
                    )
                    if forced is None:
                        continue
                    victim = sorted(strata[stratum])[0]
                    trial = [record for record in records if record.speaker != victim]
                    trial.extend(_rebuild_victim_items(pool, candidate_speaker, forced, per_speaker))
                    trial.sort(key=lambda record: (record.speaker, record.utt_id))
                    if coverage_stats(trial).complete:
                        records = trial
                        supplements.append(
                            {
                                "kind": "speaker_swap",
                                "category": category,
                                "value": value,
                                "stratum": stratum,
                                "removed_speaker": victim,
                                "added_speaker": candidate_speaker,
                                "forced_utt": forced.utt_id,
                            }
                        )
                        progress = True
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break
        if not progress:
            break
    return records, supplements


def ensure_special_subset(
    records: list[AishellRecord],
    pool: dict[str, list[AishellRecord]],
    *,
    min_special: int = AISHELL_SPECIAL_MIN,
    per_speaker: int = AISHELL_UTTS_PER_SPEAKER,
) -> tuple[list[AishellRecord], list[dict[str, Any]]]:
    """Guarantee at least ``min_special`` items carry a neutral/一不/erhua flag."""
    records = list(records)
    selected_ids = {record.utt_id for record in records}
    count = sum(1 for record in records if record.flags.special)
    supplements: list[dict[str, Any]] = []
    if count >= min_special:
        return records, supplements
    for speaker in sorted({record.speaker for record in records}):
        for candidate in pool[speaker]:
            if count >= min_special:
                break
            if candidate.utt_id in selected_ids or not candidate.flags.special:
                continue
            non_special = [
                index
                for index, record in enumerate(records)
                if record.speaker == speaker and not record.flags.special
            ]
            if not non_special:
                continue
            drop_index = non_special[0]
            dropped_utt = records[drop_index].utt_id
            trial = [record for index, record in enumerate(records) if index != drop_index]
            if not coverage_stats(trial).complete:
                continue
            records = trial + [candidate]
            records.sort(key=lambda record: (record.speaker, record.utt_id))
            supplements.append(
                {
                    "kind": "same_speaker_swap",
                    "reason": "special_subset",
                    "speaker": speaker,
                    "removed_utt": dropped_utt,
                    "added_utt": candidate.utt_id,
                }
            )
            selected_ids.discard(dropped_utt)
            selected_ids.add(candidate.utt_id)
            count += 1
    return records, supplements


# ---------------------------------------------------------------------------
# Audio probing / downloading
# ---------------------------------------------------------------------------


def probe_audio(path: str | Path) -> dict[str, Any]:
    """Read wav metadata with soundfile (never raises)."""
    file_path = Path(path)
    if not file_path.exists():
        return {"ok": False, "error": "missing file"}
    try:
        import soundfile as sf  # local import keeps the parse/sample path dependency-free

        info = sf.info(str(file_path))
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        return {
            "ok": True,
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
            "subtype": str(info.subtype),
            "frames": int(info.frames),
            "duration_s": round(duration, 6),
        }
    except Exception as exc:  # noqa: BLE001 - report, do not abort the build
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _download_file(
    urls: Sequence[str],
    dest: Path,
    *,
    retries: int = 6,
    timeout: float = 180.0,
    user_agent: str = "pinyin-eval-data-prep/1.0",
) -> dict[str, Any]:
    """Download one file, trying ``urls`` in order; resumable and retrying."""
    if dest.exists() and dest.stat().st_size > 0:
        return {"status": "skip", "url_used": None, "bytes": dest.stat().st_size, "attempts": 0, "errors": []}
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    for url_index, url in enumerate(urls):
        for attempt in range(1, retries + 1):
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": user_agent})
                started = time.time()
                with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        handle.write(chunk)
                size = tmp.stat().st_size
                if size == 0:
                    raise OSError("empty response body")
                tmp.replace(dest)
                return {
                    "status": "ok",
                    "url_used": url,
                    "url_index": url_index,
                    "bytes": size,
                    "attempts": attempt,
                    "elapsed_s": round(time.time() - started, 3),
                    "errors": errors,
                }
            except Exception as exc:  # noqa: BLE001 - retry / fall back
                errors.append({"url": url, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}"})
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(min(2**attempt * 0.5, 15.0))
        # fall through to the next mirror after exhausting the retries
    return {"status": "fail", "url_used": None, "bytes": 0, "attempts": retries * len(urls), "errors": errors}


def _source_label(source: str) -> str:
    """Human-readable provenance label for one wav (mirror name, not full URL)."""
    if source in {"pre-existing", "unknown"}:
        return source
    for name, base in AISHELL_MIRRORS:
        if source.startswith(f"{base}/"):
            return name
    if source.startswith(f"{OMPAL_RAW_BASE}/"):
        return "raw.githubusercontent"
    return source.split("?")[0]


def download_audio(
    root: Path = REPO_ROOT,
    *,
    concurrency: int = 12,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Download every wav referenced by the four manifests (resumable).

    Existing files are skipped, so interrupted runs can simply be re-started.
    A JSONL provenance log (source URL / attempts / errors) is appended for
    every file.
    """
    manifest_dir = root / MANIFEST_DIR
    items: list[dict[str, Any]] = []
    for name in ("aishell3_items.jsonl", "ompal_b_items.jsonl", "ompal_native_items.jsonl"):
        items.extend(load_jsonl(manifest_dir / name))
    om_wav_root = (root / OMPAL_DIR / "wav").resolve()
    aishell_wav_root = (root / AISHELL_DIR / "wav").resolve()

    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        audio_path = item["audio_path"]
        if audio_path in seen:
            continue
        seen.add(audio_path)
        dest = (root / audio_path).resolve()
        if str(dest).startswith(str(om_wav_root)):
            rel = dest.relative_to(om_wav_root).as_posix()
            urls = [f"{OMPAL_RAW_BASE}/wav/{rel}"]
            task = "ompal"
        elif str(dest).startswith(str(aishell_wav_root)):
            rel = dest.relative_to(aishell_wav_root).as_posix()
            urls = [f"{base}/train/wav/{rel}" for _name, base in AISHELL_MIRRORS]
            task = "aishell3"
        else:
            raise ValueError(f"unexpected audio path outside the data roots: {audio_path}")
        jobs.append({"task": task, "audio_path": audio_path, "rel": rel, "dest": dest, "urls": urls})

    run_id = utc_now()
    log_file = Path(log_path) if log_path is not None else root / DOWNLOAD_LOG
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Per-file provenance: seed from a previous sidecar / download log so a
    # resumable run never loses the source of an already-present wav.
    sources_file = root / AUDIO_SOURCES
    provenance: dict[str, str] = {}
    if sources_file.exists():
        try:
            provenance = json.loads(sources_file.read_text(encoding="utf-8"))
        except ValueError:
            provenance = {}
    elif log_file.exists():
        try:
            for line in log_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("status") == "ok" and entry.get("url_used"):
                    provenance.setdefault(str(entry["audio_path"]), str(entry["url_used"]))
        except ValueError:
            provenance = {}

    stats: dict[str, Any] = {
        "run_id": run_id,
        "log": _repo_rel(log_file) if log_file.is_absolute() else log_file.as_posix(),
        "total": len(jobs),
        "by_task": defaultdict(Counter),
        "failures": [],
        "sources": defaultdict(Counter),
        "bytes": Counter(),
    }

    def worker(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = _download_file(job["urls"], job["dest"])
        return job, result

    with log_file.open("a", encoding="utf-8") as log_handle, concurrent.futures.ThreadPoolExecutor(concurrency) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            job, result = future.result()
            entry = {
                "run_id": run_id,
                "task": job["task"],
                "audio_path": job["audio_path"],
                "status": result["status"],
                "url_used": result.get("url_used"),
                "attempts": result["attempts"],
                "bytes": result["bytes"],
                "errors": result["errors"],
                "timestamp": utc_now(),
            }
            with _LOG_LOCK:
                log_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_handle.flush()
            stats["by_task"][job["task"]][result["status"]] += 1
            if result["status"] == "fail":
                stats["failures"].append(
                    {"audio_path": job["audio_path"], "urls": list(job["urls"]), "errors": result["errors"]}
                )
            else:
                stats["bytes"][job["task"]] += result["bytes"]
                if result["status"] == "ok":
                    source = str(AISHELL_MIRRORS[result.get("url_index", 0)][0]) if job["task"] == "aishell3" else "raw.githubusercontent"
                    stats["sources"][job["task"]][source] += 1
                    provenance[job["audio_path"]] = str(result["url_used"])
                else:
                    stats["sources"][job["task"]]["pre-existing"] += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  download {done}/{len(jobs)} ({job['task']}) ...", flush=True)

    # Persist per-file provenance so a rebuild (or a later resumable run) keeps
    # the source of every already-downloaded wav.
    for job in jobs:
        if job["dest"].exists():
            provenance.setdefault(job["audio_path"], "pre-existing")
    for entry in stats["failures"]:
        provenance.pop(entry["audio_path"], None)
    # keep the sidecar in sync with the current sample (drop stale paths)
    current_paths = {job["audio_path"] for job in jobs}
    provenance = {path: source for path, source in provenance.items() if path in current_paths}
    sources_file.parent.mkdir(parents=True, exist_ok=True)
    sources_file.write_text(
        json.dumps(dict(sorted(provenance.items())), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    source_counts = Counter(_source_label(source) for source in provenance.values())

    return {
        "run_id": run_id,
        "total": stats["total"],
        "by_task": {task: dict(counter) for task, counter in stats["by_task"].items()},
        "failures": stats["failures"],
        "sources": {task: dict(counter) for task, counter in stats["sources"].items()},
        "bytes": {task: int(value) for task, value in stats["bytes"].items()},
        "provenance_file": _repo_rel(sources_file),
        "provenance_counts": dict(sorted(source_counts.items())),
    }


def ensure_metadata(root: Path = REPO_ROOT, *, verbose: bool = True) -> list[str]:
    """Fetch missing OMPAL source JSON files from GitHub raw (network stage)."""
    repo_dir = root / OMPAL_REPO_DIR
    repo_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in OMPAL_REQUIRED_FILES if not (repo_dir / name).exists()]
    for name in missing:
        dest = repo_dir / name
        result = _download_file([f"{OMPAL_RAW_BASE}/{name}"], dest)
        if verbose:
            print(f"  metadata {name}: {result['status']} ({result['bytes']} bytes)")
        if result["status"] == "fail":
            raise RuntimeError(f"could not fetch OMPAL metadata file {name!r}: {result['errors'][-1:]}")
    return missing


# ---------------------------------------------------------------------------
# OMPAL parsing / cleaning
# ---------------------------------------------------------------------------


def detect_ompal_anomalies(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Scan the released OMPAL score records for word-slot / text corruption.

    Returns lists of ``{"utt_id", ...}`` dicts for every anomaly class that
    matters for cleaning (design §3.2): multi-character word slots, word/text
    join mismatches, empty slots and labels outside ``{0, 1}``.
    """
    multi_char_slots: list[dict[str, Any]] = []
    text_mismatches: list[dict[str, Any]] = []
    empty_slots: list[dict[str, Any]] = []
    invalid_labels: list[dict[str, Any]] = []
    for utt_id in sorted(records):
        record = records[utt_id]
        text = record.get("text", "")
        words = record.get("words", [])
        joined = "".join("".join(word.get("text", [])) for word in words)
        if joined != text:
            text_mismatches.append(
                {
                    "utt_id": utt_id,
                    "reason": "joined word-slot characters differ from the text field",
                    "text": text,
                    "joined": joined,
                    "n_chars_text": len(text),
                    "n_chars_words": sum(len(word.get("text", [])) for word in words),
                }
            )
        for slot_index, word in enumerate(words):
            slot_text = list(word.get("text", []))
            if not slot_text:
                empty_slots.append({"utt_id": utt_id, "slot_index": slot_index})
            if len(slot_text) > 1:
                multi_char_slots.append(
                    {"utt_id": utt_id, "slot_index": slot_index, "slot_text": slot_text}
                )
            for dim in OMPAL_WORD_DIMS:
                raw = word.get(dim)
                values = raw if isinstance(raw, list) else [raw]
                for value in values:
                    if value is None or str(value) not in {"0", "1"}:
                        invalid_labels.append(
                            {"utt_id": utt_id, "slot_index": slot_index, "dim": dim, "value": value}
                        )
    return {
        "multi_char_word_slot_rows": sorted({entry["utt_id"] for entry in multi_char_slots}),
        "multi_char_word_slots": multi_char_slots,
        "text_join_mismatch_rows": text_mismatches,
        "empty_word_slot_rows": sorted({entry["utt_id"] for entry in empty_slots}),
        "invalid_label_rows": sorted({entry["utt_id"] for entry in invalid_labels}),
    }


def classify_learner_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Split learner records into ``kept`` ids and documented exclusions.

    Excluded rows are exactly the rows whose word slots cannot be aligned one
    character per slot to the reference text:
      * ``corrupted_word_slot`` — a slot holds more than one character;
      * ``text_slot_mismatch`` — the joined slots are not the text field.
    (The two classes overlap for multi-character slots; the reason string then
    lists both, and every removed utt id is recorded.)
    """
    anomalies = detect_ompal_anomalies(records)
    multi = set(anomalies["multi_char_word_slot_rows"])
    mismatch = {entry["utt_id"] for entry in anomalies["text_join_mismatch_rows"]}
    empty = set(anomalies["empty_word_slot_rows"])
    invalid = set(anomalies["invalid_label_rows"])
    details = {entry["utt_id"]: entry for entry in anomalies["text_join_mismatch_rows"]}
    exclusions: list[dict[str, Any]] = []
    kept: list[str] = []
    for utt_id in sorted(records):
        reasons: list[str] = []
        if utt_id in multi or utt_id in empty:
            reasons.append("corrupted_word_slot")
        if utt_id in mismatch:
            reasons.append("text_slot_mismatch")
        if utt_id in invalid:
            reasons.append("invalid_word_label")
        if reasons:
            exclusion = {"utt_id": utt_id, "reasons": reasons}
            if utt_id in details:
                exclusion.update(
                    {
                        "text": details[utt_id]["text"],
                        "joined": details[utt_id]["joined"],
                        "n_chars_text": details[utt_id]["n_chars_text"],
                        "n_chars_words": details[utt_id]["n_chars_words"],
                    }
                )
            exclusions.append(exclusion)
        else:
            kept.append(utt_id)
    return {"kept": kept, "exclusions": exclusions, "anomalies": anomalies}


def check_aggregated_scores(
    aggregated: dict[str, dict[str, Any]],
    detail: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate the aggregated score file against the per-rater detail file.

    Design §3.2 claims the released sentence scores are 3-rater means and the
    word labels are 3-rater majority votes.  This checks that claim on the
    1,112 directly matching ids and — accounting for the detail file's
    re-numbered ids — on every record via global signature/mean containment.
    """
    overlap = sorted(set(aggregated) & set(detail))
    score_mismatches: list[dict[str, Any]] = []
    label_mismatches: list[dict[str, Any]] = []
    for utt_id in overlap:
        agg = aggregated[utt_id]
        det = detail[utt_id]
        for dim in OMPAL_SCORE_DIMS:
            det_values = det[dim]
            det_mean = round(sum(float(value) for value in det_values) / len(det_values), 2)
            if det_mean != round(float(agg[dim]), 2):
                score_mismatches.append({"utt_id": utt_id, "dim": dim, "agg": agg[dim], "detail": det_values})
        for agg_word, det_word in zip(agg["words"], det["words"], strict=False):
            for dim in OMPAL_WORD_DIMS:
                det_values = det_word[dim]
                det_majority = majority_label(det_values)
                if int(agg_word[dim]) != det_majority:
                    label_mismatches.append(
                        {"utt_id": utt_id, "dim": dim, "agg": agg_word[dim], "detail": det_values}
                    )
    det_means_by_text: dict[str, set[tuple[float, float, float]]] = defaultdict(set)
    for det in detail.values():
        det_means_by_text[det["text"]].add(tuple(round(sum(float(v) for v in det[dim]) / len(det[dim]), 2) for dim in OMPAL_SCORE_DIMS))
    score_triples_missing = [
        utt_id
        for utt_id, agg in aggregated.items()
        if tuple(round(float(agg[dim]), 2) for dim in OMPAL_SCORE_DIMS) not in det_means_by_text[agg["text"]]
    ]
    det_signatures = {
        (det["text"], tuple(tuple(word["text"]) for word in det["words"]), tuple(tuple(majority_label(word[dim]) for dim in OMPAL_WORD_DIMS) for word in det["words"]))
        for det in detail.values()
    }
    sig_missing = [
        utt_id
        for utt_id, agg in aggregated.items()
        if (agg["text"], tuple(tuple(word["text"]) for word in agg["words"]), tuple(tuple(int(word[dim]) for dim in OMPAL_WORD_DIMS) for word in agg["words"]))
        not in det_signatures
    ]
    only_detail = sorted(set(detail) - set(aggregated))
    only_aggregated = sorted(set(aggregated) - set(detail))
    return {
        "aggregated_records": len(aggregated),
        "detail_records": len(detail),
        "direct_id_overlap": len(overlap),
        "direct_id_score_mean_mismatches": score_mismatches[:20],
        "n_direct_id_score_mean_mismatches": len(score_mismatches),
        "direct_id_label_majority_mismatches": label_mismatches[:20],
        "n_direct_id_label_majority_mismatches": len(label_mismatches),
        "score_triples_not_found_in_detail_means": score_triples_missing[:20],
        "n_score_triples_not_found_in_detail_means": len(score_triples_missing),
        "aggregated_signatures_not_found_in_detail": sig_missing[:20],
        "n_aggregated_signatures_not_found_in_detail": len(sig_missing),
        "detail_only_ids": len(only_detail),
        "aggregated_only_ids": len(only_aggregated),
        "detail_only_id_note": (
            "non-native_scores-detail.json re-numbers 656 learner utterances to speaker ids "
            "047-066 and compacts their utterance numbers; the aggregated file matches the wav tree"
        ),
        "rater_count": 3,
        "aggregated_is_mean_of_raters": len(score_mismatches) == 0 and len(score_triples_missing) == 0,
        "aggregated_is_majority_vote": len(label_mismatches) == 0 and len(sig_missing) == 0,
    }


def compute_zero_initials(text: str) -> list[bool]:
    """Canonical zero-initial flag per character, using pypinyin with context.

    ``pypinyin.lazy_pinyin(text, style=Style.TONE3)`` is context aware (多音字
    disambiguation) and yields one syllable per character; the syllable is then
    split with the frozen :func:`normalize_pinyin_token` rules so that the
    zero-initial definition matches the rest of the project (y/w rewrites).
    Raises ``RuntimeError`` when pypinyin is unavailable.
    """
    try:
        from pypinyin import Style, lazy_pinyin
    except Exception as exc:  # noqa: BLE001 - documented fallback in the caller
        raise RuntimeError("pypinyin is required to compute zero_initial flags") from exc
    syllables = lazy_pinyin(text, style=Style.TONE3, errors="default")
    if len(syllables) != len(text):
        syllables = [lazy_pinyin(char, style=Style.TONE3, errors="default")[0] for char in text]
    return [normalize_pinyin_token(str(syllable)).initial == "" for syllable in syllables]


def words_to_labels(
    words: Sequence[dict[str, Any]],
    zero_initials: Sequence[bool],
) -> list[dict[str, Any]]:
    """Convert OMPAL word slots to the per-character B labels (all slots 1-char)."""
    if len(words) != len(zero_initials):
        raise ValueError("zero_initial flags do not match the number of word slots")
    labels: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        slot_text = list(word["text"])
        if len(slot_text) != 1:
            raise ValueError(f"word slot {index} is not a single character: {slot_text!r}")
        labels.append(
            {
                "text": slot_text[0],
                "tone": int(word["tone"]),
                "consonant": int(word["phoneme_consonant"]),
                "vowel": int(word["phoneme_vowel"]),
                "zero_initial": bool(zero_initials[index]),
            }
        )
    return labels


def build_sentence_inventory(
    learner: dict[str, dict[str, Any]],
    native: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Map every distinct released reference text to a stable ``Sxx`` id.

    The release does not expose the corpus' internal sentence numbering, so the
    inventory is the sorted set of distinct ``text`` values found in the score
    files (84 entries); the summary records the id -> text mapping.
    """
    texts = sorted({record["text"] for record in learner.values()} | {record["text"] for record in native.values()})
    return {text: f"S{index:02d}" for index, text in enumerate(texts)}


def _ompal_record_score(record: dict[str, Any]) -> dict[str, float]:
    return {dim: round(float(record[dim]), 4) for dim in OMPAL_SCORE_DIMS}


def build_ompal_items(
    learner_raw: dict[str, dict[str, Any]],
    native_raw: dict[str, dict[str, Any]],
    *,
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Clean the learner set and build B / C / native manifests."""
    cleaned = classify_learner_records(learner_raw)
    inventory = build_sentence_inventory(learner_raw, native_raw)
    audio_root = root / OMPAL_DIR / "wav"

    def audio_meta(utt_id: str) -> tuple[str, float, dict[str, Any]]:
        speaker = speaker_from_utt(utt_id)
        relative = f"{speaker}/{utt_id}.wav"
        path = audio_root / relative
        probe = probe_audio(path)
        return _repo_rel(path), float(probe.get("duration_s", 0.0)) if probe["ok"] else 0.0, probe

    b_items: list[SentenceItem] = []
    c_items: list[SentenceItem] = []
    audio_reports: list[dict[str, Any]] = []
    zero_initial_failures = 0
    total_units = 0
    zero_initial_units = 0
    for utt_id in cleaned["kept"]:
        record = learner_raw[utt_id]
        speaker = speaker_from_utt(utt_id)
        audio_path, duration, probe = audio_meta(utt_id)
        audio_reports.append({"audio_path": audio_path, "probe": probe})
        try:
            flags = compute_zero_initials(record["text"])
        except RuntimeError:
            zero_initial_failures += 1
            flags = [False] * len(record["words"])
        labels = words_to_labels(record["words"], flags)
        total_units += len(labels)
        zero_initial_units += sum(1 for label in labels if label["zero_initial"])
        sentence_id = inventory.get(record["text"], "")
        chars = "".join("".join(word["text"]) for word in record["words"])
        common_meta = {
            "utt_id": utt_id,
            "native": False,
            "learner_id": speaker,
            "sentence_id": sentence_id,
            "n_units": len(labels),
            "rater_count": 3,
        }
        optional_flags = {"native": False, "learner": True}
        b_items.append(
            SentenceItem(
                item_id=f"B|{utt_id}",
                task="B",
                source="ompal",
                speaker=speaker,
                audio_path=audio_path,
                duration_s=duration,
                text=record["text"],
                chars=chars,
                labels=labels,
                learner=speaker,
                flags=optional_flags,
                meta=common_meta,
            )
        )
        c_items.append(
            SentenceItem(
                item_id=f"C|{utt_id}",
                task="C",
                source="ompal",
                speaker=speaker,
                audio_path=audio_path,
                duration_s=duration,
                text=record["text"],
                chars=chars,
                scores=_ompal_record_score(record),
                learner=speaker,
                flags=dict(optional_flags),
                meta=dict(common_meta, labels_in_b=True),
            )
        )

    native_items: list[SentenceItem] = []
    native_anomalies = detect_ompal_anomalies(native_raw)
    for utt_id in sorted(native_raw):
        record = native_raw[utt_id]
        speaker = speaker_from_utt(utt_id)
        audio_path, duration, probe = audio_meta(utt_id)
        audio_reports.append({"audio_path": audio_path, "probe": probe})
        native_items.append(
            SentenceItem(
                item_id=f"C|{utt_id}",
                task="C",
                source="ompal",
                speaker=speaker,
                audio_path=audio_path,
                duration_s=duration,
                text=record["text"],
                chars=record["text"],
                scores=_ompal_record_score(record),
                native=True,
                flags={"native": True, "learner": False},
                meta={
                    "utt_id": utt_id,
                    "native": True,
                    "learner_id": "",
                    "sentence_id": inventory.get(record["text"], ""),
                    "rater_count": 3,
                },
            )
        )

    return {
        "b_items": b_items,
        "c_items": c_items,
        "native_items": native_items,
        "cleaned": cleaned,
        "inventory": inventory,
        "audio_reports": audio_reports,
        "zero_initial_failures": zero_initial_failures,
        "zero_initial_units": zero_initial_units,
        "total_units": total_units,
        "native_anomalies": native_anomalies,
    }


# ---------------------------------------------------------------------------
# AISHELL-3 item construction
# ---------------------------------------------------------------------------


def build_aishell_records_section(
    root: Path = REPO_ROOT,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    """Parse AISHELL-3 metadata and produce the stratified 300-item sample."""
    label_path = root / AISHELL_DIR / "label_train-set.txt"
    spk_path = root / AISHELL_DIR / "spk-info.txt"
    if not label_path.exists() or not spk_path.exists():
        raise FileNotFoundError(
            f"AISHELL-3 metadata missing under {root / AISHELL_DIR}; expected "
            "label_train-set.txt and spk-info.txt"
        )
    lines, malformed_lines = parse_aishell_labels(label_path.read_text(encoding="utf-8"))
    spk_info, malformed_spk = parse_spk_info(spk_path.read_text(encoding="utf-8"))
    filtered = build_aishell_records(lines)
    pool = group_by_speaker(filtered["eligible"])
    selection = select_speakers(pool, spk_info, seed=seed)
    records = sample_speaker_utterances(pool, selection["speakers"], seed=seed)
    records, coverage_supplements = apply_coverage_supplements(records, pool, spk_info, per_speaker=AISHELL_UTTS_PER_SPEAKER)
    records, special_supplements = ensure_special_subset(records, pool, min_special=AISHELL_SPECIAL_MIN)
    coverage = coverage_stats(records)
    special_records = [record for record in records if record.flags.special]
    special_breakdown: Counter[tuple[str, ...]] = Counter()
    for record in special_records:
        flags = record.flags
        special_breakdown[tuple(name for name in ("neutral", "yibu", "erhua") if getattr(flags, name))] += 1

    audio_reports: list[dict[str, Any]] = []
    items: list[SentenceItem] = []
    for record in records:
        speaker = record.speaker
        info = spk_info[speaker]
        relative = f"{speaker}/{record.utt_id}.wav"
        path = root / AISHELL_DIR / "wav" / relative
        probe = probe_audio(path)
        audio_reports.append({"audio_path": _repo_rel(path), "probe": probe})
        flags = record.flags
        items.append(
            SentenceItem(
                item_id=f"A|{record.utt_id}",
                task="A",
                source="aishell3",
                speaker=speaker,
                audio_path=_repo_rel(path),
                duration_s=float(probe.get("duration_s", 0.0)) if probe["ok"] else 0.0,
                text=record.chars,
                chars=record.chars,
                pinyin_surface=list(record.tokens),
                flags=flags.as_dict(),
                meta={
                    "utt_id": record.utt_id,
                    "gender": info.gender,
                    "accent": info.accent,
                    "age": info.age,
                    "in_special": bool(flags.special),
                    "coverage_supplement": False,
                    "stratum": info.stratum,
                    "n_chars": len(record.chars),
                },
            )
        )

    # Flag the coverage / special supplement items by utt id.
    supplement_ids = {
        entry.get("added_utt") or entry.get("forced_utt")
        for entry in coverage_supplements + special_supplements
    }
    supplement_ids.discard(None)
    for item in items:
        if item.meta["utt_id"] in supplement_ids:
            item.meta["coverage_supplement"] = True

    return {
        "items": items,
        "records": records,
        "speakers": selection["speakers"],
        "selection": selection,
        "coverage": coverage,
        "coverage_supplements": coverage_supplements,
        "special_supplements": special_supplements,
        "special_count": len(special_records),
        "special_breakdown": {"/".join(key): value for key, value in sorted(special_breakdown.items())},
        "audio_reports": audio_reports,
        "frame": {
            "label_train_utt_lines": len(lines),
            "label_train_speakers": len({line.utt_id[:7] for line in lines}),
            "malformed_label_lines": malformed_lines,
            "spk_info_rows": len(spk_info),
            "malformed_spk_info_rows": malformed_spk,
            "unalignable_rows": len(filtered["unalignable"]),
            "unalignable": filtered["unalignable"],
            "length_filtered_rows": filtered["length_filtered"],
            "eligible_utterances": len(filtered["eligible"]),
            "eligible_speakers": len(pool),
            "speakers_with_min_utts": len({speaker for speaker, recs in pool.items() if len(recs) >= AISHELL_UTTS_PER_SPEAKER}),
        },
    }


# ---------------------------------------------------------------------------
# Manifest assembly / summary / README
# ---------------------------------------------------------------------------


def _audio_section(
    reports: list[dict[str, Any]],
    expected: dict[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    format_violations: list[dict[str, Any]] = []
    formats: Counter[str] = Counter()
    durations: list[float] = []
    for report in reports:
        probe = report["probe"]
        path = report["audio_path"]
        if not probe["ok"]:
            failures.append({"audio_path": path, "error": probe.get("error")})
            continue
        formats[f"{probe['samplerate']}Hz/{probe['channels']}ch/{probe['subtype']}"] += 1
        durations.append(float(probe["duration_s"]))
        if expected is not None and any(probe[key] != value for key, value in expected.items()):
            format_violations.append(
                {
                    "audio_path": path,
                    "expected": expected,
                    "observed": {
                        "samplerate": probe["samplerate"],
                        "channels": probe["channels"],
                        "subtype": probe["subtype"],
                    },
                }
            )
    return {
        "label": label,
        "checked": len(reports),
        "ok": len(reports) - len(failures),
        "failures": failures,
        "format_violations": format_violations,
        "formats": dict(sorted(formats.items())),
        "total_duration_s": round(sum(durations), 3),
    }


def _duration_block(items: Sequence[SentenceItem]) -> dict[str, Any]:
    stats = duration_stats([item.duration_s for item in items])
    stats["n_chars"] = sum(len(item.chars) for item in items)
    stats["n_pinyin_syllables"] = sum(len(item.pinyin_surface) for item in items)
    stats["n_label_units"] = sum(len(item.labels) for item in items)
    return stats


def build(root: Path = REPO_ROOT, *, seed: int = SEED, download_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full offline build (metadata parsing, sampling, manifests, summary)."""
    aishell = build_aishell_records_section(root, seed=seed)
    om_wav_dir = root / OMPAL_REPO_DIR
    if not (om_wav_dir / "non-native_scores.json").exists():
        raise FileNotFoundError(
            f"OMPAL metadata missing under {om_wav_dir}; run the `metadata` or `all` stage first"
        )
    learner_raw = json.loads((om_wav_dir / "non-native_scores.json").read_text(encoding="utf-8"))
    native_raw = json.loads((om_wav_dir / "native_scores.json").read_text(encoding="utf-8"))
    detail_path = om_wav_dir / "non-native_scores-detail.json"
    detail_raw = json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.exists() else None
    om = build_ompal_items(learner_raw, native_raw, root=root)

    validation = {
        "learner_records": len(learner_raw),
        "native_records": len(native_raw),
        "learner_id_prefix_002": all(utt_id.startswith("002") for utt_id in learner_raw),
        "native_id_prefix_001": all(utt_id.startswith("001") for utt_id in native_raw),
        "native_all_scores_5": all(
            tuple(float(record[dim]) for dim in OMPAL_SCORE_DIMS) == (5.0, 5.0, 5.0) for record in native_raw.values()
        ),
    }
    if detail_raw is not None:
        validation["aggregated_vs_detail"] = check_aggregated_scores(learner_raw, detail_raw)

    a_items = aishell["items"]
    b_items = om["b_items"]
    c_items = om["c_items"]
    native_items = om["native_items"]
    counts = {"A": len(a_items), "B": len(b_items), "C": len(c_items), "C_native": len(native_items)}

    aishell_audio = _audio_section(aishell["audio_reports"], AISHELL_EXPECTED_FORMAT, label="aishell3")
    om_audio = _audio_section(om["audio_reports"], None, label="ompal")
    expected_ompal_format = (
        f"{OMPAL_EXPECTED_FORMAT['samplerate']}Hz/"
        f"{OMPAL_EXPECTED_FORMAT['channels']}ch/{OMPAL_EXPECTED_FORMAT['subtype']}"
    )
    om_format_mismatch = [
        {"observed_format": key, "n": value}
        for key, value in om_audio["formats"].items()
        if key != expected_ompal_format
    ]

    source_files = {
        _repo_rel(root / AISHELL_DIR / "label_train-set.txt"): sha256_file(root / AISHELL_DIR / "label_train-set.txt"),
        _repo_rel(root / AISHELL_DIR / "spk-info.txt"): sha256_file(root / AISHELL_DIR / "spk-info.txt"),
    }
    for name in OMPAL_REQUIRED_FILES:
        path = root / OMPAL_REPO_DIR / name
        if path.exists():
            source_files[_repo_rel(path)] = sha256_file(path)

    deviations = _deviations(aishell, om, counts)
    summary: dict[str, Any] = {
        "prep_version": PREP_VERSION,
        "generated_at": utc_now(),
        "seed": seed,
        "repo_root": str(REPO_ROOT),
        "counts": counts,
        "sources": {
            "aishell3": {
                "corpus": "AISHELL-3 (OpenSLR SLR93, Apache-2.0)",
                "split": "train",
                "wav_mirrors": [{"name": name, "base_url": base} for name, base in AISHELL_MIRRORS],
                "files_sha256": source_files,
            },
            "ompal": {
                "corpus": "OMPAL corpus (CC BY 4.0)",
                "repo": OMPAL_REPO,
                "commit": OMPAL_COMMIT,
                "raw_base": OMPAL_RAW_BASE,
                "files_sha256": source_files,
                "wav_format": OMPAL_EXPECTED_FORMAT,
            },
        },
        "aishell3": {
            "frame": aishell["frame"],
            "stratification": {
                "requested_quotas": aishell["selection"]["requested"],
                "achieved": aishell["selection"]["achieved"],
                "achieved_slack": aishell["selection"]["slack"],
                "category_counts": aishell["selection"]["counts"],
                "candidates_by_category": aishell["selection"]["by_category"],
                "candidate_exclusions": aishell["selection"]["exclusions"],
                "selected_speakers": aishell["speakers"],
            },
            "coverage": {
                **aishell["coverage"].as_dict(),
                "supplements": aishell["coverage_supplements"],
                "supplement_count": len(aishell["coverage_supplements"]),
            },
            "special_subset": {
                "min_required": AISHELL_SPECIAL_MIN,
                "count": aishell["special_count"],
                "breakdown": aishell["special_breakdown"],
                "supplements": aishell["special_supplements"],
            },
            "items": _duration_block(a_items),
            "audio": aishell_audio,
        },
        "ompal": {
            "cleaning": {
                "raw_learner": len(learner_raw),
                "raw_native": len(native_raw),
                "excluded_learner": len(om["cleaned"]["exclusions"]),
                "cleaned_learner": len(om["cleaned"]["kept"]),
                "exclusions": om["cleaned"]["exclusions"],
                "anomaly_scan": {
                    "multi_char_word_slot_rows": om["cleaned"]["anomalies"]["multi_char_word_slot_rows"],
                    "multi_char_word_slots": om["cleaned"]["anomalies"]["multi_char_word_slots"],
                    "text_join_mismatch_rows": om["cleaned"]["anomalies"]["text_join_mismatch_rows"],
                    "empty_word_slot_rows": om["cleaned"]["anomalies"]["empty_word_slot_rows"],
                    "invalid_label_rows": om["cleaned"]["anomalies"]["invalid_label_rows"],
                },
                "native_anomalies_kept": {
                    "multi_char_word_slot_rows": om["native_anomalies"]["multi_char_word_slot_rows"],
                    "text_join_mismatch_rows": om["native_anomalies"]["text_join_mismatch_rows"],
                    "note": "native rows are anchor-only (all scores 5.0), so their corrupted slots are kept as-is",
                },
            },
            "validation": validation,
            "sentence_inventory": {
                "n_sentences": len(om["inventory"]),
                "ids": {sentence_id: text for text, sentence_id in om["inventory"].items()},
            },
            "labels": {
                "units_b": om["total_units"],
                "zero_initial_units": om["zero_initial_units"],
                "zero_initial_source": "pypinyin.lazy_pinyin(text, style=TONE3) with project split_surface rules",
                "zero_initial_failures": om["zero_initial_failures"],
            },
            "items": {
                "B": _duration_block(b_items),
                "C": _duration_block(c_items),
                "C_native": _duration_block(native_items),
            },
            "audio": {**om_audio, "format_mismatch_rows": om_format_mismatch},
        },
        "download": (
            {
                **download_stats,
                "note": (
                    "by_task counts refer to the last run only (existing files are skipped); "
                    "provenance_counts covers every present file and download_log.jsonl keeps "
                    "the per-file history"
                ),
            }
            if download_stats is not None
            else None
        ),
        "deviations": deviations,
    }
    readme = _readme(summary)
    return {
        "items": {"aishell3": a_items, "ompal_b": b_items, "ompal_c": c_items, "ompal_native": native_items},
        "summary": summary,
        "readme": readme,
    }


def _deviations(aishell: dict[str, Any], om: dict[str, Any], counts: dict[str, int]) -> list[dict[str, str]]:
    """Documented deviations from the design (design §3)."""
    deviations = [
        {
            "area": "aishell3/source",
            "deviation": (
                "AISHELL-3 task-A source is the train split only. The official HF repo "
                "AISHELL/AISHELL-3 hosts only 9,802 of the 63,262 train wavs, so wavs are "
                "fetched from the complete mirror shenyunhang/AISHELL-3 with "
                "AISHELL/AISHELL-3 as fallback; the label/spk files are the released train "
                "metadata (identical across both mirrors)."
            ),
        },
        {
            "area": "aishell3/unalignable",
            "deviation": (
                f"align_gold rejects {aishell['frame']['unalignable_rows']} label lines, not only the "
                "two known ids; every rejected utterance id + reason is recorded under "
                "aishell3.frame.unalignable and excluded from the sampling frame."
            ),
        },
        {
            "area": "aishell3/strata",
            "deviation": (
                "Speaker quotas are met "
                + ("exactly" if aishell["selection"]["slack"] == 0 else f"with slack {aishell['selection']['slack']}")
                + " (design allows approximate); achieved: "
                + json.dumps(aishell["selection"]["achieved"], ensure_ascii=False, sort_keys=True)
            ),
        },
        {
            "area": "ompal/cleaning-count",
            "deviation": (
                "OMPAL cleaning differs from the design prediction of 12+2 removals (1,754): the "
                "learner file has 9 corrupted multi-character word-slot rows plus 1 text/slot "
                f"mismatch (00204117) = {len(om['cleaned']['exclusions'])} removals -> "
                f"{len(om['cleaned']['kept'])} learner items. The design's 12 corrupted rows are "
                "corpus-wide (9 learner + 3 native) and its second text/slot mismatch 00100217 is "
                "native, so only 10 removals apply to the learner set. No valid rows are dropped to "
                "force 1,754."
            ),
        },
        {
            "area": "ompal/detail-json-ids",
            "deviation": (
                "non-native_scores-detail.json re-numbers 656 learner utterances to speaker ids "
                "047-066 and compacts their utterance numbers. non-native_scores.json matches the "
                "wav tree exactly and is used for the manifests; the 3-rater means / majority votes "
                "are validated against the detail file wherever the ids match."
            ),
        },
        {
            "area": "ompal/native",
            "deviation": (
                "The 82 native rows keep their released anomalies (4 rows: 00100217, 00100218, "
                "00100222, 00100322) because natives are only a human-ceiling anchor (all scores 5.0) "
                "and are never scored."
            ),
        },
        {
            "area": "ompal/sentence-id",
            "deviation": (
                "The release does not expose the corpus' internal sentence numbering; meta.sentence_id "
                "is an ordinal over the 84 distinct reference texts in the score files (full mapping "
                "in ompal.sentence_inventory)."
            ),
        },
        {
            "area": "counts",
            "deviation": (
                f"Final item counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, "
                f"native-C={counts['C_native']} (design: A=300, B=C=1,754, native=82)."
            ),
        },
    ]
    return deviations


def _readme(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    achieved = json.dumps(summary["aishell3"]["stratification"]["achieved"], ensure_ascii=False, sort_keys=True)
    coverage = summary["aishell3"]["coverage"]
    deviations = "\n".join(
        f"{index}. **{entry['area']}** — {entry['deviation']}"
        for index, entry in enumerate(summary["deviations"], 1)
    )
    return f"""# Sentence-level evaluation data (pilot)

Generated by `scripts/prepare_sentence_data.py` (version {summary['prep_version']},
seed `{summary['seed']}`) on {summary['generated_at']}.

## Manifests

| file | task | corpus | items |
|---|---|---|---|
| `aishell3_items.jsonl` | A (pinyin transcription) | AISHELL-3 train | {counts['A']} |
| `ompal_b_items.jsonl` | B (pronunciation assessment) | OMPAL learner | {counts['B']} |
| `ompal_c_items.jsonl` | C (reading scoring) | OMPAL learner | {counts['C']} |
| `ompal_native_items.jsonl` | C anchor | OMPAL native | {counts['C_native']} |

Each line matches `pinyin_eval.sentence.records.SentenceItem` (`to_dict()`);
`audio_path` is repository-relative.  `prep_summary.json` holds counts, achieved
strata ({achieved}), the coverage table (initials {coverage['n_initials_present']}/21,
finals {coverage['n_finals_present']}/36, tones {coverage['n_tones_present']}/4,
supplemented: {coverage['supplement_count']}), the special subset
({summary['aishell3']['special_subset']['count']} >= {summary['aishell3']['special_subset']['min_required']}),
auto-checked.

## Rebuild

```bash
# all-in-one: fetch missing metadata + build + download audio + rebuild durations
.venv/bin/python scripts/prepare_sentence_data.py all

# offline (metadata + previously downloaded audio already present)
.venv/bin/python scripts/prepare_sentence_data.py build

# only fetch missing audio (resumable: existing files are skipped)
.venv/bin/python scripts/prepare_sentence_data.py download

# only fetch missing source JSON from GitHub raw
.venv/bin/python scripts/prepare_sentence_data.py metadata
```

Network needs the system proxy, or `HTTPS_PROXY` / `HTTP_PROXY` (the script uses
`urllib`, which honours both).  Downloads are concurrent (default 12) and
resumable; the per-file provenance (final source URL, attempts, errors) is
appended to `../download_log.jsonl` and summarised in `../audio_sources.json`
plus `prep_summary.json["download"]`.

Determinism: sampling uses `random.Random({summary['seed']})` over sorted
inputs, so a rebuild from the same source files reproduces the same item ids
and the same manifests; only the `generated_at` / download timestamps differ.

## Deviations from `docs/sentence-experiment-design.md`

{deviations}

## Licences / attribution

* AISHELL-3: Apache-2.0, OpenSLR SLR93 — <https://www.openslr.org/93/>
* OMPAL: CC BY 4.0, <https://github.com/phantomhsieh/OMPAL-corpus>
  (commit `{OMPAL_COMMIT}`); cite Hsieh et al., Interspeech 2025.
* Wavs are kept in their original format (AISHELL-3 44.1 kHz/16-bit/mono,
  OMPAL 16 kHz/16-bit/mono); nothing is resampled or re-encoded.
"""


def write_outputs(root: Path, result: dict[str, Any]) -> dict[str, str]:
    """Write the four manifests, the summary JSON and the data README."""
    manifest_dir = root / MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, key in (
        ("aishell3_items.jsonl", "aishell3"),
        ("ompal_b_items.jsonl", "ompal_b"),
        ("ompal_c_items.jsonl", "ompal_c"),
        ("ompal_native_items.jsonl", "ompal_native"),
    ):
        destination = manifest_dir / name
        write_jsonl(destination, result["items"][key])
        paths[name] = _repo_rel(destination)
    summary_path = manifest_dir / "prep_summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    paths["prep_summary.json"] = _repo_rel(summary_path)
    readme_path = manifest_dir / "README.md"
    readme_path.write_text(result["readme"], encoding="utf-8")
    paths["README.md"] = _repo_rel(readme_path)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_counts(result: dict[str, Any]) -> None:
    counts = result["summary"]["counts"]
    print("counts:", json.dumps(counts, sort_keys=True))
    print("achieved strata:", json.dumps(result["summary"]["aishell3"]["stratification"]["achieved"], sort_keys=True))
    coverage = result["summary"]["aishell3"]["coverage"]
    print(
        "coverage:",
        f"initials {coverage['n_initials_present']}/21",
        f"finals {coverage['n_finals_present']}/36",
        f"tones {coverage['n_tones_present']}/4",
        f"supplements={coverage['supplement_count']}",
    )
    print("special subset:", result["summary"]["aishell3"]["special_subset"]["count"])
    print("ompal exclusions:", len(result["summary"]["ompal"]["cleaning"]["exclusions"]))


def _preserved_download_stats(root: Path) -> dict[str, Any] | None:
    """Download statistics recorded by a previous run (kept across pure rebuilds)."""
    summary_path = root / MANIFEST_DIR / "prep_summary.json"
    if not summary_path.exists():
        return None
    try:
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    download = previous.get("download")
    return download if isinstance(download, dict) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=("all", "build", "download", "metadata"),
        help="pipeline stage to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="sampling seed")
    parser.add_argument("--concurrency", type=int, default=12, help="parallel downloads")
    args = parser.parse_args(argv)

    root = REPO_ROOT
    if args.stage == "metadata":
        missing = ensure_metadata(root)
        print(f"metadata: fetched {len(missing)} missing file(s)")
        return 0

    if args.stage == "all":
        ensure_metadata(root)

    if args.stage in ("all", "build"):
        result = build(root, seed=args.seed, download_stats=_preserved_download_stats(root))
        _print_counts(result)
        paths = write_outputs(root, result)
        print("wrote:", json.dumps(paths, ensure_ascii=False, sort_keys=True))
        if args.stage == "build":
            return 0

    # download stage / tail of `all`
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    print("downloading missing audio ...", flush=True)
    download_stats = download_audio(root, concurrency=args.concurrency)
    print(
        "download:",
        json.dumps(
            {
                "total": download_stats["total"],
                "by_task": download_stats["by_task"],
                "failures": len(download_stats["failures"]),
            },
            ensure_ascii=False,
        ),
    )
    result = build(root, seed=args.seed, download_stats=download_stats)
    _print_counts(result)
    paths = write_outputs(root, result)
    print("wrote:", json.dumps(paths, ensure_ascii=False, sort_keys=True))
    if download_stats["failures"]:
        print("WARNING: some audio downloads failed; see data/sentence/download_log.jsonl and prep_summary.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
