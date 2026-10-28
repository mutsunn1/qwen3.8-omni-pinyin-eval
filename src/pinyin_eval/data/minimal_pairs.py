"""Minimal-pair construction for the ``mandarin_tone_syn`` experiments.

This module implements section 5 of ``docs/contracts.md`` (pairing rules are
derived from the design document's "最小对立组" table):

===============  =====================================  ==================
pair_type        rule                                   design count
===============  =====================================  ==================
``sh_zh``        (z, zh), (c, ch), (s, sh) same final;  36 pairs / 576 inst.
                 finals ``i`` (-i group) excluded
``aspiration``   (b, p), (d, t), (g, k), (j, q) same    67 / 1072
                 final
``nasal_lateral``(n, l) same final                     21 / 336
``front_back_nasal_ptr`` (an, ang), (en, eng),        40 / 640
                 (in, ing) same initial; the wide     (52 / 832 with
                 variant adds (ian, iang),              ``include_wide_nasal``)
                 (uan, uang)
``i_ü``          same initial, final i vs ü            5 / 80
``ian_üan``      same initial, final ian vs üan        3 / 48
``tone``         same initial+final, tone pairs        2268 / 9072
                 (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
===============  =====================================  ==================

A :class:`MinimalPair` is a *target-level* pair (no voice/tone instantiation).
A :class:`MinimalPairInstance` is a concrete pairing of two :class:`AudioRecord`
objects: one per side, at a fixed voice (and a fixed tone for segmental pairs).
Instances only exist when both sides' audio is available.  Merged spellings
(the seven targets with two dataset spellings) prefer ``raw_target == target``.

Every builder returns a deterministically ordered list: the same input set
produces the same output order regardless of input ordering.  ``count_table``
reproduces the design document's pair/instance counts.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pinyin_eval.records import PAIR_TYPES, AudioRecord, make_pair_id

__all__ = [
    "FRONT_BACK_NASAL_FINALS",
    "IAN_UAN_FINAL_PAIR",
    "I_U_FINAL_PAIR",
    "PAIR_TYPE_DIMENSION",
    "TONE_PAIRS",
    "WIDE_NASAL_FINALS",
    "MappingEntry",
    "MinimalPair",
    "MinimalPairInstance",
    "PairItem",
    "build_all_pairs",
    "build_instances",
    "build_pairs",
    "count_table",
    "instance_sort_key",
    "instances_by_type",
    "iter_pair_items",
    "load_mapping",
    "pair_sort_key",
    "pairs_by_type",
]

# ---------------------------------------------------------------------------
# Pairing rules (frozen by docs/contracts.md section 5)
# ---------------------------------------------------------------------------

#: initial contrasts that share one final; side A is the first member.
INITIAL_CONTRASTS: dict[str, tuple[tuple[str, str], ...]] = {
    "sh_zh": (("z", "zh"), ("c", "ch"), ("s", "sh")),
    "aspiration": (("b", "p"), ("d", "t"), ("g", "k"), ("j", "q")),
    "nasal_lateral": (("n", "l"),),
}

#: the only initial-pair type that excludes the -i group (final == "i").
_EXCLUDE_I_FINAL_PAIR_TYPE = "sh_zh"

#: finals contrasted for "front/back nasal"; the default scope is the three
#: pairs that reproduce the design count of 40.
FRONT_BACK_NASAL_FINALS: tuple[tuple[str, str], ...] = (
    ("an", "ang"),
    ("en", "eng"),
    ("in", "ing"),
)

#: extra front/back nasal pairs enabled by ``include_wide_nasal=True``
#: (+12 pairs / +192 instances, design count 40 -> 52).
WIDE_NASAL_FINALS: tuple[tuple[str, str], ...] = (
    ("ian", "iang"),
    ("uan", "uang"),
)

I_U_FINAL_PAIR: tuple[str, str] = ("i", "ü")
IAN_UAN_FINAL_PAIR: tuple[str, str] = ("ian", "üan")

#: final contrasts paired inside one initial, mapped to their pair type.
FINAL_CONTRASTS: dict[str, tuple[tuple[str, str], ...]] = {
    "i_ü": (I_U_FINAL_PAIR,),
    "ian_üan": (IAN_UAN_FINAL_PAIR,),
}

#: six non-identical tone pairs, in the order given by the design document.
TONE_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)

#: pair_type -> recognition dimension tested by the matching experiment.
PAIR_TYPE_DIMENSION: dict[str, str] = {
    "sh_zh": "initial",
    "aspiration": "initial",
    "nasal_lateral": "initial",
    "front_back_nasal_ptr": "final",
    "i_ü": "final",
    "ian_üan": "final",
    "tone": "tone",
}

_PAIR_TYPE_ORDER: dict[str, int] = {pair_type: i for i, pair_type in enumerate(PAIR_TYPES)}
_TONES: tuple[int, ...] = (1, 2, 3, 4)

_MAPPING_REQUIRED_COLUMNS = ("target", "initial", "final")


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappingEntry:
    """One row of ``mandarin_tone_syn_target_mapping.csv``.

    ``target`` is the canonical (zero-initial normalised) syllable; ``initial``
    is ``""`` for zero initials; ``final`` uses Unicode ``ü``.
    """

    target: str
    initial: str
    final: str
    count: int = 0
    source: str = ""


def load_mapping(path: str | Path) -> dict[str, MappingEntry]:
    """Load the canonical target -> (initial, final) mapping CSV.

    The file is UTF-8 **with BOM**; it is decoded with ``utf-8-sig``.  Returns a
    dict in file order keyed by canonical target.  Duplicate or blank targets
    raise :class:`ValueError` instead of silently overwriting data.
    """
    mapping_path = Path(path)
    entries: dict[str, MappingEntry] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [name for name in _MAPPING_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"{mapping_path}: missing mapping columns {missing}")
        for row in reader:
            target = (row.get("target") or "").strip()
            if not target:
                raise ValueError(f"{mapping_path}: row with empty 'target': {row!r}")
            if target in entries:
                raise ValueError(f"{mapping_path}: duplicate target {target!r}")
            count_text = (row.get("count") or "").strip()
            entries[target] = MappingEntry(
                target=target,
                initial=(row.get("initial") or "").strip(),
                final=(row.get("final") or "").strip(),
                count=int(count_text) if count_text else 0,
                source=(row.get("source") or "").strip(),
            )
    if not entries:
        raise ValueError(f"{mapping_path}: empty mapping table")
    return entries


def _resolve_initial_final(entry: MappingEntry | tuple[str, str] | Sequence[str]) -> tuple[str, str]:
    """Accept ``MappingEntry`` rows as well as plain ``(initial, final)`` tuples."""
    if isinstance(entry, MappingEntry):
        return entry.initial, entry.final
    if isinstance(entry, tuple) and len(entry) == 2:
        return str(entry[0]), str(entry[1])
    if isinstance(entry, list) and len(entry) == 2:
        return str(entry[0]), str(entry[1])
    if hasattr(entry, "initial") and hasattr(entry, "final"):
        return str(entry.initial), str(entry.final)
    raise TypeError(f"mapping value must be MappingEntry or (initial, final), got {entry!r}")


# ---------------------------------------------------------------------------
# Pair / instance records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinimalPair:
    """One target-level minimal pair (no voice/tone instantiation yet).

    For segmental pair types (``pair_type != "tone"``) both targets share one
    tone and differ in exactly one of initial/final; ``tone_a``/``tone_b`` are
    ``None``.  For ``pair_type == "tone"`` the two targets are identical and
    ``tone_a``/``tone_b`` hold the contrasted tones.
    """

    pair_type: str
    target_a: str
    target_b: str
    initial_a: str
    initial_b: str
    final_a: str
    final_b: str
    tone_a: int | None = None
    tone_b: int | None = None

    def __post_init__(self) -> None:
        if self.pair_type not in PAIR_TYPE_DIMENSION:
            raise ValueError(f"unknown pair_type {self.pair_type!r}")
        if self.pair_type == "tone":
            if self.tone_a is None or self.tone_b is None:
                raise ValueError("tone pairs require tone_a and tone_b")
            if self.tone_a == self.tone_b:
                raise ValueError("tone pairs require two different tones")
        elif self.tone_a is not None or self.tone_b is not None:
            raise ValueError(f"{self.pair_type} pairs must not carry tones")

    @property
    def dimension(self) -> str:
        """Recognition dimension tested: ``initial`` | ``final`` | ``tone``."""
        return PAIR_TYPE_DIMENSION[self.pair_type]

    @property
    def targets(self) -> tuple[str, str]:
        return (self.target_a, self.target_b)

    @property
    def pair_id(self) -> str:
        """Stable pair-level id (no voice/tone instantiation).

        Segmental pairs: ``{pair_type}:{target_a}/{target_b}``.
        Tone pairs: ``tone:{target}:{tone_a}-{tone_b}`` (the target is identical
        on both sides, so the contrasted tones are what distinguishes pairs).
        """
        if self.pair_type == "tone":
            return f"tone:{self.target_a}:{self.tone_a}-{self.tone_b}"
        return f"{self.pair_type}:{self.target_a}/{self.target_b}"

    @property
    def identity(self) -> tuple[str, str, str, int | None, int | None]:
        """Identity used when de-duplicating pairs (voice/tone independent)."""
        return (self.pair_type, self.target_a, self.target_b, self.tone_a, self.tone_b)


@dataclass(frozen=True, slots=True)
class MinimalPairInstance:
    """Concrete pairing of two audios: one instance per (pair, voice[, tone]).

    * segmental pair types: ``tone`` is the common tone of both sides;
    * ``tone`` pairs: ``tone`` is ``None`` and the contrasted tones live on
      ``pair.tone_a`` / ``pair.tone_b``.

    ``audio_a`` always belongs to ``pair.target_a`` and ``audio_b`` to
    ``pair.target_b``; the two sides carry the same voice.
    """

    pair: MinimalPair
    voice: str
    tone: int | None
    audio_a: AudioRecord
    audio_b: AudioRecord

    def __post_init__(self) -> None:
        if self.pair.pair_type == "tone":
            if self.tone is not None:
                raise ValueError("tone-pair instances must not carry a common tone")
        elif self.tone not in _TONES:
            raise ValueError(f"{self.pair.pair_type} instances require a tone in 1..4")
        if self.audio_a.target != self.pair.target_a:
            raise ValueError(
                f"audio_a target {self.audio_a.target!r} != pair.target_a {self.pair.target_a!r}"
            )
        if self.audio_b.target != self.pair.target_b:
            raise ValueError(
                f"audio_b target {self.audio_b.target!r} != pair.target_b {self.pair.target_b!r}"
            )
        if self.audio_a.voice != self.voice or self.audio_b.voice != self.voice:
            raise ValueError(f"instance voice {self.voice!r} does not match its audios")
        if self.audio_a.tone != self.tone_a:
            raise ValueError(f"audio_a tone {self.audio_a.tone} != expected {self.tone_a}")
        if self.audio_b.tone != self.tone_b:
            raise ValueError(f"audio_b tone {self.audio_b.tone} != expected {self.tone_b}")

    @property
    def tone_a(self) -> int:
        return self.pair.tone_a if self.pair.tone_a is not None else int(self.tone)

    @property
    def tone_b(self) -> int:
        return self.pair.tone_b if self.pair.tone_b is not None else int(self.tone)

    @property
    def pair_id(self) -> str:
        """Deterministic pair id shared by both sides (``records.make_pair_id``)."""
        pair = self.pair
        if pair.pair_type == "tone":
            return make_pair_id(
                "tone", pair.target_a, pair.target_b, self.voice, f"{pair.tone_a}-{pair.tone_b}"
            )
        return make_pair_id(pair.pair_type, pair.target_a, pair.target_b, self.voice, str(self.tone))

    def iter_sides(self) -> Iterator[tuple[str, AudioRecord, AudioRecord]]:
        """Yield ``(side, audio, other_audio)`` for sides ``"A"`` and ``"B"``."""
        yield "A", self.audio_a, self.audio_b
        yield "B", self.audio_b, self.audio_a

    def side_audio(self, side: str) -> AudioRecord:
        if side == "A":
            return self.audio_a
        if side == "B":
            return self.audio_b
        raise ValueError(f"side must be 'A' or 'B', got {side!r}")


@dataclass(frozen=True, slots=True)
class PairItem:
    """One side of one instance, ready to become a ``TaskItem``.

    ``audio`` is the side's own record, ``other_audio`` the minimal-pair
    counterpart.  ``pair_id`` is identical for both sides of an instance.
    """

    pair_id: str
    pair: MinimalPair
    voice: str
    side: str
    audio: AudioRecord
    other_audio: AudioRecord

    def __post_init__(self) -> None:
        if self.side not in ("A", "B"):
            raise ValueError(f"side must be 'A' or 'B', got {self.side!r}")

    @property
    def audio_id(self) -> str:
        return self.audio.audio_id

    @property
    def pair_type(self) -> str:
        return self.pair.pair_type

    @property
    def dimension(self) -> str:
        return self.pair.dimension


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------


def pair_sort_key(pair: MinimalPair) -> tuple[int, str, str, int, int]:
    """Total order over pairs: pair type, targets, then contrasted tones."""
    return (
        _PAIR_TYPE_ORDER[pair.pair_type],
        pair.target_a,
        pair.target_b,
        pair.tone_a or 0,
        pair.tone_b or 0,
    )


def instance_sort_key(instance: MinimalPairInstance) -> tuple[int, str, str, int, int, int, str]:
    """Total order over instances: pair order, then tone (if any) and voice."""
    return (*pair_sort_key(instance.pair), instance.tone or 0, instance.voice)


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------


def _resolve_targets(
    targets: Iterable[str],
    mapping: Mapping[str, MappingEntry | tuple[str, str] | Sequence[str]],
) -> dict[str, tuple[str, str]]:
    unique_targets = sorted({str(target) for target in targets})
    missing = [target for target in unique_targets if target not in mapping]
    if missing:
        preview = ", ".join(repr(target) for target in missing[:5])
        raise ValueError(f"{len(missing)} target(s) missing from mapping: {preview}")
    return {target: _resolve_initial_final(mapping[target]) for target in unique_targets}


def _index_by_final(
    resolved: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, list[str]]]:
    """final -> initial -> sorted canonical targets."""
    by_final: dict[str, dict[str, list[str]]] = {}
    for target in sorted(resolved):
        initial, final = resolved[target]
        by_final.setdefault(final, {}).setdefault(initial, []).append(target)
    return by_final


def _index_by_initial(
    resolved: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, list[str]]]:
    """initial -> final -> sorted canonical targets."""
    by_initial: dict[str, dict[str, list[str]]] = {}
    for target in sorted(resolved):
        initial, final = resolved[target]
        by_initial.setdefault(initial, {}).setdefault(final, []).append(target)
    return by_initial


def _pairs_with_initial_contrast(
    pair_type: str,
    by_final: Mapping[str, Mapping[str, Sequence[str]]],
    contrasts: Sequence[tuple[str, str]],
    *,
    exclude_finals: frozenset[str] = frozenset(),
) -> list[MinimalPair]:
    pairs: list[MinimalPair] = []
    for final in sorted(by_final):
        if final in exclude_finals:
            continue
        targets_by_initial = by_final[final]
        for initial_a, initial_b in contrasts:
            for target_a in sorted(targets_by_initial.get(initial_a, ())):
                for target_b in sorted(targets_by_initial.get(initial_b, ())):
                    pairs.append(
                        MinimalPair(
                            pair_type=pair_type,
                            target_a=target_a,
                            target_b=target_b,
                            initial_a=initial_a,
                            initial_b=initial_b,
                            final_a=final,
                            final_b=final,
                        )
                    )
    return pairs


def _pairs_with_final_contrast(
    pair_type: str,
    by_initial: Mapping[str, Mapping[str, Sequence[str]]],
    final_contrasts: Sequence[tuple[str, str]],
) -> list[MinimalPair]:
    pairs: list[MinimalPair] = []
    for initial in sorted(by_initial):
        targets_by_final = by_initial[initial]
        for final_a, final_b in final_contrasts:
            for target_a in sorted(targets_by_final.get(final_a, ())):
                for target_b in sorted(targets_by_final.get(final_b, ())):
                    pairs.append(
                        MinimalPair(
                            pair_type=pair_type,
                            target_a=target_a,
                            target_b=target_b,
                            initial_a=initial,
                            initial_b=initial,
                            final_a=final_a,
                            final_b=final_b,
                        )
                    )
    return pairs


def _pairs_with_tone_contrast(resolved: Mapping[str, tuple[str, str]]) -> list[MinimalPair]:
    pairs: list[MinimalPair] = []
    for target in sorted(resolved):
        initial, final = resolved[target]
        for tone_a, tone_b in TONE_PAIRS:
            pairs.append(
                MinimalPair(
                    pair_type="tone",
                    target_a=target,
                    target_b=target,
                    initial_a=initial,
                    initial_b=initial,
                    final_a=final,
                    final_b=final,
                    tone_a=tone_a,
                    tone_b=tone_b,
                )
            )
    return pairs


def build_pairs(
    targets: Iterable[str],
    mapping: Mapping[str, MappingEntry | tuple[str, str] | Sequence[str]],
    *,
    include_wide_nasal: bool = False,
) -> list[MinimalPair]:
    """Build every target-level minimal pair from canonical ``targets``.

    Parameters
    ----------
    targets:
        Canonical no-tone targets (e.g. ``sorted({record.target for record in
        audio_records})``); duplicates are ignored, order is irrelevant.
    mapping:
        ``target -> MappingEntry`` (from :func:`load_mapping`) or
        ``target -> (initial, final)``.  Every requested target must exist.
    include_wide_nasal:
        ``False`` (default) reproduces the design count: front/back nasal pairs
        are only (an, ang), (en, eng), (in, ing).  ``True`` adds (ian, iang) and
        (uan, uang): +12 pairs / +192 instances.

    Returns
    -------
    list[MinimalPair]
        Sorted by :func:`pair_sort_key`; deterministic for any input order.
    """
    resolved = _resolve_targets(targets, mapping)
    by_final = _index_by_final(resolved)
    by_initial = _index_by_initial(resolved)

    pairs: list[MinimalPair] = []
    for pair_type, contrasts in INITIAL_CONTRASTS.items():
        exclude_finals = frozenset({"i"}) if pair_type == _EXCLUDE_I_FINAL_PAIR_TYPE else frozenset()
        pairs.extend(
            _pairs_with_initial_contrast(
                pair_type, by_final, contrasts, exclude_finals=exclude_finals
            )
        )

    nasal_finals = FRONT_BACK_NASAL_FINALS + (WIDE_NASAL_FINALS if include_wide_nasal else ())
    pairs.extend(_pairs_with_final_contrast("front_back_nasal_ptr", by_initial, nasal_finals))
    for pair_type, final_contrasts in FINAL_CONTRASTS.items():
        pairs.extend(_pairs_with_final_contrast(pair_type, by_initial, final_contrasts))
    pairs.extend(_pairs_with_tone_contrast(resolved))

    pairs.sort(key=pair_sort_key)
    return pairs


def build_all_pairs(
    audio_records: Iterable[AudioRecord],
    mapping: Mapping[str, MappingEntry | tuple[str, str] | Sequence[str]],
    *,
    include_wide_nasal: bool = False,
) -> list[MinimalPair]:
    """Convenience wrapper: ``build_pairs`` over the canonical targets present in audio."""
    targets = sorted({record.target for record in audio_records})
    return build_pairs(targets, mapping, include_wide_nasal=include_wide_nasal)


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------


def _selection_rank(record: AudioRecord) -> tuple[int, str]:
    """Prefer ``raw_target == target`` (merged spellings), then the lowest audio id."""
    return (0 if record.raw_target == record.target else 1, record.audio_id)


def _build_audio_index(
    audio_records: Iterable[AudioRecord],
) -> dict[tuple[str, str, int], AudioRecord]:
    """(target, voice, tone) -> one deterministically chosen audio record."""
    ordered = sorted(
        audio_records,
        key=lambda record: (
            record.audio_id,
            record.target,
            record.voice,
            record.tone,
            record.raw_target,
        ),
    )
    index: dict[tuple[str, str, int], AudioRecord] = {}
    for record in ordered:
        key = (record.target, record.voice, record.tone)
        current = index.get(key)
        if current is None or _selection_rank(record) < _selection_rank(current):
            index[key] = record
    return index


def _dedupe_pairs(pairs: Iterable[MinimalPair]) -> list[MinimalPair]:
    seen: set[tuple[str, str, str, int | None, int | None]] = set()
    unique: list[MinimalPair] = []
    for pair in sorted(pairs, key=pair_sort_key):
        if pair.identity in seen:
            continue
        seen.add(pair.identity)
        unique.append(pair)
    return unique


def build_instances(
    pairs: Iterable[MinimalPair],
    audio_records: Iterable[AudioRecord],
) -> list[MinimalPairInstance]:
    """Instantiate pairs as concrete two-audio pairings.

    One instance is produced per ``(pair, voice, tone)`` for segmental pairs and
    per ``(pair, voice)`` for tone pairs, **only when both sides have audio**.
    Duplicate pairs/records are tolerated: each ``(target, voice, tone)`` maps to
    a single audio (``raw_target == target`` preferred, then lowest audio id) and
    pair duplicates are removed.

    Returns instances sorted by :func:`instance_sort_key`.
    """
    records = list(audio_records)
    index = _build_audio_index(records)
    voices = sorted({record.voice for record in records})

    instances: list[MinimalPairInstance] = []
    for pair in _dedupe_pairs(pairs):
        if pair.pair_type == "tone":
            for voice in voices:
                audio_a = index.get((pair.target_a, voice, int(pair.tone_a)))
                audio_b = index.get((pair.target_b, voice, int(pair.tone_b)))
                if audio_a is None or audio_b is None:
                    continue
                instances.append(MinimalPairInstance(pair, voice, None, audio_a, audio_b))
        else:
            for voice in voices:
                for tone in _TONES:
                    audio_a = index.get((pair.target_a, voice, tone))
                    audio_b = index.get((pair.target_b, voice, tone))
                    if audio_a is None or audio_b is None:
                        continue
                    instances.append(MinimalPairInstance(pair, voice, tone, audio_a, audio_b))

    instances.sort(key=instance_sort_key)
    return instances


# ---------------------------------------------------------------------------
# Counting / grouping helpers
# ---------------------------------------------------------------------------


def count_table(
    pairs: Iterable[MinimalPair],
    instances: Iterable[MinimalPairInstance],
) -> dict[str, dict[str, int]]:
    """Per-``pair_type`` pair and instance counts, in ``PAIR_TYPES`` order.

    Every pair type is present (zeros included) so report tables are stable.
    """
    table: dict[str, dict[str, int]] = {
        pair_type: {"n_pairs": 0, "n_instances": 0} for pair_type in PAIR_TYPES
    }
    for pair in pairs:
        table[pair.pair_type]["n_pairs"] += 1
    for instance in instances:
        table[instance.pair.pair_type]["n_instances"] += 1
    return table


def pairs_by_type(pairs: Iterable[MinimalPair]) -> dict[str, list[MinimalPair]]:
    """Group pairs by ``pair_type`` (all types present, each list sorted)."""
    grouped: dict[str, list[MinimalPair]] = {pair_type: [] for pair_type in PAIR_TYPES}
    for pair in pairs:
        grouped[pair.pair_type].append(pair)
    for group in grouped.values():
        group.sort(key=pair_sort_key)
    return grouped


def instances_by_type(
    instances: Iterable[MinimalPairInstance],
) -> dict[str, list[MinimalPairInstance]]:
    """Group instances by pair type (all types present, each list sorted)."""
    grouped: dict[str, list[MinimalPairInstance]] = {pair_type: [] for pair_type in PAIR_TYPES}
    for instance in instances:
        grouped[instance.pair.pair_type].append(instance)
    for group in grouped.values():
        group.sort(key=instance_sort_key)
    return grouped


def iter_pair_items(instances: Iterable[MinimalPairInstance]) -> Iterator[PairItem]:
    """Flatten instances into per-side :class:`PairItem` records for task building.

    Yields sides ``"A"`` then ``"B"`` for each instance, instances sorted by
    :func:`instance_sort_key`.  Exactly ``2 * len(instances)`` items are produced.
    """
    ordered = sorted(instances, key=instance_sort_key)
    for instance in ordered:
        for side, audio, other_audio in instance.iter_sides():
            yield PairItem(
                pair_id=instance.pair_id,
                pair=instance.pair,
                voice=instance.voice,
                side=side,
                audio=audio,
                other_audio=other_audio,
            )
