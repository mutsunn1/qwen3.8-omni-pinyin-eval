"""Per-item scoring for the sentence-level pinyin experiment (design §6, §7).

The module turns one :class:`~pinyin_eval.sentence.records.SentenceItem` plus the
parsed model reply (one plain dict, see
:mod:`pinyin_eval.sentence.parsing`) into a flat, JSON-serialisable row that the
aggregation code in :mod:`pinyin_eval.sentence.metrics` consumes.  It performs
no I/O and no statistics: every field below is defined so a report can be
reconstructed from the rows alone.

Task A — whole-utterance pinyin transcription
---------------------------------------------
Gold is ``align_gold(chars, pinyin_surface)`` (design §6.1: NFKC/tonemark
normalisation, merged erhua split into base syllable + ``er5``).  The hypothesis
syllables are re-canonicalised through :func:`split_surface` so the score does
not depend on how the parser spelled a surface (``y``/``w``, ``j/q/x + u``,
``iu``/``iou`` aliases, ...).  Gold and hypothesis character sequences are
aligned with :func:`align_char_sequences` after ``t2s`` on both sides.

Definitions (all counts are over gold syllables, which equal gold characters):
``n_matched``
    Gold syllables that received a hypothesis syllable from the character
    alignment (``n_gold - n_deletions``); ``alignment_coverage = n_matched /
    n_gold``.
``n_initial_correct`` / ``n_final_correct`` / ``n_tone_correct``
    Aligned gold/hypothesis pairs whose initial / final / tone is identical.  A
    tone is only correct when the gold tone is valid (1–5) and equals the
    hypothesis tone (a missing hypothesis tone is stored as ``0`` and never
    matches).
``n_joint_correct``
    Aligned pairs with all three of initial, final and tone correct.  A
    substituted character pair is still compared by pinyin (per the design,
    character substitutions are not a separate error class).
``n_insertions`` / ``n_deletions``
    Hypothesis characters with no gold partner / gold characters with no
    hypothesis partner (design §6.3); insertions are reported but never enter a
    denominator.
``cer_tone_num`` / ``cer_tone_den``
    Tone-sensitive pinyin CER numerator (``edit_distance`` between the gold and
    hypothesis syllable ``surface+tone`` sequences) over ``n_gold``.
``cer_stripped_num`` / ``cer_stripped_den``
    Same with tone-stripped surfaces only.
``sentence_exact``
    ``n_gold > 0``, no deletion, no insertion and every aligned pair has a fully
    correct initial/final/tone (character identity itself is *not* additionally
    required — the pinyin is the evaluated target; ``n_char_mismatches`` reports
    the count for diagnostics).
``n_char_mismatches``
    Aligned pairs whose ``t2s`` characters differ (homophone substitutions).
``cat_initial_stats`` / ``cat_final_stats`` / ``cat_tone_stats``
    Per-category correctness counters for design §7.4's category-weighted
    comparison with the single-syllable results.  Each is a **JSON string** (keys
    sorted via ``json.dumps(..., sort_keys=True)``, so deterministic and flat for
    pandas/pyarrow) mapping the *gold* category to ``{"n": int, "correct": int}``:
    the gold initial / final, or the gold tone as a string (``"1"``–``"5"``; a
    gold surface with an invalid tone would appear as ``"0"``).  The zero-initial
    category is the empty string ``""``, verbatim.  One increment per gold
    character: ``n`` counts every gold character, matched or deleted (the same
    denominator as ``n_joint_correct``), while ``correct`` counts only characters
    whose initial / final / tone matched the aligned hypothesis syllable (a
    deletion is therefore counted in ``n`` but never in ``correct``).
    Parse-fail rows carry ``"{}"`` for all three.
N1 diagnostics
    Textbook 一/不 sandhi (design §6.1) is applied independently to the gold and
    to the hypothesis (never mixed into the N0 fields); the same alignment is
    reused because sandhi only changes tones.  ``n1_*`` fields mirror their N0
    counterparts.

Rows whose ``parse_status`` is not ``ok`` keep ``n_gold`` (needed for data
accounting) but every count is zero, ``sentence_exact = False`` and the row is
excluded from metric denominators upstream.

Task B — whole-utterance pronunciation evaluation
-------------------------------------------------
Gold units are ``item.labels`` (position ordered; label ``0`` = *wrong* is the
positive class).  Hypothesis units are ``parsed["words"]``.  Alignment is a
Levenshtein alignment of the unit texts after ``t2s`` (``align_char_sequences``
is reused with one element per unit); an aligned pair only counts as *matched*
when its texts are equal after ``t2s``.  A gold unit without an equal-text
hypothesis unit increments ``n_missing`` and is counted as ``fn`` for **every**
dimension (the model produced no verdict, so it gets no credit for detecting an
error); ``n_inserted`` counts hypothesis units that are not matched to a gold
unit.  ``tp/fp/fn/tn`` are computed over the resulting ``n_units`` gold units:
positive class = gold label ``0`` (wrong), ``pred_error = not *_ok``.

The zero-initial consonant sub-counts (``n_zero_initial_*``) are only filled for
the ``consonant`` dimension (design §4.2 / §6.4: reported separately, never mixed
into the main conclusion); they are ``0`` for ``tone`` / ``vowel``.

Task C — whole-utterance reading scoring
----------------------------------------
One row per item: gold three-rater means and the predicted scores.  On parse
failure the predicted dimensions are ``None`` (the gold values are still stored
so failure rates and the native anchor stay countable).  ``gold_mean`` /
``pred_mean`` are the means of the three dimensions (``None`` when incomplete).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .normalize import (
    GoldSyllable,
    Syllable,
    align_char_sequences,
    align_gold,
    apply_n1_sandhi,
    edit_distance,
    split_surface,
    t2s,
)
from .records import SentenceItem

__all__ = ["build_rows", "score_a", "score_b", "score_c"]

#: task-B dimensions, in report order
_DIMENSIONS: tuple[str, ...] = ("tone", "consonant", "vowel")

#: task-C dimensions, in report order
_SCORE_DIMENSIONS: tuple[str, ...] = ("accuracy", "fluency", "prosody")

#: flags that define the "special" subsets (design §3.1 / §7.1)
_SPECIAL_FLAGS: tuple[str, ...] = ("neutral", "yibu", "erhua")


# ---------------------------------------------------------------------------
# Small coercion helpers
# ---------------------------------------------------------------------------


def _as_tone(value: Any) -> int:
    """Coerce a predicted tone to ``1..5``; anything else becomes ``0``.

    ``0`` means "missing / invalid tone" and can never match a gold tone
    (design §6.1); booleans are rejected because ``True == 1`` in Python.
    """
    if value is None or isinstance(value, bool):
        return 0
    try:
        tone = int(value)
    except (TypeError, ValueError):
        return 0
    return tone if 1 <= tone <= 5 else 0


def _as_score(value: Any) -> float | None:
    """Coerce one task-C score to ``float``; missing / non-numeric becomes ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_defined(values: Iterable[float | None]) -> float | None:
    """Mean of the non-``None`` values, or ``None`` when nothing is defined."""
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return float(sum(defined)) / len(defined)


def _mean3(scores: Mapping[str, float | None]) -> float | None:
    """Mean of the three task-C scores; ``None`` unless all three are present."""
    values = [scores.get(dimension) for dimension in _SCORE_DIMENSIONS]
    if any(value is None for value in values):
        return None
    return float(sum(values)) / 3.0  # type: ignore[arg-type]


def _ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator`` or ``None`` when the denominator is zero."""
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _hyp_syllable(entry: Mapping[str, Any]) -> Syllable:
    """Canonicalise one predicted task-A syllable.

    ``parsing.normalize_predicted_pair`` already normalises most model
    spellings, but it trusts ``initial``/``final`` verbatim when the initial is
    a real initial (e.g. ``j`` + ``u`` stays ``u`` rather than ``ü``).  Splitting
    ``initial + final`` through :func:`split_surface` once more makes the
    comparison canonical regardless of the parser spelling; it is idempotent
    for already-normalised pairs.
    """
    initial = str(entry.get("initial") or "")
    final = str(entry.get("final") or "")
    tone = _as_tone(entry.get("tone"))
    if initial or final:
        initial, final = split_surface(initial + final)
    return Syllable(
        initial=initial,
        final=final,
        tone=tone,
        char=str(entry.get("char") or ""),
    )


def _base_row(
    item: SentenceItem, task: str, condition: str, parse_status: str
) -> dict[str, Any]:
    """Envelope shared by the task-A/B/C rows.

    ``flags`` is always emitted with the same three boolean keys (design §3.1),
    even when the item carries no flags: a homogeneous schema keeps the scored
    rows valid for Parquet export (PyArrow rejects struct columns whose rows
    disagree on their child fields).
    """
    item_flags = item.flags or {}
    flags = {name: bool(item_flags.get(name)) for name in _SPECIAL_FLAGS}
    return {
        "item_id": item.item_id,
        "task": task,
        "condition": condition,
        "cluster_id": item.cluster_id,
        "parse_status": parse_status,
        "speaker": item.speaker,
        "duration_s": float(item.duration_s or 0.0),
        "flags": flags,
        "stable": not any(flags.values()),
    }


# ---------------------------------------------------------------------------
# Task A — whole-utterance pinyin transcription
# ---------------------------------------------------------------------------

_A_COUNT_FIELDS: tuple[str, ...] = (
    "n_hyp",
    "n_matched",
    "n_deletions",
    "n_insertions",
    "n_joint_correct",
    "n_initial_correct",
    "n_final_correct",
    "n_tone_correct",
    "cer_tone_num",
    "cer_stripped_num",
    "n_char_mismatches",
    "n1_n_joint_correct",
    "n1_n_initial_correct",
    "n1_n_final_correct",
    "n1_n_tone_correct",
    "n1_cer_tone_num",
)

#: task-A per-category correctness columns (JSON-string values keep the row flat)
_A_CATEGORY_FIELDS: tuple[str, ...] = (
    "cat_initial_stats",
    "cat_final_stats",
    "cat_tone_stats",
)


def _bump_category(counters: dict[str, list[int]], key: str, correct: bool) -> None:
    """Increment one category's ``[n, correct]`` counter pair."""
    counts = counters.setdefault(key, [0, 0])
    counts[0] += 1
    if correct:
        counts[1] += 1


def _category_stats_json(counters: Mapping[str, list[int]]) -> str:
    """Serialise ``category -> [n, correct]`` as a deterministic JSON object string."""
    payload = {
        key: {"n": int(counts[0]), "correct": int(counts[1])}
        for key, counts in counters.items()
    }
    return json.dumps(payload, sort_keys=True)


def score_a(
    item: SentenceItem, parsed: Mapping[str, Any], condition: str = ""
) -> dict[str, Any]:
    """Score one task-A item against its gold pinyin (see module docstring).

    Parameters
    ----------
    item:
        Evaluation item; ``chars`` and ``pinyin_surface`` must be alignable
        (``align_gold`` returns ``None`` otherwise — such items are excluded at
        data-prep time, and this function raises ``ValueError`` for them).
    parsed:
        Output of ``parsing.parse_task_a``.  ``parse_status != "ok"`` produces a
        zero-count row that still carries ``n_gold``, ``flags`` and ``stable``.
    condition:
        Generation condition label stored verbatim in the row (``"C1"``/``"C2"``).

    Returns
    -------
    dict
        JSON-serialisable row with the fields described in the module docstring,
        including the ``cat_*_stats`` JSON-string category counters.
    """
    gold = align_gold(item.chars, list(item.pinyin_surface or []))
    if gold is None:
        raise ValueError(
            f"score_a: gold for item {item.item_id!r} is not alignable "
            f"(align_gold returned None: {len(item.chars)} chars vs "
            f"{len(item.pinyin_surface or [])} tokens)"
        )

    parse_status = str(parsed.get("parse_status") or "ok")
    n_gold = len(gold)
    row = _base_row(item, "A", condition, parse_status)
    row["n_gold"] = n_gold
    row["alignment_coverage"] = 0.0
    row["sentence_exact"] = False
    row["n1_sentence_exact"] = False
    row["cer_tone_den"] = n_gold
    row["cer_stripped_den"] = n_gold
    row["n1_cer_tone_den"] = n_gold
    row.update(dict.fromkeys(_A_COUNT_FIELDS, 0))
    row.update(dict.fromkeys(_A_CATEGORY_FIELDS, "{}"))

    if parse_status != "ok":
        return row

    syllables = parsed.get("syllables") or []
    hyp = [_hyp_syllable(entry) for entry in syllables if isinstance(entry, Mapping)]
    n_hyp = len(hyp)

    gold_chars = [t2s(syllable.char) for syllable in gold]
    hyp_chars = [t2s(syllable.char) for syllable in hyp]
    pairs, n_insertions, n_deletions = align_char_sequences(gold_chars, hyp_chars)

    # N1 diagnostic: textbook 一/不 sandhi applied independently on each side.
    gold_n1 = apply_n1_sandhi(gold)
    hyp_as_gold = [
        GoldSyllable(initial=syllable.initial, final=syllable.final,
                     tone=syllable.tone, char=syllable.char)
        for syllable in hyp
    ]
    hyp_n1 = apply_n1_sandhi(hyp_as_gold)

    n_matched = 0
    n_initial_correct = n_final_correct = n_tone_correct = n_joint_correct = 0
    n1_initial_correct = n1_final_correct = n1_tone_correct = n1_joint_correct = 0
    n_char_mismatches = 0

    cat_initial: dict[str, list[int]] = {}
    cat_final: dict[str, list[int]] = {}
    cat_tone: dict[str, list[int]] = {}

    for gold_index, hyp_index in pairs:
        gold_syllable = gold[gold_index]
        hyp_syllable = hyp[hyp_index] if hyp_index is not None else None

        initial_ok = (
            hyp_syllable is not None and gold_syllable.initial == hyp_syllable.initial
        )
        final_ok = hyp_syllable is not None and gold_syllable.final == hyp_syllable.final
        tone_ok = (
            hyp_syllable is not None
            and gold_syllable.tone in (1, 2, 3, 4, 5)
            and gold_syllable.tone == hyp_syllable.tone
        )

        # design §7.4 category-weighted counters: one increment per gold character
        # (the ``n_joint_correct`` denominator), deletions counted in ``n`` only.
        _bump_category(cat_initial, gold_syllable.initial, initial_ok)
        _bump_category(cat_final, gold_syllable.final, final_ok)
        _bump_category(cat_tone, str(gold_syllable.tone), tone_ok)

        if hyp_syllable is None:
            continue
        n_matched += 1
        if gold_chars[gold_index] != hyp_chars[hyp_index]:
            n_char_mismatches += 1

        if initial_ok:
            n_initial_correct += 1
        if final_ok:
            n_final_correct += 1
        if tone_ok:
            n_tone_correct += 1
        if initial_ok and final_ok and tone_ok:
            n_joint_correct += 1

        n1_gold = gold_n1[gold_index]
        n1_hyp = hyp_n1[hyp_index]
        n1_initial_ok = n1_gold.initial == n1_hyp.initial
        n1_final_ok = n1_gold.final == n1_hyp.final
        n1_tone_ok = n1_gold.tone in (1, 2, 3, 4, 5) and n1_gold.tone == n1_hyp.tone
        if n1_initial_ok:
            n1_initial_correct += 1
        if n1_final_ok:
            n1_final_correct += 1
        if n1_tone_ok:
            n1_tone_correct += 1
        if n1_initial_ok and n1_final_ok and n1_tone_ok:
            n1_joint_correct += 1

    # Every hypothesis syllable is either paired with a gold syllable or an
    # insertion; ``align_char_sequences`` already returns that insertion count.
    n_insertions_aligned = n_hyp - n_matched
    if n_insertions_aligned != n_insertions:  # pragma: no cover - defensive
        raise AssertionError(
            "score_a: inconsistent insertion count "
            f"({n_insertions_aligned} vs alignment {n_insertions})"
        )

    cer_tone_num = edit_distance(
        [syllable.with_tone for syllable in gold],
        [syllable.with_tone for syllable in hyp],
    )
    cer_stripped_num = edit_distance(
        [syllable.surface for syllable in gold],
        [syllable.surface for syllable in hyp],
    )
    n1_cer_tone_num = edit_distance(
        [syllable.with_tone for syllable in gold_n1],
        [syllable.with_tone for syllable in hyp_n1],
    )

    sentence_exact = (
        n_gold > 0
        and n_deletions == 0
        and n_insertions_aligned == 0
        and n_joint_correct == n_gold
    )
    n1_sentence_exact = (
        n_gold > 0
        and n_deletions == 0
        and n_insertions_aligned == 0
        and n1_joint_correct == n_gold
    )

    row.update(
        {
            "n_hyp": n_hyp,
            "n_matched": n_matched,
            "n_deletions": n_deletions,
            "n_insertions": n_insertions_aligned,
            "n_joint_correct": n_joint_correct,
            "n_initial_correct": n_initial_correct,
            "n_final_correct": n_final_correct,
            "n_tone_correct": n_tone_correct,
            "cer_tone_num": cer_tone_num,
            "cer_stripped_num": cer_stripped_num,
            "n_char_mismatches": n_char_mismatches,
            "n1_n_joint_correct": n1_joint_correct,
            "n1_n_initial_correct": n1_initial_correct,
            "n1_n_final_correct": n1_final_correct,
            "n1_n_tone_correct": n1_tone_correct,
            "n1_cer_tone_num": n1_cer_tone_num,
            "alignment_coverage": _ratio(n_matched, n_gold),
            "sentence_exact": bool(sentence_exact),
            "n1_sentence_exact": bool(n1_sentence_exact),
            "cat_initial_stats": _category_stats_json(cat_initial),
            "cat_final_stats": _category_stats_json(cat_final),
            "cat_tone_stats": _category_stats_json(cat_tone),
        }
    )
    return row


# ---------------------------------------------------------------------------
# Task B — whole-utterance pronunciation evaluation
# ---------------------------------------------------------------------------


def _gold_label(unit: Mapping[str, Any], dimension: str) -> int | None:
    """Binary gold label for one dimension: ``0`` wrong / ``1`` correct / ``None``."""
    value = unit.get(dimension)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    return None


def _matched_units(
    labels: list[Mapping[str, Any]], words: list[Mapping[str, Any]]
) -> tuple[dict[int, int], int]:
    """Align gold units to predicted units by text after ``t2s``.

    Returns ``(gold_index -> hyp_index, n_inserted_hyp_units)``.  Only pairs
    whose ``t2s`` texts are equal count as matched; a same-position substitution
    (different character) leaves the gold unit unmatched on purpose, because the
    model's verdict refers to a different character.
    """
    gold_texts = [t2s(str(unit.get("text") or "")) for unit in labels]
    hyp_texts = [t2s(str(word.get("text") or "")) for word in words]
    pairs, _, _ = align_char_sequences(gold_texts, hyp_texts)
    matched: dict[int, int] = {}
    for gold_index, hyp_index in pairs:
        if hyp_index is None:
            continue
        if gold_texts[gold_index] == hyp_texts[hyp_index]:
            matched[gold_index] = hyp_index
    n_inserted = len(words) - len(set(matched.values()))
    return matched, n_inserted


def score_b(
    item: SentenceItem, parsed: Mapping[str, Any], condition: str = ""
) -> list[dict[str, Any]]:
    """Score one task-B item, returning one row per dimension.

    Returns three rows (``tone`` / ``consonant`` / ``vowel``) when
    ``parsed["parse_status"] == "ok"``, and an **empty list** otherwise (task-B
    parse failure means there is no per-character verdict to score; failures are
    counted from the record envelope / usage, not from these rows).

    See the module docstring for the alignment and missing-unit conventions.
    """
    parse_status = str(parsed.get("parse_status") or "ok")
    if parse_status != "ok":
        return []

    labels = [unit for unit in (item.labels or []) if isinstance(unit, Mapping)]
    words = [word for word in (parsed.get("words") or []) if isinstance(word, Mapping)]
    matched, n_inserted = _matched_units(labels, words)

    n_units = len(labels)
    n_matched = len(matched)
    n_missing = n_units - n_matched
    coverage = _ratio(n_matched, n_units)
    base = _base_row(item, "B", condition, parse_status)
    base["learner"] = item.learner
    base["native"] = bool(item.native)

    rows: list[dict[str, Any]] = []
    for dimension in _DIMENSIONS:
        tp = fp = fn = tn = 0
        zero_tp = zero_fp = zero_fn = zero_tn = 0
        for gold_index, unit in enumerate(labels):
            zero_initial = dimension == "consonant" and bool(unit.get("zero_initial"))
            gold_value = _gold_label(unit, dimension)
            hyp_index = matched.get(gold_index)
            if hyp_index is None:
                # No equal-text prediction: no credit for detecting an error.
                fn += 1
                if zero_initial:
                    zero_fn += 1
                continue
            pred_error = not bool(words[hyp_index].get(f"{dimension}_ok"))
            if gold_value == 0:
                if pred_error:
                    tp += 1
                    if zero_initial:
                        zero_tp += 1
                else:
                    fn += 1
                    if zero_initial:
                        zero_fn += 1
            elif gold_value == 1:
                if pred_error:
                    fp += 1
                    if zero_initial:
                        zero_fp += 1
                else:
                    tn += 1
                    if zero_initial:
                        zero_tn += 1
            else:
                # Invalid / missing gold label: not evaluable, count as fn.
                fn += 1
                if zero_initial:
                    zero_fn += 1

        row = dict(base)
        row.update(
            {
                "dimension": dimension,
                "n_units": n_units,
                "n_matched": n_matched,
                "alignment_coverage": coverage,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "n_missing": n_missing,
                "n_inserted": n_inserted,
                "n_zero_initial_tp": zero_tp if dimension == "consonant" else 0,
                "n_zero_initial_fp": zero_fp if dimension == "consonant" else 0,
                "n_zero_initial_fn": zero_fn if dimension == "consonant" else 0,
                "n_zero_initial_tn": zero_tn if dimension == "consonant" else 0,
            }
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Task C — whole-utterance reading scoring
# ---------------------------------------------------------------------------


def score_c(
    item: SentenceItem, parsed: Mapping[str, Any], condition: str = ""
) -> dict[str, Any]:
    """Score one task-C item (three rating dimensions, one row per item).

    Predicted dimensions are ``None`` when the reply failed to parse; the gold
    values are always recorded so that failure rates and the native anchor stay
    countable.  ``gold_mean`` / ``pred_mean`` are ``None`` unless all three
    dimensions are present.
    """
    parse_status = str(parsed.get("parse_status") or "ok")
    base = _base_row(item, "C", condition, parse_status)
    base["learner"] = item.learner
    base["native"] = bool(item.native)

    gold_source = item.scores or {}
    gold = {dimension: _as_score(gold_source.get(dimension))
            for dimension in _SCORE_DIMENSIONS}
    if parse_status == "ok":
        parsed_scores = parsed.get("scores") or {}
        pred = {dimension: _as_score(parsed_scores.get(dimension))
                for dimension in _SCORE_DIMENSIONS}
    else:
        pred = dict.fromkeys(_SCORE_DIMENSIONS)

    row = dict(base)
    row.update({f"gold_{dimension}": gold[dimension] for dimension in _SCORE_DIMENSIONS})
    row.update({f"pred_{dimension}": pred[dimension] for dimension in _SCORE_DIMENSIONS})
    row["gold_mean"] = _mean3(gold)
    row["pred_mean"] = _mean3(pred)
    return row


# ---------------------------------------------------------------------------
# Convenience: build all rows for a run
# ---------------------------------------------------------------------------


def _default_parsers() -> dict[str, Callable[[str], dict[str, Any]]]:
    """Lazy default parser map (keeps this module import-light and cycle-free)."""
    from .parsing import parse_task_a, parse_task_b, parse_task_c

    return {"A": parse_task_a, "B": parse_task_b, "C": parse_task_c}


def _record_dict(record: Any) -> dict[str, Any]:
    """Accept ``GenerationRecord`` objects or plain dicts."""
    if hasattr(record, "to_dict"):
        return dict(record.to_dict())
    return dict(record)


def build_rows(
    items: Iterable[SentenceItem],
    generations: Iterable[Any],
    condition: str,
    parsers: Mapping[str, Callable[[str], dict[str, Any]]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Score a whole run: ``items`` x ``generations`` for one condition.

    Parameters
    ----------
    items:
        The evaluation items (all three tasks may be mixed).
    generations:
        ``GenerationRecord`` objects or dicts; matched to items by
        ``(task, item_id)`` and filtered on ``condition``.
    condition:
        Condition label to score (``"C1"`` / ``"C2"``).  Stored in every row.
    parsers:
        Optional ``{task: callable(response_text) -> parsed dict}`` map; defaults
        to the parsers in :mod:`pinyin_eval.sentence.parsing`.

    Returns
    -------
    dict[str, list[dict]]
        ``{"A": [...], "B": [...], "C": [...]}`` — all three keys are always
        present (possibly empty).  A generation with ``error`` set is scored as
        ``api_error``; a missing generation is scored as ``empty``.
    """
    parser_map = dict(parsers) if parsers is not None else _default_parsers()

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for record in generations:
        payload = _record_dict(record)
        if str(payload.get("condition") or "") != condition:
            continue
        task = str(payload.get("task") or "").upper()
        item_id = payload.get("item_id")
        if not task or item_id is None:
            continue
        index[(task, str(item_id))] = payload

    rows: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for item in items:
        task = str(item.task).upper()
        if task not in rows:
            raise ValueError(f"build_rows: unknown sentence task {item.task!r}")
        parser = parser_map.get(task)
        if parser is None:
            raise ValueError(f"build_rows: no parser supplied for task {task!r}")
        record = index.get((task, item.item_id))
        if record is None:
            parsed: dict[str, Any] = {"parse_status": "empty"}
        elif record.get("error"):
            parsed = {"parse_status": "api_error"}
        else:
            parsed = parser(str(record.get("response_text") or ""))
        if task == "A":
            rows["A"].append(score_a(item, parsed, condition))
        elif task == "B":
            rows["B"].extend(score_b(item, parsed, condition))
        else:
            rows["C"].append(score_c(item, parsed, condition))
    return rows
