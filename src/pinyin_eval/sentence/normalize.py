"""Pinyin / text normalisation for the sentence-level tasks (design §6).

Pure functions only (no I/O, no API), so they are easy to unit-test:

* ``normalize_pinyin_token`` — surface pinyin (``nv3`` / ``nǚ`` / ``lve4``) to
  ``Syllable(initial, final, tone)`` under the **N0** orthographic rules:
  NFKC, lowercase, tone marks -> tone digits, ``v`` / ``u:`` -> ``ü``,
  ``j/q/x`` + ``u`` -> ``ü``, final aliases, zero-initial y/w rewrites;
* ``align_gold`` — gold characters + raw AISHELL-3 syllables -> per-character
  gold, including the **erhua split** (``dianr3`` -> ``dian3`` + ``er5``);
* ``apply_n1_sandhi`` — textbook 一/不 sandhi on a syllable sequence (N1
  diagnostic, applied to gold and hypothesis independently);
* ``align_char_sequences`` — Levenshtein alignment of gold vs predicted
  character sequences (OpenCC t2s when available, identity fallback);
* ``edit_distance`` — syllable CER numerator.

The module deliberately does not depend on the single-syllable mapping CSV; it
only reuses the frozen tone-mark / initial / y-w tables from
:mod:`pinyin_eval.normalize`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from ..normalize import (
    FINAL_ALIASES,
    INITIALS_BY_LENGTH,
    YW_SURFACE_RULES,
    tone_from_marks,
)

__all__ = [
    "GoldSyllable",
    "Syllable",
    "add_erhua_flag",
    "align_char_sequences",
    "align_gold",
    "apply_n1_sandhi",
    "edit_distance",
    "erhua_flags",
    "is_erhua_surface",
    "normalize_pinyin_token",
    "split_surface",
    "strip_tone_marks",
    "t2s",
    "tone_stripped",
]

_DIGIT_TONE_RE = re.compile(r"([1-5])\s*$")
_TONE_MARK_CHARS = frozenset("āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ")
_APICAL_INITIALS = frozenset({"zh", "ch", "sh", "r", "z", "c", "s"})


# ---------------------------------------------------------------------------
# OpenCC t2s (optional dependency; identity fallback)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _opencc():
    try:  # pragma: no cover - depends on the optional package
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:  # noqa: BLE001 - fall back to identity
        return None


#: characters that OpenCC-python-reimplemented converts correctly even in the
#: identity fallback are none; the fallback is only for environments without it.
def t2s(text: str) -> str:
    """Traditional -> simplified Chinese conversion (identity when OpenCC absent)."""
    if not text:
        return ""
    converter = _opencc()
    if converter is None:
        return text
    return converter.convert(text)


def strip_tone_marks(text: str) -> str:
    """Remove tone diacritics from a pinyin string (keeps tone digits!)."""
    text = unicodedata.normalize("NFKC", text or "")
    out = [ch for ch in text if ch not in _TONE_MARK_CHARS]
    return "".join(out)


# ---------------------------------------------------------------------------
# Syllable normalisation (N0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Syllable:
    """One normalised pinyin syllable.

    ``tone`` is ``1..5``; ``0`` means the tone is missing / unparseable and can
    never match a gold syllable.  ``surface`` is the canonical toneless
    syllable (``initial + final``), used by CER.
    """

    initial: str
    final: str
    tone: int
    char: str = ""

    @property
    def surface(self) -> str:
        return f"{self.initial}{self.final}"

    @property
    def with_tone(self) -> str:
        return f"{self.surface}{self.tone}" if self.tone else self.surface

    def as_dict(self) -> dict:
        return {
            "char": self.char,
            "initial": self.initial,
            "final": self.final,
            "tone": self.tone,
        }


@dataclass(frozen=True, slots=True)
class GoldSyllable(Syllable):
    """A gold syllable bound to one gold character."""

    char: str = ""


def _final_canonical(final: str, initial: str) -> str:
    f = (final or "").strip().replace("v", "ü").replace("u:", "ü")
    f, _tone = tone_from_marks(f)
    f = re.sub(r"[^a-zü]", "", f)
    f = FINAL_ALIASES.get(f, f)
    if initial in {"j", "q", "x"} and f.startswith("u"):
        f = "ü" + f[1:]
    # "yo" is a rare zero-initial surface whose canonical final is "io"
    if initial == "" and f == "yo":
        f = "io"
    return f


def split_surface(surface: str) -> tuple[str, str]:
    """Split a canonical **toneless** syllable surface into ``(initial, final)``.

    ``y`` / ``w`` spellings are rewritten to zero-initial finals, ``j/q/x`` +
    ``u`` becomes ``ü``, and apical syllables keep ``i`` as the final.
    """
    s = (surface or "").strip().lower().replace("v", "ü").replace("u:", "ü")
    s = re.sub(r"[^a-zü]", "", s)
    if not s:
        return "", ""
    # zero-initial y/w surface spellings (yi, wu, yue, weng, ...)
    if s in YW_SURFACE_RULES:
        return YW_SURFACE_RULES[s]
    # greedy longest-initial split; a consonant-only answer keeps the initial
    for initial in INITIALS_BY_LENGTH:
        if s.startswith(initial) and len(s) > len(initial):
            final = s[len(initial):]
            if initial in _APICAL_INITIALS and final in {"", "i"}:
                final = "i"
            return initial, _final_canonical(final, initial)
    if s.startswith("y"):
        rest = s[1:]
        rest = "ü" if rest == "u" else rest
        return "", _final_canonical(rest, "")
    if s.startswith("w"):
        return "", _final_canonical(s[1:], "")
    return "", _final_canonical(s, "")


def normalize_pinyin_token(token: str, *, char: str = "") -> Syllable:
    """Normalise one surface pinyin token to a :class:`Syllable` (N0).

    ``nǚ`` / ``nv3`` / ``nü3`` -> ``Syllable("n", "ü", 3)``;
    ``dianr3`` keeps the merged surface (call :func:`align_gold` for the split).
    """
    t = unicodedata.normalize("NFKC", token or "").strip().lower()
    t = t.replace("u:", "ü").replace("v", "ü")
    stripped, mark_tone = tone_from_marks(t)
    m = _DIGIT_TONE_RE.search(stripped)
    if m:
        tone = int(m.group(1))
        stripped = stripped[: m.start()]
    else:
        tone = int(mark_tone) if mark_tone else 0
    surface = re.sub(r"[^a-zü]", "", stripped)
    initial, final = split_surface(surface)
    return Syllable(initial=initial, final=final, tone=tone, char=char)


def is_erhua_surface(surface: str) -> bool:
    """True for a merged erhua surface such as ``dianr`` / ``huar`` / ``zher``.

    ``er`` itself (and a bare ``r``) are *not* erhua forms.
    """
    s = (surface or "").strip().lower()
    if len(s) < 3 or not s.endswith("r"):
        return False
    base = s[:-1]
    if base in {"", "e"}:
        return False
    # a plausible base: must not itself end with r and must start with a valid
    # pinyin letter run
    return bool(re.fullmatch(r"[a-zü]+", base)) and not base.endswith("r")


def align_gold(chars: str, tokens: list[str]) -> list[GoldSyllable] | None:
    """Bind gold characters to raw pinyin tokens, splitting merged erhua forms.

    Returns ``None`` when the token count cannot be reconciled with the
    character count (such utterances are excluded from the pilot sample and
    counted in the data-prep report).

    Erhua rule (design §6.1): ``点儿`` + ``dianr3`` -> ``点 dian3`` + ``儿 er5``;
    the base syllable keeps the merged tone, the ``儿`` entry is neutral (5).
    """
    out: list[GoldSyllable] = []
    i = 0
    for token in tokens:
        syl = normalize_pinyin_token(token)
        if is_erhua_surface(syl.surface) and i + 1 < len(chars):
            base = normalize_pinyin_token(syl.surface[:-1])
            if base.tone == 0:
                base = Syllable(base.initial, base.final, syl.tone)
            out.append(GoldSyllable(initial=base.initial, final=base.final,
                                    tone=base.tone, char=chars[i]))
            out.append(GoldSyllable(initial="", final="er", tone=5, char=chars[i + 1]))
            i += 2
            continue
        if i >= len(chars):
            return None
        out.append(GoldSyllable(initial=syl.initial, final=syl.final,
                                tone=syl.tone, char=chars[i]))
        i += 1
    if i != len(chars):
        return None
    return out


# ---------------------------------------------------------------------------
# N1: textbook 一 / 不 sandhi (diagnostic)
# ---------------------------------------------------------------------------


def _sandhi_tone(char: str, next_tone: int | None) -> int | None:
    if char == "一":
        return 2 if next_tone == 4 else 4
    if char == "不":
        return 2 if next_tone == 4 else 4
    return None


def apply_n1_sandhi(syllables: list[GoldSyllable]) -> list[GoldSyllable]:
    """Return a copy with textbook 一/不 sandhi applied (design §6.1 N1)."""
    out = list(syllables)
    for idx, syl in enumerate(syllables):
        nxt = syllables[idx + 1].tone if idx + 1 < len(syllables) else None
        tone = _sandhi_tone(syl.char, nxt)
        if tone is not None and tone != syl.tone:
            out[idx] = GoldSyllable(initial=syl.initial, final=syl.final,
                                    tone=tone, char=syl.char)
    return out


def tone_stripped(syllables: list[Syllable]) -> list[str]:
    """Toneless surfaces (tone-stripped CER)."""
    return [s.surface for s in syllables]


# ---------------------------------------------------------------------------
# Character-sequence alignment
# ---------------------------------------------------------------------------


def align_char_sequences(
    gold: list[str], hyp: list[str]
) -> tuple[list[tuple[int, int | None]], int, int]:
    """Levenshtein alignment of two character sequences.

    Returns ``(pairs, n_insertions, n_deletions)`` where ``pairs`` maps every
    gold index to a hypothesis index (or ``None`` for a deletion).  Insertions
    are hypothesis characters with no gold partner (design §6.3).
    """
    n, m = len(gold), len(hyp)
    # dp[i][j] = edit distance between gold[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        gi = gold[i - 1]
        for j in range(1, m + 1):
            cost = 0 if gi == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )
    pairs: list[tuple[int, int | None]] = []
    insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if gold[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
            continue
        # j > 0, insertion in hypothesis
        insertions += 1
        j -= 1
    pairs.reverse()
    deletions = sum(1 for _g, h in pairs if h is None)
    return pairs, insertions, deletions


def edit_distance(a: list[str], b: list[str]) -> int:
    """Plain Levenshtein distance between two string sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return prev[m]


# ---------------------------------------------------------------------------
# Special-subset flags
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpecialFlags:
    neutral: bool = False
    yibu: bool = False
    erhua: bool = False

    @property
    def special(self) -> bool:
        return self.neutral or self.yibu or self.erhua

    def as_dict(self) -> dict[str, bool]:
        return {"neutral": self.neutral, "yibu": self.yibu, "erhua": self.erhua}


def erhua_flags(chars: str, tokens: list[str]) -> SpecialFlags:
    """Compute the special-subset flags (design §3.1) from a gold transcript."""
    gold = align_gold(chars, tokens)
    syllables = gold or [normalize_pinyin_token(t) for t in tokens]
    neutral = any(s.tone == 5 for s in syllables)
    yibu = any(c in "一不" for c in chars)
    erhua = any(is_erhua_surface(normalize_pinyin_token(t).surface) for t in tokens)
    return SpecialFlags(neutral=neutral, yibu=yibu, erhua=erhua)


def add_erhua_flag(flags: SpecialFlags, value: bool) -> SpecialFlags:
    return SpecialFlags(neutral=flags.neutral, yibu=flags.yibu, erhua=flags.erhua or value)
