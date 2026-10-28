# 拼音规范化与输出解析的公共规则。
#
# 设计文档要求：两种输出格式都先规范化（ü/v/u、y/w、大小写、声调数字），
# 再与金标严格匹配。本模块的所有函数都是纯函数，便于单测。
#
# 模块负责四件事：
# 1. 数据集原始 target（385 种拼写）-> 规范 target（378 种）：只实现设计文档
#    明确列出的 7 条零声母规则，其余一律查映射表，不在代码里硬编码 initial/final。
# 2. 模型输出的 initial / final / tone 规范化。
# 3. 由“表层拼音”拆出声母/韵母（用于模型把整音节写在 initial 字段等情形）。
# 4. 声调符号（ā á ǎ à 等）与数字声调的互转。

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

#: 设计文档明确给出的零声母规范化规则。
#: 数据集 target 采用规范零声母写法，其余“原始 target”都能在映射表里查到。
ZERO_INITIAL_RULES: dict[str, str] = {
    "in": "yin",
    "ing": "ying",
    "iu": "you",
    "u": "wu",
    "ui": "wei",
    "un": "wen",
    "ün": "yun",
}

#: 21 个声母（不含零声母），按长度降序用于贪心切分
INITIALS: tuple[str, ...] = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s",
)
INITIALS_BY_LENGTH = tuple(sorted(INITIALS, key=len, reverse=True))

#: 模型可能输出的“无/零声母”写法
EMPTY_INITIAL_TOKENS = {
    "", "none", "null", "nil", "empty", "na", "n/a", "∅", "-", "无", "零", "零声母", "无生母",
}

#: 声调符号 -> (去掉声调符号后的字母, 声调)。声调 0 表示该符号本身是字母。
TONE_MARK_BASE: dict[str, tuple[str, int]] = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("ü", 1), "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4),
    "ü": ("ü", 0),  # 无调
    "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4), "ḿ": ("m", 2),
}

#: 声调符号 -> 声调（兼容旧接口）
TONE_MARK_TABLE: dict[str, int] = {ch: tone for ch, (_base, tone) in TONE_MARK_BASE.items()}

#: 模型常用的完整韵母写法 -> 规范韵母
FINAL_ALIASES: dict[str, str] = {
    "iou": "iu", "uei": "ui", "uen": "un", "üen": "ün", "yo": "io",
}

TONE_WORD_TABLE: dict[str, int] = {
    "一": 1, "1": 1, "一声": 1, "阴平": 1, "第一声": 1,
    "二": 2, "2": 2, "二声": 2, "阳平": 2, "第二声": 2,
    "三": 3, "3": 3, "三声": 3, "上声": 3, "第三声": 3,
    "四": 4, "4": 4, "四声": 4, "去声": 4, "第四声": 4,
}

_PINYIN_STRIP_RE = re.compile(r"[\s,，。;；:：、\"'“”‘’`()（）\[\]【】<>《》!！?？/\\|]+")
_INDEX_PREFIX_RE = re.compile(r"^[\(\[]?\d+[\)\].、:：]\s*")  # e.g. "1. zh" / "(2) ang"


# ---------------------------------------------------------------------------
# 映射表
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MappingEntry:
    target: str
    initial: str
    final: str
    count: int
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "initial": self.initial,
            "final": self.final,
            "count": self.count,
            "source": self.source,
        }


def load_mapping(path: str | Path) -> dict[str, MappingEntry]:
    """Read ``mandarin_tone_syn_target_mapping.csv`` (UTF-8 with BOM)."""
    entries: dict[str, MappingEntry] = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"target", "initial", "final", "count", "source"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"mapping CSV missing columns {required - set(reader.fieldnames or [])}")
        for row in reader:
            target = row["target"].strip()
            if not target:
                continue
            entry = MappingEntry(
                target=target,
                initial=row["initial"].strip(),
                final=row["final"].strip(),
                count=int(row["count"]),
                source=row["source"].strip(),
            )
            if target in entries:
                raise ValueError(f"duplicate target in mapping CSV: {target!r}")
            entries[target] = entry
    if not entries:
        raise ValueError(f"empty mapping CSV: {path}")
    return entries


def canonical_target(raw: str, mapping: dict[str, MappingEntry] | None = None) -> str:
    """Map one of the 385 raw dataset spellings to a canonical target.

    Only the seven documented zero-initial rewrites are applied; the result must
    exist in the mapping table (when provided).  Anything else is returned
    unchanged so that ``prepare-data`` can fail loudly on unknown targets.
    """
    raw = (raw or "").strip()
    canonical = ZERO_INITIAL_RULES.get(raw, raw)
    if mapping is not None and canonical not in mapping:
        raise KeyError(f"raw target {raw!r} -> {canonical!r} not found in target mapping")
    return canonical


# ---------------------------------------------------------------------------
# 输出规范化
# ---------------------------------------------------------------------------

_TONE_NUMBER_SUFFIX_RE = re.compile(r"([1-4])\s*$")
_FULLWIDTH_RE = re.compile(r"[０-９]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_noise(text: str) -> str:
    """Collapse punctuation/quotes/whitespace and leading list indices."""
    text = _nfkc(text or "").strip()
    text = _INDEX_PREFIX_RE.sub("", text)
    text = _PINYIN_STRIP_RE.sub(" ", text).strip()
    # 取第一个非空 token（模型常输出 "zh  (zh)" 之类）
    parts = [p for p in text.split() if p]
    if not parts:
        return ""
    return parts[0]


def tone_from_marks(text: str) -> tuple[str, int | None]:
    """Return ``(text_without_tone_marks, tone_or_None)``."""
    tone: int | None = None
    out_chars: list[str] = []
    for ch in _nfkc(text or ""):
        if ch in TONE_MARK_BASE:
            base, t = TONE_MARK_BASE[ch]
            if t:
                tone = t
            out_chars.append(base)
            continue
        out_chars.append(ch)
    return "".join(out_chars), tone


def normalize_initial(value: str | None) -> str:
    """Normalise a predicted initial.  Returns ``""`` for zero-initial forms."""
    text = strip_noise(value or "")
    text = text.lower().replace("v", "ü")
    stripped, tone = tone_from_marks(text)
    text = stripped
    if text in EMPTY_INITIAL_TOKENS:
        return ""
    text = re.sub(r"[^a-zü]", "", text)
    # y/w 不是声母；统一交给 normalize_syllable 处理。这里先原样保留，
    # 由 normalize_output_pair 负责转写。
    return text


def _final_canonicalisation(final: str, initial: str) -> str:
    f = final.strip().replace("v", "ü").replace("u:", "ü")
    f, _tone = tone_from_marks(f)
    f = FINAL_ALIASES.get(f, f)
    # j/q/x + u -> ü
    if initial in {"j", "q", "x"}:
        if f == "u":
            f = "ü"
        elif f.startswith("u"):
            f = "ü" + f[1:]
    return f


def normalize_final(value: str | None) -> str:
    final = strip_noise(value or "")
    final = final.lower().replace("v", "ü").replace("u:", "ü")
    stripped, _tone = tone_from_marks(final)
    final = re.sub(r"[^a-zü]", "", stripped)
    return FINAL_ALIASES.get(final, final)


def normalize_yw_surface(syllable: str, mapping: dict[str, MappingEntry]) -> tuple[str, str]:
    """Map a y/w surface spelling (``yin``, ``wu``…) back to (initial, final).

    The mapping CSV itself records the canonical (initial, final) for these
    spellings (source == "y"/"w"/"merged"/"final"), so we do not hardcode it.
    """
    syllable = syllable.strip().lower()
    entry = mapping.get(syllable)
    if entry is not None and syllable != "":
        if entry.initial == "" and entry.source in {"y", "w", "final", "merged"}:
            return entry.initial, entry.final
    # fall back: y/w prefix stripping (only when table lookup failed)
    if syllable.startswith("y"):
        rest = syllable[1:]
        rest = "ü" if rest == "u" else rest
        return "", _final_canonicalisation(rest, "")
    if syllable.startswith("w"):
        rest = syllable[1:]
        return "", _final_canonicalisation(rest, "")
    return "", _final_canonicalisation(syllable, "")


def split_syllable(syllable: str) -> tuple[str, str]:
    """Greedy longest-initial split of a pinyin syllable surface."""
    s = (syllable or "").strip().lower().replace("v", "ü")
    s = re.sub(r"[1-4]$", "", s)
    s, _ = tone_from_marks(s)
    if not s:
        return "", ""
    for initial in INITIALS_BY_LENGTH:
        if s.startswith(initial) and len(s) > len(initial):
            return initial, s[len(initial):]
    return "", s


def normalize_output_pair(
    initial: str | None,
    final: str | None,
    mapping: dict[str, MappingEntry],
) -> tuple[str, str]:
    """Normalise a predicted (initial, final) pair to the canonical convention.

    Rules (design doc): lowercase, ü/v/u, y/w, tone digits removed; then compare
    strictly with gold.  y/w spellings and full syllables are rewritten via the
    mapping table so that e.g. ``("y", "in")`` -> ``("", "in")`` and
    ``("", "you")`` -> ``("", "iu")``.  A final that is already a canonical final
    (``ang``, ``ü``) is kept as-is; anything else is treated as a syllable surface.
    """
    raw_initial = normalize_initial(initial)
    raw_final = normalize_final(final)
    known_finals = {entry.final for entry in mapping.values()}

    # full syllable written into one field
    if raw_initial and not raw_final:
        return normalize_syllable(raw_initial, mapping)
    if raw_final and not raw_initial:
        if raw_final in known_finals:
            return "", raw_final
        return normalize_syllable(raw_final, mapping)
    if not raw_initial and not raw_final:
        return "", ""

    # y/w are not initials: rewrite through the mapping table
    if raw_initial in {"y", "w"}:
        return normalize_syllable(raw_initial + raw_final, mapping)

    initial_n = raw_initial
    final_n = _final_canonicalisation(raw_final, initial_n)
    return initial_n, final_n


def normalize_syllable(syllable: str, mapping: dict[str, MappingEntry]) -> tuple[str, str]:
    """Full-syllable surface -> canonical (initial, final)."""
    surface = (syllable or "").strip().lower().replace("v", "ü")
    surface = re.sub(r"[1-4]$", "", surface)
    surface, _tone = tone_from_marks(surface)
    surface = re.sub(r"[^a-zü]", "", surface)
    if not surface:
        return "", ""
    # 1) exact canonical target
    entry = mapping.get(surface)
    if entry is not None:
        return entry.initial, entry.final
    # 2) y/w surface that maps to a canonical target (mapping lookup by target)
    entry = mapping.get(surface)
    if entry is not None:
        return entry.initial, entry.final
    # 3) zero-initial rewrite then canonical lookup
    rewritten = ZERO_INITIAL_RULES.get(surface)
    if rewritten and rewritten in mapping:
        return mapping[rewritten].initial, mapping[rewritten].final
    # 4) greedy split, then canonicalise final
    initial, final = split_syllable(surface)
    if initial in {"y", "w"}:
        return normalize_yw_surface(surface, mapping)
    return initial, _final_canonicalisation(final, initial)


def _looks_like_bare_final(final: str) -> bool:
    """Heuristic: bare finals never start with a consonant cluster that looks initial-like."""
    if final.startswith(("zh", "ch", "sh")):
        return False
    if len(final) >= 2 and final[0] in "bpmfdtnlgkhjqxrzcs" and final[1] in "aeiouü":
        return False
    return True


def normalize_tone(value: object) -> int | None:
    """Normalise a predicted tone to ``1..4`` or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if 1 <= n <= 4 else None
    text = _nfkc(str(value)).strip()
    if not text:
        return None
    stripped, marked = tone_from_marks(text)
    if marked:
        return marked
    text = stripped.strip().lower()
    text = re.sub(r"^(tone|声调|第)\s*[:：=]?\s*", "", text)
    text = re.sub(r"\s*(声|声调|tone)$", "", text)
    text = text.strip("()[]{}<> .,，。:：;；")
    if text in TONE_WORD_TABLE:
        return TONE_WORD_TABLE[text]
    if re.fullmatch(r"[1-4]", text):
        return int(text)
    return None


def normalize_toneless_string(value: str) -> str:
    """Normalise a written toneless pinyin (for target claims / free text)."""
    text = strip_noise(value or "")
    text = text.lower().replace("v", "ü")
    fullwidth_digits = _FULLWIDTH_RE.sub(lambda m: str(ord(m.group()) - 0xFF10), text)
    text = fullwidth_digits
    stripped, _tone = tone_from_marks(text)
    text = re.sub(r"[1-4]$", "", stripped)
    return re.sub(r"[^a-zü]", "", text)


def parse_target_with_tone(value: str | None) -> tuple[str, int | None]:
    """Parse a prompt target that may carry a tone (digit or diacritic).

    ``zhang4`` -> ("zhang", 4); ``zhāng`` -> ("zhang", 1); ``zhang`` -> ("zhang", None).
    """
    text = (value or "").strip()
    if not text:
        return "", None
    text = text.replace("v", "ü")
    text, mark_tone = tone_from_marks(text)
    m = _TONE_NUMBER_SUFFIX_RE.search(text)
    digit_tone = int(m.group(1)) if m else None
    if m:
        text = text[: m.start()]
    toneless = re.sub(r"[^a-zü]", "", text.lower())
    return toneless, mark_tone or digit_tone


__all__ = [
    "EMPTY_INITIAL_TOKENS",
    "INITIALS",
    "MappingEntry",
    "TONE_MARK_TABLE",
    "ZERO_INITIAL_RULES",
    "canonical_target",
    "load_mapping",
    "normalize_final",
    "normalize_initial",
    "normalize_output_pair",
    "normalize_syllable",
    "normalize_tone",
    "normalize_toneless_string",
    "normalize_yw_surface",
    "parse_target_with_tone",
    "split_syllable",
    "strip_noise",
    "tone_from_marks",
]
