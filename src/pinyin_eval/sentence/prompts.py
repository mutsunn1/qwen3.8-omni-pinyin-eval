"""Prompt templates for the sentence-level tasks (design §8).

The three templates follow the frozen design document verbatim; the only
implementation-level decision is task A's output envelope: the API requires a
JSON *object* for ``response_format={"type": "json_object"}``, so the syllable
array is wrapped in ``{"syllables": [...]}`` as allowed by design §5.

``PROMPT_VERSION`` (package root) must be bumped whenever a template changes;
the runner records the version plus a sha256 fingerprint of the rendered prompt.
"""

from __future__ import annotations

import hashlib
import json

from . import PROMPT_VERSION
from .records import SentenceItem

__all__ = [
    "D_PROMPT_VERSION",
    "PROMPT_TEMPLATES",
    "PROMPT_VERSION",
    "build_prompt",
    "prompt_fingerprints",
    "prompt_sha256",
    "prompt_version_for_task",
]

#: task D has its own frozen prompt/schema (design v0.2 附录 A.7)
D_PROMPT_VERSION = "sentence-d-v1"

#: design §8.1 — whole-utterance pinyin transcription (audio only)
TEMPLATE_A = """你是普通话整句拼音转写助手。请只根据音频，逐字输出每个汉字的拼音。
要求：
1. 按音频中汉字的顺序，每个汉字一个条目。
2. 每个条目包含 char、initial、final、tone；tone 取 1–5（5=轻声）。
3. 零声母 initial 写空字符串；ü 直接写 ü；儿化音把基础音节和“儿”（er5）分开写。
4. 只输出 JSON 对象：{{"syllables":[{{"char":"你","initial":"n","final":"i","tone":3}}]}}；
   不要输出标点、解释或 Markdown。"""

#: design §8.2 — whole-utterance pronunciation evaluation (audio + text)
TEMPLATE_B = """你是普通话发音评测助手。音频是学习者朗读，文本是目标句子。
目标句子：{text}
请逐字判断实际发音，输出 JSON 对象：
{{"words":[{{"text":"字","tone_ok":true,"consonant_ok":true,"vowel_ok":true}}]}}
要求：
1. text 必须按目标文本逐字原样输出（繁体保持繁体），不得改写、简繁转换或合并；
   顺序与目标文本一致。
2. 只判断 tone / consonant / vowel 三个维度，不输出拼音。
3. 只输出 JSON，不要解释。"""

#: design §8.3 — whole-utterance reading scoring (audio + text)
TEMPLATE_C = """你是普通话朗读评分助手。音频是学习者朗读，文本是目标句子。
目标句子：{text}
请给 accuracy / fluency / prosody 三个维度打分，各 1–5 分，允许 1 位小数。
只输出 JSON 对象：{{"accuracy": 4.0, "fluency": 4.5, "prosody": 4.0}}
不要输出评分理由。"""

#: design v0.2 附录 A.7 — ASR-text-conditioned alignment + scoring (task D)
TEMPLATE_D = """你是普通话发音评测助手。音频是学习者朗读；文本来自 ASR，可能包含错字、多字或少字。
输入文本：{text}
请以音频为准，按输入文本顺序逐字输出 alignment 数组：
- text：原样复制输入字；
- aligned：该字是否对应音频中的实际发音；
- text_error：该字本身是否错误（错字/多字）；
- error_type：若对齐且文本正确，取 none/tone/consonant/vowel，否则填 null；
- score：该字发音 1–5；无法对齐填 null。
并输出整句 scores：accuracy / fluency / prosody，各 1–5，允许 1 位小数。
只输出 JSON 对象：{{"alignment":[{{"text":"字","aligned":true,"text_error":false,"error_type":"none","score":4.5}}],"scores":{{"accuracy":4.3,"fluency":4.5,"prosody":4.0}}}}
不要解释、不要 Markdown。"""

PROMPT_TEMPLATES: dict[str, str] = {"A": TEMPLATE_A, "B": TEMPLATE_B, "C": TEMPLATE_C, "D": TEMPLATE_D}


def prompt_version_for_task(task: str) -> str:
    return D_PROMPT_VERSION if task.upper() == "D" else PROMPT_VERSION


def build_prompt(task: str, item: SentenceItem | None = None) -> tuple[str, str]:
    """Render ``(system_prompt, user_prompt)`` for one task/item.

    B/C/D embed the target text (design §4.2/§4.3, 附录 A.3); A sees audio only.
    """
    task = task.upper()
    if task not in PROMPT_TEMPLATES:
        raise KeyError(f"unknown sentence task {task!r}")
    template = PROMPT_TEMPLATES[task]
    if task == "A":
        return "", template.format()
    text = item.text if item is not None else ""
    return "", template.format(text=text)


def prompt_sha256(system_prompt: str, user_prompt: str, version: str | None = None) -> str:
    """Fingerprint of the rendered prompt (recorded per request in the manifest)."""
    payload = json.dumps(
        {"system": system_prompt, "user": user_prompt,
         "version": version or PROMPT_VERSION},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prompt_fingerprints() -> dict[str, str]:
    """sha256 of each raw template (frozen in ``run_manifest.json``)."""
    out: dict[str, str] = {}
    for task, template in PROMPT_TEMPLATES.items():
        body = template.format() if task == "A" else template
        out[task] = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return out
