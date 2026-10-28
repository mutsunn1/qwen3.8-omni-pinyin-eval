"""Prompt templates (zero-shot, no few-shot examples).

Every prompt variant is generated in two output forms (JSON / free text) and is a
pure function of a :class:`~pinyin_eval.records.TaskItem`'s claim fields.  The
prompt text is stored inside each task item, so a run can be replayed and
request fingerprints stay stable even if this module evolves; bump
``PROMPT_VERSION`` on any wording change.

Design constraints (docs/experiment-design.md):
* zero-shot, no examples, no target-audio leakage;
* target-condition prompts: first judge whether the audio matches the claimed
  pinyin, then report the audio's *true* tone;
* free-recognition prompts: initial, final, tone 1-4;
* A/B prompts: pick A or B among two candidates carrying tone digits.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .records import TaskItem

PROMPT_VERSION = "p1"
PROMPT_VERSION_NOTES = "初版：中文零样本指令；JSON/自由文本双格式；一致性任务先判定后报声调"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
FREE_JSON = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频，识别其声母、韵母和声调。

约定：
- 声母用汉语拼音字母书写（如 zh、b、n）；零声母时该字段填空字符串 ""。
- 韵母用汉语拼音字母书写，ü 写作 ü（如 ang、ian、ü、üan）。
- 声调取 1、2、3、4 之一（1=阴平，2=阳平，3=上声，4=去声）。

只输出一个 JSON 对象，不要输出任何解释或多余文字：
{"initial": "声母", "final": "韵母", "tone": 声调数字}"""

FREE_TEXT = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频，识别其声母、韵母和声调。

约定：
- 声母用汉语拼音字母书写（如 zh、b、n）；零声母时写“无”。
- 韵母用汉语拼音字母书写，ü 写作 ü（如 ang、ian、ü、üan）。
- 声调取 1、2、3、4 之一（1=阴平，2=阳平，3=上声，4=去声）。

只输出以下三行，不要输出任何解释或多余文字：
声母=<声母>
韵母=<韵母>
声调=<1/2/3/4>"""

CONSISTENCY_JSON = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频。说话人声称这段音频读作“{claim}”。

请依次完成两件事：
1. 判断音频的实际读音与上述声称是否一致（只比较声称中给出的信息）；
2. 给出音频的真实声调，取 1、2、3、4 之一；必须以音频为准，不以文字为准。

只输出一个 JSON 对象，不要输出任何解释或多余文字：
{{"consistent": true 或 false, "tone": 声调数字}}"""

CONSISTENCY_TEXT = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频。说话人声称这段音频读作“{claim}”。

请依次完成两件事：
1. 判断音频的实际读音与上述声称是否一致（只比较声称中给出的信息）；
2. 给出音频的真实声调，取 1、2、3、4 之一；必须以音频为准，不以文字为准。

只输出以下两行，不要输出任何解释或多余文字：
一致性=<一致 或 不一致>
声调=<1/2/3/4>"""

AB_JSON = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频，并从两个候选拼音中选出与音频实际读音一致的一个。
候选拼音中的数字 1-4 表示声调（1=阴平，2=阳平，3=上声，4=去声）。

A. {cand_a}
B. {cand_b}

只输出一个 JSON 对象，不要输出任何解释或多余文字：
{{"choice": "A" 或 "B"}}"""

AB_TEXT = """你是一名普通话语音分析专家。请听下面这段单音节普通话音频，并从两个候选拼音中选出与音频实际读音一致的一个。
候选拼音中的数字 1-4 表示声调（1=阴平，2=阳平，3=上声，4=去声）。

A. {cand_a}
B. {cand_b}

只输出以下一行，不要输出任何解释或多余文字：
选择=<A 或 B>"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def task_kind(prompt_condition: str) -> str:
    """free | consistency | ab_choice"""
    if prompt_condition == "free":
        return "free"
    if prompt_condition in ("correct_target", "wrong_target"):
        return "consistency"
    if prompt_condition == "ab_choice":
        return "ab_choice"
    raise ValueError(f"unknown prompt_condition: {prompt_condition!r}")


def claim_text(task: TaskItem) -> str:
    """Human-readable claim used inside consistency prompts.

    * toneless target: ``zhang``（无调拼音）
    * target with tone: ``zhang4``（数字表示声调）
    """
    if task.given_target is None:
        raise ValueError(f"consistency task {task.item_id} has no given_target")
    if task.given_tone is None:
        return task.given_target
    return f"{task.given_target}{int(task.given_tone)}"


def build_prompt_text(task: TaskItem, output_form: str) -> str:
    if output_form not in ("json", "text"):
        raise ValueError(f"unknown output_form: {output_form!r}")
    kind = task_kind(task.prompt_condition)
    if kind == "free":
        return FREE_JSON if output_form == "json" else FREE_TEXT
    if kind == "consistency":
        claim = claim_text(task)
        if task.given_tone is None:
            # segmental claim only: state explicitly that tone is not part of the claim
            claim = f"{claim}（只给出了无调拼音，声调不在声称范围内）"
        else:
            claim = f"{claim}（末尾数字表示声称的声调）"
        template = CONSISTENCY_JSON if output_form == "json" else CONSISTENCY_TEXT
        return template.format(claim=claim)
    if kind == "ab_choice":
        if not task.alternatives or len(task.alternatives) != 2:
            raise ValueError(f"ab_choice task {task.item_id} needs two alternatives")
        template = AB_JSON if output_form == "json" else AB_TEXT
        return template.format(cand_a=task.alternatives[0], cand_b=task.alternatives[1])
    raise AssertionError("unreachable")


def build_prompt_texts(task: TaskItem) -> dict[str, str]:
    return {form: build_prompt_text(task, form) for form in ("json", "text")}


# ---------------------------------------------------------------------------
# Request fingerprint (dedupe key)
# ---------------------------------------------------------------------------


def request_id_for(
    *,
    media_relpath: str,
    output_form: str,
    prompt_text: str,
) -> str:
    """Stable 16-hex fingerprint of one model call.

    Two task items that resolve to the same audio variant, output form and prompt
    text share one API call (e.g. exp3 ``free`` and exp4 ``free`` on the same
    audio).  Thinking / model params are part of the run, not the request id.
    """
    payload = f"{media_relpath}\x00{output_form}\x00{prompt_text}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def describe_prompts() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "notes": PROMPT_VERSION_NOTES,
        "templates": {
            "free_json": FREE_JSON,
            "free_text": FREE_TEXT,
            "consistency_json": CONSISTENCY_JSON,
            "consistency_text": CONSISTENCY_TEXT,
            "ab_json": AB_JSON,
            "ab_text": AB_TEXT,
        },
    }


__all__ = [
    "PROMPT_VERSION",
    "PROMPT_VERSION_NOTES",
    "build_prompt_text",
    "build_prompt_texts",
    "claim_text",
    "describe_prompts",
    "prompt_sha",
    "request_id_for",
    "task_kind",
]
