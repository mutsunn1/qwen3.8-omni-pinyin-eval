"""Prompt templates (zero-shot, no few-shot examples).

Source: ``docs/experiment-design.md`` section 五 "Prompt 模板" (design doc
updated 2026-10-28, snapshot sha256 4d8f4414…).  Every prompt is a pure function
of a :class:`~pinyin_eval.records.TaskItem`'s claim fields and output form.

Design constraints:

* shared system prompt for all experiments;
* experiment 1/2 share one prompt (toneless target -> tone only), the prompt does
  not reveal whether the target is correct;
* experiment 3: free recognition (initial/final/tone, zero initial -> empty);
* experiments 4-6: A/B choice among two tone-marked candidates;
* detection experiment: target + target_tone -> consistent true/false;
* thinking is an API parameter, audio quality is preprocessing — neither changes
  the prompt text.

``PROMPT_VERSION`` must be bumped on any wording change: it invalidates
request fingerprints, so old generations are never mixed into a new run.

Note on ``{format_instruction}``: the design doc lists, per task, the required
JSON object / fixed text marker.  We append that schema to the doc's generic
format sentence so the model reliably emits parseable output; this is recorded
in ``docs/decisions.md``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .records import TaskItem

PROMPT_VERSION = "p2"
PROMPT_VERSION_NOTES = (
    "2026-10-28 设计更新：实验一/二只评声调（共用无调目标 tone prompt）；新增检测实验"
    "（target+target_tone 一致性）；实验三自由识别；实验四~六 A/B；共用 system prompt"
)

#: shared system prompt (design doc section 五)
SYSTEM_PROMPT = """你是一个普通话单音节发音评估助手。请根据音频和用户给出的信息完成任务，
不要解释，不要输出思考过程，只输出指定格式的最终答案。"""

#: format instructions from the design doc (the schema suffix makes output parseable)
FORMAT_JSON_BASE = "只输出一个 JSON 对象，不要使用 Markdown 代码块，不要添加任何解释。"
FORMAT_TEXT_BASE = "只输出一行，使用指定的固定标记，不要添加解释。"

OUTPUT_SCHEMA: dict[str, dict[str, str]] = {
    "json": {
        "tone": '{"tone": 1|2|3|4}',
        "free": '{"initial": "...", "final": "...", "tone": 1|2|3|4}',
        "detect": '{"consistent": true|false}',
        "ab": '{"choice": "A"|"B"}',
    },
    "text": {
        "tone": "声调=1",
        "free": "声母=zh；韵母=ang；声调=1（零声母写 声母=）",
        "detect": "一致=是 或 一致=否",
        "ab": "选择=A",
    },
}

#: user prompt bodies, per task type (design doc section 五)
TONE_PROMPT = """音频：一段普通话单音节。
目标拼音（无调）：{target}

请给出该音节的声调（1–4）。
{format_instruction}"""

FREE_PROMPT = """音频：一段普通话单音节。
请识别该音节的声母、韵母和声调（1–4）。
零声母用空字符串表示。
{format_instruction}"""

AB_PROMPT = """音频：一段普通话单音节。
候选 A：{candidate_a}
候选 B：{candidate_b}

两个候选只有一个音素或声调不同。请选择与音频实际发音一致的候选。
{format_instruction}"""

DETECT_PROMPT = """音频：一段普通话单音节。
目标拼音：{target}
目标声调：{target_tone}

请判断目标拼音和声调是否与音频中的实际发音一致。
{format_instruction}"""


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def task_kind(task_type: str) -> str:
    """Backwards-compatible alias: output schema name is the parser kind."""
    if task_type not in OUTPUT_SCHEMA["json"]:
        raise ValueError(f"unknown task_type: {task_type!r}")
    return task_type


def format_instruction(task_type: str, output_form: str) -> str:
    if output_form not in ("json", "text"):
        raise ValueError(f"unknown output_form: {output_form!r}")
    if task_type not in OUTPUT_SCHEMA[output_form]:
        raise ValueError(f"unknown task_type: {task_type!r}")
    base = FORMAT_JSON_BASE if output_form == "json" else FORMAT_TEXT_BASE
    return f"{base}输出格式：{OUTPUT_SCHEMA[output_form][task_type]}"


def _tone_marked(target: str | None, tone: int | None) -> str:
    if target is None:
        raise ValueError("task is missing given_target")
    return f"{target}{int(tone)}" if tone else target


def build_prompt_text(task: TaskItem, output_form: str) -> str:
    if not task.task_type:
        raise ValueError(f"task {task.item_id} has no task_type; rebuild task catalogs")
    kind = task.task_type
    fmt = format_instruction(kind, output_form)
    if kind == "tone":
        if task.given_target is None:
            raise ValueError(f"tone task {task.item_id} has no given_target")
        return TONE_PROMPT.format(target=task.given_target, format_instruction=fmt)
    if kind == "free":
        return FREE_PROMPT.format(format_instruction=fmt)
    if kind == "ab":
        if not task.alternatives or len(task.alternatives) != 2:
            raise ValueError(f"ab task {task.item_id} needs two alternatives")
        return AB_PROMPT.format(
            candidate_a=task.alternatives[0], candidate_b=task.alternatives[1],
            format_instruction=fmt,
        )
    if kind == "detect":
        if task.given_target is None or task.given_tone is None:
            raise ValueError(f"detect task {task.item_id} needs target and target_tone")
        return DETECT_PROMPT.format(
            target=task.given_target, target_tone=int(task.given_tone),
            format_instruction=fmt,
        )
    raise ValueError(f"unknown task_type: {kind!r}")


def build_prompt_texts(task: TaskItem) -> dict[str, str]:
    return {form: build_prompt_text(task, form) for form in ("json", "text")}


def claim_text(task: TaskItem) -> str:
    """Human-readable claim (target / target+tone) for reporting/debugging."""
    if task.given_target is None:
        raise ValueError(f"task {task.item_id} has no given_target")
    if task.given_tone is None:
        return task.given_target
    return f"{task.given_target}{int(task.given_tone)}"


# ---------------------------------------------------------------------------
# Request fingerprint (dedupe key)
# ---------------------------------------------------------------------------


def request_id_for(
    *,
    media_relpath: str,
    output_form: str,
    prompt_text: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Stable 16-hex fingerprint of one model call.

    Two task items that resolve to the same audio variant, output form, system
    prompt and user prompt share one API call (e.g. exp3 ``free`` and exp4
    ``free`` on the same audio).  Thinking / model params are part of the run,
    not the request id.
    """
    payload = f"{media_relpath}\x00{output_form}\x00{system_prompt}\x00{prompt_text}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def prompt_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def describe_prompts() -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "notes": PROMPT_VERSION_NOTES,
        "system_prompt": SYSTEM_PROMPT,
        "format_json": FORMAT_JSON_BASE,
        "format_text": FORMAT_TEXT_BASE,
        "templates": {
            "tone": TONE_PROMPT,
            "free": FREE_PROMPT,
            "ab": AB_PROMPT,
            "detect": DETECT_PROMPT,
        },
    }


__all__ = [
    "AB_PROMPT",
    "DETECT_PROMPT",
    "FORMAT_JSON_BASE",
    "FORMAT_TEXT_BASE",
    "FREE_PROMPT",
    "OUTPUT_SCHEMA",
    "PROMPT_VERSION",
    "PROMPT_VERSION_NOTES",
    "SYSTEM_PROMPT",
    "TONE_PROMPT",
    "build_prompt_text",
    "build_prompt_texts",
    "claim_text",
    "describe_prompts",
    "format_instruction",
    "prompt_sha",
    "request_id_for",
    "task_kind",
]
