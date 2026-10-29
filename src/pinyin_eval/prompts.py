"""Prompt templates (zero-shot, no few-shot examples).

Source: ``docs/experiment-design.md`` v3 (2026-10-28 20:37 snapshot,
sha256 ``acf2015a…``).  The v3 revision replaced the A/B task with *three
sub-experiments per minimal-pair experiment*:

* **自由检测** (``free``): no target information, output the experiment's
  dimension (exp4 initial / exp5 final / exp6 tone);
* **正确信息** (``correct_target``): give target initial/final/tone (all
  correct), output the dimension;
* **错误信息** (``wrong_target``): give one wrong target field (exp4 initial /
  exp5 final / exp6 tone; the other fields stay correct), output the dimension.

Experiments 1-3 and the detection experiment keep the v2 structure:

* exp1: correct toneless target -> tone only;
* exp2: wrong toneless segmental target -> tone only;
* exp3: free recognition (initial/final/tone);
* detect: target + target_tone -> consistency true/false.

Thinking is an API parameter and audio quality is preprocessing; neither
changes the prompt text.  ``PROMPT_VERSION`` must be bumped on any wording
change (it changes request fingerprints, so old runs never mix with new ones).

Note: the design doc lists the required JSON object / fixed text marker per
task; we append that schema to the doc's generic format sentence as
``{format_instruction}`` so the model reliably emits parseable output
(recorded in ``docs/decisions.md``).
"""

from __future__ import annotations

import hashlib
from typing import Any

from .records import TaskItem

PROMPT_VERSION = "p3"
PROMPT_VERSION_NOTES = (
    "2026-10-28 v3：取消 A/B；实验四~六改为自由检测/正确信息/错误信息三个子实验，"
    "各自输出 initial/final/tone；检测实验不参与消融；实验一/二只评声调；实验三自由识别"
)

#: shared system prompt (design doc section 五)
SYSTEM_PROMPT = """你是一个普通话单音节发音评估助手。请根据音频和用户给出的信息完成任务，
不要解释，不要输出思考过程，只输出指定格式的最终答案。"""

FORMAT_JSON_BASE = "只输出一个 JSON 对象，不要使用 Markdown 代码块，不要添加任何解释。"
FORMAT_TEXT_BASE = "只输出一行，使用指定的固定标记，不要添加解释。"

OUTPUT_SCHEMA: dict[str, dict[str, str]] = {
    "json": {
        "tone": '{"tone": 1|2|3|4}',
        "free": '{"initial": "...", "final": "...", "tone": 1|2|3|4}',
        "initial": '{"initial": "..."}',
        "final": '{"final": "..."}',
        "detect": '{"consistent": true|false}',
    },
    "text": {
        "tone": "声调=1",
        "free": "声母=zh；韵母=ang；声调=1（零声母写 声母=）",
        "initial": "声母=zh",
        "final": "韵母=ang",
        "detect": "一致=是 或 一致=否",
    },
}

#: dimension word used inside the experiments 4-6 prompts
DIMENSION_WORD: dict[str, str] = {
    "initial": "声母",
    "final": "韵母",
    "tone": "声调",
}

TONE_PROMPT = """音频：一段普通话单音节。
目标拼音（无调）：{target}

请给出该音节的声调（1–4）。
{format_instruction}"""

FREE_PROMPT = """音频：一段普通话单音节。
请识别该音节的声母、韵母和声调（1–4）。
零声母用空字符串表示。
{format_instruction}"""

FREE_DIMENSION_PROMPT = """音频：一段普通话单音节。
请给出该音节的{dimension}。
{format_instruction}"""

INFO_PROMPT = """音频：一段普通话单音节。
目标声母：{target_initial}
目标韵母：{target_final}
目标声调：{target_tone}

请根据音频给出该音节的{dimension}。
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


def _dimension_word(task_type: str) -> str:
    return DIMENSION_WORD.get(task_type, "声调")


def _is_pair_info_prompt(task: TaskItem) -> bool:
    """Experiments 4-6 use the target-information prompt for correct/wrong info."""
    return task.experiment in ("exp4", "exp5", "exp6") and task.prompt_condition in (
        "correct_target", "wrong_target",
    )


def build_prompt_text(task: TaskItem, output_form: str) -> str:
    if not task.task_type:
        raise ValueError(f"task {task.item_id} has no task_type; rebuild task catalogs")
    kind = task.task_type
    fmt = format_instruction(kind, output_form)

    if kind == "detect":
        if task.given_target is None or task.given_tone is None:
            raise ValueError(f"detect task {task.item_id} needs target and target_tone")
        return DETECT_PROMPT.format(
            target=task.given_target, target_tone=int(task.given_tone),
            format_instruction=fmt,
        )

    if kind == "free":
        return FREE_PROMPT.format(format_instruction=fmt)

    # tone / initial / final are dimension outputs
    if _is_pair_info_prompt(task):
        if task.given_initial is None or task.given_final is None or task.given_tone is None:
            raise ValueError(
                f"pair info task {task.item_id} needs given_initial/final/tone"
            )
        return INFO_PROMPT.format(
            target_initial=task.given_initial,
            target_final=task.given_final,
            target_tone=int(task.given_tone),
            dimension=_dimension_word(kind),
            format_instruction=fmt,
        )
    if task.experiment in ("exp4", "exp5", "exp6"):
        # free sub-experiment of experiments 4-6
        return FREE_DIMENSION_PROMPT.format(
            dimension=_dimension_word(kind), format_instruction=fmt,
        )
    if kind == "tone":
        if task.given_target is None:
            raise ValueError(f"tone task {task.item_id} has no given_target")
        return TONE_PROMPT.format(target=task.given_target, format_instruction=fmt)
    raise ValueError(f"unexpected task_type {kind!r} for {task.item_id}")


def build_prompt_texts(task: TaskItem) -> dict[str, str]:
    return {form: build_prompt_text(task, form) for form in ("json", "text")}


def claim_text(task: TaskItem) -> str:
    """Human-readable claim (target / target info) for reporting/debugging."""
    if task.given_initial is not None or task.given_final is not None:
        return f"{task.given_initial}/{task.given_final}/{task.given_tone}"
    if task.given_target is None:
        raise ValueError(f"task {task.item_id} has no target information")
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
            "free_dimension": FREE_DIMENSION_PROMPT,
            "info": INFO_PROMPT,
            "detect": DETECT_PROMPT,
        },
    }


__all__ = [
    "DETECT_PROMPT",
    "DIMENSION_WORD",
    "FORMAT_JSON_BASE",
    "FORMAT_TEXT_BASE",
    "FREE_DIMENSION_PROMPT",
    "FREE_PROMPT",
    "INFO_PROMPT",
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
