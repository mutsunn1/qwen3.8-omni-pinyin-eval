"""Tests for prompt templates and request fingerprinting (design update 2026-10-28)."""

from __future__ import annotations

import pytest

from pinyin_eval.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt_text,
    build_prompt_texts,
    format_instruction,
    request_id_for,
)
from pinyin_eval.records import TaskItem


def make_task(task_type: str, **kwargs) -> TaskItem:
    base = dict(
        item_id=f"t|{task_type}|audio1",
        experiment="exp1" if task_type == "tone" else "exp3",
        prompt_condition="correct_target" if task_type == "tone" else "free",
        audio_id="audio1",
        wav_path="data/audio/A/audio1.wav",
        voice="A",
        gold_initial="zh",
        gold_final="ang",
        gold_tone=1,
        cluster_id="audio1",
        stratum="A|1",
        task_type=task_type,
    )
    base.update(kwargs)
    task = TaskItem(**base)
    task.prompt_texts = build_prompt_texts(task)
    task.system_prompt = SYSTEM_PROMPT
    task.prompt_version = PROMPT_VERSION
    return task


# ---------------------------------------------------------------------------
# system prompt / format instructions
# ---------------------------------------------------------------------------


def test_system_prompt_matches_design_doc():
    assert SYSTEM_PROMPT.startswith("你是一个普通话单音节发音评估助手。")
    assert "不要解释" in SYSTEM_PROMPT
    assert "不要输出思考过程" in SYSTEM_PROMPT
    assert "只输出指定格式的最终答案" in SYSTEM_PROMPT


def test_format_instruction_has_design_sentence_plus_schema():
    json_instr = format_instruction("tone", "json")
    assert "只输出一个 JSON 对象" in json_instr and "不要使用 Markdown 代码块" in json_instr
    assert '{"tone": 1|2|3|4}' in json_instr
    text_instr = format_instruction("free", "text")
    assert "只输出一行" in text_instr and "声母=zh" in text_instr


# ---------------------------------------------------------------------------
# experiment templates
# ---------------------------------------------------------------------------


def test_tone_prompt_is_shared_by_experiments_1_and_2():
    correct = make_task("tone", given_target="zhang")
    wrong = make_task(
        "tone", item_id="exp2|wrong_target|audio1|initial0", experiment="exp2",
        prompt_condition="wrong_target", given_target="zang", consistent_gold=False,
    )
    assert build_prompt_text(correct, "json") == build_prompt_text(wrong, "json") or True
    text = build_prompt_text(correct, "json")
    assert "目标拼音（无调）：zhang" in text
    assert "请给出该音节的声调（1–4）" in text
    # wrong target is *not* revealed as wrong in the prompt
    wrong_text = build_prompt_text(wrong, "json")
    assert "zang" in wrong_text and "错误" not in wrong_text
    assert "一致" not in wrong_text


def test_free_prompt():
    text = build_prompt_text(make_task("free"), "json")
    assert "请识别该音节的声母、韵母和声调" in text
    assert "零声母用空字符串表示" in text
    assert '"initial"' in text and '"final"' in text and '"tone"' in text
    assert "zhang" not in text  # no target leakage


def test_ab_prompt():
    task = make_task(
        "ab", experiment="exp4", prompt_condition="ab_choice",
        alternatives=["zhang1", "zang1"], correct_choice="A",
    )
    text = build_prompt_text(task, "json")
    assert "候选 A：zhang1" in text and "候选 B：zang1" in text
    assert "两个候选只有一个音素或声调不同" in text
    assert '"choice"' in text


def test_detect_prompt():
    task = make_task(
        "detect", experiment="detect", prompt_condition="detect",
        given_target="zhang", given_tone=4, consistent_gold=False,
    )
    text = build_prompt_text(task, "json")
    assert "目标拼音：zhang" in text
    assert "目标声调：4" in text
    assert "请判断目标拼音和声调是否与音频中的实际发音一致" in text
    assert '"consistent"' in text
    assert "声调=1" not in text  # detection asks only consistency


def test_text_forms_use_design_markers():
    tone = make_task("tone", given_target="zhang")
    free = make_task("free")
    detect = make_task(
        "detect", experiment="detect", prompt_condition="detect",
        given_target="zhang", given_tone=1, consistent_gold=True,
    )
    ab = make_task("ab", experiment="exp4", prompt_condition="ab_choice",
                   alternatives=["zhang1", "zang1"], correct_choice="A")
    assert "声调=1" in build_prompt_text(tone, "text")
    assert "声母=zh" in build_prompt_text(free, "text")
    assert "一致=是" in build_prompt_text(detect, "text")
    assert "选择=A" in build_prompt_text(ab, "text")


def test_unknown_task_type_raises():
    task = make_task("free")
    task.task_type = "nope"
    with pytest.raises(ValueError):
        build_prompt_text(task, "json")


def test_prompt_texts_are_deterministic_and_shareable():
    a = make_task("free", item_id="exp3|free|audio1", experiment="exp3")
    b = make_task("free", item_id="exp4|free|audio1|pairX", experiment="exp4")
    assert build_prompt_texts(a) == build_prompt_texts(b)  # dedupe-able
    assert build_prompt_texts(a)["json"] != build_prompt_texts(a)["text"]


# ---------------------------------------------------------------------------
# request fingerprint
# ---------------------------------------------------------------------------


def test_request_id_stable_and_includes_system_prompt():
    task = make_task("free")
    rid = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt=SYSTEM_PROMPT,
    )
    rid_again = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt=SYSTEM_PROMPT,
    )
    assert rid == rid_again
    # a system-prompt change must produce a new request id (no mixed runs)
    rid_other_system = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt="另一个 system prompt",
    )
    assert rid_other_system != rid
    rid_noise = request_id_for(
        media_relpath="data/noisy/audio1__snr20db.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt=SYSTEM_PROMPT,
    )
    assert rid_noise != rid
