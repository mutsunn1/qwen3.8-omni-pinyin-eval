"""Tests for v3 prompt templates (experiments 4-6 sub-experiments, no A/B)."""

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


def test_system_prompt_matches_design_doc():
    assert SYSTEM_PROMPT.startswith("你是一个普通话单音节发音评估助手。")
    assert "不要输出思考过程" in SYSTEM_PROMPT


def test_format_instructions_per_task():
    assert '{"tone": 1|2|3|4}' in format_instruction("tone", "json")
    assert '{"initial": "..."}' in format_instruction("initial", "json")
    assert '{"final": "..."}' in format_instruction("final", "json")
    assert '"consistent"' in format_instruction("detect", "json")
    assert "声母=zh" in format_instruction("initial", "text")
    assert "韵母=ang" in format_instruction("final", "text")


def test_tone_prompt_shared_by_exp1_and_exp2():
    correct = make_task("tone", given_target="zhang")
    wrong = make_task(
        "tone", item_id="exp2|wrong_target|audio1|initial0", experiment="exp2",
        prompt_condition="wrong_target", given_target="zang", consistent_gold=False,
    )
    text = build_prompt_text(correct, "json")
    assert "目标拼音（无调）：zhang" in text
    assert "请给出该音节的声调（1–4）" in text
    wrong_text = build_prompt_text(wrong, "json")
    assert "zang" in wrong_text and "错误" not in wrong_text
    assert "一致" not in wrong_text


def test_free_recognition_prompt():
    text = build_prompt_text(make_task("free"), "json")
    assert "请识别该音节的声母、韵母和声调" in text
    assert '零声母时写空字符串 ""' in text
    assert "ü 直接写作 ü" in text  # pilot: model tends to write v
    assert "zhang" not in text


@pytest.mark.parametrize(
    ("experiment", "task_type", "dimension_word"),
    [("exp4", "initial", "声母"), ("exp5", "final", "韵母"), ("exp6", "tone", "声调")],
)
def test_pair_free_sub_experiment(experiment, task_type, dimension_word):
    task = make_task(task_type, experiment=experiment, prompt_condition="free")
    text = build_prompt_text(task, "json")
    assert f"请给出该音节的{dimension_word}" in text
    assert "目标声母" not in text  # no target information


@pytest.mark.parametrize(
    ("experiment", "task_type", "wrong_field"),
    [
        ("exp4", "initial", "initial"),
        ("exp5", "final", "final"),
        ("exp6", "tone", "tone"),
    ],
)
def test_pair_info_sub_experiments(experiment, task_type, wrong_field):
    # correct information: all three target fields given
    correct = make_task(
        task_type, experiment=experiment, prompt_condition="correct_target",
        given_initial="zh", given_final="ang", given_tone=1, consistent_gold=True,
    )
    text = build_prompt_text(correct, "json")
    assert "目标声母：zh" in text
    assert "目标韵母：ang" in text
    assert "目标声调：1" in text
    # wrong information: exactly the experiment's dimension changes
    wrong = make_task(
        task_type, item_id=f"{experiment}|wrong_target|audio1", experiment=experiment,
        prompt_condition="wrong_target",
        given_initial="z" if wrong_field == "initial" else "zh",
        given_final="an" if wrong_field == "final" else "ang",
        given_tone=2 if wrong_field == "tone" else 1,
        consistent_gold=False,
    )
    wtext = build_prompt_text(wrong, "json")
    assert ("目标声母：z\n" if wrong_field == "initial" else "目标声母：zh\n") in wtext
    assert ("目标韵母：an\n" if wrong_field == "final" else "目标韵母：ang\n") in wtext
    assert ("目标声调：2" if wrong_field == "tone" else "目标声调：1") in wtext


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


def test_no_ab_task_type():
    with pytest.raises(ValueError):
        make_task("ab")


def test_prompt_texts_deterministic_and_shareable():
    a = make_task("free", item_id="exp3|free|audio1", experiment="exp3")
    b = make_task("free", item_id="exp4|free|audio1|pairX", experiment="exp4")
    assert build_prompt_texts(a) == build_prompt_texts(b)


def test_request_id_includes_system_prompt():
    task = make_task("free")
    rid = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt=SYSTEM_PROMPT,
    )
    rid_other = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt="另一个 system prompt",
    )
    assert rid != rid_other
    rid_noise = request_id_for(
        media_relpath="data/noisy/audio1__snr20db.wav", output_form="json",
        prompt_text=task.prompt_texts["json"], system_prompt=SYSTEM_PROMPT,
    )
    assert rid_noise != rid
