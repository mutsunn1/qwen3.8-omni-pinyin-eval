"""Tests for prompt templates and request fingerprinting."""

from __future__ import annotations

import pytest

from pinyin_eval.prompts import (
    PROMPT_VERSION,
    build_prompt_text,
    build_prompt_texts,
    claim_text,
    request_id_for,
    task_kind,
)
from pinyin_eval.records import TaskItem


def make_task(prompt_condition: str, **kwargs) -> TaskItem:
    base = dict(
        item_id=f"exp1|{prompt_condition}|audio1",
        experiment="exp1",
        prompt_condition=prompt_condition,
        audio_id="audio1",
        wav_path="data/audio/A/audio1.wav",
        voice="A",
        gold_initial="zh",
        gold_final="ang",
        gold_tone=1,
        cluster_id="audio1",
        stratum="A|1",
    )
    base.update(kwargs)
    task = TaskItem(**base)
    task.prompt_texts = build_prompt_texts(task)
    task.prompt_version = PROMPT_VERSION
    return task


def test_task_kind_mapping():
    assert task_kind("free") == "free"
    assert task_kind("correct_target") == "consistency"
    assert task_kind("wrong_target") == "consistency"
    assert task_kind("ab_choice") == "ab_choice"
    with pytest.raises(ValueError):
        task_kind("nope")


def test_free_prompt_has_no_target_and_no_markers_for_other_tasks():
    task = make_task("free")
    json_text = build_prompt_text(task, "json")
    text_text = build_prompt_text(task, "text")
    assert '"initial"' in json_text and '"final"' in json_text and '"tone"' in json_text
    assert "声母=" in text_text and "韵母=" in text_text and "声调=" in text_text
    assert "zhang" not in json_text + text_text  # no target leakage
    assert "一致" not in text_text


def test_consistency_toneless_claim():
    task = make_task("correct_target", given_target="zhang", consistent_gold=True)
    assert claim_text(task) == "zhang"
    for form in ("json", "text"):
        text = build_prompt_text(task, form)
        assert "zhang" in text
        assert "无调拼音" in text  # states tone is not part of the claim
    assert '"consistent"' in build_prompt_text(task, "json")
    assert "一致性=" in build_prompt_text(task, "text")


def test_consistency_tone_claim():
    task = make_task(
        "wrong_target", given_target="zhang", given_tone=4, consistent_gold=False,
    )
    assert claim_text(task) == "zhang4"
    assert "zhang4" in build_prompt_text(task, "json")
    assert "数字表示声称的声调" in build_prompt_text(task, "json")


def test_ab_prompt_lists_candidates():
    task = make_task(
        "ab_choice", alternatives=["zhang1", "zang1"], correct_choice="A",
    )
    json_text = build_prompt_text(task, "json")
    text_text = build_prompt_text(task, "text")
    assert "A. zhang1" in json_text and "B. zang1" in json_text
    assert '"choice"' in json_text
    assert "选择=" in text_text


def test_prompt_texts_are_deterministic():
    task = make_task("free")
    assert build_prompt_texts(task) == build_prompt_texts(task)
    other = make_task("free", item_id="exp3|free|audio1", experiment="exp3")
    # prompt depends only on the claim fields, not on item ids
    assert build_prompt_texts(task)["json"] == build_prompt_texts(other)["json"]


def test_request_id_is_stable_and_dedupes_identical_prompts():
    task = make_task("free")
    rid_a = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=task.prompt_texts["json"],
    )
    rid_b = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="json",
        prompt_text=build_prompt_texts(make_task("free", item_id="exp4|free|audio1"))["json"],
    )
    assert rid_a == rid_b  # same audio + same prompt => one API call
    rid_noise = request_id_for(
        media_relpath="data/noisy/audio1__snr20db.wav", output_form="json",
        prompt_text=task.prompt_texts["json"],
    )
    assert rid_noise != rid_a
    rid_text = request_id_for(
        media_relpath="data/audio/A/audio1.wav", output_form="text",
        prompt_text=task.prompt_texts["text"],
    )
    assert rid_text != rid_a


def test_prompt_version_present():
    assert PROMPT_VERSION
