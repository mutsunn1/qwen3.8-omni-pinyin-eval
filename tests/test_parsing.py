"""Tests for model-output parsing / normalisation."""

from __future__ import annotations

import pytest

from pinyin_eval.normalize import load_mapping
from pinyin_eval.parsing import (
    OutputNormalizer,
    extract_json_object,
    parse_prediction,
    parse_response,
)

MAPPING = load_mapping("data/mapping/mandarin_tone_syn_target_mapping.csv")
NORMALIZER = OutputNormalizer(MAPPING)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('好的：{"a": 1} 完毕') == {"a": 1}
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None


# ---------------------------------------------------------------------------
# free recognition
# ---------------------------------------------------------------------------


def test_free_json_ok():
    parsed = parse_prediction(
        '{"initial": "zh", "final": "ang", "tone": 3}',
        task_kind="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("zh", "ang", 3)


def test_free_json_normalises_variants():
    parsed = parse_prediction(
        '{"initial": "y", "final": "in", "tone": "一"}',
        task_kind="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("", "in", 1)


def test_free_json_jqx_u_becomes_ü():
    parsed = parse_prediction(
        '{"initial": "q", "final": "uan", "tone": 2}',
        task_kind="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final) == ("q", "üan")


def test_free_json_accepts_chinese_aliases():
    parsed = parse_prediction(
        '{"声母": "b", "韵母": "a", "声调": 4}',
        task_kind="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("b", "a", 4)


def test_free_json_missing_tone_is_parse_error():
    parsed = parse_prediction(
        '{"initial": "b", "final": "a"}',
        task_kind="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.status == "parse_error"


def test_free_text_ok():
    text = "声母=zh\n韵母=ang\n声调=3"
    parsed = parse_prediction(text, task_kind="free", output_form="text", normalizer=NORMALIZER)
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("zh", "ang", 3)


def test_free_text_zero_initial_marker():
    parsed = parse_prediction(
        "声母=无\n韵母=ai\n声调=2", task_kind="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("", "ai", 2)


def test_free_text_tone_mark():
    parsed = parse_prediction(
        "声母=l\n韵母=ǜ\n声调=四", task_kind="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("l", "ü", 4)


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------


def test_consistency_json_ok():
    parsed = parse_prediction(
        '{"consistent": false, "tone": 2}',
        task_kind="consistency", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.consistent is False
    assert parsed.tone == 2


def test_consistency_json_bool_strings():
    parsed = parse_prediction(
        '{"consistent": "不一致", "tone": "2"}',
        task_kind="consistency", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.consistent is False


def test_consistency_text_negative_marker_precedes_positive():
    parsed = parse_prediction(
        "一致性=不一致\n声调=3", task_kind="consistency", output_form="text",
        normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.consistent is False
    assert parsed.tone == 3


def test_consistency_text_positive_marker():
    parsed = parse_prediction(
        "一致性=一致\n声调=1", task_kind="consistency", output_form="text",
        normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.consistent is True


def test_consistency_missing_tone_is_parse_error():
    parsed = parse_prediction(
        "一致性=一致", task_kind="consistency", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.status == "parse_error"
    assert parsed.consistent is True


# ---------------------------------------------------------------------------
# A/B
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ['{"choice": "A"}', '{"choice": "a"}', '{"选择": "B"}'])
def test_ab_json(text):
    parsed = parse_prediction(text, task_kind="ab_choice", output_form="json", normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.choice in ("A", "B")


def test_ab_text():
    parsed = parse_prediction("选择=B", task_kind="ab_choice", output_form="text",
                              normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.choice == "B"


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_empty_response():
    parsed = parse_prediction("   ", task_kind="free", output_form="json", normalizer=NORMALIZER)
    assert parsed.status == "empty"


def test_refusal_json_mode():
    parsed = parse_prediction(
        "抱歉，我无法判断这段音频。", task_kind="consistency", output_form="json",
        normalizer=NORMALIZER,
    )
    assert parsed.status == "refused"


def test_refusal_text_mode():
    parsed = parse_prediction(
        "抱歉，我无法判断这段音频。", task_kind="consistency", output_form="text",
        normalizer=NORMALIZER,
    )
    assert parsed.status == "refused"


def test_garbage_markers_parse_error():
    parsed = parse_prediction(
        "我认为这个发音不太好", task_kind="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.status == "parse_error"


def test_parse_response_dispatch():
    parsed = parse_response(
        "声母=zh\n韵母=ang\n声调=1",
        prompt_condition="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.tone == 1
