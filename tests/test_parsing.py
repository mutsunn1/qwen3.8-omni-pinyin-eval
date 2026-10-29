"""Tests for model-output parsing / normalisation."""

from __future__ import annotations

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
        task_type="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("zh", "ang", 3)


def test_free_json_normalises_variants():
    parsed = parse_prediction(
        '{"initial": "y", "final": "in", "tone": "一"}',
        task_type="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("", "in", 1)


def test_free_json_jqx_u_becomes_ü():
    parsed = parse_prediction(
        '{"initial": "q", "final": "uan", "tone": 2}',
        task_type="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final) == ("q", "üan")


def test_free_json_accepts_chinese_aliases():
    parsed = parse_prediction(
        '{"声母": "b", "韵母": "a", "声调": 4}',
        task_type="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("b", "a", 4)


def test_free_json_missing_tone_is_parse_error():
    parsed = parse_prediction(
        '{"initial": "b", "final": "a"}',
        task_type="free", output_form="json", normalizer=NORMALIZER,
    )
    assert parsed.status == "parse_error"


def test_free_text_ok():
    text = "声母=zh\n韵母=ang\n声调=3"
    parsed = parse_prediction(text, task_type="free", output_form="text", normalizer=NORMALIZER)
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("zh", "ang", 3)


def test_free_text_zero_initial_marker():
    parsed = parse_prediction(
        "声母=无\n韵母=ai\n声调=2", task_type="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("", "ai", 2)


def test_free_text_tone_mark():
    parsed = parse_prediction(
        "声母=l\n韵母=ǜ\n声调=四", task_type="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert (parsed.initial, parsed.final, parsed.tone) == ("l", "ü", 4)


# ---------------------------------------------------------------------------
# tone-only tasks (experiments 1/2)
# ---------------------------------------------------------------------------


def test_tone_json_ok():
    parsed = parse_prediction('{"tone": 3}', task_type="tone", output_form="json",
                              normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.tone == 3


def test_tone_json_chinese_number():
    parsed = parse_prediction('{"tone": "二"}', task_type="tone", output_form="json",
                              normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.tone == 2


def test_tone_text_ok():
    parsed = parse_prediction("声调=4", task_type="tone", output_form="text",
                              normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.tone == 4


def test_tone_missing_is_parse_error():
    parsed = parse_prediction("我认为是第二声", task_type="tone", output_form="text",
                              normalizer=NORMALIZER)
    assert parsed.status == "parse_error"


# ---------------------------------------------------------------------------
# detection experiment (consistency only)
# ---------------------------------------------------------------------------


def test_detect_json_ok():
    parsed = parse_prediction('{"consistent": false}', task_type="detect",
                              output_form="json", normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.consistent is False


def test_detect_json_bool_strings():
    parsed = parse_prediction('{"consistent": "不一致"}', task_type="detect",
                              output_form="json", normalizer=NORMALIZER)
    assert parsed.ok
    assert parsed.consistent is False


def test_detect_text_yes_no_markers():
    assert parse_prediction("一致=是", task_type="detect", output_form="text",
                            normalizer=NORMALIZER).consistent is True
    assert parse_prediction("一致=否", task_type="detect", output_form="text",
                            normalizer=NORMALIZER).consistent is False


def test_detect_text_tolerates_legacy_marker():
    parsed = parse_prediction("一致性=不一致", task_type="detect", output_form="text",
                              normalizer=NORMALIZER)
    assert parsed.ok and parsed.consistent is False


def test_detect_missing_consistent_is_parse_error():
    parsed = parse_prediction("目标拼音正确", task_type="detect", output_form="text",
                              normalizer=NORMALIZER)
    assert parsed.status == "parse_error"


# ---------------------------------------------------------------------------
# dimension-only tasks (experiments 4/5)
# ---------------------------------------------------------------------------


def test_initial_json_and_text():
    assert parse_prediction('{"initial": "zh"}', task_type="initial", output_form="json",
                            normalizer=NORMALIZER).initial == "zh"
    assert parse_prediction("声母=zh", task_type="initial", output_form="text",
                            normalizer=NORMALIZER).initial == "zh"
    # zero initial: empty marker / y spelling
    assert parse_prediction("声母=", task_type="initial", output_form="text",
                            normalizer=NORMALIZER).initial == ""
    assert parse_prediction('{"initial": "y"}', task_type="initial", output_form="json",
                            normalizer=NORMALIZER).initial == ""


def test_final_json_and_text():
    assert parse_prediction('{"final": "ang"}', task_type="final", output_form="json",
                            normalizer=NORMALIZER).final == "ang"
    assert parse_prediction("韵母=üan", task_type="final", output_form="text",
                            normalizer=NORMALIZER).final == "üan"
    # a full syllable in the final field is split to its final
    assert parse_prediction("韵母=yuan", task_type="final", output_form="text",
                            normalizer=NORMALIZER).final == "üan"


def test_dimension_task_missing_field_is_parse_error():
    assert parse_prediction("{}", task_type="initial", output_form="json",
                            normalizer=NORMALIZER).status == "parse_error"
    assert parse_prediction('{"final": ""}', task_type="final", output_form="json",
                            normalizer=NORMALIZER).final == ""


# ---------------------------------------------------------------------------
# failure modes# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_empty_response():
    parsed = parse_prediction("   ", task_type="free", output_form="json", normalizer=NORMALIZER)
    assert parsed.status == "empty"


def test_refusal_json_mode():
    parsed = parse_prediction(
        "抱歉，我无法判断这段音频。", task_type="detect", output_form="json",
        normalizer=NORMALIZER,
    )
    assert parsed.status == "refused"


def test_refusal_text_mode():
    parsed = parse_prediction(
        "抱歉，我无法判断这段音频。", task_type="detect", output_form="text",
        normalizer=NORMALIZER,
    )
    assert parsed.status == "refused"


def test_garbage_markers_parse_error():
    parsed = parse_prediction(
        "我认为这个发音不太好", task_type="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.status == "parse_error"


def test_parse_response_dispatch():
    parsed = parse_response(
        "声母=zh；韵母=ang；声调=1",
        task_type="free", output_form="text", normalizer=NORMALIZER,
    )
    assert parsed.ok
    assert parsed.tone == 1
