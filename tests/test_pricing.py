"""Tests for token accounting / CNY cost estimation (`pinyin_eval.pricing`)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pinyin_eval.config import Config
from pinyin_eval.pricing import (
    Pricing,
    budget_status,
    extract_usage,
    summarize_generations,
)

from .conftest import CONFIG_PATH


@pytest.fixture()
def pricing() -> Pricing:
    return Pricing.from_config(Config.load(CONFIG_PATH, repo_root="."))


def test_default_pricing_matches_documented_snapshot(pricing):
    assert pricing.currency == "CNY"
    assert pricing.input_per_mtok == 0.8
    assert pricing.cache_hit_input_per_mtok == 0.1
    assert pricing.output_per_mtok == 2.7
    assert pricing.audio_tokens_per_second == 7


@pytest.mark.parametrize(
    ("duration", "expected"),
    [(0.2, 7), (0.0, 7), (0.99, 7), (1.0, 7), (1.01, 14), (1.334, 14)],
)
def test_audio_tokens_rule(pricing, duration, expected):
    # rule: ceil(duration) * 7, audios shorter than 1 s count as 1 s
    assert pricing.audio_tokens(duration) == expected


def test_input_cost_with_cache_discount(pricing):
    # 1M uncached tokens -> 0.8 CNY; 1M cached -> 0.1 CNY
    assert pricing.input_cost(1_000_000) == pytest.approx(0.8)
    assert pricing.input_cost(1_000_000, cached_tokens=1_000_000) == pytest.approx(0.1)
    assert pricing.input_cost(1_000_000, cached_tokens=500_000) == pytest.approx(0.45)


def test_output_cost(pricing):
    assert pricing.output_cost(1_000_000) == pytest.approx(2.7)
    assert pricing.output_cost(256) == pytest.approx(256 * 2.7 / 1e6)


def test_call_cost_breakdown(pricing):
    cost = pricing.call_cost(prompt_tokens=200, completion_tokens=256, cached_tokens=50)
    expected_input = (150 * 0.8 + 50 * 0.1) / 1e6
    expected_output = 256 * 2.7 / 1e6
    assert cost["input_cost_cny"] == pytest.approx(expected_input)
    assert cost["output_cost_cny"] == pytest.approx(expected_output)
    assert cost["total_cost_cny"] == pytest.approx(expected_input + expected_output)


def test_estimate_prompt_tokens_modes(pricing):
    system = "x" * 60
    prompt = "y" * 100
    conservative = pricing.estimate_prompt_tokens(system, prompt, 0.6, mode="conservative")
    expected = pricing.estimate_prompt_tokens(system, prompt, 0.6, mode="expected")
    assert conservative >= expected
    # 160 chars * 1.0 + 7 audio tokens + 4 overhead
    assert conservative == 160 + 7 + 4


def test_assumed_output_tokens(pricing):
    assert pricing.assumed_output_tokens(False) == 30
    assert pricing.assumed_output_tokens(True) == 300
    assert pricing.assumed_output_tokens(True, high=True) == 800


def test_extract_usage_full_and_missing():
    usage = {
        "prompt_tokens": 210,
        "completion_tokens": 260,
        "total_tokens": 470,
        "prompt_tokens_details": {"cached_tokens": 100, "audio_tokens": 7, "text_tokens": 203},
        "completion_tokens_details": {"reasoning_tokens": 200, "text_tokens": 60},
    }
    extracted = extract_usage(usage)
    assert extracted == {
        "prompt_tokens": 210,
        "cached_tokens": 100,
        "completion_tokens": 260,
        "reasoning_tokens": 200,
        "total_tokens": 470,
    }
    assert extract_usage(None)["prompt_tokens"] == 0
    # cached can never exceed prompt tokens
    assert extract_usage({"prompt_tokens": 10,
                          "prompt_tokens_details": {"cached_tokens": 99}})["cached_tokens"] == 10


@dataclass
class FakeRecord:
    condition_id: str
    usage: dict | None
    error: str | None = None
    latency_s: float = 1.0


def test_summarize_generations_cost(pricing):
    records = [
        FakeRecord("free__think0__clean__json", {
            "prompt_tokens": 200, "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 100},
        }),
        FakeRecord("free__think0__clean__json", {
            "prompt_tokens": 300, "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }, latency_s=3.0),
        FakeRecord("detect__think1__clean__json", None, error="api_error"),
    ]
    summary = summarize_generations(records, pricing)
    overall = summary["overall"]
    assert overall["n_calls"] == 3
    assert overall["n_errors"] == 1
    assert overall["prompt_tokens"] == 500
    assert overall["cached_tokens"] == 100
    assert overall["completion_tokens"] == 60
    assert overall["reasoning_tokens"] == 30
    expected_cost = (
        (100 * 0.8 + 100 * 0.1) / 1e6 + 20 * 2.7 / 1e6
        + 300 * 0.8 / 1e6 + 40 * 2.7 / 1e6
    )
    assert overall["cost_cny"] == pytest.approx(expected_cost, abs=1e-6)
    assert overall["avg_latency_s"] == pytest.approx(5.0 / 3.0, abs=1e-3)
    assert set(summary["by_condition"]) == {
        "free__think0__clean__json", "detect__think1__clean__json",
    }


def test_budget_status():
    assert budget_status(10.0, None)["within_budget"] is None
    status = budget_status(10.0, 70.0)
    assert status["within_budget"] is True
    assert status["used_fraction"] == pytest.approx(10.0 / 70.0, abs=1e-4)
    assert budget_status(80.0, 70.0)["within_budget"] is False
