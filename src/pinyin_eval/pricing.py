"""Token accounting & CNY cost estimation for Qwen3.8-Omni-Flash.

Pricing snapshot (Alibaba Cloud Model Studio, 2026-10-28):
https://help.aliyun.com/zh/model-studio/model-pricing

* ``qwen3.8-omni-flash``, 华北2（北京）/中国内地:
  input 0.8 元 / 1M tokens, cache-hit input 0.1 元 / 1M tokens,
  output (incl. reasoning tokens) 2.7 元 / 1M tokens;
* audio input tokens: ``ceil(duration_s) * 7`` (audios shorter than 1 s count as 1 s),
  see https://help.aliyun.com/zh/model-studio/qwen-omni (计费与限流).

The real cost is always computed from the API ``usage`` returned per request.
``estimate_*`` helpers are only for pre-run budgeting: they convert prompt
characters to tokens with a configurable (conservative by default) ratio.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .config import Config

DEFAULT_PRICING: dict[str, Any] = {
    "currency": "CNY",
    "input_per_mtok": 0.8,
    "cache_hit_input_per_mtok": 0.1,
    "output_per_mtok": 2.7,
    "audio_tokens_per_second": 7,
    "audio_min_seconds": 1,
    "prompt_tokens_per_char_conservative": 1.0,
    "prompt_tokens_per_char_expected": 0.7,
    "assumed_output_tokens_off": 30,
    "assumed_output_tokens_on": 300,
    "assumed_output_tokens_on_high": 800,
}


@dataclass(frozen=True, slots=True)
class Pricing:
    currency: str = "CNY"
    input_per_mtok: float = 0.8
    cache_hit_input_per_mtok: float = 0.1
    output_per_mtok: float = 2.7
    audio_tokens_per_second: float = 7.0
    audio_min_seconds: float = 1.0
    prompt_tokens_per_char_conservative: float = 1.0
    prompt_tokens_per_char_expected: float = 0.7
    assumed_output_tokens_off: int = 30
    assumed_output_tokens_on: int = 300
    assumed_output_tokens_on_high: int = 800

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config | None = None) -> Pricing:
        if cfg is None:
            return cls()
        values = dict(DEFAULT_PRICING)
        for key in values:
            value = cfg.get(f"pricing.{key}", values[key])
            if value is not None:
                values[key] = value
        return cls(
            currency=str(values["currency"]),
            input_per_mtok=float(values["input_per_mtok"]),
            cache_hit_input_per_mtok=float(values["cache_hit_input_per_mtok"]),
            output_per_mtok=float(values["output_per_mtok"]),
            audio_tokens_per_second=float(values["audio_tokens_per_second"]),
            audio_min_seconds=float(values["audio_min_seconds"]),
            prompt_tokens_per_char_conservative=float(
                values["prompt_tokens_per_char_conservative"]
            ),
            prompt_tokens_per_char_expected=float(values["prompt_tokens_per_char_expected"]),
            assumed_output_tokens_off=int(values["assumed_output_tokens_off"]),
            assumed_output_tokens_on=int(values["assumed_output_tokens_on"]),
            assumed_output_tokens_on_high=int(values["assumed_output_tokens_on_high"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- cost formulas ----------------------------------------------------
    def audio_tokens(self, duration_s: float | None) -> int:
        # official rule: total tokens = ceil(duration seconds) * rate,
        # audios shorter than the minimum count as the minimum
        seconds = max(float(self.audio_min_seconds), float(duration_s or 0.0))
        return int(math.ceil(seconds) * self.audio_tokens_per_second)

    def input_cost(self, prompt_tokens: int, cached_tokens: int = 0) -> float:
        cached = max(0, min(int(cached_tokens), int(prompt_tokens)))
        uncached = int(prompt_tokens) - cached
        return (
            uncached * self.input_per_mtok + cached * self.cache_hit_input_per_mtok
        ) / 1_000_000.0

    def output_cost(self, completion_tokens: int) -> float:
        return int(completion_tokens) * self.output_per_mtok / 1_000_000.0

    def call_cost(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> dict[str, float]:
        input_cost = self.input_cost(prompt_tokens, cached_tokens)
        output_cost = self.output_cost(completion_tokens)
        return {
            "input_tokens": int(prompt_tokens),
            "cached_tokens": min(int(cached_tokens), int(prompt_tokens)),
            "completion_tokens": int(completion_tokens),
            "input_cost_cny": round(input_cost, 8),
            "output_cost_cny": round(output_cost, 8),
            "total_cost_cny": round(input_cost + output_cost, 8),
        }

    # -- prompt-token estimation (pre-run budgeting only) -----------------
    def estimate_prompt_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        duration_s: float | None,
        *,
        mode: str = "conservative",
    ) -> int:
        factor = (
            self.prompt_tokens_per_char_conservative
            if mode == "conservative"
            else self.prompt_tokens_per_char_expected
        )
        chars = len(system_prompt or "") + len(user_prompt or "")
        # +4 chat-template overhead (role markers / separators)
        return int(math.ceil(chars * factor)) + self.audio_tokens(duration_s) + 4

    def assumed_output_tokens(self, thinking: bool, *, high: bool = False) -> int:
        if not thinking:
            return self.assumed_output_tokens_off
        return self.assumed_output_tokens_on_high if high else self.assumed_output_tokens_on


# ---------------------------------------------------------------------------
# Actual usage extraction / aggregation
# ---------------------------------------------------------------------------


def _get(mapping: Any, *keys: str, default: Any = None) -> Any:
    node: Any = mapping
    for key in keys:
        if node is None:
            return default
        if isinstance(node, Mapping):
            node = node.get(key)
        else:
            node = getattr(node, key, None)
    return default if node is None else node


def extract_usage(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Normalise an OpenAI-compatible ``usage`` object to token counters."""
    if not usage:
        return {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
    prompt_tokens = int(_get(usage, "prompt_tokens", default=0) or 0)
    completion_tokens = int(_get(usage, "completion_tokens", default=0) or 0)
    cached_tokens = int(
        _get(usage, "prompt_tokens_details", "cached_tokens", default=0) or 0
    )
    reasoning_tokens = int(
        _get(usage, "completion_tokens_details", "reasoning_tokens", default=0) or 0
    )
    total_tokens = int(
        _get(usage, "total_tokens", default=prompt_tokens + completion_tokens) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": min(cached_tokens, prompt_tokens),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _empty_totals() -> dict[str, Any]:
    return {
        "n_calls": 0,
        "n_errors": 0,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "input_cost_cny": 0.0,
        "output_cost_cny": 0.0,
        "cost_cny": 0.0,
        "latency_s": 0.0,
    }


def _add_usage(totals: dict[str, Any], usage: dict[str, int], latency_s: float,
               pricing: Pricing, *, error: bool) -> None:
    totals["n_calls"] += 1
    if error:
        totals["n_errors"] += 1
    totals["prompt_tokens"] += usage["prompt_tokens"]
    totals["cached_tokens"] += usage["cached_tokens"]
    totals["completion_tokens"] += usage["completion_tokens"]
    totals["reasoning_tokens"] += usage["reasoning_tokens"]
    input_cost = pricing.input_cost(usage["prompt_tokens"], usage["cached_tokens"])
    output_cost = pricing.output_cost(usage["completion_tokens"])
    totals["input_cost_cny"] += input_cost
    totals["output_cost_cny"] += output_cost
    totals["cost_cny"] += input_cost + output_cost
    totals["latency_s"] += float(latency_s or 0.0)


def _finalise_totals(totals: dict[str, Any]) -> dict[str, Any]:
    n = totals["n_calls"]
    out = {
        "n_calls": n,
        "n_errors": totals["n_errors"],
        "prompt_tokens": totals["prompt_tokens"],
        "cached_tokens": totals["cached_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "reasoning_tokens": totals["reasoning_tokens"],
        "input_cost_cny": round(totals["input_cost_cny"], 6),
        "output_cost_cny": round(totals["output_cost_cny"], 6),
        "cost_cny": round(totals["cost_cny"], 6),
        "avg_latency_s": round(totals["latency_s"] / n, 3) if n else None,
    }
    return out


def _record_fields(record: Any) -> tuple[Any, str, str, bool]:
    if isinstance(record, Mapping):
        usage = record.get("usage")
        condition_id = str(record.get("condition_id") or "")
        error = bool(record.get("error"))
        latency = float(record.get("latency_s") or 0.0)
    else:  # GenerationRecord dataclass
        usage = getattr(record, "usage", None)
        condition_id = str(getattr(record, "condition_id", ""))
        error = bool(getattr(record, "error", None))
        latency = float(getattr(record, "latency_s", 0.0) or 0.0)
    return usage, condition_id, latency, error  # type: ignore[return-value]


def summarize_generations(
    generations: Iterable[Any],
    pricing: Pricing | None = None,
) -> dict[str, Any]:
    """Sum actual tokens/cost per condition from generation records.

    Accepts ``GenerationRecord`` objects or their JSON dicts.  Records without a
    ``usage`` object are still counted (as calls) but contribute 0 tokens, so old
    or mock runs degrade gracefully.
    """
    pricing = pricing or Pricing()
    overall = _empty_totals()
    by_condition: dict[str, dict[str, Any]] = {}
    by_generation: dict[str, dict[str, Any]] = {}
    for record in generations:
        usage_raw, condition_id, latency, error = _record_fields(record)
        usage = extract_usage(usage_raw)
        _add_usage(overall, usage, latency, pricing, error=error)
        cond = by_condition.setdefault(condition_id, _empty_totals())
        _add_usage(cond, usage, latency, pricing, error=error)
        generation = condition_id.split("__", 1)[1] if "__" in condition_id else condition_id
        gen = by_generation.setdefault(generation, _empty_totals())
        _add_usage(gen, usage, latency, pricing, error=error)
    return {
        "pricing": pricing.as_dict(),
        "overall": _finalise_totals(overall),
        "by_condition": {k: _finalise_totals(v) for k, v in sorted(by_condition.items())},
        "by_generation": {k: _finalise_totals(v) for k, v in sorted(by_generation.items())},
    }


def budget_status(cost_cny: float, budget_cny: float | None) -> dict[str, Any]:
    if budget_cny is None or budget_cny <= 0:
        return {
            "budget_cny": None,
            "cost_cny": round(float(cost_cny), 6),
            "within_budget": None,
            "used_fraction": None,
        }
    cost = float(cost_cny)
    return {
        "budget_cny": float(budget_cny),
        "cost_cny": round(cost, 6),
        "within_budget": bool(cost <= budget_cny),
        "used_fraction": round(cost / budget_cny, 4),
    }


__all__ = [
    "DEFAULT_PRICING",
    "Pricing",
    "budget_status",
    "extract_usage",
    "summarize_generations",
]
