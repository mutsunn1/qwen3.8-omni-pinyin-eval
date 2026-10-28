"""Model client: Alibaba Bailian (DashScope) OpenAI-compatible API.

Endpoint facts verified against the Qwen-Omni documentation (2026-10-28):

* model id for text output: ``qwen3.8-omni-flash`` (Qwen3.8-Omni-Flash);
* request media item:
  ``{"type": "input_audio", "input_audio": {"data": "data:audio/wav;base64,...",
  "format": "wav"}}``;
* requires ``modalities=["text"]``;
* thinking: Qwen3.8-Omni thinks by default; ``reasoning_effort="none"`` disables it;
* ``response_format={"type": "json_object"}`` is supported for structured output;
* base_url is workspace/region specific and must be configured via
  ``DASHSCOPE_BASE_URL`` (public compatible endpoint is only a fallback).

A deterministic ``MockClient`` (oracle / noisy / random) is provided so the full
pipeline can be exercised end-to-end without an API key.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import openai

from .config import Config

log = logging.getLogger(__name__)


class ClientSetupError(RuntimeError):
    """Raised when API credentials / endpoint are not configured."""


@dataclass(slots=True)
class ChatRequest:
    """One API call (request-id already de-duplicated by the runner)."""

    request_id: str
    audio_path: str
    prompt_text: str
    output_form: str
    thinking: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResponse:
    request_id: str
    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    latency_s: float = 0.0
    error: str | None = None
    retries: int = 0
    timestamp: str = ""
    model: str = ""
    request_log_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def encode_audio_data_uri(path: str | Path) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _usage_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        try:
            return usage.model_dump()
        except Exception:  # noqa: BLE001 - fall through to manual extraction
            pass
    if isinstance(usage, dict):
        return dict(usage)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


# ---------------------------------------------------------------------------
# Base / mock
# ---------------------------------------------------------------------------


class BaseClient:
    model: str = ""

    def generate(self, request: ChatRequest) -> ChatResponse:  # pragma: no cover
        raise NotImplementedError


class MockClient(BaseClient):
    """Deterministic mock used by tests and ``--mock`` runs.

    ``mock.mode``:
    * ``perfect``  — always answers gold labels (oracle);
    * ``noisy``    — flips answers with probability ``mock.error_rate``;
    * ``random``   — random tone / consistency / choice.

    ``mock.fail_rate`` injects API errors (to exercise retry/resume paths).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = f"mock-{cfg.model_name}"
        self.mode = str(cfg.get("mock.mode", "perfect"))
        self.error_rate = float(cfg.get("mock.error_rate", 0.0))
        self.fail_rate = float(cfg.get("mock.fail_rate", 0.0))
        self.latency_s = float(cfg.get("mock.latency_s", 0.0))
        self.rng = random.Random(int(cfg.get("mock.seed", cfg.seed())))

    def generate(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)
        rid = request.request_id
        if self.rng.random() < self.fail_rate:
            return ChatResponse(
                request_id=rid, error="mock transient failure", retries=0,
                timestamp=utc_now(), model=self.model,
                latency_s=time.perf_counter() - started,
            )
        text = self._answer(request)
        return ChatResponse(
            request_id=rid,
            text=text,
            reasoning="mock reasoning" if request.thinking else "",
            finish_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            latency_s=time.perf_counter() - started,
            timestamp=utc_now(),
            model=self.model,
        )

    # -- answer synthesis --------------------------------------------------
    def _answer(self, request: ChatRequest) -> str:
        meta = request.meta
        kind = meta.get("task_kind", "free")
        gold_initial = meta.get("gold_initial", "")
        gold_final = meta.get("gold_final", "")
        gold_tone = int(meta.get("gold_tone", 1) or 1)
        consistent = meta.get("consistent_gold")
        correct_choice = meta.get("correct_choice")

        wrong = self.rng.random() < self.error_rate
        if self.mode == "random":
            gold_tone = self.rng.randint(1, 4)
            consistent = self.rng.random() < 0.5
            correct_choice = self.rng.choice(["A", "B"])
        elif wrong:
            gold_tone = self.rng.choice([t for t in (1, 2, 3, 4) if t != gold_tone])
            if consistent is not None:
                consistent = not consistent
            correct_choice = "B" if correct_choice == "A" else "A"

        if kind == "consistency":
            consistent = bool(consistent)
            if request.output_form == "json":
                return json.dumps({"consistent": consistent, "tone": gold_tone})
            return f"一致性={'一致' if consistent else '不一致'}\n声调={gold_tone}"
        if kind == "free":
            if request.output_form == "json":
                return json.dumps(
                    {"initial": gold_initial, "final": gold_final, "tone": gold_tone},
                    ensure_ascii=False,
                )
            initial = gold_initial if gold_initial else "无"
            return f"声母={initial}\n韵母={gold_final}\n声调={gold_tone}"
        if kind == "ab_choice":
            choice = correct_choice or self.rng.choice(["A", "B"])
            if request.output_form == "json":
                return json.dumps({"choice": choice})
            return f"选择={choice}"
        raise ValueError(f"mock cannot answer unknown task kind {kind!r}")


# ---------------------------------------------------------------------------
# DashScope OpenAI-compatible client
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", 0)
        return status == 429 or status >= 500
    return False


class DashScopeClient(BaseClient):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.model_name
        api_key = os.environ.get(str(cfg.get("model.api_key_env", "DASHSCOPE_API_KEY")), "")
        if not api_key:
            raise ClientSetupError(
                f"环境变量 {cfg.get('model.api_key_env')} 未设置。"
                "请先 export DASHSCOPE_API_KEY=...（阿里云百炼 API Key），"
                "并建议同时设置 DASHSCOPE_BASE_URL 为业务空间兼容地址。"
            )
        base_url_env = str(cfg.get("model.base_url_env", "DASHSCOPE_BASE_URL"))
        base_url = os.environ.get(base_url_env) or str(cfg.get("model.base_url_default"))
        self.base_url = str(base_url).rstrip("/")
        timeout = float(cfg.get("model.request_timeout_s", 120))
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=0,  # retries handled here so they are recorded
        )
        self.temperature = float(cfg.get("model.temperature", 0.0))
        self.top_p = float(cfg.get("model.top_p", 1.0))
        self.max_output_tokens = int(cfg.get("model.max_output_tokens", 256))
        self.max_retries = int(cfg.get("model.max_retries", 6))
        self.backoff = float(cfg.get("model.retry_backoff_s", 2.0))
        self.max_backoff = float(cfg.get("model.retry_max_backoff_s", 90.0))
        self.stream = bool(cfg.get("model.stream", False))
        self.send_enable_thinking = bool(cfg.get("model.send_enable_thinking", False))
        self.thinking_on = cfg.get("model.thinking_on_reasoning_effort")
        self.thinking_off = cfg.get("model.thinking_off_reasoning_effort", "none")
        self.modalities = list(cfg.get("model.modalities", ["text"]))

    # -- request construction ---------------------------------------------
    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        audio_uri = encode_audio_data_uri(request.audio_path)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio",
                         "input_audio": {"data": audio_uri, "format": "wav"}},
                        {"type": "text", "text": request.prompt_text},
                    ],
                }
            ],
            "modalities": self.modalities,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
            "stream": self.stream,
        }
        if self.stream:
            payload["stream_options"] = {"include_usage": True}
        if request.output_form == "json":
            payload["response_format"] = {"type": "json_object"}
        effort = self.thinking_on if request.thinking else self.thinking_off
        if effort:
            payload["reasoning_effort"] = effort
        if self.send_enable_thinking:
            payload["extra_body"] = {"enable_thinking": bool(request.thinking)}
        return payload

    # -- call with retries -------------------------------------------------
    def generate(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        retries = 0
        last_error: str | None = None
        while retries <= self.max_retries:
            try:
                payload = self.build_payload(request)
                if self.stream:
                    response = self._call_streaming(payload)
                else:
                    response = self._call_blocking(payload)
                response.latency_s = time.perf_counter() - started
                response.retries = retries
                response.request_id = request.request_id
                response.model = self.model
                response.timestamp = utc_now()
                return response
            except openai.AuthenticationError as exc:
                last_error = f"authentication failed: {exc}"
                break  # never retry auth errors
            except (openai.BadRequestError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
                last_error = f"request rejected ({type(exc).__name__}): {exc}"
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if not _is_retryable(exc) or retries >= self.max_retries:
                    break
                delay = min(self.max_backoff, self.backoff * (2 ** retries))
                delay += random.uniform(0, delay * 0.25)
                retry_after = _retry_after_header(exc)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                log.warning("request %s failed (%s); retry %d/%d in %.1fs",
                            request.request_id, last_error, retries + 1, self.max_retries, delay)
                time.sleep(delay)
                retries += 1
        return ChatResponse(
            request_id=request.request_id,
            error=last_error or "unknown error",
            retries=retries,
            latency_s=time.perf_counter() - started,
            timestamp=utc_now(),
            model=self.model,
        )

    def _call_blocking(self, payload: dict[str, Any]) -> ChatResponse:
        completion = self.client.chat.completions.create(**payload)
        choice = completion.choices[0] if completion.choices else None
        text = getattr(getattr(choice, "message", None), "content", None) or ""
        reasoning = getattr(getattr(choice, "message", None), "reasoning_content", None) or ""
        return ChatResponse(
            request_id="",
            text=text,
            reasoning=reasoning,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=_usage_dict(getattr(completion, "usage", None)),
            request_log_id=getattr(completion, "id", None),
        )

    def _call_streaming(self, payload: dict[str, Any]) -> ChatResponse:
        stream = self.client.chat.completions.create(**payload)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] | None = None
        finish_reason: str | None = None
        request_log_id: str | None = None
        for chunk in stream:
            request_log_id = request_log_id or getattr(chunk, "id", None)
            if getattr(chunk, "usage", None):
                usage = _usage_dict(chunk.usage)
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        return ChatResponse(
            request_id="",
            text="".join(text_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            usage=usage,
            request_log_id=request_log_id,
        )


def _retry_after_header(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        try:
            value = headers.get(key)
        except Exception:  # noqa: BLE001
            value = None
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Factory / setup checks
# ---------------------------------------------------------------------------


def create_client(cfg: Config, *, mock: bool = False) -> BaseClient:
    if mock:
        return MockClient(cfg)
    return DashScopeClient(cfg)


def check_setup(cfg: Config) -> dict[str, Any]:
    """Report whether live API calls are possible (no network request)."""
    key_env = str(cfg.get("model.api_key_env", "DASHSCOPE_API_KEY"))
    base_env = str(cfg.get("model.base_url_env", "DASHSCOPE_BASE_URL"))
    key_present = bool(os.environ.get(key_env))
    base_url = os.environ.get(base_env) or str(cfg.get("model.base_url_default"))
    return {
        "api_key_env": key_env,
        "api_key_present": key_present,
        "base_url_env": base_env,
        "base_url": base_url,
        "base_url_is_workspace_specific": bool(os.environ.get(base_env)),
        "model": cfg.model_name,
        "mock_available": True,
        "live_calls_possible": key_present,
    }


def smoke_call(cfg: Config, audio_path: str, *, prompt: str | None = None) -> dict[str, Any]:
    """One real API call for connectivity/model verification."""
    client = create_client(cfg)
    if not isinstance(client, DashScopeClient):
        raise ClientSetupError("smoke_call requires a live client")
    prompt_text = prompt or (
        "你是一名普通话语音分析专家。请听这段单音节普通话音频，"
        "只输出一行：声调=<1/2/3/4>"
    )
    request = ChatRequest(
        request_id="smoke",
        audio_path=str(audio_path),
        prompt_text=prompt_text,
        output_form="text",
        thinking=False,
    )
    response = client.generate(request)
    return {
        "ok": response.ok,
        "error": response.error,
        "text": response.text,
        "reasoning": response.reasoning,
        "usage": response.usage,
        "latency_s": response.latency_s,
        "model": response.model,
        "base_url": client.base_url,
    }


__all__ = [
    "BaseClient",
    "ChatRequest",
    "ChatResponse",
    "ClientSetupError",
    "DashScopeClient",
    "MockClient",
    "check_setup",
    "create_client",
    "encode_audio_data_uri",
    "smoke_call",
    "utc_now",
]
