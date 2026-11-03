"""Deterministic mock client for the sentence-level tasks.

Used by ``sentence generate --mock`` and the offline tests: it emits outputs in
exactly the JSON schema the parsers/scorers expect, with the gold answer drawn
from the plan metadata.  Modes (``mock.mode`` in the config):

* ``perfect`` — always answers the gold labels;
* ``noisy``   — flips a deterministic fraction (``mock.error_rate``) of the
  answers;
* ``random``  — random but schema-valid answers.

It never touches the network and its token usage is a rough character-based
estimate, so mock runs validate the *pipeline* (not the cost model).
"""

from __future__ import annotations

import json
import random
import time

from ..client import BaseClient, ChatRequest, ChatResponse, utc_now
from ..config import Config
from .normalize import align_gold, normalize_pinyin_token

__all__ = ["SentenceMockClient"]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // 2)


class SentenceMockClient(BaseClient):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = f"mock-{cfg.model_name}"
        self.mode = str(cfg.get("mock.mode", "perfect"))
        self.error_rate = float(cfg.get("mock.error_rate", 0.0))
        self.latency_s = float(cfg.get("mock.latency_s", 0.0))
        self.rng = random.Random(int(cfg.get("mock.seed", cfg.seed())))

    # -- answers -----------------------------------------------------------
    def _answer_a(self, meta: dict) -> dict:
        chars = str(meta.get("gold_chars") or "")
        tokens = list(meta.get("gold_pinyin") or [])
        gold = align_gold(chars, tokens)
        if gold is None:
            gold = [normalize_pinyin_token(token) for token in tokens]
        syllables = []
        for syl in gold:
            initial, final, tone = syl.initial, syl.final, syl.tone
            if self.mode == "random" or (
                self.mode != "perfect" and self.rng.random() < self.error_rate
            ):
                tone = self.rng.choice([1, 2, 3, 4, 5])
                if self.rng.random() < 0.3:
                    initial = ""
            syllables.append({
                "char": getattr(syl, "char", "") or "",
                "initial": initial,
                "final": final,
                "tone": tone,
            })
        return {"syllables": syllables}

    def _answer_b(self, meta: dict) -> dict:
        labels = list(meta.get("gold_labels") or [])
        words = []
        for unit in labels:
            flags = {key: int(unit.get(key, 1)) == 1 for key in ("tone", "consonant", "vowel")}
            if self.mode == "random" or (
                self.mode != "perfect" and self.rng.random() < self.error_rate
            ):
                for key in flags:
                    if self.rng.random() < 0.5:
                        flags[key] = not flags[key]
            words.append({
                "text": unit.get("text", ""),
                "tone_ok": flags["tone"],
                "consonant_ok": flags["consonant"],
                "vowel_ok": flags["vowel"],
            })
        return {"words": words}

    def _answer_c(self, meta: dict) -> dict:
        gold = dict(meta.get("gold_scores") or {})
        scores = {}
        for key in ("accuracy", "fluency", "prosody"):
            value = float(gold.get(key, 4.0))
            if self.mode == "random":
                value = self.rng.uniform(1.0, 5.0)
            elif self.mode != "perfect" and self.rng.random() < self.error_rate:
                value = min(5.0, max(1.0, value + self.rng.choice([-1.0, 1.0])))
            scores[key] = round(value, 1)
        return scores

    def _answer_d(self, meta: dict) -> dict:
        text = list(str(meta.get("gold_text") or ""))
        gold_alignment = list(meta.get("gold_alignment") or [])
        gold_scores = dict(meta.get("gold_scores") or {})
        alignment = []
        for index, char in enumerate(text):
            info = gold_alignment[index] if index < len(gold_alignment) else {}
            aligned = bool(info.get("aligned"))
            text_error = (not bool(info.get("text_correct"))) if aligned else True
            error_type = info.get("error_type")
            if self.mode == "random":
                aligned = self.rng.random() < 0.8
                text_error = self.rng.random() < 0.1
                error_type = self.rng.choice(["none", "tone", "consonant", "vowel"])
            elif self.mode != "perfect" and self.rng.random() < self.error_rate:
                if self.rng.random() < 0.5:
                    aligned = not aligned
                else:
                    error_type = self.rng.choice(["none", "tone", "consonant", "vowel"])
            alignment.append({
                "text": char,
                "aligned": aligned,
                "text_error": text_error,
                "error_type": error_type if (aligned and not text_error) else None,
                "score": round(self.rng.uniform(3.0, 5.0) if self.mode == "random" else 4.0, 1),
            })
        scores = {}
        for key in ("accuracy", "fluency", "prosody"):
            value = float(gold_scores.get(key, 4.0))
            if self.mode == "random":
                value = self.rng.uniform(1.0, 5.0)
            elif self.mode != "perfect" and self.rng.random() < self.error_rate:
                value = min(5.0, max(1.0, value + self.rng.choice([-1.0, 1.0])))
            scores[key] = round(value, 1)
        return {"alignment": alignment, "scores": scores}

    def generate(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)
        meta = request.meta or {}
        task = str(meta.get("sentence_task") or "").upper()
        if task == "A":
            payload = self._answer_a(meta)
        elif task == "B":
            payload = self._answer_b(meta)
        elif task == "C":
            payload = self._answer_c(meta)
        elif task == "D":
            payload = self._answer_d(meta)
        else:  # pragma: no cover - defensive
            return ChatResponse(request_id=request.request_id, error=f"unknown task {task!r}",
                                model=self.model, timestamp=utc_now())
        text = json.dumps(payload, ensure_ascii=False)
        reasoning = "（mock reasoning）" if request.thinking else ""
        prompt_tokens = _estimate_tokens(request.system_prompt + request.prompt_text) + 50
        completion_tokens = _estimate_tokens(text) + _estimate_tokens(reasoning)
        return ChatResponse(
            request_id=request.request_id,
            text=text,
            reasoning=reasoning,
            finish_reason="stop",
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            latency_s=time.perf_counter() - started,
            timestamp=utc_now(),
            model=self.model,
        )
