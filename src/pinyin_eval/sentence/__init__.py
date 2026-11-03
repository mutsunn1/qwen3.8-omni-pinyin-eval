"""Sentence-level (utterance) pinyin evaluation for Qwen3.8-Omni.

Design: ``docs/sentence-experiment-design.md`` (v0.1, 2026-11-01).

Three tasks share one runner but use different corpora and gold labels:

* **A** — whole-utterance pinyin transcription (AISHELL-3 gold, audio only);
* **B** — whole-utterance pronunciation evaluation (OMPAL word labels);
* **C** — whole-utterance reading scoring (OMPAL sentence-level human scores).

Conditions: ``C1`` = thinking off / clean / JSON; ``C2`` = thinking on / clean / JSON.
"""

from __future__ import annotations

__all__ = ["PROMPT_VERSION"]

#: Bumped whenever the prompt text or output schema changes (design §5).
PROMPT_VERSION = "sentence-v1"
