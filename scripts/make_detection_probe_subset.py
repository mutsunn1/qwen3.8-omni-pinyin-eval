"""Build the fixed 512-audio subset used by the detection prompt probes.

The subset is deterministic (seed 20261028) and written to
``data/manifests/tasks/probe_detect_items.txt`` (all 6 detection task ids per
audio) and ``probe_detect_audios.txt`` (the audio ids).  ``data/`` is gitignored,
so this script is the reproducible way to recreate the subset after
``build-tasks`` has produced ``data/manifests/tasks/detection.jsonl``.

Usage::

    PYTHONPATH=src python scripts/make_detection_probe_subset.py
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data" / "manifests" / "tasks"
SEED = 20261028
N_AUDIOS = 512


def main() -> None:
    by_audio: dict[str, list[str]] = defaultdict(list)
    with (TASKS / "detection.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            task = json.loads(line)
            by_audio[task["audio_id"]].append(task["item_id"])
    audios = sorted(by_audio)
    rng = random.Random(SEED)
    picked = sorted(rng.sample(audios, N_AUDIOS))
    items = [item for audio in picked for item in by_audio[audio]]
    (TASKS / "probe_detect_audios.txt").write_text("\n".join(picked) + "\n", encoding="utf-8")
    (TASKS / "probe_detect_items.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
    print(f"audios={len(picked)} items={len(items)} -> {TASKS / 'probe_detect_items.txt'}")


if __name__ == "__main__":
    main()
