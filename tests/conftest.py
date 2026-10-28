"""Shared test fixtures: synthetic audio records / tiny configs (no real audio)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from pinyin_eval.config import Config
from pinyin_eval.normalize import load_mapping
from pinyin_eval.records import AudioRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
MAPPING_CSV = REPO_ROOT / "data" / "mapping" / "mandarin_tone_syn_target_mapping.csv"


@pytest.fixture(scope="session")
def mapping():
    return load_mapping(MAPPING_CSV)


@pytest.fixture(scope="session")
def repo_cfg():
    """Config rooted at the repo (for pure in-memory task construction tests)."""
    return Config.load(CONFIG_PATH, repo_root=REPO_ROOT)


def make_synthetic_records(mapping, *, voices=("A", "B", "C", "D"), tones=(1, 2, 3, 4),
                           targets=None, wav_prefix="data/audio") -> list[AudioRecord]:
    """AudioRecords for every (canonical target, voice, tone) — no real files."""
    targets = sorted(mapping) if targets is None else list(targets)
    records: list[AudioRecord] = []
    for target in targets:
        entry = mapping[target]
        for voice in voices:
            for tone in tones:
                audio_id = f"ZH_TTS_{voice}_{target}{tone}"
                records.append(AudioRecord(
                    audio_id=audio_id,
                    row_index=len(records),
                    speaker=f"ZH_TTS_{voice}",
                    voice=voice,
                    raw_target=target,
                    target=target,
                    initial=entry.initial,
                    final=entry.final,
                    tone=tone,
                    wav_path=f"{wav_prefix}/ZH_TTS_{voice}/{audio_id}.wav",
                    duration_s=0.5,
                    sample_rate=16000,
                    channels=1,
                ))
    return records


@pytest.fixture()
def tiny_records(mapping):
    """8 audios covering initial/final/tone contrasts needed by core tests."""
    targets = ["zhang", "zang", "chan", "chang", "ni", "nü", "yin", "ying"]
    records = []
    idx = 0
    for target in targets:
        entry = mapping[target]
        for voice in ("A", "B"):
            for tone in (1, 2):
                audio_id = f"ZH_TTS_{voice}_{target}{tone}"
                records.append(AudioRecord(
                    audio_id=audio_id,
                    row_index=idx,
                    speaker=f"ZH_TTS_{voice}",
                    voice=voice,
                    raw_target=target,
                    target=target,
                    initial=entry.initial,
                    final=entry.final,
                    tone=tone,
                    wav_path=f"data/audio/ZH_TTS_{voice}/{audio_id}.wav",
                    duration_s=0.5,
                    sample_rate=16000,
                    channels=1,
                ))
                idx += 1
    return records


@pytest.fixture()
def tiny_cfg(tmp_path, tiny_records):
    overrides = {
        "paths": {
            "raw_parquet": str(tmp_path / "raw" / "data.parquet"),
            "mapping_csv": str(MAPPING_CSV),
            "audio_dir": str(tmp_path / "audio"),
            "noisy_dir": str(tmp_path / "noisy"),
            "manifest_dir": str(tmp_path / "manifests"),
            "task_dir": str(tmp_path / "manifests" / "tasks"),
            "results_dir": str(tmp_path / "results"),
        },
    }
    return Config.load(CONFIG_PATH, repo_root=tmp_path, overrides=overrides)


# ---------------------------------------------------------------------------
# Tiny parquet dataset (real WAV bytes) for dataset tests
# ---------------------------------------------------------------------------


def wav_bytes(duration_s: float = 0.2, sample_rate: int = 16000, freq: float = 220.0) -> bytes:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * freq * t)
    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_tiny_parquet(path: Path, rows: list[dict]) -> None:
    """rows: dict(id, speaker, target, label)."""
    audio_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table({
        "id": pa.array([r["id"] for r in rows], pa.string()),
        "audio": pa.array([
            {"bytes": wav_bytes(freq=200.0 + i * 20), "path": f"{r['id']}.wav"}
            for i, r in enumerate(rows)
        ], audio_struct),
        "speaker": pa.array([r["speaker"] for r in rows], pa.string()),
        "target": pa.array([r["target"] for r in rows], pa.string()),
        "label": pa.array([r["label"] for r in rows], pa.int64()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture()
def tiny_parquet_cfg(tmp_path):
    rows = [
        {"id": "ZH_TTS_A_yin1", "speaker": "ZH_TTS_A", "target": "yin", "label": 1},
        {"id": "ZH_TTS_A_in1", "speaker": "ZH_TTS_A", "target": "in", "label": 1},
        {"id": "ZH_TTS_A_zhang1", "speaker": "ZH_TTS_A", "target": "zhang", "label": 1},
        {"id": "ZH_TTS_A_zhang2", "speaker": "ZH_TTS_A", "target": "zhang", "label": 2},
        {"id": "ZH_TTS_B_zhang1", "speaker": "ZH_TTS_B", "target": "zhang", "label": 1},
        {"id": "ZH_TTS_B_zhang2", "speaker": "ZH_TTS_B", "target": "zhang", "label": 2},
    ]
    parquet_path = tmp_path / "raw" / "data.parquet"
    write_tiny_parquet(parquet_path, rows)
    overrides = {
        "paths": {
            "raw_parquet": str(parquet_path),
            "mapping_csv": str(MAPPING_CSV),
            "audio_dir": str(tmp_path / "audio"),
            "noisy_dir": str(tmp_path / "noisy"),
            "manifest_dir": str(tmp_path / "manifests"),
            "task_dir": str(tmp_path / "manifests" / "tasks"),
            "results_dir": str(tmp_path / "results"),
        },
        "model": {"concurrency": 2, "qps": None},
    }
    cfg = Config.load(CONFIG_PATH, repo_root=tmp_path, overrides=overrides)
    return cfg


def config_with(cfg: Config, overrides: dict) -> Config:
    return cfg.with_overrides(overrides)
