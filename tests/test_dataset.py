"""Tests for dataset preparation, canonical manifest and noise variants."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from pinyin_eval.config import Config
from pinyin_eval.data import dataset

from .conftest import CONFIG_PATH, MAPPING_CSV

# ---------------------------------------------------------------------------
# prepare-data
# ---------------------------------------------------------------------------


def test_prepare_data_manifest(tiny_parquet_cfg):
    df = dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    assert len(df) == 6
    assert df["sample_rate"].unique().tolist() == [16000]
    assert df["channels"].unique().tolist() == [1]
    # canonicalisation of the merged raw spelling "in" -> "yin"
    in_row = df[df["raw_target"] == "in"].iloc[0]
    assert in_row["target"] == "yin"
    assert (in_row["initial"], in_row["final"]) == ("", "in")
    # both raw spellings are retained with the same canonical target
    assert set(df[df["target"] == "yin"]["raw_target"]) == {"in", "yin"}
    # audio exported unchanged
    for _, row in df.iterrows():
        path = tiny_parquet_cfg.repo_root / row["wav_path"]
        assert path.exists()
        info = sf.info(path)
        assert info.samplerate == 16000 and info.channels == 1
    assert (tiny_parquet_cfg.path("paths.manifest_dir") / "audio.parquet").exists()
    assert (tiny_parquet_cfg.path("paths.manifest_dir") / "audio.csv").exists()


def test_prepare_data_idempotent(tiny_parquet_cfg):
    first = dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    second = dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    assert first.equals(second)


def test_load_audio_records_roundtrip(tiny_parquet_cfg):
    dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    records = dataset.load_audio_records(tiny_parquet_cfg)
    assert len(records) == 6
    assert records[0].audio_id.startswith("ZH_TTS_")
    assert records[0].stratum
    from pinyin_eval.records import AudioRecord

    assert isinstance(records[0], AudioRecord)


def test_dataset_summary(tiny_parquet_cfg):
    df = dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    summary = dataset.dataset_summary(df)
    assert summary["n_audio"] == 6
    assert summary["voices"] == ["A", "B"]
    assert summary["tones"] == {1: 4, 2: 2}
    assert summary["sample_rates"] == [16000]


# ---------------------------------------------------------------------------
# noise injection
# ---------------------------------------------------------------------------


def _read_wav(path) -> np.ndarray:
    data, _ = sf.read(path, dtype="float32")
    return data


def test_add_white_noise_snr():
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    signal = np.sin(2 * np.pi * 220 * t)
    noisy = dataset.add_white_noise(signal, snr_db=20.0, rng=rng)
    noise = noisy - signal
    measured = 10 * np.log10(np.mean(signal ** 2) / np.mean(noise ** 2))
    assert abs(measured - 20.0) < 1.5


def test_ensure_noisy_for_creates_deterministic_files(tiny_parquet_cfg):
    dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    records = dataset.load_audio_records(tiny_parquet_cfg)[:3]
    paths = dataset.ensure_noisy_for(
        tiny_parquet_cfg, [(r.audio_id, r.wav_path) for r in records]
    )
    assert all(p.exists() for p in paths)
    # SNR of the generated file is close to 20 dB
    snr = dataset.verify_noise_snr(paths[0], tiny_parquet_cfg.repo_root / records[0].wav_path)
    assert abs(snr - 20.0) < 1.0
    # deterministic: re-creating from scratch yields identical bytes
    first = paths[0].read_bytes()
    paths[0].unlink()
    dataset.ensure_noisy_for(tiny_parquet_cfg, [(records[0].audio_id, records[0].wav_path)])
    assert paths[0].read_bytes() == first
    # existing files are reused
    mtime = paths[0].stat().st_mtime_ns
    dataset.ensure_noisy_for(tiny_parquet_cfg, [(records[0].audio_id, records[0].wav_path)])
    assert paths[0].stat().st_mtime_ns == mtime


def test_media_path_resolution(tiny_parquet_cfg):
    dataset.prepare_data(tiny_parquet_cfg, skip_download=True)
    record = dataset.load_audio_records(tiny_parquet_cfg)[0]
    clean = dataset.media_path(tiny_parquet_cfg, record, "clean")
    assert clean.exists() and clean == tiny_parquet_cfg.repo_root / record.wav_path
    noisy = dataset.media_path(tiny_parquet_cfg, record, "noise_20db")
    assert noisy.exists()
    with pytest.raises(ValueError):
        dataset.media_path(tiny_parquet_cfg, record, "bogus")


# ---------------------------------------------------------------------------
# config / download helpers
# ---------------------------------------------------------------------------


def test_hf_parquet_url_pins_revision(tiny_cfg):
    url = dataset.hf_parquet_url(tiny_cfg)
    assert tiny_cfg["dataset.hf_revision"] in url
    assert "mandarin_tone_syn/mandarin_tone_syn.parquet" in url


def test_prepare_data_without_parquet_raises(tmp_path):
    cfg = Config.load(CONFIG_PATH, repo_root=tmp_path, overrides={
        "paths": {
            "raw_parquet": str(tmp_path / "missing.parquet"),
            "mapping_csv": str(MAPPING_CSV),
            "audio_dir": str(tmp_path / "audio"),
            "noisy_dir": str(tmp_path / "noisy"),
            "manifest_dir": str(tmp_path / "manifests"),
            "task_dir": str(tmp_path / "tasks"),
            "results_dir": str(tmp_path / "results"),
        },
    })
    with pytest.raises(dataset.DatasetError):
        dataset.prepare_data(cfg, skip_download=True)
