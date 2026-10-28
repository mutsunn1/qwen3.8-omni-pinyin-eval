"""Dataset preparation: download, audio export, canonical manifest, noise variants.

Data source (pinned in ``configs/base.yaml``):
``HaitongSUN/prosodic-abx``, config ``mandarin_tone_syn``, commit
``d607806619aba678030dac10ea086fb19347205a``.

Verified facts (2026-10-28):
* 6160 rows, columns ``id, audio{bytes,path}, speaker, target, label``
* audio: WAV, 16 kHz, mono, PCM_16, 0.215-1.334 s
* 4 TTS voices x 1540 rows; tones 1-4 x 1540 rows
* 385 raw target spellings -> 378 canonical spellings
"""

from __future__ import annotations

import hashlib
import io
import logging
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from ..config import Config
from ..normalize import canonical_target, load_mapping
from ..records import AudioRecord

log = logging.getLogger(__name__)

AUDIO_COLUMNS = [
    "audio_id", "row_index", "speaker", "voice", "raw_target", "target",
    "initial", "final", "tone", "wav_path", "duration_s", "sample_rate", "channels",
]

HF_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"


class DatasetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def hf_parquet_url(cfg: Config) -> str:
    repo = cfg["dataset.hf_repo"]
    rev = cfg["dataset.hf_revision"]
    config_name = cfg["dataset.config_name"]
    return HF_RESOLVE_URL.format(
        repo=repo, rev=rev, path=f"{config_name}/{config_name}.parquet"
    )


def download_dataset(cfg: Config, *, force: bool = False) -> Path:
    """Download the pinned parquet file if missing (streaming, retrying)."""
    dest = cfg.path("paths.raw_parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.info("dataset parquet already present: %s", dest)
        return dest
    url = hf_parquet_url(cfg)
    log.info("downloading %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
                total = int(resp.headers.get("content-length") or 0)
                written = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if total and written % (32 << 20) < (1 << 20):
                        log.info("downloaded %.0f/%.0f MB", written / 1e6, total / 1e6)
            tmp.replace(dest)
            log.info("downloaded %.1f MB -> %s", dest.stat().st_size / 1e6, dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried below
            last_error = exc
            log.warning("download attempt %d failed: %s", attempt, exc)
            tmp.unlink(missing_ok=True)
    raise DatasetError(f"failed to download {url}: {last_error}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _voice_of(speaker: str) -> str:
    return speaker.rsplit("_", 1)[-1]


def _relative(cfg: Config, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(cfg.repo_root))
    except ValueError:
        return str(path)


def read_parquet_rows(cfg: Config) -> list[dict[str, Any]]:
    path = cfg.path("paths.raw_parquet")
    if not path.exists():
        raise DatasetError(
            f"raw parquet missing: {path}. Run `pinyin-eval prepare-data` first."
        )
    table = pq.read_table(path)
    return table.to_pylist()


def prepare_data(cfg: Config, *, force: bool = False, skip_download: bool = False) -> pd.DataFrame:
    """Download + export audio + build canonical audio manifest."""
    if not skip_download:
        download_dataset(cfg, force=force)

    mapping = load_mapping(cfg.path("paths.mapping_csv"))
    rows = read_parquet_rows(cfg)
    audio_dir = cfg.path("paths.audio_dir")
    manifest_dir = cfg.path("paths.manifest_dir")
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    profiles: Counter[tuple[int, int, str]] = Counter()
    for row_index, row in enumerate(rows):
        audio_id = str(row["id"])
        audio_obj = row.get("audio") or {}
        data = audio_obj.get("bytes")
        if not data:
            raise DatasetError(f"row {row_index} ({audio_id}) has no audio bytes")
        raw_target = str(row["target"]).strip()
        target = canonical_target(raw_target, mapping)
        entry = mapping[target]
        speaker = str(row["speaker"])
        tone = int(row["label"])

        wav_path = audio_dir / speaker / f"{audio_id}.wav"
        if force or not wav_path.exists() or wav_path.stat().st_size != len(data):
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_path.write_bytes(data)

        try:
            info = sf.info(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            raise DatasetError(f"cannot decode audio for {audio_id}: {exc}") from exc
        profiles[(info.samplerate, info.channels, info.subtype)] += 1

        expected_sr = int(cfg.get("dataset.expect_sample_rate", info.samplerate))
        expected_ch = int(cfg.get("dataset.expect_channels", info.channels))
        if (info.samplerate, info.channels) != (expected_sr, expected_ch):
            log.warning(
                "audio %s has %d Hz / %d ch (expected %d Hz / %d ch)",
                audio_id, info.samplerate, info.channels, expected_sr, expected_ch,
            )

        records.append({
            "audio_id": audio_id,
            "row_index": row_index,
            "speaker": speaker,
            "voice": _voice_of(speaker),
            "raw_target": raw_target,
            "target": target,
            "initial": entry.initial,
            "final": entry.final,
            "tone": tone,
            "wav_path": _relative(cfg, wav_path),
            "duration_s": round(info.frames / info.samplerate, 6),
            "sample_rate": info.samplerate,
            "channels": info.channels,
        })

    df = pd.DataFrame(records, columns=AUDIO_COLUMNS)
    _validate_manifest(df, cfg)

    out_parquet = manifest_dir / "audio.parquet"
    out_csv = manifest_dir / "audio.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    log.info("wrote %s rows to %s and %s", len(df), out_parquet, out_csv)

    summary = dataset_summary(df)
    log.info("dataset summary: %s", summary)
    return df


def _validate_manifest(df: pd.DataFrame, cfg: Config) -> None:
    if len(df) == 0:
        raise DatasetError("empty audio manifest")
    dup = df["audio_id"].duplicated()
    if dup.any():
        raise DatasetError(f"duplicate audio ids: {df.loc[dup, 'audio_id'].tolist()[:5]}")
    missing = df[df["wav_path"].map(lambda p: not (cfg.repo_root / p).exists())]
    if len(missing):
        raise DatasetError(
            f"{len(missing)} manifest rows point to missing audio files, e.g. "
            f"{missing['wav_path'].iloc[0]}"
        )
    bad_tone = df[~df["tone"].isin([1, 2, 3, 4])]
    if len(bad_tone):
        raise DatasetError(f"invalid tones: {bad_tone[['audio_id', 'tone']].head().to_dict('records')}")
    empty_initial = df["initial"].isna() | (df["initial"].astype(str) == "")
    if empty_initial.isna().any():
        raise DatasetError("initial column has NaN")
    if (df["final"].astype(str) == "").any():
        raise DatasetError("empty final in manifest")


def load_audio_records(cfg: Config) -> list[AudioRecord]:
    path = cfg.path("paths.manifest_dir") / "audio.parquet"
    if not path.exists():
        raise DatasetError(
            f"audio manifest missing: {path}. Run `pinyin-eval prepare-data` first."
        )
    df = pd.read_parquet(path)
    records: list[AudioRecord] = []
    for row in df.to_dict("records"):
        records.append(AudioRecord(**{k: row[k] for k in AUDIO_COLUMNS}))
    return records


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """JSON-serialisable summary used in reports/manifests."""
    return {
        "n_audio": int(len(df)),
        "n_voices": int(df["voice"].nunique()),
        "voices": sorted(df["voice"].unique().tolist()),
        "n_raw_targets": int(df["raw_target"].nunique()),
        "n_canonical_targets": int(df["target"].nunique()),
        "tones": {int(k): int(v) for k, v in df["tone"].value_counts().sort_index().items()},
        "per_voice": {str(k): int(v) for k, v in df["voice"].value_counts().sort_index().items()},
        "sample_rates": sorted(df["sample_rate"].unique().tolist()),
        "channels": sorted(df["channels"].unique().tolist()),
        "duration_s": {
            "min": float(df["duration_s"].min()),
            "median": float(df["duration_s"].median()),
            "mean": float(df["duration_s"].mean()),
            "max": float(df["duration_s"].max()),
        },
    }


# ---------------------------------------------------------------------------
# Media paths & noise injection
# ---------------------------------------------------------------------------


def clean_path(cfg: Config, record: AudioRecord) -> Path:
    p = Path(record.wav_path)
    return p if p.is_absolute() else cfg.repo_root / p


def noisy_path(cfg: Config, audio_id: str) -> Path:
    return cfg.path("paths.noisy_dir") / f"{audio_id}__snr20db.wav"


def media_path(cfg: Config, record: AudioRecord, audio_quality: str) -> Path:
    if audio_quality == "clean":
        return clean_path(cfg, record)
    if audio_quality == "noise_20db":
        path = noisy_path(cfg, record.audio_id)
        if not path.exists():
            ensure_noisy_audio(cfg, [record])
        return path
    raise ValueError(f"unknown audio_quality: {audio_quality!r}")


def _seed_for(audio_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{audio_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def add_white_noise(signal: np.ndarray, *, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at the requested SNR (power ratio)."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    signal_power = float(np.mean(x ** 2))
    if signal_power <= 0:
        return x.astype(np.float32)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32)


def ensure_noisy_audio(cfg: Config, records: Iterable[AudioRecord]) -> list[Path]:
    """Create (and cache) the SNR-20 dB white-noise variant of each audio."""
    snr_db = float(cfg.get("generation.noise_snr_db", 20))
    seed = cfg.seed()
    out_dir = cfg.path("paths.noisy_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for record in records:
        dest = noisy_path(cfg, record.audio_id)
        paths.append(dest)
        if dest.exists():
            continue
        src = clean_path(cfg, record)
        signal, sr = sf.read(src, dtype="float32", always_2d=False)
        rng = np.random.default_rng(_seed_for(record.audio_id, seed))
        noisy = add_white_noise(signal, snr_db=snr_db, rng=rng)
        clipped = np.clip(noisy, -1.0, 1.0)
        sf.write(dest, clipped, sr, subtype="PCM_16")
    return paths


def ensure_noisy_for(
    cfg: Config,
    items: Iterable[tuple[str, str]],
    *,
    snr_db: float | None = None,
) -> list[Path]:
    """Create noisy variants from ``(audio_id, clean_wav_relpath)`` pairs."""
    if snr_db is None:
        snr_db = float(cfg.get("generation.noise_snr_db", 20))
    seed = cfg.seed()
    out_dir = cfg.path("paths.noisy_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for audio_id, clean_rel in items:
        dest = noisy_path(cfg, audio_id)
        paths.append(dest)
        if dest.exists():
            continue
        src = Path(clean_rel)
        if not src.is_absolute():
            src = cfg.repo_root / src
        signal, sr = sf.read(src, dtype="float32", always_2d=False)
        rng = np.random.default_rng(_seed_for(audio_id, seed))
        noisy = np.clip(add_white_noise(signal, snr_db=snr_db, rng=rng), -1.0, 1.0)
        sf.write(dest, noisy, sr, subtype="PCM_16")
    return paths


def verify_noise_snr(noisy_file: Path, clean_file: Path | None = None) -> float:
    """Diagnostic: re-estimate the SNR of a noisy file against its clean source."""
    noisy, sr = sf.read(noisy_file, dtype="float32")
    if clean_file is None:
        clean_id = noisy_file.name.split("__")[0]
        candidates = [
            p for p in noisy_file.parent.parent.glob(f"*/*{clean_id}*.wav")
            if p.name != noisy_file.name
        ]
        if not candidates:
            raise FileNotFoundError(f"clean audio for {noisy_file.name} not found")
        clean_file = candidates[0]
    clean, _ = sf.read(clean_file, dtype="float32")
    n = min(len(clean), len(noisy))
    noise = noisy[:n] - clean[:n]
    sig_pow = float(np.mean(clean[:n] ** 2))
    noise_pow = float(np.mean(noise ** 2))
    if noise_pow <= 0:
        return float("inf")
    return float(10 * np.log10(sig_pow / noise_pow))


__all__ = [
    "AUDIO_COLUMNS",
    "DatasetError",
    "add_white_noise",
    "clean_path",
    "dataset_summary",
    "download_dataset",
    "ensure_noisy_audio",
    "hf_parquet_url",
    "load_audio_records",
    "media_path",
    "noisy_path",
    "prepare_data",
    "read_parquet_rows",
    "verify_noise_snr",
]
