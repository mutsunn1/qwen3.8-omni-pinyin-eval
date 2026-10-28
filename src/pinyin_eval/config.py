"""Configuration loading.

A single YAML file (``configs/base.yaml`` by default) describes data paths,
model parameters, task construction, statistics and reporting.  All relative
paths are resolved against ``repo_root`` (defaults to the current working
directory) so that the same config works from any checkout location.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = "configs/base.yaml"


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """Thin wrapper around a nested dict with dotted-path access."""

    def __init__(self, data: dict[str, Any], *, repo_root: Path, source_path: Path | None = None):
        if not isinstance(data, dict):
            raise ConfigError("config root must be a mapping")
        self._data = data
        self.repo_root = Path(repo_root).resolve()
        self.source_path = Path(source_path).resolve() if source_path else None

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG, *, repo_root: str | Path | None = None,
             overrides: dict[str, Any] | None = None) -> Config:
        config_path = Path(path)
        if not config_path.is_absolute():
            root = Path(repo_root) if repo_root is not None else Path.cwd()
            config_path = Path(root) / config_path
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if overrides:
            raw = _deep_merge(raw, overrides)
        root = Path(repo_root) if repo_root is not None else Path.cwd()
        cfg = cls(raw, repo_root=root, source_path=config_path)
        cfg.validate()
        return cfg

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        return Config(_deep_merge(self._data, overrides), repo_root=self.repo_root,
                      source_path=self.source_path)

    # -- access ------------------------------------------------------------
    def __getitem__(self, dotted: str) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"config key not found: {dotted!r}")
            node = node[part]
        return node

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return self[dotted]
        except KeyError:
            return default

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def path(self, dotted: str, default: str | None = None) -> Path:
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(f"config path not found: {dotted!r}")
        p = Path(value)
        return p if p.is_absolute() else (self.repo_root / p)

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- derived helpers ---------------------------------------------------
    @property
    def model_name(self) -> str:
        return str(self["model.model"])

    @property
    def main_condition(self) -> dict[str, Any]:
        return dict(self["generation.main_condition"])

    def all_conditions(self) -> list[dict[str, Any]]:
        """Full factorial condition grid (design: 2 thinking x 2 quality x 2 form)."""
        thinking_levels = self.get("generation.thinking_levels", [True, False])
        qualities = self.get("generation.audio_qualities", ["clean", "noise_20db"])
        forms = self.get("generation.output_forms", ["json", "text"])
        return [
            {"thinking": bool(t), "audio_quality": str(q), "output_form": str(f)}
            for t in thinking_levels
            for q in qualities
            for f in forms
        ]

    def seed(self) -> int:
        return int(self.get("project.seed", 20261028))

    def config_sha256(self) -> str:
        payload = json.dumps(self._data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()

    # -- validation --------------------------------------------------------
    def validate(self) -> None:
        required = [
            "project.name",
            "paths.raw_parquet",
            "paths.mapping_csv",
            "paths.audio_dir",
            "paths.manifest_dir",
            "paths.task_dir",
            "paths.results_dir",
            "dataset.hf_repo",
            "dataset.hf_revision",
            "model.model",
            "generation.main_condition",
            "tasks.ablation.items_per_condition",
            "stats.n_boot",
        ]
        missing = [key for key in required if self.get(key) is None]
        if missing:
            raise ConfigError(f"missing required config keys: {missing}")
        main = self.main_condition
        for field in ("thinking", "audio_quality", "output_form"):
            if field not in main:
                raise ConfigError(f"generation.main_condition missing {field!r}")
        if main["output_form"] not in ("json", "text"):
            raise ConfigError("generation.main_condition.output_form must be json|text")
        if main["audio_quality"] not in ("clean", "noise_20db"):
            raise ConfigError("generation.main_condition.audio_quality must be clean|noise_20db")
        if int(self.get("stats.n_boot", 0)) < 100:
            raise ConfigError("stats.n_boot must be >= 100")

    def env(self, dotted: str, default: str | None = None) -> str | None:
        """Read a value that may be an ``env:VAR`` reference."""
        value = self.get(dotted, default)
        if isinstance(value, str) and value.startswith("env:"):
            return os.environ.get(value[4:])
        return value


__all__ = ["Config", "ConfigError", "DEFAULT_CONFIG"]
