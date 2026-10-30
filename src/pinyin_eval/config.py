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

#: environment variable that can override the default config path (see ``.env``)
CONFIG_ENV_VAR = "PINYIN_EVAL_CONFIG"


def load_dotenv(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Minimal ``.env`` loader (no external dependency).

    Reads ``KEY=VALUE`` lines, ignores blank lines / ``#`` comments, strips
    optional ``export`` prefixes and surrounding quotes.  Existing environment
    variables win unless ``override=True``.  This keeps the local environment,
    the CLI and the test suite on one single source of truth for credentials.
    """
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def default_config_path() -> str:
    """Default config path, overridable with ``PINYIN_EVAL_CONFIG``."""
    return os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG


class ConfigError(ValueError):
    pass


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_with_extends(path: str | Path, *, _seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config and resolve ``extends`` recursively (single inheritance).

    ``extends: configs/base.yaml`` is resolved relative to the repo root first
    (i.e. ``config_path.parent.parent`` for configs living in ``configs/``) and
    then relative to the config's own directory.  Cycles are rejected.
    """
    config_path = Path(path)
    resolved = config_path.resolve()
    if resolved in _seen:
        raise ConfigError(f"config extends cycle detected: {config_path}")
    raw = _load_yaml(config_path)
    parent = raw.pop("extends", None)
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            candidates = [
                config_path.parent.parent / parent_path,
                config_path.parent / parent_path,
            ]
            parent_path = next((c for c in candidates if c.exists()), candidates[0])
        if not parent_path.exists():
            raise ConfigError(f"parent config not found: {parent} (from {config_path})")
        base_raw = _load_with_extends(parent_path, _seen=_seen | {resolved})
        raw = _deep_merge(base_raw, raw)
    return raw


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
    def load(cls, path: str | Path | None = None, *, repo_root: str | Path | None = None,
             overrides: dict[str, Any] | None = None) -> Config:
        root = Path(repo_root) if repo_root is not None else Path.cwd()
        load_dotenv(root / ".env")
        config_path = Path(path or default_config_path())
        if not config_path.is_absolute():
            config_path = Path(root) / config_path
        if not config_path.exists():
            raise ConfigError(f"config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # optional inheritance chain: ``extends: configs/base.yaml``
        raw = _load_with_extends(config_path)
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


__all__ = [
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG",
    "Config",
    "ConfigError",
    "default_config_path",
    "load_dotenv",
]
