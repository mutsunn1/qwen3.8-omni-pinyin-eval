"""Tests for the unified environment: .env loading + config env override."""

from __future__ import annotations

import os

import pytest

from pinyin_eval.config import (
    CONFIG_ENV_VAR,
    Config,
    ConfigError,
    default_config_path,
    load_dotenv,
)

from .conftest import CONFIG_PATH, REPO_ROOT


def test_load_dotenv_parses_key_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "# comment",
            "",
            "export FOO=bar",
            "QUOTED='hello world'",
            "DOUBLE=\"x=y\"",
            "EMPTY=",
            "NO_EQUALS_LINE",
        ]),
        encoding="utf-8",
    )
    loaded = load_dotenv(env_file)
    assert loaded == {"FOO": "bar", "QUOTED": "hello world", "DOUBLE": "x=y", "EMPTY": ""}
    assert os.environ["FOO"] == "bar"
    assert os.environ["QUOTED"] == "hello world"


def test_load_dotenv_does_not_override_existing(monkeypatch, tmp_path):
    monkeypatch.setenv("EXISTING_KEY", "from-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=from-file\nNEW_KEY=from-file\n", encoding="utf-8")
    load_dotenv(env_file)
    assert os.environ["EXISTING_KEY"] == "from-shell"
    assert os.environ["NEW_KEY"] == "from-file"
    load_dotenv(env_file, override=True)
    assert os.environ["EXISTING_KEY"] == "from-file"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "missing.env") == {}


def test_default_config_path_env_override(monkeypatch):
    monkeypatch.setenv(CONFIG_ENV_VAR, "custom.yaml")
    assert default_config_path() == "custom.yaml"
    monkeypatch.delenv(CONFIG_ENV_VAR)
    assert default_config_path() == "configs/base.yaml"


def test_config_load_uses_dotenv(monkeypatch, tmp_path):
    old_key = os.environ.pop("DASHSCOPE_API_KEY", None)
    try:
        (tmp_path / ".env").write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")
        (tmp_path / "configs").mkdir(exist_ok=True)
        cfg_path = tmp_path / "configs" / "base.yaml"
        cfg_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        cfg = Config.load(repo_root=tmp_path)
        assert cfg.repo_root == tmp_path.resolve()
        assert os.environ["DASHSCOPE_API_KEY"] == "sk-test"
    finally:
        if old_key is None:
            os.environ.pop("DASHSCOPE_API_KEY", None)
        else:
            os.environ["DASHSCOPE_API_KEY"] = old_key


def test_config_extends_minimal_design():
    base = Config.load(REPO_ROOT / "configs" / "base.yaml", repo_root=REPO_ROOT)
    mini = Config.load(REPO_ROOT / "configs" / "minimal.yaml", repo_root=REPO_ROOT)
    # inherited
    assert mini.model_name == base.model_name
    assert mini.path("paths.mapping_csv") == base.path("paths.mapping_csv")
    # overridden
    assert mini.get("generation.budget_cny") == 70
    assert mini.main_condition == {
        "thinking": False, "audio_quality": "clean", "output_form": "json",
    }
    conditions = mini.get("minimal.conditions")
    assert len(conditions) == 4
    assert {c["role"] for c in conditions} == {
        "main", "thinking_ablation", "noise_ablation", "output_ablation",
    }
    assert mini.get("pricing.input_per_mtok") == 0.8


def test_config_extends_missing_parent_raises(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text("extends: missing.yaml\nproject:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="parent config not found"):
        Config.load(child, repo_root=tmp_path)


def test_config_missing_required_key(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("project:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required config keys"):
        Config.load(cfg_path, repo_root=tmp_path)
