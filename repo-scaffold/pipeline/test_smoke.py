"""Phase 0 infrastructure smoke tests."""
from __future__ import annotations

import datetime as dt
import sys
import types
from pathlib import Path

import pytest

from pipeline.config import Config

try:
    import wandb  # noqa: F401
except ModuleNotFoundError:
    sys.modules["wandb"] = types.ModuleType("wandb")

from pipeline.log import run_name


@pytest.mark.smoke
def test_config_defaults_can_instantiate() -> None:
    cfg = Config()

    assert cfg.data_dir == Path("data")
    assert cfg.seed == 42


@pytest.mark.smoke
def test_smoke_yaml_loads_with_path_fields() -> None:
    cfg = Config.from_yaml(Path("config/smoke.yaml"))

    assert cfg.task_id == "smoke"
    assert cfg.stage == "smoke"
    assert cfg.approach == "smoke"
    assert cfg.data_dir == Path("data/smoke_sample")
    assert isinstance(cfg.data_dir, Path)


@pytest.mark.smoke
def test_run_name_follows_stage_approach_date_convention() -> None:
    cfg = Config(stage="stage", approach="approach")

    assert run_name(cfg) == f"stage-approach-{dt.date.today().isoformat()}"


@pytest.mark.smoke
def test_tiny_sample_fixture_has_expected_shape(tiny_sample_df) -> None:
    assert tiny_sample_df.shape == (20, 3)
    assert list(tiny_sample_df.columns) == ["id", "text", "target"]
