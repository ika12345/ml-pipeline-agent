# Phase 0 Scaffold — reference templates

Use these as starting points when scaffolding a brand-new repo. Adapt names/paths
to the specific dataset, but keep the shape identical.

## Directory creation

```bash
mkdir -p modalities/{text,image,video,audio} tabular eda analysis results \
         artifacts queue current \
         pipeline/{dataset,preprocessing,models,pipelines}
touch modalities/{text,image,video,audio}/__init__.py tabular/__init__.py \
      eda/__init__.py pipeline/__init__.py pipeline/{dataset,preprocessing,models,pipelines}/__init__.py
```

## `pipeline/config.py`

```python
"""Single source of truth for paths, hyperparameters, and seeds.
Nothing hardcoded outside this file."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class Config:
    # paths
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")

    # run identity
    task_id: str = "task_0000"
    stage: str = "eda"
    approach: str = "baseline"

    # experiment
    seed: int = 42
    metric: str = "rmse"

    # compute
    disk_budget_gb: float = 50.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**{**dataclasses.asdict(cls()), **raw})
```

## `pipeline/log.py`

```python
"""wandb init/log/finish wrapper with a consistent run-naming scheme:
<stage>-<approach>-<date>"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

import wandb

from pipeline.config import Config


def run_name(cfg: Config) -> str:
    date = dt.date.today().isoformat()
    return f"{cfg.stage}-{cfg.approach}-{date}"


@contextmanager
def wandb_run(cfg: Config, project: str):
    run = wandb.init(project=project, name=run_name(cfg), config=cfg.__dict__)
    try:
        yield run
    finally:
        wandb.finish()
```

## `pipeline/main.py`

```python
"""Entrypoint: parses config, calls dataset -> preprocessing -> model -> eval.
No logic lives here beyond wiring stages together."""
from __future__ import annotations

import argparse

from pipeline.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # Wire stages here as they're built, e.g.:
    # from pipeline.dataset import run as dataset_run
    # from pipeline.preprocessing import run as preprocessing_run
    # artifacts = dataset_run(cfg)
    # artifacts = preprocessing_run(cfg, artifacts)
    raise NotImplementedError("Wire pipeline stages here as they're implemented.")


if __name__ == "__main__":
    main()
```

## `config/smoke.yaml`

```yaml
task_id: smoke
stage: smoke
approach: smoke
data_dir: data/smoke_sample
```

## `pyproject.toml` (ruff config)

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
markers = ["smoke: fast tiny-sample tests run after every change"]
```

## Smoke test pattern — `eda/test_smoke.py` (one per folder)

```python
"""Tiny-sample smoke test. Asserts shapes / no-NaN, never runs on full data."""
import pytest


@pytest.mark.smoke
def test_eda_runs_on_sample(tiny_sample_df):
    from eda.missingness import run

    result = run(tiny_sample_df)
    assert result is not None
```

## `conftest.py` fixture harness (repo root)

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_sample_df() -> pd.DataFrame:
    """~20-row fixture standing in for the real dataset during smoke tests."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(20),
            "text": [f"sample item {i}" for i in range(20)],
            "target": rng.uniform(1, 100, size=20),
        }
    )
```

## Checklist before calling Phase 0 done

- [ ] Directory tree created with `__init__.py` in every importable package
- [ ] `pipeline/config.py`, `pipeline/log.py`, `pipeline/main.py` in place
- [ ] `pyproject.toml` with ruff + pytest config
- [ ] `conftest.py` with a tiny fixture dataset
- [ ] `config/smoke.yaml` exists
- [ ] `ruff check .` passes on the empty scaffold
- [ ] `pytest -k smoke` runs (even with zero tests collected) without error
- [ ] W&B project created and `WANDB_API_KEY` available in the environment
