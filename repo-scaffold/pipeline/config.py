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
        values = {**dataclasses.asdict(cls()), **raw}
        for field_name in ("data_dir", "artifacts_dir", "results_dir"):
            values[field_name] = Path(values[field_name])
        return cls(**values)
