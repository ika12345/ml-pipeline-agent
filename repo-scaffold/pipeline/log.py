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
