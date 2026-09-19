"""Entrypoint: parses config, calls dataset -> preprocessing -> model -> eval.
No logic lives here beyond wiring stages together."""
from __future__ import annotations

import argparse

from pipeline.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    Config.from_yaml(args.config)

    # Wire stages here as they're built, e.g.:
    # from pipeline.dataset import run as dataset_run
    # from pipeline.preprocessing import run as preprocessing_run
    # artifacts = dataset_run(cfg)
    # artifacts = preprocessing_run(cfg, artifacts)
    raise NotImplementedError("Wire pipeline stages here as they're implemented.")


if __name__ == "__main__":
    main()
