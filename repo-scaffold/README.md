# ML Pipeline Agent — repo scaffold (Phase 0)

This is the fixed layout described in the `ml-pipeline-agent` skill. Don't
change the top-level structure — add code inside these folders.

- `modalities/{text,image,video,audio}/` — pretrained encoders, embedding extractors
- `tabular/` — tabular-specific transforms, encoders, feature builders
- `eda/` — one file per analysis, `python -m eda.<name>`
- `analysis/` — per-approach writeups (what was tried, what broke, what fixed it)
- `results/` — metrics, plots, comparison tables, archived per task_id
- `artifacts/` — cached embeddings, fitted transformers, checkpoints (gitignored)
- `queue/`, `current/` — the Kaggle git-poll worker's task queue (see the
  ml-pipeline-agent skill's `references/worker-protocol.md`)
- `pipeline/` — `dataset/`, `preprocessing/`, `models/`, `pipelines/`, plus
  `config.py`, `log.py`, `main.py`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
ruff check .          # should pass clean on the scaffold
pytest -k smoke        # runs (0 collected until stage 1 tests are written)
```

Set `WANDB_API_KEY` in your environment before running any stage that logs.

## Next step

Work through Phase 1 (EDA) per the skill's stage sequence: one module per data
concern in `eda/`, each exposing `run() -> dict/plots`, logged to W&B, with a
smoke test in `eda/test_smoke.py`.
