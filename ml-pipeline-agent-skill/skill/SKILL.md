---
name: ml-pipeline-agent
description: >
  Turns a raw competition/research dataset (tabular, text, image, video, and/or audio)
  into a clean, tested, W&B-tracked ML pipeline following a fixed EDA -> feature
  extraction -> dimensionality reduction -> pretrained-model probing -> baseline
  pipeline -> iteration sequence. Use this skill whenever the user asks to build,
  scaffold, or continue an ML pipeline / Kaggle competition harness / multimodal
  baseline, whenever they mention EDA, feature extraction, embeddings, pretrained
  encoder probing, sklearn ColumnTransformer/Pipeline work, or W&B experiment
  tracking for a dataset, and whenever they're working inside a repo that already
  matches the folder layout below (modalities/, tabular/, eda/, pipeline/, etc.) —
  even if they don't explicitly say "use the ML pipeline skill." Also use this to
  scaffold a brand-new repo of this kind from scratch, or to set up the Kaggle
  git-poll worker that executes long training runs on Kaggle's free GPU quota.
---

# ML Pipeline Agent

A reusable workflow for turning a raw dataset into a clean, tested, tracked ML
pipeline with minimal hand-holding. This skill generalizes the harness built for
the Amazon ML Challenge qualifier (data loading, EDA, tabular baselines, swappable
VLM/OCR wrapper) into something reusable across any tabular + multimodal
competition or research dataset.

## Inputs to collect before starting

If any of these are missing or ambiguous, ask before writing code:

- **Goal**: task type (classification / regression / ranking / generation) + metric to optimize
- **Dataset path(s) + modality manifest**: which of text / image / video / audio / tabular exist in this dataset
- **Run command**: `python -m pipeline.main --config <cfg>`
- **Test command**: `pytest -k smoke` (or a per-folder smoke script)
- **Compute budget**: default assumption is 2xT4-class GPUs on Kaggle with a ~50GB disk budget; confirm or adjust with the user

## Design principles (non-negotiable)

1. **No orphan code.** Every script lives under a folder that's importable as a
   module (`python -m eda.something`) and its outputs are consumed by the next
   stage's code, not copy-pasted between notebooks or files.
2. **Explain, don't just transform.** `eda/` modules aren't throwaway notebooks —
   `preprocessing/` imports directly from `eda/` so a reviewer (or a future agent
   run) can trace *why* a transform exists.
3. **Cheap before expensive.** Always check what a frozen pretrained model /
   embedding gives you before training anything. Only escalate to a trained
   `nn.Module` if the free signal isn't enough.
4. **sklearn for structure, PyTorch for learning.** `Pipeline`/`ColumnTransformer`
   own the plumbing (imputation, scaling, encoding, feature unions across
   modalities); PyTorch owns anything that needs gradients.
5. **Everything is a function, everything is logged.** No bare top-level script
   logic — each stage exposes a `run(config) -> artifacts` function so it's
   testable and composable. No exceptions, even for "quick" exploratory code.
6. **Compute-aware.** Cache embeddings/features to disk once and re-use them,
   prefer frozen encoders over fine-tuning, keep the heavy training loop on
   Kaggle while orchestration lives on the VPS/local machine.

## Repository structure (fixed — never invent a different layout)

```
root/
├── modalities/
│   ├── text/          # tokenizers, pretrained text encoders, embedding extractors
│   ├── image/          # pretrained vision encoders (CLIP/ResNet/etc.), embedding extractors
│   ├── video/          # frame sampling + per-frame or video encoder wrappers
│   └── audio/          # spectrogram/embedding extractors (e.g. wav2vec, CLAP)
├── tabular/             # tabular-specific transforms, encoders, feature builders
├── eda/                 # one file per analysis; each is `python -m eda.<name>`
├── analysis/            # per-approach writeups: what was tried, what broke, what fixed it
├── results/             # metrics, plots, leaderboard-style comparison tables — archived per task_id
├── artifacts/           # cached embeddings, fitted transformers, model checkpoints
├── queue/               # task_<id>.json files — the work queue between VPS and Kaggle
├── current/             # log.txt, status.json for whatever task is running/just finished
└── pipeline/
    ├── dataset/         # Dataset/DataLoader + tabular loaders, train/val/test splits
    ├── preprocessing/   # imports eda/* for rationale; sklearn ColumnTransformer/Pipeline
    ├── models/          # nn.Module definitions, one class per file
    ├── pipelines/       # end-to-end sklearn Pipeline objects (feature union of modalities)
    ├── log.py           # W&B init/logging helpers, run naming convention
    ├── config.py        # dataclass/YAML-driven config (paths, hyperparams, seed)
    └── main.py          # entrypoint: parses config, calls dataset -> preprocessing -> model -> eval
```

Invocation convention stays flat and predictable — never nest deeper than this:

```
python -m eda.<analysis_name>
python -m modalities.image.<probe_name>   # same pattern for text/video/audio
python -m tabular.<feature_name>
python -m pipeline.main --config config/<experiment>.yaml
```

If the target repo doesn't have this layout yet, scaffold it first (see
`references/scaffold.md`) before writing any pipeline code into it.

## Coding conventions (enforce on every change)

- Every stage exposes `run(cfg: Config) -> Artifacts`; no top-level side effects
  outside `if __name__ == "__main__":`.
- `preprocessing/*.py` imports the specific `eda.*` function whose finding
  motivated the transform, as a docstring-visible dependency — not a comment
  restating it.
- Models live in `pipeline/models/`, one `nn.Module` per file, constructor takes
  only primitives/config, no hidden globals.
- `pipeline/config.py` is the single source of truth for paths/hyperparams/seeds
  — nothing hardcoded elsewhere.
- `ruff check` (and ideally `ruff format --check`) must pass before a change is
  considered complete.
- Each folder's smoke test lives alongside it (e.g. `eda/test_smoke.py`) and
  runs on a tiny sample subset — fast enough to run on every change.

## Experiment tracking (W&B)

`pipeline/log.py` wraps `wandb.init`/`log`/`finish` with a consistent run-naming
scheme: `<stage>-<approach>-<date>`. Every stage's `run()` calls it:

- EDA logs tables/plots
- Feature extraction logs cache stats
- Training logs metrics/curves
- Probing logs a comparison table across pretrained models

Use one W&B project per competition/dataset, grouped by stage, so "which
pretrained model helped" is answerable from the dashboard alone.

## The agent loop — follow this exactly, per stage

1. Write/modify code only inside the matching folder for the current stage.
2. Run `ruff check .` — fix until clean.
3. Run that folder's smoke test — fix until passing.
4. Log results via `pipeline/log.py` (W&B).
5. Write a short entry to `analysis/` (approach, outcome, next step).
6. Only then move to the next stage. Do not skip ahead even if a later stage
   seems obvious — the probing step (stage 4 below) exists specifically to stop
   you from prematurely committing to an expensive approach.

### Stage sequence

1. **EDA** (`eda/`) — distributions, missingness, target leakage checks,
   per-modality sanity (image sizes/corruption, text length/language, audio
   duration, class balance). Each analysis is one function
   `run() -> dict/plots`, logged to W&B as tables/images.
2. **Feature extraction** — tabular: hand-built + encoded features in
   `tabular/`; per-modality: pretrained-encoder embeddings in
   `modalities/<mod>/`, cached to `artifacts/` (e.g. `.npy`/parquet,
   respecting the disk budget — checksum before re-extracting).
3. **Reduction** — PCA/UMAP/feature-selection on top of extracted features,
   still logged, still one function.
4. **Pretrained-model probing** — before any training, run a small battery of
   frozen pretrained models/embeddings on a sample and check linear-probe or
   nearest-neighbor performance per modality. This tells you which modalities
   actually carry signal, cheaply, before you spend compute.
5. **Baseline pipeline** — assemble the cheapest thing that could plausibly
   work: sklearn `Pipeline` (imputers/encoders/scalers) +
   `ColumnTransformer`/`FeatureUnion` combining tabular features and the
   surviving pretrained embeddings from stage 4, feeding a simple model
   (logistic/GBM, or a thin `nn.Module` head).
6. **Iteration** — for each new approach, `analysis/` gets an entry: what was
   tried, what failed, what fixed it, plus the smoke test that proves the code
   path works end-to-end (small sample, few steps, asserts shapes/no-NaN/loss
   decreases). Only swap in trainable `nn.Module` heads or fine-tuning where
   probing showed real headroom.

## Testing / validation

- **Smoke tests**: tiny-sample, few-iteration runs per module confirming
  shapes, no-NaN, and (for models) that loss moves. Run these after every
  change — never the full dataset during iteration.
- **Pipeline-level check**: `python -m pipeline.main --config config/smoke.yaml`
  must run the entire EDA -> features -> baseline chain end-to-end on a small
  held-out sample before any full run is kicked off on Kaggle.

## Execution model — Kaggle git-poll worker

Kaggle kernels have no inbound network access, so orchestration uses a git-based
task queue rather than SSH or a websocket (no server to secure, kernel restarts
are a non-issue since state lives in the repo, `git log` is a free event log,
and 30-60s poll latency doesn't matter for jobs running tens of minutes to hours).
See `references/worker-protocol.md` for the full protocol, the task JSON schema,
and conflict-handling rules before writing or modifying the worker script.

## Roadmap phases (for scaffolding a new pipeline from scratch)

When starting a brand-new repo, work through these phases in order and confirm
with the user before moving past Phase 0:

- **Phase 0 — Scaffold**: directory tree, `config.py`, `log.py`, empty
  `main.py`, ruff config, W&B project, smoke test harness (pytest + tiny
  fixture data) — all before any real pipeline code.
- **Phase 1 — EDA**: one module per data concern, logged to W&B, output a short
  `analysis/00_eda_findings.md`.
- **Phase 2 — Feature extraction & reduction**: tabular builders, per-modality
  embedding extractors with disk caching, PCA/UMAP where useful. Measure cache
  size after each extraction against the disk budget.
- **Phase 3 — Pretrained-model probing**: frozen encoders across modalities on
  a sample, linear-probe/kNN score each, log a comparison table.
- **Phase 4 — Baseline pipeline**: the reference score, recorded in `results/`.
- **Phase 5 — Iteration**: trainable heads/fine-tuning only where probing
  showed headroom.
- **Phase 6 — Packaging**: freeze the winning pipeline as
  `pipeline/pipelines/final.py`, document the Kaggle<->VPS run contract, and
  archive it as the reusable harness for the next competition.

## Reference files

- `references/scaffold.md` — exact file contents/templates for Phase 0
  (`config.py`, `log.py`, `main.py`, ruff config, pytest smoke harness). Read
  this before scaffolding a new repo.
- `references/worker-protocol.md` — full Kaggle git-poll worker protocol, task
  JSON schema, and the reference worker script. Read this before writing or
  editing the worker.
