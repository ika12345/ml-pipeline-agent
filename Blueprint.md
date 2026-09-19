# ML Pipeline Agent — Complete Implementation Blueprint

Status: pre-implementation specification. No pipeline/ML code has been written
yet beyond the Phase 0 scaffold (directory tree, `config.py`, `log.py`,
`main.py`, `pyproject.toml`, `conftest.py`, `config/smoke.yaml`,
`scripts/kaggle_worker.py`, `scripts/enqueue_task.py`, and `pipeline/test_smoke.py`).
This document is self-contained: a coding agent should be able to execute it
without access to the conversation that produced it.

---

## 1. SYSTEM ARCHITECTURE

### Components

| Component | What it is |
|---|---|
| **Local machine / GCP VM** | Where the coding agent (Copilot, or any agent) runs interactively or as a long-lived session. Owns decision-making: what stage to run next, what approach to try, how to react to failures. |
| **GitHub repository** | The single source of truth and the only communication channel between the VM and Kaggle. Holds all code, the task queue, and all results. No other network path exists between VM and Kaggle. |
| **Kaggle kernel (worker)** | A dumb, non-agentic polling loop with GPU access (2×T4-class). Executes exactly the shell command it's given, nothing more. Never makes decisions. |
| **Weights & Biases (W&B)** | Experiment tracking. Every stage's `run()` logs to it. This is where a human (or the agent, via the W&B API) inspects results — it is not part of the task-queue communication path. |

### What runs where

- **Locally / on the GCP VM**: the coding agent itself (Copilot session or equivalent); `scripts/enqueue_task.py` (writes task JSON, commits, pushes); reading `results/<task_id>/` and `current/status.json` after a `git pull`; all "thinking" — deciding what stage/approach to run next, reading logs, diagnosing failures, writing `analysis/*.md` entries; local smoke tests and `ruff check .` before anything is pushed.
- **On Kaggle**: `scripts/kaggle_worker.py` running in a loop for the duration of a kernel session; the actual `python -m pipeline.main --config ...` (or `python -m eda.<name>`, etc.) invocations, which do real GPU/CPU work; writing `current/log.txt`, `current/status.json`, and copying them into `results/<task_id>/`; `git commit`/`git push` of those results.
- **GitHub**: stores everything above. It is the mechanism, not a compute location. Branch protection is unnecessary — treat `main` as the only branch both sides read/write.
- **W&B**: stores every run's metrics, tables, and plots. Neither the VM nor Kaggle needs to read W&B to keep the pipeline itself functioning — it exists for evaluation and comparison, and the coding agent may optionally query it (via `wandb` API) when deciding what to try next.

### Communication model

There are exactly two communication paths, and both are git:

1. **VM → Kaggle**: VM writes `queue/task_NNNN.json` with `status: pending`, commits, pushes. Kaggle's poll loop eventually `git pull --rebase`s and sees it.
2. **Kaggle → VM**: Kaggle updates the task's `status`, writes `current/log.txt` + `current/status.json`, copies them to `results/<task_id>/`, commits, pushes. VM's next `git pull` sees it.

No SSH, no websocket, no shared filesystem, no message broker. This is deliberate (see Section 6) — Kaggle kernels have no inbound network access, so any push-based mechanism (SSH tunnel, websocket server) would require the VM to expose something to Kaggle, which isn't possible; git polling inverts this so Kaggle only ever makes outbound calls to GitHub, which it already permits.

### Role of the coding agent

The coding agent is the only intelligent component in the system. It:
- decides which roadmap phase/stage to work on next,
- writes/edits code inside the correct folder for that stage,
- runs `ruff check .` and the smoke tests locally before anything is queued,
- decides what experiment (`command`) to enqueue for Kaggle,
- polls for the task's completion (`git pull` + check `current/status.json` or `results/<task_id>/`),
- reads `current/log.txt` and any results/metrics to diagnose success or failure,
- writes an `analysis/*.md` entry summarizing what happened,
- decides the next action (retry, adjust approach, move to the next stage, or stop and ask the human).

It never runs training itself — Kaggle does that. It never SSHs into Kaggle or waits synchronously in a blocking sense; it polls between other work or with sleep intervals.

---

## 2. REPOSITORY ARCHITECTURE

```
root/
├── .gitignore
├── README.md
├── pyproject.toml                  # ruff + pytest config, project deps
├── conftest.py                     # shared pytest fixtures (tiny_sample_df)
├── config/
│   ├── smoke.yaml                  # Phase 0: minimal config for pipeline-level smoke check
│   └── <experiment>.yaml           # one per real experiment, created as needed from Phase 4 onward
├── modalities/
│   ├── __init__.py
│   ├── text/                       # tokenizers, pretrained text encoders, embedding extractors
│   ├── image/                      # pretrained vision encoders (CLIP/ResNet/etc.), embedding extractors
│   ├── video/                      # frame sampling + per-frame or video encoder wrappers
│   └── audio/                      # spectrogram/embedding extractors (wav2vec, CLAP, etc.)
├── tabular/                        # tabular-specific transforms, encoders, feature builders
├── eda/                            # one file per analysis; each `python -m eda.<name>`; test_smoke.py alongside
├── analysis/                       # per-approach writeups: what was tried, what broke, what fixed it
├── results/                        # ML metrics/plots/tables archived per task_id (Kaggle worker writes here)
├── artifacts/                      # cached embeddings, fitted transformers, checkpoints (gitignored contents)
├── queue/                          # task_<id>.json files — the task queue (VM writes, worker updates)
├── current/                        # log.txt + status.json for the most recently started task (gitignored contents)
├── scripts/
│   ├── kaggle_worker.py            # runs ON KAGGLE — dumb poll/execute/push loop
│   └── enqueue_task.py             # runs on the VM — writes+pushes a new pending task
└── pipeline/
    ├── __init__.py
    ├── config.py                   # dataclass/YAML-driven config — single source of truth for paths/hyperparams/seed
    ├── log.py                      # W&B init/log/finish wrapper, run-naming convention
    ├── main.py                     # entrypoint: parses config, wires dataset → preprocessing → model → eval
    ├── test_smoke.py               # Phase 0 plumbing smoke tests (config + log naming + fixture — NOT ML logic)
    ├── dataset/                    # Dataset/DataLoader + tabular loaders, train/val/test splits
    ├── preprocessing/              # imports eda/* for rationale; sklearn ColumnTransformer/Pipeline
    ├── models/                     # nn.Module definitions, one class per file
    └── pipelines/                  # end-to-end sklearn Pipeline objects (feature union of modalities)
```

### Infrastructure vs. ML pipeline code

**Infrastructure (build once, rarely touched again):**
`.gitignore`, `pyproject.toml`, `conftest.py`, `config/smoke.yaml`, `scripts/kaggle_worker.py`, `scripts/enqueue_task.py`, `pipeline/config.py`, `pipeline/log.py`, `queue/`, `current/`, `results/` (as a mechanism — its contents are ML output, but the directory's existence and copy-in behavior is infra).

**ML pipeline code (built phase by phase, grows throughout the project):**
everything under `eda/`, `tabular/`, `modalities/*/`, `pipeline/dataset/`, `pipeline/preprocessing/`, `pipeline/models/`, `pipeline/pipelines/`, `pipeline/main.py`'s stage-wiring body (the file exists as infra but its content is completed incrementally as ML code), `analysis/*.md`, per-experiment `config/<experiment>.yaml` files.

### Invocation convention (fixed, never deviate)

```
python -m eda.<analysis_name>
python -m modalities.image.<probe_name>        # same pattern for text/video/audio
python -m tabular.<feature_name>
python -m pipeline.main --config config/<experiment>.yaml
```

---

## 3. PHASE-BY-PHASE IMPLEMENTATION

The template below applies to every phase; Section 5 below fills in the
phase-specific detail. Every phase must satisfy this checklist before being
marked done:

- Objective stated and confirmed against the roadmap (Section 5)
- Files created only inside the correct folder for that stage
- Every stage exposes `run(cfg: Config) -> Artifacts` (or `run() -> dict` for
  simple EDA analyses); no top-level side effects outside `if __name__ == "__main__":`
- `ruff check .` passes clean
- That folder's `test_smoke.py` passes on the tiny fixture sample
- Results logged to W&B via `pipeline/log.py`
- A new `analysis/*.md` entry written: what was tried, what happened, what's next
- Nothing merged/pushed to `main` from the VM side without the smoke tests and
  ruff passing locally first

---

## 4. PHASE 0 — EXACT CHECKLIST

This phase is **complete** except for two external, human-only steps. Current
scaffold state and what's missing:

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Directory tree created, `__init__.py` in every importable package | ✅ Done | `modalities/{text,image,video,audio}`, `tabular`, `eda`, `pipeline/{dataset,preprocessing,models,pipelines}` all present |
| 2 | `pipeline/config.py` — dataclass `Config` with `from_yaml` | ✅ Done | Fields: `data_dir`, `artifacts_dir`, `results_dir`, `task_id`, `stage`, `approach`, `seed`, `metric`, `disk_budget_gb` |
| 3 | `pipeline/log.py` — `wandb_run` context manager + `run_name(cfg)` | ✅ Done | Naming convention: `<stage>-<approach>-<date>` |
| 4 | `pipeline/main.py` — argparse entrypoint | ✅ Done | Loads `Config`, prints it, raises `NotImplementedError` until stages exist. `cfg` is used (logged), no F841 lint issue |
| 5 | `pyproject.toml` — ruff + pytest config | ✅ Done | ruff rules `E, F, I, W`; pytest marker `smoke` registered |
| 6 | `conftest.py` — `tiny_sample_df` fixture | ✅ Done | 20-row synthetic DataFrame with `id`, `text`, `target` |
| 7 | `config/smoke.yaml` | ✅ Done | `task_id: smoke`, `stage: smoke`, `approach: smoke` |
| 8 | `pipeline/test_smoke.py` — Phase 0 plumbing tests | ✅ Done | Tests `Config` defaults, `Config.from_yaml`, `run_name` convention, fixture shape. **Does not** test any EDA/feature/model logic — none exists yet by design |
| 9 | `.gitignore` | ✅ Done | Excludes `artifacts/*`, `data/`, `current/*`, `wandb/`, caches |
| 10 | `scripts/kaggle_worker.py` | ✅ Done | Full poll/run/archive/push loop per Section 6 |
| 11 | `scripts/enqueue_task.py` | ✅ Done | VM-side task creation + push |
| 12 | `ruff check .` passes clean | ⏳ **Needs your local confirmation** | Fixed the one prior finding (F841 on unused `cfg`); no network in the authoring sandbox to run ruff itself |
| 13 | `pytest -k smoke` passes (not just "runs with 0 collected") | ⏳ **Needs your local confirmation** | Should now show 4 passed |
| 14 | W&B project created | ❌ **Human action required** | Nothing to inspect in the repo for this — must be created in the W&B UI/CLI |
| 15 | `WANDB_API_KEY` available in the VM's and Kaggle's environment | ❌ **Human action required** | VM: shell env var or `.env` (gitignored). Kaggle: added as a Kaggle Secret, exported at kernel start |
| 16 | Git remote reachable from both VM and Kaggle with push access | ❌ **Human action required** | SSH deploy key or PAT; Kaggle-side credential must be injected via Kaggle Secrets, never hardcoded in the notebook |

**Definition of done for Phase 0**: items 1–13 are code-complete; items 14–16
are external setup with no code artifact to produce — Phase 1 cannot begin
until all 16 are true, but only 12–16 require action from you before Copilot
proceeds to Phase 1.

---

## 5. PHASE 1 → FINAL PHASE — FULL ROADMAP

### Phase 1 — EDA

- **Objective**: Understand the dataset before writing any transform or model
  code. Every subsequent decision (which modalities carry signal, what to
  impute, what to encode) must trace back to a finding recorded here.
- **Files**: one file per data concern under `eda/`, e.g. `eda/missingness.py`,
  `eda/target_distribution.py`, `eda/leakage_check.py`, `eda/text_sanity.py`,
  `eda/image_sanity.py` (only for modalities actually present in the dataset —
  see "what the agent decides autonomously" below).
- **Functions**: each file exposes exactly one `run() -> dict` (or returns a
  dict of summary stats + a list of plot objects/paths). No classes needed
  here unless a concern genuinely needs stateful analysis across calls.
- **Inputs**: raw dataset path(s) from `Config.data_dir`.
- **Outputs**: a dict of findings returned to the caller; plots/tables logged
  to W&B as a side effect of `run()`; a written summary in `analysis/00_eda_findings.md`.
- **Agent decides autonomously**: which modalities are present in this
  particular dataset (from the modality manifest given at project start) and
  therefore which `eda/*.py` files to write at all — e.g., a pure tabular
  competition needs no `eda/image_sanity.py`; the specific missingness/leakage
  checks worth running for this dataset's schema; how to visualize each
  finding.
- **Logged**: summary tables and any distribution plots, via `pipeline/log.py`,
  under a W&B run named `eda-<concern>-<date>`.
- **Artifacts produced**: none persisted beyond W&B tables/plots and the
  `analysis/00_eda_findings.md` write-up — EDA should not write to `artifacts/`.
- **Tests**: `eda/test_smoke.py` — each `run()` executes on `tiny_sample_df`
  (or an equivalent tiny fixture for non-tabular modalities) and asserts the
  returned dict has the expected keys and no unhandled exceptions.
- **Definition of done**: every planned `eda/*.py` module has a passing smoke
  test, `ruff check .` passes, `analysis/00_eda_findings.md` exists and names
  concrete findings that will shape Phase 2 decisions.

### Phase 2 — Feature extraction & reduction

- **Objective**: Turn raw data into numeric features, cheaply and cacheably,
  for every modality identified as present.
- **Files**: `tabular/<feature_name>.py` for hand-built/encoded tabular
  features; `modalities/<mod>/<extractor_name>.py` for pretrained-encoder
  embedding extraction per modality; optionally `tabular/reduction.py` or
  `modalities/<mod>/reduction.py` for PCA/UMAP on top of extracted features.
- **Functions**: each exposes `run(cfg: Config) -> Artifacts`, where
  `Artifacts` at minimum carries a path to the cached feature/embedding file
  and metadata (shape, dtype, cache checksum).
- **Inputs**: raw dataset (tabular) or raw modality data + a pretrained
  encoder identifier (image/text/video/audio).
- **Outputs**: cached `.npy`/parquet files under `artifacts/`, keyed so
  re-running with unchanged inputs is a checksum-verified no-op rather than
  a re-extraction.
- **Agent decides autonomously**: which pretrained encoders to try per
  modality (this is provisional — final selection happens in Phase 3); which
  hand-built tabular features are worth encoding given the Phase 1 findings;
  when the disk budget (`cfg.disk_budget_gb`) is close enough to force pruning
  cached artifacts before continuing.
- **Logged**: cache stats (size on disk, extraction time, row/embedding count)
  via `pipeline/log.py`, run named `features-<modality_or_feature_name>-<date>`.
- **Artifacts produced**: the cached feature/embedding files themselves,
  under `artifacts/<modality_or_feature_name>/`.
- **Tests**: `tabular/test_smoke.py` and `modalities/<mod>/test_smoke.py`,
  each running extraction on the tiny fixture sample and asserting output
  shape, dtype, and no-NaN.
- **Definition of done**: every feature/embedding source has a passing smoke
  test and a cache-stats W&B log entry; disk usage measured and under budget;
  an `analysis/` entry summarizing what was extracted and any surprises
  (e.g., an encoder that failed on a subset of the data).

### Phase 3 — Pretrained-model probing

- **Objective**: Before training anything, determine cheaply which
  modalities/encoders actually carry predictive signal.
- **Files**: `modalities/<mod>/probe.py` per modality (linear-probe or
  nearest-neighbor evaluation harness), plus a comparison aggregator, e.g.
  `pipeline/preprocessing/probe_comparison.py`.
- **Functions**: `run(cfg: Config) -> Artifacts` per probe, returning a score
  (e.g. validation accuracy/AUC/R² of a simple linear model fit on the cached
  embeddings from Phase 2) plus the identifying metadata (encoder name,
  modality).
- **Inputs**: cached embeddings from Phase 2's `artifacts/`, plus the target
  column.
- **Outputs**: a per-modality, per-encoder score table.
- **Agent decides autonomously**: which encoders' probe scores clear the bar
  to advance to Phase 4 (this is the single most consequential autonomous
  decision in the pipeline — it determines what the baseline is built from);
  if a modality's best probe score is at or near a naive baseline (e.g.
  mean-prediction for regression, majority-class for classification), the
  agent should drop that modality rather than force it into the baseline.
- **Logged**: one comparison table across all probed encoders per modality,
  W&B run named `probe-comparison-<date>`.
- **Artifacts produced**: none new — probing reuses Phase 2's cached
  embeddings; the comparison table itself lives in W&B and `results/`.
- **Tests**: `modalities/<mod>/test_smoke.py` extended (or a dedicated
  `probe_test_smoke.py`) verifying the probe harness runs end-to-end on the
  tiny fixture and returns a well-formed score dict.
- **Definition of done**: a comparison table exists (in W&B and mirrored to
  `results/phase3_probe_comparison.csv` or similar), and an `analysis/` entry
  states explicitly which modalities/encoders were kept or dropped and why.

### Phase 4 — Baseline pipeline

- **Objective**: Assemble the cheapest plausible end-to-end model as the
  reference score everything else is measured against.
- **Files**: `pipeline/preprocessing/build_column_transformer.py` (or
  similarly named), `pipeline/pipelines/baseline.py` (the full sklearn
  `Pipeline` object), `pipeline/models/baseline_head.py` only if a thin
  `nn.Module` head is chosen over logistic/GBM.
- **Functions**: `build_pipeline(cfg: Config) -> sklearn.pipeline.Pipeline`
  in `pipeline/pipelines/baseline.py`; `run(cfg: Config) -> Artifacts` in
  `pipeline/main.py`'s wiring, which fits/evaluates this pipeline and reports
  the metric from `cfg.metric`.
- **Inputs**: the surviving Phase 3 embeddings + Phase 2 tabular features.
- **Outputs**: a fitted pipeline object (pickled), predictions on a held-out
  split, the metric value.
- **Agent decides autonomously**: the exact `ColumnTransformer`/`FeatureUnion`
  wiring combining tabular + surviving embeddings; the specific simple model
  (logistic regression / GBM / thin head) to lead with, based on task type
  (`cfg.metric`) and Phase 3 findings.
- **Logged**: the metric, a prediction-vs-actual plot (or confusion matrix for
  classification), via `pipeline/log.py`, run named `baseline-<approach>-<date>`.
- **Artifacts produced**: `artifacts/baseline_pipeline.pkl`,
  `results/baseline/metrics.json`.
- **Tests**: `pipeline/pipelines/test_smoke.py` — fits and scores the baseline
  on the tiny fixture, asserting the metric is computed and finite (not NaN).
- **Definition of done**: `results/baseline/metrics.json` exists with a real
  number, logged to W&B, and an `analysis/` entry records it as *the*
  reference score for all Phase 5 comparisons.

### Phase 5 — Iteration

- **Objective**: Improve on the Phase 4 baseline, escalating to trainable
  components only where Phase 3 probing showed headroom.
- **Files**: new files under `pipeline/models/` (one `nn.Module` per file) and
  updated/new `pipeline/pipelines/<approach>.py` per experiment; this is the
  only phase expected to run for multiple iterations, each getting its own
  `config/<experiment>.yaml`.
- **Functions**: each new pipeline variant exposes the same
  `build_pipeline(cfg) -> Pipeline`-or-equivalent shape as Phase 4, so
  `pipeline/main.py`'s wiring doesn't need to branch on which experiment is
  running — only the config selects it.
- **Inputs**: same as Phase 4, plus any newly extracted features specific to
  the experiment.
- **Outputs**: a fitted model/pipeline, predictions, the metric, a delta
  against the Phase 4 baseline.
- **Agent decides autonomously**: which experiment to try next, given the
  previous experiment's result (this is the core of the "coding agent loop"
  in Section 7); when to stop iterating (diminishing returns, budget
  exhausted, or a human-set experiment cap reached — see Section 10 for the
  human-approval boundary here).
- **Logged**: metric + delta-vs-baseline per experiment, run named
  `iterate-<approach>-<date>`.
- **Artifacts produced**: `artifacts/<experiment_name>_pipeline.pkl` (or
  checkpoint) per kept experiment; failed/discarded experiments' artifacts
  may be deleted to respect the disk budget, but their `analysis/` entry and
  W&B log stay.
- **Tests**: each new `pipeline/models/*.py` gets a smoke test asserting
  forward-pass shape correctness and (for anything trained) that loss
  decreases over a handful of steps on the tiny fixture.
- **Definition of done for the phase overall**: at least one experiment beats
  the Phase 4 baseline on the held-out metric, or the agent has exhausted the
  agreed experiment budget and documented why nothing improved.

### Phase 6 — Packaging

- **Objective**: Freeze the winning pipeline as the reusable deliverable.
- **Files**: `pipeline/pipelines/final.py` (the single frozen winning
  pipeline, referencing whichever Phase 5 experiment won), a top-level
  `RUNBOOK.md` documenting the Kaggle↔VM run contract for reuse on the next
  competition.
- **Functions**: `build_pipeline(cfg) -> Pipeline` (final, no branching),
  `run(cfg) -> Artifacts` producing the final submission/prediction file.
- **Inputs**: the winning experiment's config and cached artifacts.
- **Outputs**: final predictions/submission file, `RUNBOOK.md`.
- **Agent decides autonomously**: nothing new — this phase is purely
  consolidation of a decision already made in Phase 5.
- **Logged**: final metric, run named `final-<date>`.
- **Artifacts produced**: `results/final/predictions.csv` (or competition's
  required submission format), `artifacts/final_pipeline.pkl`.
- **Tests**: `pipeline/pipelines/test_smoke.py` extended to cover
  `final.py` specifically.
- **Definition of done**: `pipeline/pipelines/final.py` runs end-to-end via
  `python -m pipeline.main --config config/final.yaml`, `RUNBOOK.md` exists,
  and the harness (folder structure, worker, queue protocol) is confirmed
  reusable as-is for the next competition with only `config/` and dataset
  paths changing.

---

## 6. KAGGLE WORKER

### Task creation (VM side)

`scripts/enqueue_task.py <command>`:
1. Computes the next `task_NNNN` id by scanning `queue/task_*.json` for the
   highest existing number and incrementing.
2. Writes `queue/task_NNNN.json` with `status: "pending"`.
3. `git add` that one file, `git commit`, `git pull --rebase`, `git push`.

### Task JSON format

```json
{
  "id": "task_0007",
  "command": "python -m pipeline.main --config config/exp3.yaml",
  "created_at": "2026-09-18T10:00:00Z",
  "status": "pending"
}
```

`command` is a full shell command, executed with `cwd` set to the repo root
inside the Kaggle kernel — it can be any of the four invocation forms from
Section 2 (`eda`, `modalities.<mod>`, `tabular`, or `pipeline.main`).

### Lifecycle: pending → running → done/failed

- **pending**: written by the VM, untouched by anything else until a worker
  picks it up.
- **running**: set by the worker the moment it claims the task, *before*
  executing the command, and immediately committed+pushed — this is what
  gives the VM visibility that work has started, even mid-run.
- **done / failed**: set by the worker after the command exits, based on
  return code (`0` → `done`, anything else → `failed`), committed+pushed
  together with the archived results.

The VM never edits a task's status after creating it. The worker is the only
writer of `status` past the initial `pending`. This single-writer-per-field
rule is what avoids merge conflicts on the task file itself.

### How commands are executed

`scripts/kaggle_worker.py`'s `run_task()`:
1. Wipes and recreates `current/`.
2. Runs `task["command"]` via `subprocess.run(..., shell=True, cwd=REPO_DIR)`,
   redirecting combined stdout+stderr to `current/log.txt`.
3. Writes `current/status.json` with `{"returncode": ..., "status": "done"|"failed", "finished_at": ...}`.

### How results return to GitHub

`archive_current(task_id)` copies everything in `current/` into
`results/<task_id>/` (a permanent, never-overwritten archive), then the main
loop commits and pushes both the task status update and the new
`results/<task_id>/` contents in one `git commit`.

### How the coding agent reads those results

After enqueueing, the agent's poll loop (see Section 7) does `git pull` and
checks, in order:
1. `queue/task_NNNN.json`'s `status` field — cheapest check, tells it
   pending/running/done/failed without opening any large file.
2. `current/status.json` — if this task is the most recent one run, this is
   the fastest "did it just finish" signal.
3. `results/task_NNNN/log.txt` and any metric files the command itself wrote
   (e.g. `results/task_NNNN/metrics.json` if the pipeline stage writes one) —
   read once `status` is `done` or `failed`, for the actual diagnostic content.

### Failure / retry behavior

- **Git push conflicts** (both sides pushing near-simultaneously): both
  `scripts/enqueue_task.py` and the worker's `git_commit_push()` retry up to
  5 times, doing `git pull --rebase` between attempts. Because there are only
  two writers and they touch disjoint files in the common case (VM writes new
  task files; worker writes status+results of existing ones), true conflicts
  are rare, but the retry loop makes it safe regardless.
- **Task command failure** (non-zero exit): recorded as `status: failed`
  with the full log preserved in `results/<task_id>/log.txt`. The worker does
  **not** retry a failed command automatically — retry-with-a-different-approach
  is a decision, and decisions belong to the agent (Section 10), not the
  dumb worker. The agent reads the failure, diagnoses it from the log, and
  either enqueues a corrected command as a new task or asks the human.
- **Worker crash mid-task** (kernel restarted, session killed): the task is
  left in `status: "running"` with no corresponding `done`/`failed` update.
  The agent should treat a task stuck in `running` for longer than a
  reasonable multiple of its expected runtime as effectively failed, and
  either re-enqueue it as a new task or flag it for human attention — the
  worker itself has no mechanism to detect or recover from its own crash.

---

## 7. CODING AGENT LOOP — PSEUDOCODE

```
loop:
    inspect_repo_state()
        # git pull; read queue/*.json statuses; read latest results/;
        # read analysis/*.md for what's already been tried

    if no phase currently in progress:
        phase = decide_next_phase(roadmap_position, past_results)
        # e.g. "EDA is done and analysis/00_eda_findings.md exists
        #       with no open questions -> move to Phase 2"

    plan = plan_stage_work(phase, findings_so_far)
        # e.g. "write eda/missingness.py and eda/target_distribution.py
        #       because the modality manifest says this is a tabular-only task"

    implement(plan)
        # write/modify files ONLY inside the folder(s) the plan named

    lint_result = run("ruff check .")
    if lint_result.failed:
        fix_lint_issues()
        retry lint

    smoke_result = run("pytest -k smoke")
    if smoke_result.failed:
        diagnose_and_fix()
        retry smoke

    if stage_needs_kaggle_compute(plan):
        # local sklearn/eda work on tiny samples can often run locally;
        # only enqueue to Kaggle for GPU-bound or full-dataset work
        task_id = enqueue_kaggle_task(command_for(plan))
        wait_for_task(task_id)
            # git pull on an interval; check queue/<task_id>.json status;
            # this is a polling wait, not a blocking network call
        result = read_results(task_id)
            # results/<task_id>/log.txt, any metrics file, current/status.json
    else:
        result = run_locally(plan)

    analysis_entry = summarize(plan, result)
    write_file(f"analysis/{next_seq}_{short_name}.md", analysis_entry)

    decision = decide_next_action(result, phase, roadmap_position)
        # options: advance to next stage/phase, retry with adjusted approach,
        # stop and request human approval (see Section 10 for what forces this)

    if decision == "advance":
        phase = next_phase(phase)
    elif decision == "retry":
        continue  # loop again within the same phase with an adjusted plan
    elif decision == "needs_human":
        pause_and_report_to_human(result, analysis_entry)
        break
```

Key properties this pseudocode preserves from the original spec: the fixed
per-stage sequence (write code → lint → smoke test → log → analysis entry →
only then advance) is never skipped or reordered; Kaggle is only ever reached
through the queue, never directly; the agent never marks a stage done without
a passing smoke test and a logged result.

---

## 8. GIT WORKFLOW

### Commits/pushes on the agent (VM) side

1. Local code changes (new/edited files inside `eda/`, `tabular/`,
   `modalities/`, `pipeline/`) are committed normally as work progresses —
   these are ordinary source commits, no special protocol.
2. `scripts/enqueue_task.py` makes a dedicated, single-file commit
   (`queue/task_NNNN.json` only) with message `enqueue task_NNNN` —
   kept separate from source commits so the task-creation event is
   unambiguous in `git log`.
3. Before any push, `git pull --rebase` first; on push rejection, retry after
   another pull (see Section 6's retry behavior — same mechanism, both sides
   share it).

### Commits/pushes on the Kaggle (worker) side

1. On claiming a task: commit+push *only* the updated `queue/task_NNNN.json`
   (`status: running`), message `worker: start task_NNNN`.
2. On completion: commit+push the updated `queue/task_NNNN.json`
   (`status: done`/`failed`) together with the new `results/task_NNNN/`
   directory, message `worker: done task_NNNN` or `worker: failed task_NNNN`.
3. The worker never touches source code (`eda/`, `pipeline/`, etc.) — it only
   ever writes to `queue/`, `current/`, and `results/`.

### How conflicts are avoided

- **Disjoint write ownership**: the VM writes source code and new task files;
  the worker writes task *status* and `results/`. In the common case these
  are different files entirely, so there's nothing to merge.
- **Single writer per field**: within a task JSON, only the VM ever sets it
  to `pending` (at creation), and only the worker ever transitions it
  afterward. Neither side edits a field the other owns.
- **Rebase-before-push, always**: both sides pull-rebase immediately before
  every push attempt, so the working copy is never stale when a push is
  attempted.
- **Retry with backoff on rejection**: up to 5 attempts, re-pulling between
  each, on both sides — handles the rare case of a near-simultaneous push.

---

## 9. WEIGHTS & BIASES (W&B)

### Naming convention

Every run is named `<stage>-<approach>-<date>` via `pipeline/log.py`'s
`run_name(cfg)`, e.g. `eda-missingness-2026-09-19`, `probe-comparison-2026-09-20`,
`baseline-logistic-2026-09-21`, `iterate-clip-embed-fusion-2026-09-22`.

### Project organization

One W&B **project** per competition/dataset. Within it, group runs by stage
(EDA, features, probe, baseline, iterate, final) — either via a W&B "group"
field set to the stage name, or by relying on the `<stage>-` prefix in the run
name for filtering on the dashboard. The goal stated in the original spec is
that "which pretrained model helped" must be answerable from the dashboard
alone — this means the probe-comparison table (Phase 3) and every iteration's
metric-vs-baseline delta (Phase 5) must be logged as first-class W&B tables,
not just buried in console log text.

### What gets logged, per stage

- **EDA**: summary tables (missingness %, class balance, leakage-check
  results) and distribution plots, as W&B Tables/Images.
- **Feature extraction**: cache stats — extraction time, output shape, disk
  size — as scalar/summary metrics.
- **Probing**: one comparison table per modality across all tried encoders,
  each row a (encoder, probe_type, score) tuple.
- **Baseline / Iteration**: the task metric (`cfg.metric`), a
  prediction-vs-actual plot or confusion matrix, and — for iteration
  specifically — the delta against the Phase 4 baseline score, logged
  explicitly as its own field so it's sortable/filterable on the dashboard.
- **Final**: the final metric plus a pointer (config name/artifact path) to
  exactly which Phase 5 experiment was frozen.

---

## 10. AUTONOMY — WHO DECIDES WHAT

### Decisions the coding agent makes autonomously

- Which `eda/*.py` modules to write, based on the dataset's modality manifest
- Which hand-built tabular features to encode, and which pretrained encoders
  to try per modality in Phase 2
- Which encoders/modalities survive Phase 3 probing to reach the baseline
- The exact `ColumnTransformer`/`FeatureUnion` wiring and the specific simple
  model choice for Phase 4
- What experiment to try next during Phase 5 iteration, based on the
  previous result
- Whether a lint/smoke-test failure is fixable inline vs. needs a design
  change
- How to interpret a Kaggle task's failure log and whether to re-enqueue a
  corrected command

### Deterministic scripts (no decision-making, same output for same input)

- `scripts/kaggle_worker.py` — polls, executes exactly the given command,
  archives, pushes; makes zero judgment calls about what to run or whether
  a result is "good"
- `scripts/enqueue_task.py` — writes exactly the task it's given
- `pipeline/config.py`'s `Config.from_yaml` — pure parsing
- `pipeline/log.py`'s `wandb_run`/`run_name` — pure formatting/wrapping
- Every stage's smoke test — fixed assertions against a fixed tiny fixture

### Things Kaggle executes (never decides)

- The actual `python -m ...` command from a task's `command` field — whatever
  it is, verbatim
- Nothing else. Kaggle's kernel never chooses a hyperparameter, never
  retries with a different approach, never inspects whether a result is good

### Things that require human approval before proceeding

- Creating the W&B project and provisioning `WANDB_API_KEY` (Phase 0,
  external — no agent action can substitute for this)
- Provisioning git push credentials for both VM and Kaggle (Phase 0,
  external)
- Advancing past Phase 3's probing decision when the result is ambiguous —
  i.e., if no modality clears a plausible signal bar and the agent is
  considering proceeding with a baseline that's barely better than a naive
  prediction, it should report this and ask before spending Phase 4/5 effort
  on a likely-weak foundation
- Ending Phase 5 iteration — the agent should propose stopping (diminishing
  returns or budget exhausted) but a human should confirm before Phase 6
  packaging freezes a specific experiment as "final"
- Any change to the fixed repository structure (Section 2) or the task
  JSON/lifecycle protocol (Section 6) — these are the contract the whole
  system depends on and shouldn't drift without an explicit decision to
  change the architecture itself

---

## 11. COPILOT IMPLEMENTATION INSTRUCTIONS

Give these to Copilot **one at a time**, in order. Do not paste the next
prompt until the current phase's Definition of Done is met and confirmed
locally (`ruff check .` and `pytest -k smoke` both passing).

### Prompt for Phase 0 (verification only — scaffold already exists)

> Inspect the existing repository against `ML_PIPELINE_BLUEPRINT.md` Sections
> 2 and 4. Do not create any new ML/EDA/feature code. Only: (a) confirm every
> directory and file listed in Section 2's tree exists with an `__init__.py`
> in every importable package; (b) run `ruff check .` and fix anything it
> reports without changing the intended architecture (e.g. an unused variable
> should be used meaningfully, not suppressed with `# noqa`); (c) run
> `pytest -k smoke` and confirm the existing `pipeline/test_smoke.py` tests
> pass — do not add new tests in this prompt. Report the Phase 0 checklist
> from Section 4 item-by-item, and explicitly flag `WANDB_API_KEY`, the W&B
> project, and git push credentials as external steps you cannot complete.
> **Definition of done**: `ruff check .` exits clean, `pytest -k smoke` shows
> all tests passing, and you've reported the checklist without touching
> `eda/`, `tabular/`, `modalities/`, or `pipeline/pipelines/`.

### Prompt for Phase 1 (EDA)

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 1 entry. The dataset's
> modality manifest is: [FILL IN — e.g. "tabular + text (product description)
> + image (product photo)"]. Implement only `eda/` modules for the modalities
> actually listed — do not write `eda/audio_sanity.py` or similar for a
> modality that isn't present. For each concern (missingness, target
> distribution, leakage check, and one sanity check per present modality),
> create `eda/<concern>.py` exposing a single `run() -> dict` function with
> no top-level side effects, and `eda/test_smoke.py` covering all of them
> against `tiny_sample_df` (extend the fixture in `conftest.py` if a
> non-tabular modality needs its own tiny sample — keep it small and
> synthetic, not real data). Do not implement any feature extraction,
> dimensionality reduction, or model code — that is Phase 2 and later. Run
> `ruff check .` and `pytest -k smoke` and fix until both pass. Write
> `analysis/00_eda_findings.md` summarizing concrete findings that will
> inform Phase 2 (e.g. which columns/modalities look weak). **Definition of
> done**: every `eda/*.py` module has a passing smoke test, ruff is clean,
> and `analysis/00_eda_findings.md` exists with specific, actionable findings
> — not placeholder text.

### Prompt for Phase 2 (feature extraction & reduction)

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 2 entry and
> `analysis/00_eda_findings.md` from Phase 1. Implement tabular feature
> builders under `tabular/` and, for each modality confirmed present in
> Phase 1, an embedding extractor under `modalities/<mod>/`, each exposing
> `run(cfg: Config) -> Artifacts` and caching output to `artifacts/` as
> `.npy`/parquet with a checksum check to skip redundant re-extraction. Do
> not implement pretrained-model probing, baseline pipeline assembly, or any
> training loop — those are Phase 3 and later. Add `test_smoke.py` beside
> each new folder's code, running extraction on the tiny fixture and
> asserting output shape/dtype/no-NaN. Run `ruff check .` and
> `pytest -k smoke`. Log cache stats (size, time, shape) via
> `pipeline/log.py`. Write an `analysis/` entry on what was extracted and any
> extraction failures. **Definition of done**: every feature/embedding source
> has a passing smoke test, cache files exist under `artifacts/` and stay
> within `cfg.disk_budget_gb`, and the `analysis/` entry is written.

### Prompt for Phase 3 (pretrained-model probing)

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 3 entry. Using the cached
> embeddings from Phase 2 (do not re-extract), implement
> `modalities/<mod>/probe.py` per present modality — a linear-probe or
> nearest-neighbor evaluation against the target column — plus a comparison
> aggregator that produces a single score table across all probed
> encoders/modalities. Do not implement the baseline `ColumnTransformer` or
> any final model — that is Phase 4. Add smoke tests verifying the probe
> harness runs end-to-end on the tiny fixture and returns a well-formed score
> dict. Run `ruff check .` and `pytest -k smoke`. Log the comparison table to
> W&B as `probe-comparison-<date>` and mirror it to
> `results/phase3_probe_comparison.csv`. Write an `analysis/` entry stating
> explicitly which modalities/encoders you are keeping or dropping for
> Phase 4, and why. If no modality clears a plausible signal bar above a
> naive baseline, stop and flag this for human review rather than proceeding.
> **Definition of done**: the comparison table exists in both W&B and
> `results/`, and the `analysis/` entry names a clear keep/drop decision per
> modality.

### Prompt for Phase 4 (baseline pipeline)

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 4 entry and the Phase 3
> `analysis/` entry's keep/drop decision. Implement
> `pipeline/preprocessing/build_column_transformer.py` and
> `pipeline/pipelines/baseline.py` (a `build_pipeline(cfg) -> Pipeline`
> combining tabular features with only the surviving Phase 3 embeddings),
> feeding a simple model appropriate to `cfg.metric` (logistic/GBM, or a thin
> `nn.Module` head only if justified). Wire `pipeline/main.py` to actually
> call this pipeline for `stage: baseline`, replacing the current
> `NotImplementedError` for that case only — do not remove the
> `NotImplementedError` for stages that still don't exist. Do not implement
> any iteration/experimentation logic — that is Phase 5. Add
> `pipeline/pipelines/test_smoke.py` fitting and scoring the baseline on the
> tiny fixture, asserting the metric computes and is finite. Run
> `ruff check .` and `pytest -k smoke`. Log the metric and a
> prediction-vs-actual plot to W&B as `baseline-<approach>-<date>`, and write
> `results/baseline/metrics.json`. Write an `analysis/` entry recording this
> as the reference score. **Definition of done**:
> `python -m pipeline.main --config config/baseline.yaml` (create this
> config) runs end-to-end via the git-poll worker on Kaggle for the full
> dataset, `results/baseline/metrics.json` has a real number, and it's
> logged to W&B.

### Prompt for Phase 5 (iteration) — repeat this prompt per experiment

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 5 entry and the current
> Phase 4 (or latest Phase 5) baseline score from `results/`. Propose exactly
> one next experiment (e.g. a new feature, a swapped encoder, a trainable
> head) with a one-paragraph rationale tied to a specific Phase 1/3 finding,
> and confirm the approach before implementing. Once confirmed: implement it
> as a new `pipeline/models/*.py` (if a trainable component) and/or
> `pipeline/pipelines/<experiment_name>.py`, using its own
> `config/<experiment_name>.yaml`. Do not modify `pipeline/pipelines/baseline.py`
> — the baseline stays as the fixed reference. Add a smoke test for any new
> model file (forward-pass shape check, and loss-decreases-over-a-few-steps
> if trained). Run `ruff check .` and `pytest -k smoke`. Enqueue the real run
> via `scripts/enqueue_task.py` and, once results land, log the metric and
> its delta against the baseline to W&B as `iterate-<experiment_name>-<date>`.
> Write an `analysis/` entry: what was tried, the result, whether it beat the
> baseline, and what you'd try next. **Definition of done for this single
> experiment**: it ran to completion on Kaggle (done or failed, both are
> valid outcomes to record), the delta-vs-baseline is logged, and the
> `analysis/` entry is written. Stop and ask before starting another
> experiment iteration or before deciding Phase 5 is finished.

### Prompt for Phase 6 (packaging)

> Read `ML_PIPELINE_BLUEPRINT.md` Section 5's Phase 6 entry and every
> `analysis/` entry from Phase 5. Confirm with me which experiment won before
> writing any code. Once confirmed: implement `pipeline/pipelines/final.py`
> as a `build_pipeline(cfg) -> Pipeline` with no branching, referencing only
> the winning experiment's approach, and wire `pipeline/main.py`'s
> `stage: final` case to call it. Do not modify any other experiment's files.
> Add/extend `pipeline/pipelines/test_smoke.py` to cover `final.py`. Run
> `ruff check .` and `pytest -k smoke`. Write `RUNBOOK.md` documenting the
> Kaggle↔VM run contract (queue protocol, secrets needed, how to reuse this
> repo for the next competition). Run the final pipeline end-to-end via the
> worker and produce `results/final/predictions.csv`. **Definition of done**:
> `python -m pipeline.main --config config/final.yaml` produces the final
> predictions file end-to-end, `RUNBOOK.md` exists, and I've confirmed the
> harness is reusable as-is for a future competition.
