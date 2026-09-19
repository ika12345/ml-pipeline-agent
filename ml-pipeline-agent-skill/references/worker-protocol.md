# Kaggle git-poll worker protocol

Chosen over SSH (Kaggle kernels have no inbound network access — a reverse
tunnel is more fragile than a queue for no real benefit) and over a websocket
(git gives auth, heartbeats/reconnects, and state-recovery after Kaggle
restarts a session for free).

## Protocol

```
VPS / Agent                                   Kaggle notebook (worker)
────────────                                  ────────────────────────
write queue/<task_id>.json {status: pending}
git add + commit + push
                                               loop:
                                                 git pull --rebase
                                                 scan queue/*.json for status == "pending"
                                                 if found:
                                                   mark status "running", commit+push
                                                   run task["command"], tee output to current/log.txt
                                                   write current/status.json {returncode, status}
                                                   copy current/* -> results/<task_id>/
                                                   mark task status "done"/"failed", commit+push
                                                 else: sleep POLL_INTERVAL_SEC
git pull -> read results/<task_id>/ and current/status.json
```

## Task file — `queue/<task_id>.json`

```json
{
  "id": "task_0007",
  "command": "python -m pipeline.main --config config/exp3.yaml",
  "created_at": "2026-09-18T10:00:00Z",
  "status": "pending"
}
```

`status` moves `pending -> running -> done | failed`. The VPS agent never edits
a task after creating it — only the worker updates status, so there's a single
writer per field and no merge conflicts on that file.

`current/` always reflects the most recently started task (`log.txt` streamed
stdout/stderr, `status.json` with returncode + status) and gets wiped at the
start of each new task — this is what the VPS-side agent polls first for a
quick "is it alive / did it just finish" check.

`results/<task_id>/` is the permanent archive — `current/`'s contents copied
over once the task finishes, so nothing gets overwritten by the next run.

**Conflict handling**: both sides always `git pull --rebase` before pushing,
and retry the push a few times on rejection.

## Reference worker script — runs inside the Kaggle kernel

```python
"""Kaggle-side git-poll worker. Deliberately dumb — no LLM calls here.
Pulls the repo, looks for a pending task, runs it, pushes results."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path("/kaggle/working/repo")
QUEUE_DIR = REPO_DIR / "queue"
CURRENT_DIR = REPO_DIR / "current"
RESULTS_DIR = REPO_DIR / "results"
POLL_INTERVAL_SEC = 45
PUSH_RETRIES = 5


def run_git(*args: str) -> None:
    subprocess.run(["git", "-C", str(REPO_DIR), *args], check=True)


def git_pull() -> None:
    run_git("pull", "--rebase")


def git_commit_push(message: str) -> None:
    subprocess.run(["git", "-C", str(REPO_DIR), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "commit", "-m", message],
        check=False,  # no-op if nothing changed
    )
    for attempt in range(PUSH_RETRIES):
        try:
            run_git("push")
            return
        except subprocess.CalledProcessError:
            if attempt == PUSH_RETRIES - 1:
                raise
            git_pull()
            time.sleep(2)


def find_pending_task() -> Path | None:
    for path in sorted(QUEUE_DIR.glob("task_*.json")):
        task = json.loads(path.read_text())
        if task.get("status") == "pending":
            return path
    return None


def update_task_status(task_path: Path, status: str) -> dict:
    task = json.loads(task_path.read_text())
    task["status"] = status
    task_path.write_text(json.dumps(task, indent=2))
    return task


def run_task(task: dict) -> int:
    if CURRENT_DIR.exists():
        shutil.rmtree(CURRENT_DIR)
    CURRENT_DIR.mkdir(parents=True)

    log_path = CURRENT_DIR / "log.txt"
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            task["command"],
            shell=True,
            cwd=REPO_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    status = {
        "returncode": proc.returncode,
        "status": "done" if proc.returncode == 0 else "failed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (CURRENT_DIR / "status.json").write_text(json.dumps(status, indent=2))
    return proc.returncode


def archive_current(task_id: str) -> None:
    dest = RESULTS_DIR / task_id
    dest.mkdir(parents=True, exist_ok=True)
    for item in CURRENT_DIR.iterdir():
        shutil.copy2(item, dest / item.name)


def main_loop() -> None:
    while True:
        git_pull()
        task_path = find_pending_task()

        if task_path is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        task = update_task_status(task_path, "running")
        git_commit_push(f"worker: start {task['id']}")

        returncode = run_task(task)
        archive_current(task["id"])

        final_status = "done" if returncode == 0 else "failed"
        update_task_status(task_path, final_status)
        git_commit_push(f"worker: {final_status} {task['id']}")


if __name__ == "__main__":
    main_loop()
```

## Secrets / setup checklist

- [ ] SSH deploy key or PAT with push access, added as a Kaggle Secret and
      injected as `GIT_ASKPASS`/credential helper at kernel start (never
      hardcode in the notebook)
- [ ] `WANDB_API_KEY` added as a Kaggle Secret
- [ ] Same repo cloned on the VPS with push access, for creating tasks and
      reading `results/`
- [ ] VPS-side task creation is a thin wrapper: write `queue/task_NNNN.json`
      with `status: pending`, commit, push — this is where the *agent* (Claude
      Code, deciding what to run next per the roadmap phase) lives; the
      Kaggle worker itself must stay dumb and mechanical
