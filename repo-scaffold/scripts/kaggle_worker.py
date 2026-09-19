"""Kaggle-side git-poll worker. Deliberately dumb — no LLM calls here.
Pulls the repo, looks for a pending task, runs it, pushes results.

Run this as the last cell of a Kaggle notebook (internet + GPU enabled),
with git credentials (deploy key / PAT) configured via Kaggle Secrets before
this cell runs. It loops forever, so run it in a session with a long enough
time budget for whatever queue of tasks you expect to process.
"""
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
    print(f"[worker] starting, polling every {POLL_INTERVAL_SEC}s")
    while True:
        git_pull()
        task_path = find_pending_task()

        if task_path is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        task = update_task_status(task_path, "running")
        print(f"[worker] running {task['id']}: {task['command']}")
        git_commit_push(f"worker: start {task['id']}")

        returncode = run_task(task)
        archive_current(task["id"])

        final_status = "done" if returncode == 0 else "failed"
        update_task_status(task_path, final_status)
        git_commit_push(f"worker: {final_status} {task['id']}")
        print(f"[worker] {task['id']} -> {final_status}")


if __name__ == "__main__":
    main_loop()
