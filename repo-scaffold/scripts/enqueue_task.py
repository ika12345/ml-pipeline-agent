"""VPS/local-side helper: write a new pending task and push it for the
Kaggle worker to pick up. This is the boundary where the *agent* (deciding
what to run next per the roadmap phase) hands off to the dumb worker.

Usage:
    python scripts/enqueue_task.py "python -m pipeline.main --config config/exp3.yaml"
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
QUEUE_DIR = REPO_DIR / "queue"


def next_task_id() -> str:
    existing = sorted(QUEUE_DIR.glob("task_*.json"))
    if not existing:
        return "task_0001"
    last_num = max(int(p.stem.split("_")[1]) for p in existing)
    return f"task_{last_num + 1:04d}"


def enqueue(command: str) -> str:
    task_id = next_task_id()
    task = {
        "id": task_id,
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }
    task_path = QUEUE_DIR / f"{task_id}.json"
    task_path.write_text(json.dumps(task, indent=2))

    subprocess.run(["git", "-C", str(REPO_DIR), "add", str(task_path)], check=True)
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "commit", "-m", f"enqueue {task_id}"],
        check=True,
    )
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--rebase"], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "push"], check=True)
    return task_id


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/enqueue_task.py '<command>'")
        sys.exit(1)
    task_id = enqueue(sys.argv[1])
    print(f"enqueued {task_id}")
