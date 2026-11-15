import sys
from pathlib import Path

import pytest

from pipeline import verify
from pipeline.verify import WorkerPool

WORKER_SCRIPT = """
import json, sys, time
for line in sys.stdin:
    job = json.loads(line)
    if job.get("mode") == "hang":
        time.sleep(60)
    if job.get("mode") == "exit":
        sys.exit(0)
    reply_id = "other" if job.get("mode") == "wrong_id" else job["id"]
    print(json.dumps({"id": reply_id, "compiled": True, "status": "EXACT", "similarity": 1.0, "note": ""}), flush=True)
"""


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    script = tmp_path / "worker.py"
    script.write_text(WORKER_SCRIPT)
    monkeypatch.setitem(verify.WORKER_COMMANDS, "sly1", [sys.executable, str(script)])
    monkeypatch.setitem(verify.REPOSITORIES, "sly1", tmp_path)
    created = WorkerPool("sly1", 1, timeout_seconds=1.5)
    yield created
    created.close()


def test_a_normal_job_gets_the_worker_reply(pool: WorkerPool) -> None:
    assert pool.submit({"id": "a"})["status"] == "EXACT"


def test_a_hanging_worker_fails_the_job_instead_of_blocking(pool: WorkerPool) -> None:
    result = pool.submit({"id": "a", "mode": "hang"})
    assert (result["status"], result["note"]) == ("ERROR", "pool: timeout")


def test_the_pool_recovers_with_a_fresh_worker_after_a_timeout(pool: WorkerPool) -> None:
    pool.submit({"id": "a", "mode": "hang"})
    assert pool.submit({"id": "b"})["status"] == "EXACT"


def test_a_reply_for_another_job_is_rejected(pool: WorkerPool) -> None:
    assert "another job" in pool.submit({"id": "a", "mode": "wrong_id"})["note"]


def test_a_worker_that_exits_fails_the_job(pool: WorkerPool) -> None:
    assert "worker exited" in pool.submit({"id": "a", "mode": "exit"})["note"]
