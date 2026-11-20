"""Harness side of verification: splice a candidate function into its original source, run persistent workers.

A candidate is C++ text for ONE function. The original source file with that function replaced is compiled by the
repo's own toolchain and compared with retail by the repo's own compare code (see eval/workers).
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.cpp_parse import find_function_definitions
from pipeline.records import scan_rules
from pipeline.sly1 import load_reverse_names, to_address_names

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORIES = {
    "sly1": PROJECT_ROOT / "data" / "raw" / "sly1-decomp",
    "sly2": PROJECT_ROOT / "data" / "raw" / "sly2-decomp",
    "sly3": PROJECT_ROOT / "data" / "raw" / "sly3-decomp",
}
WORKER_COMMANDS = {
    "sly1": [
        str(PROJECT_ROOT / "data" / "raw" / "sly1-decomp" / "venv" / "bin" / "python"),
        str(PROJECT_ROOT / "eval" / "workers" / "sly1_worker.py"),
    ],
    "sly2": [
        str(PROJECT_ROOT / "data" / "raw" / "sly2-decomp" / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "eval" / "workers" / "sly2_worker.py"),
    ],
    "sly3": [
        str(PROJECT_ROOT / "data" / "raw" / "sly3-decomp" / "venv" / "bin" / "python"),
        str(PROJECT_ROOT / "eval" / "workers" / "sly3_worker.py"),
    ],
}
# Games laid out like sly1-decomp: func_XXXXXXXX symbols, readable names in config/names/, one source file per job.
ADDRESS_SYMBOL_GAMES = frozenset({"sly1", "sly3"})
HUNGARIAN_PATTERN = r"\b(?:psw|plo|grf|pso|ppo)[A-Z]\w*"


@dataclass
class Candidate:
    """One generated function for a record."""

    game: str
    function_id: str
    address: int
    size_bytes: int
    definition_name: str  # the function's name as written in the original source
    definition_ordinal: int  # among same-named definitions (overloads)
    source_file: str  # relative to the repository (src/main/x.cpp for sly1/sly3, src/<module>/x.cpp for sly2)
    text: str


def splice_candidate(
    game: str,
    source_file: str,
    function_id: str,
    definition_name: str,
    definition_ordinal: int,
    candidate_text: str,
    repository: Path | None = None,
) -> str | None:
    """The original file with the target function replaced by the candidate; None when the target is not found."""
    repository = repository or REPOSITORIES[game]
    source_text = (repository / source_file).read_text(errors="replace")
    reverse_names = load_reverse_names(repository) if game in ADDRESS_SYMBOL_GAMES else {}
    definitions = find_function_definitions(source_text)
    matching = (
        [definition for definition in definitions if definition.name == definition_name] if definition_name else []
    )
    if not matching:
        matching = [
            definition
            for definition in definitions
            if function_id
            in {
                definition.name,
                to_address_names(definition.name, reverse_names) if game in ADDRESS_SYMBOL_GAMES else definition.name,
            }
        ]
    if not matching and game == "sly2" and len(definitions) == 1:
        matching = definitions  # a per-function file: its only definition (same rule as the extractor)
    if len(matching) > 1 and definition_ordinal < len(matching):
        matching = [matching[definition_ordinal]]
    if len(matching) != 1:
        return None
    definition = matching[0]
    source_bytes = source_text.encode()
    return (
        source_bytes[: definition.start_byte] + candidate_text.encode() + source_bytes[definition.end_byte :]
    ).decode()


def candidate_defines(game: str, candidate_text: str, function_id: str) -> bool:
    names = [definition.name for definition in find_function_definitions(candidate_text)]
    return bool(names) and (function_id in names or game == "sly2")


JOB_TIMEOUT_SECONDS = 180


class Worker:
    """One verifier subprocess; a reader thread moves its output lines into a queue so waiting can time out."""

    def __init__(self, game: str) -> None:
        self.process = subprocess.Popen(
            [*WORKER_COMMANDS[game], str(REPOSITORIES[game])],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read_lines, daemon=True).start()

    def _read_lines(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)  # end of output: the process exited

    def alive(self) -> bool:
        return self.process.poll() is None

    def kill(self) -> None:
        self.process.kill()


class WorkerPool:
    """One persistent worker process per thread; jobs are JSON lines. A reply that takes too long fails the job."""

    def __init__(self, game: str, size: int = 6, timeout_seconds: float = JOB_TIMEOUT_SECONDS) -> None:
        self.game = game
        self.size = size
        self.timeout_seconds = timeout_seconds
        self._local = threading.local()
        self._workers: list[Worker] = []
        self._lock = threading.Lock()

    def _worker(self) -> Worker:
        worker: Worker | None = getattr(self._local, "worker", None)
        if worker is None or not worker.alive():
            worker = Worker(self.game)
            self._local.worker = worker
            with self._lock:
                self._workers.append(worker)
        return worker

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        worker = self._worker()
        assert worker.process.stdin is not None
        try:
            worker.process.stdin.write(json.dumps(job) + "\n")
            worker.process.stdin.flush()
            line = worker.lines.get(timeout=self.timeout_seconds)
            if line is None:
                raise RuntimeError("worker exited")
            result: dict[str, Any] = json.loads(line)
            if result.get("id") != job["id"]:
                raise RuntimeError("reply for another job")
            return result
        except (OSError, RuntimeError, json.JSONDecodeError, queue.Empty) as error:
            worker.kill()  # a killed worker cannot answer late and desynchronise the next job
            self._local.worker = None
            note = "timeout" if isinstance(error, queue.Empty) else repr(error)
            return {
                "id": job["id"],
                "compiled": False,
                "status": "ERROR",
                "similarity": 0.0,
                "note": f"pool: {note}"[:200],
            }

    def close(self) -> None:
        for worker in self._workers:
            worker.kill()


def build_job(candidate: Candidate, source_text: str, with_diff: bool = False) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": f"{candidate.function_id}-{uuid.uuid4().hex[:8]}",
        "function": candidate.function_id,
        "source_text": source_text,
        "with_diff": with_diff,
    }
    if candidate.game in ADDRESS_SYMBOL_GAMES:
        job["source_file"] = candidate.source_file
    else:
        candidate_definitions = find_function_definitions(candidate.text)
        job.update(
            {
                "definition": candidate_definitions[0].name if candidate_definitions else candidate.definition_name,
                "address": candidate.address,
                "size": candidate.size_bytes,
                "language": "c" if candidate.source_file.endswith(".c") else "c++",
            }
        )
    return job


def verify_candidates(candidates: list[Candidate], workers: int = 6, with_diff: bool = False) -> list[dict[str, Any]]:
    """Result per candidate (same order): compiled, status, similarity, rule_hits, rule_clean, exact, note.

    With ``with_diff`` a DIFFER result also has ``diff``, the condensed target | ours instruction diff."""
    game = candidates[0].game if candidates else "sly1"
    pool = WorkerPool(game, workers)

    def verify_one(candidate: Candidate) -> dict[str, Any]:
        rule_hits = scan_rules(candidate.text)
        if game in ADDRESS_SYMBOL_GAMES:
            import re

            if re.search(HUNGARIAN_PATTERN, candidate.text):
                rule_hits.append("hungarian_prefix")
        base: dict[str, Any] = {"rule_hits": rule_hits, "rule_clean": not rule_hits}
        if not candidate.text.strip() or not candidate_defines(game, candidate.text, candidate.function_id):
            return {
                **base,
                "compiled": False,
                "status": "ERROR",
                "similarity": 0.0,
                "exact": False,
                "note": "no function definition",
            }
        spliced = splice_candidate(
            game,
            candidate.source_file,
            candidate.function_id,
            candidate.definition_name,
            candidate.definition_ordinal,
            candidate.text,
        )
        if spliced is None:
            return {
                **base,
                "compiled": False,
                "status": "ERROR",
                "similarity": 0.0,
                "exact": False,
                "note": "target not found in source",
            }
        result = pool.submit(build_job(candidate, spliced, with_diff))
        rule_hits = base["rule_hits"] + [hit for hit in result.get("worker_rules", []) if hit not in base["rule_hits"]]
        return {
            **base,
            **{key: result[key] for key in ("compiled", "status", "similarity", "note")},
            "rule_hits": rule_hits,
            "rule_clean": not rule_hits,
            "exact": result["status"] == "EXACT",
            **({"diff": result["diff"]} if "diff" in result else {}),
        }

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(verify_one, candidates))
    finally:
        pool.close()
