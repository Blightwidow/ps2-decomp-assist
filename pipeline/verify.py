"""Harness side of verification: splice a candidate function into its original source, run persistent workers.

A candidate is C++ text for ONE function. The original source file with that function replaced is compiled by the
repo's own toolchain and compared with retail by the repo's own compare code (see eval/workers).
"""

from __future__ import annotations

import json
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
}
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
    source_file: str  # relative to the repository (src/main/x.cpp for sly1, src/<module>/x.cpp for sly2)
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
    reverse_names = load_reverse_names(repository) if game == "sly1" else {}
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
                to_address_names(definition.name, reverse_names) if game == "sly1" else definition.name,
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


class WorkerPool:
    """One persistent worker process per thread; jobs are JSON lines."""

    def __init__(self, game: str, size: int = 6) -> None:
        self.game = game
        self.size = size
        self._local = threading.local()
        self._processes: list[subprocess.Popen[str]] = []
        self._lock = threading.Lock()

    def _worker(self) -> subprocess.Popen[str]:
        process = getattr(self._local, "process", None)
        if process is None or process.poll() is not None:
            process = subprocess.Popen(
                [*WORKER_COMMANDS[self.game], str(REPOSITORIES[self.game])],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._local.process = process
            with self._lock:
                self._processes.append(process)
        return process

    def submit(self, job: dict[str, Any]) -> dict[str, Any]:
        process = self._worker()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(job) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("worker exited")
            result: dict[str, Any] = json.loads(line)
            return result
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            process.kill()
            self._local.process = None
            return {
                "id": job["id"],
                "compiled": False,
                "status": "ERROR",
                "similarity": 0.0,
                "note": f"pool: {error!r}"[:200],
            }

    def close(self) -> None:
        for process in self._processes:
            process.kill()


def build_job(candidate: Candidate, source_text: str, with_diff: bool = False) -> dict[str, Any]:
    job: dict[str, Any] = {
        "id": f"{candidate.function_id}-{uuid.uuid4().hex[:8]}",
        "function": candidate.function_id,
        "source_text": source_text,
        "with_diff": with_diff,
    }
    if candidate.game == "sly1":
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
        if game == "sly1":
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
