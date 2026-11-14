"""Persistent verification worker for sly1-decomp. Run with the repo's python, cwd anywhere.

    python sly1_worker.py REPOSITORY_ROOT

stdin: one JSON job per line: {"id", "source_file", "function", "source_text", optional "with_diff"}
stdout: one JSON result per line: {"id", "compiled", "status", "similarity", "note", "diff" when asked and DIFFER}; never prints retail bytes.
"""

from __future__ import annotations

import difflib
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
os.chdir(REPOSITORY)

import check
import compile as compiler
import elf

from pipeline.instruction_diff import condense, side_by_side

IMAGE = check.retail_image("us")
RESOLVER = check.Resolver("us")
TEMPORARY_DIRECTORY = REPOSITORY / "build" / "eval_tmp"
TEMPORARY_DIRECTORY.mkdir(parents=True, exist_ok=True)


def words(data: bytes) -> list[int]:
    return [int.from_bytes(data[index : index + 4], "little") for index in range(0, len(data) - len(data) % 4, 4)]


def similarity(ours: bytes, retail: bytes) -> float:
    ours_words, retail_words = words(ours), words(retail)
    while retail_words and retail_words[-1] == 0:
        retail_words.pop()
    if not ours_words or not retail_words:
        return 0.0
    return difflib.SequenceMatcher(None, retail_words, ours_words, autojunk=False).ratio()


def run(job: dict[str, str]) -> dict[str, object]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cpp", dir=TEMPORARY_DIRECTORY, delete=False, prefix=f"{job['id'][:24]}_"
    ) as handle:
        handle.write(job["source_text"])
        source_path = Path(handle.name)
    object_path = source_path.with_suffix(".o")
    try:
        try:
            compiler.compile_file(source_path, object_path, None, source=REPOSITORY / job["source_file"])
        except compiler.CompileError as error:
            return {"id": job["id"], "compiled": False, "status": "ERROR", "similarity": 0.0, "note": str(error)[:300]}
        compiled_object = elf.Elf(object_path.read_bytes())
        try:
            result = check.check_function(compiled_object, job["function"], "us", RESOLVER, IMAGE)
        except KeyError as error:
            return {"id": job["id"], "compiled": True, "status": "ERROR", "similarity": 0.0, "note": str(error)[:200]}
        status = result["status"]
        score = 1.0 if status == "EXACT" else similarity(result["ours"], result["retail"]) if "ours" in result else 0.0
        output: dict[str, object] = {
            "id": job["id"],
            "compiled": True,
            "status": status,
            "similarity": score,
            "note": str(result.get("note", ""))[:200],
        }
        if job.get("with_diff") and status == "DIFFER" and "ours" in result:
            retail_words = words(result["retail"])
            while retail_words and retail_words[-1] == 0:
                retail_words.pop()
            output["diff"] = condense(side_by_side(retail_words, words(result["ours"])))
        return output
    finally:
        source_path.unlink(missing_ok=True)
        object_path.unlink(missing_ok=True)


def main() -> None:
    for line in sys.stdin:
        job = json.loads(line)
        try:
            output = run(job)
        except Exception as error:  # noqa: BLE001 - a worker must survive any candidate
            output = {
                "id": job["id"],
                "compiled": False,
                "status": "ERROR",
                "similarity": 0.0,
                "note": f"worker: {error!r}"[:300],
            }
        sys.stdout.write(json.dumps(output) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
