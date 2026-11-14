"""Persistent verification worker for sly2-decomp (replicates scripts/matching/try.py without the "lab" registry).

    python sly2_worker.py REPOSITORY_ROOT

stdin: {"id", "function", "address", "size", "language", "source_text", optional "with_diff"}; stdout: {"id", "compiled", "status", "similarity", "note", "diff" when asked and DIFFER}.
status is EXACT (score 0), DIFFER or ERROR. Never prints retail bytes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

REPOSITORY = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(REPOSITORY / "scripts" / "build"), str(REPOSITORY / "scripts" / "matching")]
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
os.chdir(REPOSITORY)

import check_source
import compare
import compile_unit
from elftools.elf.elffile import ELFFile

from pipeline.instruction_diff import diff_from_compare_report

_ORIGINAL_RESOLVE = compare.resolve
CALL_SITE_SUFFIX = re.compile(r"^(?P<base>.+)_[0-9A-Fa-f]{6}$")


def resolve_with_site_suffix(addresses: dict[str, int], name: str) -> int | None:
    """Whole-file units name same-file callees and globals ``NAME_<address of the function>``: resolve the base name."""
    found = _ORIGINAL_RESOLVE(addresses, name)
    if found is None:
        match = CALL_SITE_SUFFIX.match(name)
        if match:
            return _ORIGINAL_RESOLVE(addresses, match.group("base"))
    return found


compare.resolve = resolve_with_site_suffix
ELF_PATH = str(REPOSITORY / "work" / "elf" / "SCUS_973.16")
MERGED_SYMBOLS = Path(tempfile.gettempdir()) / f"sly2_eval_symbols_{os.getpid()}.txt"
MERGED_SYMBOLS.write_text(
    (REPOSITORY / "config" / "us" / "symbol_addrs.txt").read_text()
    + "\n"
    + (REPOSITORY / "config" / "matching" / "symbol_addrs.txt").read_text()
)
SYMBOLS_PATH = str(MERGED_SYMBOLS)
TEMPORARY_DIRECTORY = REPOSITORY / "work" / "eval_tmp"
TEMPORARY_DIRECTORY.mkdir(parents=True, exist_ok=True)
HEADER_PATTERN = re.compile(r"score (\d+).*?target (\d+) instr.*?ours (\d+) instr")


def choose_symbol(object_path: str, job: dict[str, object]) -> str | None:
    """The object's function for this job: exact name, the definition name, else the only one of the target's size."""
    with open(object_path, "rb") as object_file:
        symbol_table = ELFFile(object_file).get_section_by_name(".symtab")
        functions = [
            (symbol.name, symbol["st_size"])
            for symbol in symbol_table.iter_symbols()
            if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_size"] > 0
        ]
    names = [name for name, _ in functions]
    for wanted in (job["function"], job.get("definition")):
        if not wanted:
            continue
        wanted = str(wanted)
        for matches in (
            [name for name in names if name == wanted],
            [name for name in names if name.startswith(wanted + "__")],
            [name for name in names if wanted in name],
        ):
            if len(matches) == 1:
                return matches[0]
    size = int(job["size"])  # type: ignore[call-overload]
    same_size = [name for name, function_size in functions if abs(function_size - size) <= 8]
    return same_size[0] if len(same_size) == 1 else None


def run(job: dict[str, object]) -> dict[str, object]:
    result = compile_and_compare(job)
    found = check_source.problems(str(job.get("candidate_text", "")))
    result["worker_rules"] = sorted({f"{kind}:{what}"[:60] for _, kind, what in found})
    return result


def compile_and_compare(job: dict[str, object]) -> dict[str, object]:
    suffix = ".c" if job["language"] == "c" else ".cpp"
    with tempfile.NamedTemporaryFile(
        "w", suffix=suffix, dir=TEMPORARY_DIRECTORY, delete=False, prefix=f"{str(job['id'])[:24]}_"
    ) as handle:
        handle.write(str(job["source_text"]))
        source_path = Path(handle.name)
    object_path = source_path.with_suffix(".o")
    try:
        address = int(job["address"])  # type: ignore[call-overload]
        ok, message = compile_unit.compile_object(
            str(job["language"]),
            str(source_path),
            str(object_path),
            assembler=compile_unit.assembler_for(address),
            include=None,
            localize=False,
            vu0=compile_unit.vu0_for(address),
            movables=compile_unit.movables_for(address),
        )
        if not ok:
            return {"id": job["id"], "compiled": False, "status": "ERROR", "similarity": 0.0, "note": message[:300]}
        chosen = choose_symbol(str(object_path), job)
        if chosen is None:
            return {
                "id": job["id"],
                "compiled": True,
                "status": "ERROR",
                "similarity": 0.0,
                "note": "function not found in object",
            }
        arguments = Namespace(
            object=str(object_path),
            address=address,
            elf=ELF_PATH,
            symbol=chosen,
            size=int(job["size"]),  # type: ignore[call-overload]
            export=None,
            quiet=not job.get("with_diff"),
            symbols=SYMBOLS_PATH,
        )
        try:
            header = compare.compare(arguments)
        except (SystemExit, KeyError, ValueError, IndexError) as error:
            return {
                "id": job["id"],
                "compiled": True,
                "status": "ERROR",
                "similarity": 0.0,
                "note": f"compare: {error!r}"[:200],
            }
        match = HEADER_PATTERN.search(header)
        if not match:
            return {"id": job["id"], "compiled": True, "status": "ERROR", "similarity": 0.0, "note": header[:200]}
        score, target_instructions, our_instructions = (int(value) for value in match.groups())
        similarity = 1.0 if score == 0 else max(0.0, 1.0 - score / max(target_instructions, our_instructions, 1))
        result: dict[str, object] = {
            "id": job["id"],
            "compiled": True,
            "status": "EXACT" if score == 0 else "DIFFER",
            "similarity": similarity,
            "note": f"score {score}",
        }
        if job.get("with_diff") and score:
            result["diff"] = diff_from_compare_report(header)
        return result
    finally:
        source_path.unlink(missing_ok=True)
        object_path.unlink(missing_ok=True)
        object_path.with_suffix(".o.s").unlink(missing_ok=True)


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
