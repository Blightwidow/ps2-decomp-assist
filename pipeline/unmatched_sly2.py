"""Sly 2 functions that are NOT matched yet (S6 set): asm only, no reference C. Prints counts only."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from pipeline.asm_parse import normalize_assembly
from pipeline.records import assembly_tags
from pipeline.sly2 import load_assembly_by_address, load_excluded_addresses

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = PROJECT_ROOT / "data" / "raw" / "sly2-decomp"
SDK_RANGE = (0x2319E8, 0x254CA0)
MAXIMUM_SIZE_BYTES = 400
MINIMUM_INSTRUCTIONS = 3


def main() -> None:
    with (REPOSITORY / "src" / "MANIFEST.tsv").open() as handle:
        matched_addresses = {int(row["address"], 16) for row in csv.DictReader(handle, delimiter="\t")}
    handwritten_path = REPOSITORY / "config" / "us" / "handwritten_asm.txt"
    handwritten_text = handwritten_path.read_text() if handwritten_path.exists() else ""
    excluded_addresses = load_excluded_addresses(REPOSITORY)
    statistics: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for address, function in sorted(load_assembly_by_address(REPOSITORY).items()):
        if address in matched_addresses:
            continue
        if SDK_RANGE[0] <= address < SDK_RANGE[1]:
            statistics["skip:sdk"] += 1
        elif address in excluded_addresses or function.name in handwritten_text:
            statistics["skip:blocker_or_handwritten"] += 1
        elif function.size_bytes > MAXIMUM_SIZE_BYTES or function.instruction_count < MINIMUM_INSTRUCTIONS:
            statistics["skip:size"] += 1
        else:
            assembly = normalize_assembly(function)
            rows.append(
                {
                    "game": "sly2",
                    "function_id": function.name,
                    "address": address,
                    "size_bytes": function.size_bytes,
                    "assembly": assembly,
                    "context": "",
                    "tags": assembly_tags(assembly, function.instruction_count),
                }
            )
            statistics["kept"] += 1
    output_path = PROJECT_ROOT / "data" / "splits" / "unmatched_sly2.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(dict(statistics))


if __name__ == "__main__":
    main()
