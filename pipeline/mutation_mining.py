"""CLI: synthetic T2 failures by mutating train ground truth. Prints counts only.

    python -m pipeline.mutation_mining NAME [--count N] [--mutants-per-function M] [--per-function K]

Keeps a mutant only if it compiles, is rule-clean and DIFFERs from the target. Writes ``data/t2/NAME.jsonl`` in the
same record format as ``pipeline.failure_mining`` with ``source`` ``mutation:<operator>``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from eval.run_eval import load_split, to_candidate
from pipeline.failure_mining import failure_record, select_unseen_rows
from pipeline.mutations import mutants
from pipeline.verify import verify_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "t2"
CHUNK_FUNCTIONS = 250


def keep_differing_mutants(
    names: list[str], results: list[dict[str, Any]], per_function_limit: int, seed: int
) -> list[tuple[str, int]]:
    """``(operator name, index)`` of compiled, rule-clean, DIFFER mutants, at most ``per_function_limit``, random order."""
    kept = [
        (name, index)
        for index, (name, result) in enumerate(zip(names, results, strict=True))
        if result["compiled"] and result["rule_clean"] and result["status"] == "DIFFER"
    ]
    random.Random(seed).shuffle(kept)
    return kept[:per_function_limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--mutants-per-function", type=int, default=4)
    parser.add_argument("--per-function", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rows = select_unseen_rows(load_split("train"), set(), arguments.count, arguments.seed)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{arguments.name}.jsonl"
    output_path.write_text("")
    operator_counts: Counter[str] = Counter()
    written = 0
    for start in range(0, len(rows), CHUNK_FUNCTIONS):
        chunk = rows[start : start + CHUNK_FUNCTIONS]
        records: list[dict[str, Any]] = []
        for game in ("sly1", "sly2"):
            game_rows = [row for row in chunk if row["game"] == game]
            pairs = [
                (row, name, text)
                for row in game_rows
                for name, text in mutants(row["c_text"], arguments.mutants_per_function, arguments.seed)
            ]
            if not pairs:
                continue
            results = verify_candidates(
                [to_candidate(row, text) for row, _, text in pairs], arguments.workers, with_diff=True
            )
            for row in game_rows:
                indices = [index for index, (other, _, _) in enumerate(pairs) if other is row]
                kept = keep_differing_mutants(
                    [pairs[index][1] for index in indices],
                    [results[index] for index in indices],
                    arguments.per_function,
                    arguments.seed,
                )
                for name, position in kept:
                    index = indices[position]
                    records.append(failure_record(row, pairs[index][2], results[index], f"mutation:{name}"))
                    operator_counts[name] += 1
        with output_path.open("a") as handle:
            handle.write("".join(json.dumps(record) + "\n" for record in records))
        written += len(records)
        print(f"{min(start + CHUNK_FUNCTIONS, len(rows))}/{len(rows)} functions, {written} mutants kept", flush=True)
    print(json.dumps(dict(operator_counts)))


if __name__ == "__main__":
    main()
