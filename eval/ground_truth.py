"""Sanity gate: the reference C of every record must verify EXACT + rule-clean. Prints counts only.

python -m eval.ground_truth SPLIT [--limit N]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pipeline.verify import Candidate, verify_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "data" / "splits" / f"{arguments.split}.jsonl").read_text().splitlines()
    ]
    failures: list[str] = []
    for game in ("sly1", "sly2"):
        game_rows = [row for row in rows if row["game"] == game][: arguments.limit or None]
        candidates = [
            Candidate(
                game,
                row["function_id"],
                row["address"],
                row["size_bytes"],
                row["definition_name"],
                row["definition_ordinal"],
                row["source_file"],
                row["c_text"],
            )
            for row in game_rows
        ]
        results = verify_candidates(candidates, arguments.workers)
        statuses = Counter((result["status"], result["compiled"], result["rule_clean"]) for result in results)
        print(game, len(results), dict(statuses))
        notes = Counter(result["note"][:60] for result in results if result["status"] != "EXACT")
        print("  top notes:", notes.most_common(4))
        failures += [
            f"{game}:{row['function_id']}"
            for row, result in zip(game_rows, results, strict=True)
            if not (result["exact"] and result["rule_clean"])
        ]
    if not arguments.limit:  # a partial run must not overwrite the full list
        (PROJECT_ROOT / "data" / "splits" / f"ground_truth_failures_{arguments.split}.txt").write_text(
            "\n".join(failures)
        )
    print("failures:", len(failures))


if __name__ == "__main__":
    main()
