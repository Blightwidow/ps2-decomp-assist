"""CLI: python -m pipeline.extract sly1|sly2. Prints counts only (never record contents)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pipeline import sly1, sly2
from pipeline.cpp_parse import find_function_definitions
from pipeline.matched_dates import earliest_date, first_seen_dates
from pipeline.records import write_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game", choices=["sly1", "sly2"])
    parser.add_argument("--max-size", type=int, default=400)
    arguments = parser.parse_args()
    module = sly1 if arguments.game == "sly1" else sly2
    records, statistics = module.extract(PROJECT_ROOT / "data" / "raw" / f"{arguments.game}-decomp", arguments.max_size)
    repository = PROJECT_ROOT / "data" / "raw" / f"{arguments.game}-decomp"
    dates = first_seen_dates(repository, "src/main" if arguments.game == "sly1" else "src")
    readable_names: dict[str, set[str]] = {}
    if arguments.game == "sly1":
        for readable, symbol in sly1.load_reverse_names(repository).items():
            readable_names.setdefault(symbol, set()).add(readable)
    for record in records:
        candidates = {record.function_id} | readable_names.get(record.function_id, set())
        candidates |= {definition.name for definition in find_function_definitions(record.c_text)}
        record.matched_date = earliest_date(candidates, dates)
    write_records(records, PROJECT_ROOT / "data" / "records" / f"{arguments.game}.jsonl")
    for key, count in sorted(statistics.items()):
        print(f"{key}: {count}")
    print("dated:", sum(record.matched_date is not None for record in records))
    print(
        "dates:",
        sorted(Counter(record.matched_date for record in records).items(), key=lambda item: str(item[0]))[:3],
        "...",
    )
    print("rule_clean:", sum(record.rule_clean for record in records), "of", len(records))
    print("groups:", len({record.group for record in records}))
    print("tags:", dict(Counter(tag for record in records for tag in record.tags)))


if __name__ == "__main__":
    main()
