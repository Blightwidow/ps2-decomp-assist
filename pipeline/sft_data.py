"""CLI: python -m pipeline.sft_data NAME COUNT [--valid N]. Writes mlx-lm chat-format train/valid files. Prints counts only."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.metrics import size_bucket
from pipeline.token_stats import SYSTEM_PROMPT, render_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIRECTORY = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "sft"


def select_stratified(rows: list[dict[str, Any]], count: int, seed: int = 0) -> list[dict[str, Any]]:
    """Deduped-slice rows, sampled in proportion to each (game, size bucket) stratum."""
    eligible = [row for row in rows if row["deduped_slice"] and row["rule_clean"]]
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        strata[(row["game"], size_bucket(row["size_bytes"]))].append(row)
    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        members = strata[key]
        generator.shuffle(members)
        selected.extend(members[: max(1, round(count * len(members) / len(eligible)))])
    generator.shuffle(selected)
    return selected[:count]


def select_all(rows: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Every rule-clean row (trivial and duplicate-tagged rows included), shuffled."""
    selected = [row for row in rows if row["rule_clean"]]
    random.Random(seed).shuffle(selected)
    return selected


def to_chat_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_prompt(row)},
            {"role": "assistant", "content": row["c_text"]},
        ]
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("count", type=int, help="0 = all rule-clean train rows")
    parser.add_argument("--valid", type=int, default=60)
    arguments = parser.parse_args()
    train_rows = read_jsonl(SPLITS_DIRECTORY / "train.jsonl")
    train = select_all(train_rows) if arguments.count == 0 else select_stratified(train_rows, arguments.count)
    valid = select_stratified(read_jsonl(SPLITS_DIRECTORY / "validation.jsonl"), arguments.valid)
    output = OUTPUT_DIRECTORY / arguments.name
    output.mkdir(parents=True, exist_ok=True)
    for label, rows in (("train", train), ("valid", valid)):
        (output / f"{label}.jsonl").write_text("".join(json.dumps(to_chat_example(row)) + "\n" for row in rows))
        print(label, len(rows), "tokens", sum(row["token_count"] for row in rows))
    (output / "ids.json").write_text(json.dumps({"train": [f"{r['game']}:{r['function_id']}" for r in train]}))


if __name__ == "__main__":
    main()
