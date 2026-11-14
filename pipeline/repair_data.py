"""CLI: build the SFT-2 chat data (T2 repair examples mixed with T1 replay). Prints counts only.

    python -m pipeline.repair_data NAME FAILURES... [--t1-replay N] [--valid N] [--maximum-tokens 2048]

T2 example: system = repair prompt; user = the T1 prompt plus ``CANDIDATE:`` and ``DIFF:``; assistant = the ground truth.
The ``DIFF:`` section is the instruction diff of a DIFFER candidate, or the compiler message of one that did not compile.
Synthetic (mutation) examples are capped at ``synthetic-share`` of the T2 total; the last ``valid`` examples, grouped by
function, are held out for validation loss.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pipeline.token_stats import render_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPAIR_SYSTEM_PROMPT = (
    "Matching decompiler: PS2 EE (MIPS R5900), SN GCC 2.95.3, C++. "
    "The candidate does not match the target. Output only the corrected function."
)
DIFF_LINE_BUDGETS = (40, 20, 10, 5)
COMPILE_NOTE_LIMIT = 300


def feedback_text(record: dict[str, Any]) -> str:
    """The ``DIFF:`` body: compiler message for a candidate that did not compile, else its instruction diff."""
    if not record["compiled"]:
        return f"COMPILE ERROR: {record['note'][:COMPILE_NOTE_LIMIT]}"
    return str(record["diff"])


def truncate_diff(diff: str, maximum_lines: int) -> str:
    lines = diff.split("\n")
    if len(lines) <= maximum_lines:
        return diff
    return "\n".join([*lines[:maximum_lines], f"... ({len(lines) - maximum_lines} more lines)"])


def render_repair_prompt(row: dict[str, Any], candidate: str, feedback: str) -> str:
    return f"{render_user_prompt(row)}\nCANDIDATE:\n{candidate}\nDIFF:\n{feedback}"


def to_repair_example(
    row: dict[str, Any], record: dict[str, Any], count_tokens: Callable[[str], int], maximum_tokens: int
) -> dict[str, Any] | None:
    """The chat example, with the diff cut to fit ``maximum_tokens`` in total, or None if even the shortest cut does not fit."""
    feedback = feedback_text(record)
    for maximum_lines in DIFF_LINE_BUDGETS:
        user = render_repair_prompt(row, record["candidate"], truncate_diff(feedback, maximum_lines))
        if count_tokens(REPAIR_SYSTEM_PROMPT + user + row["c_text"]) <= maximum_tokens:
            return {
                "messages": [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": row["c_text"]},
                ]
            }
    return None


def cap_synthetic(records: list[dict[str, Any]], synthetic_share: float, seed: int) -> list[dict[str, Any]]:
    """Keep every model failure and as many synthetic ones as keep their share at most ``synthetic_share``."""
    real = [record for record in records if not record["source"].startswith("mutation:")]
    synthetic = [record for record in records if record["source"].startswith("mutation:")]
    allowed = int(len(real) * synthetic_share / (1 - synthetic_share) + 1e-9)
    random.Random(seed).shuffle(synthetic)
    return real + synthetic[:allowed]


def split_by_function(
    records: list[dict[str, Any]], valid_count: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(train, valid): whole functions go to valid until it holds at least ``valid_count`` records."""
    by_function: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_function.setdefault(f"{record['game']}:{record['function_id']}", []).append(record)
    keys = sorted(by_function)
    random.Random(seed).shuffle(keys)
    valid: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for key in keys:
        (valid if len(valid) < valid_count else train).extend(by_function[key])
    return train, valid


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("failures", nargs="+", help="names under data/t2/")
    parser.add_argument("--t1-replay", type=int, default=2500)
    parser.add_argument("--valid", type=int, default=100)
    parser.add_argument("--synthetic-share", type=float, default=0.7)
    parser.add_argument("--maximum-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mlx-community/gemma-4-e2b-it-bf16")

    def count_tokens(text: str) -> int:
        return len(tokenizer(text)["input_ids"])

    rows = {
        f"{row['game']}:{row['function_id']}": row
        for row in read_jsonl(PROJECT_ROOT / "data" / "splits" / "train.jsonl")
    }
    records = [
        record for name in arguments.failures for record in read_jsonl(PROJECT_ROOT / "data" / "t2" / f"{name}.jsonl")
    ]
    records = cap_synthetic(records, arguments.synthetic_share, arguments.seed)
    train_records, valid_records = split_by_function(records, arguments.valid, arguments.seed)

    def build(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
        examples = (
            to_repair_example(
                rows[f"{record['game']}:{record['function_id']}"], record, count_tokens, arguments.maximum_tokens
            )
            for record in selected
        )
        return [example for example in examples if example is not None]

    train_t2, valid_t2 = build(train_records), build(valid_records)
    t1_train = read_jsonl(PROJECT_ROOT / "data" / "sft" / "full" / "train.jsonl")
    random.Random(arguments.seed).shuffle(t1_train)
    t1_valid = read_jsonl(PROJECT_ROOT / "data" / "sft" / "full" / "valid.jsonl")[: len(valid_t2)]
    train = train_t2 + t1_train[: arguments.t1_replay]
    random.Random(arguments.seed).shuffle(train)
    valid = valid_t2 + t1_valid
    output = PROJECT_ROOT / "data" / "sft" / arguments.name
    output.mkdir(parents=True, exist_ok=True)
    for label, examples in (("train", train), ("valid", valid)):
        (output / f"{label}.jsonl").write_text("".join(json.dumps(example) + "\n" for example in examples))
    sources = {"model": 0, "mutation": 0}
    for record in train_records:
        sources["mutation" if record["source"].startswith("mutation:") else "model"] += 1
    print(
        f"records: {sources}; dropped over token cap: {len(train_records) - len(train_t2)} train, "
        f"{len(valid_records) - len(valid_t2)} valid"
    )
    print(
        f"train: {len(train_t2)} T2 + {min(arguments.t1_replay, len(t1_train))} T1 = {len(train)}; valid: {len(valid)}"
    )


if __name__ == "__main__":
    main()
