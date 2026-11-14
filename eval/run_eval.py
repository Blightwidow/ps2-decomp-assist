"""CLI: evaluate a method on a split sample. Prints aggregates only; writes eval/results/<run_id>.json.

python -m eval.run_eval --method retrieval|zeroshot|fewshot|model --run-id ID [--adapter DIR] [--sample 150] [--n 8]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from eval.generate import Generator, build_messages, extract_code
from eval.metrics import summarize
from eval.retrieval import Retriever, retarget
from pipeline.verify import Candidate, verify_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIRECTORY = PROJECT_ROOT / "eval" / "results"


def load_split(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (PROJECT_ROOT / "data" / "splits" / f"{name}.jsonl").read_text().splitlines()]


def select_rows(split: str, sample: int, seed: int, deduped_only: bool) -> list[dict[str, Any]]:
    """Ground-truth-verified (sly1 and sly2) rows, optionally the deduped slice, a seeded sample balanced across games."""
    failures_path = PROJECT_ROOT / "data" / "splits" / f"ground_truth_failures_{split}.txt"
    failures = set(failures_path.read_text().split("\n")) if failures_path.exists() else set()
    rows = [row for row in load_split(split) if f"{row['game']}:{row['function_id']}" not in failures]
    if deduped_only:
        rows = [row for row in rows if row["deduped_slice"]]
    if not sample:
        return rows
    random_generator = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for game in ("sly1", "sly2"):
        pool = [row for row in rows if row["game"] == game]
        random_generator.shuffle(pool)
        chosen.extend(pool[: sample // 2])
    return chosen


def to_candidate(row: dict[str, Any], text: str) -> Candidate:
    return Candidate(
        row["game"],
        row["function_id"],
        row["address"],
        row["size_bytes"],
        row["definition_name"],
        row["definition_ordinal"],
        row["source_file"],
        text,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", required=True, choices=["ground_truth", "retrieval", "zeroshot", "fewshot", "model"]
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--adapter")
    parser.add_argument("--model", default="mlx-community/gemma-4-e2b-it-bf16")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--all-functions", action="store_true", help="not only the deduped slice")
    arguments = parser.parse_args()

    rows = select_rows(arguments.split, arguments.sample, arguments.seed, deduped_only=not arguments.all_functions)
    print(f"{arguments.run_id}: {len(rows)} functions, method {arguments.method}")
    started = time.time()
    texts_per_row: list[list[str]]
    if arguments.method == "ground_truth":
        texts_per_row = [[row["c_text"]] for row in rows]
    elif arguments.method == "retrieval":
        retriever = Retriever(load_split("train"))
        texts_per_row = [[retarget(neighbor, row) for neighbor in retriever.nearest(row, arguments.n)] for row in rows]
    else:
        generator = Generator(arguments.model, arguments.adapter)
        shot_retriever = Retriever(load_split("train")) if arguments.method == "fewshot" else None
        conversations = [
            build_messages(row, shot_retriever.nearest(row, arguments.shots) if shot_retriever else []) for row in rows
        ]
        texts_per_row = [
            [extract_code(text) for text in texts]
            for texts in generator.sample(conversations, arguments.n, arguments.temperature, arguments.max_tokens)
        ]
    generation_seconds = time.time() - started

    per_function: list[dict[str, Any]] = [
        {
            "function": f"{row['game']}:{row['function_id']}",
            "game": row["game"],
            "size_bytes": row["size_bytes"],
            "results": [],
        }
        for row in rows
    ]
    for game in ("sly1", "sly2"):
        indices = [index for index, row in enumerate(rows) if row["game"] == game]
        flat = [(index, text) for index in indices for text in (texts_per_row[index] or [""])]
        results = verify_candidates([to_candidate(rows[index], text) for index, text in flat], arguments.workers)
        for (index, _), result in zip(flat, results, strict=True):
            per_function[index]["results"].append(result)
    summary = summarize(per_function)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIRECTORY / f"{arguments.run_id}.json").write_text(
        json.dumps(
            {
                "arguments": vars(arguments),
                "seconds": {"generation": generation_seconds, "total": time.time() - started},
                "summary": summary,
                "per_function": [
                    {
                        **entry,
                        "results": [
                            {key: r[key] for key in ("compiled", "status", "similarity", "exact", "rule_clean")}
                            for r in entry["results"]
                        ],
                    }
                    for entry in per_function
                ],
            },
            indent=1,
        )
    )
    (RESULTS_DIRECTORY / f"{arguments.run_id}.candidates.jsonl").write_text(
        "".join(
            json.dumps({"function": row_id, "texts": texts}) + "\n"
            for row_id, texts in zip((p["function"] for p in per_function), texts_per_row, strict=True)
        )
    )
    print(
        json.dumps(
            {
                name: {key: round(value, 3) if isinstance(value, float) else value for key, value in block.items()}
                for name, block in summary.items()
                if name in ("all", "game:sly1", "game:sly2")
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
