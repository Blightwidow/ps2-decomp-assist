"""CLI: sample a model on train functions and keep rule-clean EXACT candidates for rejection-sampling fine-tuning.

    python -m pipeline.rft_mining NAME --adapter DIR [--count N] [--n SAMPLES] [--temperature T]

Writes ``data/rft/NAME.jsonl`` (one record per kept candidate; disc-derived, git-ignored) and prints counts only.
A kept candidate is ``novel`` when it differs from the ground-truth text beyond whitespace; copies of the ground
truth teach nothing the SFT data does not already contain.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.generate import Generator, build_messages, extract_code
from eval.metrics import size_bucket
from eval.repair_eval import verify_per_game
from eval.run_eval import load_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "rft"
CHUNK_FUNCTIONS = 50


def normalize(text: str) -> str:
    return " ".join(text.split())


def select_balanced_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Deduped rule-clean rows, the same share from each size bucket, seeded."""
    eligible = [row for row in rows if row["deduped_slice"] and row["rule_clean"]]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        buckets[size_bucket(row["size_bytes"])].append(row)
    generator = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for name in sorted(buckets):
        generator.shuffle(buckets[name])
        selected.extend(buckets[name][: count // len(buckets)])
    generator.shuffle(selected)
    return selected


def keep_exact_candidates(row: dict[str, Any], texts: list[str], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct rule-clean EXACT candidates of one function, flagged novel when they differ from the ground truth."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for text, result in zip(texts, results, strict=True):
        key = normalize(text)
        if not (result["exact"] and result["rule_clean"]) or key in seen:
            continue
        seen.add(key)
        kept.append(
            {
                "game": row["game"],
                "function_id": row["function_id"],
                "candidate": text,
                "novel": key != normalize(row["c_text"]),
            }
        )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-e2b-it-bf16")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    rows = select_balanced_rows(load_split("train"), arguments.count, arguments.seed)
    generator = Generator(arguments.model, arguments.adapter)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{arguments.name}.jsonl"
    output_path.write_text("")
    solved: dict[str, int] = defaultdict(int)
    novel_solved: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for start in range(0, len(rows), CHUNK_FUNCTIONS):
        chunk = rows[start : start + CHUNK_FUNCTIONS]
        texts_per_row = [
            [extract_code(text) for text in texts]
            for texts in generator.sample(
                [build_messages(row, []) for row in chunk], arguments.n, arguments.temperature
            )
        ]
        results_per_row = verify_per_game(chunk, texts_per_row, arguments.workers, with_diff=False)
        records: list[dict[str, Any]] = []
        for row, texts, results in zip(chunk, texts_per_row, results_per_row, strict=True):
            kept = keep_exact_candidates(row, texts, results)
            records.extend(kept)
            bucket = size_bucket(row["size_bytes"])
            totals[bucket] += 1
            solved[bucket] += bool(kept)
            novel_solved[bucket] += any(record["novel"] for record in kept)
        with output_path.open("a") as handle:
            handle.write("".join(json.dumps(record) + "\n" for record in records))
        print(f"{min(start + CHUNK_FUNCTIONS, len(rows))}/{len(rows)} functions", flush=True)
    for bucket in sorted(totals):
        print(
            f"{bucket}: {totals[bucket]} functions, {solved[bucket]} with an exact sample, "
            f"{novel_solved[bucket]} with a novel exact sample"
        )


if __name__ == "__main__":
    main()
