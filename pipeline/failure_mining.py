"""CLI: mine model-generated failures on train functions for T2 repair data. Prints counts only.

    python -m pipeline.failure_mining NAME --adapter DIR [--model M] [--exclude-ids data/sft/pilot/ids.json]
        [--count N] [--n SAMPLES] [--per-function K]

Writes ``data/t2/NAME.jsonl``: one record per failed candidate (compile error or DIFFER) with its verifier feedback.
The file holds disc-derived material and stays under the git-ignored ``data/``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from eval.generate import Generator, build_messages, extract_code
from eval.run_eval import load_split, to_candidate
from pipeline.verify import verify_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "t2"
CHUNK_FUNCTIONS = 100


def select_failures(
    texts: list[str], results: list[dict[str, Any]], per_function_limit: int, seed: int
) -> list[tuple[str, dict[str, Any]]]:
    """Distinct non-empty failed candidates of one function, at most ``per_function_limit``, in a seeded random order."""
    seen: set[str] = set()
    failures: list[tuple[str, dict[str, Any]]] = []
    for text, result in zip(texts, results, strict=True):
        if result["exact"] or not text.strip() or text in seen:
            continue
        seen.add(text)
        failures.append((text, result))
    random.Random(seed).shuffle(failures)
    return failures[:per_function_limit]


def failure_record(row: dict[str, Any], text: str, result: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "game": row["game"],
        "function_id": row["function_id"],
        "source": source,
        "candidate": text,
        "status": result["status"],
        "compiled": result["compiled"],
        "similarity": result["similarity"],
        "diff": result.get("diff", ""),
        "note": result["note"] if not result["compiled"] else "",
    }


def select_unseen_rows(
    rows: list[dict[str, Any]], excluded_ids: set[str], count: int, seed: int
) -> list[dict[str, Any]]:
    """Rule-clean deduped train rows outside ``excluded_ids``, seeded shuffle, first ``count``."""
    eligible = [
        row
        for row in rows
        if row["deduped_slice"] and row["rule_clean"] and f"{row['game']}:{row['function_id']}" not in excluded_ids
    ]
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-e2b-it-bf16")
    parser.add_argument("--exclude-ids")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--per-function", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    excluded: set[str] = set()
    if arguments.exclude_ids:
        excluded = set(json.loads(Path(arguments.exclude_ids).read_text())["train"])
    rows = select_unseen_rows(load_split("train"), excluded, arguments.count, arguments.seed)
    generator = Generator(arguments.model, arguments.adapter)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIRECTORY / f"{arguments.name}.jsonl"
    output_path.write_text("")
    written = 0
    for start in range(0, len(rows), CHUNK_FUNCTIONS):
        chunk = rows[start : start + CHUNK_FUNCTIONS]
        conversations = [build_messages(row, []) for row in chunk]
        texts_per_row = [
            [extract_code(text) for text in texts]
            for texts in generator.sample(conversations, arguments.n, arguments.temperature)
        ]
        records: list[dict[str, Any]] = []
        for game in ("sly1", "sly2"):
            flat = [(row, text) for row, texts in zip(chunk, texts_per_row, strict=True) for text in texts]
            flat = [(row, text) for row, text in flat if row["game"] == game]
            if not flat:
                continue
            results = verify_candidates(
                [to_candidate(row, text) for row, text in flat], arguments.workers, with_diff=True
            )
            for row in {id(row): row for row, _ in flat}.values():
                function_pairs = [
                    (text, result) for (other, text), result in zip(flat, results, strict=True) if other is row
                ]
                for text, result in select_failures(
                    [text for text, _ in function_pairs],
                    [result for _, result in function_pairs],
                    arguments.per_function,
                    arguments.seed,
                ):
                    records.append(failure_record(row, text, result, f"model:{arguments.name}"))
        with output_path.open("a") as handle:
            handle.write("".join(json.dumps(record) + "\n" for record in records))
        written += len(records)
        print(f"{min(start + CHUNK_FUNCTIONS, len(rows))}/{len(rows)} functions, {written} failures", flush=True)


if __name__ == "__main__":
    main()
