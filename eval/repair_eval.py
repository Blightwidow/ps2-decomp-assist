"""CLI: evaluate T2 repair on the failures of an earlier T1 run. Prints aggregates only; writes eval/results/<run_id>.json.

    python -m eval.repair_eval --run-id ID --candidates eval/results/T1RUN.candidates.jsonl --adapter DIR
        [--split validation] [--sample 150] [--n 8]

Per function: every T1 candidate is verified again with instruction diffs; a function with an exact one is solved already.
Otherwise one failed candidate (seeded pick) goes to the repair prompt and ``n`` repairs are sampled and verified.
The report covers the unsolved functions (repair alone) and the combined solved fraction (T1 plus one repair round).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.generate import Generator, extract_code
from eval.metrics import summarize
from eval.run_eval import RESULTS_DIRECTORY, select_rows, to_candidate
from pipeline.failure_mining import select_failures
from pipeline.repair_data import REPAIR_SYSTEM_PROMPT, feedback_text, render_repair_prompt, truncate_diff
from pipeline.verify import verify_candidates

DIFF_LINE_BUDGET = 40


def build_repair_messages(row: dict[str, Any], candidate: str, result: dict[str, Any]) -> list[dict[str, str]]:
    """The T2 chat prompt for one failed candidate, in the training format."""
    record = {"compiled": result["compiled"], "note": result.get("note", ""), "diff": result.get("diff", "")}
    feedback = truncate_diff(feedback_text(record), DIFF_LINE_BUDGET)
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": render_repair_prompt(row, candidate, feedback)},
    ]


def verify_per_game(
    rows: list[dict[str, Any]], texts_per_row: list[list[str]], workers: int, with_diff: bool
) -> list[list[dict[str, Any]]]:
    """Verifier results per row, aligned with ``texts_per_row``."""
    results_per_row: list[list[dict[str, Any]]] = [[] for _ in rows]
    for game in ("sly1", "sly2"):
        flat = [
            (index, text)
            for index, row in enumerate(rows)
            if row["game"] == game
            for text in (texts_per_row[index] or [""])
        ]
        results = verify_candidates(
            [to_candidate(rows[index], text) for index, text in flat], workers, with_diff=with_diff
        )
        for (index, _), result in zip(flat, results, strict=True):
            results_per_row[index].append(result)
    return results_per_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default="mlx-community/gemma-4-e2b-it-bf16")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    started = time.time()
    rows = select_rows(arguments.split, arguments.sample, arguments.seed, deduped_only=True)
    by_function = {
        entry["function"]: entry["texts"]
        for entry in map(json.loads, Path(arguments.candidates).read_text().splitlines())
    }
    first_texts = [by_function[f"{row['game']}:{row['function_id']}"] for row in rows]
    first_results = verify_per_game(rows, first_texts, arguments.workers, with_diff=True)
    solved = [any(result["exact"] and result["rule_clean"] for result in results) for results in first_results]

    pending: list[tuple[int, str, dict[str, Any]]] = []
    for index, (texts, results) in enumerate(zip(first_texts, first_results, strict=True)):
        if solved[index]:
            continue
        picked = select_failures(texts, results, 1, arguments.seed)
        if picked:
            pending.append((index, picked[0][0], picked[0][1]))
    print(f"{len(rows)} functions, {sum(solved)} solved by T1, {len(pending)} to repair", flush=True)

    generator = Generator(arguments.model, arguments.adapter)
    conversations = [build_repair_messages(rows[index], text, result) for index, text, result in pending]
    repairs = [
        [extract_code(text) for text in texts]
        for texts in generator.sample(conversations, arguments.n, arguments.temperature, arguments.max_tokens)
    ]
    generation_seconds = time.time() - started

    pending_rows = [rows[index] for index, _, _ in pending]
    repair_results = verify_per_game(pending_rows, repairs, arguments.workers, with_diff=False)
    per_function = [
        {
            "function": f"{row['game']}:{row['function_id']}",
            "game": row["game"],
            "size_bytes": row["size_bytes"],
            "results": results,
        }
        for row, results in zip(pending_rows, repair_results, strict=True)
    ]
    summary = summarize(per_function)
    repaired = sum(
        any(result["exact"] and result["rule_clean"] for result in entry["results"]) for entry in per_function
    )
    combined = (sum(solved) + repaired) / len(rows)
    print(f"repair on unsolved: {json.dumps(summary['all'])}")
    print(
        f"combined solved (T1@{len(first_texts[0])} + one repair round): {combined:.1%} vs T1 {sum(solved) / len(rows):.1%}"
    )
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIRECTORY / f"{arguments.run_id}.json").write_text(
        json.dumps(
            {
                "arguments": vars(arguments),
                "seconds": {"generation": generation_seconds, "total": time.time() - started},
                "solved_by_t1": sum(solved),
                "functions": len(rows),
                "repaired": repaired,
                "combined_solved": combined,
                "summary": summary,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
