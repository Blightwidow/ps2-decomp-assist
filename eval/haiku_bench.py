"""CLI: tiny zero-shot benchmark of a hosted Claude model on validation functions, via the headless ``claude`` CLI.

    python -m eval.haiku_bench --run-id ID [--model haiku] [--sample 20] [--n 8]

Sends the asm and context of the sampled validation functions (no ground truth) to the hosted model. The user accepted
this for a small sample; sly1 LEGAL.md otherwise forbids sending disc-derived data to hosted services.
Writes aggregates to eval/results/<run_id>.json; prints counts only. Candidate text stays in git-ignored results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from eval.generate import extract_code
from eval.metrics import summarize
from eval.repair_eval import verify_per_game
from eval.run_eval import RESULTS_DIRECTORY, select_rows
from pipeline.token_stats import SYSTEM_PROMPT, render_user_prompt

CALL_TIMEOUT_SECONDS = 180


def ask_model(model: str, row: dict[str, Any], working_directory: str) -> str:
    """One zero-shot completion through ``claude -p`` (no tools, no project settings); empty text on failure."""
    command = [
        "claude",
        "-p",
        "--model",
        model,
        "--tools",
        "",
        "--system-prompt",
        SYSTEM_PROMPT,
        "--no-session-persistence",
        "--disable-slash-commands",
        "--setting-sources",
        "",
    ]
    try:
        completed = subprocess.run(
            command,
            input=render_user_prompt(row),
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_SECONDS,
            cwd=working_directory,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""
    return extract_code(completed.stdout) if completed.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()

    rows = select_rows(arguments.split, arguments.sample, arguments.seed, deduped_only=True)
    jobs = [index for index in range(len(rows)) for _ in range(arguments.n)]
    with tempfile.TemporaryDirectory() as working_directory, ThreadPoolExecutor(arguments.parallel) as pool:
        texts = list(pool.map(lambda index: ask_model(arguments.model, rows[index], working_directory), jobs))
    texts_per_row: list[list[str]] = [[] for _ in rows]
    for index, text in zip(jobs, texts, strict=True):
        texts_per_row[index].append(text)
    empty = sum(not text for text in texts)
    results_per_row = verify_per_game(rows, texts_per_row, arguments.workers, with_diff=False)
    per_function = [
        {
            "function": f"{row['game']}:{row['function_id']}",
            "game": row["game"],
            "size_bytes": row["size_bytes"],
            "results": results,
        }
        for row, results in zip(rows, results_per_row, strict=True)
    ]
    summary = summarize(per_function)
    print(f"{len(rows)} functions, {len(texts)} calls, {empty} empty")
    print(json.dumps(summary["all"]))
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIRECTORY / f"{arguments.run_id}.json").write_text(
        json.dumps(
            {
                "arguments": vars(arguments),
                "empty": empty,
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


if __name__ == "__main__":
    main()
