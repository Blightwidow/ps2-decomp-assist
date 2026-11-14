"""Token-length statistics of extracted records under the v1 prompt format (prints counts only)."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT = "Matching decompiler: PS2 EE (MIPS R5900), SN GCC 2.95.3, C++. Output only the function."


def render_user_prompt(record: dict[str, object]) -> str:
    return (
        f"GAME: {record['game']}  FUNC: {record['function_id']}  SIZE: {record['size_bytes']:#x}\n"
        f"CONTEXT:\n{record['context']}\nASM:\n{record['assembly']}"
    )


def main() -> None:
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mlx-community/gemma-4-e2b-it-bf16")
    for game in ("sly1", "sly2"):
        records = [
            json.loads(line) for line in (PROJECT_ROOT / "data" / "records" / f"{game}.jsonl").read_text().splitlines()
        ]
        totals = []
        for record in records:
            prompt_tokens = len(tokenizer(SYSTEM_PROMPT + render_user_prompt(record))["input_ids"])
            completion_tokens = len(tokenizer(record["c_text"])["input_ids"])
            totals.append((prompt_tokens, completion_tokens))
        sums = sorted(prompt + completion for prompt, completion in totals)
        instruction_tokens = [
            len(tokenizer(record["assembly"])["input_ids"]) / max(1, record["assembly"].count("\n") + 1)
            for record in records
        ]
        print(
            f"{game}: n={len(records)} total tokens median={statistics.median(sums):.0f} p90={sums[int(0.9 * len(sums))]} "
            f"max={sums[-1]} over2048={sum(total > 2048 for total in sums)} "
            f"asm tokens/line mean={statistics.mean(instruction_tokens):.1f} "
            f"completion median={statistics.median(c for _, c in totals):.0f}"
        )


if __name__ == "__main__":
    main()
