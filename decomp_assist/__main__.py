"""CLI: ``python -m decomp_assist --game sly1|sly2 --function FUNC [--source FILE]``.

Samples candidates from the fine-tuned model, verifies each with the repo's own compiler and comparer, repairs the
closest failure for a few rounds, and prints one JSON object (``--json``) or a short report. Only an ``exact``
outcome is a verified, rule-clean match; it still needs the repo's full build before it is committed.

The output includes instruction diffs and C derived from the disc image. Do not paste it anywhere the repo's
LEGAL.md forbids (hosted chats included) unless you accept that.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
from pathlib import Path

from decomp_assist.adapters import DEFAULT_ADAPTER, DEFAULT_REPAIR_ADAPTER, resolve_adapter
from decomp_assist.loop import Outcome, run_assist
from decomp_assist.target import TargetError, build_target
from eval.generate import Generator, extract_code
from pipeline import verify
from pipeline.verify import verify_candidates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-bf16"
WORKER_PYTHON = {"sly1": "venv/bin/python", "sly2": ".venv/bin/python"}


def use_repository(game: str, repository: Path) -> None:
    """Point the verifier at another checkout of the game's repository."""
    verify.REPOSITORIES[game] = repository
    verify.WORKER_COMMANDS[game][0] = str(repository / WORKER_PYTHON[game])


def format_report(outcome: Outcome, function: str) -> str:
    lines = [
        f"{function}: {outcome.status.upper()} (similarity {outcome.similarity:.3f}, {outcome.rounds_used} repair rounds, {outcome.candidates_tried} candidates)"
    ]
    lines += [f"note: {note}" for note in outcome.notes]
    if outcome.status == "exact":
        lines.append("Verified rule-clean EXACT. Run the repo's full build before committing.")
    if outcome.diff:
        lines += ["DIFF (target | ours):", outcome.diff]
    lines += ["CODE:", outcome.code]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, choices=["sly1", "sly2"])
    parser.add_argument("--function", required=True, help="sly1: func_XXXXXXXX or address; sly2: name or 0xADDRESS")
    parser.add_argument("--source", help="file (relative to the repository) holding one definition of the function")
    parser.add_argument("--repository", help="checkout of the game's decomp repo (default: data/raw/<game>-decomp)")
    parser.add_argument("--n", type=int, default=8, help="samples per round")
    parser.add_argument("--rounds", type=int, default=2, help="repair rounds after the first sampling")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--repair-adapter", default=DEFAULT_REPAIR_ADAPTER)
    parser.add_argument("--json", action="store_true", help="print one JSON object")
    arguments = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
    repository = (
        Path(arguments.repository)
        if arguments.repository
        else PROJECT_ROOT / "data" / "raw" / f"{arguments.game}-decomp"
    )
    use_repository(arguments.game, repository)
    try:
        target = build_target(arguments.game, repository, arguments.function, arguments.source)
    except (TargetError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"target {target.row['function_id']}: {target.row['size_bytes']} bytes, source {target.source_file}",
        file=sys.stderr,
    )

    generators: dict[str, Generator] = {}

    def sampler(adapter: str):  # type: ignore[no-untyped-def]
        def sample(messages: list[dict[str, str]], count: int) -> list[str]:
            if adapter not in generators:
                generators.clear()  # one model in memory at a time
                gc.collect()
                generators[adapter] = Generator(arguments.model, resolve_adapter(adapter))
            texts = generators[adapter].sample([messages], count, arguments.temperature, arguments.max_tokens)[0]
            return [extract_code(text) for text in texts]

        return sample

    def verifier(texts: list[str]) -> list[dict[str, object]]:
        return verify_candidates([target.candidate(text) for text in texts], arguments.workers, with_diff=True)

    outcome = run_assist(
        target.row,
        sampler(arguments.adapter),
        sampler(arguments.repair_adapter),
        verifier,
        arguments.n,
        arguments.rounds,
    )
    if arguments.json:
        print(json.dumps({"function": target.row["function_id"], **dataclasses.asdict(outcome)}))
    else:
        print(format_report(outcome, target.row["function_id"]))
    return 0 if outcome.status == "exact" else 1


if __name__ == "__main__":
    sys.exit(main())
