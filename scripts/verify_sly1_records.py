"""Re-verify every sly1 record with the repo's own check.py (EXACT expected). Prints counts only."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = PROJECT_ROOT / "data" / "raw" / "sly1-decomp"


def verify(record: dict[str, str]) -> tuple[str, str]:
    completed = subprocess.run(
        [sys.executable, "tools/check.py", record["source_file"], record["function_id"]],
        check=False,
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
    )
    first_line = (completed.stdout.strip().splitlines() or ["NO_OUTPUT"])[0]
    return record["function_id"], first_line.split(":")[0]


def main() -> None:
    records = [json.loads(line) for line in (PROJECT_ROOT / "data" / "records" / "sly1.jsonl").read_text().splitlines()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(verify, records))
    print(Counter(verdict for _, verdict in results))
    failures = [function_id for function_id, verdict in results if verdict != "EXACT"]
    (PROJECT_ROOT / "data" / "records" / "sly1_verify_failures.txt").write_text("\n".join(failures))
    print("non-exact:", len(failures))


if __name__ == "__main__":
    main()
