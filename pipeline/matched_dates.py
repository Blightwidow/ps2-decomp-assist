"""Date a function was first committed under its source directory (feeds the chronological split)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

CALL_TOKEN_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
COMMIT_MARKER = "\x01"


def first_seen_dates(repository: Path, source_directory: str) -> dict[str, str]:
    """Identifier-followed-by-``(`` on an added line -> earliest commit date (YYYY-MM-DD) it was added."""
    process = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "log",
            "-p",
            "--no-color",
            f"--format={COMMIT_MARKER}%cs",
            "--",
            source_directory,
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    earliest: dict[str, str] = {}
    current_date = ""
    for line in process.stdout.splitlines():
        if line.startswith(COMMIT_MARKER):
            current_date = line[1:]
        elif line.startswith("+") and not line.startswith("+++"):
            for name in CALL_TOKEN_PATTERN.findall(line):
                if name not in earliest or current_date < earliest[name]:
                    earliest[name] = current_date
    return earliest


def earliest_date(candidate_names: set[str], dates: dict[str, str]) -> str | None:
    known = [dates[name] for name in candidate_names if name in dates]
    return min(known) if known else None
