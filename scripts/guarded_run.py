"""Run a command under a memory watchdog: kill its whole process group when available RAM gets critical.

Usage: python scripts/guarded_run.py [--minimum-available-gib 6] -- COMMAND...
Exit code 99 when the watchdog killed the command for low RAM, 98 when it ran past --maximum-seconds.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time

GIBIBYTE = 1024**3


def available_bytes() -> int:
    """Free + inactive + speculative + purgeable pages: what macOS can hand out without swapping."""
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))  # type: ignore[union-attr]
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        match = re.search(rf"{label}:\s+(\d+)", output)
        pages += int(match.group(1)) if match else 0
    return pages * page_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-available-gib", type=float, default=6.0)
    parser.add_argument("--interval-seconds", type=float, default=2.0)
    parser.add_argument("--strikes", type=int, default=2, help="consecutive bad samples before killing")
    parser.add_argument("--maximum-seconds", type=float, default=0.0, help="wall-clock limit; 0 means none")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command

    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    strikes = 0
    lowest_available = available_bytes()
    while process.poll() is None:
        time.sleep(arguments.interval_seconds)
        if arguments.maximum_seconds and time.monotonic() - started > arguments.maximum_seconds:
            print(f"[watchdog] KILL: ran past {arguments.maximum_seconds:.0f} s", file=sys.stderr, flush=True)
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            sys.exit(98)
        available = available_bytes()
        lowest_available = min(lowest_available, available)
        bad = available < arguments.minimum_available_gib * GIBIBYTE
        strikes = strikes + 1 if bad else 0
        if strikes >= arguments.strikes:
            print(
                f"[watchdog] KILL: available {available / GIBIBYTE:.1f} GiB",
                file=sys.stderr,
                flush=True,
            )
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            sys.exit(99)
    print(f"[watchdog] finished; lowest available {lowest_available / GIBIBYTE:.1f} GiB", file=sys.stderr, flush=True)
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
