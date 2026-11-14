"""Guard: nothing disc-derived or proprietary may enter the repository (cf. the game repos' LEGAL.md)."""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_SUFFIXES = {".iso", ".bin", ".elf", ".s", ".o", ".exe", ".dll", ".zip", ".7z", ".safetensors", ".gguf"}
MAXIMUM_FILE_BYTES = 1_000_000
MIPS_LINE = re.compile(
    r"^\s*(?:\d+:\s*|/\*.*?\*/\s*)?(?:addiu|addu|lw|sw|lui|jal|jr|beq|bne|lwc1|swc1|sll|ori)\s+(?:\$\w+|\w+,)"
)
ASSEMBLY_RUN_LENGTH = 4


def repository_files() -> list[Path]:
    """Tracked files plus untracked ones that are not ignored: everything a commit could carry."""
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [PROJECT_ROOT / line for line in listing.splitlines() if (PROJECT_ROOT / line).is_file()]


def longest_assembly_run(text: str) -> int:
    longest = current = 0
    for line in text.splitlines():
        current = current + 1 if MIPS_LINE.match(line) else 0
        longest = max(longest, current)
    return longest


def test_no_binary_or_disc_file_types_are_committable() -> None:
    assert [str(path) for path in repository_files() if path.suffix.lower() in FORBIDDEN_SUFFIXES] == []


def test_no_large_files_are_committable() -> None:
    assert [str(path) for path in repository_files() if path.stat().st_size > MAXIMUM_FILE_BYTES] == []


def test_no_pasted_disassembly_is_committable() -> None:
    offenders = []
    for path in repository_files():
        if path.name == "test_no_disc_material.py":
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if longest_assembly_run(text) >= ASSEMBLY_RUN_LENGTH:
            offenders.append(str(path))
    assert offenders == []


def test_the_assembly_detector_flags_a_listing() -> None:
    listing = "addiu $sp, $sp, -0x20\nsw $ra, 0x10($sp)\nlui $v0, 0x1\njr $ra"
    assert longest_assembly_run(listing) == 4
