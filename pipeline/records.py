"""Record schema shared by both games' extractors, plus rule scanning and cheap asm tags."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

BANNED_PATTERNS: dict[str, re.Pattern[str]] = {
    "inline_asm": re.compile(r"\b(?:asm|__asm__|__asm)\b"),
    "volatile": re.compile(r"\bvolatile\b"),
    "register_pin": re.compile(r"\bregister\b[^;{]*\basm\b"),
    "include_asm_marker": re.compile(r"\bINCLUDE_ASM\b"),
    "m2c_leftover": re.compile(r"\bM2C_\w+|/\*\s*Error"),
}
FLOAT_MNEMONIC_PATTERN = re.compile(r"^\s*(?:[a-z0-9]+\.s|mtc1|mfc1|cvt\.\w+\.\w+|c\.\w+\.s)\b", re.MULTILINE)
CALL_PATTERN = re.compile(r"^\s*(?:jal|jalr)\b", re.MULTILINE)
BRANCH_PATTERN = re.compile(r"\.L\d+\b")


@dataclass
class Record:
    game: str
    function_id: str
    address: int
    size_bytes: int
    group: str
    source_file: str
    assembly: str
    c_text: str
    context: str
    tags: list[str] = field(default_factory=list)
    rule_hits: list[str] = field(default_factory=list)
    matched_date: str | None = None
    definition_name: str = ""
    definition_ordinal: int = 0  # index among same-named definitions (overloads) in the source file

    @property
    def rule_clean(self) -> bool:
        return not self.rule_hits


def scan_rules(c_text: str) -> list[str]:
    return [name for name, pattern in BANNED_PATTERNS.items() if pattern.search(c_text)]


def assembly_tags(assembly_text: str, instruction_count: int) -> list[str]:
    tags: list[str] = []
    tags.append("calls" if CALL_PATTERN.search(assembly_text) else "leaf")
    if FLOAT_MNEMONIC_PATTERN.search(assembly_text):
        tags.append("float")
    if BRANCH_PATTERN.search(assembly_text):
        tags.append("branches")
    if instruction_count <= 6:
        tags.append("trivial")
    return tags


def write_records(records: list[Record], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps({**asdict(record), "rule_clean": record.rule_clean}) + "\n")
