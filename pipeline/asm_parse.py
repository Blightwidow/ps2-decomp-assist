"""Parse splat/spimdisasm MIPS assembly into per-function records and normalize it for prompts.

Input format (both decomp repos):

    glabel NAME                      function start (sly1 also uses ``nonmatching NAME, SIZE`` above it)
        /* ROM VRAM RAW */  mnemonic operands
    .LXXXXXXXX:                      local label definition (``jlabel``/``alabel`` lines define labels too)
    endlabel NAME  |  .size NAME, .  function end

Output is a canonical text: one ``mnemonic operands`` instruction per line, local labels renamed
``.L1``, ``.L2``... in order of definition, nothing disc-derived is kept besides the instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INSTRUCTION_PATTERN = re.compile(
    r"^\s*/\*\s*[0-9A-Fa-f]+\s+(?P<vram>[0-9A-Fa-f]{8})\s+[0-9A-Fa-f]{8}\s*\*/\s*(?P<text>.*?)\s*$"
)
FUNCTION_START_PATTERN = re.compile(r"^glabel\s+(?P<name>\S+)\s*$")
LABEL_DEFINITION_PATTERN = re.compile(r"^(?:\.(?P<plain>L\w+):|(?:jlabel|alabel)\s+\.?(?P<marked>L\w+))\s*$")
LOCAL_LABEL_REFERENCE_PATTERN = re.compile(r"\.L\w+")
FUNCTION_END_PATTERN = re.compile(r"^(?:endlabel\s+(?P<end_name>\S+)|\.size\s+(?P<size_name>[^\s,]+),)")
TRAILING_COMMENT_PATTERN = re.compile(r"\s+#.*$")


@dataclass
class AssemblyFunction:
    """One function: ``lines`` mixes instruction texts and label definitions (``.LXXXX:``)."""

    name: str
    address: int
    lines: list[str] = field(default_factory=list)
    instruction_count: int = 0

    @property
    def size_bytes(self) -> int:
        return self.instruction_count * 4


def parse_assembly_functions(assembly_text: str) -> list[AssemblyFunction]:
    """Split an assembly file into functions. Functions without any instruction are skipped."""
    functions: list[AssemblyFunction] = []
    current: AssemblyFunction | None = None
    current_address: int | None = None

    def close_current() -> None:
        nonlocal current, current_address
        if current is not None and current.instruction_count > 0 and current_address is not None:
            current.address = current_address
            functions.append(current)
        current = None
        current_address = None

    for raw_line in assembly_text.splitlines():
        stripped_line = raw_line.strip()
        start_match = FUNCTION_START_PATTERN.match(stripped_line)
        if start_match:
            close_current()
            current = AssemblyFunction(name=start_match.group("name"), address=0)
            continue
        if current is None:
            continue
        end_match = FUNCTION_END_PATTERN.match(stripped_line)
        if end_match:
            close_current()
            continue
        label_match = LABEL_DEFINITION_PATTERN.match(stripped_line)
        if label_match:
            label_name = label_match.group("plain") or label_match.group("marked")
            current.lines.append(f".{label_name}:")
            continue
        instruction_match = INSTRUCTION_PATTERN.match(raw_line)
        if instruction_match:
            if current_address is None:
                current_address = int(instruction_match.group("vram"), 16)
            instruction_text = TRAILING_COMMENT_PATTERN.sub("", instruction_match.group("text"))
            if instruction_text:
                current.lines.append(instruction_text)
                current.instruction_count += 1
    close_current()
    return functions


def normalize_assembly(function: AssemblyFunction, symbol_aliases: dict[str, str] | None = None) -> str:
    """Canonical prompt text: collapsed whitespace, local labels renumbered, optional symbol renaming.

    ``symbol_aliases`` maps a symbol name to the name to print (for example recovered name ->
    ``func_XXXXXXXX``); it applies to whole identifiers only.
    """
    label_numbers: dict[str, int] = {}
    for line in function.lines:
        if line.endswith(":") and line.startswith(".L"):
            label_numbers.setdefault(line[:-1], len(label_numbers) + 1)
    # labels only referenced (never defined in this function) are numbered after the defined ones
    for line in function.lines:
        for reference in LOCAL_LABEL_REFERENCE_PATTERN.findall(line):
            label_numbers.setdefault(reference, len(label_numbers) + 1)

    alias_pattern = None
    if symbol_aliases:
        alias_pattern = re.compile(
            r"\b(" + "|".join(re.escape(name) for name in sorted(symbol_aliases, key=len, reverse=True)) + r")\b"
        )

    output_lines: list[str] = []
    for line in function.lines:
        if line.endswith(":") and line.startswith(".L"):
            output_lines.append(f".L{label_numbers[line[:-1]]}:")
            continue
        collapsed = re.sub(r"\s+", " ", line).strip()
        collapsed = LOCAL_LABEL_REFERENCE_PATTERN.sub(lambda match: f".L{label_numbers[match.group(0)]}", collapsed)
        if alias_pattern is not None and symbol_aliases is not None:
            collapsed = alias_pattern.sub(lambda match: symbol_aliases[match.group(1)], collapsed)
        output_lines.append(collapsed)
    return "\n".join(output_lines)
