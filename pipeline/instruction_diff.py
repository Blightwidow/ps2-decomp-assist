"""Compact side-by-side instruction diff (target | ours), the DIFF section of a T2 repair prompt.

Two sources feed one format: sly1 builds it from the two word lists, sly2 condenses the report of its own compare tool.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence

COLUMN_WIDTH = 38
MARKER_INDEX = COLUMN_WIDTH + 1
ADDRESS_COLUMN_WIDTH = 9
DEFAULT_CONTEXT_LINES = 1
DEFAULT_MAXIMUM_LINES = 40


def disassemble_word(word: int) -> str:
    """One R5900 instruction as normalized text (the sly2 compare tool uses the same call)."""
    import rabbitizer  # type: ignore[import-not-found,unused-ignore]

    instruction = rabbitizer.Instruction(word, category=rabbitizer.InstrCategory.R5900)
    return " ".join(str(instruction.disassemble(immOverride=None)).split())


def side_by_side(
    target_words: Sequence[int], our_words: Sequence[int], disassemble: Callable[[int], str] = disassemble_word
) -> list[str]:
    """One line per aligned instruction pair: ``target  marker  ours``, marker ' ' equal, '|' changed, '<' / '>' only one side."""
    matcher = difflib.SequenceMatcher(None, list(target_words), list(our_words), autojunk=False)
    lines: list[str] = []
    for operation, target_start, target_end, our_start, our_end in matcher.get_opcodes():
        width = max(target_end - target_start, our_end - our_start)
        for step in range(width):
            has_target = target_start + step < target_end
            has_ours = our_start + step < our_end
            left = disassemble(target_words[target_start + step]) if has_target else ""
            right = disassemble(our_words[our_start + step]) if has_ours else ""
            if operation == "equal":
                marker = " "
            elif has_target and has_ours:
                marker = "|"
            else:
                marker = "<" if has_target else ">"
            lines.append(f"{left:<{COLUMN_WIDTH}} {marker} {right}")
    return lines


def condense(
    lines: Sequence[str], context_lines: int = DEFAULT_CONTEXT_LINES, maximum_lines: int = DEFAULT_MAXIMUM_LINES
) -> str:
    """Changed lines plus context, skipped runs shown as ``...``, cut to ``maximum_lines`` with a count of what was dropped."""
    changed = [index for index, line in enumerate(lines) if len(line) > MARKER_INDEX and line[MARKER_INDEX] != " "]
    keep: set[int] = set()
    for index in changed:
        keep.update(range(max(0, index - context_lines), min(len(lines), index + context_lines + 1)))
    output: list[str] = []
    previous = -1
    for index in sorted(keep):
        if index != previous + 1:
            output.append("...")
        output.append(lines[index].rstrip())
        previous = index
    if previous != -1 and previous < len(lines) - 1:
        output.append("...")
    if len(output) > maximum_lines:
        dropped = len(output) - maximum_lines
        output = [*output[:maximum_lines], f"... ({dropped} more lines)"]
    return "\n".join(output)


def strip_address_column(line: str) -> str:
    """Drop the leading ``xxxxxxxx `` address column of a sly2 compare report line."""
    prefix = line[: ADDRESS_COLUMN_WIDTH - 1]
    if len(line) >= ADDRESS_COLUMN_WIDTH and (prefix.strip() == "" or all(c in "0123456789abcdef" for c in prefix)):
        return line[ADDRESS_COLUMN_WIDTH:]
    return line


def diff_from_compare_report(report: str, maximum_lines: int = DEFAULT_MAXIMUM_LINES) -> str:
    """The condensed diff of a sly2 compare report (header line first, then the side-by-side body)."""
    body = report.splitlines()[1:]
    return condense([strip_address_column(line) for line in body], maximum_lines=maximum_lines)
