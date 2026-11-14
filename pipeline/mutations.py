"""Source mutation operators for synthetic T2 repair data.

Each operator lists every variant of a function's C++ text that changes one thing the compiler may emit differently
(operand form, comparison form, type width, statement order). Whether a variant really moves the bytes is decided later,
by compiling it: variants that stay EXACT or stop compiling are dropped.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

Operator = Callable[[str], list[str]]

TYPE_COUNTERPARTS = {
    "int": "unsigned int",
    "unsigned int": "int",
    "s32": "u32",
    "u32": "s32",
    "s16": "u16",
    "u16": "s16",
    "s8": "u8",
    "u8": "s8",
    "short": "unsigned short",
    "unsigned short": "short",
    "char": "unsigned char",
    "unsigned char": "char",
    "float": "int",
}
TYPE_PATTERN = re.compile(
    r"\b(unsigned int|unsigned short|unsigned char|int|short|char|float|[su](?:8|16|32))\b(?=\s+\w+\s*[=;,)])"
)
CAST_PATTERN = re.compile(r"\((?:unsigned int|unsigned short|unsigned char|int|short|char|float|[su](?:8|16|32))\)\s*")
ZERO_COMPARE_PATTERN = re.compile(r"\b(\w+)\s*==\s*0\b")
NEGATION_PATTERN = re.compile(r"!\s*(\w+)\b(?!\s*[(\w])")
LOGICAL_PATTERN = re.compile(r"\b(\w+)\s*(&&|\|\|)\s*(\w+)\b")
RELATIONAL_PATTERN = re.compile(r"(?<![<>=!-])(<=|>=|<|>)(?![<>=])")
INCREMENT_PATTERN = re.compile(r"^(\s*)(\w+)(\+\+|--);\s*$", re.MULTILINE)
PREFIX_INCREMENT_PATTERN = re.compile(r"^(\s*)(\+\+|--)(\w+);\s*$", re.MULTILINE)
COMPOUND_PATTERN = re.compile(r"^(\s*)(\w+)\s*([+\-*|&^])=\s*([^;]+);\s*$", re.MULTILINE)
EXPANDED_PATTERN = re.compile(r"^(\s*)(\w+)\s*=\s*\2\s*([+\-*|&^])\s*([^;]+);\s*$", re.MULTILINE)
RELATIONAL_FLIP = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}
SIMPLE_STATEMENT_PATTERN = re.compile(r"^\s*[\w\[\]\->.*]+\s*=\s*[^;{}]+;\s*$")


def replace_each_match(text: str, pattern: re.Pattern[str], replacement: Callable[[re.Match[str]], str]) -> list[str]:
    """One variant per match of ``pattern``, each with only that match replaced."""
    return [text[: match.start()] + replacement(match) + text[match.end() :] for match in pattern.finditer(text)]


def swap_type(text: str) -> list[str]:
    return replace_each_match(text, TYPE_PATTERN, lambda match: TYPE_COUNTERPARTS[match.group(1)])


def remove_cast(text: str) -> list[str]:
    return replace_each_match(text, CAST_PATTERN, lambda match: "")


def zero_compare_to_negation(text: str) -> list[str]:
    return replace_each_match(text, ZERO_COMPARE_PATTERN, lambda match: f"!{match.group(1)}")


def negation_to_zero_compare(text: str) -> list[str]:
    return replace_each_match(text, NEGATION_PATTERN, lambda match: f"{match.group(1)} == 0")


def swap_logical_operands(text: str) -> list[str]:
    return replace_each_match(
        text, LOGICAL_PATTERN, lambda match: f"{match.group(3)} {match.group(2)} {match.group(1)}"
    )


def flip_relational(text: str) -> list[str]:
    return replace_each_match(text, RELATIONAL_PATTERN, lambda match: RELATIONAL_FLIP[match.group(1)])


def increment_to_prefix(text: str) -> list[str]:
    return replace_each_match(
        text, INCREMENT_PATTERN, lambda match: f"{match.group(1)}{match.group(3)}{match.group(2)};"
    ) + replace_each_match(
        text, PREFIX_INCREMENT_PATTERN, lambda match: f"{match.group(1)}{match.group(3)}{match.group(2)};"
    )


def compound_to_expanded(text: str) -> list[str]:
    return replace_each_match(
        text,
        COMPOUND_PATTERN,
        lambda match: f"{match.group(1)}{match.group(2)} = {match.group(2)} {match.group(3)} {match.group(4).strip()};",
    )


def expanded_to_compound(text: str) -> list[str]:
    return replace_each_match(
        text,
        EXPANDED_PATTERN,
        lambda match: f"{match.group(1)}{match.group(2)} {match.group(3)}= {match.group(4).strip()};",
    )


def swap_adjacent_statements(text: str) -> list[str]:
    lines = text.split("\n")
    variants: list[str] = []
    for index in range(len(lines) - 1):
        if SIMPLE_STATEMENT_PATTERN.match(lines[index]) and SIMPLE_STATEMENT_PATTERN.match(lines[index + 1]):
            swapped = lines[:]
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            variants.append("\n".join(swapped))
    return variants


OPERATORS: dict[str, Operator] = {
    "swap_type": swap_type,
    "remove_cast": remove_cast,
    "zero_compare_to_negation": zero_compare_to_negation,
    "negation_to_zero_compare": negation_to_zero_compare,
    "swap_logical_operands": swap_logical_operands,
    "flip_relational": flip_relational,
    "increment_to_prefix": increment_to_prefix,
    "compound_to_expanded": compound_to_expanded,
    "expanded_to_compound": expanded_to_compound,
    "swap_adjacent_statements": swap_adjacent_statements,
}


def mutants(text: str, count: int, seed: int) -> list[tuple[str, str]]:
    """Up to ``count`` distinct ``(operator name, mutated text)`` pairs, operators and variants drawn at random."""
    generator = random.Random(seed)
    pool = [(name, variant) for name, operator in OPERATORS.items() for variant in operator(text) if variant != text]
    generator.shuffle(pool)
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    used_operators: dict[str, int] = {}
    for name, variant in pool:
        if variant in seen or used_operators.get(name, 0) >= max(1, count // 2):
            continue
        seen.add(variant)
        used_operators[name] = used_operators.get(name, 0) + 1
        chosen.append((name, variant))
        if len(chosen) == count:
            break
    return chosen
