"""Near-duplicate detection: masked-assembly and C-token shingles, exact and Jaccard links."""

from __future__ import annotations

import re
from collections import defaultdict

REGISTER_PATTERN = re.compile(r"\$[a-z0-9]+")
LABEL_PATTERN = re.compile(r"\.L\d+")
HEX_PATTERN = re.compile(r"-?0x[0-9a-fA-F]+")
DECIMAL_PATTERN = re.compile(r"(?<![\w$.])-?\d+")
SYMBOL_PATTERN = re.compile(r"\b(?:func|D|B|jtbl|g|sym)_\w+|%(?:hi|lo|gp_rel)\([^)]*\)")
C_TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+\.?\d*[fF]?|->|==|!=|<=|>=|&&|\|\||<<|>>|[^\s\w]")
C_KEYWORDS = frozenset(
    [
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "return",
        "break",
        "continue",
        "goto",
        "sizeof",
        "void",
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "unsigned",
        "signed",
        "struct",
        "union",
        "enum",
        "typedef",
        "static",
        "extern",
        "const",
        "volatile",
        "class",
        "virtual",
        "new",
        "delete",
        "this",
    ]
)


def mask_assembly_line(line: str) -> str:
    """Mnemonic plus operand classes: registers -> R, immediates -> I, labels -> L, symbols -> S."""
    masked = SYMBOL_PATTERN.sub("S", line)
    masked = LABEL_PATTERN.sub("L", masked)
    masked = REGISTER_PATTERN.sub("R", masked)
    masked = HEX_PATTERN.sub("I", masked)
    return DECIMAL_PATTERN.sub("I", masked)


def masked_assembly_tokens(assembly_text: str) -> list[str]:
    return [mask_assembly_line(line) for line in assembly_text.splitlines() if not line.endswith(":")]


def normalized_c_tokens(c_text: str) -> list[str]:
    tokens: list[str] = []
    for token in C_TOKEN_PATTERN.findall(c_text):
        if token in C_KEYWORDS or not (token[0].isalpha() or token[0] == "_" or token[0].isdigit()):
            tokens.append(token)
        elif token[0].isdigit():
            tokens.append("NUM")
        else:
            tokens.append("ID")
    return tokens


def shingles(tokens: list[str], size: int) -> frozenset[tuple[str, ...]]:
    if len(tokens) < size:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def similar_pairs(
    shingle_sets: list[frozenset[tuple[str, ...]]],
    threshold: float,
    minimum_shingles: int = 8,
    document_frequency_cap: int = 400,
) -> list[tuple[int, int]]:
    """Index pairs with Jaccard >= threshold. Sets smaller than ``minimum_shingles`` are left to exact matching."""
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, shingle_set in enumerate(shingle_sets):
        if len(shingle_set) >= minimum_shingles:
            for shingle in shingle_set:
                postings[shingle].append(index)
    shared_counts: dict[tuple[int, int], int] = defaultdict(int)
    for members in postings.values():
        if len(members) > document_frequency_cap:
            continue
        for position, first in enumerate(members):
            for second in members[position + 1 :]:
                shared_counts[(first, second)] += 1
    pairs: list[tuple[int, int]] = []
    for (first, second), shared in shared_counts.items():
        union = len(shingle_sets[first]) + len(shingle_sets[second]) - shared
        if union and shared / union >= threshold:
            pairs.append((first, second))
    return pairs


def exact_duplicate_groups(keys: list[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    return [members for members in groups.values() if len(members) > 1]
