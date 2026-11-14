"""Retrieval baseline: copy the C of the nearest training function (masked-asm shingles), retargeting its symbols."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pipeline.similarity import masked_assembly_tokens, shingles

SYMBOL_OPERAND_PATTERN = re.compile(r"(?:\bjal\s+|%(?:hi|lo)\()([A-Za-z_]\w*)")
POSTING_FREQUENCY_CAP = 600


def assembly_symbols(assembly: str) -> list[str]:
    """Call targets and %hi/%lo symbols in order of first appearance."""
    seen: dict[str, None] = {}
    for name in SYMBOL_OPERAND_PATTERN.findall(assembly):
        seen.setdefault(name, None)
    return list(seen)


def target_name(row: dict[str, Any]) -> str:
    return str(row["function_id"] if row["game"] == "sly1" else row["definition_name"])


def retarget(neighbor: dict[str, Any], target: dict[str, Any]) -> str:
    """The neighbor's C with its function and symbols renamed to the target's (pairwise by order of appearance)."""
    mapping: dict[str, str] = {target_name(neighbor): target_name(target)}
    for source_symbol, target_symbol in zip(
        assembly_symbols(neighbor["assembly"]), assembly_symbols(target["assembly"]), strict=False
    ):
        mapping.setdefault(source_symbol, target_symbol)
    return re.sub(r"[A-Za-z_]\w*", lambda match: mapping.get(match.group(0), match.group(0)), neighbor["c_text"])


class Retriever:
    def __init__(self, training_rows: list[dict[str, Any]]) -> None:
        self.rows = training_rows
        self.shingle_sets = [shingles(masked_assembly_tokens(row["assembly"]), 4) for row in training_rows]
        self.postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for index, shingle_set in enumerate(self.shingle_sets):
            for shingle in shingle_set:
                self.postings[shingle].append(index)

    def nearest(self, row: dict[str, Any], count: int = 1) -> list[dict[str, Any]]:
        query = shingles(masked_assembly_tokens(row["assembly"]), 4)
        shared: dict[int, int] = defaultdict(int)
        for shingle in query:
            members = self.postings.get(shingle, [])
            if len(members) > POSTING_FREQUENCY_CAP:
                continue
            for index in members:
                shared[index] += 1
        scored = sorted(
            (
                (shared_count / (len(query) + len(self.shingle_sets[index]) - shared_count), index)
                for index, shared_count in shared.items()
            ),
            reverse=True,
        )
        return [self.rows[index] for _, index in scored[:count]]
