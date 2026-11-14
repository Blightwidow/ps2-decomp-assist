"""Leak-safe split assignment: whole connected components (near-duplicates + same source group) go to one side."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, element: int) -> int:
        while self.parents[element] != element:
            self.parents[element] = self.parents[self.parents[element]]
            element = self.parents[element]
        return element

    def union(self, first: int, second: int) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parents[root_second] = root_first


def connected_components(
    size: int,
    links: Iterable[tuple[int, int]],
    group_keys: Sequence[str],
) -> list[list[int]]:
    """Components over explicit links plus every pair of records that share a group key."""
    union_find = UnionFind(size)
    for first, second in links:
        union_find.union(first, second)
    first_by_group: dict[str, int] = {}
    for index, key in enumerate(group_keys):
        if key in first_by_group:
            union_find.union(first_by_group[key], index)
        else:
            first_by_group[key] = index
    members: dict[int, list[int]] = {}
    for index in range(size):
        members.setdefault(union_find.find(index), []).append(index)
    return list(members.values())


def assign_splits(
    components: list[list[int]],
    test_fraction: float = 0.10,
    validation_fraction: float = 0.05,
    train_only_component_size: int = 25,
    seed: int = 0,
) -> dict[int, str]:
    """Record index -> ``train``/``validation``/``test``.

    Big components (boilerplate clusters or huge source groups) are train-only so that test and validation stay
    de-duplicated and diverse.
    """
    total = sum(len(component) for component in components)
    assignment: dict[int, str] = {}
    candidates: list[list[int]] = []
    for component in components:
        if len(component) > train_only_component_size:
            assignment.update({index: "train" for index in component})
        else:
            candidates.append(component)
    random.Random(seed).shuffle(candidates)
    test_target, validation_target = test_fraction * total, validation_fraction * total
    test_count = validation_count = 0
    for component in candidates:
        if test_count < test_target:
            label = "test"
            test_count += len(component)
        elif validation_count < validation_target:
            label = "validation"
            validation_count += len(component)
        else:
            label = "train"
        assignment.update({index: label for index in component})
    return assignment


def resolve_cross_split_duplicates(
    assignment: dict[int, str],
    duplicate_components: list[list[int]],
    boilerplate_size: int = 25,
) -> dict[int, str]:
    """Drop (label ``dropped``) the records that would leak a duplicate across splits.

    Near-duplicate clusters only *remove* records, they never merge source groups (merging chains the whole corpus
    into one component). A boilerplate cluster (> ``boilerplate_size`` members) is removed from validation/test so
    the evaluation sets stay diverse; a small cluster is removed from the lower-priority side
    (train < validation < test).
    """
    resolved = dict(assignment)
    priority = {"train": 0, "validation": 1, "test": 2}
    for component in duplicate_components:
        labels = {resolved[index] for index in component if resolved[index] != "dropped"}
        if len(labels) < 2:
            continue
        if len(component) > boilerplate_size:
            for index in component:
                if resolved[index] in {"validation", "test"}:
                    resolved[index] = "dropped"
        else:
            keep = max(labels, key=lambda label: priority[label])
            for index in component:
                if resolved[index] not in {keep, "dropped"}:
                    resolved[index] = "dropped"
    return resolved


def audit_no_leakage(
    components: list[list[int]],
    assignment: dict[int, str],
    exact_keys: Sequence[str],
) -> list[str]:
    """Human-readable problems; empty list means clean."""
    problems: list[str] = []
    for component in components:
        labels = {assignment[index] for index in component if assignment[index] != "dropped"}
        if len(labels) > 1:
            problems.append(f"component of {len(component)} spans {sorted(labels)}")
    labels_by_key: dict[str, set[str]] = {}
    for index, key in enumerate(exact_keys):
        if assignment[index] != "dropped":
            labels_by_key.setdefault(key, set()).add(assignment[index])
    problems.extend(
        f"exact duplicate key spans {sorted(labels)}" for labels in labels_by_key.values() if len(labels) > 1
    )
    return problems
