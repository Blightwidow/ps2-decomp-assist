from pipeline.splits import assign_splits, audit_no_leakage, connected_components


def test_links_and_shared_groups_merge_components() -> None:
    components = connected_components(5, [(0, 1)], ["a", "b", "c", "b", "d"])
    assert {frozenset(component) for component in components} == {frozenset({0, 1, 3}), frozenset({2}), frozenset({4})}


def test_components_stay_whole_in_assignment() -> None:
    components = [[0, 1, 2], [3], [4, 5], [6], [7], [8], [9], [10, 11, 12, 13]]
    assignment = assign_splits(components, test_fraction=0.3, validation_fraction=0.2, seed=1)
    assert audit_no_leakage(components, assignment, [str(index) for index in range(14)]) == []


def test_oversized_components_go_to_train_only() -> None:
    components = [list(range(30)), [30], [31]]
    assignment = assign_splits(components, test_fraction=0.5, validation_fraction=0.0, train_only_component_size=25)
    assert {assignment[index] for index in range(30)} == {"train"}


def test_audit_detects_split_component_and_exact_duplicates() -> None:
    components = [[0, 1], [2]]
    assignment = {0: "train", 1: "test", 2: "test"}
    problems = audit_no_leakage(components, assignment, ["x", "y", "x"])
    assert len(problems) == 2


def test_assignment_is_deterministic_for_a_seed() -> None:
    components = [[index] for index in range(50)]
    assert assign_splits(components, seed=3) == assign_splits(components, seed=3)


def test_small_duplicate_cluster_drops_the_train_side() -> None:
    from pipeline.splits import resolve_cross_split_duplicates

    resolved = resolve_cross_split_duplicates({0: "train", 1: "test", 2: "train"}, [[0, 1]])
    assert resolved == {0: "dropped", 1: "test", 2: "train"}


def test_boilerplate_cluster_is_removed_from_evaluation_sides() -> None:
    from pipeline.splits import resolve_cross_split_duplicates

    assignment = {index: "train" for index in range(30)} | {30: "test"}
    resolved = resolve_cross_split_duplicates(assignment, [list(range(31))], boilerplate_size=25)
    assert resolved[30] == "dropped" and resolved[0] == "train"
