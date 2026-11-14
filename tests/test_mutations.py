from pipeline.mutations import (
    compound_to_expanded,
    expanded_to_compound,
    flip_relational,
    increment_to_prefix,
    mutants,
    remove_cast,
    swap_adjacent_statements,
    swap_logical_operands,
    swap_type,
    zero_compare_to_negation,
)


def test_swap_type_changes_one_declaration_per_variant() -> None:
    assert swap_type("void f() {\n    int count = 0;\n    float scale;\n}") == [
        "void f() {\n    unsigned int count = 0;\n    float scale;\n}",
        "void f() {\n    int count = 0;\n    int scale;\n}",
    ]


def test_remove_cast_drops_the_cast() -> None:
    assert remove_cast("x = (float)y;") == ["x = y;"]


def test_zero_compare_becomes_negation() -> None:
    assert zero_compare_to_negation("if (flag == 0) return;") == ["if (!flag) return;"]


def test_logical_operands_are_swapped() -> None:
    assert swap_logical_operands("if (a && b) go();") == ["if (b && a) go();"]


def test_relational_operator_is_mirrored_without_touching_shifts() -> None:
    assert flip_relational("if (a <= b) x = c << 2;") == ["if (a >= b) x = c << 2;"]


def test_postfix_statement_increment_becomes_prefix() -> None:
    assert increment_to_prefix("    i++;") == ["    ++i;"]


def test_compound_and_expanded_assignment_round_trip() -> None:
    expanded = compound_to_expanded("    total += step;")
    assert expand_back(expanded) == ["    total += step;"]


def expand_back(variants: list[str]) -> list[str]:
    return [back for variant in variants for back in expanded_to_compound(variant)]


def test_adjacent_simple_statements_are_swapped() -> None:
    assert swap_adjacent_statements("{\n    a = 1;\n    b = 2;\n}") == ["{\n    b = 2;\n    a = 1;\n}"]


def test_statement_swap_ignores_returns_and_braces() -> None:
    assert swap_adjacent_statements("{\n    a = 1;\n    return a;\n}") == []


def test_mutants_never_include_the_original_and_respect_count() -> None:
    source = "void f() {\n    int a = 0;\n    int b = 1;\n    if (a < b && b == 0) a++;\n}"
    produced = mutants(source, 3, 0)
    assert len(produced) == 3 and all(variant != source for _, variant in produced)


def test_mutants_are_deterministic_for_a_seed() -> None:
    source = "void f() {\n    int a = 0;\n    if (a < 1) a++;\n}"
    assert mutants(source, 2, 5) == mutants(source, 2, 5)


def test_swap_type_handles_every_type_the_pattern_matches() -> None:
    source = "\n".join(
        f"    {name} v{index} = 0;"
        for index, name in enumerate(
            [
                "unsigned int",
                "unsigned short",
                "unsigned char",
                "int",
                "short",
                "char",
                "float",
                "s8",
                "u8",
                "s16",
                "u16",
                "s32",
                "u32",
            ]
        )
    )
    assert len(swap_type(source)) == 13
