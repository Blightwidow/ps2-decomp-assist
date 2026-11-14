"""Tests use hand-written synthetic assembly only (no disc-derived content)."""

from pipeline.asm_parse import normalize_assembly, parse_assembly_functions

SYNTHETIC_ASSEMBLY = """
.section .text
.align 3
nonmatching func_00100000, 0x10

glabel func_00100000
    /* 000000 00100000 03E00008 */  jr         $ra
    /* 000004 00100004 00000000 */   nop
endlabel func_00100000

glabel Example_Function
    /* 000100 00100100 10800003 */  beqz       $a0, .L00100110
    /* 000104 00100104 00000000 */   nop
    /* 000108 00100108 0C040040 */  jal        func_00100100
    /* 00010C 0010010C 00000000 */   nop
.L00100110:
    /* 000110 00100110 03E00008 */  jr         $ra
    /* 000114 00100114 00000000 */   nop
.size Example_Function, . - Example_Function
"""


def test_parses_each_function_with_address_and_size() -> None:
    functions = parse_assembly_functions(SYNTHETIC_ASSEMBLY)
    assert [(function.name, function.address, function.size_bytes) for function in functions] == [
        ("func_00100000", 0x100000, 8),
        ("Example_Function", 0x100100, 24),
    ]


def test_normalization_renumbers_local_labels() -> None:
    function = parse_assembly_functions(SYNTHETIC_ASSEMBLY)[1]
    text = normalize_assembly(function)
    assert "beqz $a0, .L1" in text and text.splitlines()[4] == ".L1:"


def test_normalization_applies_symbol_aliases_on_whole_identifiers() -> None:
    function = parse_assembly_functions(SYNTHETIC_ASSEMBLY)[1]
    text = normalize_assembly(function, {"func_00100100": "func_ALIASED"})
    assert "jal func_ALIASED" in text


def test_aliases_do_not_touch_substrings() -> None:
    function = parse_assembly_functions(SYNTHETIC_ASSEMBLY)[1]
    text = normalize_assembly(function, {"func_001001": "X"})
    assert "func_00100100" in text
