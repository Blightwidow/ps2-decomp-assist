from pipeline.records import assembly_tags, scan_rules


def test_scan_flags_inline_asm_and_volatile() -> None:
    assert scan_rules('void f() { asm("nop"); volatile int x; }') == ["inline_asm", "volatile"]


def test_scan_clean_function() -> None:
    assert scan_rules("int f(int a) { return a + 1; }") == []


def test_tags_for_leaf_float_branching_code() -> None:
    assembly = "add.s $f0, $f12, $f13\nbeqz $a0, .L1\nnop\n.L1:\njr $ra\nnop"
    assert assembly_tags(assembly, 5) == ["leaf", "float", "branches", "trivial"]


def test_tags_for_calling_function() -> None:
    assert "calls" in assembly_tags("jal func_00100000\nnop", 20)
