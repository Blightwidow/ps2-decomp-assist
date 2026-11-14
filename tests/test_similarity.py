from pipeline.similarity import (
    exact_duplicate_groups,
    mask_assembly_line,
    masked_assembly_tokens,
    normalized_c_tokens,
    shingles,
    similar_pairs,
)


def test_masking_removes_registers_immediates_and_symbols() -> None:
    assert mask_assembly_line("lw $v0, 0x708($a0)") == "lw R, I(R)"
    assert mask_assembly_line("jal func_00100100") == "jal S"
    assert mask_assembly_line("beqz $a0, .L3") == "beqz R, L"
    assert mask_assembly_line("addiu $sp, $sp, -32") == "addiu R, R, I"


def test_masked_tokens_skip_label_definitions() -> None:
    assert masked_assembly_tokens("nop\n.L1:\njr $ra") == ["nop", "jr R"]


def test_c_normalization_keeps_keywords_and_structure() -> None:
    assert normalized_c_tokens("int f(int a) { return a + 1; }")[:3] == ["int", "ID", "("]
    assert "NUM" in normalized_c_tokens("x = 12;")


def test_similar_pairs_finds_near_duplicates_only() -> None:
    base = [f"op{index}" for index in range(20)]
    near = base[:19] + ["different"]
    far = [f"other{index}" for index in range(20)]
    sets = [shingles(tokens, 2) for tokens in (base, near, far)]
    assert similar_pairs(sets, 0.7) == [(0, 1)]


def test_exact_duplicate_groups() -> None:
    assert exact_duplicate_groups(["a", "b", "a", "c", "a"]) == [[0, 2, 4]]
