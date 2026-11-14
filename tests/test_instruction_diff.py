from pipeline.instruction_diff import (
    COLUMN_WIDTH,
    condense,
    diff_from_compare_report,
    side_by_side,
    strip_address_column,
)


def fake_disassemble(word: int) -> str:
    return f"op{word}"


def test_identical_words_have_no_changed_lines() -> None:
    assert condense(side_by_side([1, 2, 3], [1, 2, 3], fake_disassemble)) == ""


def test_changed_word_is_marked() -> None:
    lines = side_by_side([1, 2, 3], [1, 9, 3], fake_disassemble)
    assert lines[1].split() == ["op2", "|", "op9"]


def test_missing_instruction_is_marked_as_target_only() -> None:
    lines = side_by_side([1, 2, 3], [1, 3], fake_disassemble)
    assert any(line.split()[:2] == ["op2", "<"] for line in lines)


def test_extra_instruction_is_marked_as_ours_only() -> None:
    lines = side_by_side([1, 3], [1, 2, 3], fake_disassemble)
    assert any(line.split() == [">", "op2"] for line in lines)


def test_condense_keeps_one_line_of_context_and_marks_skipped_runs() -> None:
    target = list(range(10))
    ours = [*target[:5], 99, *target[6:]]
    output = condense(side_by_side(target, ours, fake_disassemble)).splitlines()
    assert [output[0], output[-1]] == ["...", "..."]
    assert len(output) == 5


def test_condense_truncates_long_diffs_with_a_dropped_count() -> None:
    lines = [f"{'a':<{COLUMN_WIDTH}} | b" for _ in range(50)]
    assert condense(lines, maximum_lines=10).splitlines()[-1] == "... (40 more lines)"


def test_address_column_is_stripped_from_compare_lines() -> None:
    assert strip_address_column("002a1b30 addiu $sp  | addiu $sp") == "addiu $sp  | addiu $sp"


def test_padding_only_address_column_is_stripped() -> None:
    assert strip_address_column("         " + "x") == "x"


def test_compare_report_diff_skips_header_and_matching_lines() -> None:
    changed = f"002a1b34 {'lw $v0':<{COLUMN_WIDTH}} | lw $v1"
    matching = f"002a1b30 {'nop':<{COLUMN_WIDTH}}   nop"
    report = f"score 1  target 2 instr\n{matching}\n{changed}"
    assert diff_from_compare_report(report).splitlines()[-1].split() == ["lw", "$v0", "|", "lw", "$v1"]
