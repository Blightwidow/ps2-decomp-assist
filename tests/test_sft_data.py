from typing import Any

from pipeline.sft_data import select_all, select_stratified, to_chat_example


def make_row(index: int, game: str, size_bytes: int, deduped: bool = True) -> dict[str, Any]:
    return {
        "game": game,
        "function_id": f"func_{index:08X}",
        "size_bytes": size_bytes,
        "deduped_slice": deduped,
        "rule_clean": True,
        "assembly": "nop",
        "context": "",
        "c_text": "void f() {}",
    }


def test_selection_skips_rows_outside_deduped_slice() -> None:
    rows = [make_row(index, "sly1", 40, deduped=index % 2 == 0) for index in range(20)]
    assert all(row["deduped_slice"] for row in select_stratified(rows, 20))


def test_selection_covers_every_stratum() -> None:
    rows = [make_row(index, "sly1", 40) for index in range(50)] + [
        make_row(100 + index, "sly2", 300) for index in range(5)
    ]
    assert {row["game"] for row in select_stratified(rows, 10)} == {"sly1", "sly2"}


def test_selection_respects_count() -> None:
    rows = [make_row(index, "sly1", 40) for index in range(50)]
    assert len(select_stratified(rows, 10)) == 10


def test_chat_example_ends_with_the_reference_function() -> None:
    assert to_chat_example(make_row(1, "sly1", 40))["messages"][-1] == {
        "role": "assistant",
        "content": "void f() {}",
    }


def test_select_all_keeps_non_deduped_rule_clean_rows() -> None:
    rows = [make_row(index, "sly1", 40, deduped=False) for index in range(7)]
    assert len(select_all(rows)) == 7
