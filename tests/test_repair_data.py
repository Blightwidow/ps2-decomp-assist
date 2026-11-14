from typing import Any

from pipeline.repair_data import (
    cap_synthetic,
    feedback_text,
    split_by_function,
    to_repair_example,
    truncate_diff,
)


def make_record(source: str = "model:x", function_id: str = "f1", compiled: bool = True) -> dict[str, Any]:
    return {
        "game": "sly1",
        "function_id": function_id,
        "source": source,
        "candidate": "bad",
        "compiled": compiled,
        "note": "error: boom",
        "diff": "a | b",
    }


ROW = {"game": "sly1", "function_id": "f1", "size_bytes": 8, "context": "", "assembly": "nop", "c_text": "void f() {}"}


def test_compile_failure_feedback_is_the_compiler_message() -> None:
    assert feedback_text(make_record(compiled=False)) == "COMPILE ERROR: error: boom"


def test_differing_feedback_is_the_instruction_diff() -> None:
    assert feedback_text(make_record()) == "a | b"


def test_short_diff_is_not_truncated() -> None:
    assert truncate_diff("a\nb", 5) == "a\nb"


def test_long_diff_is_cut_with_a_count() -> None:
    assert truncate_diff("a\nb\nc\nd", 2).splitlines() == ["a", "b", "... (2 more lines)"]


def test_example_ends_with_ground_truth_and_carries_the_candidate() -> None:
    example = to_repair_example(ROW, make_record(), len, 10_000)
    assert example is not None
    assert example["messages"][-1]["content"] == "void f() {}" and "CANDIDATE:\nbad\nDIFF:\na | b" in str(
        example["messages"][1]["content"]
    )


def test_example_over_the_token_cap_is_dropped() -> None:
    assert to_repair_example(ROW, make_record(), len, 10) is None


def test_long_diff_is_shortened_until_the_example_fits() -> None:
    record = {**make_record(), "diff": "\n".join(f"line {index}" for index in range(100))}
    full = to_repair_example(ROW, record, len, 100_000)
    assert full is not None
    full_length = sum(len(message["content"]) for message in full["messages"])
    shortened = to_repair_example(ROW, record, len, full_length - 100)
    assert shortened is not None and "more lines" in str(shortened["messages"][1]["content"])


def test_synthetic_share_is_capped() -> None:
    records = [make_record("model:x") for _ in range(3)] + [make_record("mutation:swap_type") for _ in range(20)]
    capped = cap_synthetic(records, 0.7, 0)
    assert sum(record["source"].startswith("mutation:") for record in capped) == 7


def test_no_real_failures_means_no_synthetic_data() -> None:
    assert cap_synthetic([make_record("mutation:swap_type")], 0.7, 0) == []


def test_valid_split_never_separates_a_function() -> None:
    records = [make_record(function_id=f"f{index // 3}") for index in range(12)]
    train, valid = split_by_function(records, 4, 0)
    assert {record["function_id"] for record in train}.isdisjoint({record["function_id"] for record in valid})
