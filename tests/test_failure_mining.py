from typing import Any

from pipeline.failure_mining import failure_record, select_failures, select_unseen_rows


def result(exact: bool = False, compiled: bool = True) -> dict[str, Any]:
    return {
        "exact": exact,
        "compiled": compiled,
        "status": "EXACT" if exact else "DIFFER",
        "similarity": 0.5,
        "note": "boom",
    }


def test_exact_candidates_are_not_failures() -> None:
    assert select_failures(["a", "b"], [result(exact=True), result()], 5, 0) == [("b", result())]


def test_duplicate_and_empty_candidates_are_dropped() -> None:
    picked = select_failures(["a", "a", " ", "b"], [result()] * 4, 5, 0)
    assert sorted(text for text, _ in picked) == ["a", "b"]


def test_failures_are_capped_per_function() -> None:
    assert len(select_failures(["a", "b", "c"], [result()] * 3, 2, 0)) == 2


def test_compile_failure_record_keeps_the_note_and_no_diff() -> None:
    row = {"game": "sly1", "function_id": "func_1"}
    record = failure_record(row, "x", result(compiled=False), "model:t")
    assert (record["note"], record["diff"]) == ("boom", "")


def test_differing_record_drops_the_note() -> None:
    row = {"game": "sly1", "function_id": "func_1"}
    assert failure_record(row, "x", {**result(), "diff": "d"}, "model:t")["note"] == ""


def test_unseen_rows_exclude_listed_ids_and_non_deduped_rows() -> None:
    rows = [
        {"game": "sly1", "function_id": f"f{index}", "deduped_slice": index != 1, "rule_clean": True}
        for index in range(4)
    ]
    chosen = select_unseen_rows(rows, {"sly1:f0"}, 10, 0)
    assert sorted(row["function_id"] for row in chosen) == ["f2", "f3"]
