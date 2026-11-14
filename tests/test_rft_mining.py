from pipeline.rft_mining import keep_exact_candidates, normalize, select_balanced_rows

ROW = {"game": "sly1", "function_id": "f1", "c_text": "void f() { a = 1; }"}


def result(exact: bool = True, rule_clean: bool = True) -> dict[str, object]:
    return {"exact": exact, "rule_clean": rule_clean}


def test_whitespace_differences_are_not_novel() -> None:
    kept = keep_exact_candidates(ROW, ["void f() {\n  a = 1;\n}"], [result()])
    assert [record["novel"] for record in kept] == [False]


def test_a_different_text_is_novel() -> None:
    kept = keep_exact_candidates(ROW, ["void f() { a = 1 + 0; }"], [result()])
    assert [record["novel"] for record in kept] == [True]


def test_only_rule_clean_exact_candidates_are_kept() -> None:
    kept = keep_exact_candidates(ROW, ["x", "y", "z"], [result(exact=False), result(rule_clean=False), result()])
    assert [record["candidate"] for record in kept] == ["z"]


def test_duplicates_after_normalization_are_kept_once() -> None:
    assert len(keep_exact_candidates(ROW, ["a  b", "a b"], [result(), result()])) == 1


def test_normalize_collapses_whitespace() -> None:
    assert normalize(" a \n b ") == "a b"


def test_balanced_selection_takes_the_same_count_per_size_bucket() -> None:
    rows = [
        {"deduped_slice": True, "rule_clean": True, "size_bytes": size, "function_id": f"{size}-{index}"}
        for size in (50, 150, 300)
        for index in range(10)
    ]
    chosen = select_balanced_rows(rows, 6, 0)
    assert sorted(row["size_bytes"] for row in chosen) == [50, 50, 150, 150, 300, 300]
