from eval.metrics import pass_at_k, size_bucket, summarize


def test_pass_at_k_extremes() -> None:
    assert pass_at_k(8, 0, 4) == 0.0
    assert pass_at_k(8, 8, 4) == 1.0


def test_pass_at_one_is_the_success_fraction() -> None:
    assert abs(pass_at_k(10, 3, 1) - 0.3) < 1e-9


def test_pass_at_k_is_monotonic() -> None:
    assert pass_at_k(8, 2, 1) < pass_at_k(8, 2, 4) < pass_at_k(8, 2, 8)


def test_size_buckets() -> None:
    assert [size_bucket(size) for size in (50, 150, 399)] == ["<=100", "<=200", "<=400"]


def test_summary_counts_only_rule_clean_exact() -> None:
    good = {"compiled": True, "exact": True, "rule_clean": True, "similarity": 1.0}
    dirty = {"compiled": True, "exact": True, "rule_clean": False, "similarity": 1.0}
    rows = [
        {"game": "sly1", "size_bytes": 40, "results": [good]},
        {"game": "sly1", "size_bytes": 40, "results": [dirty]},
    ]
    assert summarize(rows, (1,))["all"]["exact_rule_clean@1"] == 0.5
