from typing import Any

from pipeline.mutation_mining import keep_differing_mutants


def result(status: str = "DIFFER", compiled: bool = True, rule_clean: bool = True) -> dict[str, Any]:
    return {"status": status, "compiled": compiled, "rule_clean": rule_clean}


def test_only_compiled_rule_clean_differing_mutants_are_kept() -> None:
    results = [result("EXACT"), result(compiled=False), result(rule_clean=False), result()]
    assert keep_differing_mutants(["a", "b", "c", "d"], results, 5, 0) == [("d", 3)]


def test_kept_mutants_are_capped_per_function() -> None:
    assert len(keep_differing_mutants(["a", "b", "c"], [result()] * 3, 2, 0)) == 2
