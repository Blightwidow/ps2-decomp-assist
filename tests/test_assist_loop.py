from typing import Any

from decomp_assist.loop import best_attempt, is_match, repair_messages, run_assist

ROW = {"game": "sly1", "function_id": "func_00000000", "size_bytes": 8, "context": "", "assembly": "nop"}


def result(exact: bool = False, compiled: bool = True, similarity: float = 0.5, diff: str = "a | b") -> dict[str, Any]:
    return {
        "exact": exact,
        "rule_clean": True,
        "compiled": compiled,
        "similarity": similarity,
        "diff": diff,
        "note": "boom",
    }


def scripted_verifier(table: dict[str, dict[str, Any]]):  # type: ignore[no-untyped-def]
    return lambda texts: [table[text] for text in texts]


def test_first_round_exact_returns_without_repair() -> None:
    calls: list[str] = []

    def repair(messages: list[dict[str, str]], n: int) -> list[str]:
        calls.append("repair")
        return []

    outcome = run_assist(
        ROW, lambda messages, n: ["good"], repair, scripted_verifier({"good": result(exact=True)}), 1, 2
    )
    assert (outcome.status, outcome.rounds_used, calls) == ("exact", 0, [])


def test_repair_round_can_reach_an_exact_match() -> None:
    table = {"bad": result(), "fixed": result(exact=True)}
    outcome = run_assist(ROW, lambda m, n: ["bad"], lambda m, n: ["fixed"], scripted_verifier(table), 1, 2)
    assert (outcome.status, outcome.code, outcome.rounds_used) == ("exact", "fixed", 1)


def test_rounds_are_capped_and_the_closest_attempt_is_reported() -> None:
    table = {"bad": result(similarity=0.4), "worse": result(similarity=0.2), "better": result(similarity=0.7)}
    outputs = iter([["worse"], ["better"]])
    outcome = run_assist(ROW, lambda m, n: ["bad"], lambda m, n: next(outputs), scripted_verifier(table), 1, 2)
    assert (outcome.status, outcome.code, outcome.rounds_used, outcome.candidates_tried) == ("closest", "better", 2, 3)


def test_empty_generation_gives_status_none() -> None:
    outcome = run_assist(ROW, lambda m, n: [""], lambda m, n: [""], scripted_verifier({"": result()}), 1, 2)
    assert outcome.status == "none"


def test_compiled_attempt_beats_a_higher_similarity_one_that_does_not_compile() -> None:
    pair = best_attempt(["a", "b"], [result(compiled=False, similarity=0.9), result(similarity=0.3)])
    assert pair is not None and pair[0] == "b"


def test_exact_but_rule_breaking_is_not_a_match() -> None:
    assert not is_match({**result(exact=True), "rule_clean": False})


def test_repair_prompt_carries_the_diff_of_the_failed_candidate() -> None:
    messages = repair_messages(ROW, "bad", result(diff="x | y"))
    assert "CANDIDATE:\nbad\nDIFF:\nx | y" in messages[1]["content"]


def test_matching_seed_returns_without_calling_the_model() -> None:
    def never(messages: list[dict[str, str]], n: int) -> list[str]:
        raise AssertionError("the model was called")

    outcome = run_assist(ROW, never, never, scripted_verifier({"draft": result(exact=True)}), 1, 2, seeds=["draft"])
    assert (outcome.status, outcome.code, outcome.candidates_tried) == ("exact", "draft", 1)


def test_closer_seed_is_the_one_repaired() -> None:
    repaired: list[str] = []

    def repair(messages: list[dict[str, str]], n: int) -> list[str]:
        repaired.append(messages[1]["content"])
        return ["fixed"]

    table = {"draft": result(similarity=0.9), "sampled": result(similarity=0.2), "fixed": result(exact=True)}
    run_assist(ROW, lambda m, n: ["sampled"], repair, scripted_verifier(table), 1, 1, seeds=["draft"])
    assert "CANDIDATE:\ndraft\n" in repaired[0]


def test_blank_seed_is_ignored() -> None:
    outcome = run_assist(
        ROW, lambda m, n: ["good"], lambda m, n: [], scripted_verifier({"good": result(exact=True)}), 1, 0, seeds=[" "]
    )
    assert (outcome.code, outcome.candidates_tried) == ("good", 1)
