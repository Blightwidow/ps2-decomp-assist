"""The assist loop: sample candidates, verify, then repair the closest failure for a few rounds. Model-free and testable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eval.generate import build_messages
from pipeline.repair_data import REPAIR_SYSTEM_PROMPT, feedback_text, render_repair_prompt, truncate_diff

DIFF_LINE_BUDGET = 40
Sampler = Callable[[list[dict[str, str]], int], list[str]]  # (messages, n) -> n texts
Verifier = Callable[[list[str]], list[dict[str, Any]]]  # texts -> verifier results (with diffs)


@dataclass
class Outcome:
    status: str  # "exact" | "closest" | "none"
    code: str
    similarity: float
    diff: str
    rounds_used: int
    candidates_tried: int
    notes: list[str] = field(default_factory=list)


def is_match(result: dict[str, Any]) -> bool:
    return bool(result["exact"] and result["rule_clean"])


def rank(result: dict[str, Any]) -> tuple[bool, bool, float]:
    return (is_match(result), bool(result["compiled"]), float(result["similarity"]))


def best_attempt(texts: list[str], results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]] | None:
    """The best non-empty attempt: matches first, then compiled ones, then by similarity."""
    scored = [(text, result) for text, result in zip(texts, results, strict=True) if text.strip()]
    return max(scored, key=lambda pair: rank(pair[1])) if scored else None


def repair_messages(row: dict[str, Any], candidate: str, result: dict[str, Any]) -> list[dict[str, str]]:
    record = {"compiled": result["compiled"], "note": result.get("note", ""), "diff": result.get("diff", "")}
    feedback = truncate_diff(feedback_text(record), DIFF_LINE_BUDGET)
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": render_repair_prompt(row, candidate, feedback)},
    ]


def run_assist(
    row: dict[str, Any], sample: Sampler, sample_repair: Sampler, verify: Verifier, n: int, rounds: int
) -> Outcome:
    """First round: ``n`` samples of the plain prompt. Each later round: ``n`` repairs of the closest failure so far."""
    texts = sample(build_messages(row, []), n)
    results = verify(texts)
    tried = len(texts)
    best = best_attempt(texts, results)
    used = 0
    while best is not None and not is_match(best[1]) and used < rounds:
        used += 1
        texts = sample_repair(repair_messages(row, best[0], best[1]), n)
        results = verify(texts)
        tried += len(texts)
        candidate = best_attempt(texts, results)
        if candidate is not None and rank(candidate[1]) > rank(best[1]):
            best = candidate
    if best is None:
        return Outcome("none", "", 0.0, "", used, tried, ["the model produced no code"])
    text, result = best
    if is_match(result):
        return Outcome("exact", text, 1.0, "", used, tried)
    notes = [] if result["compiled"] else ["closest candidate does not compile: " + str(result.get("note", ""))[:300]]
    return Outcome("closest", text, float(result["similarity"]), str(result.get("diff", "")), used, tried, notes)
