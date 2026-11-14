"""Evaluation metrics: unbiased pass@k, aggregates by game / size bucket."""

from __future__ import annotations

from collections import defaultdict
from math import comb
from typing import Any


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    """Unbiased estimator (Chen et al. 2021). Requires ``sample_count >= k``; smaller n uses k = n."""
    k = min(k, sample_count)
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - comb(sample_count - correct_count, k) / comb(sample_count, k)


def size_bucket(size_bytes: int) -> str:
    return "<=100" if size_bytes <= 100 else "<=200" if size_bytes <= 200 else "<=400"


def summarize(per_function: list[dict[str, Any]], ks: tuple[int, ...] = (1, 4, 8)) -> dict[str, Any]:
    """``per_function`` rows: game, size_bytes, results (list of verify results, one per sample)."""

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {"functions": 0}
        summary: dict[str, Any] = {"functions": len(rows)}
        for k in ks:
            summary[f"exact_rule_clean@{k}"] = sum(
                pass_at_k(len(row["results"]), sum(r["exact"] and r["rule_clean"] for r in row["results"]), k)
                for row in rows
            ) / len(rows)
        summary["compiles@1"] = sum(row["results"][0]["compiled"] for row in rows) / len(rows)
        summary["rule_clean_rate"] = sum(r["rule_clean"] for row in rows for r in row["results"]) / sum(
            len(row["results"]) for row in rows
        )
        summary["mean_similarity_first"] = sum(row["results"][0]["similarity"] for row in rows) / len(rows)
        summary["mean_similarity_best"] = sum(max(r["similarity"] for r in row["results"]) for row in rows) / len(rows)
        return summary

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_function:
        groups["all"].append(row)
        groups[f"game:{row['game']}"].append(row)
        groups[f"size:{size_bucket(row['size_bytes'])}"].append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}
