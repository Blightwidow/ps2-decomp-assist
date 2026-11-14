"""CLI: python -m pipeline.build_splits. Builds leak-safe train/validation/test files. Prints counts only."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline.similarity import (
    exact_duplicate_groups,
    masked_assembly_tokens,
    normalized_c_tokens,
    shingles,
    similar_pairs,
)
from pipeline.splits import assign_splits, audit_no_leakage, connected_components, resolve_cross_split_duplicates
from pipeline.token_stats import SYSTEM_PROMPT, render_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDS_DIRECTORY = PROJECT_ROOT / "data" / "records"
SPLITS_DIRECTORY = PROJECT_ROOT / "data" / "splits"
MAXIMUM_TOKENS = 2048
ASSEMBLY_THRESHOLD = 0.8
BOILERPLATE_CLUSTER_SIZE = 25
TRAIN_ONLY_COMPONENT_SIZE = 60
SEED = 0


def load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for game in ("sly1", "sly2"):
        records.extend(json.loads(line) for line in (RECORDS_DIRECTORY / f"{game}.jsonl").read_text().splitlines())
    return records


def token_counts(records: list[dict[str, Any]]) -> list[int]:
    cache_path = RECORDS_DIRECTORY / "token_counts.json"
    cache: dict[str, int] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [record for record in records if f"{record['game']}:{record['function_id']}" not in cache]
    if missing:
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("mlx-community/gemma-4-e2b-it-bf16")
        for record in missing:
            prompt = len(tokenizer(SYSTEM_PROMPT + render_user_prompt(record))["input_ids"])
            completion = len(tokenizer(record["c_text"])["input_ids"])
            cache[f"{record['game']}:{record['function_id']}"] = prompt + completion
        cache_path.write_text(json.dumps(cache))
    return [cache[f"{record['game']}:{record['function_id']}"] for record in records]


def group_key(record: dict[str, Any]) -> str:
    if record["game"] == "sly2":
        return f"sly2:{record['group']}:{record['address'] // 0x10000:X}"
    return f"sly1:{record['group']}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def main() -> None:
    all_records = load_records()
    totals = token_counts(all_records)
    exclusion_reasons: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record, total in zip(all_records, totals, strict=True):
        reason = "rule_dirty" if not record["rule_clean"] else "over_token_limit" if total > MAXIMUM_TOKENS else None
        if reason:
            exclusion_reasons[f"{record['game']}:{reason}"] += 1
            excluded.append({**record, "excluded_reason": reason})
        else:
            records.append({**record, "token_count": total})
    print("eligible:", len(records), "excluded:", dict(exclusion_reasons))

    masked = [masked_assembly_tokens(record["assembly"]) for record in records]
    c_tokens = [normalized_c_tokens(record["c_text"]) for record in records]
    assembly_keys = ["\n".join(tokens) for tokens in masked]
    c_keys = [" ".join(tokens) for tokens in c_tokens]

    duplicate_links: set[tuple[int, int]] = set()
    for keys in (assembly_keys, c_keys):
        for members in exact_duplicate_groups(keys):
            duplicate_links.update((members[0], other) for other in members[1:])
    duplicate_links.update(similar_pairs([shingles(tokens, 4) for tokens in masked], ASSEMBLY_THRESHOLD))
    print("duplicate links:", len(duplicate_links))

    # Components are source groups only: merging near-duplicate links too chains the whole corpus together.
    duplicate_components = connected_components(
        len(records), duplicate_links, [str(index) for index in range(len(records))]
    )
    components = connected_components(len(records), [], [group_key(record) for record in records])
    sizes = sorted((len(component) for component in components), reverse=True)
    print("source-group components:", len(components), "largest:", sizes[:5])

    initial_assignment = assign_splits(components, train_only_component_size=TRAIN_ONLY_COMPONENT_SIZE, seed=SEED)
    assignment = resolve_cross_split_duplicates(initial_assignment, duplicate_components, BOILERPLATE_CLUSTER_SIZE)
    dropped = [index for index, label in assignment.items() if label == "dropped"]
    print(
        "dropped for cross-split duplicates:",
        len(dropped),
        dict(Counter(initial_assignment[index] for index in dropped)),
    )
    problems = audit_no_leakage(components, assignment, assembly_keys)
    problems += audit_no_leakage(duplicate_components, assignment, c_keys)
    print("leak audit problems:", len(problems), problems[:3])
    if problems:
        raise SystemExit("leakage audit failed")

    component_of: dict[int, int] = {}
    for component_id, component in enumerate(components):
        for index in component:
            component_of[index] = component_id
    games_in_duplicate_cluster: dict[int, set[str]] = {}
    for cluster in duplicate_components:
        cluster_games = {records[index]["game"] for index in cluster}
        for index in cluster:
            games_in_duplicate_cluster[index] = cluster_games
    ground_truth_failures: set[str] = set()
    for failure_path in SPLITS_DIRECTORY.glob("ground_truth_failures_*.txt"):
        ground_truth_failures |= set(failure_path.read_text().split())
    print("ground-truth failures to drop:", len(ground_truth_failures))
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    seen_assembly: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for index, record in enumerate(records):
        label = assignment[index]
        if label == "dropped":
            continue
        if f"{record['game']}:{record['function_id']}" in ground_truth_failures:
            # applied after assignment so that the frozen assignment does not shift
            excluded.append({**record, "excluded_reason": "ground_truth_failed"})
            continue
        component_id = component_of[index]
        first_of_kind = assembly_keys[index] not in seen_assembly[label]
        seen_assembly[label].add(assembly_keys[index])
        split_rows[label].append(
            {
                **record,
                "split": label,
                "component_id": component_id,
                "component_size": len(components[component_id]),
                "cross_game_component": len(games_in_duplicate_cluster[index]) > 1,
                "deduped_slice": first_of_kind and "trivial" not in record["tags"],
            }
        )
    SPLITS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for label, rows in split_rows.items():
        write_jsonl(SPLITS_DIRECTORY / f"{label}.jsonl", rows)
    write_jsonl(SPLITS_DIRECTORY / "excluded.jsonl", excluded)
    cross_game_test = [
        row
        for row in (r for rows in split_rows.values() for r in rows)
        if row["game"] == "sly1" and not row["cross_game_component"]
    ]
    write_jsonl(SPLITS_DIRECTORY / "cross_game_sly1_test.jsonl", cross_game_test)

    summary: dict[str, Any] = {
        label: {
            "total": len(rows),
            "by_game": dict(Counter(row["game"] for row in rows)),
            "deduped_slice": sum(row["deduped_slice"] for row in rows),
            "median_tokens": sorted(row["token_count"] for row in rows)[len(rows) // 2] if rows else 0,
        }
        for label, rows in split_rows.items()
    }
    test_ids = sorted(f"{row['game']}:{row['function_id']}" for row in split_rows["test"])
    manifest: dict[str, Any] = {
        "seed": SEED,
        "thresholds": {
            "assembly_jaccard": ASSEMBLY_THRESHOLD,
            "boilerplate_cluster_size": BOILERPLATE_CLUSTER_SIZE,
            "train_only_component_size": TRAIN_ONLY_COMPONENT_SIZE,
            "maximum_tokens": MAXIMUM_TOKENS,
        },
        "summary": summary,
        "cross_game_sly1_test": len(cross_game_test),
        "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(sorted(f"{row['game']}:{row['function_id']}" for row in split_rows["validation"])).encode()
        ).hexdigest(),
    }
    (SPLITS_DIRECTORY / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(summary, indent=1))
    print("cross-game sly1 test:", len(cross_game_test), "test hash:", manifest["test_ids_sha256"][:16])


if __name__ == "__main__":
    main()
