"""Sly 1 (OpenThievius/sly1-decomp) extraction: matched game-code functions paired with their asm."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from pipeline.asm_parse import normalize_assembly, parse_assembly_functions
from pipeline.context import IncludeResolver, build_context, collect_chunks
from pipeline.cpp_parse import find_function_definitions, strip_comments
from pipeline.records import Record, assembly_tags, scan_rules

IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]\w*")
WINDOW_GROUP_BYTES = 0x10000


def load_reverse_names(repository: Path) -> dict[str, str]:
    """Readable name -> address symbol (func_XXXXXXXX / D_XXXXXXXX), from config/names/**/*.tsv."""
    mapping: dict[str, str] = {}
    for table_path in sorted((repository / "config" / "names").rglob("*.tsv")):
        with table_path.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                symbol, name = row.get("symbol"), row.get("name")
                if symbol and name and symbol != name:
                    mapping[name] = symbol
    return mapping


def to_address_names(text: str, reverse_names: dict[str, str]) -> str:
    return IDENTIFIER_PATTERN.sub(lambda match: reverse_names.get(match.group(0), match.group(0)), text)


def load_matched_game_functions(repository: Path) -> set[str]:
    report = json.loads((repository / "progress" / "us" / "report.json").read_text())
    matched: set[str] = set()
    for unit in report["units"]:
        if not unit["name"].startswith("game/"):
            continue
        for function in unit.get("functions", []):
            if function.get("fuzzy_match_percent") == 100.0:
                matched.add(function["name"])
    return matched


def load_function_table(repository: Path) -> dict[str, dict[str, str]]:
    with (repository / "config" / "us" / "functions.tsv").open() as handle:
        return {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}


def load_real_file_ranges(repository: Path) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    with (repository / "config" / "us" / "files.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["text"] != "-" and row["end"] != "-":
                ranges.append((int(row["text"], 16), int(row["end"], 16), row["file"]))
    return ranges


def group_for_address(address: int, real_file_ranges: list[tuple[int, int, str]]) -> str:
    for start, end, file_name in real_file_ranges:
        if start <= address < end:
            return f"file:{Path(file_name).stem}"
    return f"window:{address // WINDOW_GROUP_BYTES:X}"


def extract(repository: Path, maximum_size_bytes: int = 400) -> tuple[list[Record], Counter[str]]:
    statistics: Counter[str] = Counter()
    reverse_names = load_reverse_names(repository)
    matched = load_matched_game_functions(repository)
    function_table = load_function_table(repository)
    real_file_ranges = load_real_file_ranges(repository)
    assembly_by_name = {
        function.name: function
        for function in parse_assembly_functions((repository / "asm" / "us" / "text.s").read_text())
    }
    resolver = IncludeResolver([repository / "include"], skipped_names=frozenset({"names.h"}))

    records: list[Record] = []
    seen: set[str] = set()
    for source_path in sorted((repository / "src" / "main").glob("*.cpp")):
        source_text = source_path.read_text(errors="replace")
        chunks = None
        definitions = find_function_definitions(source_text)
        for definition in definitions:
            symbol = to_address_names(definition.name, reverse_names)
            statistics["definitions"] += 1
            if symbol not in function_table:
                statistics["skip:not_a_known_symbol"] += 1
                continue
            if symbol not in matched:
                statistics["skip:not_matched_in_report"] += 1
                continue
            row = function_table[symbol]
            if row["kind"] != "code":
                statistics[f"skip:kind_{row['kind']}"] += 1
                continue
            if symbol in seen:
                statistics["skip:duplicate_definition"] += 1
                continue
            size_bytes = int(row["size"])
            if size_bytes > maximum_size_bytes:
                statistics["skip:too_large"] += 1
                continue
            assembly_function = assembly_by_name.get(symbol)
            if assembly_function is None:
                statistics["skip:no_assembly"] += 1
                continue
            if assembly_function.size_bytes != size_bytes:
                statistics["skip:asm_size_mismatch"] += 1
                continue
            if chunks is None:
                chunks = collect_chunks(source_path, source_text, resolver)
            clean_function_text = strip_comments(definition.text)
            context = build_context(clean_function_text, definition.name, chunks)
            address = int(row["address"], 16)
            assembly_text = normalize_assembly(assembly_function)
            c_text = to_address_names(clean_function_text, reverse_names).strip()
            records.append(
                Record(
                    game="sly1",
                    function_id=symbol,
                    address=address,
                    size_bytes=size_bytes,
                    group=group_for_address(address, real_file_ranges),
                    source_file=str(source_path.relative_to(repository)),
                    assembly=assembly_text,
                    c_text=c_text,
                    definition_name=definition.name,
                    definition_ordinal=[
                        other.start_byte for other in definitions if other.name == definition.name
                    ].index(definition.start_byte),
                    context=to_address_names(context, reverse_names),
                    tags=assembly_tags(assembly_text, assembly_function.instruction_count),
                    rule_hits=scan_rules(c_text),
                )
            )
            seen.add(symbol)
            statistics["extracted"] += 1
    return records, statistics
