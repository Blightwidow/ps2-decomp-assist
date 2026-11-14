"""Sly 2 (OpenThievius/sly2-decomp) extraction: matched game-code functions paired with their asm.

Names are kept as the repo has them (recovered names mixed with ``func_XXXXXXXX``). Address-only variants are
produced later with ``function_aliases`` (Stage 3 ablation).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from pipeline.asm_parse import AssemblyFunction, normalize_assembly, parse_assembly_functions
from pipeline.context import IncludeResolver, build_context, collect_chunks
from pipeline.cpp_parse import find_function_definitions, strip_comments
from pipeline.records import Record, assembly_tags, scan_rules

SYMBOL_LINE_PATTERN = re.compile(r"^(?P<name>\S+)\s*=\s*0x(?P<address>[0-9A-Fa-f]+);\s*//\s*type:func")


def function_aliases(repository: Path) -> dict[str, str]:
    """Recovered function name -> ``func_XXXXXXXX`` (address-only form), from config/us/symbol_addrs.txt."""
    aliases: dict[str, str] = {}
    for line in (repository / "config" / "us" / "symbol_addrs.txt").read_text().splitlines():
        match = SYMBOL_LINE_PATTERN.match(line)
        if match:
            aliases[match.group("name")] = f"func_{int(match.group('address'), 16):08X}"
    return aliases


def load_assembly_by_address(repository: Path) -> dict[int, AssemblyFunction]:
    by_address: dict[int, AssemblyFunction] = {}
    for assembly_path in sorted((repository / "build" / "splat" / "asm").glob("*.s")):
        for function in parse_assembly_functions(assembly_path.read_text(errors="replace")):
            by_address[function.address] = function
    return by_address


def load_excluded_addresses(repository: Path) -> set[int]:
    """Inline-asm helper functions and permanent blockers: never part of the dataset."""
    excluded: set[int] = set()
    for table_name in ("inline_functions.tsv", "blockers.tsv"):
        with (repository / "config" / "us" / table_name).open() as handle:
            excluded |= {int(row["address"], 16) for row in csv.DictReader(handle, delimiter="\t")}
    return excluded


def extract(repository: Path, maximum_size_bytes: int = 400) -> tuple[list[Record], Counter[str]]:
    statistics: Counter[str] = Counter()
    assembly_by_address = load_assembly_by_address(repository)
    excluded_addresses = load_excluded_addresses(repository)
    resolver = IncludeResolver([repository / "include"])

    with (repository / "src" / "MANIFEST.tsv").open() as handle:
        manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

    records: list[Record] = []
    parsed_files: dict[str, tuple[str, list, list]] = {}  # type: ignore[type-arg]
    for row in manifest_rows:
        statistics["manifest_rows"] += 1
        relative_file = row["file"]
        if relative_file.startswith("sdk/"):
            statistics["skip:sdk"] += 1
            continue
        address = int(row["address"], 16)
        size_bytes = int(row["size"])
        if address in excluded_addresses:
            statistics["skip:inline_or_blocker"] += 1
            continue
        if size_bytes > maximum_size_bytes:
            statistics["skip:too_large"] += 1
            continue
        assembly_function = assembly_by_address.get(address)
        if assembly_function is None:
            statistics["skip:no_assembly"] += 1
            continue
        if assembly_function.size_bytes != size_bytes:
            statistics["skip:asm_size_mismatch"] += 1
            continue
        source_path = repository / "src" / relative_file
        if relative_file not in parsed_files:
            source_text = source_path.read_text(errors="replace")
            parsed_files[relative_file] = (
                source_text,
                find_function_definitions(source_text),
                collect_chunks(source_path, source_text, resolver),
            )
        source_text, definitions, chunks = parsed_files[relative_file]
        wanted_names = {name for name in (row["symbol"], row["name"], assembly_function.name) if name}
        matching = [definition for definition in definitions if definition.name in wanted_names]
        if not matching and not row["whole_file"] and len(definitions) == 1:
            matching = definitions  # a per-function file: its only definition, whatever name it was given
        if len(matching) != 1:
            statistics["skip:definition_not_found" if not matching else "skip:definition_ambiguous"] += 1
            continue
        definition = matching[0]
        clean_function_text = strip_comments(definition.text).strip()
        assembly_text = normalize_assembly(assembly_function)
        records.append(
            Record(
                game="sly2",
                function_id=row["name"],
                address=address,
                size_bytes=size_bytes,
                group=f"module:{relative_file.split('/')[0]}",
                source_file=f"src/{relative_file}",
                assembly=assembly_text,
                c_text=clean_function_text,
                definition_name=definition.name,
                definition_ordinal=[other.start_byte for other in definitions if other.name == definition.name].index(
                    definition.start_byte
                ),
                context=build_context(clean_function_text, definition.name, chunks),
                tags=assembly_tags(assembly_text, assembly_function.instruction_count),
                rule_hits=scan_rules(clean_function_text),
            )
        )
        statistics["extracted"] += 1
    return records, statistics
