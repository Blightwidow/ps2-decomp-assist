"""Build the model prompt row and the verifier fields for one function of a local sly1/sly2 checkout."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.asm_parse import normalize_assembly, parse_assembly_functions
from pipeline.context import IncludeResolver, build_context, collect_chunks
from pipeline.cpp_parse import FunctionDefinition, find_function_definitions, strip_comments
from pipeline.sly1 import load_function_table, load_reverse_names, to_address_names
from pipeline.sly2 import load_assembly_by_address
from pipeline.verify import Candidate

SLY1_SYMBOL_PATTERN = re.compile(r"^(?:func_)?(?:0x)?(?P<address>[0-9A-Fa-f]{1,8})$")


class TargetError(Exception):
    """The function cannot be turned into a prompt (unknown name, no definition in the source file, ...)."""


@dataclass
class Target:
    """``row`` feeds the prompt; the other fields locate the function for the verifier."""

    row: dict[str, Any]
    address: int
    definition_name: str
    definition_ordinal: int
    source_file: str  # relative to the repository

    def candidate(self, text: str) -> Candidate:
        return Candidate(
            self.row["game"],
            self.row["function_id"],
            self.address,
            self.row["size_bytes"],
            self.definition_name,
            self.definition_ordinal,
            self.source_file,
            text,
        )


def ordinal_of(definition: FunctionDefinition, definitions: list[FunctionDefinition]) -> int:
    return [other.start_byte for other in definitions if other.name == definition.name].index(definition.start_byte)


def sly1_symbol(function: str) -> str:
    """``func_XXXXXXXX`` from a symbol or a bare hex address."""
    match = SLY1_SYMBOL_PATTERN.match(function)
    if not match:
        raise TargetError(f"sly1 function must be func_XXXXXXXX or an address, got {function!r}")
    return f"func_{int(match.group('address'), 16):08X}"


def build_sly1_target(repository: Path, function: str, source_file: str) -> Target:
    symbol = sly1_symbol(function)
    function_table = load_function_table(repository)
    if symbol not in function_table:
        raise TargetError(f"{symbol} is not in config/us/functions.tsv")
    table_row = function_table[symbol]
    assembly_by_name = {
        item.name: item for item in parse_assembly_functions((repository / "asm" / "us" / "text.s").read_text())
    }
    if symbol not in assembly_by_name:
        raise TargetError(f"{symbol} has no assembly in asm/us/text.s")
    reverse_names = load_reverse_names(repository)
    source_path = repository / source_file
    source_text = source_path.read_text(errors="replace")
    definitions = find_function_definitions(source_text)
    matching = [item for item in definitions if to_address_names(item.name, reverse_names) == symbol]
    if len(matching) != 1:
        raise TargetError(f"{source_file} must hold exactly one definition of {symbol}, found {len(matching)}")
    definition = matching[0]
    resolver = IncludeResolver([repository / "include"], skipped_names=frozenset({"names.h"}))
    chunks = collect_chunks(source_path, source_text, resolver)
    clean_text = strip_comments(definition.text)
    context = to_address_names(build_context(clean_text, definition.name, chunks), reverse_names)
    assembly_function = assembly_by_name[symbol]
    row = {
        "game": "sly1",
        "function_id": symbol,
        "size_bytes": int(table_row["size"]),
        "context": context,
        "assembly": normalize_assembly(assembly_function),
    }
    return Target(row, int(table_row["address"], 16), definition.name, ordinal_of(definition, definitions), source_file)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_sly2_entry(repository: Path, function: str) -> tuple[int, str, str]:
    """(address, name, default source file relative to the repository) from the nonmatching or src manifest."""
    wanted_address = int(function, 16) if function.lower().startswith("0x") else None
    for prefix, manifest in (("nonmatching", "nonmatching/MANIFEST.tsv"), ("src", "src/MANIFEST.tsv")):
        for row in read_manifest(repository / manifest):
            address = int(row["address"], 16)
            if row["name"] == function or address == wanted_address:
                return address, row["name"], f"{prefix}/{row['file']}"
    raise TargetError(f"{function} is in neither nonmatching/MANIFEST.tsv nor src/MANIFEST.tsv")


def build_sly2_target(repository: Path, function: str, source_file: str | None) -> Target:
    address, name, default_source_file = find_sly2_entry(repository, function)
    source_file = source_file or default_source_file
    assembly_function = load_assembly_by_address(repository).get(address)
    if assembly_function is None:
        raise TargetError(f"no assembly at {address:#x} in build/splat/asm (run the repo's setup first)")
    source_path = repository / source_file
    source_text = source_path.read_text(errors="replace")
    definitions = find_function_definitions(source_text)
    wanted_names = {name, assembly_function.name}
    matching = [item for item in definitions if item.name in wanted_names]
    if not matching and len(definitions) == 1:
        matching = definitions
    if len(matching) != 1:
        raise TargetError(f"{source_file} must hold exactly one definition of {name}, found {len(matching)}")
    definition = matching[0]
    chunks = collect_chunks(source_path, source_text, IncludeResolver([repository / "include"]))
    clean_text = strip_comments(definition.text).strip()
    row = {
        "game": "sly2",
        "function_id": name,
        "size_bytes": assembly_function.size_bytes,
        "context": build_context(clean_text, definition.name, chunks),
        "assembly": normalize_assembly(assembly_function),
    }
    return Target(row, address, definition.name, ordinal_of(definition, definitions), source_file)


def build_target(game: str, repository: Path, function: str, source_file: str | None = None) -> Target:
    """The target for ``function`` (sly1: ``func_XXXXXXXX`` or address; sly2: name or ``0x`` address).

    ``source_file`` (relative to the repository) must already hold one definition of the function: the staged
    ``nonmatching/`` attempt, or the shared file it lives in. It defaults to the sly2 manifest entry; sly1 needs it.
    """
    if game == "sly1":
        if source_file is None:
            raise TargetError(
                "sly1 needs --source: the file that holds the function (e.g. nonmatching/func_XXXXXXXX.cpp)"
            )
        return build_sly1_target(repository, function, source_file)
    if game == "sly2":
        return build_sly2_target(repository, function, source_file)
    raise TargetError(f"unknown game {game!r}")
