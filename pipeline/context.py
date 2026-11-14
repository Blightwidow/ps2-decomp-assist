"""Declarations-only context for a function: only the types, macros, externs and prototypes it needs.

Never includes another function's body. Headers are read at the checked-out commit (a struct may list
fields that other functions' matching added later; recorded as a known limitation in docs/PLAN.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.cpp_parse import (
    DeclarationChunk,
    declaration_chunks,
    included_paths,
    referenced_identifiers,
    strip_comments,
    trim_struct_text,
)

HEADER_DEPTH_LIMIT = 6


@dataclass
class IncludeResolver:
    """Finds ``#include "x"`` targets under the project's own include directories."""

    search_directories: list[Path]
    skipped_names: frozenset[str] = frozenset()

    def resolve(self, include_path: str, including_directory: Path | None = None) -> Path | None:
        if Path(include_path).name in self.skipped_names:
            return None
        candidates = [directory / include_path for directory in self.search_directories]
        if including_directory is not None:
            candidates.insert(0, including_directory / include_path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


def collect_chunks(
    source_path: Path,
    source_text: str,
    resolver: IncludeResolver,
    read_text: Callable[[Path], str] = lambda path: path.read_text(errors="replace"),
) -> list[DeclarationChunk]:
    """Chunks of the headers (depth-first, in include order) followed by the source file's own."""
    visited: set[Path] = set()
    chunks: list[DeclarationChunk] = []

    def visit(path: Path, text: str, depth: int) -> None:
        if depth > HEADER_DEPTH_LIMIT:
            return
        for include_path in included_paths(text):
            header_path = resolver.resolve(include_path, path.parent)
            if header_path is None or header_path in visited:
                continue
            visited.add(header_path)
            header_text = read_text(header_path)
            visit(header_path, header_text, depth + 1)
            chunks.extend(declaration_chunks(strip_comments(header_text)))

    visit(source_path, source_text, 0)
    chunks.extend(declaration_chunks(strip_comments(source_text)))
    return chunks


def build_context(
    function_text: str,
    function_name: str,
    chunks: list[DeclarationChunk],
    character_cap: int = 4000,
) -> str:
    """Select the chunks the function needs (transitively); struct bodies are trimmed to the members it uses.

    When over the cap, the most distant (deepest) chunks are dropped first.
    """
    used_names = referenced_identifiers(function_text)
    defined_names = {name for chunk in chunks if not chunk.is_forward_declaration for name in chunk.declared_names}
    chunk_indices_by_name: dict[str, list[int]] = {}
    for index, chunk in enumerate(chunks):
        if chunk.is_forward_declaration and chunk.declared_names <= defined_names:
            continue  # a full definition exists: the bare forward declaration adds nothing
        for name in chunk.declared_names:
            chunk_indices_by_name.setdefault(name, []).append(index)

    depth_by_chunk: dict[int, int] = {}
    trimmed_text_by_chunk: dict[int, str] = {}
    frontier = set(used_names)
    depth = 0
    while frontier:
        next_frontier: set[str] = set()
        for name in frontier:
            for index in chunk_indices_by_name.get(name, []):
                chunk = chunks[index]
                if chunk.declared_names == {function_name} or index in depth_by_chunk:
                    continue  # the function's own prototype adds nothing; chunks are taken once
                depth_by_chunk[index] = depth
                trimmed = trim_struct_text(chunk.text, used_names) if chunk.kind.endswith("_specifier") else chunk.text
                trimmed_text_by_chunk[index] = trimmed
                next_frontier |= referenced_identifiers(trimmed) - chunk.declared_names
        frontier = next_frontier
        depth += 1

    selected = sorted(depth_by_chunk, key=lambda index: (depth_by_chunk[index], index))
    kept: list[int] = []
    total_characters = 0
    for index in selected:
        length = len(trimmed_text_by_chunk[index]) + 1
        if total_characters + length > character_cap:
            continue
        kept.append(index)
        total_characters += length
    return "\n".join(trimmed_text_by_chunk[index] for index in sorted(kept))
