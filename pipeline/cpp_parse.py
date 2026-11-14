"""C++ source parsing (tree-sitter) for function extraction and declaration-only context building."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

CPP_LANGUAGE = Language(tree_sitter_cpp.language())
IDENTIFIER_NODE_TYPES = {"identifier", "type_identifier", "field_identifier", "namespace_identifier"}
DECLARATOR_WRAPPER_TYPES = {
    "pointer_declarator",
    "reference_declarator",
    "function_declarator",
    "parenthesized_declarator",
    "array_declarator",
    "attributed_declarator",
}
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s+"(?P<path>[^"]+)"', re.MULTILINE)


@dataclass
class FunctionDefinition:
    name: str
    text: str
    start_byte: int
    end_byte: int


@dataclass
class DeclarationChunk:
    """A top-level declaration usable as context (never a function definition)."""

    kind: str
    declared_names: set[str]
    text: str
    start_byte: int
    referenced_names: set[str] = field(default_factory=set)
    is_forward_declaration: bool = False


def _parse(source_bytes: bytes) -> Node:
    return Parser(CPP_LANGUAGE).parse(source_bytes).root_node


def _walk(node: Node):  # type: ignore[no-untyped-def]
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def strip_comments(source_text: str) -> str:
    """Blank out comments, keeping newlines so offsets of lines stay stable."""
    source_bytes = source_text.encode()
    root = _parse(source_bytes)
    buffer = bytearray(source_bytes)
    for node in _walk(root):
        if node.type == "comment":
            for index in range(node.start_byte, node.end_byte):
                if buffer[index] != 0x0A:
                    buffer[index] = 0x20
    return "\n".join(line.rstrip() for line in buffer.decode().split("\n"))


def referenced_identifiers(node_text: str) -> set[str]:
    root = _parse(node_text.encode())
    return {node.text.decode() for node in _walk(root) if node.type in IDENTIFIER_NODE_TYPES and node.text}


def _function_name(definition: Node) -> str:
    declarator = definition.child_by_field_name("declarator")
    while declarator is not None and declarator.type in DECLARATOR_WRAPPER_TYPES:
        inner = declarator.child_by_field_name("declarator")
        if inner is None:
            break
        declarator = inner
    return declarator.text.decode() if declarator is not None and declarator.text else ""


def _top_level_nodes(root: Node):  # type: ignore[no-untyped-def]
    """Top-level nodes, looking inside ``extern "C" {}`` blocks and preprocessor conditionals."""
    for child in root.children:
        if child.type == "linkage_specification":
            body = child.child_by_field_name("body")
            if body is not None and body.type == "declaration_list":
                yield from _top_level_nodes(body)
            elif body is not None:
                yield body  # extern "C" applied to one declaration or definition
        elif child.type in {
            "preproc_ifdef",
            "preproc_if",
            "preproc_else",
            "namespace_definition",
        }:
            body = child.child_by_field_name("body")
            inner_root = body if body is not None else child
            yield from _top_level_nodes(inner_root)
        elif child.type == "declaration_list":
            yield from _top_level_nodes(child)
        else:
            yield child


def find_function_definitions(source_text: str) -> list[FunctionDefinition]:
    source_bytes = source_text.encode()
    root = _parse(source_bytes)
    definitions: list[FunctionDefinition] = []
    for node in _top_level_nodes(root):
        if node.type == "function_definition":
            definitions.append(
                FunctionDefinition(
                    name=_function_name(node),
                    text=source_bytes[node.start_byte : node.end_byte].decode(),
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
            )
    return definitions


def _declared_names(node: Node) -> set[str]:
    names: set[str] = set()
    if node.type in {"preproc_def", "preproc_function_def"}:
        name_node = node.child_by_field_name("name")
        if name_node is not None and name_node.text:
            names.add(name_node.text.decode())
    elif node.type == "type_definition":
        for declarator in node.children_by_field_name("declarator"):
            current = declarator
            while current.type in DECLARATOR_WRAPPER_TYPES and current.child_by_field_name("declarator") is not None:
                inner = current.child_by_field_name("declarator")
                assert inner is not None
                current = inner
            if current.text:
                names.add(current.text.decode())
        names |= _specifier_names(node)
    elif node.type in {"struct_specifier", "class_specifier", "union_specifier", "enum_specifier"}:
        name_node = node.child_by_field_name("name")
        if name_node is not None and name_node.text:
            names.add(name_node.text.decode())
    elif node.type in {"declaration", "field_declaration"}:
        names |= _specifier_names(node)
        for declarator in node.children_by_field_name("declarator"):
            current = declarator
            while current.type in DECLARATOR_WRAPPER_TYPES | {"init_declarator"}:
                inner = current.child_by_field_name("declarator")
                if inner is None:
                    break
                current = inner
            if current.text:
                names.add(current.text.decode())
    return names


def _specifier_names(node: Node) -> set[str]:
    names: set[str] = set()
    for descendant in _walk(node):
        if descendant.type in {"struct_specifier", "class_specifier", "union_specifier", "enum_specifier"}:
            name_node = descendant.child_by_field_name("name")
            if name_node is not None and name_node.text and descendant.child_by_field_name("body") is not None:
                names.add(name_node.text.decode())
    return names


def declaration_chunks(source_text: str) -> list[DeclarationChunk]:
    """Top-level declarations of a file (types, typedefs, macros, externs, prototypes), excluding function bodies."""
    source_bytes = source_text.encode()
    root = _parse(source_bytes)
    chunks: list[DeclarationChunk] = []
    for node in _top_level_nodes(root):
        if node.type in {"preproc_include", "function_definition", "comment", "preproc_call", "ERROR"}:
            continue
        kind = node.type
        if kind in {"struct_specifier", "class_specifier", "union_specifier", "enum_specifier"}:
            pass
        names = _declared_names(node)
        if not names:
            continue
        end_byte = node.end_byte
        following = node.next_sibling
        if following is not None and following.type == ";" and node.type.endswith("_specifier"):
            end_byte = following.end_byte
        text = source_bytes[node.start_byte : end_byte].decode()
        chunks.append(
            DeclarationChunk(
                kind=kind,
                declared_names=names,
                text=text,
                start_byte=node.start_byte,
                referenced_names=referenced_identifiers(text) - names,
                is_forward_declaration=kind.endswith("_specifier") and "{" not in text,
            )
        )
    return chunks


def included_paths(source_text: str) -> list[str]:
    return INCLUDE_PATTERN.findall(source_text)


def trim_struct_text(chunk_text: str, wanted_names: set[str]) -> str:
    """Keep only the members of a struct/class whose declared names the function uses.

    Dropped members are replaced by one ``/* ... */`` marker. Base-class lists and non-struct chunks stay as-is.
    A struct with no used member becomes a forward declaration.
    """
    source_bytes = chunk_text.encode()
    root = _parse(source_bytes)
    specifier = next(
        (
            node
            for node in _walk(root)
            if node.type in {"struct_specifier", "class_specifier"} and node.child_by_field_name("body")
        ),
        None,
    )
    if specifier is None:
        return chunk_text
    body = specifier.child_by_field_name("body")
    assert body is not None
    kept_members: list[str] = []
    dropped_any = False
    for member in body.children:
        if member.type in {"{", "}", "access_specifier", "comment"}:
            continue
        member_names = _member_names(member)
        if member_names & wanted_names:
            kept_members.append(source_bytes[member.start_byte : member.end_byte].decode())
        else:
            dropped_any = True
    header = source_bytes[specifier.start_byte : body.start_byte].decode().rstrip()
    if not kept_members:
        base_clause_present = ":" in header
        return f"{header};" if not base_clause_present else f"{header} {{ /* ... */ }};"
    lines = [f"{header} {{"]
    lines.extend(f"    {member}" for member in kept_members)
    if dropped_any:
        lines.append("    /* ... */")
    lines.append("};")
    return "\n".join(lines)


def _member_names(member: Node) -> set[str]:
    names: set[str] = set()
    for descendant in _walk(member):
        if descendant.type in {"field_identifier", "identifier"} and descendant.text:
            names.add(descendant.text.decode())
    # a member's type name is not its own name: drop the type identifiers collected above only when they are type nodes
    return names
