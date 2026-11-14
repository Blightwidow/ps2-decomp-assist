from pathlib import Path

from pipeline.context import IncludeResolver, build_context, collect_chunks

HEADER = """// doc
#define SCALE 4
typedef int s32;
struct Point { s32 x; s32 y; };
struct Unused { s32 z; };
extern s32 g_total;
"""

SOURCE = """#include "types.h"
extern s32 helper(Point* point);
s32 target(Point* point) { return point->x * SCALE + g_total; }
s32 other_function(Point* point) { return helper(point); }
"""


def _chunks(tmp_path: Path):  # type: ignore[no-untyped-def]
    (tmp_path / "types.h").write_text(HEADER)
    source_path = tmp_path / "file.cpp"
    source_path.write_text(SOURCE)
    return collect_chunks(source_path, SOURCE, IncludeResolver([tmp_path]))


def test_context_contains_needed_declarations_transitively(tmp_path: Path) -> None:
    context = build_context(
        "s32 target(Point* point) { return point->x * SCALE + g_total; }", "target", _chunks(tmp_path)
    )
    assert (
        "struct Point" in context
        and "typedef int s32" in context
        and "#define SCALE" in context
        and "g_total" in context
    )


def test_context_excludes_unreferenced_declarations(tmp_path: Path) -> None:
    context = build_context(
        "s32 target(Point* point) { return point->x * SCALE + g_total; }", "target", _chunks(tmp_path)
    )
    assert "Unused" not in context and "helper" not in context


def test_context_never_contains_other_function_bodies(tmp_path: Path) -> None:
    context = build_context("s32 target(Point* point) { return helper(point); }", "target", _chunks(tmp_path))
    assert "other_function" not in context and "return helper" not in context


def test_character_cap_drops_distant_chunks_first(tmp_path: Path) -> None:
    context = build_context(
        "int target(Point* point) { return point->x; }", "target", _chunks(tmp_path), character_cap=50
    )
    assert "struct Point" in context and "typedef int s32" not in context


def test_struct_chunks_keep_trailing_semicolon(tmp_path: Path) -> None:
    context = build_context("int target(Point* point) { return point->x + point->y; }", "target", _chunks(tmp_path))
    assert context.count("};") >= 1 and "s32 y;" in context


def test_forward_declaration_dropped_when_definition_exists(tmp_path: Path) -> None:
    (tmp_path / "types.h").write_text("struct Point;\nstruct Point { int x; };\n")
    source_path = tmp_path / "file.cpp"
    source_path.write_text('#include "types.h"\n')
    chunks = collect_chunks(source_path, '#include "types.h"\n', IncludeResolver([tmp_path]))
    context = build_context("int target(Point* point) { return point->x; }", "target", chunks)
    assert context.count("struct Point") == 1


def test_struct_bodies_are_trimmed_to_used_members(tmp_path: Path) -> None:
    (tmp_path / "types.h").write_text("struct Big { int used; int unused_one; int unused_two; };\n")
    source_path = tmp_path / "file.cpp"
    source_path.write_text('#include "types.h"\n')
    chunks = collect_chunks(source_path, '#include "types.h"\n', IncludeResolver([tmp_path]))
    context = build_context("int target(Big* big) { return big->used; }", "target", chunks)
    assert "int used;" in context and "unused_one" not in context and "/* ... */" in context


def test_struct_without_used_members_becomes_forward_declaration(tmp_path: Path) -> None:
    (tmp_path / "types.h").write_text("struct Big { int a; int b; };\n")
    source_path = tmp_path / "file.cpp"
    source_path.write_text('#include "types.h"\n')
    chunks = collect_chunks(source_path, '#include "types.h"\n', IncludeResolver([tmp_path]))
    context = build_context("int target(Big* big) { return helper(big); }", "target", chunks)
    assert context.strip() == "struct Big;"


def test_member_types_are_pulled_in_transitively(tmp_path: Path) -> None:
    (tmp_path / "types.h").write_text("struct Inner { int v; };\nstruct Outer { Inner inner; int pad; };\n")
    source_path = tmp_path / "file.cpp"
    source_path.write_text('#include "types.h"\n')
    chunks = collect_chunks(source_path, '#include "types.h"\n', IncludeResolver([tmp_path]))
    context = build_context("int target(Outer* outer) { return outer->inner.v; }", "target", chunks)
    assert "struct Inner" in context and "int v;" in context and "pad" not in context


def test_extern_c_single_definition_is_found() -> None:
    from pipeline.cpp_parse import find_function_definitions

    definitions = find_function_definitions(
        'extern "C" int func_00100A90(int a) { return a; }\nextern "C" { int other(void) { return 1; } }'
    )
    assert [definition.name for definition in definitions] == ["func_00100A90", "other"]
