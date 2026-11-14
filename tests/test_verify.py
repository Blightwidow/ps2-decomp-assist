from pathlib import Path

from eval.generate import extract_code
from pipeline.verify import candidate_defines, splice_candidate

SOURCE = """#include "common.h"
int helper(int a) { return a; }
int first(int a) { return a + 1; }
int first(float a) { return 7; }
int target(int a) { return a * 2; }
"""


def _splice(tmp_path: Path, function_id: str, definition_name: str, ordinal: int) -> str | None:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "file.cpp").write_text(SOURCE)
    return splice_candidate(
        "sly2", "src/file.cpp", function_id, definition_name, ordinal, "int NEW() { return 0; }", tmp_path
    )


def test_splice_replaces_only_the_target_definition(tmp_path: Path) -> None:
    spliced = _splice(tmp_path, "target", "target", 0)
    assert spliced is not None and "int NEW()" in spliced and "a * 2" not in spliced and "int helper" in spliced


def test_splice_uses_overload_ordinal(tmp_path: Path) -> None:
    spliced = _splice(tmp_path, "first__Ff", "first", 1)
    assert spliced is not None and "return a + 1" in spliced and "return 7" not in spliced


def test_splice_returns_none_when_target_is_missing(tmp_path: Path) -> None:
    assert _splice(tmp_path, "nothing", "nothing", 0) is None


def test_candidate_defines_requires_a_definition() -> None:
    assert not candidate_defines("sly1", "just words", "func_1")
    assert candidate_defines("sly1", "int func_1() { return 1; }", "func_1")
    assert not candidate_defines("sly1", "int other() { return 1; }", "func_1")


def test_extract_code_takes_the_fenced_block() -> None:
    assert extract_code("Here:\n```cpp\nint f() { return 1; }\n```\nDone.") == "int f() { return 1; }"
    assert extract_code("int f() { return 1; }") == "int f() { return 1; }"
