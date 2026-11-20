from pathlib import Path

import pytest

from decomp_assist.target import TargetError, address_symbol, build_target
from pipeline.verify import Candidate, build_job


def test_address_symbol_accepts_a_bare_address() -> None:
    assert address_symbol("sly3", "187cb8") == "func_00187CB8"


def test_address_symbol_accepts_a_symbol() -> None:
    assert address_symbol("sly3", "func_00187CB8") == "func_00187CB8"


def test_address_symbol_rejects_a_readable_name() -> None:
    with pytest.raises(TargetError, match="sly3"):
        address_symbol("sly3", "UpdateThief")


def test_sly3_target_needs_a_source_file(tmp_path: Path) -> None:
    with pytest.raises(TargetError, match="sly3 needs --source"):
        build_target("sly3", tmp_path, "func_00187CB8")


def test_sly3_job_carries_the_source_file() -> None:
    candidate = Candidate(
        "sly3", "func_00187CB8", 0x187CB8, 16, "func_00187CB8", 0, "nonmatching/func_00187CB8.cpp", "int f() {}"
    )
    assert build_job(candidate, "source text")["source_file"] == "nonmatching/func_00187CB8.cpp"
