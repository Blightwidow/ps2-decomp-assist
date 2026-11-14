import pytest

from decomp_assist.adapters import parse_hugging_face_spec, resolve_adapter


def test_local_path_is_returned_unchanged() -> None:
    assert resolve_adapter("runs/sft2") == "runs/sft2"


def test_hugging_face_spec_splits_into_repository_and_subfolder() -> None:
    assert parse_hugging_face_spec("hf:owner/repo/t1") == ("owner/repo", "t1")


@pytest.mark.parametrize("spec", ["hf:owner/repo", "hf:owner/repo/a/b", "hf://x"])
def test_malformed_hugging_face_spec_is_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_hugging_face_spec(spec)
