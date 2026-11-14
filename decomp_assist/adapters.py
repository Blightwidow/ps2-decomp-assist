"""Resolve an adapter argument: a local folder, or a folder inside a Hugging Face model repo (``hf:OWNER/REPO/SUBFOLDER``)."""

from __future__ import annotations

from pathlib import Path

HUGGING_FACE_PREFIX = "hf:"
DEFAULT_REPOSITORY = "blightwidow/ps2-decomp-assist-gemma4-e2b-lora"
DEFAULT_ADAPTER = f"{HUGGING_FACE_PREFIX}{DEFAULT_REPOSITORY}/t1"
DEFAULT_REPAIR_ADAPTER = f"{HUGGING_FACE_PREFIX}{DEFAULT_REPOSITORY}/repair"


def parse_hugging_face_spec(spec: str) -> tuple[str, str]:
    """(repository id, subfolder) from ``hf:OWNER/REPO/SUBFOLDER``."""
    parts = spec.removeprefix(HUGGING_FACE_PREFIX).split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"expected hf:OWNER/REPO/SUBFOLDER, got {spec!r}")
    return f"{parts[0]}/{parts[1]}", parts[2]


def resolve_adapter(spec: str) -> str:
    """The local folder holding ``adapters.safetensors`` and ``adapter_config.json``; downloads on first use."""
    if not spec.startswith(HUGGING_FACE_PREFIX):
        return spec
    repository, subfolder = parse_hugging_face_spec(spec)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repository, allow_patterns=[f"{subfolder}/*"])
    return str(Path(snapshot) / subfolder)
