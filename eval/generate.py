"""Candidate generation with mlx-lm: chat prompt in the v1 format (thinking off), n samples per prompt."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pipeline.token_stats import SYSTEM_PROMPT, render_user_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/gemma-4-e2b-it-bf16"
FENCE_PATTERN = re.compile(r"```[\w+]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """The code of the first fenced block, else the raw text."""
    match = FENCE_PATTERN.search(text)
    return (match.group(1) if match else text).strip()


def build_messages(row: dict[str, Any], shots: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": render_user_prompt(shot)})
        messages.append({"role": "assistant", "content": shot["c_text"]})
    messages.append({"role": "user", "content": render_user_prompt(row)})
    return messages


class Generator:
    def __init__(self, model_path: str = DEFAULT_MODEL, adapter_path: str | None = None) -> None:
        os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / "hf"))
        from mlx_lm import load

        loaded = load(model_path, adapter_path=adapter_path)
        self.model, self.tokenizer = loaded[0], loaded[1]

    def prompt_tokens(self, messages: list[dict[str, str]]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(  # type: ignore[no-untyped-call]
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
        )

    def sample(
        self,
        conversations: list[list[dict[str, str]]],
        samples_per_prompt: int,
        temperature: float,
        max_tokens: int = 512,
        batch_size: int = 8,
    ) -> list[list[str]]:
        """Texts per conversation (``samples_per_prompt`` each), generated in batches of ``batch_size`` sequences."""
        from mlx_lm.generate import batch_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        outputs: list[list[str]] = []
        for messages in conversations:
            tokens = self.prompt_tokens(messages)
            texts: list[str] = []
            remaining = samples_per_prompt
            while remaining > 0:
                batch = min(batch_size, remaining)
                response = batch_generate(
                    self.model, self.tokenizer, [tokens] * batch, max_tokens=max_tokens, sampler=sampler
                )
                texts.extend(response.texts)
                remaining -= batch
            outputs.append(texts)
        return outputs
