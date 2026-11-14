"""Candidate generation through an OpenAI-compatible server (llama.cpp ``llama-server``), same interface as ``Generator.sample``.

Start the server with the exported GGUF and thinking off, one slot per concurrent sample, for example:

    llama-server -m model.gguf --jinja --chat-template-kwargs '{"enable_thinking": false}' -ngl 99 -c 20480 -np 8
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

REQUEST_TIMEOUT_SECONDS = 600


def post_chat_completion(server_url: str, payload: dict[str, object]) -> str:
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return str(json.loads(response.read())["choices"][0]["message"]["content"])


class ServerGenerator:
    def __init__(
        self,
        server_url: str,
        parallel: int = 8,
        post: Callable[[str, dict[str, object]], str] = post_chat_completion,
    ) -> None:
        self.server_url = server_url
        self.parallel = parallel
        self.post = post

    def sample(
        self,
        conversations: list[list[dict[str, str]]],
        samples_per_prompt: int,
        temperature: float,
        max_tokens: int = 512,
        batch_size: int = 8,  # unused: concurrency is ``parallel`` requests across all prompts
    ) -> list[list[str]]:
        """Texts per conversation (``samples_per_prompt`` each); an empty string where a request failed."""

        def one(messages: list[dict[str, str]]) -> str:
            payload: dict[str, object] = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            try:
                return self.post(self.server_url, payload)
            except OSError:
                return ""

        jobs = [messages for messages in conversations for _ in range(samples_per_prompt)]
        with ThreadPoolExecutor(self.parallel) as pool:
            texts = list(pool.map(one, jobs))
        return [texts[start : start + samples_per_prompt] for start in range(0, len(texts), samples_per_prompt)]
