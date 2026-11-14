from eval.generate_server import ServerGenerator


def fake_post(server_url: str, payload: dict[str, object]) -> str:
    return f"{server_url}|{payload['messages'][0]['content']}|{payload['temperature']}"  # type: ignore[index]


def test_samples_are_grouped_per_conversation_in_order() -> None:
    generator = ServerGenerator("http://x", parallel=3, post=fake_post)
    texts = generator.sample([[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]], 2, 0.5)
    assert texts == [["http://x|a|0.5"] * 2, ["http://x|b|0.5"] * 2]


def test_failed_request_gives_an_empty_text() -> None:
    def failing(server_url: str, payload: dict[str, object]) -> str:
        raise ConnectionError

    texts = ServerGenerator("http://x", post=failing).sample([[{"role": "user", "content": "a"}]], 2, 0.5)
    assert texts == [["", ""]]
