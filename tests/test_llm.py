"""Tests for the provider abstraction.

The Groq client is exercised through an injected fake SDK object, so the suite
never makes a network call and never needs an API key.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.llm.base import LLMClient, LLMError, build_llm
from app.llm.groq_client import GroqClient
from app.llm.stub import FailingLLMClient, StubLLMClient


class Answer(BaseModel):
    verdict: str = "unknown"
    score: int = 0


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeGroqSDK:
    """Minimal stand-in for `groq.Groq`, returning scripted response bodies."""

    def __init__(self, bodies, raises: Exception | None = None):
        self._bodies = list(bodies)
        self._raises = raises
        self.requests: list[dict] = []
        self.chat = type("chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return _Completion(self._bodies.pop(0) if self._bodies else None)


def make_client(bodies, raises=None, max_attempts=2):
    sdk = FakeGroqSDK(bodies, raises)
    return GroqClient(api_key="unused", model="test-model", max_attempts=max_attempts, client=sdk), sdk


class TestProtocolConformance:
    def test_both_stubs_satisfy_the_protocol(self):
        assert isinstance(StubLLMClient(), LLMClient)
        assert isinstance(FailingLLMClient(), LLMClient)

    def test_groq_client_satisfies_the_protocol(self):
        client, _ = make_client(['{"verdict": "ok"}'])
        assert isinstance(client, LLMClient)
        assert client.model_name == "groq:test-model"


class TestGroqClient:
    def test_parses_a_valid_response_into_the_schema(self):
        client, sdk = make_client([json.dumps({"verdict": "SAFE", "score": 3})])
        result = client.structured(system="s", user="u", schema=Answer)
        assert isinstance(result, Answer)
        assert result.verdict == "SAFE" and result.score == 3
        assert len(sdk.requests) == 1

    def test_requests_json_mode_and_embeds_the_schema(self):
        client, sdk = make_client(['{"verdict": "ok"}'])
        client.structured(system="SYSTEM RULES", user="the question", schema=Answer)
        request = sdk.requests[0]
        assert request["response_format"] == {"type": "json_object"}
        assert request["temperature"] == 0.0
        assert "SYSTEM RULES" in request["messages"][0]["content"]
        assert "verdict" in request["messages"][0]["content"]  # schema included
        assert request["messages"][1]["content"] == "the question"

    def test_retries_once_with_the_parse_error_fed_back(self):
        client, sdk = make_client(["not json at all", '{"verdict": "SAFE"}'])
        result = client.structured(system="s", user="u", schema=Answer)
        assert result.verdict == "SAFE"
        assert len(sdk.requests) == 2
        repair = sdk.requests[1]["messages"][-1]["content"]
        assert "could not be parsed" in repair

    def test_raises_when_every_attempt_is_unparseable(self):
        client, sdk = make_client(["nope", "still nope"])
        with pytest.raises(LLMError, match="does not match Answer"):
            client.structured(system="s", user="u", schema=Answer)
        assert len(sdk.requests) == 2

    def test_wraps_sdk_failures(self):
        client, _ = make_client([], raises=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="connection reset"):
            client.structured(system="s", user="u", schema=Answer)

    def test_rejects_an_empty_completion(self):
        client, _ = make_client([""])
        with pytest.raises(LLMError, match="empty completion"):
            client.structured(system="s", user="u", schema=Answer)


class TestStubClient:
    def test_replays_scripted_responses_in_order(self):
        stub = StubLLMClient([Answer(verdict="first"), Answer(verdict="second")])
        assert stub.structured(system="s", user="u", schema=Answer).verdict == "first"
        assert stub.structured(system="s", user="u", schema=Answer).verdict == "second"
        assert stub.call_count == 2

    def test_records_the_prompts_it_was_given(self):
        stub = StubLLMClient([Answer()])
        stub.structured(system="the rules", user="the question", schema=Answer)
        assert stub.calls[0].system == "the rules"
        assert stub.calls[0].user == "the question"
        assert stub.calls[0].schema is Answer

    def test_falls_back_to_schema_defaults_when_unscripted(self):
        assert StubLLMClient().structured(system="s", user="u", schema=Answer).verdict == "unknown"

    def test_rejects_a_response_of_the_wrong_type(self):
        class Other(BaseModel):
            x: int = 1

        stub = StubLLMClient([Other()])
        with pytest.raises(LLMError, match="scripted a Other"):
            stub.structured(system="s", user="u", schema=Answer)


class TestFactory:
    def test_stub_provider_needs_no_api_key(self, monkeypatch):
        from app.config import Settings

        settings = Settings(LLM_PROVIDER="stub", GROQ_API_KEY="")
        assert isinstance(build_llm(settings), StubLLMClient)

    def test_groq_provider_without_a_key_fails_loudly(self):
        from app.config import Settings

        settings = Settings(LLM_PROVIDER="groq", GROQ_API_KEY="")
        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            build_llm(settings)
