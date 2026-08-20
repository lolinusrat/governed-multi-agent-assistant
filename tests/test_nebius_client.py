"""Nebius client tests.

The client exists so that changing provider is a configuration change rather than
a code change. It is exercised through an injected fake transport, so the suite
stays offline and needs no Nebius key.
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm.base import LLMClient, LLMError, build_llm
from app.llm.nebius_client import NebiusClient


class Answer(BaseModel):
    verdict: str = "unknown"
    score: int = 0


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def json(self):
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError("not json")


class FakeTransport:
    """Stands in for `httpx.Client`, returning scripted response bodies."""

    def __init__(self, bodies, status_code=200, raises=None):
        self._bodies = list(bodies)
        self._status = status_code
        self._raises = raises
        self.requests: list[dict] = []

    def post(self, path, json=None):  # noqa: A002 - matches the httpx signature
        self.requests.append({"path": path, "json": json})
        if self._raises is not None:
            raise self._raises
        body = self._bodies.pop(0) if self._bodies else ""
        return FakeResponse(
            {"choices": [{"message": {"content": body}}]} if self._status == 200 else {"error": body},
            status_code=self._status,
        )


def make_client(bodies, status_code=200, raises=None, max_attempts=2):
    transport = FakeTransport(bodies, status_code, raises)
    client = NebiusClient(
        api_key="unused", model="test-model", max_attempts=max_attempts, client=transport
    )
    return client, transport


class TestProtocolConformance:
    def test_satisfies_the_llm_protocol(self):
        client, _ = make_client(['{"verdict": "ok"}'])
        assert isinstance(client, LLMClient)
        assert client.model_name == "nebius:test-model"


class TestStructuredCalls:
    def test_parses_a_valid_response_into_the_schema(self):
        client, transport = make_client([json.dumps({"verdict": "SAFE", "score": 3})])
        result = client.structured(system="s", user="u", schema=Answer)
        assert result.verdict == "SAFE" and result.score == 3
        assert len(transport.requests) == 1

    def test_requests_json_mode_and_embeds_the_schema(self):
        client, transport = make_client(['{"verdict": "ok"}'])
        client.structured(system="SYSTEM RULES", user="the question", schema=Answer)
        body = transport.requests[0]["json"]
        assert transport.requests[0]["path"] == "/chat/completions"
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.0
        assert "SYSTEM RULES" in body["messages"][0]["content"]
        assert "verdict" in body["messages"][0]["content"]
        assert body["messages"][1]["content"] == "the question"

    def test_retries_once_with_the_parse_error_fed_back(self):
        client, transport = make_client(["not json at all", '{"verdict": "SAFE"}'])
        assert client.structured(system="s", user="u", schema=Answer).verdict == "SAFE"
        assert len(transport.requests) == 2
        assert "could not be parsed" in transport.requests[1]["json"]["messages"][-1]["content"]

    def test_raises_when_every_attempt_is_unparseable(self):
        client, _ = make_client(["nope", "still nope"])
        with pytest.raises(LLMError, match="Nebius returned a response that does not match"):
            client.structured(system="s", user="u", schema=Answer)


class TestFailureHandling:
    def test_a_non_200_becomes_an_llm_error(self):
        client, _ = make_client(["rate limited"], status_code=429)
        with pytest.raises(LLMError, match="HTTP 429"):
            client.structured(system="s", user="u", schema=Answer)

    def test_a_transport_failure_becomes_an_llm_error(self):
        client, _ = make_client([], raises=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="connection reset"):
            client.structured(system="s", user="u", schema=Answer)

    def test_an_empty_completion_is_rejected(self):
        client, _ = make_client([""])
        with pytest.raises(LLMError, match="empty completion"):
            client.structured(system="s", user="u", schema=Answer)


class TestFactory:
    def test_the_provider_is_selected_by_configuration_alone(self):
        client = build_llm(
            Settings(LLM_PROVIDER="nebius", NEBIUS_API_KEY="k", NEBIUS_MODEL="some-model")
        )
        assert isinstance(client, NebiusClient)
        assert client.model_name == "nebius:some-model"

    def test_groq_remains_the_configured_default_path(self):
        from app.llm.groq_client import GroqClient

        client = build_llm(Settings(LLM_PROVIDER="groq", GROQ_API_KEY="k", GROQ_MODEL="m"))
        assert isinstance(client, GroqClient)

    def test_a_missing_key_fails_loudly(self):
        with pytest.raises(LLMError, match="NEBIUS_API_KEY"):
            build_llm(Settings(LLM_PROVIDER="nebius", NEBIUS_API_KEY=""))
