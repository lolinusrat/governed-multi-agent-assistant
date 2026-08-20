"""Nebius AI Studio implementation of `LLMClient`.

Nebius exposes an OpenAI-shaped chat-completions endpoint, so this client is a
transport only: it posts to `{base_url}/chat/completions` with `httpx` and hands
the reply to the shared JSON-chat logic. No vendor SDK, and therefore no new
dependency — `httpx` is already used by the UI.

The base URL is configuration rather than a constant on purpose. Provider
endpoints move, and when one does it should be an environment change, not a code
change.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.llm.base import LLMError, T
from app.llm.json_chat import Messages, structured_via_chat

DEFAULT_BASE_URL = "https://api.studio.nebius.com/v1"


class NebiusClient:
    """Calls Nebius AI Studio's chat completions API and validates into a model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        timeout: float = 60.0,
        max_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        # `client` is an injection point so tests exercise this class without a
        # network call or an API key.
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_attempts = max(1, max_attempts)

    @property
    def model_name(self) -> str:
        return f"nebius:{self.model}"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        return structured_via_chat(
            self._complete,
            system=system,
            user=user,
            schema=schema,
            max_attempts=self.max_attempts,
            provider="Nebius",
        )

    def _complete(self, messages: Messages) -> str:
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "response_format": {"type": "json_object"},
                },
            )
        except Exception as exc:  # noqa: BLE001 - any transport failure is a provider failure
            raise LLMError(f"Nebius request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(f"Nebius request failed: HTTP {response.status_code} - {response.text[:300]}")

        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("Nebius returned a malformed completion") from exc

        if not content or not content.strip():
            raise LLMError("Nebius returned an empty completion")
        return content
