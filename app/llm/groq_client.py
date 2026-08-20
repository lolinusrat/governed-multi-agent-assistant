"""Groq implementation of `LLMClient`.

The only module in the project that imports the Groq SDK. Everything provider
specific is contained here: JSON mode, the retry on an unparseable response, and
the translation of SDK exceptions into `LLMError`.
"""

from __future__ import annotations

from typing import Any

from app.llm.base import LLMError, T
from app.llm.json_chat import Messages, structured_via_chat

class GroqClient:
    """Calls Groq's chat completions API and validates the result into a model."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_attempts: int = 2,
        client: Any | None = None,
    ) -> None:
        # `client` is an injection point so tests can exercise this class without
        # a network call or an API key.
        if client is None:
            from groq import Groq

            client = Groq(api_key=api_key, timeout=timeout)
        self._client = client
        self.model = model
        self.temperature = temperature
        self.max_attempts = max(1, max_attempts)

    @property
    def model_name(self) -> str:
        return f"groq:{self.model}"

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        return structured_via_chat(
            self._complete,
            system=system,
            user=user,
            schema=schema,
            max_attempts=self.max_attempts,
            provider="Groq",
        )

    def _complete(self, messages: Messages) -> str:
        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001 - any SDK failure is a provider failure
            raise LLMError(f"Groq request failed: {exc}") from exc

        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError("Groq returned a malformed completion") from exc

        if not content or not content.strip():
            raise LLMError("Groq returned an empty completion")
        return content
