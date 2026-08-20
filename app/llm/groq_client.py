"""Groq implementation of `LLMClient`.

The only module in the project that imports the Groq SDK. Everything provider
specific is contained here: JSON mode, the retry on an unparseable response, and
the translation of SDK exceptions into `LLMError`.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.llm.base import LLMError, T

_JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. No prose, no code fences. "
    "It must validate against this JSON schema:\n{schema}"
)

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed: {error}\n"
    "Reply again with a single valid JSON object matching the schema exactly."
)


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
        instruction = _JSON_INSTRUCTION.format(
            schema=json.dumps(schema.model_json_schema(), indent=2)
        )
        messages = [
            {"role": "system", "content": f"{system}\n\n{instruction}"},
            {"role": "user", "content": user},
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            content = self._complete(messages)
            try:
                return schema.model_validate_json(content)
            except (ValidationError, ValueError) as exc:
                last_error = exc
                # Temperature is zero, so a bare retry would return the same text.
                # Feed the failure back instead.
                if attempt + 1 < self.max_attempts:
                    messages = [
                        *messages,
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": _REPAIR_INSTRUCTION.format(error=exc)},
                    ]

        raise LLMError(
            f"Groq returned a response that does not match {schema.__name__} "
            f"after {self.max_attempts} attempt(s): {last_error}"
        )

    def _complete(self, messages: list[dict[str, str]]) -> str:
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
