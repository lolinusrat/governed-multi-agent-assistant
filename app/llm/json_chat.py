"""Shared JSON-mode chat logic for OpenAI-shaped providers.

Groq and Nebius both expose a chat-completions endpoint that can be asked for a
JSON object. Only the transport differs, so the part that matters — instructing
the model with the schema, validating the reply, and repairing one bad response —
lives here once rather than being copied per vendor.

A provider module supplies a `complete(messages) -> str` callable and gets the
whole contract. Nothing here imports a vendor SDK.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from app.llm.base import LLMError, T

JSON_INSTRUCTION = (
    "Reply with a single JSON object and nothing else. No prose, no code fences. "
    "It must validate against this JSON schema:\n{schema}"
)

REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed: {error}\n"
    "Reply again with a single valid JSON object matching the schema exactly."
)

Messages = list[dict[str, str]]


def structured_via_chat(
    complete: Callable[[Messages], str],
    *,
    system: str,
    user: str,
    schema: type[T],
    max_attempts: int = 2,
    provider: str = "provider",
) -> T:
    """Ask for a JSON object, validate it into `schema`, repair once if needed."""
    instruction = JSON_INSTRUCTION.format(schema=json.dumps(schema.model_json_schema(), indent=2))
    messages: Messages = [
        {"role": "system", "content": f"{system}\n\n{instruction}"},
        {"role": "user", "content": user},
    ]

    last_error: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        content = complete(messages)
        try:
            return schema.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            # Temperature is zero, so a bare retry returns the same text. Feed the
            # failure back instead of asking the same question twice.
            if attempt + 1 < max_attempts:
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": REPAIR_INSTRUCTION.format(error=exc)},
                ]

    raise LLMError(
        f"{provider} returned a response that does not match {schema.__name__} "
        f"after {max_attempts} attempt(s): {last_error}"
    )
