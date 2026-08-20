"""A deterministic client used by the test suite and for offline runs.

Keeps the whole suite free of network calls and API keys, and records what it was
asked so tests can assert on the prompts the agents actually build.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.llm.base import LLMError, T


@dataclass(frozen=True)
class RecordedCall:
    """One `structured()` invocation, kept for assertions."""

    system: str
    user: str
    schema: type[BaseModel]


class StubLLMClient:
    """Replays scripted responses in order.

    When the script runs out it falls back to constructing the schema with its
    own defaults. Contracts in this project default to the conservative option —
    `PolicyDraft` defaults to `answerable=False` — so an unscripted call abstains
    rather than inventing an answer.
    """

    model_name = "stub:deterministic"

    def __init__(self, responses: Iterable[BaseModel] | None = None) -> None:
        self._responses: deque[BaseModel] = deque(responses or ())
        self.calls: list[RecordedCall] = []

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(RecordedCall(system=system, user=user, schema=schema))

        if self._responses:
            response = self._responses.popleft()
            if not isinstance(response, schema):
                raise LLMError(
                    f"stub scripted a {type(response).__name__} but {schema.__name__} was requested"
                )
            return response

        try:
            return schema()
        except Exception as exc:  # noqa: BLE001 - surfaced as a provider-level failure
            raise LLMError(f"stub has no scripted response for {schema.__name__}") from exc

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FailingLLMClient:
    """Stands in for a provider outage."""

    model_name = "stub:failing"

    def __init__(self, message: str = "provider unavailable") -> None:
        self.message = message

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        raise LLMError(self.message)
