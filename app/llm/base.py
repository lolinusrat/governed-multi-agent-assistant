"""The seam between the application and whichever model provider is configured.

One method, one shape: give it a system prompt, a user prompt and a Pydantic
schema, get a validated instance of that schema back. Callers never see provider
SDK objects, raw JSON, or provider-specific errors.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from app.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A provider call failed, or returned something that would not validate.

    Deliberately not swallowed into an abstention: a provider outage is an
    availability problem, and reporting it as "no policy covers this" would be a
    misleading answer to a member of staff.
    """


@runtime_checkable
class LLMClient(Protocol):
    """The only LLM interface the application depends on."""

    model_name: str

    def structured(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of `schema`, or raise `LLMError`."""
        ...


def build_llm(settings: Settings | None = None) -> LLMClient:
    """Construct the configured client. The one place a provider is chosen."""
    settings = settings or get_settings()

    if settings.llm_provider == "stub":
        from app.llm.stub import StubLLMClient

        return StubLLMClient()

    from app.llm.groq_client import GroqClient

    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set; set it in .env or use LLM_PROVIDER=stub")
    return GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    """Process-wide client, built once at first use."""
    global _client
    if _client is None:
        _client = build_llm()
    return _client
