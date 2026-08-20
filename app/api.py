"""FastAPI surface.

Two endpoints and nothing else: `GET /health` and `POST /ask`.

The API is a thin adapter over the workflow. It does not decide anything - the
governance fields it returns are copied from the `FinalResponse` the graph
produced, and there is no code path here that can alter them.

Three things it does own:

* **Input validation.** Malformed requests are rejected with a field-level message
  and never reach the workflow.
* **Controlled failure.** Nothing raises out of a handler. An unexpected error
  becomes a generic message plus a request id that ties it to a server-side log
  entry holding the real traceback.
* **Output hygiene.** No stack traces, no internal file paths, and any string that
  looks like a provider key is redacted before it leaves the process.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.contracts import AbstainReason, AskRequest, FinalResponse, ResponseStatus, RiskStatus
from app.graph import GovernedAssistant

logger = logging.getLogger("governed_assistant.api")

GENERIC_ERROR = "The assistant could not process this request."

# Anything shaped like a provider key never leaves the process, whatever produced it.
_KEY_SHAPED = re.compile(r"\b(?:gsk|sk|xai|ghp|github_pat)[-_][A-Za-z0-9_\-]{16,}\b")


def redact(text: str, settings: Settings | None = None) -> str:
    """Strip the configured key and anything key-shaped from outgoing text."""
    settings = settings or get_settings()
    if settings.groq_api_key and settings.groq_api_key in text:
        text = text.replace(settings.groq_api_key, "[redacted]")
    return _KEY_SHAPED.sub("[redacted]", text)


# --------------------------------------------------------------------------- #
# Wire contracts
# --------------------------------------------------------------------------- #


class PolicySource(BaseModel):
    """A citation as the caller sees it.

    Deliberately narrower than `PolicyEvidence`: the retrieval score and the file
    path on disk are internal and are not published.
    """

    model_config = ConfigDict(extra="forbid")

    policy_id: str
    title: str
    section: str
    excerpt: str


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    answer: str
    policy_sources: list[PolicySource] = Field(default_factory=list)
    risk_status: RiskStatus
    human_review_required: bool
    status: ResponseStatus
    recommended_next_steps: list[str] = Field(default_factory=list)
    abstain_reason: AbstainReason | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    error: str
    detail: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    policies_loaded: int
    llm_provider: str
    model: str


def to_wire(response: FinalResponse, request_id: str, settings: Settings) -> AskResponse:
    """Project the internal response onto the public contract."""
    return AskResponse(
        request_id=request_id,
        answer=redact(response.answer, settings),
        policy_sources=[
            PolicySource(
                policy_id=s.policy_id,
                title=s.title,
                section=s.section,
                excerpt=s.text,
            )
            for s in response.policy_sources
        ],
        risk_status=response.risk_status,
        human_review_required=response.human_review_required,
        status=response.status,
        recommended_next_steps=[redact(s, settings) for s in response.recommended_next_steps],
        abstain_reason=response.abstain_reason,
    )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


def create_app(assistant: GovernedAssistant | None = None) -> FastAPI:
    """Build the app. `assistant` is injectable so tests never touch a provider."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Built once at startup: the corpus is parsed and the graph compiled here
        # rather than on the first request.
        app.state.assistant = assistant or GovernedAssistant()
        yield

    api = FastAPI(
        title="Governed Banking Policy Assistant",
        version="0.1.0",
        summary="Internal staff-facing assistant over synthetic banking policies.",
        lifespan=lifespan,
    )

    def get_assistant(request: Request) -> GovernedAssistant:
        return request.app.state.assistant

    @api.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Field-level feedback, with the offending input echoed back removed."""
        detail = [
            {"field": ".".join(str(p) for p in err["loc"][1:]) or "body", "message": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=ErrorResponse(
                request_id=uuid4().hex, error="The request was not valid.", detail=detail
            ).model_dump(),
        )

    @api.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort. The traceback goes to the log, never to the caller."""
        request_id = uuid4().hex
        logger.exception("unhandled error on %s (request_id=%s)", request.url.path, request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(request_id=request_id, error=GENERIC_ERROR).model_dump(),
        )

    @api.get("/health", response_model=HealthResponse)
    def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        """Liveness plus enough context to tell which corpus and model are loaded.

        Reports configuration, never credentials.
        """
        try:
            from app.retrieval import load_corpus

            policies = len(load_corpus(settings.policy_path))
            state = "ok"
        except Exception:  # noqa: BLE001 - health must not raise
            logger.exception("health check could not load the policy corpus")
            policies, state = 0, "degraded"

        return HealthResponse(
            status=state,
            policies_loaded=policies,
            llm_provider=settings.llm_provider,
            model=settings.groq_model if settings.llm_provider == "groq" else "stub",
        )

    @api.post(
        "/ask",
        response_model=AskResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    def ask(
        payload: AskRequest,
        assistant: GovernedAssistant = Depends(get_assistant),
        settings: Settings = Depends(get_settings),
    ) -> Any:
        """Answer a staff question, or say why it will not.

        A degraded run returns 503 and still carries the full governed body: the
        caller needs to read "human review required", not just a status code.
        """
        request_id = uuid4().hex
        result = assistant.ask(
            payload.question, staff_role=payload.staff_role, trace_id=request_id
        )
        body = to_wire(result, request_id, settings)

        if result.status is ResponseStatus.UNAVAILABLE:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body.model_dump(mode="json")
            )
        return body

    return api


app = create_app()
