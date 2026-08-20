"""Structured contracts exchanged between the agents and the governance layer.

Every boundary in the system is a Pydantic model. Agents never pass free-form
dictionaries to each other, so a malformed model response fails validation at the
boundary rather than silently propagating into a staff-facing answer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

StaffRole = Literal["branch_staff", "contact_centre", "operations"]


def _new_id() -> str:
    return uuid4().hex


class RiskVerdict(str, Enum):
    """The only verdicts the Risk Agent may return.

    Deliberately three values, not a numeric score: each maps to exactly one
    downstream behaviour, so the guardrail can act on the verdict without
    interpreting a scale.
    """

    SAFE = "SAFE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class ActionCategory(str, Enum):
    """Whether the question seeks information or proposes to change something."""

    INFORMATIONAL = "INFORMATIONAL"
    CONSEQUENTIAL = "CONSEQUENTIAL"


class ResponseStatus(str, Enum):
    """Terminal state of a single request."""

    ANSWERED = "ANSWERED"
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    ABSTAINED = "ABSTAINED"
    REJECTED = "REJECTED"


class AbstainReason(str, Enum):
    """Why the assistant declined to answer. Set deterministically, not by a model."""

    NO_RELEVANT_POLICY = "NO_RELEVANT_POLICY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    OUT_OF_POLICY_SCOPE = "OUT_OF_POLICY_SCOPE"
    UNVERIFIABLE_CITATION = "UNVERIFIABLE_CITATION"


class AskRequest(BaseModel):
    """A staff question arriving at the API."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    staff_role: StaffRole = "branch_staff"


class PolicyEvidence(BaseModel):
    """One retrieved passage, carrying everything needed to cite it.

    `text` is copied verbatim from the source document. Nothing downstream may
    paraphrase into this field, so a citation can always be checked against the
    file on disk.
    """

    model_config = ConfigDict(extra="forbid")

    policy_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    section: str = Field(min_length=1, description="Section heading, e.g. '5. Consequential actions requiring human approval'")
    text: str = Field(min_length=1, description="Verbatim excerpt from the policy document")
    score: float = Field(ge=0.0, le=1.0, description="Retrieval confidence, 0-1")
    source_path: str | None = Field(default=None, description="Path of the source document, for audit")


class RetrievalResult(BaseModel):
    """Outcome of a retrieval call.

    Insufficient evidence is represented explicitly rather than as an empty list,
    so the caller cannot mistake 'found nothing' for 'nothing worth saying'.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    sufficient: bool
    evidence: list[PolicyEvidence] = Field(default_factory=list)
    best_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    abstain_reason: AbstainReason | None = None
    explanation: str = Field(default="", description="Human-readable reason, shown to staff on abstention")

    @property
    def policy_ids(self) -> list[str]:
        return list(dict.fromkeys(e.policy_id for e in self.evidence))


class PolicyFinding(BaseModel):
    """Policy Agent output: what the corpus says and what staff must do.

    Self-contained on purpose. It carries the evidence it was built from, so a
    reviewer can check the guidance against the cited sections without re-running
    retrieval, and downstream stages never need to re-derive the citations.

    It has no field for approving or authorising anything. The Policy Agent
    reports what the policy requires; whether an action may proceed is decided by
    the deterministic guardrail and, where required, a human.
    """

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    proposed_guidance: str = ""
    required_procedures: list[str] = Field(default_factory=list)
    cited_policy_ids: list[str] = Field(default_factory=list)
    evidence: list[PolicyEvidence] = Field(
        default_factory=list, description="The retrieved sections the guidance rests on"
    )
    abstain_reason: AbstainReason | None = None
    notes: str = ""

    @property
    def is_grounded(self) -> bool:
        """True when every claim can be traced to at least one cited section."""
        return bool(self.evidence) and bool(self.cited_policy_ids)


class RiskAssessment(BaseModel):
    """Risk Agent output: an independent review of the proposed guidance."""

    model_config = ConfigDict(extra="forbid")

    verdict: RiskVerdict
    concerns: list[str] = Field(default_factory=list)
    missing_controls: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def blocks_answer(self) -> bool:
        return self.verdict is RiskVerdict.REJECTED

    @property
    def demands_review(self) -> bool:
        return self.verdict in (RiskVerdict.HUMAN_REVIEW_REQUIRED, RiskVerdict.REJECTED)


class GuardrailDecision(BaseModel):
    """Deterministic control output. Produced by rules, never by a model."""

    model_config = ConfigDict(extra="forbid")

    requires_human_review: bool
    action_category: ActionCategory = ActionCategory.INFORMATIONAL
    triggered_rules: list[str] = Field(default_factory=list)
    approval_authorities: list[str] = Field(default_factory=list)
    rationale: str = ""


class FinalResponse(BaseModel):
    """Response Agent output as returned by the API and rendered by the UI."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=_new_id)
    status: ResponseStatus
    answer: str
    citations: list[PolicyEvidence] = Field(default_factory=list)
    required_procedures: list[str] = Field(default_factory=list)
    risk: RiskAssessment | None = None
    guardrail: GuardrailDecision | None = None
    abstain_reason: AbstainReason | None = None
    model: str = Field(default="", description="Provider and model used, for explainability")


class AuditEvent(BaseModel):
    """One record of a decision taken during a request.

    Held in the response and written to the log. Persisting these is an insert
    away, which is why the shape is fixed now rather than later.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=_new_id)
    trace_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: Literal["retrieve", "policy_agent", "risk_agent", "guardrail", "response_agent"]
    deterministic: bool = Field(description="True when produced by rules rather than a model")
    summary: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)
