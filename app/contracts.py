"""Structured contracts exchanged between the agents and the governance layer.

Every boundary in the system is a Pydantic model. Agents never pass free-form
dictionaries to each other, so a malformed model response fails validation at the
boundary rather than silently propagating into a staff-facing answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

StaffRole = Literal["branch_staff", "contact_centre", "operations"]


def _new_id() -> str:
    return uuid4().hex


class RiskStatus(str, Enum):
    """The only statuses the Risk Agent may return.

    Deliberately three values, not a numeric score: each maps to exactly one
    downstream behaviour, so the guardrail can act on the status without
    interpreting a scale.
    """

    SAFE = "SAFE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REJECTED = "REJECTED"

    @property
    def severity(self) -> int:
        """Ordering used to combine findings. Risk only ever escalates."""
        return _SEVERITY[self]

    @classmethod
    def most_severe(cls, statuses: "Iterable[RiskStatus]") -> "RiskStatus":
        return max(statuses, key=lambda s: s.severity, default=cls.SAFE)


_SEVERITY = {
    RiskStatus.SAFE: 0,
    RiskStatus.HUMAN_REVIEW_REQUIRED: 1,
    RiskStatus.REJECTED: 2,
}


class RiskCategory(str, Enum):
    """The risk types the Risk Agent is required to check for."""

    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    CONSEQUENTIAL_ACTION = "CONSEQUENTIAL_ACTION"
    UNSUPPORTED_GUARANTEE = "UNSUPPORTED_GUARANTEE"
    PERSONAL_FINANCIAL_ADVICE = "PERSONAL_FINANCIAL_ADVICE"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    SENSITIVE_INFORMATION = "SENSITIVE_INFORMATION"


class ConsequentialAction(str, Enum):
    """Actions that may never be taken on the assistant's say-so alone.

    Scoped to the four in this demonstration. Adding one is a change to this
    enum and to the rule table beside it - never a change to a prompt.
    """

    TRANSFER_FUNDS = "TRANSFER_FUNDS"
    APPROVE_CREDIT = "APPROVE_CREDIT"
    CLOSE_ACCOUNT = "CLOSE_ACCOUNT"
    BLOCK_ACCOUNT = "BLOCK_ACCOUNT"


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
    UNAVAILABLE = "UNAVAILABLE"
    """A stage failed, so the governance checks did not complete. Fails closed."""


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

    @field_validator("question")
    @classmethod
    def _must_carry_a_question(cls, value: str) -> str:
        """Whitespace passes a length check but is not a question."""
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("question must contain at least 3 non-whitespace characters")
        return stripped


class PolicyEvidence(BaseModel):
    """One retrieved passage, carrying everything needed to cite it.

    `text` is copied verbatim from the source document. Nothing downstream may
    paraphrase into this field, so a citation can always be checked against the
    file on disk.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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


class RiskFlag(BaseModel):
    """One identified risk, with the severity it carries on its own."""

    model_config = ConfigDict(extra="forbid")

    category: RiskCategory
    detail: str = Field(min_length=1, description="What was found, in reviewer-readable terms")
    severity: RiskStatus = RiskStatus.HUMAN_REVIEW_REQUIRED
    source: Literal["deterministic", "model"] = "deterministic"


class RiskAssessment(BaseModel):
    """Risk Agent output: an independent review of the proposed guidance.

    The Risk Agent reads the Policy Agent's output and never writes to it. This
    model is its entire contribution to the request.
    """

    model_config = ConfigDict(extra="forbid")

    status: RiskStatus
    reason: str = Field(default="", description="Why this status was reached")
    identified_risks: list[RiskFlag] = Field(default_factory=list)

    @property
    def blocks_answer(self) -> bool:
        return self.status is RiskStatus.REJECTED

    @property
    def demands_review(self) -> bool:
        return self.status in (RiskStatus.HUMAN_REVIEW_REQUIRED, RiskStatus.REJECTED)

    @property
    def categories(self) -> list[RiskCategory]:
        return list(dict.fromkeys(flag.category for flag in self.identified_risks))


class GuardrailDecision(BaseModel):
    """Deterministic control output. Produced by rules, never by a model.

    `requires_human_review` is the binding field in the system. Nothing
    downstream may recompute or relax it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requires_human_review: bool
    action_category: ActionCategory = ActionCategory.INFORMATIONAL
    detected_actions: list[ConsequentialAction] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    approval_authorities: list[str] = Field(default_factory=list)
    rationale: str = ""

    @property
    def permits_autonomous_execution(self) -> bool:
        """True only when staff may act without a person reviewing first."""
        return not self.requires_human_review


class FinalResponse(BaseModel):
    """What the API returns and the UI renders.

    The governance fields are flat and authoritative: `risk_status` and
    `human_review_required` are what a caller reads, and a validator refuses to
    build a response whose flat fields disagree with the assessment and guardrail
    decision they came from. A response that overrides the guardrail is not
    something this contract can represent.
    """

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(default_factory=_new_id)
    status: ResponseStatus
    answer: str = Field(min_length=1)
    recommended_next_steps: list[str] = Field(default_factory=list)
    policy_sources: list[PolicyEvidence] = Field(
        default_factory=list, description="The policy sections the answer rests on"
    )
    risk_status: RiskStatus
    human_review_required: bool

    # Full decision record, for audit and for the UI to explain itself.
    risk: RiskAssessment
    guardrail: GuardrailDecision
    abstain_reason: AbstainReason | None = None
    model: str = Field(default="", description="Provider and model used, for explainability")

    @model_validator(mode="after")
    def _governance_fields_cannot_disagree(self) -> "FinalResponse":
        if self.risk_status is not self.risk.status:
            raise ValueError(
                f"risk_status {self.risk_status.value} contradicts the assessment "
                f"({self.risk.status.value})"
            )
        if self.human_review_required != self.guardrail.requires_human_review:
            raise ValueError(
                "human_review_required contradicts the guardrail decision "
                f"({self.guardrail.requires_human_review})"
            )
        if self.status is ResponseStatus.ANSWERED:
            if self.human_review_required:
                raise ValueError("a response requiring human review cannot be ANSWERED")
            if self.risk.blocks_answer:
                raise ValueError("a rejected assessment cannot be ANSWERED")
            if not self.policy_sources:
                raise ValueError("an answered response must cite at least one policy source")
        if self.status is ResponseStatus.ABSTAINED and self.abstain_reason is None:
            raise ValueError("an abstention must say why")
        return self

    @property
    def cited_policy_ids(self) -> list[str]:
        return list(dict.fromkeys(s.policy_id for s in self.policy_sources))


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
