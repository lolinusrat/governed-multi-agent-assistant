"""Response Agent.

Composes the staff-facing answer. It is the last stage, and it is the one with
the least authority: the status, the risk status and the human-review flag are
all decided before it runs, and it cannot change any of them.

What it may do is write prose. What it may not do is change what the prose is
allowed to say:

* The terminal status comes from `guardrail.resolve_status`, not from this agent.
* An abstention and a rejection are composed deterministically, with no model
  call at all - there is nothing to draft when the answer must not be issued.
* When human review is required, the response leads with a deterministic banner
  saying so, and the first recommended step is obtaining that review.
* `FinalResponse` itself refuses to be constructed with governance fields that
  disagree with the assessment and guardrail decision they came from.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import (
    AbstainReason,
    AskRequest,
    FinalResponse,
    GuardrailDecision,
    PolicyFinding,
    ResponseStatus,
    RiskAssessment,
)
from app.guardrail import resolve_status
from app.llm.base import LLMClient

SYSTEM_PROMPT = """\
You are the Response Agent in an internal banking assistant used by bank staff.

You are given a staff question, guidance already drafted from bank policy, the \
procedural steps that go with it, and the policy evidence behind it. Write the \
final answer the member of staff will read.

Rules:

1. Say only what the guidance and evidence support. Add nothing from general \
knowledge. If something is not in the evidence, it is not in your answer.
2. Refer to policies by their policy id and section, as they appear in the evidence.
3. Do not tell staff that an action is approved, permitted, cleared or that they \
may proceed. You do not have that authority and neither does the guidance.
4. If human review is required, write on the basis that a person must confirm \
before anything is done. Never suggest the review can be skipped, hurried or \
treated as a formality.
5. Be brief and direct. Short paragraphs, no preamble, no restating the question \
back, no closing pleasantries.

Write for a member of bank staff who needs to act correctly, not for a customer.\
"""

_USER_TEMPLATE = """\
STAFF QUESTION: {question}

GUIDANCE DRAWN FROM POLICY:
{guidance}

PROCEDURAL STEPS:
{procedures}

POLICY EVIDENCE:
{evidence}

HUMAN REVIEW REQUIRED: {review}{review_detail}

Write the final answer."""

_REVIEW_BANNER = (
    "Human review is required before you act on this. {detail}"
)

_ABSTAIN_ANSWER = (
    "I cannot answer this from the bank's policies. {explanation}\n\n"
    "Do not answer this from general knowledge or from what seems reasonable. "
    "Escalate it instead."
)

_REJECTED_ANSWER = (
    "This guidance has been withheld. The risk review found it must not be issued "
    "as drafted.\n\n{reason}\n\n"
    "Do not act on any part of it. Escalate the question instead."
)

_ESCALATION_STEPS = (
    "Do not give the customer an answer based on this request.",
    "Escalate the question to your team leader or the policy owner named in the relevant policy.",
    "Record the question and the outcome so the gap can be reviewed.",
)

# The model may not claim authority it does not have. If it does, its draft is
# discarded rather than edited.
_AUTHORITY_CLAIMS = re.compile(
    r"\b(?:you may (?:proceed|go ahead|action this)|no (?:approval|review) (?:is )?(?:needed|required)"
    r"|this is approved|approval is not required|you can proceed without|safe to proceed"
    r"|no need to escalate|skip the review)\b",
    re.I,
)


class ResponseDraft(BaseModel):
    """What the model is asked to return. Prose only - no governance fields."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(default="", description="The staff-facing answer")
    next_steps: list[str] = Field(default_factory=list, description="Ordered actions for staff")


class ResponseAgent:
    """Turns a reviewed finding into the response staff actually read."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def respond(
        self,
        request: AskRequest,
        finding: PolicyFinding,
        risk: RiskAssessment,
        guardrail: GuardrailDecision,
    ) -> FinalResponse:
        status = resolve_status(finding, risk, guardrail)

        if status is ResponseStatus.ABSTAINED:
            return self._abstain(finding, risk, guardrail)
        if status is ResponseStatus.REJECTED:
            return self._reject(finding, risk, guardrail)

        return self._compose(request, finding, risk, guardrail, status)

    # -- composed answers ---------------------------------------------------- #

    def _compose(
        self,
        request: AskRequest,
        finding: PolicyFinding,
        risk: RiskAssessment,
        guardrail: GuardrailDecision,
        status: ResponseStatus,
    ) -> FinalResponse:
        review_detail = self._review_detail(guardrail)
        draft = self.llm.structured(
            system=SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(
                question=request.question.strip(),
                guidance=finding.proposed_guidance.strip(),
                procedures="\n".join(f"- {s}" for s in finding.required_procedures) or "(none)",
                evidence=self._render_evidence(finding),
                review="yes" if guardrail.requires_human_review else "no",
                review_detail=f"\nREVIEW REASON: {review_detail}" if review_detail else "",
            ),
            schema=ResponseDraft,
        )

        body = draft.answer.strip()
        if not body or _AUTHORITY_CLAIMS.search(body):
            # Fall back to the policy guidance verbatim rather than correcting the
            # model's claim: the guidance is already grounded and already reviewed.
            body = finding.proposed_guidance.strip()

        if guardrail.requires_human_review:
            body = f"{_REVIEW_BANNER.format(detail=review_detail).strip()}\n\n{body}"

        return FinalResponse(
            status=status,
            answer=body,
            recommended_next_steps=self._next_steps(finding, guardrail, draft.next_steps),
            policy_sources=list(finding.evidence),
            risk_status=risk.status,
            human_review_required=guardrail.requires_human_review,
            risk=risk,
            guardrail=guardrail,
            model=getattr(self.llm, "model_name", ""),
        )

    def _next_steps(
        self, finding: PolicyFinding, guardrail: GuardrailDecision, drafted: list[str]
    ) -> list[str]:
        """Procedure first, review requirement ahead of all of it."""
        steps = [s.strip() for s in [*finding.required_procedures, *drafted] if s.strip()]
        steps = list(dict.fromkeys(steps))

        if guardrail.requires_human_review:
            authorities = ", ".join(guardrail.approval_authorities)
            first = (
                f"Obtain approval from {authorities} before taking any action."
                if authorities
                else "Obtain human review and approval before taking any action."
            )
            steps = [first, *steps]
        return steps

    # -- deterministic answers ----------------------------------------------- #

    @staticmethod
    def _abstain(
        finding: PolicyFinding, risk: RiskAssessment, guardrail: GuardrailDecision
    ) -> FinalResponse:
        reason = finding.abstain_reason or AbstainReason.INSUFFICIENT_EVIDENCE
        return FinalResponse(
            status=ResponseStatus.ABSTAINED,
            answer=_ABSTAIN_ANSWER.format(explanation=finding.notes.strip() or _EXPLANATIONS[reason]),
            recommended_next_steps=list(_ESCALATION_STEPS),
            policy_sources=[],
            risk_status=risk.status,
            human_review_required=guardrail.requires_human_review,
            risk=risk,
            guardrail=guardrail,
            abstain_reason=reason,
            model="deterministic",
        )

    @staticmethod
    def _reject(
        finding: PolicyFinding, risk: RiskAssessment, guardrail: GuardrailDecision
    ) -> FinalResponse:
        return FinalResponse(
            status=ResponseStatus.REJECTED,
            answer=_REJECTED_ANSWER.format(reason=risk.reason.strip() or "No reason was recorded."),
            recommended_next_steps=list(_ESCALATION_STEPS),
            policy_sources=list(finding.evidence),
            risk_status=risk.status,
            human_review_required=guardrail.requires_human_review,
            risk=risk,
            guardrail=guardrail,
            model="deterministic",
        )

    # -- helpers -------------------------------------------------------------- #

    @staticmethod
    def _render_evidence(finding: PolicyFinding) -> str:
        return "\n\n".join(
            f"[{i}] {e.policy_id} - {e.section} ({e.title})\n{e.text}"
            for i, e in enumerate(finding.evidence, start=1)
        )

    @staticmethod
    def _review_detail(guardrail: GuardrailDecision) -> str:
        if not guardrail.requires_human_review:
            return ""
        if guardrail.approval_authorities:
            return f"Approval must come from {', '.join(guardrail.approval_authorities)}."
        return guardrail.rationale


_EXPLANATIONS = {
    AbstainReason.NO_RELEVANT_POLICY: "No policy in the corpus covers this question.",
    AbstainReason.INSUFFICIENT_EVIDENCE: (
        "The policies do not say enough about this to give a reliable answer."
    ),
    AbstainReason.OUT_OF_POLICY_SCOPE: (
        "The relevant policy states explicitly that this topic is not covered by it."
    ),
    AbstainReason.UNVERIFIABLE_CITATION: (
        "The drafted guidance could not be traced back to a policy section."
    ),
}
