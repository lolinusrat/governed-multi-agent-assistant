"""Policy Agent.

Takes a staff question, retrieves policy evidence, and reports what the corpus
says and what procedure it requires. It is the only agent that reads the corpus.

Its boundaries are enforced by code, not only by the prompt:

* Retrieval decides whether there is enough evidence to attempt an answer. If
  there is not, the agent abstains and the model is never called.
* Citations the model returns are checked against what was actually retrieved.
  An id the retriever did not supply is discarded, and guidance left with no
  verifiable citation becomes an abstention.
* `PolicyFinding` has no field in which an action could be approved. This agent
  reports the approval a policy requires; it never grants one.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import AbstainReason, AskRequest, PolicyEvidence, PolicyFinding, RetrievalResult
from app.llm.base import LLMClient
from app.retrieval import PolicyRetriever

SYSTEM_PROMPT = """\
You are the Policy Agent in an internal banking assistant used by bank staff.

You are given a staff question and numbered POLICY EVIDENCE extracted from the \
bank's policy documents. Those extracts are the only source you may use.

Rules, in order of priority:

1. Use only the supplied evidence. Do not use general knowledge about banking, \
regulation or other institutions. If the evidence does not state something, it is \
not known to you.
2. Never invent a policy, a policy id, a section number, a threshold, a timeframe \
or an approval authority. Every id you cite must appear in the evidence.
3. If the evidence does not answer the question, set answerable to false. Abstaining \
is the correct outcome, not a failure.
4. If the evidence explicitly says the topic is not covered by the policy - for \
example a section listing matters the policy does not address - set out_of_scope to \
true and answerable to false, even if the section otherwise seems related.
5. You do not decide, approve or authorise anything. Where a policy requires \
approval before an action, state that requirement as a procedural step naming the \
approval authority from the evidence. Never tell staff that an action is approved, \
permitted without approval, or that you are approving it.
6. Procedural steps must be concrete, in the order staff should perform them, and \
each must be supported by the evidence.

Write for a member of bank staff: plain, direct, no hedging padding.\
"""

_USER_TEMPLATE = """\
STAFF QUESTION ({staff_role}): {question}

POLICY EVIDENCE:
{evidence}

Produce your finding from this evidence alone."""

_EVIDENCE_TEMPLATE = """\
[{index}] policy_id: {policy_id}
    title: {title}
    section: {section}
    text: {text}"""


class PolicyDraft(BaseModel):
    """What the model is asked to return.

    Kept separate from `PolicyFinding` so the model cannot populate the fields the
    agent derives itself - the evidence and the abstention reason. Every field
    defaults to the conservative option, so a degenerate response abstains.
    """

    model_config = ConfigDict(extra="ignore")

    answerable: bool = Field(default=False, description="True only if the evidence answers the question")
    out_of_scope: bool = Field(default=False, description="True if the evidence states the topic is not covered")
    proposed_guidance: str = Field(default="", description="Grounded guidance for staff; empty if not answerable")
    required_procedures: list[str] = Field(default_factory=list, description="Ordered steps staff must follow")
    cited_policy_ids: list[str] = Field(default_factory=list, description="Policy ids drawn from the evidence")
    notes: str = Field(default="", description="Anything a reviewer should know, including gaps in the evidence")


def format_evidence(evidence: list[PolicyEvidence]) -> str:
    """Render retrieved sections into the prompt, ids and sections included."""
    return "\n\n".join(
        _EVIDENCE_TEMPLATE.format(
            index=i,
            policy_id=e.policy_id,
            title=e.title,
            section=e.section,
            text=e.text,
        )
        for i, e in enumerate(evidence, start=1)
    )


class PolicyAgent:
    """Retrieves policy evidence and turns it into a structured finding."""

    def __init__(self, retriever: PolicyRetriever, llm: LLMClient) -> None:
        self.retriever = retriever
        self.llm = llm

    def run(self, request: AskRequest) -> PolicyFinding:
        """Answer from policy, or abstain. Never raises for lack of evidence."""
        retrieval = self.retriever.search(request.question)

        if not retrieval.sufficient:
            return self._abstain(
                reason=retrieval.abstain_reason or AbstainReason.INSUFFICIENT_EVIDENCE,
                notes=retrieval.explanation,
            )

        draft = self.llm.structured(
            system=SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(
                staff_role=request.staff_role.replace("_", " "),
                question=request.question.strip(),
                evidence=format_evidence(retrieval.evidence),
            ),
            schema=PolicyDraft,
        )
        return self._finalise(draft, retrieval)

    # -- deterministic post-checks ------------------------------------------ #

    def _finalise(self, draft: PolicyDraft, retrieval: RetrievalResult) -> PolicyFinding:
        """Apply the checks that do not depend on the model behaving well."""
        if draft.out_of_scope:
            return self._abstain(
                reason=AbstainReason.OUT_OF_POLICY_SCOPE,
                notes=draft.notes or "The retrieved policy states this topic is not covered.",
            )

        if not draft.answerable:
            return self._abstain(
                reason=AbstainReason.INSUFFICIENT_EVIDENCE,
                notes=draft.notes or retrieval.explanation,
            )

        # Discard any id the retriever did not actually return.
        retrieved_ids = retrieval.policy_ids
        cited = [pid for pid in dict.fromkeys(draft.cited_policy_ids) if pid in retrieved_ids]

        if not cited:
            return self._abstain(
                reason=AbstainReason.UNVERIFIABLE_CITATION,
                notes=(
                    "The proposed guidance could not be traced to any retrieved policy section."
                    + (f" Notes: {draft.notes}" if draft.notes else "")
                ),
            )

        if not draft.proposed_guidance.strip():
            return self._abstain(
                reason=AbstainReason.INSUFFICIENT_EVIDENCE,
                notes=draft.notes or "The model produced no guidance for this question.",
            )

        return PolicyFinding(
            answerable=True,
            proposed_guidance=draft.proposed_guidance.strip(),
            required_procedures=[s.strip() for s in draft.required_procedures if s.strip()],
            cited_policy_ids=cited,
            evidence=[e for e in retrieval.evidence if e.policy_id in cited],
            notes=draft.notes,
        )

    @staticmethod
    def _abstain(*, reason: AbstainReason, notes: str) -> PolicyFinding:
        """An abstention carries no guidance and no citations, by construction."""
        return PolicyFinding(answerable=False, abstain_reason=reason, notes=notes)
