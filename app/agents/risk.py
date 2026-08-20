"""Risk Agent.

Reviews the Policy Agent's proposed guidance independently and reports risk. It
reads the finding and returns a `RiskAssessment`; it never writes back. Evidence
is frozen at the contract level, so a citation cannot be altered here even by
mistake.

Two passes, combined:

* A deterministic rule pass over the guidance and procedures. Pattern matching is
  crude, but it is repeatable, explainable and unit-testable, which matters more
  for a control than recall does.
* A model pass for the judgements patterns cannot make. It sees the question,
  the guidance and the evidence, but not the Policy Agent's own reasoning, so it
  reviews the output rather than agreeing with the rationale behind it.

The two are combined by severity, and risk only ever escalates: the model can
raise a status the rules assigned, never lower it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.contracts import (
    AskRequest,
    PolicyFinding,
    RiskAssessment,
    RiskCategory,
    RiskFlag,
    RiskStatus,
)
from app.llm.base import LLMClient

SYSTEM_PROMPT = """\
You are the Risk Agent in an internal banking assistant used by bank staff.

You review guidance that another agent has drafted from bank policy. You are a \
reviewer, not an author: you never rewrite the guidance and never add to it.

You are given the staff question, the proposed guidance, the proposed procedural \
steps, and the policy evidence they were drawn from. Judge the guidance against \
the evidence only.

Identify risks in these categories:

- UNSUPPORTED_CLAIM: a statement, figure, timeframe or authority that the evidence \
does not support, or that contradicts it.
- CONSEQUENTIAL_ACTION: the guidance would have staff take an action that changes a \
customer's money, account or records.
- UNSUPPORTED_GUARANTEE: a promised outcome the bank cannot guarantee, such as \
assuring a customer of a refund or a result.
- PERSONAL_FINANCIAL_ADVICE: a recommendation about a customer's financial position, \
products or choices, rather than a bank procedure.
- APPROVAL_REQUIRED: the evidence requires approval by a named authority before the \
action described may be taken.
- SENSITIVE_INFORMATION: customer information is disclosed, or would be disclosed, \
without the verification or authority the evidence requires.

Set status:

- SAFE: the guidance is fully supported by the evidence and describes no action \
requiring approval.
- HUMAN_REVIEW_REQUIRED: the guidance is sound but concerns a consequential action, \
or an approval, or anything a person should confirm before staff act.
- REJECTED: the guidance is unsupported by the evidence, guarantees an outcome, \
gives personal financial advice, or mishandles sensitive information.

When uncertain, choose the more cautious status.\
"""

_USER_TEMPLATE = """\
STAFF QUESTION: {question}

PROPOSED GUIDANCE:
{guidance}

PROPOSED PROCEDURAL STEPS:
{procedures}

POLICY EVIDENCE THE GUIDANCE MUST REST ON:
{evidence}

Review the guidance against this evidence."""


# --------------------------------------------------------------------------- #
# Deterministic rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rule:
    """One deterministic pattern check."""

    name: str
    category: RiskCategory
    severity: RiskStatus
    pattern: re.Pattern[str]
    detail: str


def _rule(name, category, severity, pattern, detail) -> Rule:
    return Rule(name, category, severity, re.compile(pattern, re.I), detail)


# Applied to the proposed guidance and procedural steps, never to the evidence.
RULES: tuple[Rule, ...] = (
    # Consequential banking actions: money, accounts or records change.
    _rule("waive_fee", RiskCategory.CONSEQUENTIAL_ACTION, RiskStatus.HUMAN_REVIEW_REQUIRED,
          r"\bwaiv(?:e|er|ing)\b", "The guidance involves waiving a fee or charge."),
    _rule("refund_or_credit", RiskCategory.CONSEQUENTIAL_ACTION, RiskStatus.HUMAN_REVIEW_REQUIRED,
          r"\b(?:refund|reimburs\w*|compensat\w*|provisional credit|write[ -]?off)\b",
          "The guidance involves moving money to or from a customer."),
    _rule("close_or_restrict", RiskCategory.CONSEQUENTIAL_ACTION, RiskStatus.HUMAN_REVIEW_REQUIRED,
          r"\b(?:clos(?:e|ing) the account|freez\w+|restrict\w* the account|unblock\w*|reinstat\w*|releas\w+ the hold)\b",
          "The guidance involves changing the state of an account or card."),
    _rule("override_or_delete", RiskCategory.CONSEQUENTIAL_ACTION, RiskStatus.REJECTED,
          r"\b(?:override|bypass|ignore the (?:limit|policy|threshold)|delete the record|de-identify)\b",
          "The guidance involves overriding a control or destroying records."),

    # Approval requirements stated in the guidance itself.
    _rule("approval_named", RiskCategory.APPROVAL_REQUIRED, RiskStatus.HUMAN_REVIEW_REQUIRED,
          r"\b(?:approval|approve[sd]?|authoris\w+|sign[- ]?off)\b",
          "The guidance depends on an approval that a person must give."),

    # Guarantees the bank cannot make.
    _rule("guarantee", RiskCategory.UNSUPPORTED_GUARANTEE, RiskStatus.REJECTED,
          r"\b(?:guarantee[sd]?|guaranteeing|promise[sd]?|assure[sd]?|certainly will|will definitely)\b",
          "The guidance guarantees an outcome to the customer."),
    _rule("assured_outcome", RiskCategory.UNSUPPORTED_GUARANTEE, RiskStatus.REJECTED,
          r"\b(?:in all cases|no risk|without exception|is always (?:approved|successful|refunded))\b",
          "The guidance states an outcome as certain."),

    # Personal financial advice is outside what this assistant may produce.
    _rule("recommend_product", RiskCategory.PERSONAL_FINANCIAL_ADVICE, RiskStatus.REJECTED,
          r"\b(?:should (?:invest|refinance|switch|consolidate)|best (?:option|product|choice) for (?:you|them|the customer)|better off (?:with|switching))\b",
          "The guidance recommends a financial course of action to the customer."),
    _rule("advise_on_finances", RiskCategory.PERSONAL_FINANCIAL_ADVICE, RiskStatus.REJECTED,
          r"\b(?:advise the customer to (?:invest|borrow|refinance|switch)|financial(?:ly)? advis\w+ (?:that|the customer))\b",
          "The guidance offers advice on the customer's financial position."),

    # Sensitive information handling.
    _rule("raw_card_number", RiskCategory.SENSITIVE_INFORMATION, RiskStatus.REJECTED,
          r"\b(?:\d[ -]?){13,16}\b", "The guidance contains what looks like a full card or account number."),
    _rule("credential_disclosure", RiskCategory.SENSITIVE_INFORMATION, RiskStatus.REJECTED,
          r"\b(?:one[- ]time (?:code|password)|otp|passcode|password|pin)\b.{0,40}\b(?:share|provide|read out|confirm|tell)\b",
          "The guidance would have a credential shared or read out."),
    _rule("third_party_disclosure", RiskCategory.SENSITIVE_INFORMATION, RiskStatus.HUMAN_REVIEW_REQUIRED,
          r"\b(?:disclos\w+|shar\w+|provid\w+|send\w*|releas\w+)\b.{0,60}\b(?:third party|another (?:bank|institution)|external part\w+|police|solicitor)\b",
          "The guidance would disclose customer information beyond the bank."),
    _rule("skip_verification", RiskCategory.SENSITIVE_INFORMATION, RiskStatus.REJECTED,
          r"\bwithout (?:verify\w*|verifying|verification|identifying|confirming) (?:the )?(?:customer|identity|them)\b",
          "The guidance would disclose information without verifying identity."),
)

# A phrase inside a prohibition is not a proposal to do the thing.
_NEGATION = re.compile(
    r"\b(?:do not|does not|don't|never|must not|cannot|can't|no|without|refrain from|avoid)\b[^.]{0,40}$",
    re.I,
)

# Numbers that appear in the guidance but nowhere in the evidence.
_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_POLICY_ID = re.compile(r"\b[A-Z]{3,}-[A-Z]{3,}-\d+\b")


def _is_negated(text: str, start: int) -> bool:
    """True when the match sits inside a prohibition rather than an instruction."""
    return bool(_NEGATION.search(text[max(0, start - 60) : start]))


def _check_patterns(text: str) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    for rule in RULES:
        for match in rule.pattern.finditer(text):
            if _is_negated(text, match.start()):
                continue
            flags.append(
                RiskFlag(
                    category=rule.category,
                    severity=rule.severity,
                    detail=f"{rule.detail} (matched {match.group(0).strip()!r})",
                    source="deterministic",
                )
            )
            break  # one flag per rule is enough to escalate
    return flags


def _check_unsupported_numbers(finding: PolicyFinding) -> list[RiskFlag]:
    """Every figure in the guidance must appear in the evidence it rests on.

    Catches the failure that matters most here: a plausible but wrong threshold
    or timeframe, which a reader has no way to spot without the source text.
    """
    evidence_numbers = {
        n.replace(",", "")
        for e in finding.evidence
        for n in _NUMBER.findall(_POLICY_ID.sub(" ", e.text))
    }
    claimed = _NUMBER.findall(_POLICY_ID.sub(" ", finding.proposed_guidance))
    claimed += [n for step in finding.required_procedures for n in _NUMBER.findall(_POLICY_ID.sub(" ", step))]

    unsupported = [n for n in dict.fromkeys(claimed) if n.replace(",", "") not in evidence_numbers]
    if not unsupported:
        return []
    return [
        RiskFlag(
            category=RiskCategory.UNSUPPORTED_CLAIM,
            severity=RiskStatus.REJECTED,
            detail=(
                "The guidance states figures that do not appear in the cited evidence: "
                + ", ".join(unsupported)
            ),
            source="deterministic",
        )
    ]


def _check_citations(finding: PolicyFinding) -> list[RiskFlag]:
    """Re-check grounding independently rather than trusting the Policy Agent."""
    available = {e.policy_id for e in finding.evidence}
    invented = [pid for pid in finding.cited_policy_ids if pid not in available]
    flags = []
    if invented:
        flags.append(
            RiskFlag(
                category=RiskCategory.UNSUPPORTED_CLAIM,
                severity=RiskStatus.REJECTED,
                detail=f"Cited policy ids with no supporting evidence: {', '.join(invented)}",
                source="deterministic",
            )
        )
    if finding.proposed_guidance.strip() and not finding.evidence:
        flags.append(
            RiskFlag(
                category=RiskCategory.UNSUPPORTED_CLAIM,
                severity=RiskStatus.REJECTED,
                detail="Guidance was produced with no supporting evidence attached.",
                source="deterministic",
            )
        )
    return flags


def assess_deterministically(finding: PolicyFinding) -> list[RiskFlag]:
    """The rule pass. Pure, and independent of any model."""
    surface = "\n".join([finding.proposed_guidance, *finding.required_procedures])
    return [
        *_check_citations(finding),
        *_check_unsupported_numbers(finding),
        *_check_patterns(surface),
    ]


# --------------------------------------------------------------------------- #
# Model pass
# --------------------------------------------------------------------------- #


class RiskFlagDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: RiskCategory
    detail: str = ""


class RiskDraft(BaseModel):
    """What the model is asked to return.

    Defaults to HUMAN_REVIEW_REQUIRED so a degenerate or empty response escalates
    to a person rather than passing the guidance as safe.
    """

    model_config = ConfigDict(extra="ignore")

    status: RiskStatus = Field(default=RiskStatus.HUMAN_REVIEW_REQUIRED)
    reason: str = ""
    identified_risks: list[RiskFlagDraft] = Field(default_factory=list)


class RiskAgent:
    """Independently reviews a `PolicyFinding` and reports risk."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def review(self, request: AskRequest, finding: PolicyFinding) -> RiskAssessment:
        """Assess the finding. Returns a new assessment; the finding is untouched."""
        if not finding.answerable:
            return RiskAssessment(
                status=RiskStatus.SAFE,
                reason="The Policy Agent abstained, so there is no guidance to review.",
            )

        rule_flags = assess_deterministically(finding)

        draft = self.llm.structured(
            system=SYSTEM_PROMPT,
            user=_USER_TEMPLATE.format(
                question=request.question.strip(),
                guidance=finding.proposed_guidance.strip(),
                procedures="\n".join(f"- {s}" for s in finding.required_procedures) or "(none)",
                evidence=self._render_evidence(finding),
            ),
            schema=RiskDraft,
        )
        model_flags = [
            RiskFlag(
                category=flag.category,
                severity=draft.status,
                detail=flag.detail or f"The reviewer flagged {flag.category.value}.",
                source="model",
            )
            for flag in draft.identified_risks
        ]

        flags = [*rule_flags, *model_flags]
        status = RiskStatus.most_severe([draft.status, *(f.severity for f in flags)])
        return RiskAssessment(
            status=status, reason=self._reason(status, flags, draft), identified_risks=flags
        )

    @staticmethod
    def _render_evidence(finding: PolicyFinding) -> str:
        return "\n\n".join(
            f"[{i}] {e.policy_id} - {e.section}\n{e.text}"
            for i, e in enumerate(finding.evidence, start=1)
        )

    @staticmethod
    def _reason(status: RiskStatus, flags: list[RiskFlag], draft: RiskDraft) -> str:
        if status is RiskStatus.SAFE:
            return draft.reason or "No risks were identified in the proposed guidance."

        decisive = [f for f in flags if f.severity is status]
        summary = "; ".join(dict.fromkeys(f.detail for f in decisive)) or draft.reason
        categories = ", ".join(dict.fromkeys(f.category.value for f in decisive))
        prefix = (
            "Rejected: the guidance must not be issued as drafted."
            if status is RiskStatus.REJECTED
            else "Human review required before staff act on this guidance."
        )
        return f"{prefix} {categories}. {summary}".strip()
