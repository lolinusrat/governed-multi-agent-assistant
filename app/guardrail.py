"""Deterministic guardrail.

The binding control in the system. It decides whether a person must review the
guidance before staff act on it, and it decides that by pattern matching over
text - no model call, no model input, no way for a model to influence the result.

This module imports nothing from `app.llm`, and that is a property worth keeping:
the guardrail cannot consult a model even by accident.

Two deliberate choices, both erring toward escalation:

* No negation handling. "Do not close the account" still requires review. The
  Risk Agent distinguishes a prohibition from a proposal because it is
  classifying; the guardrail is blocking, and a false escalation costs a person a
  minute while a missed one costs a customer their money.
* Risk may only escalate. A `SAFE` risk status can never clear a detected action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.contracts import (
    ActionCategory,
    ConsequentialAction,
    GuardrailDecision,
    PolicyFinding,
    ResponseStatus,
    RiskAssessment,
)


@dataclass(frozen=True)
class ActionRule:
    """One consequential-action pattern."""

    name: str
    action: ConsequentialAction
    pattern: re.Pattern[str]
    description: str


def _rule(name, action, pattern, description) -> ActionRule:
    return ActionRule(name, action, re.compile(pattern, re.I), description)


# The whole policy of this layer, as data. Read it top to bottom to know exactly
# what the system will escalate.
ACTION_RULES: tuple[ActionRule, ...] = (
    _rule(
        "transfer_funds",
        ConsequentialAction.TRANSFER_FUNDS,
        # Active: "transfer the funds". Passive and reordered: "the funds will be
        # transferred", "send the money to". Model prose uses all three.
        r"\b(?:transfer|transferring|transferred|move|moving|remit\w*|disburse\w*)\b[^.]{0,40}"
        r"\b(?:funds?|money|balance|amount|payment)\b"
        r"|\b(?:funds?|money|balance|payment|amount)\b[^.]{0,40}"
        r"\b(?:transferr?\w*|moved|remitted|disbursed|sent|paid out|credited to)\b"
        r"|\b(?:send|sending|pay|paying)\b[^.]{0,25}\b(?:funds?|money|payment|amount)\b"
        r"|\bfunds? transfer\b",
        "The guidance would move money.",
    ),
    _rule(
        "approve_credit",
        ConsequentialAction.APPROVE_CREDIT,
        r"\b(?:approv\w+|grant\w*|extend\w*|increas\w*)\b[^.]{0,40}"
        r"\b(?:credit|loan|overdraft|facility|credit limit)\b"
        r"|\b(?:credit|loan|overdraft|facility|credit limit)\b[^.]{0,40}"
        r"\b(?:approv\w+|grant\w*|extend\w*|increas\w*)\b",
        "The guidance would approve or extend credit.",
    ),
    _rule(
        "close_account",
        ConsequentialAction.CLOSE_ACCOUNT,
        r"\b(?:clos\w+|terminat\w+|cancel\w*|shut\w*)\b[^.]{0,30}\baccount\b"
        r"|\baccount\b[^.]{0,30}\b(?:clos\w+|terminat\w+|cancell?\w*|shut down)\b"
        r"|\baccount clos\w+\b",
        "The guidance would close an account.",
    ),
    _rule(
        "block_account",
        ConsequentialAction.BLOCK_ACCOUNT,
        r"\b(?:block\w*|unblock\w*|freez\w+|frozen|restrict\w*|suspend\w*)\b[^.]{0,30}\baccount\b"
        r"|\baccount\b[^.]{0,30}\b(?:block\w*|unblock\w*|freez\w+|frozen|restrict\w*|suspend\w*)\b"
        r"|\baccount (?:block|freeze|restriction|suspension)\b",
        "The guidance would block, freeze or restrict an account.",
    ),
)

# Approval authorities are named in the policy tables as "| action | authority |".
_TABLE_ROW = re.compile(r"^\|(?P<action>[^|]+)\|(?P<authority>[^|]+)\|\s*$", re.M)


def scan(text: str) -> list[ActionRule]:
    """Return the rules that match, in table order. Pure and deterministic."""
    return [rule for rule in ACTION_RULES if rule.pattern.search(text)]


def _approval_authorities(finding: PolicyFinding, matched: list[ActionRule]) -> list[str]:
    """Lift the named authority out of the cited evidence, where one is stated."""
    authorities: list[str] = []
    for evidence in finding.evidence:
        for row in _TABLE_ROW.finditer(evidence.text):
            action_cell = row.group("action").strip()
            authority = row.group("authority").strip()
            if authority.lower() in {"approval authority", "---", ""}:
                continue
            if any(rule.pattern.search(action_cell) for rule in matched):
                authorities.append(authority)
    return list(dict.fromkeys(authorities))


def evaluate(finding: PolicyFinding, risk: RiskAssessment) -> GuardrailDecision:
    """Decide whether a person must review before staff act.

    Takes no model, no client and no configuration. The same inputs always give
    the same decision, which is what makes it reviewable as a control.
    """
    if not finding.answerable:
        return GuardrailDecision(
            requires_human_review=False,
            action_category=ActionCategory.INFORMATIONAL,
            rationale="The assistant abstained, so there is no proposed action to review.",
        )

    surface = "\n".join([finding.proposed_guidance, *finding.required_procedures])
    matched = scan(surface)

    # Risk may add a reason for review. It can never remove one.
    risk_requires_review = risk.demands_review
    requires_review = bool(matched) or risk_requires_review

    return GuardrailDecision(
        requires_human_review=requires_review,
        action_category=ActionCategory.CONSEQUENTIAL if matched else ActionCategory.INFORMATIONAL,
        detected_actions=list(dict.fromkeys(rule.action for rule in matched)),
        triggered_rules=[rule.name for rule in matched],
        approval_authorities=_approval_authorities(finding, matched),
        rationale=_rationale(matched, risk, requires_review),
    )


def _rationale(matched: list[ActionRule], risk: RiskAssessment, requires_review: bool) -> str:
    if not requires_review:
        return "No consequential action was detected and the risk review raised nothing."

    parts = [rule.description for rule in matched]
    if matched:
        actions = ", ".join(a.value for a in dict.fromkeys(r.action for r in matched))
        head = f"Human review required before staff act: {actions}."
    else:
        head = "Human review required because the risk review escalated this guidance."
    if risk.demands_review:
        parts.append(f"Risk status is {risk.status.value}.")
    return " ".join([head, *dict.fromkeys(parts)])


def resolve_status(
    finding: PolicyFinding, risk: RiskAssessment, guardrail: GuardrailDecision
) -> ResponseStatus:
    """The terminal state of a request, decided by rules rather than by a model.

    Order matters: an abstention beats everything, then a rejection, then the
    review requirement. `ANSWERED` is reachable only when nothing objected.
    """
    if not finding.answerable:
        return ResponseStatus.ABSTAINED
    if risk.blocks_answer:
        return ResponseStatus.REJECTED
    if guardrail.requires_human_review:
        return ResponseStatus.PENDING_HUMAN_REVIEW
    return ResponseStatus.ANSWERED
