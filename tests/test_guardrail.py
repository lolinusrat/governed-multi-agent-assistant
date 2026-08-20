"""Guardrail tests.

This is the control the rest of the system is built to protect, so the tests are
written as claims about behaviour that must hold for every input, not as examples
of behaviour that happened to work once.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    ActionCategory,
    ConsequentialAction,
    GuardrailDecision,
    PolicyEvidence,
    PolicyFinding,
    ResponseStatus,
    RiskAssessment,
    RiskStatus,
)
from app.guardrail import ACTION_RULES, evaluate, resolve_status, scan

# One phrasing per consequential action, plus variants that must also be caught.
ACTION_PHRASES: dict[ConsequentialAction, list[str]] = {
    ConsequentialAction.TRANSFER_FUNDS: [
        "Transfer the funds back to the customer's account.",
        "Move the money to the nominated account.",
        "Arrange a funds transfer to settle the dispute.",
        "Disburse the amount to the customer.",
    ],
    ConsequentialAction.APPROVE_CREDIT: [
        "Approve the credit limit increase for the customer.",
        "Grant the overdraft the customer has asked for.",
        "Extend the loan facility by the requested amount.",
        "The loan can be approved at the counter.",
    ],
    ConsequentialAction.CLOSE_ACCOUNT: [
        "Close the account at the customer's request.",
        "Proceed with account closure.",
        "Terminate the account and issue a final statement.",
    ],
    ConsequentialAction.BLOCK_ACCOUNT: [
        "Block the account while the matter is investigated.",
        "Freeze the account immediately.",
        "Restrict the account until the customer responds.",
        "Unblock the account once the customer calls back.",
    ],
}

SAFE_RISK = RiskAssessment(status=RiskStatus.SAFE, reason="nothing found")
EVIDENCE = PolicyEvidence(
    policy_id="FRAUD-ESC-002",
    title="Fraud Escalation Policy",
    section="5. Consequential actions requiring human approval",
    text=(
        "| Action | Approval authority |\n"
        "|---|---|\n"
        "| Closing an account associated with a fraud case | Financial Crime Duty Officer |\n"
        "| Freezing or restricting an account beyond a card block | Fraud Operations Manager |"
    ),
    score=0.7,
)


def finding(guidance: str = "Record the details in the disputes system.", **overrides):
    base = {
        "answerable": True,
        "proposed_guidance": guidance,
        "required_procedures": [],
        "cited_policy_ids": ["FRAUD-ESC-002"],
        "evidence": [EVIDENCE],
    }
    return PolicyFinding(**{**base, **overrides})


class TestConsequentialActionsAlwaysRequireReview:
    """The claim the whole layer exists to make."""

    @pytest.mark.parametrize(
        ("action", "phrase"),
        [(a, p) for a, phrases in ACTION_PHRASES.items() for p in phrases],
    )
    def test_every_phrasing_requires_human_review(self, action, phrase):
        decision = evaluate(finding(phrase), SAFE_RISK)
        assert decision.requires_human_review is True
        assert decision.permits_autonomous_execution is False
        assert action in decision.detected_actions
        assert decision.action_category is ActionCategory.CONSEQUENTIAL

    @pytest.mark.parametrize(
        ("action", "phrase"),
        [(a, p) for a, phrases in ACTION_PHRASES.items() for p in phrases],
    )
    @pytest.mark.parametrize("risk_status", list(RiskStatus))
    def test_no_risk_status_can_authorise_autonomous_execution(self, action, phrase, risk_status):
        # The exhaustive form of "the LLM must not be able to override this":
        # every action phrasing, crossed with every status the Risk Agent can
        # return, including the one the model most wants to give.
        decision = evaluate(finding(phrase), RiskAssessment(status=risk_status))
        assert decision.requires_human_review is True

    @pytest.mark.parametrize("phrase", list(itertools.chain.from_iterable(ACTION_PHRASES.values())))
    def test_a_consequential_action_never_resolves_to_answered(self, phrase):
        # ANSWERED is the only status that means "act on this without a person".
        result = finding(phrase)
        decision = evaluate(result, SAFE_RISK)
        assert resolve_status(result, SAFE_RISK, decision) is ResponseStatus.PENDING_HUMAN_REVIEW

    def test_an_action_in_a_procedural_step_counts_too(self):
        decision = evaluate(
            finding("Follow the standard process.", required_procedures=["Close the account."]),
            SAFE_RISK,
        )
        assert decision.requires_human_review is True
        assert ConsequentialAction.CLOSE_ACCOUNT in decision.detected_actions

    def test_several_actions_are_all_reported(self):
        decision = evaluate(
            finding("Close the account and transfer the funds to the new one."), SAFE_RISK
        )
        assert set(decision.detected_actions) == {
            ConsequentialAction.CLOSE_ACCOUNT,
            ConsequentialAction.TRANSFER_FUNDS,
        }


class TestTheLLMCannotReachThisLayer:
    def test_the_guardrail_module_imports_nothing_from_the_llm_package(self):
        # Structural, not behavioural: the layer cannot consult a model even by
        # accident, and a future edit that wires one in fails here.
        source = Path("app/guardrail.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not any(name.startswith("app.llm") or name == "groq" for name in imported)

    def test_evaluate_takes_no_client_or_configuration(self):
        import inspect

        assert list(inspect.signature(evaluate).parameters) == ["finding", "risk"]

    def test_the_decision_cannot_be_edited_after_it_is_made(self):
        decision = evaluate(finding("Close the account."), SAFE_RISK)
        with pytest.raises(ValidationError):
            decision.requires_human_review = False

    def test_the_same_input_always_gives_the_same_decision(self):
        a = evaluate(finding("Freeze the account."), SAFE_RISK)
        b = evaluate(finding("Freeze the account."), SAFE_RISK)
        assert a.model_dump() == b.model_dump()


class TestInformationalGuidance:
    @pytest.mark.parametrize(
        "guidance",
        [
            "Record the dispute in the disputes system within one business day.",
            "Tell the customer the assessment window is 21 calendar days.",
            "Acknowledge the complaint within one business day.",
            "Verify the customer's identity before discussing transaction detail.",
        ],
    )
    def test_ordinary_procedural_guidance_is_not_escalated(self, guidance):
        decision = evaluate(finding(guidance), SAFE_RISK)
        assert decision.requires_human_review is False
        assert decision.permits_autonomous_execution is True
        assert decision.detected_actions == []
        assert decision.action_category is ActionCategory.INFORMATIONAL

    def test_informational_guidance_resolves_to_answered(self):
        result = finding("Acknowledge the complaint within one business day.")
        decision = evaluate(result, SAFE_RISK)
        assert resolve_status(result, SAFE_RISK, decision) is ResponseStatus.ANSWERED


class TestRiskCanOnlyEscalate:
    def test_risk_escalation_requires_review_without_any_action_match(self):
        decision = evaluate(
            finding("Tell the customer the outcome."),
            RiskAssessment(status=RiskStatus.HUMAN_REVIEW_REQUIRED, reason="unclear grounding"),
        )
        assert decision.requires_human_review is True
        assert decision.detected_actions == []
        assert decision.action_category is ActionCategory.INFORMATIONAL

    def test_a_rejection_resolves_to_rejected(self):
        result = finding("Close the account.")
        risk = RiskAssessment(status=RiskStatus.REJECTED, reason="unsupported")
        assert resolve_status(result, risk, evaluate(result, risk)) is ResponseStatus.REJECTED


class TestAbstention:
    def test_an_abstention_needs_no_review(self):
        decision = evaluate(PolicyFinding(answerable=False), SAFE_RISK)
        assert decision.requires_human_review is False
        assert decision.detected_actions == []
        assert "abstained" in decision.rationale

    def test_an_abstention_resolves_to_abstained(self):
        result = PolicyFinding(answerable=False)
        assert resolve_status(result, SAFE_RISK, evaluate(result, SAFE_RISK)) is ResponseStatus.ABSTAINED


class TestEscalationDirection:
    def test_a_prohibition_still_escalates(self):
        # Deliberate: the guardrail does not do negation handling. Over-escalation
        # costs a reviewer a minute; a missed escalation costs a customer.
        decision = evaluate(finding("Do not close the account without approval."), SAFE_RISK)
        assert decision.requires_human_review is True

    def test_a_question_about_an_action_is_not_enough_on_its_own(self):
        # Only the proposed guidance is scanned. An abstention about account
        # closure is not an action, so it is not escalated.
        decision = evaluate(PolicyFinding(answerable=False), SAFE_RISK)
        assert decision.requires_human_review is False


class TestExplainability:
    def test_the_rationale_names_the_actions_detected(self):
        decision = evaluate(finding("Transfer the funds to the customer."), SAFE_RISK)
        assert "TRANSFER_FUNDS" in decision.rationale
        assert decision.triggered_rules == ["transfer_funds"]

    def test_the_approval_authority_is_lifted_from_the_cited_evidence(self):
        decision = evaluate(finding("Freeze the account pending investigation."), SAFE_RISK)
        assert "Fraud Operations Manager" in decision.approval_authorities

    def test_table_headers_are_not_mistaken_for_authorities(self):
        decision = evaluate(finding("Close the account."), SAFE_RISK)
        assert "Approval authority" not in decision.approval_authorities
        assert "Financial Crime Duty Officer" in decision.approval_authorities

    def test_every_rule_has_a_name_and_a_description(self):
        for rule in ACTION_RULES:
            assert rule.name and rule.description
            assert isinstance(rule.action, ConsequentialAction)

    def test_the_rule_table_covers_exactly_the_four_demonstration_actions(self):
        assert {rule.action for rule in ACTION_RULES} == set(ConsequentialAction)


class TestScan:
    def test_scan_is_a_pure_function_over_text(self):
        assert [r.name for r in scan("Close the account.")] == ["close_account"]
        assert scan("Acknowledge the complaint.") == []
