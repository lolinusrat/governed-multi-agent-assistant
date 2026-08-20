"""Risk Agent tests.

The model boundary is stubbed throughout. What is worth testing here is the
deterministic rule pass, the escalate-only combination of the two passes, and the
guarantee that reviewing does not touch the Policy Agent's output.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.agents.risk import RiskAgent, RiskDraft, RiskFlagDraft, assess_deterministically
from app.contracts import (
    AskRequest,
    PolicyEvidence,
    PolicyFinding,
    RiskAssessment,
    RiskCategory,
    RiskStatus,
)
from app.llm.base import LLMError
from app.llm.stub import FailingLLMClient, StubLLMClient

QUESTION = AskRequest(question="A customer wants the dispute handling fee waived. Can I do that?")

EVIDENCE = PolicyEvidence(
    policy_id="CARD-DISP-001",
    title="Card Transaction Dispute Policy",
    section="5. Consequential actions requiring human approval",
    text=(
        "Waiving a card replacement or dispute-handling fee requires approval by the "
        "Branch or Contact Centre Manager. Disputes must be lodged within 90 calendar days. "
        "A provisional credit may be applied where the amount is 500 AUD or less."
    ),
    score=0.6,
)


def finding(**overrides) -> PolicyFinding:
    base = {
        "answerable": True,
        "proposed_guidance": "Disputes must be lodged within 90 calendar days.",
        "required_procedures": ["Verify the customer's identity"],
        "cited_policy_ids": ["CARD-DISP-001"],
        "evidence": [EVIDENCE],
    }
    return PolicyFinding(**{**base, **overrides})


def agent(draft: RiskDraft | None = None) -> RiskAgent:
    return RiskAgent(llm=StubLLMClient([draft] if draft is not None else []))


SAFE_DRAFT = RiskDraft(status=RiskStatus.SAFE, reason="supported by the evidence")


class TestStatusContract:
    def test_only_three_statuses_exist(self):
        assert [s.value for s in RiskStatus] == ["SAFE", "HUMAN_REVIEW_REQUIRED", "REJECTED"]

    @pytest.mark.parametrize("bad", ["MEDIUM", "LOW", "high", ""])
    def test_an_invented_status_never_validates(self, bad):
        with pytest.raises(ValidationError):
            RiskDraft(status=bad)
        with pytest.raises(ValidationError):
            RiskAssessment(status=bad)

    def test_assessment_reports_status_reason_and_identified_risks(self):
        assessment = agent(SAFE_DRAFT).review(QUESTION, finding())
        assert assessment.status in set(RiskStatus)
        assert isinstance(assessment.reason, str)
        assert isinstance(assessment.identified_risks, list)


class TestDeterministicChecks:
    def test_clean_guidance_raises_no_rule_flags(self):
        assert assess_deterministically(finding()) == []

    def test_detects_a_consequential_action(self):
        flags = assess_deterministically(
            finding(proposed_guidance="Waive the dispute handling fee for the customer.")
        )
        assert RiskCategory.CONSEQUENTIAL_ACTION in {f.category for f in flags}

    def test_detects_an_action_requiring_approval(self):
        flags = assess_deterministically(
            finding(required_procedures=["Obtain Branch Manager approval before proceeding"])
        )
        assert RiskCategory.APPROVAL_REQUIRED in {f.category for f in flags}

    def test_detects_an_unsupported_guarantee(self):
        flags = assess_deterministically(
            finding(proposed_guidance="We guarantee the disputed amount will be refunded.")
        )
        guarantee = next(f for f in flags if f.category is RiskCategory.UNSUPPORTED_GUARANTEE)
        assert guarantee.severity is RiskStatus.REJECTED

    def test_detects_personal_financial_advice(self):
        flags = assess_deterministically(
            finding(proposed_guidance="The customer should refinance to a lower rate instead.")
        )
        advice = next(f for f in flags if f.category is RiskCategory.PERSONAL_FINANCIAL_ADVICE)
        assert advice.severity is RiskStatus.REJECTED

    @pytest.mark.parametrize(
        "guidance",
        [
            "Read the card number 4111 1111 1111 1111 back to the customer.",
            "Ask the customer to share the one-time code with you to confirm.",
            "Confirm the balance without verifying the customer's identity.",
        ],
    )
    def test_detects_mishandled_sensitive_information(self, guidance):
        flags = assess_deterministically(finding(proposed_guidance=guidance))
        assert RiskCategory.SENSITIVE_INFORMATION in {f.category for f in flags}

    def test_third_party_disclosure_is_escalated_not_rejected(self):
        flags = assess_deterministically(
            finding(proposed_guidance="Provide the account details to the third party who called.")
        )
        flag = next(f for f in flags if f.category is RiskCategory.SENSITIVE_INFORMATION)
        assert flag.severity is RiskStatus.HUMAN_REVIEW_REQUIRED

    def test_a_prohibition_is_not_read_as_a_proposal(self):
        # "Do not disclose to a third party" is the policy working, not a risk.
        flags = assess_deterministically(
            finding(proposed_guidance="Do not disclose the account details to a third party.")
        )
        assert RiskCategory.SENSITIVE_INFORMATION not in {f.category for f in flags}

    def test_flags_record_that_they_came_from_rules(self):
        flags = assess_deterministically(finding(proposed_guidance="Waive the fee."))
        assert all(f.source == "deterministic" for f in flags)
        assert all(f.detail for f in flags)


class TestUnsupportedClaims:
    def test_a_figure_absent_from_the_evidence_is_rejected(self):
        flags = assess_deterministically(
            finding(proposed_guidance="Disputes must be lodged within 120 calendar days.")
        )
        claim = next(f for f in flags if f.category is RiskCategory.UNSUPPORTED_CLAIM)
        assert claim.severity is RiskStatus.REJECTED
        assert "120" in claim.detail

    def test_a_figure_present_in_the_evidence_passes(self):
        # "provisional credit" still flags as a consequential action; what must
        # not appear is an unsupported-claim flag for the figure itself.
        flags = assess_deterministically(
            finding(proposed_guidance="A provisional credit applies up to 500 AUD.")
        )
        assert RiskCategory.UNSUPPORTED_CLAIM not in {f.category for f in flags}

    def test_a_figure_in_a_procedure_is_checked_too(self):
        flags = assess_deterministically(
            finding(required_procedures=["Escalate anything above 9,999 AUD"])
        )
        assert RiskCategory.UNSUPPORTED_CLAIM in {f.category for f in flags}

    def test_a_citation_without_evidence_is_rejected(self):
        flags = assess_deterministically(
            finding(cited_policy_ids=["CARD-DISP-001", "MADE-UP-999"])
        )
        claim = next(f for f in flags if f.category is RiskCategory.UNSUPPORTED_CLAIM)
        assert "MADE-UP-999" in claim.detail

    def test_guidance_with_no_evidence_at_all_is_rejected(self):
        flags = assess_deterministically(
            finding(evidence=[], cited_policy_ids=[], proposed_guidance="Staff may waive the fee.")
        )
        assert any("no supporting evidence" in f.detail for f in flags)


class TestCombiningTheTwoPasses:
    def test_safe_when_neither_pass_finds_anything(self):
        assessment = agent(SAFE_DRAFT).review(QUESTION, finding())
        assert assessment.status is RiskStatus.SAFE
        assert assessment.identified_risks == []

    def test_rules_escalate_even_when_the_model_says_safe(self):
        # The model must not be able to clear a deterministic finding.
        assessment = agent(SAFE_DRAFT).review(
            QUESTION, finding(proposed_guidance="Waive the dispute handling fee.")
        )
        assert assessment.status is RiskStatus.HUMAN_REVIEW_REQUIRED
        assert RiskCategory.CONSEQUENTIAL_ACTION in assessment.categories

    def test_rules_can_reject_even_when_the_model_says_safe(self):
        assessment = agent(SAFE_DRAFT).review(
            QUESTION, finding(proposed_guidance="We guarantee a refund within 90 calendar days.")
        )
        assert assessment.status is RiskStatus.REJECTED

    def test_the_model_can_escalate_what_the_rules_missed(self):
        draft = RiskDraft(
            status=RiskStatus.REJECTED,
            reason="the guidance contradicts the cited section",
            identified_risks=[
                RiskFlagDraft(category=RiskCategory.UNSUPPORTED_CLAIM, detail="contradicts 5.1")
            ],
        )
        assessment = agent(draft).review(QUESTION, finding())
        assert assessment.status is RiskStatus.REJECTED
        assert any(f.source == "model" for f in assessment.identified_risks)

    def test_status_is_the_most_severe_of_everything_found(self):
        draft = RiskDraft(
            status=RiskStatus.HUMAN_REVIEW_REQUIRED,
            identified_risks=[RiskFlagDraft(category=RiskCategory.CONSEQUENTIAL_ACTION)],
        )
        assessment = agent(draft).review(
            QUESTION, finding(proposed_guidance="We guarantee the fee will be waived.")
        )
        assert assessment.status is RiskStatus.REJECTED

    def test_an_unscripted_model_response_escalates_rather_than_passing(self):
        assessment = RiskAgent(llm=StubLLMClient()).review(QUESTION, finding())
        assert assessment.status is RiskStatus.HUMAN_REVIEW_REQUIRED

    def test_reason_explains_the_status(self):
        assessment = agent(SAFE_DRAFT).review(
            QUESTION, finding(proposed_guidance="Waive the dispute handling fee.")
        )
        assert "Human review required" in assessment.reason
        assert RiskCategory.CONSEQUENTIAL_ACTION.value in assessment.reason


class TestAbstainedFindings:
    def test_an_abstention_is_safe_and_costs_no_model_call(self):
        llm = StubLLMClient()
        assessment = RiskAgent(llm=llm).review(
            QUESTION, PolicyFinding(answerable=False, proposed_guidance="")
        )
        assert assessment.status is RiskStatus.SAFE
        assert assessment.identified_risks == []
        assert llm.call_count == 0


class TestIndependenceAndImmutability:
    def test_reviewing_does_not_modify_the_finding(self):
        original = finding(proposed_guidance="Waive the fee after manager approval.")
        before = copy.deepcopy(original.model_dump())
        agent(SAFE_DRAFT).review(QUESTION, original)
        assert original.model_dump() == before

    def test_evidence_cannot_be_altered_at_all(self):
        # Frozen at the contract level, so no later stage can edit a citation.
        with pytest.raises(ValidationError):
            EVIDENCE.text = "something else"

    def test_the_reviewer_does_not_see_the_policy_agents_reasoning(self):
        llm = StubLLMClient([SAFE_DRAFT])
        RiskAgent(llm=llm).review(
            QUESTION, finding(notes="I was unsure about this but went with it anyway")
        )
        assert "unsure about this" not in llm.calls[0].user

    def test_the_reviewer_sees_guidance_procedures_and_evidence(self):
        llm = StubLLMClient([SAFE_DRAFT])
        RiskAgent(llm=llm).review(QUESTION, finding())
        prompt = llm.calls[0].user
        assert "Disputes must be lodged within 90 calendar days." in prompt
        assert "Verify the customer's identity" in prompt
        assert "CARD-DISP-001" in prompt
        assert "5. Consequential actions requiring human approval" in prompt


class TestProviderFailure:
    def test_an_outage_fails_closed_rather_than_returning_safe(self):
        risk_agent = RiskAgent(llm=FailingLLMClient("groq timed out"))
        with pytest.raises(LLMError, match="groq timed out"):
            risk_agent.review(QUESTION, finding())
