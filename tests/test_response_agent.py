"""Response Agent tests.

The agent writes prose; it decides nothing. These tests are mostly about what it
cannot do - override a status, clear a review requirement, or answer without
citing a source.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.response import ResponseAgent, ResponseDraft
from app.contracts import (
    AbstainReason,
    ActionCategory,
    AskRequest,
    ConsequentialAction,
    FinalResponse,
    GuardrailDecision,
    PolicyEvidence,
    PolicyFinding,
    ResponseStatus,
    RiskAssessment,
    RiskCategory,
    RiskFlag,
    RiskStatus,
)
from app.llm.base import LLMError
from app.llm.stub import FailingLLMClient, StubLLMClient

QUESTION = AskRequest(question="A customer wants to close their account. What do I do?")

EVIDENCE = PolicyEvidence(
    policy_id="FRAUD-ESC-002",
    title="Fraud Escalation Policy",
    section="5. Consequential actions requiring human approval",
    text="Closing an account associated with a fraud case requires the Financial Crime Duty Officer.",
    score=0.7,
)

SAFE_RISK = RiskAssessment(status=RiskStatus.SAFE, reason="nothing found")
REVIEW_RISK = RiskAssessment(
    status=RiskStatus.HUMAN_REVIEW_REQUIRED,
    reason="consequential action",
    identified_risks=[RiskFlag(category=RiskCategory.CONSEQUENTIAL_ACTION, detail="closure")],
)
REJECT_RISK = RiskAssessment(status=RiskStatus.REJECTED, reason="the guidance guarantees an outcome")

CLEAR = GuardrailDecision(requires_human_review=False)
REVIEW = GuardrailDecision(
    requires_human_review=True,
    action_category=ActionCategory.CONSEQUENTIAL,
    detected_actions=[ConsequentialAction.CLOSE_ACCOUNT],
    triggered_rules=["close_account"],
    approval_authorities=["Financial Crime Duty Officer"],
    rationale="Human review required before staff act: CLOSE_ACCOUNT.",
)


def finding(**overrides) -> PolicyFinding:
    base = {
        "answerable": True,
        "proposed_guidance": "Account closure requires approval by the Financial Crime Duty Officer.",
        "required_procedures": ["Record a case in the case management system"],
        "cited_policy_ids": ["FRAUD-ESC-002"],
        "evidence": [EVIDENCE],
    }
    return PolicyFinding(**{**base, **overrides})


def agent(draft: ResponseDraft | None = None) -> ResponseAgent:
    return ResponseAgent(llm=StubLLMClient([draft] if draft is not None else []))


DRAFT = ResponseDraft(
    answer="Account closure requires the Financial Crime Duty Officer to approve it first.",
    next_steps=["Confirm the customer's identity"],
)


class TestRequiredOutputShape:
    def test_returns_the_four_required_fields(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert isinstance(response, FinalResponse)
        assert response.recommended_next_steps
        assert response.policy_sources
        assert response.risk_status is RiskStatus.SAFE
        assert response.human_review_required is False

    def test_policy_sources_carry_id_title_and_section(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        for source in response.policy_sources:
            assert source.policy_id and source.title and source.section and source.text
        assert response.cited_policy_ids == ["FRAUD-ESC-002"]

    def test_sources_are_passed_through_untouched(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert response.policy_sources == [EVIDENCE]

    def test_the_full_decision_record_travels_with_the_response(self):
        response = agent(DRAFT).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        assert response.risk is REVIEW_RISK
        assert response.guardrail is REVIEW
        assert response.model


class TestCannotOverrideHumanReview:
    def test_review_requirement_is_carried_into_the_response(self):
        response = agent(DRAFT).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        assert response.human_review_required is True
        assert response.status is ResponseStatus.PENDING_HUMAN_REVIEW

    def test_a_model_claiming_no_approval_is_needed_is_discarded(self):
        # The draft is dropped rather than patched: the grounded guidance is
        # already correct, so there is nothing to salvage from a false claim.
        rogue = ResponseDraft(answer="No approval is required, you may proceed and close it.")
        response = agent(rogue).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        assert "you may proceed" not in response.answer.lower()
        assert response.human_review_required is True

    @pytest.mark.parametrize(
        "claim",
        [
            "You may proceed without approval.",
            "This is approved, go ahead.",
            "No review is required here.",
            "It is safe to proceed with the closure.",
            "No need to escalate this one.",
        ],
    )
    def test_authority_claims_never_survive_into_the_answer(self, claim):
        response = agent(ResponseDraft(answer=claim)).respond(
            QUESTION, finding(), REVIEW_RISK, REVIEW
        )
        assert claim.lower() not in response.answer.lower()

    def test_the_answer_leads_with_the_review_requirement(self):
        response = agent(DRAFT).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        assert response.answer.startswith("Human review is required before you act on this.")
        assert "Financial Crime Duty Officer" in response.answer

    def test_obtaining_approval_is_the_first_recommended_step(self):
        response = agent(DRAFT).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        assert response.recommended_next_steps[0].startswith("Obtain approval from")
        assert "Financial Crime Duty Officer" in response.recommended_next_steps[0]

    def test_guardrail_review_holds_even_when_risk_is_safe(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, REVIEW)
        assert response.human_review_required is True
        assert response.status is ResponseStatus.PENDING_HUMAN_REVIEW
        assert response.risk_status is RiskStatus.SAFE


class TestCannotOverrideRejection:
    def test_a_rejected_assessment_produces_a_withheld_answer(self):
        response = agent(DRAFT).respond(QUESTION, finding(), REJECT_RISK, CLEAR)
        assert response.status is ResponseStatus.REJECTED
        assert response.risk_status is RiskStatus.REJECTED
        assert "withheld" in response.answer

    def test_a_rejection_never_calls_the_model(self):
        # There is nothing to draft when the guidance must not be issued.
        llm = StubLLMClient()
        ResponseAgent(llm=llm).respond(QUESTION, finding(), REJECT_RISK, CLEAR)
        assert llm.call_count == 0

    def test_a_rejection_explains_itself_and_recommends_escalation(self):
        response = agent().respond(QUESTION, finding(), REJECT_RISK, CLEAR)
        assert "guarantees an outcome" in response.answer
        assert any("Escalate" in step for step in response.recommended_next_steps)

    def test_a_rejected_response_can_never_be_answered(self):
        with pytest.raises(ValidationError, match="rejected assessment cannot be ANSWERED"):
            FinalResponse(
                status=ResponseStatus.ANSWERED,
                answer="here you go",
                policy_sources=[EVIDENCE],
                risk_status=RiskStatus.REJECTED,
                human_review_required=False,
                risk=REJECT_RISK,
                guardrail=CLEAR,
            )


class TestGovernanceFieldsCannotDisagree:
    def test_a_response_contradicting_the_guardrail_will_not_construct(self):
        with pytest.raises(ValidationError, match="contradicts the guardrail decision"):
            FinalResponse(
                status=ResponseStatus.ANSWERED,
                answer="go ahead",
                policy_sources=[EVIDENCE],
                risk_status=RiskStatus.SAFE,
                human_review_required=False,
                risk=SAFE_RISK,
                guardrail=REVIEW,
            )

    def test_a_response_contradicting_the_risk_assessment_will_not_construct(self):
        with pytest.raises(ValidationError, match="contradicts the assessment"):
            FinalResponse(
                status=ResponseStatus.PENDING_HUMAN_REVIEW,
                answer="text",
                policy_sources=[EVIDENCE],
                risk_status=RiskStatus.SAFE,
                human_review_required=True,
                risk=REVIEW_RISK,
                guardrail=REVIEW,
            )

    def test_an_answered_response_requiring_review_will_not_construct(self):
        with pytest.raises(ValidationError, match="requiring human review cannot be ANSWERED"):
            FinalResponse(
                status=ResponseStatus.ANSWERED,
                answer="text",
                policy_sources=[EVIDENCE],
                risk_status=RiskStatus.SAFE,
                human_review_required=True,
                risk=SAFE_RISK,
                guardrail=REVIEW,
            )

    def test_an_answered_response_must_cite_a_policy_source(self):
        with pytest.raises(ValidationError, match="must cite at least one policy source"):
            FinalResponse(
                status=ResponseStatus.ANSWERED,
                answer="text",
                policy_sources=[],
                risk_status=RiskStatus.SAFE,
                human_review_required=False,
                risk=SAFE_RISK,
                guardrail=CLEAR,
            )


class TestAbstention:
    def test_an_abstention_is_composed_without_the_model(self):
        llm = StubLLMClient()
        response = ResponseAgent(llm=llm).respond(
            QUESTION,
            PolicyFinding(answerable=False, abstain_reason=AbstainReason.NO_RELEVANT_POLICY),
            SAFE_RISK,
            CLEAR,
        )
        assert response.status is ResponseStatus.ABSTAINED
        assert llm.call_count == 0

    def test_an_abstention_recommends_escalation(self):
        response = agent().respond(
            QUESTION,
            PolicyFinding(answerable=False, abstain_reason=AbstainReason.NO_RELEVANT_POLICY),
            SAFE_RISK,
            CLEAR,
        )
        assert any("Escalate" in step for step in response.recommended_next_steps)
        assert "Escalate it instead." in response.answer

    def test_an_abstention_cites_nothing(self):
        response = agent().respond(
            QUESTION,
            PolicyFinding(answerable=False, abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE),
            SAFE_RISK,
            CLEAR,
        )
        assert response.policy_sources == []

    @pytest.mark.parametrize("reason", list(AbstainReason))
    def test_every_abstain_reason_produces_an_explained_answer(self, reason):
        response = agent().respond(
            QUESTION, PolicyFinding(answerable=False, abstain_reason=reason), SAFE_RISK, CLEAR
        )
        assert response.abstain_reason is reason
        assert len(response.answer) > 60
        assert "general knowledge" in response.answer

    def test_the_abstention_carries_the_retrieval_explanation_when_there_is_one(self):
        response = agent().respond(
            QUESTION,
            PolicyFinding(
                answerable=False,
                abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE,
                notes="The closest match was CARD-DISP-001 at 0.08, below the 0.15 threshold.",
            ),
            SAFE_RISK,
            CLEAR,
        )
        assert "below the 0.15 threshold" in response.answer


class TestAnsweredPath:
    def test_a_clean_answer_is_answered_and_needs_no_review(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert response.status is ResponseStatus.ANSWERED
        assert response.human_review_required is False
        assert not response.answer.startswith("Human review is required")

    def test_procedures_come_through_as_next_steps(self):
        response = agent(DRAFT).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert "Record a case in the case management system" in response.recommended_next_steps
        assert "Confirm the customer's identity" in response.recommended_next_steps

    def test_duplicate_steps_are_collapsed(self):
        response = agent(
            ResponseDraft(answer="text", next_steps=["Record a case in the case management system"])
        ).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert response.recommended_next_steps.count("Record a case in the case management system") == 1

    def test_an_empty_draft_falls_back_to_the_grounded_guidance(self):
        response = agent(ResponseDraft(answer="   ")).respond(QUESTION, finding(), SAFE_RISK, CLEAR)
        assert response.answer == finding().proposed_guidance


class TestPromptConstruction:
    def test_the_prompt_carries_question_guidance_evidence_and_review_state(self):
        llm = StubLLMClient([DRAFT])
        ResponseAgent(llm=llm).respond(QUESTION, finding(), REVIEW_RISK, REVIEW)
        prompt = llm.calls[0].user
        assert QUESTION.question in prompt
        assert "FRAUD-ESC-002" in prompt
        assert "5. Consequential actions requiring human approval" in prompt
        assert "HUMAN REVIEW REQUIRED: yes" in prompt

    def test_the_system_prompt_forbids_granting_permission(self):
        from app.agents.response import SYSTEM_PROMPT

        assert "do not tell staff that an action is approved" in SYSTEM_PROMPT.lower()


class TestProviderFailure:
    def test_an_outage_is_not_disguised_as_an_answer(self):
        response_agent = ResponseAgent(llm=FailingLLMClient("groq timed out"))
        with pytest.raises(LLMError, match="groq timed out"):
            response_agent.respond(QUESTION, finding(), SAFE_RISK, CLEAR)
