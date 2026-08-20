"""Contract tests.

These assert the properties the governance layer depends on: a closed set of risk
verdicts, validated evidence, and an explicit representation of abstention.
"""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from app.contracts import (
    AbstainReason,
    ActionCategory,
    AskRequest,
    AuditEvent,
    FinalResponse,
    GuardrailDecision,
    PolicyEvidence,
    PolicyFinding,
    ResponseStatus,
    RetrievalResult,
    RiskAssessment,
    RiskVerdict,
)


def _evidence(**overrides) -> PolicyEvidence:
    return PolicyEvidence(
        **{
            "policy_id": "CARD-DISP-001",
            "title": "Card Transaction Dispute Policy",
            "section": "3. Time limits",
            "text": "Disputes must be lodged within 90 calendar days.",
            "score": 0.5,
            **overrides,
        }
    )


class TestRiskVerdict:
    def test_has_exactly_three_values(self):
        # The guardrail branches on this set. Adding a fourth verdict, or a numeric
        # scale, silently changes what happens to a consequential action.
        assert [v.value for v in RiskVerdict] == [
            "SAFE",
            "HUMAN_REVIEW_REQUIRED",
            "REJECTED",
        ]

    def test_rejects_unknown_verdict(self):
        with pytest.raises(ValidationError):
            RiskAssessment(verdict="LOW")

    @pytest.mark.parametrize(
        ("verdict", "blocks", "demands_review"),
        [
            (RiskVerdict.SAFE, False, False),
            (RiskVerdict.HUMAN_REVIEW_REQUIRED, False, True),
            (RiskVerdict.REJECTED, True, True),
        ],
    )
    def test_verdict_drives_downstream_behaviour(self, verdict, blocks, demands_review):
        assessment = RiskAssessment(verdict=verdict)
        assert assessment.blocks_answer is blocks
        assert assessment.demands_review is demands_review


class TestPolicyEvidence:
    def test_accepts_a_complete_citation(self):
        evidence = _evidence()
        assert evidence.policy_id and evidence.title and evidence.section and evidence.text

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_rejects_score_outside_unit_range(self, score):
        with pytest.raises(ValidationError):
            _evidence(score=score)

    @pytest.mark.parametrize("field", ["policy_id", "title", "section", "text"])
    def test_rejects_empty_citation_fields(self, field):
        # An empty section or text makes a citation uncheckable against the corpus.
        with pytest.raises(ValidationError):
            _evidence(**{field: ""})


class TestRetrievalResult:
    def test_insufficient_evidence_is_explicit(self):
        result = RetrievalResult(
            query="what is the tax treatment of a write-off?",
            sufficient=False,
            best_score=0.04,
            threshold=0.15,
            abstain_reason=AbstainReason.INSUFFICIENT_EVIDENCE,
            explanation="below threshold",
        )
        assert result.sufficient is False
        assert result.evidence == []
        assert result.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE

    def test_policy_ids_are_deduplicated_in_order(self):
        result = RetrievalResult(
            query="q",
            sufficient=True,
            evidence=[
                _evidence(section="2. Standard procedure"),
                _evidence(section="3. Time limits"),
                _evidence(policy_id="FRAUD-ESC-002", title="Fraud Escalation Policy"),
            ],
        )
        assert result.policy_ids == ["CARD-DISP-001", "FRAUD-ESC-002"]


class TestRequestAndResponse:
    def test_request_rejects_unknown_fields(self):
        # Guards against a caller smuggling in an override the governance layer ignores.
        with pytest.raises(ValidationError):
            AskRequest(question="How do I lodge a dispute?", override_guardrail=True)

    def test_request_rejects_empty_question(self):
        with pytest.raises(ValidationError):
            AskRequest(question="")

    def test_final_response_generates_a_trace_id(self):
        first = FinalResponse(status=ResponseStatus.ABSTAINED, answer="No policy covers this.")
        second = FinalResponse(status=ResponseStatus.ABSTAINED, answer="No policy covers this.")
        assert first.trace_id and first.trace_id != second.trace_id

    def test_response_carries_the_full_decision_record(self):
        response = FinalResponse(
            status=ResponseStatus.PENDING_HUMAN_REVIEW,
            answer="Refer to a Team Leader before waiving the fee.",
            citations=[_evidence()],
            required_procedures=["Obtain Team Leader approval"],
            risk=RiskAssessment(verdict=RiskVerdict.HUMAN_REVIEW_REQUIRED),
            guardrail=GuardrailDecision(
                requires_human_review=True,
                action_category=ActionCategory.CONSEQUENTIAL,
                triggered_rules=["fee_waiver"],
            ),
            model="groq:llama-3.3-70b-versatile",
        )
        assert response.guardrail.requires_human_review is True
        assert response.risk.verdict is RiskVerdict.HUMAN_REVIEW_REQUIRED
        assert response.model


class TestPolicyFindingAndGuardrail:
    def test_finding_defaults_to_no_claims(self):
        finding = PolicyFinding(answerable=False, abstain_reason=AbstainReason.OUT_OF_POLICY_SCOPE)
        assert finding.proposed_guidance == ""
        assert finding.required_procedures == []
        assert finding.cited_policy_ids == []

    def test_guardrail_defaults_to_informational(self):
        decision = GuardrailDecision(requires_human_review=False)
        assert decision.action_category is ActionCategory.INFORMATIONAL
        assert decision.triggered_rules == []


class TestAuditEvent:
    def test_records_a_timezone_aware_timestamp(self):
        event = AuditEvent(trace_id="abc", stage="guardrail", deterministic=True)
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.astimezone(UTC) == event.timestamp

    def test_rejects_an_unknown_stage(self):
        with pytest.raises(ValidationError):
            AuditEvent(trace_id="abc", stage="freeform", deterministic=True)

    def test_marks_whether_a_decision_was_deterministic(self):
        rule = AuditEvent(trace_id="abc", stage="guardrail", deterministic=True)
        model = AuditEvent(trace_id="abc", stage="risk_agent", deterministic=False)
        assert rule.deterministic and not model.deterministic
