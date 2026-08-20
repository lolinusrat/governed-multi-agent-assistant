"""End-to-end workflow tests.

Retrieval runs against the real synthetic corpus; the model boundary is stubbed
with a scripted client, so the whole workflow runs offline and deterministically.
Each test drives one question through START -> ... -> END and asserts on the
governed result rather than on the intermediate prose.
"""

from __future__ import annotations

import pytest

from app.agents.policy import PolicyDraft
from app.agents.response import ResponseDraft
from app.agents.risk import RiskDraft
from app.contracts import (
    AbstainReason,
    ConsequentialAction,
    FinalResponse,
    ResponseStatus,
    RiskStatus,
)
from app.graph import GovernedAssistant, GraphState, build_graph
from app.llm.base import LLMError
from app.llm.stub import FailingLLMClient, StubLLMClient

CLOSE_QUESTION = "A customer wants to close their account. What do I do?"
COMPLAINT_QUESTION = "How quickly must I acknowledge a customer complaint?"
UNCOVERED_QUESTION = "What is the weather forecast in Sydney tomorrow?"


def script(policy: PolicyDraft, risk: RiskDraft, response: ResponseDraft) -> StubLLMClient:
    """The three model calls the workflow makes, in the order it makes them."""
    return StubLLMClient([policy, risk, response])


def assistant(retriever, llm) -> GovernedAssistant:
    return GovernedAssistant(retriever=retriever, llm=llm)


CLEAN_POLICY = PolicyDraft(
    answerable=True,
    proposed_guidance="Acknowledge the complaint to the customer within one business day.",
    required_procedures=["Record the complaint in the complaints system on the day it is received"],
    cited_policy_ids=["COMP-HAND-004"],
)
CLOSURE_POLICY = PolicyDraft(
    answerable=True,
    proposed_guidance="Closing an account associated with a fraud case requires the Financial Crime Duty Officer.",
    required_procedures=["Record a fraud case in the case management system"],
    cited_policy_ids=["FRAUD-ESC-002"],
)
SAFE_RISK = RiskDraft(status=RiskStatus.SAFE, reason="supported by the evidence")
DRAFTED = ResponseDraft(answer="Here is what the policy requires.")


class TestWorkflowShape:
    def test_the_graph_is_the_declared_straight_line(self, retriever):
        graph = build_graph(retriever=retriever, llm=StubLLMClient())
        nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert nodes == {"policy_agent", "risk_agent", "guardrail", "response_agent"}

        edges = {(e.source, e.target) for e in graph.get_graph().edges}
        assert edges == {
            ("__start__", "policy_agent"),
            ("policy_agent", "risk_agent"),
            ("risk_agent", "guardrail"),
            ("guardrail", "response_agent"),
            ("response_agent", "__end__"),
        }

    def test_no_extra_agents_were_added(self, retriever):
        graph = build_graph(retriever=retriever, llm=StubLLMClient())
        agent_nodes = [n for n in graph.get_graph().nodes if n.endswith("_agent")]
        assert sorted(agent_nodes) == ["policy_agent", "response_agent", "risk_agent"]

    def test_state_carries_every_declared_field(self, retriever):
        state = assistant(retriever, script(CLEAN_POLICY, SAFE_RISK, DRAFTED)).run(COMPLAINT_QUESTION)
        for field in ("trace_id", "question", "finding", "risk", "guardrail", "response"):
            assert field in state, field
        assert set(GraphState.__annotations__) >= {
            "trace_id",
            "question",
            "finding",
            "risk",
            "guardrail",
            "response",
        }


class TestAnsweredRun:
    def test_a_supported_question_is_answered_with_sources(self, retriever):
        response = assistant(retriever, script(CLEAN_POLICY, SAFE_RISK, DRAFTED)).ask(COMPLAINT_QUESTION)
        assert isinstance(response, FinalResponse)
        assert response.status is ResponseStatus.ANSWERED
        assert response.human_review_required is False
        assert response.risk_status is RiskStatus.SAFE
        assert response.policy_sources
        assert response.recommended_next_steps

    def test_every_stage_writes_its_own_slot(self, retriever):
        state = assistant(retriever, script(CLEAN_POLICY, SAFE_RISK, DRAFTED)).run(COMPLAINT_QUESTION)
        assert state["finding"].answerable is True
        assert state["risk"].status is RiskStatus.SAFE
        assert state["guardrail"].requires_human_review is False
        assert state["response"].status is ResponseStatus.ANSWERED
        assert state["error"] is None

    def test_the_workflow_makes_exactly_three_model_calls(self, retriever):
        llm = script(CLEAN_POLICY, SAFE_RISK, DRAFTED)
        assistant(retriever, llm).ask(COMPLAINT_QUESTION)
        assert llm.call_count == 3

    def test_the_trace_id_is_the_same_everywhere(self, retriever):
        state = assistant(retriever, script(CLEAN_POLICY, SAFE_RISK, DRAFTED)).run(
            COMPLAINT_QUESTION, trace_id="fixed-trace"
        )
        assert state["trace_id"] == "fixed-trace"
        assert state["response"].trace_id == "fixed-trace"
        assert all(event.trace_id == "fixed-trace" for event in state["audit"])


class TestConsequentialRun:
    def test_a_consequential_action_ends_pending_review(self, retriever):
        response = assistant(retriever, script(CLOSURE_POLICY, SAFE_RISK, DRAFTED)).ask(CLOSE_QUESTION)
        assert response.status is ResponseStatus.PENDING_HUMAN_REVIEW
        assert response.human_review_required is True
        assert ConsequentialAction.CLOSE_ACCOUNT in response.guardrail.detected_actions

    def test_the_model_calling_it_safe_changes_nothing(self, retriever):
        # The risk model says SAFE at every turn; the guardrail still holds.
        response = assistant(retriever, script(CLOSURE_POLICY, SAFE_RISK, DRAFTED)).ask(CLOSE_QUESTION)
        assert response.risk_status is RiskStatus.SAFE
        assert response.human_review_required is True
        assert response.status is not ResponseStatus.ANSWERED

    def test_the_answer_leads_with_the_review_requirement(self, retriever):
        response = assistant(retriever, script(CLOSURE_POLICY, SAFE_RISK, DRAFTED)).ask(CLOSE_QUESTION)
        assert response.answer.startswith("Human review is required")
        assert response.recommended_next_steps[0].lower().startswith("obtain")


class TestAbstainedRun:
    def test_an_uncovered_question_abstains(self, retriever):
        response = assistant(retriever, StubLLMClient()).ask(UNCOVERED_QUESTION)
        assert response.status is ResponseStatus.ABSTAINED
        assert response.abstain_reason is AbstainReason.NO_RELEVANT_POLICY
        assert response.policy_sources == []
        assert any("Escalate" in step for step in response.recommended_next_steps)

    def test_an_abstention_costs_no_model_call_at_all(self, retriever):
        # Retrieval is below threshold, the Risk Agent short-circuits, and the
        # Response Agent composes from rules. Nothing reaches the provider.
        llm = StubLLMClient()
        assistant(retriever, llm).ask(UNCOVERED_QUESTION)
        assert llm.call_count == 0

    def test_the_audit_trail_records_the_abstention(self, retriever):
        state = assistant(retriever, StubLLMClient()).run(UNCOVERED_QUESTION)
        stages = [event.stage for event in state["audit"]]
        assert stages == ["policy_agent", "risk_agent", "guardrail", "response_agent"]
        assert state["audit"][0].summary == "abstained"


class TestRejectedRun:
    def test_an_unsupported_figure_is_rejected_end_to_end(self, retriever):
        # The Policy Agent's guidance states a timeframe the evidence never gives.
        # The Risk Agent's deterministic pass catches it, and no answer goes out.
        policy = PolicyDraft(
            answerable=True,
            proposed_guidance="Complaints must be acknowledged within 999 business days.",
            cited_policy_ids=["COMP-HAND-004"],
        )
        response = assistant(retriever, script(policy, SAFE_RISK, DRAFTED)).ask(COMPLAINT_QUESTION)
        assert response.status is ResponseStatus.REJECTED
        assert response.risk_status is RiskStatus.REJECTED
        assert "withheld" in response.answer

    def test_a_rejection_skips_the_response_model_call(self, retriever):
        policy = PolicyDraft(
            answerable=True,
            proposed_guidance="We guarantee the complaint will be upheld.",
            cited_policy_ids=["COMP-HAND-004"],
        )
        llm = script(policy, SAFE_RISK, DRAFTED)
        assistant(retriever, llm).ask(COMPLAINT_QUESTION)
        assert llm.call_count == 2  # policy and risk only


class TestControlledErrorHandling:
    def test_a_provider_outage_produces_a_response_rather_than_an_exception(self, retriever):
        response = assistant(retriever, FailingLLMClient("groq timed out")).ask(COMPLAINT_QUESTION)
        assert isinstance(response, FinalResponse)
        assert response.status is ResponseStatus.UNAVAILABLE

    def test_a_failure_fails_closed(self, retriever):
        # An incomplete run must never read as cleared.
        response = assistant(retriever, FailingLLMClient("groq timed out")).ask(COMPLAINT_QUESTION)
        assert response.human_review_required is True
        assert response.risk_status is RiskStatus.HUMAN_REVIEW_REQUIRED
        assert response.policy_sources == []
        assert "Do not treat this as a green light." in response.answer

    def test_the_failure_is_named_in_the_state_and_the_audit_trail(self, retriever):
        state = assistant(retriever, FailingLLMClient("groq timed out")).run(COMPLAINT_QUESTION)
        # In-process state keeps the detail; the audit trail keeps a cause. The
        # provider's words stop at the process boundary.
        assert state["error"].startswith("policy_agent:")
        assert "groq timed out" in state["error"]
        failure = next(e for e in state["audit"] if "failed" in e.summary)
        assert failure.deterministic is True
        assert failure.detail["cause"] == "timeout"
        assert "groq timed out" not in str(failure.detail)

    def test_later_stages_do_not_run_after_a_failure(self, retriever):
        state = assistant(retriever, FailingLLMClient("down")).run(COMPLAINT_QUESTION)
        assert state["risk"] is None
        assert state["guardrail"] is None
        assert [e.stage for e in state["audit"]] == ["policy_agent"]

    def test_a_failure_in_a_later_stage_is_handled_too(self, retriever):
        class RiskFails(StubLLMClient):
            def structured(self, *, system, user, schema):
                if schema.__name__ == "RiskDraft":
                    raise LLMError("risk model unavailable")
                return super().structured(system=system, user=user, schema=schema)

        state = assistant(retriever, RiskFails([CLEAN_POLICY])).run(COMPLAINT_QUESTION)
        assert state["finding"].answerable is True
        assert state["response"].status is ResponseStatus.UNAVAILABLE
        assert state["error"].startswith("risk_agent:")

    def test_a_broken_retriever_is_contained(self, retriever):
        class BrokenRetriever:
            def search(self, query, *, limit=None):
                raise RuntimeError("policy corpus is unreadable")

        response = GovernedAssistant(retriever=BrokenRetriever(), llm=StubLLMClient()).ask("anything")
        assert response.status is ResponseStatus.UNAVAILABLE
        assert response.human_review_required is True
        # Contained means contained: the internal message does not reach staff.
        assert "policy corpus is unreadable" not in response.answer
        assert "policy_agent stage did not complete" in response.answer

    @pytest.mark.parametrize("question", [CLOSE_QUESTION, COMPLAINT_QUESTION, UNCOVERED_QUESTION])
    def test_every_question_returns_a_response(self, retriever, question):
        # There is no input that makes the workflow return nothing.
        response = assistant(retriever, StubLLMClient()).ask(question)
        assert isinstance(response, FinalResponse)
        assert response.answer.strip()


class TestDeterminism:
    def test_the_same_run_twice_gives_the_same_governed_result(self, retriever):
        def once():
            return assistant(retriever, script(CLOSURE_POLICY, SAFE_RISK, DRAFTED)).ask(
                CLOSE_QUESTION, trace_id="fixed"
            )

        first, second = once(), once()
        assert first.model_dump() == second.model_dump()
