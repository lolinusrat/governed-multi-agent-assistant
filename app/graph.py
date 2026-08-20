"""LangGraph orchestration.

A straight line, on purpose:

    START -> policy_agent -> risk_agent -> guardrail -> response_agent -> END

There are no conditional edges and no branching. The decisions that could branch -
abstain, reject, escalate - are already represented inside the contracts, and each
stage knows how to do nothing when there is nothing to do. An abstention costs no
model call because the Policy Agent returns early, the Risk Agent short-circuits
on an unanswerable finding, and the Response Agent composes the abstention from
rules. Keeping the graph linear means the flow you read here is the flow that runs.

Error handling is controlled rather than absent. Any stage that raises stops the
run and produces a `FinalResponse` with status `UNAVAILABLE` that **requires human
review**: if the governance checks did not complete, the system must not imply
that anything was cleared.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.policy import PolicyAgent
from app.agents.response import ResponseAgent
from app.agents.risk import RiskAgent
from app.contracts import (
    AskRequest,
    AuditEvent,
    FinalResponse,
    GuardrailDecision,
    PolicyFinding,
    ResponseStatus,
    RiskAssessment,
    RiskStatus,
    StaffRole,
)
from app.guardrail import evaluate
from app.llm.base import LLMClient, get_llm
from app.observability import EventLog, ObservedRetriever, get_event_log, trace
from app.retrieval import PolicyRetriever, get_retriever


class GraphState(TypedDict, total=False):
    """The shared state. Every field is a typed contract, not a loose dict.

    Each node writes exactly one of `finding`, `risk`, `guardrail`, `response`,
    and appends to `audit`. Nothing rewrites what an earlier stage produced.
    """

    trace_id: str
    question: str
    staff_role: StaffRole
    finding: PolicyFinding | None
    risk: RiskAssessment | None
    guardrail: GuardrailDecision | None
    response: FinalResponse | None
    error: str | None
    audit: Annotated[list[AuditEvent], operator.add]


logger = logging.getLogger("governed_assistant.graph")

# Staff see a category, never the provider's words. A rate-limit message names the
# organisation, the model, the quota and a billing URL; none of that helps a member
# of staff decide what to do, and all of it is internal infrastructure detail.
_FAILURE_MESSAGES = {
    "rate_limited": "The assistant is temporarily over its request limit.",
    "timeout": "The assistant did not get a response in time.",
    "authentication": "The assistant could not authenticate to a service it depends on.",
    "unavailable": "The assistant could not reach a service it depends on.",
}


def classify_failure(exc: Exception) -> str:
    """Bucket a failure into a cause staff and operators can both act on.

    The classification is what travels: it is enough to diagnose a run from the
    audit trail without putting vendor, quota or account detail in it.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if "429" in text or "rate limit" in text or "quota" in text or "tokens per" in text:
        return "rate_limited"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "401" in text or "403" in text or "unauthor" in text or "api key" in text:
        return "authentication"
    return "unavailable"


def _request(state: GraphState) -> AskRequest:
    return AskRequest(question=state["question"], staff_role=state.get("staff_role", "branch_staff"))


def _audit(state: GraphState, stage: str, deterministic: bool, summary: str, **detail: Any) -> AuditEvent:
    return AuditEvent(
        trace_id=state["trace_id"],
        stage=stage,  # type: ignore[arg-type]
        deterministic=deterministic,
        summary=summary,
        detail=detail,
    )


def unavailable(state: GraphState, stage: str, exc: Exception) -> FinalResponse:
    """The fail-closed response. An incomplete run never reads as cleared.

    Deliberately says nothing about the provider. The full exception goes to the
    server log against this `trace_id`, where an operator can find it and a member
    of staff cannot.
    """
    cause = classify_failure(exc)
    logger.error(
        "stage %s failed (trace_id=%s, cause=%s): %s", stage, state["trace_id"], cause, exc
    )

    risk = RiskAssessment(
        status=RiskStatus.HUMAN_REVIEW_REQUIRED,
        reason=f"The {stage} stage failed, so the risk review did not complete.",
    )
    guardrail = GuardrailDecision(
        requires_human_review=True,
        rationale=f"Human review required: the {stage} stage did not complete.",
    )
    return FinalResponse(
        trace_id=state["trace_id"],
        status=ResponseStatus.UNAVAILABLE,
        answer=(
            "The assistant could not complete its checks, so it has no answer for you.\n\n"
            f"{_FAILURE_MESSAGES.get(cause, _FAILURE_MESSAGES['unavailable'])} "
            f"The {stage} stage did not complete.\n\n"
            "Do not treat this as a green light. Escalate the question instead. "
            f"Quote reference {state['trace_id'][:8]} if you report it."
        ),
        recommended_next_steps=[
            "Do not act on this request without a policy answer.",
            "Escalate the question to your team leader or the relevant policy owner.",
            f"Report the failure, quoting reference {state['trace_id'][:8]}.",
        ],
        policy_sources=[],
        risk_status=risk.status,
        human_review_required=True,
        risk=risk,
        guardrail=guardrail,
        model="deterministic",
    )


def _stage(name: str, log: EventLog):
    """Wrap a node with a lifecycle span and controlled failure handling.

    Observability lives here, at the wiring, rather than inside the agents.
    """

    def decorator(fn):
        def wrapped(state: GraphState) -> dict[str, Any]:
            if state.get("response") is not None:
                return {}  # an earlier stage already produced a terminal response
            try:
                with log.span(name, name):
                    return fn(state)
            except Exception as exc:  # noqa: BLE001 - the boundary is meant to be broad
                cause = classify_failure(exc)
                log.emit(name, "decision", status="failed", cause=cause)
                return {
                    # `error` stays in process state for the API layer and tests. It is
                    # never published: the response carries a category, the log a cause.
                    "error": f"{name}: {exc}",
                    "response": unavailable(state, name, exc),
                    "audit": [
                        _audit(state, name, deterministic=True, summary=f"{name} failed", cause=cause)
                    ],
                }

        wrapped.__name__ = fn.__name__
        return wrapped

    return decorator


def build_graph(
    retriever: PolicyRetriever | None = None,
    llm: LLMClient | None = None,
    event_log: EventLog | None = None,
) -> "Any":
    """Compile the workflow. Every dependency is injectable for testing."""
    log = event_log or get_event_log()
    retriever = ObservedRetriever(retriever or get_retriever(), log)
    llm = llm or get_llm()

    policy_agent = PolicyAgent(retriever=retriever, llm=llm)
    risk_agent = RiskAgent(llm=llm)
    response_agent = ResponseAgent(llm=llm)

    @_stage("policy_agent", log)
    def policy_node(state: GraphState) -> dict[str, Any]:
        finding = policy_agent.run(_request(state))
        log.emit(
            "policy_agent",
            "decision",
            status="completed" if finding.answerable else "abstained",
            cited_policy_ids=finding.cited_policy_ids,
            abstain_reason=finding.abstain_reason.value if finding.abstain_reason else None,
            procedure_count=len(finding.required_procedures),
        )
        return {
            "finding": finding,
            "audit": [
                _audit(
                    state,
                    "policy_agent",
                    deterministic=False,
                    summary="answerable" if finding.answerable else "abstained",
                    cited_policy_ids=finding.cited_policy_ids,
                    abstain_reason=finding.abstain_reason.value if finding.abstain_reason else None,
                )
            ],
        }

    @_stage("risk_agent", log)
    def risk_node(state: GraphState) -> dict[str, Any]:
        risk = risk_agent.review(_request(state), state["finding"])
        log.emit(
            "risk_agent",
            "decision",
            status=risk.status.value,
            categories=[c.value for c in risk.categories],
            risk_count=len(risk.identified_risks),
        )
        return {
            "risk": risk,
            "audit": [
                _audit(
                    state,
                    "risk_agent",
                    deterministic=False,
                    summary=risk.status.value,
                    categories=[c.value for c in risk.categories],
                )
            ],
        }

    @_stage("guardrail", log)
    def guardrail_node(state: GraphState) -> dict[str, Any]:
        decision = evaluate(state["finding"], state["risk"])
        log.emit(
            "guardrail",
            "decision",
            status=(
                "consequential action detected"
                if decision.detected_actions
                else "human review required"
                if decision.requires_human_review
                else "no action detected"
            ),
            deterministic=True,
            requires_human_review=decision.requires_human_review,
            triggered_rules=decision.triggered_rules,
            detected_actions=[a.value for a in decision.detected_actions],
        )
        return {
            "guardrail": decision,
            "audit": [
                _audit(
                    state,
                    "guardrail",
                    deterministic=True,
                    summary=(
                        "human review required"
                        if decision.requires_human_review
                        else "no review required"
                    ),
                    triggered_rules=decision.triggered_rules,
                    detected_actions=[a.value for a in decision.detected_actions],
                )
            ],
        }

    @_stage("response_agent", log)
    def response_node(state: GraphState) -> dict[str, Any]:
        response = response_agent.respond(
            _request(state), state["finding"], state["risk"], state["guardrail"]
        )
        response = response.model_copy(update={"trace_id": state["trace_id"]})
        log.emit(
            "response_agent",
            "decision",
            status=response.status.value,
            human_review_required=response.human_review_required,
            risk_status=response.risk_status.value,
            source_count=len(response.policy_sources),
        )
        return {
            "response": response,
            "audit": [
                _audit(
                    state,
                    "response_agent",
                    deterministic=False,
                    summary=response.status.value,
                    human_review_required=response.human_review_required,
                )
            ],
        }

    graph = StateGraph(GraphState)
    graph.add_node("policy_agent", policy_node)
    graph.add_node("risk_agent", risk_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("response_agent", response_node)

    graph.add_edge(START, "policy_agent")
    graph.add_edge("policy_agent", "risk_agent")
    graph.add_edge("risk_agent", "guardrail")
    graph.add_edge("guardrail", "response_agent")
    graph.add_edge("response_agent", END)

    return graph.compile()


class GovernedAssistant:
    """The application. One question in, one governed response out."""

    def __init__(
        self,
        retriever: PolicyRetriever | None = None,
        llm: LLMClient | None = None,
        event_log: EventLog | None = None,
    ) -> None:
        self.graph = build_graph(retriever=retriever, llm=llm, event_log=event_log)

    def ask(
        self, question: str, staff_role: StaffRole = "branch_staff", trace_id: str | None = None
    ) -> FinalResponse:
        """Run the workflow. Returns a response for every input, including failures."""
        state = self.run(question, staff_role=staff_role, trace_id=trace_id)
        response = state.get("response")
        if response is None:  # defensive: the graph always sets one
            return unavailable(state, "workflow", RuntimeError("no response was produced"))
        return response

    def run(
        self, question: str, staff_role: StaffRole = "branch_staff", trace_id: str | None = None
    ) -> GraphState:
        """Run the workflow and return the whole state, including the audit trail."""
        initial: GraphState = {
            "trace_id": trace_id or uuid4().hex,
            "question": question,
            "staff_role": staff_role,
            "finding": None,
            "risk": None,
            "guardrail": None,
            "response": None,
            "error": None,
            "audit": [],
        }
        with trace(initial["trace_id"]):
            return self.graph.invoke(initial)  # type: ignore[return-value]
