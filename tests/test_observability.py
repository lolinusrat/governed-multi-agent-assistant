"""Observability tests.

Two things are being checked: that the trail is complete enough to reconstruct a
request, and that it holds nothing it should not - no question text, no answer
text, no evidence excerpts, no credentials.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.policy import PolicyDraft
from app.agents.response import ResponseDraft
from app.agents.risk import RiskDraft
from app.api import create_app
from app.contracts import RiskStatus
from app.graph import GovernedAssistant
from app.llm.base import LLMError
from app.llm.stub import FailingLLMClient, StubLLMClient
from app.observability import (
    EventLog,
    ObservedRetriever,
    current_trace_id,
    digest,
    new_trace_id,
    read_events,
    redact,
    render_trace,
    trace,
)
from app.retrieval import PolicyRetriever

CLOSE_QUESTION = "A customer wants to close their account. What do I do?"
# Carries a name on purpose: used to prove the trail never holds question text.
SENSITIVE_QUESTION = "Customer Jane Roe disputes a card transaction. What do I do?"
UNCOVERED_QUESTION = "What is the weather forecast in Sydney tomorrow?"

CLOSURE_POLICY = PolicyDraft(
    answerable=True,
    proposed_guidance="Closing an account associated with a fraud case requires the Financial Crime Duty Officer.",
    required_procedures=["Record a fraud case in the case management system"],
    cited_policy_ids=["FRAUD-ESC-002"],
)
REVIEW_RISK = RiskDraft(status=RiskStatus.HUMAN_REVIEW_REQUIRED, reason="consequential action")
DRAFTED = ResponseDraft(answer="Escalate to the Financial Crime Duty Officer before acting.")


@pytest.fixture()
def log() -> EventLog:
    return EventLog()


@pytest.fixture()
def run_request(retriever):
    """Drive one full HTTP request and hand back the events it produced."""

    def _run(question: str, llm=None, log: EventLog | None = None):
        log = log or EventLog()
        assistant = GovernedAssistant(retriever, llm or StubLLMClient(), event_log=log)
        with TestClient(create_app(assistant, event_log=log)) as client:
            response = client.post("/ask", json={"question": question})
        return response, log

    return _run


class TestRecordShape:
    def test_every_record_carries_the_required_fields(self, log):
        record = log.emit("policy_agent", "decision", status="completed", trace_id="t-1")
        assert set(record) >= {"timestamp", "trace_id", "component", "event", "status"}
        assert record["trace_id"] == "t-1"

    def test_the_timestamp_is_timezone_aware_utc(self, log):
        record = log.emit("api", "request_received", trace_id="t-1")
        assert record["timestamp"].endswith("+00:00")

    def test_latency_is_recorded_where_it_is_relevant(self, log):
        with log.span("risk_agent", "risk_agent"):
            pass
        completed = log.records[-1]
        assert completed["status"] == "completed"
        assert isinstance(completed["latency_ms"], float)
        assert "latency_ms" not in log.records[0]  # the start event has no duration yet

    def test_extra_fields_are_carried_through(self, log):
        record = log.emit("guardrail", "decision", status="x", triggered_rules=["close_account"])
        assert record["triggered_rules"] == ["close_account"]


class TestTraceIdentifier:
    def test_ids_are_unique(self):
        assert new_trace_id() != new_trace_id()

    def test_the_bound_id_is_used_when_none_is_passed(self, log):
        with trace("bound-id"):
            assert current_trace_id() == "bound-id"
            record = log.emit("api", "request_received")
        assert record["trace_id"] == "bound-id"

    def test_an_unbound_event_is_marked_rather_than_dropped(self, log):
        assert log.emit("api", "x")["trace_id"] == "unbound"

    def test_binding_is_restored_afterwards(self):
        with trace("outer"):
            with trace("inner"):
                assert current_trace_id() == "inner"
            assert current_trace_id() == "outer"
        assert current_trace_id() is None


class TestSpans:
    def test_a_span_emits_start_then_completion(self, log):
        with log.span("policy_agent", "policy_agent"):
            pass
        assert [r["status"] for r in log.records] == ["started", "completed"]

    def test_a_failing_span_records_the_error_and_re_raises(self, log):
        with pytest.raises(LLMError):
            with log.span("risk_agent", "risk_agent"):
                raise LLMError("provider down")
        failed = log.records[-1]
        assert failed["status"] == "failed"
        assert "provider down" in failed["error"]
        assert "latency_ms" in failed


class TestJsonlStorage:
    def test_events_are_appended_as_one_json_object_per_line(self, tmp_path):
        path = tmp_path / "nested" / "events.jsonl"
        log = EventLog(path)
        log.emit("api", "request_received", trace_id="t-1")
        log.emit("api", "response_returned", trace_id="t-1", http_status=200)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["trace_id"] == "t-1" for line in lines)

    def test_events_can_be_read_back_and_filtered_by_trace(self, tmp_path):
        path = tmp_path / "events.jsonl"
        log = EventLog(path)
        log.emit("api", "request_received", trace_id="t-1")
        log.emit("api", "request_received", trace_id="t-2")

        assert len(read_events(path)) == 2
        assert len(read_events(path, trace_id="t-2")) == 1

    def test_reading_a_missing_file_is_not_an_error(self, tmp_path):
        assert read_events(tmp_path / "absent.jsonl") == []

    def test_a_corrupt_line_does_not_break_the_reader(self, tmp_path):
        path = tmp_path / "events.jsonl"
        EventLog(path).emit("api", "request_received", trace_id="t-1")
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        assert len(read_events(path)) == 1

    def test_a_disabled_log_writes_nothing(self, tmp_path):
        path = tmp_path / "events.jsonl"
        log = EventLog(path, enabled=False)
        log.emit("api", "request_received", trace_id="t-1")
        assert not path.exists()
        assert log.records == []

    def test_an_unwritable_path_does_not_break_the_request(self, tmp_path):
        # Observability failing must never take the assistant down.
        path = tmp_path / "events.jsonl"
        log = EventLog(path)
        path.unlink(missing_ok=True)
        path.mkdir()  # now a directory, so opening it for append fails
        log.emit("api", "request_received", trace_id="t-1")
        assert log.records  # still recorded in memory, no exception raised


class TestSensitiveData:
    def test_a_configured_key_is_scrubbed(self):
        key = "gsk_abcdefghijklmnopqrstuvwxyz012345"
        log = EventLog(api_key=key)
        record = log.emit("risk_agent", "decision", status="failed", error=f"401 for {key}")
        assert key not in json.dumps(record)
        assert "[redacted]" in record["error"]

    def test_key_shaped_strings_are_scrubbed_even_when_unknown(self, log):
        record = log.emit("api", "x", error="leaked sk_live_0123456789abcdefghij here")
        assert "sk_live_0123456789abcdefghij" not in json.dumps(record)

    def test_scrubbing_reaches_nested_values(self, log):
        record = log.emit("api", "x", detail={"messages": ["token gsk_zzzzzzzzzzzzzzzzzzzzzz"]})
        assert "gsk_zzzz" not in json.dumps(record)

    def test_the_question_text_is_never_written(self, run_request):
        _, log = run_request(
            SENSITIVE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED])
        )
        payload = json.dumps(log.records)
        assert "Jane Roe" not in payload
        assert SENSITIVE_QUESTION not in payload

    def test_a_digest_and_length_are_kept_instead(self, run_request):
        _, log = run_request(
            SENSITIVE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED])
        )
        accepted = next(r for r in log.records if r["event"] == "question_accepted")
        assert accepted["question_digest"] == digest(SENSITIVE_QUESTION)
        assert accepted["question_chars"] == len(SENSITIVE_QUESTION)

    def test_answer_and_evidence_text_are_never_written(self, run_request):
        response, log = run_request(
            CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED])
        )
        payload = json.dumps(log.records)
        assert response.json()["answer"][:60] not in payload
        for source in response.json()["policy_sources"]:
            assert source["excerpt"][:60] not in payload

    def test_redact_leaves_ordinary_text_alone(self):
        assert redact("Close the account after approval.") == "Close the account after approval."


class TestRetrievalEvents:
    def test_the_wrapper_satisfies_the_retriever_protocol(self, retriever, log):
        assert isinstance(ObservedRetriever(retriever, log), PolicyRetriever)

    def test_the_result_passes_through_unchanged(self, retriever, log):
        question = "How quickly must I acknowledge a customer complaint?"
        assert (
            ObservedRetriever(retriever, log).search(question).model_dump()
            == retriever.search(question).model_dump()
        )

    def test_retrieval_records_its_outcome_without_the_query_text(self, retriever, log):
        ObservedRetriever(retriever, log).search("How quickly must I acknowledge a complaint?")
        event = next(r for r in log.records if r["event"] == "policy_retrieval")
        assert event["status"] == "sufficient"
        assert "latency_ms" in event
        assert event["evidence_count"] > 0
        assert "How quickly" not in json.dumps(log.records)

    def test_each_retrieved_policy_id_and_section_is_recorded(self, retriever, log):
        result = ObservedRetriever(retriever, log).search("fraud escalation tiers")
        selected = [r for r in log.records if r["event"] == "evidence_selected"]
        assert len(selected) == len(result.evidence)
        assert [r["policy_id"] for r in selected] == [e.policy_id for e in result.evidence]
        assert all(r["section"] for r in selected)

    def test_an_insufficient_search_is_recorded_with_its_reason(self, retriever, log):
        ObservedRetriever(retriever, log).search(UNCOVERED_QUESTION)
        event = next(r for r in log.records if r["event"] == "policy_retrieval")
        assert event["status"] == "insufficient"
        assert event["abstain_reason"] == "NO_RELEVANT_POLICY"


class TestEndToEndTrail:
    def test_a_full_request_records_every_required_event(self, run_request):
        _, log = run_request(CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED]))
        seen = {(r["component"], r["event"]) for r in log.records}
        for expected in [
            ("api", "request_received"),
            ("retrieval", "policy_retrieval"),
            ("retrieval", "evidence_selected"),
            ("policy_agent", "policy_agent"),
            ("policy_agent", "decision"),
            ("risk_agent", "risk_agent"),
            ("risk_agent", "decision"),
            ("guardrail", "decision"),
            ("response_agent", "response_agent"),
            ("response_agent", "decision"),
            ("api", "response_returned"),
        ]:
            assert expected in seen, expected

    def test_every_stage_has_a_start_and_an_end(self, run_request):
        _, log = run_request(CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED]))
        for stage in ("policy_agent", "risk_agent", "guardrail", "response_agent"):
            states = [r["status"] for r in log.records if r["event"] == stage]
            assert states == ["started", "completed"], stage

    def test_one_trace_id_ties_the_whole_request_together(self, run_request):
        response, log = run_request(
            CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED])
        )
        ids = {r["trace_id"] for r in log.records}
        assert len(ids) == 1
        assert ids.pop() == response.json()["request_id"] == response.headers["X-Request-ID"]

    def test_the_decisions_are_recorded_with_their_outcomes(self, run_request):
        _, log = run_request(CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED]))
        decisions = {r["component"]: r for r in log.records if r["event"] == "decision"}
        assert decisions["risk_agent"]["status"] == "HUMAN_REVIEW_REQUIRED"
        assert decisions["guardrail"]["status"] == "consequential action detected"
        assert decisions["guardrail"]["requires_human_review"] is True
        assert decisions["guardrail"]["deterministic"] is True
        assert decisions["response_agent"]["status"] == "PENDING_HUMAN_REVIEW"

    def test_an_abstention_is_traceable_too(self, run_request):
        _, log = run_request(UNCOVERED_QUESTION)
        decisions = {r["component"]: r for r in log.records if r["event"] == "decision"}
        assert decisions["policy_agent"]["status"] == "abstained"
        assert decisions["policy_agent"]["abstain_reason"] == "NO_RELEVANT_POLICY"
        assert decisions["response_agent"]["status"] == "ABSTAINED"

    def test_a_failure_is_recorded_at_the_stage_that_failed(self, run_request):
        response, log = run_request(CLOSE_QUESTION, FailingLLMClient("groq timed out"))
        assert response.status_code == 503
        failed = [r for r in log.records if r["status"] == "failed"]
        assert failed and all(r["component"] == "policy_agent" for r in failed)
        assert any("groq timed out" in r.get("error", "") for r in failed)
        assert any(r["event"] == "response_returned" for r in log.records)

    def test_an_invalid_request_is_still_traced(self, retriever):
        # The identifier is minted in middleware, so a request that never reaches
        # a handler is still accounted for.
        log = EventLog()
        assistant = GovernedAssistant(retriever, StubLLMClient(), event_log=log)
        with TestClient(create_app(assistant, event_log=log)) as client:
            response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 422
        returned = next(r for r in log.records if r["event"] == "response_returned")
        assert returned["status"] == "error"
        assert returned["http_status"] == 422
        assert returned["trace_id"] == response.json()["request_id"]


class TestTraceRendering:
    def test_a_trace_renders_as_a_readable_tree(self, run_request):
        _, log = run_request(CLOSE_QUESTION, StubLLMClient([CLOSURE_POLICY, REVIEW_RISK, DRAFTED]))
        rendered = render_trace(log.records).splitlines()
        assert rendered[1] == "├── request received"
        assert rendered[-1] == "└── response returned"
        assert any(line.startswith("├── retrieval → ") and "§" in line for line in rendered)
        assert "├── risk_agent → HUMAN_REVIEW_REQUIRED" in rendered
        assert "├── guardrail → consequential action detected" in rendered

    def test_an_empty_trail_renders_without_failing(self):
        assert render_trace([]) == "(no events)"


class TestSeparationFromBusinessLogic:
    @pytest.mark.parametrize(
        "module",
        [
            "app/contracts.py",
            "app/retrieval.py",
            "app/guardrail.py",
            "app/agents/policy.py",
            "app/agents/risk.py",
            "app/agents/response.py",
        ],
    )
    def test_the_decision_code_does_not_import_observability(self, module):
        # Instrumentation is attached at the wiring, not scattered through the
        # logic. These modules stay readable and testable without it.
        tree = ast.parse(Path(module).read_text(encoding="utf-8"))
        imported = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert "app.observability" not in imported
