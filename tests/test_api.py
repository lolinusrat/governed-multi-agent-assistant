"""API tests.

The app is built with an injected assistant, so no request reaches a provider.
The focus is the boundary: what the API accepts, what it publishes, and what it
refuses to leak.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.agents.policy import PolicyDraft
from app.agents.response import ResponseDraft
from app.agents.risk import RiskDraft
from app.api import GENERIC_ERROR, create_app, redact
from app.config import Settings, get_settings
from app.contracts import ResponseStatus, RiskStatus
from app.graph import GovernedAssistant
from app.llm.stub import FailingLLMClient, StubLLMClient

COMPLAINT_QUESTION = "How quickly must I acknowledge a customer complaint?"
CLOSE_QUESTION = "A customer wants to close their account. What do I do?"
UNCOVERED_QUESTION = "What is the weather forecast in Sydney tomorrow?"

CLEAN_POLICY = PolicyDraft(
    answerable=True,
    proposed_guidance="Acknowledge the complaint to the customer within one business day.",
    required_procedures=["Record the complaint in the complaints system on the day it is received"],
    cited_policy_ids=["COMP-HAND-004"],
)
CLOSURE_POLICY = PolicyDraft(
    answerable=True,
    proposed_guidance="Closing an account associated with a fraud case requires the Financial Crime Duty Officer.",
    cited_policy_ids=["FRAUD-ESC-002"],
)
SAFE_RISK = RiskDraft(status=RiskStatus.SAFE, reason="supported")
DRAFTED = ResponseDraft(answer="Acknowledge the complaint within one business day.")


@pytest.fixture()
def client_factory(retriever):
    def _make(llm=None, raise_server_exceptions=True):
        assistant = GovernedAssistant(retriever=retriever, llm=llm or StubLLMClient())
        return TestClient(
            create_app(assistant=assistant), raise_server_exceptions=raise_server_exceptions
        )

    return _make


@pytest.fixture()
def client(client_factory):
    with client_factory(StubLLMClient([CLEAN_POLICY, SAFE_RISK, DRAFTED])) as c:
        yield c


class TestHealth:
    def test_reports_ok_and_the_loaded_corpus(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["policies_loaded"] == 4
        assert body["llm_provider"] in {"groq", "stub"}

    def test_health_never_publishes_a_credential(self, client_factory):
        app_client = client_factory()
        app_client.app.dependency_overrides[get_settings] = lambda: Settings(
            GROQ_API_KEY="gsk_supersecretvalue1234567890", GROQ_MODEL="llama-3.3-70b-versatile"
        )
        with app_client as c:
            payload = json.dumps(c.get("/health").json())
        assert "supersecret" not in payload
        assert "gsk_" not in payload
        assert "api_key" not in payload.lower()

    def test_health_is_the_only_get_route(self, client):
        assert client.get("/policies").status_code == 404


class TestAskContract:
    def test_returns_the_required_fields(self, client):
        body = client.post("/ask", json={"question": COMPLAINT_QUESTION}).json()
        for field in (
            "request_id",
            "answer",
            "policy_sources",
            "risk_status",
            "human_review_required",
        ):
            assert field in body, field

    def test_answers_a_supported_question(self, client):
        response = client.post("/ask", json={"question": COMPLAINT_QUESTION})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == ResponseStatus.ANSWERED.value
        assert body["human_review_required"] is False
        assert body["risk_status"] == RiskStatus.SAFE.value
        assert body["answer"].strip()
        assert body["recommended_next_steps"]

    def test_policy_sources_identify_the_policy_and_section(self, client):
        body = client.post("/ask", json={"question": COMPLAINT_QUESTION}).json()
        assert body["policy_sources"]
        for source in body["policy_sources"]:
            assert source["policy_id"] and source["title"]
            assert source["section"][0].isdigit()
            assert source["excerpt"].strip()

    def test_policy_sources_do_not_publish_internal_details(self, client):
        # Retrieval scores and on-disk paths are internal.
        body = client.post("/ask", json={"question": COMPLAINT_QUESTION}).json()
        for source in body["policy_sources"]:
            assert set(source) == {"policy_id", "title", "section", "excerpt"}

    def test_the_request_id_is_returned_and_unique(self, client):
        first = client.post("/ask", json={"question": COMPLAINT_QUESTION}).json()["request_id"]
        assert first
        # A fresh script is needed per call; a second request re-uses the stub's
        # defaults, which is enough to check the id changes.
        second = client.post("/ask", json={"question": COMPLAINT_QUESTION}).json()["request_id"]
        assert first != second

    def test_staff_role_is_accepted(self, client):
        response = client.post(
            "/ask", json={"question": COMPLAINT_QUESTION, "staff_role": "contact_centre"}
        )
        assert response.status_code == 200


class TestGovernanceIsCarriedThrough:
    def test_a_consequential_action_is_published_as_requiring_review(self, client_factory):
        with client_factory(StubLLMClient([CLOSURE_POLICY, SAFE_RISK, DRAFTED])) as c:
            body = c.post("/ask", json={"question": CLOSE_QUESTION}).json()
        assert body["human_review_required"] is True
        assert body["status"] == ResponseStatus.PENDING_HUMAN_REVIEW.value
        assert body["risk_status"] == RiskStatus.SAFE.value  # the model said safe; it did not matter

    def test_an_uncovered_question_abstains_with_no_sources(self, client):
        body = client.post("/ask", json={"question": UNCOVERED_QUESTION}).json()
        assert body["status"] == ResponseStatus.ABSTAINED.value
        assert body["policy_sources"] == []
        assert body["abstain_reason"] == "NO_RELEVANT_POLICY"
        assert any("Escalate" in step for step in body["recommended_next_steps"])


class TestInputValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "  "},
            {"question": "hi"},
            {"question": "x" * 1001},
            {"question": 42},
            {"question": None},
            {"question": COMPLAINT_QUESTION, "staff_role": "ceo"},
            {"question": COMPLAINT_QUESTION, "override_guardrail": True},
        ],
    )
    def test_bad_input_is_rejected_without_running_the_workflow(self, client, payload):
        response = client.post("/ask", json=payload)
        assert response.status_code == 422

    def test_validation_errors_name_the_field_and_stay_generic(self, client):
        body = client.post("/ask", json={"question": "hi"}).json()
        assert body["error"] == "The request was not valid."
        assert body["request_id"]
        assert body["detail"][0]["field"] == "question"
        assert "Traceback" not in json.dumps(body)

    def test_an_unknown_field_cannot_smuggle_in_an_override(self, client):
        response = client.post(
            "/ask", json={"question": COMPLAINT_QUESTION, "human_review_required": False}
        )
        assert response.status_code == 422

    def test_a_malformed_body_is_rejected(self, client):
        response = client.post(
            "/ask", content=b"not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestControlledFailure:
    def test_a_provider_outage_returns_503_with_the_governed_body(self, client_factory):
        with client_factory(FailingLLMClient("groq timed out")) as c:
            response = c.post("/ask", json={"question": COMPLAINT_QUESTION})
        assert response.status_code == 503
        body = response.json()
        # The caller must be able to read "human review required", not just a code.
        assert body["status"] == ResponseStatus.UNAVAILABLE.value
        assert body["human_review_required"] is True
        assert body["policy_sources"] == []
        assert "green light" in body["answer"]

    def test_an_unexpected_error_becomes_a_generic_500(self, client_factory):
        class Exploding(GovernedAssistant):
            def __init__(self):  # noqa: D107 - deliberately not calling super
                pass

            def ask(self, *args, **kwargs):
                raise RuntimeError("secret internal detail at /Users/nusrat/private")

        client = TestClient(create_app(assistant=Exploding()), raise_server_exceptions=False)
        with client as c:
            response = c.post("/ask", json={"question": COMPLAINT_QUESTION})
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == GENERIC_ERROR
        assert body["request_id"]
        assert "secret internal detail" not in json.dumps(body)
        assert "/Users/nusrat" not in json.dumps(body)

    def test_no_response_body_ever_contains_a_traceback(self, client_factory):
        with client_factory(FailingLLMClient("boom")) as c:
            payloads = [
                c.post("/ask", json={"question": COMPLAINT_QUESTION}).text,
                c.post("/ask", json={"question": "hi"}).text,
                c.get("/health").text,
            ]
        for payload in payloads:
            assert "Traceback" not in payload
            assert "File \"" not in payload
            assert ".py\", line" not in payload


class TestSecretHygiene:
    def test_redact_removes_the_configured_key(self):
        settings = Settings(GROQ_API_KEY="gsk_abcdefghijklmnopqrstuvwxyz012345")
        cleaned = redact("failed with key gsk_abcdefghijklmnopqrstuvwxyz012345 attached", settings)
        assert "gsk_abcdef" not in cleaned
        assert "[redacted]" in cleaned

    def test_redact_removes_key_shaped_strings_it_was_never_told_about(self):
        settings = Settings(GROQ_API_KEY="")
        cleaned = redact("leaked sk_live_0123456789abcdefghij and gsk_zzzzzzzzzzzzzzzzzzzz", settings)
        assert "sk_live_0123456789abcdefghij" not in cleaned
        assert "gsk_zzzzzzzzzzzzzzzzzzzz" not in cleaned

    def test_an_error_carrying_a_key_is_scrubbed_before_it_is_published(self, client_factory):
        # Belt and braces. The provider message no longer reaches the caller at
        # all, and redaction stands behind that in case it ever does again.
        key = "gsk_leakedkeyvalue0123456789abcd"
        client = client_factory(FailingLLMClient(f"401 unauthorized for key {key}"))
        client.app.dependency_overrides[get_settings] = lambda: Settings(GROQ_API_KEY=key)
        with client as c:
            payload = c.post("/ask", json={"question": COMPLAINT_QUESTION}).text
        assert key not in payload
        assert "401 unauthorized" not in payload

    def test_no_endpoint_publishes_the_environment(self, client):
        for path in ("/health", "/openapi.json"):
            payload = client.get(path).text.lower()
            assert "groq_api_key" not in payload
            assert "gsk_" not in payload
