"""Streamlit UI for the Governed Banking Policy Assistant.

Talks to the FastAPI service over HTTP and does nothing else. It holds no policy
logic, no retrieval, and no governance decisions of its own - it renders what the
API returned. That keeps the governed path behind one auditable surface: anything
this page shows, an auditor can reproduce by calling the same endpoint.

Run it alongside the API:

    uv run uvicorn app.api:app --reload
    uv run streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 60.0

EXAMPLE_QUESTIONS = [
    "A customer doesn't recognise a card transaction. What should I do?",
    "Can I immediately block the customer's account?",
    "Can I guarantee the customer will get their money back?",
]

STAFF_ROLES = ["branch_staff", "contact_centre", "operations"]

STATUS_HELP = {
    "ANSWERED": ("success", "Answered from policy."),
    "PENDING_HUMAN_REVIEW": ("warning", "A person must review this before you act."),
    "ABSTAINED": ("info", "The policies do not cover this. Escalate it."),
    "REJECTED": ("error", "The drafted guidance was withheld by the risk review."),
    "UNAVAILABLE": ("error", "The assistant could not complete its checks."),
}

# Statuses where no guidance was ever drafted. The risk status still exists on the
# wire, but reporting it here would answer a question nobody asked: "SAFE" next to
# "Abstained" reads as though the system judged the situation safe, when it means
# there was no guidance to find fault with.
NO_GUIDANCE_ISSUED = {"ABSTAINED", "UNAVAILABLE"}

# Presentation only. The wire contract keeps the enum values; these are what a
# member of staff reads. `st.metric` renders its value on one line and truncates
# rather than wrapping, so these are rendered as plain text instead.
STATUS_LABELS = {
    "ANSWERED": "Answered",
    "PENDING_HUMAN_REVIEW": "Pending Human Review",
    "ABSTAINED": "Abstained",
    "REJECTED": "Rejected",
    "UNAVAILABLE": "Unavailable",
}

RISK_LABELS = {
    "SAFE": "Safe",
    "HUMAN_REVIEW_REQUIRED": "Human Review Required",
    "REJECTED": "Rejected",
}

ABSTAIN_REASONS = {
    "NO_RELEVANT_POLICY": "No relevant policy found",
    "INSUFFICIENT_EVIDENCE": "Insufficient policy evidence",
    "OUT_OF_POLICY_SCOPE": "The relevant policy states this topic is not covered",
    "UNVERIFIABLE_CITATION": "The drafted guidance could not be traced to a policy source",
}


# --------------------------------------------------------------------------- #
# API client - the only way this page gets data
# --------------------------------------------------------------------------- #


def ask_api(question: str, staff_role: str) -> tuple[dict | None, str | None]:
    """POST /ask. Returns (body, error_message)."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "staff_role": staff_role},
            timeout=TIMEOUT,
        )
    except httpx.RequestError as exc:
        return None, f"Could not reach the API at {API_BASE_URL} ({exc.__class__.__name__})."

    if response.status_code == 422:
        detail = response.json().get("detail") or []
        messages = "; ".join(f"{d['field']}: {d['message']}" for d in detail)
        return None, f"The question was not accepted. {messages}"
    if response.status_code >= 500 and response.status_code != 503:
        return None, response.json().get("error", "The assistant could not process this request.")

    # 503 still carries the full governed body, and it must be shown.
    return response.json(), None


def fetch_trace(request_id: str) -> list[dict]:
    try:
        response = httpx.get(f"{API_BASE_URL}/trace/{request_id}", timeout=TIMEOUT)
    except httpx.RequestError:
        return []
    return response.json().get("events", []) if response.status_code == 200 else []


def api_health() -> dict | None:
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
    except httpx.RequestError:
        return None
    return response.json() if response.status_code == 200 else None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _stat(column, label: str, value: str) -> None:
    """A label/value pair that wraps instead of truncating."""
    column.caption(label)
    column.markdown(f"**{value}**")


def render_result(question: str, body: dict) -> None:
    st.divider()
    st.caption("Question")
    st.write(question)

    status = body.get("status", "UNAVAILABLE")
    review = bool(body.get("human_review_required"))
    guidance_issued = status not in NO_GUIDANCE_ISSUED
    reason = ABSTAIN_REASONS.get(body.get("abstain_reason") or "", "")

    risk_status = body.get("risk_status", "")
    outcome, risk_col, review_col = st.columns(3)
    _stat(outcome, "Outcome", STATUS_LABELS.get(status, status))
    _stat(
        risk_col,
        "Guidance risk",
        RISK_LABELS.get(risk_status, risk_status or "—") if guidance_issued else "N/A",
    )
    _stat(review_col, "Human review", "Required" if review else "Not required")

    if not guidance_issued:
        st.caption(
            "Guidance risk is not applicable here — no guidance was issued, so there was "
            "nothing for the risk review to assess."
        )

    if review:
        st.warning("**Human review required.** Do not act on this until a person has approved it.")

    level, note = STATUS_HELP.get(status, ("info", status))
    headline = STATUS_LABELS.get(status, status)
    detail = f"{reason}." if reason else note
    getattr(st, level)(f"**{headline}** — {detail}")

    st.caption("Answer")
    st.write(body.get("answer", ""))

    steps = body.get("recommended_next_steps") or []
    if steps:
        st.caption("Recommended next steps")
        for i, step in enumerate(steps, start=1):
            st.write(f"{i}. {step}")

    render_sources(body.get("policy_sources") or [])
    render_trace(body.get("request_id", ""))


def render_sources(sources: list[dict]) -> None:
    st.caption(f"Policy sources ({len(sources)})")
    if not sources:
        st.write("No policy source supports an answer to this question.")
        return
    for source in sources:
        with st.expander(f"{source['policy_id']} — {source['section']}"):
            st.caption(source["title"])
            st.text(source["excerpt"])


def render_trace(request_id: str) -> None:
    if not request_id:
        return
    with st.expander("Execution trace"):
        st.caption(f"request_id: {request_id}")
        events = fetch_trace(request_id)
        if not events:
            st.write("No trace was recorded for this request.")
            return
        # Every column is a string: a mixed float/blank column cannot be
        # serialised for display.
        rows = [
            {
                "component": str(e.get("component", "")),
                "event": str(e.get("event", "")),
                "status": str(e.get("status", "")),
                "latency_ms": f"{e['latency_ms']:.1f}" if e.get("latency_ms") is not None else "",
            }
            for e in events
        ]
        st.dataframe(rows, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="Governed Banking Policy Assistant", layout="centered")
    st.title("Governed Banking Policy Assistant")
    st.caption(
        "Internal staff tool. Answers come from synthetic banking policies for a fictional "
        "institution and are advisory only — anything flagged for human review must be "
        "reviewed before you act."
    )

    if "question" not in st.session_state:
        st.session_state.question = ""

    with st.sidebar:
        st.subheader("Service")
        health = api_health()
        if health is None:
            st.error(f"API unreachable at {API_BASE_URL}")
        else:
            st.success(f"{health['status']} · {health['policies_loaded']} policies")
            st.caption(f"{health['llm_provider']} · {health['model']}")
        st.caption(API_BASE_URL)

    st.subheader("Example questions")
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{i}", width="stretch"):
            st.session_state.question = example

    question = st.text_area(
        "Your question", key="question", height=100, placeholder="Ask about a bank policy…"
    )
    staff_role = st.selectbox("Your role", STAFF_ROLES, index=0)
    submitted = st.button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.error("Enter a question first.")
            return
        with st.spinner("Checking policy…"):
            body, error = ask_api(question.strip(), staff_role)
        if error:
            st.error(error)
            return
        render_result(question.strip(), body)


if __name__ == "__main__":
    main()
