# Governed Banking Policy Assistant

An internal, staff-facing multi-agent assistant that answers operational questions
using a **synthetic** banking policy corpus. Built as a thin vertical slice under a
three-hour timebox, with the governance controls implemented deterministically
rather than left to the language model.

> **Status: in progress.** Scaffolding, contracts, the synthetic policy corpus,
> local retrieval, the LLM abstraction, all three agents, the deterministic guardrail and
> the LangGraph workflow are in place and tested. The API and the UI are not implemented
> yet — see [Implementation sequence](#implementation-sequence).

---

## Why this exists

Staff need answers grounded in policy, not plausible-sounding prose. Two behaviours
matter more than answer quality:

1. **Abstain rather than hallucinate** when the corpus does not support the question.
2. **Route consequential banking actions to a human** before staff act on them.

Both are enforced by deterministic code, so they can be unit-tested and explained to
a reviewer without reference to model behaviour.

## Architecture

One API process, one Streamlit process, one linear LangGraph with five nodes.
No database, no vector store, no queue, no container runtime.

```
Streamlit UI ──HTTP──▶ FastAPI  POST /ask
                            │
                            ▼
                LangGraph — linear, 5 nodes

  1. retrieve         deterministic, no LLM
                      keyword scoring over the synthetic policy corpus

  2. policy_agent     LLM  — evidence → PolicyFindings
                      (required procedures, proposed guidance)

  3. risk_agent       LLM  — independent review of the proposed guidance;
                      never sees the policy agent's rationale → RiskAssessment

  4. guardrail        deterministic, no LLM
                      rule table → requires_human_review

  5. response_agent   LLM  — grounded staff-facing answer + citations
                            │
                            ▼
                  AssistantResponse (Pydantic)
```

### Deterministic controls

| Control | Mechanism |
|---|---|
| Abstain rather than hallucinate | Top retrieval score below `RETRIEVAL_MIN_SCORE` short-circuits to abstain **before any LLM call**. The policy agent's own `answerable=false` is also honoured. |
| Human review for consequential actions | A rule table is matched against the proposed guidance and procedures. In this demonstration the consequential actions are transferring funds, approving credit, closing an account and blocking an account. The rules are data, they are unit-tested, and no risk status can clear a match. |
| Risk escalation | The Risk Agent returns one of three statuses. `REJECTED` blocks the answer; `HUMAN_REVIEW_REQUIRED` forces review. Its deterministic rule pass can raise a status the model set, never lower it. |
| Grounding | The response agent sees only retrieved excerpts. Citations are validated against known policy ids after generation; a fabricated id downgrades the result to an abstention. |
| Failure | If any stage raises, the workflow stops and returns status `UNAVAILABLE` with `human_review_required = True`. Checks that did not complete never read as checks that passed. |

### Why the important controls are deterministic

The agents are language models. They are useful because they generalise, and that
is exactly why they cannot be the control: the same prompt can produce a different
answer tomorrow, a persuasive question can talk one into agreeing, and nothing in
the mechanism guarantees it will refuse when it should. Good behaviour most of the
time is not a control.

So the decisions that carry consequences are made by code instead:

- **Abstention** is a retrieval score against a threshold, applied before the model
  is called at all.
- **Grounding** is set intersection: a policy id the retriever did not return is
  discarded, whatever the model claims.
- **Human review** is a rule table matched against the proposed guidance. The
  Risk Agent's status is an input to that decision and can only ever add a reason
  for review; it cannot remove one.

This buys three things worth more here than accuracy:

1. **It can be tested.** `requires_human_review` is asserted for every consequential
   action phrasing crossed with every risk status the model can return — a claim
   about all inputs, not a sample of them.
2. **It can be explained.** Every escalation names the rule that fired and the
   approval authority from the cited policy. "The model thought it was risky" is not
   an audit trail.
3. **It can be changed deliberately.** Adding a consequential action is an edit to an
   enum and a rule table, reviewable in a diff, not a change in prompt wording whose
   effect nobody can predict.

[`app/guardrail.py`](app/guardrail.py) imports nothing from `app/llm/`, and a test
parses the module to prove it. The guardrail cannot consult a model even by accident.

The cost is false positives: the rule table over-triggers, and it deliberately does
no negation handling, so guidance saying "do not close the account" still escalates.
In a regulated context that is the correct direction — over-escalation costs a
reviewer a minute, under-escalation costs a customer.

## Repository layout

```
.
├── app/
│   ├── config.py          # settings from environment (step 1)
│   ├── contracts.py       # all Pydantic contracts between agents (step 1)
│   ├── retrieval.py       # policy parsing, keyword scoring, abstention (step 2)
│   ├── guardrail.py       # deterministic consequential-action rules (step 3)
│   ├── graph.py           # LangGraph wiring, GraphState, error handling (step 6)
│   ├── api.py             # POST /ask, GET /health (step 7)
│   ├── llm/               # provider abstraction — base, groq_client, stub (step 4)
│   └── agents/            # policy, risk, response (step 5, done)
├── data/                  # synthetic policy corpus (4 documents, done)
├── ui/                    # Streamlit app (step 8)
└── tests/                 # pytest suite, runs offline against the stub provider
```

## Configuration

Copy the example file and fill in your key. `.env` is git-ignored and must never be
committed.

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key. Read only in `app/config.py`. |
| `GROQ_MODEL` | Groq model id, e.g. `llama-3.3-70b-versatile`. |
| `LLM_PROVIDER` | `groq` for real calls, `stub` for deterministic offline runs. |
| `RETRIEVAL_MIN_SCORE` | Abstention threshold for retrieval. |
| `POLICY_DIR` | Location of the synthetic policy corpus (`data/`). |
| `API_BASE_URL` | Base URL the Streamlit UI uses to reach the API. |

### Provider abstraction

The application depends on a single protocol:

```python
class LLMClient(Protocol):
    def structured(self, *, system: str, user: str, schema: type[T]) -> T: ...
```

`GroqClient` and `StubLLMClient` implement it. Nothing outside `app/llm/` imports a
vendor SDK, so swapping providers means adding one file and changing one factory.
The stub keeps the whole test suite offline and free of API keys.

## Getting started

```bash
uv sync                       # create .venv and install from uv.lock
cp .env.example .env          # then add your GROQ_API_KEY
uv run pytest                 # test suite (offline, uses the stub provider)
```

Once the API and UI land (steps 7 and 8), two processes are run side by side:

```bash
uv run uvicorn app.api:app --reload        # API on :8000
uv run streamlit run ui/streamlit_app.py   # UI  on :8501
```

## Implementation sequence

| # | Step | Status |
|---|---|---|
| 0 | Skeleton: `pyproject.toml`, `.gitignore`, `.env.example`, README, folders | done |
| 1 | Contracts and configuration | done |
| 2 | Synthetic policy corpus and retrieval | done |
| 3 | Deterministic guardrail and its tests | done |
| 4 | LLM abstraction: protocol, Groq client, stub | done |
| 5 | Policy, Risk and Response agents | done |
| 6 | LangGraph wiring and end-to-end test | done |
| 7 | FastAPI endpoints | pending |
| 8 | Streamlit UI | pending |
| 9 | Documentation pass | pending |

The guardrail and abstention logic are built before the agents deliberately. If the
timebox runs out, the governance layer is complete and tested, and the model layer is
what degrades.

## Scope and limitations

Declared rather than half-built, given the timebox:

- **Synthetic policies only.** The four documents in `data/` describe a fictional
  institution, Meridian Retail Bank. They do not reproduce any real bank's policy.
  No real customer data, and none should ever be entered.
- **Keyword retrieval, not embeddings.** Deterministic and testable, but weak on
  synonyms. Mitigated by a small curated corpus plus the explicit abstain path.
  First thing to replace after the slice.
- **Risk agent independence is prompt-level, not process-level.** A separate call with
  a separate system prompt and a restricted view — not a separate model or provider.
- **Audit trail is a `trace_id` and a structured log line**, not a persisted store. The
  full decision record travels in the response, so persistence is one insert away.
- **No authentication, rate limiting or PII scrubbing.** Out of scope for an internal
  slice over synthetic data.
- **Advisory only.** Output supports a member of staff making a decision; it is not an
  approval, and anything flagged for human review must be reviewed before staff act.
