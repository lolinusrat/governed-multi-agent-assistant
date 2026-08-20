# Governed Banking Policy Assistant

An internal, staff-facing multi-agent assistant that answers operational questions
using a **synthetic** banking policy corpus. Built as a thin vertical slice under a
three-hour timebox, with the governance controls implemented deterministically
rather than left to the language model.

> **Status: skeleton only.** Project scaffolding, dependencies and configuration are
> in place. The agents, retrieval, guardrail, API and UI are not implemented yet —
> see [Implementation sequence](#implementation-sequence).

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
| Human review for consequential actions | A rule table is matched against the question and the proposed guidance (fee waiver, account closure, limit override, hold release, card unblock, customer-data disclosure). The rules are data, they are unit-tested, and the model cannot clear the flag. |
| Risk escalation | `risk_level == "high"` forces `requires_human_review = True`. The risk agent can only raise the flag, never lower it. |
| Grounding | The response agent sees only retrieved excerpts. Citations are validated against known policy ids after generation; a fabricated id downgrades the result to an abstention. |

The guardrail is a pattern rule table and will over-trigger. In a regulated context
that is the correct failure direction: over-escalation is safe, under-escalation is not.

## Repository layout

```
.
├── app/
│   ├── config.py          # settings from environment (step 1)
│   ├── contracts.py       # all Pydantic contracts between agents (step 1)
│   ├── retrieval.py       # policy loading and scoring (step 2)
│   ├── guardrail.py       # deterministic rules + risk escalation (step 3)
│   ├── graph.py           # LangGraph wiring and GraphState (step 6)
│   ├── api.py             # POST /ask, GET /health (step 7)
│   ├── llm/               # provider abstraction — base, groq_client, stub (step 4)
│   └── agents/            # policy, risk, response (step 5)
├── policies/              # synthetic policy corpus (step 2)
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
| `POLICY_DIR` | Location of the synthetic policy corpus. |
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
| 1 | Contracts and configuration | pending |
| 2 | Synthetic policy corpus and retrieval | pending |
| 3 | Deterministic guardrail and its tests | pending |
| 4 | LLM abstraction: protocol, Groq client, stub | pending |
| 5 | Policy, Risk and Response agents | pending |
| 6 | LangGraph wiring and end-to-end test | pending |
| 7 | FastAPI endpoints | pending |
| 8 | Streamlit UI | pending |
| 9 | Documentation pass | pending |

The guardrail and abstention logic are built before the agents deliberately. If the
timebox runs out, the governance layer is complete and tested, and the model layer is
what degrades.

## Scope and limitations

Declared rather than half-built, given the timebox:

- **Synthetic policies only.** No real customer data, and none should ever be entered.
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
