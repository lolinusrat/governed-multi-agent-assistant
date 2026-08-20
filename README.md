# Governed Banking Policy Assistant

An internal, staff-facing multi-agent assistant that answers operational questions
from a **synthetic** banking policy corpus. Built as a thin vertical slice under a
three-hour timebox, with the governance controls implemented in deterministic code
rather than left to the language model.

Python · uv · FastAPI · Streamlit · LangGraph · Pydantic · Groq · pytest

---

## Problem

A member of branch or contact-centre staff has a customer in front of them and a
question about bank policy. They need an answer they can act on, grounded in the
policy that actually governs the situation.

A general-purpose assistant is the wrong tool for this. It will answer confidently
whether or not a policy covers the question, and it has no notion of an action that
a person must authorise before staff take it. In a regulated setting those two
failures are the ones that matter:

1. **Answering something the policies do not cover.** A plausible answer with no
   basis in policy is worse than no answer, because staff cannot tell the difference.
2. **Letting staff act on guidance nobody reviewed.** Moving money, approving credit,
   closing or blocking an account are not decisions an assistant may make.

This system is built around those two failures. It abstains when the corpus does not
support the question, and it routes consequential actions to a human. Both are
enforced by rules, so they can be unit-tested and explained to a reviewer without
appealing to how the model usually behaves.

## Architecture

One API process, one Streamlit process, one linear LangGraph with four nodes. No
database, no vector store, no queue, no container runtime.

```
Streamlit UI ──HTTP──▶ FastAPI  POST /ask
                            │
                            ▼
                LangGraph — linear, no branching

  1. policy_agent     retrieval (deterministic) → LLM
                      evidence → PolicyFinding
                      abstains before the model is called when
                      retrieval is insufficient or out of scope

  2. risk_agent       deterministic rule pass + independent LLM review
                      → RiskAssessment {SAFE | HUMAN_REVIEW_REQUIRED | REJECTED}

  3. guardrail        deterministic, no LLM, no model input
                      → GuardrailDecision {requires_human_review, actions}

  4. response_agent   LLM prose, then deterministic post-checks
                      → FinalResponse
                            │
                            ▼
                  request_id · answer · policy_sources
                  risk_status · human_review_required
```

The graph is a straight line on purpose. The decisions that could branch — abstain,
reject, escalate — are represented inside the contracts, and each stage knows how to
do nothing when there is nothing to do. An abstention costs zero model calls because
all three agents short-circuit.

### Repository layout

```
.
├── app/
│   ├── contracts.py       # every contract between stages, one file
│   ├── config.py          # settings from environment; the only module that knows a key exists
│   ├── retrieval.py       # corpus parsing, BM25-style scoring, abstention
│   ├── guardrail.py       # deterministic consequential-action rules
│   ├── graph.py           # LangGraph wiring, shared state, error handling
│   ├── api.py             # POST /ask, GET /health, GET /trace/{id}
│   ├── observability.py   # JSONL audit trail, kept out of the decision code
│   ├── llm/               # provider abstraction: base, groq_client, stub
│   └── agents/            # policy, risk, response
├── data/                  # four synthetic policy documents
├── ui/streamlit_app.py    # calls the API over HTTP; holds no policy logic
└── tests/                 # 407 tests, fully offline
```

## How to run with uv

First-time setup:

```bash
uv sync                 # creates .venv and installs from uv.lock
cp .env.example .env    # then add your Groq key — see below
```

Run the two processes side by side:

```bash
uv run uvicorn app.api:app --reload         # API on :8000
uv run streamlit run ui/streamlit_app.py    # UI  on :8501
```

Open <http://localhost:8501>. The sidebar shows `ok · 4 policies` when the UI can
reach the API. The UI talks to the API over HTTP and holds no policy logic of its
own, so anything it displays can be reproduced with `curl`.

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, policies loaded, configured provider and model. Never credentials. |
| `POST /ask` | `{"question": "…", "staff_role": "branch_staff"}` |
| `GET /trace/{request_id}` | The audit trail for one request, so the UI can show an execution trace without reading the log file behind the API's back. |

`POST /ask` returns **200** for an answer, abstention or rejection; **422** for
invalid input with field-level detail; and **503** when a stage failed — carrying the
full governed body, because the caller needs to read "human review required" rather
than just a status code.

## How to configure Groq

Groq is configured entirely through the environment. `.env` is git-ignored and must
never be committed; `.env.example` holds names and shapes only.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Your Groq key. Read only in `app/config.py`. |
| `GROQ_MODEL` | Model id, e.g. `openai/gpt-oss-120b`. |
| `LLM_PROVIDER` | `groq` for real calls, `stub` for deterministic offline runs. |
| `RETRIEVAL_MIN_SCORE` | Abstention threshold, 0–1. Default `0.15`. |
| `RETRIEVAL_MAX_RESULTS` | How many policy sections may be used as evidence. Default `6`. |
| `POLICY_DIR` | Location of the policy corpus. Default `data`. |
| `API_BASE_URL` | Base URL the UI uses to reach the API. |
| `OBSERVABILITY_ENABLED` / `OBSERVABILITY_FILE` | JSONL audit trail and its path. |

Model ids change. If `POST /ask` returns `UNAVAILABLE` with a 404 from Groq, the
configured model has been decommissioned — list what your account can reach and set
`GROQ_MODEL` accordingly.

### The provider abstraction

The application depends on one protocol:

```python
class LLMClient(Protocol):
    model_name: str
    def structured(self, *, system: str, user: str, schema: type[T]) -> T: ...
```

`GroqClient` and `StubLLMClient` implement it. Nothing outside `app/llm/` imports a
vendor SDK, so changing provider means adding one file and changing one factory. The
stub keeps the whole test suite offline and free of API keys.

## How to test

```bash
uv run pytest
```

**407 tests, offline, well under a second.** No API key, no network, no fixtures that
reach a provider. Retrieval, the guardrail and the contracts run for real; only the
model boundary is stubbed.

| File | Tests | Focus |
|---|---|---|
| `test_guardrail.py` | 116 | Every action phrasing × every risk status; adversarial passive voice; AST proof the layer cannot reach an LLM |
| `test_response_agent.py` | 50 | Cannot override review, rejection, or introduce a new action |
| `test_observability.py` | 43 | Trail completeness, and that it holds no question text, answer text or credentials |
| `test_retrieval.py` | 40 | Routing, golden questions, abstention boundary, determinism, stemming |
| `test_risk_agent.py` | 35 | Six risk categories, escalate-only combination, evidence immutability |
| `test_api.py` | 30 | Wire contract, malformed input, controlled failure, secret hygiene |
| `test_contracts.py` | 29 | Closed status sets, governance-field invariants |
| `test_policy_agent.py` | 25 | Grounding, four abstention paths, out-of-scope backstop |
| `test_graph.py` | 25 | End-to-end runs, graph shape, controlled error handling |
| `test_llm.py` | 14 | Retry, exhaustion, SDK error wrapping against a fake SDK |

Running without a Groq key: set `LLM_PROVIDER=stub` and everything works offline, but
the stub returns conservative defaults, so every question abstains. Useful for
checking wiring, not for demonstrating behaviour.

## Example scenarios

Real output from the running system, not illustrations.

### 1. Consequential action → human review

> *"Can I immediately block the customer's account?"*

```
status: PENDING_HUMAN_REVIEW   risk_status: HUMAN_REVIEW_REQUIRED   human_review_required: true

Human review is required before you act on this. Approval must come from
Fraud Operations Manager.

Blocking the account is classified as "freezing or restricting an account" and,
per FRAUD-ESC-002 §5.1, must be reviewed and approved by the Fraud Operations
Manager before any action is taken.

next step 1: Obtain approval from Fraud Operations Manager before taking any action.
sources:     FRAUD-ESC-002 §3, §5, §4
```

The approval authority is lifted from the `| action | authority |` table in the cited
policy, not generated. The guardrail would have required review here even if the Risk
Agent had returned `SAFE`.

### 2. Unsupported question → abstention

> *"Can I guarantee the customer will get their money back?"*

```
status: ABSTAINED   abstain_reason: NO_RELEVANT_POLICY   policy_sources: []

I cannot answer this from the bank's policies. The closest match was FRAUD-ESC-002
section '4. Escalation tiers' at 0.10, below the 0.15 evidence threshold.
```

Decided by retrieval score before any model call. The response names the near-miss
and its score, so a reader can tell "not covered" from "the system failed".

### 3. Out of scope → abstention, deterministically

> *"What is the tax treatment of a written-off disputed amount?"*

Every policy ends with a section listing what it does not cover, and that section
scores top for this question. When the best match is a section that disowns the
topic, the assistant abstains **before the model is consulted** with
`OUT_OF_POLICY_SCOPE`, naming the section and pointing at the policy owner.

### 4. Ordinary procedural question → answered

> *"How quickly must I acknowledge a customer complaint?"*

```
status: ANSWERED   human_review_required: false
sources: COMP-HAND-004 §2, §4, §1
```

No consequential action, nothing for risk to raise, sources cited. This is the only
route to `ANSWERED`.

### 5. Provider failure → fails closed

```
HTTP 503
status: UNAVAILABLE   human_review_required: true   policy_sources: []

The assistant could not complete its checks, so it has no answer for you.
… Do not treat this as a green light. Escalate the question instead.
```

An incomplete run must never read as a cleared one.

## Safety model

The system assumes the model will sometimes be wrong, confident, and persuasive. Every
control that matters is therefore code, not a prompt.

| Control | Mechanism | Where |
|---|---|---|
| **Abstain rather than hallucinate** | Top retrieval score below `RETRIEVAL_MIN_SCORE` short-circuits **before any LLM call**. The model's own `answerable=false` is also honoured. | `retrieval.py`, `agents/policy.py` |
| **Out-of-scope abstention** | If the best-matching section's heading says the policy does not cover the topic, abstain before consulting the model. | `agents/policy.py` |
| **Grounding** | Cited policy ids are intersected with what the retriever actually returned. Invented ids are discarded; guidance left with no verifiable citation becomes an abstention. | `agents/policy.py` |
| **Unsupported figures** | Every number in the guidance must appear in the cited evidence. A plausible-but-wrong threshold is the failure a reader cannot catch unaided. | `agents/risk.py` |
| **Human review for consequential actions** | A rule table over the guidance and procedures: transferring funds, approving credit, closing an account, blocking an account. No risk status can clear a match. | `guardrail.py` |
| **Risk escalation is one-way** | The deterministic rule pass can raise the status the model returned, never lower it. An unscripted or degenerate model response escalates. | `agents/risk.py` |
| **The answer cannot smuggle in an action** | The composed answer is re-scanned with the guardrail's own rules. A draft introducing an action the guardrail never cleared is discarded. | `agents/response.py` |
| **The response cannot override upstream** | `FinalResponse` refuses to construct when `status`, `risk_status` or `human_review_required` disagree with the assessment and guardrail decision they came from. | `contracts.py` |
| **Failure fails closed** | Any stage raising produces `UNAVAILABLE` with `human_review_required = true`. | `graph.py` |
| **Secret hygiene** | The configured key and anything key-shaped is stripped from every response and every log line. | `observability.py`, `api.py` |

Two structural properties back this up, both enforced by tests that parse the source:

- `app/guardrail.py` imports nothing from `app/llm/` or `groq`. The layer cannot
  consult a model even by accident.
- None of `contracts.py`, `retrieval.py`, `guardrail.py` or `agents/*` imports the
  observability module. Instrumentation attaches at the wiring.

### Why the important controls are deterministic

Models are useful because they generalise, and that is exactly why they cannot be the
control: the same prompt can produce a different answer tomorrow, and nothing in the
mechanism guarantees a refusal when one is needed. Good behaviour most of the time is
not a control. Determinism buys three things worth more here than accuracy:

1. **It can be tested.** `requires_human_review` is asserted for every consequential
   action phrasing crossed with every risk status the model can return — a claim about
   all inputs, not a sample.
2. **It can be explained.** Every escalation names the rule that fired and the approval
   authority from the cited policy. "The model thought it was risky" is not an audit
   trail.
3. **It can be changed deliberately.** Adding a consequential action is an edit to an
   enum and a rule table, reviewable in a diff.

The cost is false positives. The rule table over-triggers, and it deliberately does no
negation handling, so guidance saying "do not close the account" still escalates.
Over-escalation costs a reviewer a minute; under-escalation costs a customer.

## Observability and audit

Every request gets a `trace_id` in HTTP middleware, before any handler runs, so a
request that fails validation is still accounted for. It returns as `request_id` in
the body and as the `X-Request-ID` header.

Events are appended one JSON object per line to `logs/events.jsonl`, each carrying
`timestamp`, `trace_id`, `component`, `event`, `status` and `latency_ms` where a
duration is meaningful. `GET /trace/{request_id}` reads one request back:

```
a89813b7e33e4358ad11d5648c63df5c
├── request received
├── retrieval → FRAUD-ESC-002 §3
├── retrieval → FRAUD-ESC-002 §5
├── policy_agent → completed
├── risk_agent → HUMAN_REVIEW_REQUIRED
├── guardrail → consequential action detected
├── response_agent → PENDING_HUMAN_REVIEW
└── response returned
```

**What is not recorded:** the question text, the drafted answer and evidence excerpts
never reach the file. A staff member may paste customer detail into a question. A
length and a short SHA-256 digest are kept instead, enough to correlate and spot
repeats. Every string is passed through redaction on the way out, so a provider error
quoting a credential cannot land in the log.

## Limitations

Declared rather than half-built, given the timebox.

- **Synthetic policies only.** The four documents in `data/` describe a fictional
  institution, Meridian Retail Bank, and reproduce no real bank's policy. No real
  customer data, and none should ever be entered.
- **Keyword retrieval, not embeddings.** BM25-style scoring with length normalisation
  and a crude stemmer, so `close` matches `closing`. Still blind to synonyms — `card`
  will not match `payment instrument`. First thing to replace.
- **The guardrail is a pattern rule table.** It covers four action types in English
  and will both over-trigger and miss novel phrasings. It is a demonstration of the
  control shape, not a complete one.
- **Risk agent independence is prompt-level, not process-level.** A separate call with
  a separate system prompt and a restricted view — not a separate model or provider.
- **Recommended next steps can repeat themselves.** The Policy Agent's procedures and
  the Response Agent's steps are merged with exact-string de-duplication, so near-
  duplicate phrasings both survive. Cosmetic, visible in longer answers.
- **No authentication, authorisation, rate limiting or PII scrubbing.** `GET /trace`
  is unauthenticated. Acceptable only for a local demonstration over synthetic data.
- **Audit trail is a local file.** No retention policy, no tamper-evidence, no
  shipping. The full decision record travels in the response, so durable storage is an
  insert away.
- **Single process, in-memory.** No persistence of requests, no caching, no
  horizontal-scaling story.
- **Advisory only.** Output supports a member of staff making a decision. It is not an
  approval, and anything flagged for human review must be reviewed before staff act.

See [DECISIONS.md](DECISIONS.md) for why the system is shaped this way and what would
change in production, and [AI_ASSISTED_ENGINEERING.md](AI_ASSISTED_ENGINEERING.md) for
how it was built.
