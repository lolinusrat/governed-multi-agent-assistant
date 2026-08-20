# Design decisions

Why the system is shaped the way it is, and what would change if it left the
demonstration and entered a bank. Written against a three-hour timebox: most of these
decisions are trades made deliberately, not defaults accepted quietly.

---

## Why three specialised agents

One model call with a long prompt would have been faster to build and cheaper to run.
Three exist because they produce **three separable artefacts that can be checked
independently**, not because more agents are better.

- The **Policy Agent** answers "what does the corpus say, and what procedure does it
  require". Its output can be checked against the corpus without knowing anything
  about risk.
- The **Risk Agent** answers "should this guidance go out". It sees the proposed
  guidance and the evidence, but never the Policy Agent's reasoning, so it reviews the
  output rather than agreeing with the rationale behind it.
- The **Response Agent** answers "how should this read to staff". It has the least
  authority in the system: the status, the risk status and the review flag are all
  decided before it runs.

The value is that a reviewer can point at the stage that got something wrong. With a
single call, a bad answer is one undifferentiated failure; here it is a retrieval
miss, or an ungrounded claim, or a missed escalation, each with its own contract and
its own tests.

The cost is latency — three sequential calls, roughly 3–8 seconds — and it is
genuinely sequential, since the Risk Agent must see the guidance before it can review
it. For a staff tool where the alternative is finding the policy manually, that trade
is easy.

## Why structured contracts

Every boundary is a Pydantic model. No stage passes a dictionary to another.

The immediate benefit is that a malformed model response fails at the boundary rather
than propagating into a staff-facing answer. The larger benefit is that **the contract
can carry the control**. Several governance properties are enforced by the type rather
than by code that must remember to run:

- `RiskStatus` is a closed set of three values. There is no numeric scale to
  misinterpret and no fourth value to add by accident.
- `PolicyFinding` has no field in which an action could be approved. The Policy Agent
  cannot grant permission because there is nowhere to express it.
- `PolicyEvidence` is frozen. A citation cannot be altered by any later stage.
- `FinalResponse` refuses to construct when `status`, `risk_status` or
  `human_review_required` disagree with the assessment and guardrail decision they came
  from. A response that overrides the guardrail is not a state this system can
  represent.

That last one is the pattern worth taking away: where a rule can be made unrepresentable
rather than merely checked, it should be.

## Why deterministic guardrails

The agents are language models. They are useful because they generalise, and that is
precisely why they cannot be the control. The same prompt can produce a different
answer tomorrow; a persuasively worded question can talk one into agreeing; and nothing
in the mechanism guarantees a refusal when one is required. **Good behaviour most of the
time is not a control.**

So the decisions that carry consequences are made by code:

- Abstention is a retrieval score against a threshold, applied before the model is
  called at all.
- Grounding is set intersection — a policy id the retriever did not return is
  discarded, whatever the model claims.
- Human review is a rule table matched against the proposed guidance. The Risk Agent's
  status is an *input* to that decision and can only add a reason for review, never
  remove one.

This buys three things worth more here than accuracy. It can be **tested** as a claim
about all inputs rather than a sample. It can be **explained** — every escalation names
the rule that fired and the authority from the cited policy. And it can be **changed
deliberately**, as an edit to an enum and a rule table that shows up in a diff, rather
than a change in prompt wording whose effect nobody can predict.

`app/guardrail.py` imports nothing from `app/llm/`, and a test parses the module to
prove it. The guardrail cannot consult a model even by accident.

The cost is false positives: the rules over-trigger and deliberately do no negation
handling, so "do not close the account" still escalates. The trade-off deliberately
favours over-escalation: unnecessary review has an operational cost, but under-escalation
can create customer and risk impact.

## Why unsupported questions abstain

A plausible answer with no basis in policy is worse than no answer, because staff
cannot tell the two apart. Confidence is not a signal of correctness, and a member of
staff reading fluent prose about a policy that does not exist has no way to notice.

Abstention is therefore the default outcome, and it is reached down four paths, each
with its own reason code so the UI and the audit trail can distinguish them:

- `NO_RELEVANT_POLICY` — nothing in the corpus matched.
- `INSUFFICIENT_EVIDENCE` — something matched, below the threshold.
- `OUT_OF_POLICY_SCOPE` — the best-matching section explicitly disowns the topic.
- `UNVERIFIABLE_CITATION` — guidance was produced that cannot be traced to a source.

Three of the four are decided without a model call. An abstention names the near-miss
section and its score, so a reader can tell "not covered" from "the system failed", and
it always recommends escalation rather than leaving staff with nothing.

The out-of-scope path deserves its own note. Each synthetic policy ends with a section
listing what it does not cover, and those sections score highly on exactly the questions
they disown. Relying on the model to notice that in the prompt was the weakest link in
the abstention path, so it is now decided before the model is consulted.

## Why synthetic Markdown policies

**Synthetic** because real customer or bank-confidential data was unnecessary and
inappropriate for a demonstration repository, and because a demonstration that quoted a
real bank's policy would imply an authority it does not have. The corpus describes a fictional institution and says so at the top of every
document.

**Markdown** because it parses with a regex, diffs in review, and is readable by the
person writing the rules and the person auditing them. YAML front matter carries the
metadata a citation needs — `policy_id`, `title`, `version`, `status`,
`effective_date`, `classification`, `owner` — and `# N. Heading` gives natural,
citable chunk boundaries. A section is the right retrieval unit here: large enough to
answer a question, small enough to quote verbatim.

The corpus is also written to exercise the controls. Each document contains ordinary
procedural guidance, a table of consequential actions with named approval authorities,
and a section of matters it does not cover. The approval tables are what the guardrail
lifts authority names from; the "not covered" sections are what the out-of-scope
backstop keys on. The corpus is a test fixture as much as it is content.

## Why no database was required

Nothing in the slice needs one. The policy corpus is four files read once at startup;
retrieval scores in memory in under a millisecond; and no request depends on a previous
one. Adding Postgres would have added a migration, a connection pool, a container and a
failure mode, in exchange for nothing the slice does.

The two things that would normally justify a database are handled honestly instead. The
**audit trail** is append-only JSONL, which is enough to reconstruct any request and is
one insert away from a table, since the record shape is already fixed as `AuditEvent`.
The **decision record** travels inside `FinalResponse`, so persistence is a write, not a
redesign.

This is a deliberate limit, not a claim that state is unnecessary. Durable audit
storage is the first thing production needs.

## Why Groq is behind an LLM abstraction

The application depends on one protocol with one method:

```python
class LLMClient(Protocol):
    model_name: str
    def structured(self, *, system: str, user: str, schema: type[T]) -> T: ...
```

Nothing outside `app/llm/` imports the Groq SDK. Everything provider-specific — JSON
mode, the retry on an unparseable response, the translation of SDK exceptions into
`LLMError` — is contained in one file.

Two payoffs, one expected and one immediate.

The expected one: swapping provider is a new file and a changed factory. In an
enterprise this is not hypothetical, since traffic will be routed through a gateway
rather than to a vendor directly.

The immediate one: `StubLLMClient` implements the same protocol, which is why **407
tests run offline in under a second with no API key**. That is not a testing
convenience, it is what makes the governance logic testable at all — the deterministic
controls can be asserted across every input without a network call in the loop.

The abstraction also drew a useful boundary in the other direction. Because
`structured()` returns a validated Pydantic model, no caller ever sees raw JSON, and a
provider outage arrives as one exception type that the workflow knows how to fail closed
on.

---

# What would change in production

High level only. Each of these is a system the bank already owns, and the point is that
the slice was built with a seam where it would attach.

### Enterprise policy repository

The corpus would come from the bank's policy management system rather than four files
in a folder, with the properties that system already provides: version history, an
approval workflow, effective and expiry dates, and a named owner per document. Citations
would pin a policy *version*, not just an id, so an answer given last month can be
reproduced against the policy that was in force at the time. Retrieval would need
freshness handling — reindexing on publication, and refusing to answer from a superseded
document.

### Identity and entitlements

There is no authentication today. In production the caller is a named staff member
arriving through SSO, and their entitlements shape the request end to end: which policies
they may see, which questions they may ask, and — most importantly — whether the approval
they hold satisfies the requirement the guardrail raised. `staff_role` is currently a
self-declared string; it would become an assertion from the identity provider. The
human-review step becomes a real workflow with a named approver, a timestamp and a
record, rather than a flag on a response.

### Enterprise search / vector retrieval

Keyword scoring is the first thing to replace. Production would use the bank's search
platform — hybrid lexical and vector retrieval, with entitlement filtering applied at
query time so a user never retrieves a document they may not read. The `PolicyRetriever`
protocol exists for this: a replacement satisfies one method and returns the same
`RetrievalResult`, including the explicit insufficient-evidence case. The abstention
threshold would need recalibrating against the new scoring, and that recalibration is a
governance change, not a tuning change.

### Model gateway

Calls would go to an internal gateway rather than to Groq directly: central credential
management, quota and cost attribution per team, rate limiting, model allow-listing, a
fallback when a model is decommissioned, and policy-controlled audit telemetry with —
where permitted — appropriately protected content logging. Whether raw prompts and
responses may be retained at all is a privacy decision, not a logging default, and it is
the same decision this slice already makes by keeping question and answer text out of
its own trail. This slice hit that last problem live — the configured model was
retired mid-exercise — which is precisely the failure a gateway absorbs. `app/llm/` is
the seam.

### Secrets management

The key currently lives in a git-ignored `.env`. Production would take it from the
bank's secrets manager with short-lived credentials, automatic rotation, no key material
on disk or in environment variables, and an access audit. The redaction that strips
key-shaped strings from responses and logs would stay — defence in depth, since it
catches credentials the code never knew about.

### Durable audit storage

The JSONL file becomes an append-only store with a retention policy aligned to the
bank's record-keeping obligations, tamper-evidence, and the ability to answer "show me
every consequential action this assistant escalated last quarter, and who approved it".
The record shape is already fixed as `AuditEvent`, and the decision record already
travels in the response, so this is a sink change rather than a redesign.

### Enterprise observability

Structured events would ship to the bank's platform — traces, metrics and logs
correlated on the same `trace_id` — with dashboards and alerting on the numbers that
matter for a governed assistant: abstention rate, escalation rate, rejection rate,
guardrail trigger distribution, provider latency and error rate. A sharp move in
abstention rate is an early signal that the corpus and the questions have drifted apart.

### Security and privacy controls

Input and output scanning for customer data, since staff will paste it in whatever the
policy says. Prompt-injection defences on retrieved content as well as on questions.
Data-residency and retention rules on anything the model sees. A DPIA and model risk
assessment before launch, and periodic re-review. The current design already refuses to
log question text, answers or evidence excerpts, keeping a length and digest instead —
that principle would extend rather than change.
