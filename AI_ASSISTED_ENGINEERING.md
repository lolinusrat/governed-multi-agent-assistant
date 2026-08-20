# AI-assisted engineering

A factual record of how Claude Code was used to build this system during the exercise,
including where it was useful, where it was wrong, and where the engineer's judgement
was the deciding factor.

The short version: **the coding agent accelerated implementation; the engineer remained
responsible for architecture, security boundaries, review, verification and the final
engineering decisions.** The sections below say what that meant in practice rather than
asserting it as a slogan.

---

## Framing

The engineer set the brief: the problem, the stack, the constraints (three-hour
timebox, synthetic policies only, no real customer data, no unnecessary
infrastructure, abstain rather than hallucinate, human review for consequential
actions, deterministic controls where practical, structured contracts) — and one
instruction that shaped everything after it:

> *First, DO NOT build the application. Propose … Stop after presenting the design and
> wait for my review.*

That gate was the single most useful control in the exercise. The agent produced a
design — minimum architecture, repository structure, data contracts, implementation
sequence, trade-offs — and stopped. Nothing was written to disk until the engineer
approved it. Design review happened while changing direction was still free.

## Architecture

The agent proposed the architecture. The engineer reviewed it, approved it, and then
made the decisions that actually determine the system's governance behaviour. Those
came from the engineer, not the agent:

- the four consequential actions the guardrail treats as requiring review (transfer
  funds, approve credit, close an account, block an account)
- the three risk statuses, and that there be exactly three — `SAFE`,
  `HUMAN_REVIEW_REQUIRED`, `REJECTED` — rather than a numeric scale
- the shape of `RiskAssessment` (status, reason, identified risks) and of the final
  response (request id, answer, policy sources, risk status, human-review flag)
- that the guardrail must be independent of the LLM and that the LLM must not be able
  to override it
- that the workflow stay a straight line, with no additional agents
- that the API expose two endpoints, and that Streamlit call the API rather than
  bypassing it

The agent's contribution at this layer was proposing options and implementing the
decision. The constraints that make the system *governed* were specified by the
engineer.

## Scaffolding

On approval, the agent generated the project skeleton in one step: `pyproject.toml`
with pinned dependency groups, `.gitignore`, `.env.example`, the README, and the
package layout. It ran `uv lock` and `uv sync`, then verified the result rather than
assuming it — importing every top-level dependency and running `pytest` to confirm it
collected zero tests cleanly.

It also verified secret hygiene at this point by writing a throwaway `.env`, confirming
git ignored it, and deleting it. That check mattered later.

## Implementation

Built in nine steps, each explicitly gated by the engineer and each ending with a
"stop when complete" instruction:

| Step | Delivered |
|---|---|
| 0 | Skeleton, dependencies, configuration |
| 1 | Pydantic contracts and settings |
| 2 | Four synthetic policy documents and local retrieval |
| 3 | Deterministic guardrail |
| 4 | LLM abstraction: protocol, Groq client, offline stub |
| 5 | Policy, Risk and Response agents (one step each, separately gated) |
| 6 | LangGraph orchestration and controlled error handling |
| 7 | FastAPI endpoints |
| 8 | Streamlit UI |
| 9 | Documentation |

Roughly 3,200 lines of application code and 2,900 lines of tests. The agent wrote
substantially all of it. Speed was the clear benefit: the guardrail rule table, the
BM25 scorer, the LangGraph wiring and the FastAPI layer were each produced in minutes
rather than hours, which is what made a nine-step vertical slice feasible in the
timebox at all.

The step gating was not ceremony. Twice it caught scope drift before it compounded:
the agent flagged that the LLM abstraction had to be built before the Policy Agent
could use it, and flagged that showing an execution trace in the UI required a third
API endpoint — rather than silently adding either.

## Test generation

407 tests, all offline. The agent wrote them alongside each step, and the stub LLM
client — which implements the same protocol as the Groq client — is what keeps the
suite free of network calls and API keys.

Test generation was genuinely accelerating: exhaustive parameterised coverage, such as
every consequential-action phrasing crossed with every risk status the model can
return, is tedious to write by hand and quick to generate.

It was also where the agent was most often wrong in an instructive way. Three examples,
all real:

- A test asserted that guidance mentioning "a provisional credit of 500 AUD" produced
  no risk flags. It failed — correctly, because a provisional credit *is* a
  consequential action. The code was right and the test's assertion was too broad.
- A test question was written as *"A customer named Jane Roe wants to close their
  account"* to prove the audit log holds no names. It failed because the two unknown
  name tokens dropped the retrieval score to 0.1485, just under the 0.15 threshold, so
  the system abstained. The fixture was wrong, not the retriever.
- A response-agent fixture paired a "no review required" guardrail decision with
  account-closure guidance — a state the real system cannot produce. Those tests were
  passing against an impossible combination and were rewritten.

In each case the failing test was the useful output. Generated tests need reading, not
just running.

## Code review

The engineer explicitly commissioned a review — "review this implementation as a senior
engineer responsible for a regulated-enterprise AI system, do not change the code yet"
— naming the areas to examine and asking for findings categorised by severity.

The agent wrote throwaway probe scripts rather than reading code and speculating. Four
findings were demonstrated with executable evidence:

1. **The guardrail never saw the text staff read.** It evaluated the Policy Agent's
   finding; the Response Agent then composed prose afterwards. A probe produced a
   response reading *"close the account and transfer the funds"* with
   `human_review_required: false`. The system's headline control was bypassable by the
   last model in the chain.
2. **Passive phrasings evaded the guardrail.** Four of nine plausible phrasings missed
   — *"The account will be closed"*, *"Send the money to…"* — because the rules required
   the verb before the noun. The existing 98 guardrail tests all used active-voice
   phrasings the agent had written itself, so they tested the regex against the
   phrasings the regex was written for.
3. **Out-of-scope abstention depended entirely on the model.** Each policy's "matters
   not covered" section scored top for the questions it disowns, and nothing
   deterministic acted on that.
4. **`FinalResponse` validated two governance fields but not `status`.**

A suspected concurrency bug was also investigated and **found not to exist** — eight
parallel requests kept their traces separate. Reporting a non-finding matters as much
as reporting a finding.

The honest limitation: this review was performed by the same agent that wrote the code.
That is weaker than independent review, and it is not a substitute for it. It found real
defects because it was grounded in executed probes rather than self-assessment, but a
production system would need review by someone who did not write it.

## Identifying edge cases

Several issues surfaced only by running the system rather than reasoning about it:

- **The configured Groq model had been decommissioned.** Live calls returned a 404.
  This was found by actually invoking the API, and it doubled as an unplanned test of
  the fail-closed path: the system returned `UNAVAILABLE` with human review required,
  and leaked no credential in the error.
- **A retrieval defect was diagnosed wrongly, then corrected.** The agent first
  attributed a failing golden question to a `lodge`/`lodged` stemming gap and wrote that
  cause into an `xfail` reason. Measuring the actual IDF contributions and section
  scores showed the recorded cause was wrong: the discriminating term *did* match, and
  the real problems were that long sections won on repetition of cheap words, and that
  the evidence window was cut at two sections per policy. The fix that followed — BM25
  length normalisation, a proper stemmer, a wider window — addressed the measured cause.
  The first diagnosis was plausible and wrong, and only measurement separated them.
- **An `UNSUPPORTED_CLAIM` false-rejection pattern** appeared in live runs before it
  appeared in any test, because it depended on model variance across runs.

The pattern worth naming: the agent's confident explanations needed the same
verification as its code.

## Documentation

The README was maintained incrementally through the build — status, controls table,
configuration and limitations updated at each step rather than written at the end, so it
never described a system that did not exist. This document, `DECISIONS.md` and the final
README restructure were produced in the documentation step.

Example scenarios in the README are **real output captured from the running system**,
not illustrations written to look plausible.

---

## Division of responsibility

**The agent did:** propose designs and trade-offs; generate application code, tests,
synthetic policy content and documentation; run the test suite and the live services;
write probe scripts; diagnose failures; report findings with severity.

**The engineer did:** set the problem, the stack and the constraints; require a design
before any code; approve the architecture; specify the governance semantics — which
actions are consequential, which statuses exist, what may never be overridden; gate every
implementation step; commission the review and choose which findings to act on within
the remaining time; and remain accountable for what the system does.

The clearest illustration of that boundary was the secret incident. A real Groq API key
was committed into `.env.example` and GitHub's push protection rejected the push. The
agent diagnosed it — identifying that the first two commits had succeeded and only the
third was rejected, locating the key, moving it to a git-ignored `.env` and restoring the
placeholder — and recommended rotating the key on the grounds that a blocked push still
uploads the object to GitHub before rejection. **The engineer executed the history
change and owns the rotation decision.** The agent's write access to git history was
withheld by policy, which is the correct arrangement.

Two things follow from this exercise that are worth stating plainly.

First, **the agent's speed is real and so is its confidence when wrong.** It generated a
correct BM25 implementation quickly and also recorded a wrong root-cause analysis in a
test annotation with equal fluency. The difference was only ever established by
measuring.

Second, **generated tests are evidence about the code, not proof of it.** The guardrail
had 98 passing tests and a demonstrable bypass, because the tests and the implementation
shared an author and therefore shared a blind spot. Adversarial review, executed probes
and live runs found what the unit tests could not.
