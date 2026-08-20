"""Policy Agent tests.

Retrieval runs for real against the synthetic corpus; the model boundary is
stubbed. That split is deliberate - the properties worth testing here are the
deterministic ones the agent applies around the model, not the model's prose.
"""

from __future__ import annotations

import pytest

from app.agents.policy import PolicyAgent, PolicyDraft, format_evidence
from app.contracts import AbstainReason, AskRequest, PolicyEvidence, PolicyFinding, RetrievalResult
from app.llm.base import LLMError
from app.llm.stub import FailingLLMClient, StubLLMClient

CARD_QUESTION = "How long does a customer have to lodge a card transaction dispute?"
WAIVER_QUESTION = "A customer wants the card dispute handling fee waived. Can I do that?"


@pytest.fixture()
def agent_factory(retriever):
    def _make(drafts=None, llm=None):
        return PolicyAgent(retriever=retriever, llm=llm or StubLLMClient(drafts or []))

    return _make


class _EmptyRetriever:
    """Retriever that always reports insufficient evidence."""

    def __init__(self, reason=AbstainReason.NO_RELEVANT_POLICY):
        self.reason = reason

    def search(self, query: str, *, limit: int | None = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            sufficient=False,
            best_score=0.0,
            threshold=0.15,
            abstain_reason=self.reason,
            explanation="nothing in the corpus matched",
        )


class TestAnswerablePath:
    def test_returns_a_grounded_finding(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    proposed_guidance="Disputes must be lodged within 90 calendar days.",
                    required_procedures=[
                        "Verify the customer's identity",
                        "Lodge the dispute within one business day",
                    ],
                    cited_policy_ids=["CARD-DISP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))

        assert isinstance(finding, PolicyFinding)
        assert finding.answerable is True
        assert finding.abstain_reason is None
        assert finding.cited_policy_ids == ["CARD-DISP-001"]
        assert finding.required_procedures[0] == "Verify the customer's identity"
        assert finding.is_grounded

    def test_finding_carries_policy_id_and_section_for_every_citation(self, agent_factory):
        agent = agent_factory(
            [PolicyDraft(answerable=True, proposed_guidance="g", cited_policy_ids=["CARD-DISP-001"])]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))

        assert finding.evidence
        for evidence in finding.evidence:
            assert evidence.policy_id == "CARD-DISP-001"
            assert evidence.section[0].isdigit() and ". " in evidence.section
            assert evidence.text.strip()

    def test_evidence_is_narrowed_to_what_was_actually_cited(self, agent_factory):
        # A cross-cutting question retrieves several policies; only the cited one
        # should travel onward as support for the guidance.
        agent = agent_factory(
            [PolicyDraft(answerable=True, proposed_guidance="g", cited_policy_ids=["FRAUD-ESC-002"])]
        )
        finding = agent.run(AskRequest(question="what approval is required before closing an account"))
        assert {e.policy_id for e in finding.evidence} == {"FRAUD-ESC-002"}

    def test_blank_procedures_are_dropped(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    proposed_guidance="g",
                    required_procedures=["  Verify identity  ", "", "   "],
                    cited_policy_ids=["CARD-DISP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))
        assert finding.required_procedures == ["Verify identity"]


class TestGroundingIsEnforced:
    def test_discards_a_policy_id_that_was_never_retrieved(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    proposed_guidance="g",
                    cited_policy_ids=["CARD-DISP-001", "LOAN-ORIG-999"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))
        assert finding.cited_policy_ids == ["CARD-DISP-001"]
        assert all(e.policy_id != "LOAN-ORIG-999" for e in finding.evidence)

    def test_abstains_when_every_citation_is_invented(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    proposed_guidance="Staff may waive the fee at their discretion.",
                    cited_policy_ids=["MADE-UP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))

        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.UNVERIFIABLE_CITATION
        assert finding.proposed_guidance == ""
        assert finding.evidence == []

    def test_abstains_when_answerable_but_no_citations_given(self, agent_factory):
        agent = agent_factory([PolicyDraft(answerable=True, proposed_guidance="g")])
        finding = agent.run(AskRequest(question=CARD_QUESTION))
        assert finding.abstain_reason is AbstainReason.UNVERIFIABLE_CITATION

    def test_abstains_when_answerable_but_guidance_is_empty(self, agent_factory):
        agent = agent_factory(
            [PolicyDraft(answerable=True, proposed_guidance="   ", cited_policy_ids=["CARD-DISP-001"])]
        )
        finding = agent.run(AskRequest(question=CARD_QUESTION))
        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE


class TestAbstention:
    def test_abstains_without_calling_the_model_when_evidence_is_insufficient(self):
        # The expensive and least predictable step is skipped entirely.
        llm = StubLLMClient()
        agent = PolicyAgent(retriever=_EmptyRetriever(), llm=llm)
        finding = agent.run(AskRequest(question="What is the weather in Sydney tomorrow?"))

        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.NO_RELEVANT_POLICY
        assert llm.call_count == 0
        assert finding.notes

    def test_honours_the_retrieval_abstain_reason(self):
        agent = PolicyAgent(
            retriever=_EmptyRetriever(AbstainReason.INSUFFICIENT_EVIDENCE), llm=StubLLMClient()
        )
        finding = agent.run(AskRequest(question="anything at all"))
        assert finding.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE

    def test_abstains_when_the_model_says_it_cannot_answer(self, agent_factory):
        agent = agent_factory([PolicyDraft(answerable=False, notes="evidence does not cover this")])
        finding = agent.run(AskRequest(question=CARD_QUESTION))
        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
        assert finding.notes == "evidence does not cover this"

    def test_out_of_scope_evidence_produces_a_distinct_reason(self, agent_factory):
        # The corpus retrieves the "matters not covered" section above threshold;
        # the agent must recognise that as a refusal, not as supporting evidence.
        agent = agent_factory([PolicyDraft(answerable=False, out_of_scope=True)])
        finding = agent.run(
            AskRequest(question="What is the tax treatment of a written-off disputed amount?")
        )
        assert finding.abstain_reason is AbstainReason.OUT_OF_POLICY_SCOPE
        assert finding.proposed_guidance == ""

    def test_out_of_scope_overrides_an_answerable_claim(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    out_of_scope=True,
                    proposed_guidance="The amount is treated as assessable income.",
                    cited_policy_ids=["CARD-DISP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question="tax treatment of a written-off disputed amount"))
        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.OUT_OF_POLICY_SCOPE

    def test_an_unscripted_model_response_abstains(self, agent_factory):
        # StubLLMClient falls through to PolicyDraft defaults, which are conservative.
        finding = agent_factory([]).run(AskRequest(question=CARD_QUESTION))
        assert finding.answerable is False


class TestNoConsequentialDecisions:
    def test_the_finding_has_no_field_that_could_approve_an_action(self):
        fields = set(PolicyFinding.model_fields)
        assert not fields & {"approved", "authorised", "decision", "requires_human_review"}

    def test_approval_requirements_are_reported_as_procedure_not_permission(self, agent_factory):
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    proposed_guidance="A fee waiver requires manager approval before it is applied.",
                    required_procedures=[
                        "Obtain Branch or Contact Centre Manager approval before waiving the fee",
                        "Record the approval against the dispute reference",
                    ],
                    cited_policy_ids=["CARD-DISP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=WAIVER_QUESTION))
        assert any("approval" in step.lower() for step in finding.required_procedures)

    def test_the_system_prompt_forbids_approving_actions(self):
        from app.agents.policy import SYSTEM_PROMPT

        lowered = SYSTEM_PROMPT.lower()
        assert "do not decide, approve or authorise" in lowered
        assert "never invent" in lowered


class TestPromptConstruction:
    def test_prompt_contains_only_retrieved_evidence(self, agent_factory, retriever):
        llm = StubLLMClient([PolicyDraft(answerable=False)])
        agent = agent_factory(llm=llm)
        agent.run(AskRequest(question=CARD_QUESTION))

        prompt = llm.calls[0].user
        retrieved = retriever.search(CARD_QUESTION)
        for evidence in retrieved.evidence:
            assert evidence.policy_id in prompt
            assert evidence.section in prompt
        assert "CARD-DISP-001" in prompt
        assert "FRAUD-ESC-002" not in prompt or "FRAUD-ESC-002" in {
            e.policy_id for e in retrieved.evidence
        }

    def test_prompt_carries_the_question_and_staff_role(self, agent_factory):
        llm = StubLLMClient([PolicyDraft(answerable=False)])
        agent_factory(llm=llm).run(AskRequest(question=CARD_QUESTION, staff_role="contact_centre"))
        prompt = llm.calls[0].user
        assert CARD_QUESTION in prompt
        assert "contact centre" in prompt

    def test_format_evidence_labels_id_section_and_text(self):
        rendered = format_evidence(
            [
                PolicyEvidence(
                    policy_id="P-1",
                    title="A Policy",
                    section="5. Consequential actions",
                    text="Approval is required.",
                    score=0.9,
                )
            ]
        )
        assert "policy_id: P-1" in rendered
        assert "section: 5. Consequential actions" in rendered
        assert "Approval is required." in rendered


class TestProviderFailure:
    def test_a_provider_outage_is_not_disguised_as_an_abstention(self, retriever):
        # Reporting an outage as "no policy covers this" would mislead staff.
        agent = PolicyAgent(retriever=retriever, llm=FailingLLMClient("groq timed out"))
        with pytest.raises(LLMError, match="groq timed out"):
            agent.run(AskRequest(question=CARD_QUESTION))


class TestOutOfScopeBackstop:
    """Every policy ends with a section saying what it does not cover.

    When that section is the best match, the corpus is telling us not to answer.
    Leaving that to the model noticing it in the prompt was the weakest point in
    the abstention path, so it is decided before the model is consulted.
    """

    TAX_QUESTION = "What is the tax treatment of a written-off disputed amount?"

    def test_a_disowning_section_abstains_without_a_model_call(self, retriever):
        llm = StubLLMClient()
        finding = PolicyAgent(retriever=retriever, llm=llm).run(AskRequest(question=self.TAX_QUESTION))
        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.OUT_OF_POLICY_SCOPE
        assert llm.call_count == 0

    def test_the_model_cannot_talk_the_backstop_out_of_abstaining(self, agent_factory):
        # The model is never asked, so a confident answer has nowhere to enter.
        agent = agent_factory(
            [
                PolicyDraft(
                    answerable=True,
                    out_of_scope=False,
                    proposed_guidance="The written-off amount is treated as assessable income.",
                    cited_policy_ids=["CARD-DISP-001"],
                )
            ]
        )
        finding = agent.run(AskRequest(question=self.TAX_QUESTION))
        assert finding.answerable is False
        assert finding.abstain_reason is AbstainReason.OUT_OF_POLICY_SCOPE
        assert finding.proposed_guidance == ""

    def test_the_abstention_names_the_section_that_disowned_it(self, retriever):
        finding = PolicyAgent(retriever=retriever, llm=StubLLMClient()).run(
            AskRequest(question=self.TAX_QUESTION)
        )
        assert "Matters not covered" in finding.notes
        assert "policy owner" in finding.notes

    def test_an_ordinary_question_is_unaffected(self, agent_factory):
        agent = agent_factory(
            [PolicyDraft(answerable=True, proposed_guidance="g", cited_policy_ids=["CARD-DISP-001"])]
        )
        assert agent.run(AskRequest(question=CARD_QUESTION)).answerable is True
