"""Retrieval tests.

The behaviour that matters here is not ranking quality but the abstention
boundary: retrieval must be able to say "not enough evidence" explicitly, and it
must be deterministic so that decision is testable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.contracts import AbstainReason, RetrievalResult
from app.retrieval import (
    KeywordPolicyRetriever,
    PolicyCorpusError,
    PolicyRetriever,
    load_corpus,
    parse_policy_document,
    tokenize,
)

EXPECTED_POLICY_IDS = {"CARD-DISP-001", "FRAUD-ESC-002", "PRIV-DATA-003", "COMP-HAND-004"}


class TestCorpusParsing:
    def test_loads_every_policy_with_metadata(self, corpus):
        assert {d.policy_id for d in corpus} == EXPECTED_POLICY_IDS
        for doc in corpus:
            assert doc.title and doc.version and doc.status
            assert doc.effective_date and doc.classification and doc.owner
            assert doc.sections

    def test_sections_are_numbered_and_labelled(self, corpus):
        for doc in corpus:
            numbers = [s.number for s in doc.sections]
            assert numbers == sorted(numbers, key=int)
            assert doc.sections[0].label.startswith("1. ")

    def test_corpus_is_labelled_synthetic(self, corpus):
        # A demonstration corpus must not be mistakable for a real bank's policy.
        for doc in corpus:
            raw = Path(doc.source_path).read_text(encoding="utf-8")
            assert "SYNTHETIC DEMONSTRATION POLICY" in raw
            assert all(s.text.strip() for s in doc.sections)

    def test_rejects_a_file_without_front_matter(self, tmp_path):
        bad = tmp_path / "broken.md"
        bad.write_text("# 1. Purpose\nSome text.\n", encoding="utf-8")
        with pytest.raises(PolicyCorpusError, match="front matter"):
            parse_policy_document(bad)

    def test_rejects_a_file_without_numbered_sections(self, tmp_path):
        bad = tmp_path / "broken.md"
        bad.write_text("---\npolicy_id: X-1\ntitle: X\n---\n\nJust prose.\n", encoding="utf-8")
        with pytest.raises(PolicyCorpusError, match="no numbered sections"):
            parse_policy_document(bad)

    def test_rejects_an_empty_directory(self, tmp_path):
        with pytest.raises(PolicyCorpusError):
            load_corpus(tmp_path)


class TestTokenize:
    def test_drops_stopwords_and_singularises(self):
        assert tokenize("What are the time limits for disputes?") == ["time", "limit", "dispute"]

    def test_keeps_numeric_thresholds(self):
        assert "500" in tokenize("provisional credit of 500 AUD")


class TestSuccessfulRetrieval:
    @pytest.mark.parametrize(
        ("question", "expected_policy_id"),
        [
            ("How long does a customer have to lodge a card transaction dispute?", "CARD-DISP-001"),
            ("Can I unblock a card that was blocked for suspected fraud?", "FRAUD-ESC-002"),
            ("Can I tell a third party about a customer's account?", "PRIV-DATA-003"),
            ("What is the timeframe for responding to a complaint?", "COMP-HAND-004"),
        ],
    )
    def test_routes_a_question_to_the_right_policy(self, retriever, question, expected_policy_id):
        result = retriever.search(question)
        assert result.sufficient is True
        assert expected_policy_id in result.policy_ids

    def test_evidence_carries_everything_needed_to_cite(self, retriever):
        result = retriever.search("How long does a customer have to lodge a card dispute?")
        for evidence in result.evidence:
            assert evidence.policy_id
            assert evidence.title
            assert evidence.section[0].isdigit() and ". " in evidence.section
            assert evidence.text.strip()
            assert 0.0 <= evidence.score <= 1.0

    def test_evidence_text_is_verbatim_from_the_source_document(self, retriever, corpus):
        result = retriever.search("What approval is needed to waive a fee?")
        by_id = {d.policy_id: d for d in corpus}
        for evidence in result.evidence:
            section = next(
                s for s in by_id[evidence.policy_id].sections if s.label == evidence.section
            )
            excerpt = evidence.text.removesuffix(" […]")
            assert section.text.startswith(excerpt)

    def test_results_are_ordered_by_descending_score(self, retriever):
        result = retriever.search("fraud escalation tiers and containment steps")
        scores = [e.score for e in result.evidence]
        assert scores == sorted(scores, reverse=True)

    def test_respects_the_result_limit(self, retriever):
        result = retriever.search("customer policy approval review")
        assert len(result.evidence) <= 4
        assert len(retriever.search("customer policy approval review", limit=1).evidence) == 1

    def test_spreads_citations_across_documents(self, retriever):
        # A cross-cutting question should not be answered from one document alone.
        result = retriever.search("what approval is required before closing an account")
        assert len(set(result.policy_ids)) >= 2

    def test_finds_the_consequential_action_sections(self, retriever):
        result = retriever.search("consequential actions requiring human approval authority")
        assert any("Consequential actions" in e.section for e in result.evidence)


class TestAbstention:
    def test_abstains_on_a_question_the_corpus_does_not_cover(self, retriever):
        result = retriever.search("What is the weather forecast in Sydney tomorrow?")
        assert result.sufficient is False
        assert result.evidence == []
        assert result.abstain_reason is AbstainReason.NO_RELEVANT_POLICY
        assert result.explanation

    def test_abstains_rather_than_returning_weak_evidence(self, corpus):
        # With the threshold raised, a partially-matching question must abstain
        # instead of handing back low-confidence text to ground an answer on.
        strict = KeywordPolicyRetriever(corpus, min_score=0.95)
        result = strict.search("What is the tax treatment of a written-off disputed amount?")
        assert result.sufficient is False
        assert result.evidence == []
        assert result.abstain_reason is AbstainReason.INSUFFICIENT_EVIDENCE
        assert "threshold" in result.explanation

    def test_abstains_on_a_query_with_no_searchable_terms(self, retriever):
        result = retriever.search("what is it")
        assert result.sufficient is False
        assert result.evidence == []
        assert result.explanation

    def test_reports_the_threshold_it_applied(self, retriever):
        result = retriever.search("Sydney weather forecast")
        assert result.threshold == pytest.approx(0.15)
        assert result.best_score < result.threshold


class TestDeterminism:
    def test_the_same_query_gives_the_same_result(self, corpus):
        a = KeywordPolicyRetriever(corpus).search("fee waiver approval")
        b = KeywordPolicyRetriever(corpus).search("fee waiver approval")
        assert a.model_dump() == b.model_dump()

    def test_corpus_order_does_not_change_results(self, corpus):
        forward = KeywordPolicyRetriever(corpus).search("privacy access request timeframe")
        reversed_ = KeywordPolicyRetriever(list(reversed(corpus))).search(
            "privacy access request timeframe"
        )
        assert forward.model_dump() == reversed_.model_dump()


class TestRetrieverInterface:
    def test_keyword_retriever_satisfies_the_protocol(self, retriever):
        assert isinstance(retriever, PolicyRetriever)

    def test_an_alternative_backend_can_be_substituted(self):
        # Stands in for a future enterprise search or vector implementation: the
        # rest of the system depends only on this protocol.
        class FakeVectorRetriever:
            def search(self, query: str, *, limit: int | None = None) -> RetrievalResult:
                return RetrievalResult(
                    query=query,
                    sufficient=False,
                    best_score=0.0,
                    threshold=0.5,
                    abstain_reason=AbstainReason.NO_RELEVANT_POLICY,
                    explanation="stub backend",
                )

        backend: PolicyRetriever = FakeVectorRetriever()
        assert isinstance(backend, PolicyRetriever)
        assert backend.search("anything").sufficient is False
