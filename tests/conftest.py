"""Shared fixtures. Every test runs offline against the real synthetic corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval import KeywordPolicyRetriever, PolicyDocument, load_corpus

POLICY_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def corpus() -> list[PolicyDocument]:
    return load_corpus(POLICY_DIR)


@pytest.fixture()
def retriever(corpus: list[PolicyDocument]) -> KeywordPolicyRetriever:
    return KeywordPolicyRetriever(corpus, min_score=0.15)
