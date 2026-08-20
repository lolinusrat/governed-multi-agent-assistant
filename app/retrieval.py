"""Local retrieval over the synthetic Markdown policy corpus.

Deliberately a keyword scorer over section-sized chunks: no database, no vector
store, no network call. It is deterministic, which means the abstention decision
that depends on it can be unit-tested rather than observed.

Everything outside this module depends on the `PolicyRetriever` protocol, so an
enterprise search or vector implementation can be dropped in by satisfying
`search()` and returning the same `RetrievalResult`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.contracts import AbstainReason, PolicyEvidence, RetrievalResult

# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_FM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_SECTION_HEADING = re.compile(r"^#\s+(\d+)\.\s+(.+?)\s*$", re.M)

MAX_EXCERPT_CHARS = 700


class PolicySection(BaseModel):
    """One numbered section of a policy document."""

    model_config = ConfigDict(frozen=True)

    number: str
    heading: str
    text: str

    @property
    def label(self) -> str:
        """Citable section label, e.g. '5. Consequential actions requiring human approval'."""
        return f"{self.number}. {self.heading}"


class PolicyDocument(BaseModel):
    """A parsed policy document: front-matter metadata plus numbered sections."""

    model_config = ConfigDict(frozen=True)

    policy_id: str
    title: str
    version: str = ""
    status: str = ""
    effective_date: str = ""
    classification: str = ""
    owner: str = ""
    source_path: str = ""
    sections: tuple[PolicySection, ...] = ()


class PolicyCorpusError(RuntimeError):
    """Raised when the corpus is missing or a document cannot be parsed."""


def parse_policy_document(path: Path) -> PolicyDocument:
    """Parse one Markdown policy file into a `PolicyDocument`."""
    raw = path.read_text(encoding="utf-8")

    fm_match = _FRONT_MATTER.match(raw)
    if fm_match is None:
        raise PolicyCorpusError(f"{path.name}: missing YAML front matter")

    meta: dict[str, str] = {}
    for line in fm_match.group(1).splitlines():
        if (m := _FM_LINE.match(line.strip())) is not None:
            meta[m.group(1)] = m.group(2).strip().strip("'\"")

    for required in ("policy_id", "title"):
        if not meta.get(required):
            raise PolicyCorpusError(f"{path.name}: front matter is missing '{required}'")

    body = raw[fm_match.end() :]
    headings = list(_SECTION_HEADING.finditer(body))
    sections = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[heading.end() : end].strip()
        if text:
            sections.append(
                PolicySection(number=heading.group(1), heading=heading.group(2), text=text)
            )

    if not sections:
        raise PolicyCorpusError(f"{path.name}: no numbered sections found")

    return PolicyDocument(
        policy_id=meta["policy_id"],
        title=meta["title"],
        version=meta.get("version", ""),
        status=meta.get("status", ""),
        effective_date=meta.get("effective_date", ""),
        classification=meta.get("classification", ""),
        owner=meta.get("owner", ""),
        source_path=str(path),
        sections=tuple(sections),
    )


def load_corpus(policy_dir: Path) -> list[PolicyDocument]:
    """Load every Markdown policy in `policy_dir`, sorted by filename for determinism."""
    if not policy_dir.is_dir():
        raise PolicyCorpusError(f"policy directory not found: {policy_dir}")

    docs = [parse_policy_document(p) for p in sorted(policy_dir.glob("*.md"))]
    if not docs:
        raise PolicyCorpusError(f"no policy documents found in {policy_dir}")
    return docs


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_STOPWORDS = frozenset(
    """a an and any are as at be by can could do does for from has have how i if in into is it
    its may must not of on or our shall should so than that the their them then there these this
    to under up was we what when where which who why will with would you your""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")

# BM25 term-frequency parameters. `k1` saturates repeated terms so a long section
# cannot win on repetition alone; `b` normalises by section length so a four-line
# section that answers the question can outrank a long procedural one that merely
# mentions the same words. Both are the standard defaults.
_K1 = 1.2
_B = 0.75
_HEADING_BOOST = 1.5


def stem(token: str) -> str:
    """Reduce a word to a crude stem so inflections of it match each other.

    Not a real stemmer, and it does not need to be. It exists so that a question
    asking to "close" an account matches a policy that says "closing", and one
    asking to "lodge" a dispute matches "lodged". Without it the retriever misses
    the section holding the answer while happily returning the one that repeats
    the question's easy words.

    Verb and plural endings are stripped, then a trailing "e", so close/closed/
    closing all reduce to "clos". Words ending "ss" are left alone.
    """
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ss"):
        return token
    if token.endswith("s"):
        token = token[:-1]
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
    elif token.endswith("ed") and len(token) > 4:
        token = token[:-2]
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords, and stem what is left."""
    return [
        stem(token)
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS and (len(token) >= 3 or token.isdigit())
    ]


def _excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Trim a section to a citable excerpt on a line boundary, verbatim."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind("\n\n"), cut.rfind("\n"))
    if boundary > limit // 3:
        cut = cut[:boundary]
    return cut.rstrip() + " […]"


class _Chunk:
    """A searchable unit: one section, with its parent document's metadata."""

    __slots__ = ("doc", "section", "counts", "heading_terms", "length")

    def __init__(self, doc: PolicyDocument, section: PolicySection) -> None:
        self.doc = doc
        self.section = section
        self.counts = Counter(tokenize(section.text))
        self.heading_terms = set(tokenize(f"{doc.title} {section.heading}"))
        self.length = sum(self.counts.values())


@runtime_checkable
class PolicyRetriever(Protocol):
    """The seam a production search backend would implement.

    A replacement needs to satisfy this one method and return the same
    `RetrievalResult`, including the explicit insufficient-evidence case.
    """

    def search(self, query: str, *, limit: int | None = None) -> RetrievalResult: ...


class KeywordPolicyRetriever:
    """IDF-weighted keyword retrieval over policy sections.

    Scores are normalised to 0-1 as the share of the query's information content
    matched by a section, so `min_score` reads as a coverage threshold rather than
    an arbitrary cutoff.
    """

    def __init__(
        self,
        documents: list[PolicyDocument],
        *,
        min_score: float = 0.15,
        max_results: int = 6,
        max_per_policy: int = 3,
    ) -> None:
        self.documents = documents
        self.min_score = min_score
        self.max_results = max_results
        self.max_per_policy = max_per_policy
        self._chunks = [_Chunk(doc, s) for doc in documents for s in doc.sections]
        self._avg_length = (
            sum(c.length for c in self._chunks) / len(self._chunks) if self._chunks else 1.0
        ) or 1.0
        self._idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        n = len(self._chunks)
        df = Counter(term for chunk in self._chunks for term in chunk.counts)
        return {term: math.log(1.0 + n / (1.0 + count)) for term, count in df.items()}

    def _idf_of(self, term: str) -> float:
        # An unseen term keeps its full weight in the denominator and contributes
        # nothing to the numerator, so unanswerable queries score toward zero.
        return self._idf.get(term, math.log(1.0 + len(self._chunks)))

    def _score(self, chunk: _Chunk, query_terms: list[str], budget: float) -> float:
        """Share of the query's information content this section matches, 0-1.

        A term unseen anywhere in the corpus contributes nothing to the numerator
        but keeps its full weight in `budget`, so a question the policies do not
        cover scores toward zero and the abstention threshold does the rest.
        """
        if budget <= 0:
            return 0.0
        norm = 1.0 - _B + _B * (chunk.length / self._avg_length)
        earned = 0.0
        for term in query_terms:
            tf = chunk.counts.get(term, 0)
            if tf == 0:
                continue
            # BM25 saturation, rescaled to 0-1 so `earned / budget` stays a share.
            weight = (tf * (_K1 + 1.0)) / (tf + _K1 * norm) / (_K1 + 1.0)
            if term in chunk.heading_terms:
                weight = min(1.0, weight * _HEADING_BOOST)
            earned += self._idf_of(term) * weight
        return min(1.0, earned / budget)

    def search(self, query: str, *, limit: int | None = None) -> RetrievalResult:
        limit = limit or self.max_results
        query_terms = list(dict.fromkeys(tokenize(query)))
        budget = sum(self._idf_of(t) for t in query_terms)

        scored = sorted(
            ((self._score(c, query_terms, budget), c) for c in self._chunks),
            key=lambda pair: (-pair[0], pair[1].doc.policy_id, pair[1].section.number),
        )
        best_score = scored[0][0] if scored else 0.0

        if not query_terms or best_score < self.min_score:
            return RetrievalResult(
                query=query,
                sufficient=False,
                evidence=[],
                best_score=best_score,
                threshold=self.min_score,
                abstain_reason=(
                    AbstainReason.NO_RELEVANT_POLICY
                    if best_score <= 0.0
                    else AbstainReason.INSUFFICIENT_EVIDENCE
                ),
                explanation=self._abstain_explanation(query_terms, best_score, scored),
            )

        evidence: list[PolicyEvidence] = []
        per_policy: Counter[str] = Counter()
        for score, chunk in scored:
            if score < self.min_score or len(evidence) >= limit:
                break
            if per_policy[chunk.doc.policy_id] >= self.max_per_policy:
                continue
            per_policy[chunk.doc.policy_id] += 1
            evidence.append(
                PolicyEvidence(
                    policy_id=chunk.doc.policy_id,
                    title=chunk.doc.title,
                    section=chunk.section.label,
                    text=_excerpt(chunk.section.text),
                    score=round(score, 4),
                    source_path=chunk.doc.source_path,
                )
            )

        return RetrievalResult(
            query=query,
            sufficient=True,
            evidence=evidence,
            best_score=round(best_score, 4),
            threshold=self.min_score,
            explanation=f"Matched {len(evidence)} section(s) at or above the {self.min_score} threshold.",
        )

    def _abstain_explanation(
        self, query_terms: list[str], best_score: float, scored: list[tuple[float, _Chunk]]
    ) -> str:
        if not query_terms:
            return "The question contained no searchable terms."
        if best_score <= 0.0:
            return (
                "No section of the policy corpus matched this question. "
                "The corpus does not cover it, so no answer is given."
            )
        near = scored[0][1]
        return (
            f"The closest match was {near.doc.policy_id} section {near.section.label!r} "
            f"at {best_score:.2f}, below the {self.min_score} evidence threshold. "
            "Answering would require going beyond what the policies state."
        )


def build_retriever(settings: Settings | None = None) -> PolicyRetriever:
    """Construct the configured retriever. The composition point for a swap."""
    settings = settings or get_settings()
    return KeywordPolicyRetriever(
        load_corpus(settings.policy_path),
        min_score=settings.retrieval_min_score,
        max_results=settings.retrieval_max_results,
    )


@lru_cache
def get_retriever() -> PolicyRetriever:
    """Process-wide retriever, built once at first use."""
    return build_retriever()
