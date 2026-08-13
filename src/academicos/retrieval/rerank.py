"""Reranking: pluggable scorers over candidate hits.

Default: heading-prior + query-term proximity. Optional: cross-encoder via
sentence-transformers when the `retrieval` extra is installed.
"""
from __future__ import annotations

from typing import Callable, Optional

from .hybrid import RetrievalHit

Reranker = Callable[[str, list[RetrievalHit]], list[RetrievalHit]]


def heading_prior(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    qwords = {w for w in query.lower().split() if len(w) > 3}
    for h in hits:
        head_toks = set(h.chunk.heading.lower().split())
        h.score += 0.1 * len(qwords & head_toks)
    return sorted(hits, key=lambda r: r.score, reverse=True)


def cross_encoder(limit: int = 5) -> Reranker:
    """Lazy cross-encoder reranker (needs sentence-transformers extra)."""
    _model = {}

    def _rerank(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        try:
            if "m" not in _model:
                from sentence_transformers import CrossEncoder
                _model["m"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, h.chunk.text[:512]) for h in hits]
            scores = _model["m"].predict(pairs)
            for h, s in zip(hits, scores):
                h.score += float(s)
            return sorted(hits, key=lambda r: r.score, reverse=True)
        except ImportError:
            return heading_prior(query, hits)

    return _rerank


def get_reranker(dense: bool = False) -> Reranker:
    return cross_encoder() if dense else heading_prior
