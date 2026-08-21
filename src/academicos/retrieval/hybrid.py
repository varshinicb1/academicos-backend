"""Hybrid retrieval: BM25 + graph-kg boosts + (optional) dense rerank.

Design notes:
  * Dual-pathway: chunks win on recall; graph traversals (chapter/prereq chains,
    question->marking-point links) win on structure; hybrid merge with RRF.
  * Rerank is a pluggable scorer; default is a lightweight lexical prior +
    heading-aware boost (title/heading matches outrank body matches).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..extract.chunking import Chunk
from ..graph.store import GraphStore
from .index import ChunkIndex


@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    sources: list[str] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)

    @property
    def document_id(self) -> str:
        return self.chunk.document_id


def rrf(*rankings: list[str], k: int = 60) -> dict[str, float]:
    """Reciprocal-rank fusion over lists of chunk ids."""
    scores: dict[str, float] = {}
    for rank in rankings:
        for i, cid in enumerate(rank):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


class HybridRetriever:
    def __init__(self, index: ChunkIndex, graph: GraphStore | None = None):
        self.index = index
        self.graph = graph

    def search(self, query: str, limit: int = 12, boost_keywords: list[str] | None = None,
               dense: bool = True, subgraph: bool = True) -> list[RetrievalHit]:
        bm25 = self.index.bm25(query, limit=limit * 2)
        dense_hits = self.index.tfidf(query, limit=limit * 2) if dense else []

        bm25_ids = [c.id for c, _ in bm25]
        dense_ids = [c.id for c, _ in dense_hits]
        fused = rrf(bm25_ids, dense_ids)

        chunk_by_id = {c.id: c for c, _ in bm25}
        for c, _ in dense_hits:
            chunk_by_id.setdefault(c.id, c)

        boosted = boost_keywords or _auto_keywords(query)
        bm25_set = set(bm25_ids)
        dense_set = set(dense_ids)
        results: list[RetrievalHit] = []
        for cid, score in sorted(fused.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_by_id.get(cid)
            if not chunk:
                continue
            s = score
            low_text = (chunk.heading + " " + chunk.text).lower()
            for kw in boosted:
                if kw in low_text:
                    s += 0.15
            sources = []
            if cid in bm25_set:
                sources.append("bm25")
            if cid in dense_set:
                sources.append("dense")
            results.append(RetrievalHit(chunk=chunk, score=s, sources=sources))

        if self.graph and results:
            self._graph_boost(results)

        if self.graph and subgraph:
            # HippoRAG-style: concept-anchored sub-graph hits act as evidence seeds
            sg = self._subgraph_hits(query)
            if sg:
                for h in results[:limit]:
                    h.graph_nodes = h.graph_nodes + sg["concepts"]
                    h.score += sg["score"] if sg["concepts"] else 0.0
                    h.sources.append("subgraph")

        return sorted(results, key=lambda r: r.score, reverse=True)[:limit]

    def reasoning_paths(self, query: str, target_node: str | None = None,
                        max_depth: int = 4) -> list[list[str]]:
        """HyKGE-style: return edge-type reasoning paths from concept anchors."""
        if not self.graph:
            return []
        from ..graph.query import subgraph_for_query

        sg = subgraph_for_query(self.graph, query, max_depth=2)
        paths: list[list[str]] = []
        if target_node:
            for c in sg["concepts"]:
                nodes = self.graph.query_nodes(label_contains=c, limit=1)
                if not nodes:
                    continue
                for p in self.graph.paths(nodes[0].id, target_node, max_depth=max_depth):
                    paths.append([e.type.value for e in p])
        else:
            paths = sg["paths"]
        return paths

    def _subgraph_hits(self, query: str) -> dict | None:
        from ..graph.query import subgraph_for_query

        sg = subgraph_for_query(self.graph, query, max_depth=2)
        if not sg["concepts"]:
            return None
        score = min(0.2, 0.05 * len(sg["concepts"]))
        return {"concepts": sg["concepts"], "score": score}

    def _graph_boost(self, hits: list[RetrievalHit]) -> None:
        for h in hits:
            docs = self.graph.query_nodes(label_contains=h.chunk.document_id[:8], limit=1)
            if docs:
                nb = self.graph.neighbors(docs[0].id, max_depth=1, max_nodes=20)
                if nb:
                    h.score += 0.1
                    h.graph_nodes = [x["node"].id for x in nb[:5]]


def _auto_keywords(query: str) -> list[str]:
    from .index import _TOKEN
    toks = _TOKEN.findall(query.lower())
    stop = {"the", "a", "an", "of", "in", "for", "on", "what", "how", "why", "is", "are",
            "to", "and", "or", "explain", "define", "list", "state", "describe"}
    return [t for t in toks if t not in stop][:5]
