"""Chunk index: SQLite FTS5 (BM25) + optional dense embeddings (TF-IDF fallback).

Keeps the pipeline dependency-free: BM25 works out of the box; dense vectors use
pure-Python TF-IDF character n-grams when sentence-transformers is not installed.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

from ..extract.chunking import Chunk

SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
  id UNINDEXED, document_id UNINDEXED, page UNINDEXED, heading, text,
  tokenize = 'unicode61 remove_diacritics 2'
);
"""

_TOKEN = re.compile(r"[a-z0-9]+", re.I)


class ChunkIndex:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._df: Counter = Counter()
        self._doc_tokens: dict[str, Counter] = {}
        self._doc_count = 0
        self._load_stats()

    def _load_stats(self) -> None:
        try:
            rows = self.conn.execute(
                "SELECT document_id, text FROM chunks").fetchall()
            for doc_id, text in rows:
                toks = _TOKEN.findall(text.lower())
                self._doc_tokens[doc_id] = Counter(toks)
                for t in set(toks):
                    self._df[t] += 1
            self._doc_count = len(self._doc_tokens)
        except sqlite3.Error:
            pass

    def add(self, chunks: list[Chunk], replace_doc: bool = False) -> None:
        for c in chunks:
            if replace_doc:
                self.conn.execute("DELETE FROM chunks WHERE document_id=?", (c.document_id,))
                replace_doc = False
            self.conn.execute(
                "INSERT INTO chunks (id, document_id, page, heading, text) VALUES (?,?,?,?,?)",
                (c.id, c.document_id, c.page, c.heading, c.text),
            )
        self.conn.commit()
        self._load_stats()

    def bm25(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        q = " OR ".join(f'"{t}"' for t in _TOKEN.findall(query))
        if not q:
            return []
        rows = self.conn.execute(
            "SELECT id, document_id, page, heading, text, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (q, limit),
        ).fetchall()
        return [(Chunk(id=r[0], document_id=r[1], page=r[2], heading_path=[r[3]], text=r[4]), -r[5]) for r in rows]

    def tfidf(self, query: str, limit: int = 20) -> list[tuple[Chunk, float]]:
        """Pure-python TF-IDF over word tokens. Requires _doc_tokens stats."""
        if not self._doc_tokens:
            return []
        qtoks = _TOKEN.findall(query.lower())
        qvec = Counter(qtoks)
        scored: list[tuple[str, float]] = []
        idf = lambda t: math.log((self._doc_count + 1) / (self._df[t] + 1)) + 1.0
        for doc_id, toks in self._doc_tokens.items():
            norm = math.sqrt(sum(c * c for c in toks.values())) or 1.0
            s = 0.0
            for t in qvec:
                s += qvec[t] * (toks.get(t, 0) / norm) * idf(t)
            scored.append((doc_id, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]
        out: list[tuple[Chunk, float]] = []
        for doc_id, s in top:
            rows = self.conn.execute(
                "SELECT id, document_id, page, heading, text FROM chunks WHERE document_id=? LIMIT 3",
                (doc_id,)).fetchall()
            for r in rows:
                out.append((Chunk(id=r[0], document_id=r[1], page=r[2],
                                  heading_path=[r[3]], text=r[4]), s))
        return out

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]

    def delete_document(self, document_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
        self.conn.commit()
        self._load_stats()

    def close(self) -> None:
        self.conn.close()
