"""Evaluation metrics: retrieval quality, extraction fidelity, answer groundedness.

Answer-quality metrics (rouge_l, bleu1, token_f1, exact_match) are adapted
(dependency-free reimplementation) from praj2408/RAG-Enhanced-NCERT-Tutor
(MIT License), evaluation_script.py — retrieval metrics there are superseded
by the versions above.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any


def precision_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    hits = sum(1 for i, r in enumerate(retrieved[:k]) if r in relevant)
    return hits / k if k else 0.0


def recall_at_k(relevant: set[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr(relevant: set[str], retrieved: Sequence[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: dict[str, float], retrieved: Sequence[str], k: int) -> float:
    dcg = sum(relevant.get(r, 0.0) / math.log2(i + 1) for i, r in enumerate(retrieved[:k], start=1))
    ideal = sum(sorted(relevant.values(), reverse=True)[:k] and
                [v / math.log2(i + 1) for i, v in enumerate(sorted(relevant.values(), reverse=True)[:k], start=1)])
    return dcg / ideal if ideal else 0.0


def extraction_accuracy(gold: Sequence[dict[str, Any]], predicted: Sequence[dict[str, Any]],
                        keys: Sequence[str] = ("marks", "q_no")) -> float:
    """Fraction of exact matches on key fields between gold and predicted objects."""
    if not gold:
        return 0.0
    gold_by_id = {g.get("canonical_id"): g for g in gold}
    score = 0.0
    for p in predicted:
        g = gold_by_id.get(p.get("canonical_id"))
        if not g:
            continue
        if all(g.get(k) == p.get(k) for k in keys):
            score += 1.0
    return score / len(gold)


def answer_groundedness(answer: str, evidence_texts: Sequence[str]) -> float:
    """Fraction of answer sentences that appear verbatim (or near-verbatim)
    in the cited evidence. Strong grounding check for generated answers."""
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 4]
    if not sentences:
        return 0.0
    joined = " ".join(e.lower() for e in evidence_texts)
    grounded = sum(1 for s in sentences if _near_substring(s, joined))
    return grounded / len(sentences)


def _near_substring(sentence: str, joined_lower: str, min_len: int = 8) -> bool:
    norm = " ".join(sentence.lower().split())
    if len(norm) < min_len:
        return True  # too short to judge; count as grounded
    return norm in joined_lower


# ---- answer-quality metrics (adapted from RAG-Enhanced-NCERT-Tutor, MIT) ----

def _tokens(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, candidate: str) -> float:
    """Token-level F1 (same formulation as the tutor's compute_f1)."""
    ref = Counter(_tokens(reference))
    cand = Counter(_tokens(candidate))
    if not ref and not cand:
        return 1.0
    common = sum((ref & cand).values())
    p = common / sum(cand.values()) if cand else 0.0
    r = common / sum(ref.values()) if ref else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def exact_match(reference: str, candidate: str) -> float:
    """1.0 if identical after case-folding + whitespace collapse."""
    norm = lambda s: " ".join(s.strip().lower().split())
    return 1.0 if norm(reference) == norm(candidate) else 0.0


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L F-measure on tokens, computed via LCS (SequenceMatcher)."""
    import difflib
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return 0.0
    sm = difflib.SequenceMatcher(None, ref, cand, autojunk=False)
    lcs = sum(b.size for b in sm.get_matching_blocks())
    p = lcs / len(cand)
    r = lcs / len(ref)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def bleu1(reference: str, candidate: str) -> float:
    """BLEU-1: unigram precision with the standard brevity penalty."""
    ref, cand = _tokens(reference), _tokens(candidate)
    if not cand:
        return 0.0
    if not ref:
        return 1.0
    p = sum((Counter(ref) & Counter(cand)).values()) / len(cand)
    if p == 0.0:
        return 0.0
    bp = math.exp(1 - len(ref) / len(cand)) if len(cand) < len(ref) else 1.0
    return bp * p
