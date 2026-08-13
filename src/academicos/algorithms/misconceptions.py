"""P1.10 — Misconceptions catalog + detection.

A misconception is a persistent, internally-coherent belief that disagrees
with the domain model — not a slip. Inventories like the Force Concept
Inventory (Hestenes, Wells & Swackhamer 1992) and fraction/decimals research
(a 2024 systematic review names 78 recurring grade 3-6 mathematical
misconceptions) show these are stable across learners and highly predictable
from a short answer or distractor choice.

This module ships a **research-shaped template catalog** (short, citable,
curriculum-frequency-weighted) plus deterministic inference:

  * `for_concept(label)` — misconceptions whose concept tokens overlap the
    concept label (any token match).
  * `match_answer(label, answer)` — given the concept and the learner's
    written/diagnostic answer, returns scored misconception hits.
  * `merge(profiled, hits)` — merges LLM-extracted misconceptions
    (concept-profile) with the catalog, deduped by belief tokens.
  * `profile_concept_attr(attrs, label)` — node-attribute enrich used when
    concept-profile/rule runs.

Templates are plain data (pure functions, tests deterministic). The catalog is
deliberately small and citable; it is *not* a replacement for LLM-extraction —
it is the deterministic floor that works offline, plus the answer-diagnosis
signal for misconception-tagged interactions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Misconception:
    id: str
    name: str                      # short belief label
    belief: str                    # one-line articulation of the wrong model
    concept_tokens: frozenset[str] # glossary tokens this applies to
    indicators: frozenset[str]     # tokens in a learner answer that flag it
    source: str                    # citation-ish origin
    severity: float = 0.6


CATALOG: tuple[Misconception, ...] = (
    # ---- Fractions & numbers ----
    Misconception(
        id="frac-whole-number",
        name="whole-number thinking on fractions",
        belief="Treats numerator and denominator as two separate whole numbers "
               "(e.g. calls 3/4 bigger than 2/3 just because 3 > 2).",
        concept_tokens=frozenset({"fraction", "fractions", "denominator", "numerator"}),
        indicators=frozenset({"bigger", "greater", "smaller", "than", "compare"}),
        source="whole-number bias (Ni & Zhou 2005; systematic review 2024)",
    ),
    Misconception(
        id="frac-add-top-bottom",
        name="adds top and bottom numbers",
        belief="Adds fractions by summing numerators and denominators separately "
               "(e.g. 1/2 + 1/2 = 2/4).",
        concept_tokens=frozenset({"fraction", "fractions", "add", "addition", "denominator"}),
        indicators=frozenset({"add", "sum", "plus", "top", "bottom"}),
        source="systematic review: procedural fraction errors (2024)",
    ),
    Misconception(
        id="frac-larger-denominator",
        name="larger denominator = bigger fraction",
        belief="Equates the size of a fraction with its denominator alone, "
               "ignoring the numerator.",
        concept_tokens=frozenset({"fraction", "fractions", "compare", "comparing", "denominator"}),
        indicators=frozenset({"bigger", "larger", "denominator", "smaller"}),
        source="fraction concept inventory (ERIC EJ1401948)",
    ),
    Misconception(
        id="dec-longer-is-larger",
        name="longer-is-larger decimals",
        belief="Treats a decimal with more digits as larger regardless of place "
               "value (e.g. 0.180 > 0.2).",
        concept_tokens=frozenset({"decimal", "decimals", "place", "value"}),
        indicators=frozenset({"longer", "more", "digits", "decimal"}),
        source="longer-is-larger decimals (~40% Y5, ≤8% Y10)",
    ),
    Misconception(
        id="frac-decimal-unrelated",
        name="fractions and decimals unrelated",
        belief="Believes fractions and decimals are two different kinds of number "
               "rather than two representations of one value.",
        concept_tokens=frozenset({"fraction", "fractions", "decimal", "decimals"}),
        indicators=frozenset({"different", "not", "same"}),
        source="STEM learn misconceptions; DMTI practice review 2025",
    ),
    Misconception(
        id="percent-not-fraction",
        name="percent as raw points",
        belief="Treats a percent as a bare number of points instead of a fraction "
               "of a whole (e.g. ’she scored 25 percent’ = 25 marks).",
        concept_tokens=frozenset({"percentage", "percent", "percentages"}),
        indicators=frozenset({"percent", "marks", "points", "score"}),
        source="systematic review: percentages (2024)",
    ),
    # ---- Mechanics (Force Concept Inventory) ----
    Misconception(
        id="fci-impetus",
        name="impetus: motion needs a force in the body",
        belief="Believes motion itself implies a force ’stored in’ the object that "
               "keeps it going.",
        concept_tokens=frozenset({"force", "motion", "momentum", "inertia", "newton"}),
        indicators=frozenset({"force", "keeping", "motion", "momentum", "stored"}),
        source="FCI: impetus fallacy (Hestenes, Wells & Swackhamer 1992)",
    ),
    Misconception(
        id="fci-constant-speed-force",
        name="constant speed needs constant force",
        belief="Thinks a constant force yields constant speed rather than constant "
               "acceleration.",
        concept_tokens=frozenset({"force", "acceleration", "velocity", "speed", "newton"}),
        indicators=frozenset({"speed", "force", "constant", "velocity"}),
        source="FCI: Newton-2 intuition (1992)",
    ),
    Misconception(
        id="fci-mass-gravity",
        name="heavier objects fall faster",
        belief="Believes heavier or denser objects fall faster; conflates mass with "
               "weight and weight with the pull of gravity.",
        concept_tokens=frozenset({"gravity", "weight", "mass", "free", "fall", "newton"}),
        indicators=frozenset({"heavier", "faster", "fall", "weight", "mass"}),
        source="FCI: gravity dimension (1992)",
    ),
    Misconception(
        id="fci-rest-no-force",
        name="at rest means no force acts",
        belief="Believes an object at rest has no forces acting on it, overlooking "
               "balanced forces (normal, weight).",
        concept_tokens=frozenset({"force", "rest", "equilibrium", "balance", "newton"}),
        indicators=frozenset({"rest", "no force", "still", "balanced", "at rest"}),
        source="FCI: reaction / Newton-3 intuition (1992)",
    ),
)

# fast index: token -> misconception ids
_TOKEN_INDEX: dict[str, list[str]] = {}
for _m in CATALOG:
    for _t in _m.concept_tokens:
        _TOKEN_INDEX.setdefault(_t, []).append(_m.id)
_ID_LOOKUP: dict[str, Misconception] = {m.id: m for m in CATALOG}


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1)


def for_concept(label: str) -> list[Misconception]:
    """Misconceptions whose concept tokens overlap the concept label (any)."""
    tl = _tokens(label)
    if not tl:
        return []
    ids: set[str] = set()
    for t in tl:
        ids.update(_TOKEN_INDEX.get(t, ()))
    return [_ID_LOOKUP[i] for i in sorted(ids) if i in _ID_LOOKUP]


def match_answer(label: str, answer: str) -> list[tuple[Misconception, float]]:
    """Score catalog misconceptions against a free-text answer.

    Score = indicator-token overlap normalised 0..1 by the misconception's
    indicator count; only >0 returns. Deterministic + offline.
    """
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return []
    scored: list[tuple[Misconception, float]] = []
    for m in for_concept(label):
        hit = len(answer_tokens & m.indicators)
        if hit:
            score = round(min(1.0, hit / max(1, len(m.indicators))) * m.severity, 4)
            scored.append((m, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored


def generic_signals(answer: str) -> list[str]:
    """Safe, low-precision red flags — no catalog required."""
    a = answer.lower()
    out = []
    if re.search(r"\b(am|is|are|do|does|can|will|could|should)\s+not\b", a):
        out.append("negation")
    if re.search(r"\b\d+\s*[+x×*]\s*\d+\s*=\s*\d+\b", a):
        out.append("arithmetic-recalc")
    if re.search(r"\b(not sure|don't know|do not know|guessing)\b", a):
        out.append("uncertain")
    return out


def merge(profiled: list[str], catalog_hits: list[Misconception]) -> list[str]:
    """Deduped union: node-profiled misconceptions + catalog beliefs.
    A catalog belief is skipped when it substantially overlaps (Jaccard ≥ .4)
    an already-present profiled string."""
    out = list(profiled)
    seen = [_tokens(b) for b in profiled]
    for m in catalog_hits:
        mt = _tokens(m.belief)
        if not mt:
            continue
        dup = any(len(mt & pt) / max(1, len(mt | pt)) >= 0.4 for pt in seen)
        if not dup:
            out.append(m.belief)
            seen.append(mt)
    return out


def enrich_attributes(attrs: dict[str, Any], node_label: str) -> dict[str, Any]:
    """Merge catalog misconceptions into a node's attribute dict (in place
    friendly: returns a new dict). Powering concept-profile/rule enrichment."""
    a = dict(attrs)
    known = list(a.get("misconceptions", [])) or []
    hits = for_concept(node_label)
    a["misconceptions"] = merge(known, hits)
    a["misconception_ids"] = sorted({m.id for m in hits})
    return a