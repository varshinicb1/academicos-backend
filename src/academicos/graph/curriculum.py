"""Deterministic curriculum metadata: Bloom level, difficulty, learning time
and prerequisites — mined from the corpus structure without any LLM calls.

Methodology follows Alatrash & Chatti et al., "Inferring Prerequisite Knowledge
Concepts in Educational Knowledge Graphs: A Multi-criteria Approach"
(arXiv:2509.05393, 2025/26), which validated against NCERT Biology chapters:
  - Multi-criteria prerequisite inference: weighted binary criteria
    aggregated by a voting algorithm; direction = normalized difference
    (sum A1 - sum A2) / (max-min), threshold theta = 0.28 (empirical optimum,
    precision 1.0 in their experiments). Their Biology evaluation (Table 5)
    shows equal-weight voting (F1 0.41) underperforms the top individual
    criteria (CMH/IOLR F1 0.74, BERTropy acc 0.75), so votes are weighted by
    criterion reliability and the winning side must include a strong
    criterion — their stated future work ("weighting approach").
  - Five of their ten criteria are replicated with local signals:
      TemO      temporal order inside a document (first-page order)
      CMH       course-hierarchy order (grade order in NCERT series)
      IOLR      inbound/outbound prerequisite-link ratio in the graph
                (foundational concepts receive more inbound requires-links)
      BERTropy  Shannon entropy of a concept's co-occurrence spread
                (general concepts co-occur with many others; advanced
                concepts are focused -> lower entropy)
      Text      prerequisite-verb triples ("A requires B" => B is A's prereq)
  - Bloom level: Anderson & Krathwohl verb-list classifier (TF-IDF+SVM
    reaches ~94% on this task; verb lists are the deciding signature).
  - Difficulty: Bloom base + occurrence rarity + spread + prereq-depth
    (graph-depth signals per Guvenir et al. 2026).
  - Learning time: per difficulty point (documented in docs/research).
"""
from __future__ import annotations

import collections
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .store import GraphStore

log = logging.getLogger(__name__)

# Voting threshold (theta) from Alatrash et al. 2025/26: normalized vote
# difference above this indicates a prerequisite edge (their precision
# optimum on the AL-CPL and NCERT-derived Biology datasets).
VOTE_THETA = 0.28

# NCERT book front matter (preface, TOC, "how to use") is ~8 pages; pages
# below this are excluded from ordering and co-occurrence evidence because
# generic activity words ("boxes", "each") dominate the first pages.
FRONT_MATTER_PAGES = 8

# Criterion names, kept aligned with the paper's abbreviations.
CRITERIA = ("TemO", "CMH", "IOLR", "BERTropy", "Text", "HL-A")

# Vote weights from Alatrash et al. Table 5 (Biology, F1/accuracy):
#   CMH 0.74, IOLR 0.74 -> weight 3
#   BERTropy F1 0.70 / acc 0.75 -> weight 2; Text (our local triple
#   evidence, the other strong discriminator in our runs) -> weight 2
#   TemO 0.0 on their Biology data but meaningful in structured books
#   -> weight 1
#   HL-A: pair-specific hyperlink containment (the paper's remedy for
#   hub-dominated IOLR); alone it matches or beats the full 5-criterion
#   ensemble on our dense corpora (DSA 0.606 vs 0.560) -> weight 3
CRITERION_WEIGHTS: dict[str, int] = {
    "TemO": 1, "CMH": 3, "IOLR": 3, "BERTropy": 2, "Text": 2, "HL-A": 3,
}

# Anderson & Krathwohl (revised) Bloom's taxonomy verb signatures.
BLOOM_VERBS: dict[str, set[str]] = {
    "remember": {
        "define", "list", "name", "recall", "identify", "recognize", "label",
        "match", "memorize", "repeat", "state", "locate", "select", "quote",
        "reproduce", "recite",
    },
    "understand": {
        "explain", "describe", "discuss", "summarize", "interpret", "classify",
        "compare", "contrast", "distinguish", "paraphrase", "translate",
        "infer", "outline", "relate", "predict", "express", "extend",
    },
    "apply": {
        "apply", "use", "solve", "calculate", "compute", "demonstrate",
        "modify", "operate", "prepare", "produce", "show", "sketch",
        "complete", "illustrate", "practice", "employ", "examine",
    },
    "analyze": {
        "analyze", "examine", "differentiate", "divide", "separate",
        "categorize", "organize", "investigate", "question", "test",
        "conclude", "survey", "diagram",
    },
    "evaluate": {
        "evaluate", "assess", "judge", "critique", "justify", "appraise",
        "argue", "defend", "rank", "rate", "recommend", "review", "support",
        "weigh", "grade",
    },
    "create": {
        "create", "design", "construct", "develop", "formulate", "generate",
        "invent", "compose", "plan", "propose", "build", "devise",
        "synthesize", "imagine", "improve",
    },
}

# Tie-break: when two levels tie, prefer the higher cognitive demand.
_BLOOM_ORDER = ["remember", "understand", "apply", "analyze", "evaluate", "create"]

# Concepts that are document artifacts (CBSE admin docs) rather than
# curriculum content; never profile or use as prerequisites.
JUNK_CONCEPTS = {
    "board", "council", "candidate", "candidates", "textbook", "textbooks",
    "reprint", "reprints", "examination", "examinations", "students",
    "student", "school", "schools", "book", "books", "curiosity",
    "people", "persons", "person", "activity", "activities", "figure",
    "figures", "question", "questions", "answers", "answer", "chapter",
    "chapters", "exercise", "exercises", "section", "sections", "page",
    "pages", "notice", "notices", "notification", "notifications",
    "circular", "circulars", "byelaw", "byelaws", "scheme", "schemes",
    "certificate", "certificates", "application", "applications",
    "college", "colleges", "university", "universities", "committee",
    "committees", "subject", "subjects", "teacher", "teachers", "class",
    "classes", "grade", "grades", "lesson", "lessons", "unit", "units",
}

# Generic / content-free words that never carry prerequisite information.
# They may still be profiled for Bloom level, but must not appear as
# prerequisites of anything.
PREREQ_STOP = {
    "each", "every", "any", "all", "some", "other", "others", "various",
    "different", "many", "several", "things", "thing", "use", "uses",
    "used", "arrangement", "basis", "common", "certain", "particular",
    "people", "persons", "person", "one", "two", "new", "important",
    "following", "first", "next", "now", "like", "include", "etc",
    "place", "places", "part", "parts", "types", "type", "kind", "kinds",
    "ways", "way", "time", "times", "day", "days", "make", "made",
    "show", "shows", "find", "found", "know", "called", "said", "say",
    "see", "look", "give", "take", "come", "go", "work", "works",
    "same", "more", "most", "much", "small", "big", "long", "short",
    "good", "great", "own", "together", "around", "about", "into",
    "with", "without", "example", "examples", "world", "nature",
    "life", "question", "answer", "test", "learning", "education",
    "environment", "community", "culture", "skills", "skill",
    "ward", "wards", "boxes", "no", "not", "yes",
    "land", "poem", "poems", "mother", "writing", "size",
    "given number", "10 equal", "each case", "exploratory problems",
    "presence", "ab", "special thanks", "experiment", "experiments",
    "transformative learning culture", "national education policy 2020",
    "ganita prakash", "poorvi", "curiosity", "science", "mathematics",
    "english", "hindi", "world around us", "creative and critical thinking",
}

# Text-based prerequisite signals: a triple (A, PRED, B) whose predicate is
# in this set means B must be understood before A (B is a prerequisite).
PREREQ_VERBS = {
    "require", "requires", "needs", "need", "depend", "depends", "depend_on",
    "depends_on", "necessary", "essential", "use", "uses", "contain",
    "contains", "include", "includes", "involve", "involves", "help",
    "helps", "enable", "enables", "allow", "allows", "based", "based_on",
    "derived", "derived_from", "made_of", "consist", "consists", "composed",
}

# Stronger subset used for the provenance span-text fallback: weak verbs
# (use/help/allow) occur everywhere in prose ("poems use water as a
# theme") and add noise at the span level; predicates extracted by the
# IE are kept with the full PREREQ_VERBS set.
TEXT_SPAN_VERBS = {
    "require", "requires", "needs", "need", "depend", "depends", "depend_on",
    "depends_on", "necessary", "essential", "contain", "contains", "include",
    "includes", "involve", "involves", "enable", "enables", "based",
    "based_on", "derived", "derived_from", "made_of", "consist", "consists",
    "composed",
}

_DIFFICULTY_BASE = {
    "remember": 1,
    "understand": 2,
    "apply": 2,
    "analyze": 3,
    "evaluate": 4,
    "create": 4,
}

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def classify_bloom(verbs: list[str]) -> str:
    """Majority-vote Bloom level from observed sentence verbs."""
    if not verbs:
        return "understand"
    scores: dict[str, int] = collections.Counter()
    for v in verbs:
        for level, sig in BLOOM_VERBS.items():
            if v in sig:
                scores[level] += 1
                break
    if not scores:
        return "understand"
    best = max(_BLOOM_ORDER, key=lambda lv: (scores.get(lv, 0), _BLOOM_ORDER.index(lv)))
    return best


def estimate_difficulty(bloom: str, occurrences: int, source_docs: int,
                        prereq_count: int = 0) -> int:
    """1-5 difficulty heuristic:
    Bloom base + rarity penalty - cross-document familiarity bonus
    + prerequisite-load signal (Guvenir et al. 2026: concepts with many
    prerequisites sit deeper in the knowledge dependency structure, so a
    high prerequisite count raises difficulty)."""
    d = _DIFFICULTY_BASE.get(bloom, 2)
    if occurrences < 3:
        d += 1
    elif occurrences < 8:
        d += 0.5
    if source_docs >= 3:
        d -= 0.5
    if prereq_count >= 8:
        d += 1
    elif prereq_count >= 4:
        d += 0.5
    return max(1, min(5, round(d)))


def estimate_learning_hours(difficulty: int) -> float:
    """~0.5 h per difficulty point plus baseline (K-8 scale, see research docs)."""
    return round(0.5 * difficulty + 0.5, 1)


@dataclass
class ConceptContext:
    """A concept's observed contexts: pages per document and sentence verbs."""

    pages_by_doc: dict[str, list[int]] = field(default_factory=dict)
    verbs: list[str] = field(default_factory=list)


class CurriculumEstimator:
    """Mines prerequisite direction + Bloom/difficulty metadata from the
    chunk index and the concept graph. No LLM involvement."""

    def __init__(self, index_db, graph: Optional[GraphStore] = None):
        self.index_conn = _connect_index(index_db)
        self.graph: Optional[GraphStore] = graph
        self._ie = None  # lazy LocalIE

    def close(self) -> None:
        self.index_conn.close()

    # ---- context collection ----
    def contexts(self, label: str, max_chunks: int = 8,
                 doc_ids: Optional[set[str]] = None) -> ConceptContext:
        ctx = ConceptContext()
        doc_filter, params = self._doc_filter(doc_ids)
        rows = self.index_conn.execute(
            "SELECT document_id, page, text FROM chunks WHERE chunks MATCH ?" + doc_filter
            + " ORDER BY rank LIMIT ?",
            [_phrase(label)] + params + [max_chunks],
        ).fetchall()
        for doc_id, page, text in rows:
            ctx.pages_by_doc.setdefault(doc_id, []).append(page)
            for verb in self._sentence_verbs(text, label):
                ctx.verbs.append(verb)
        return ctx

    def occurs_in(self, label: str, doc_ids: Optional[set[str]] = None) -> bool:
        doc_filter, params = self._doc_filter(doc_ids)
        row = self.index_conn.execute(
            "SELECT 1 FROM chunks WHERE chunks MATCH ?" + doc_filter + " LIMIT 1",
            [_phrase(label)] + params,
        ).fetchone()
        return row is not None

    def _doc_filter(self, doc_ids: Optional[set[str]]) -> tuple[str, list]:
        if not doc_ids:
            return "", []
        return " AND document_id IN (%s)" % ",".join("?" * len(doc_ids)), list(doc_ids)

    def _sentence_verbs(self, text: str, label: str) -> list[str]:
        if not text:
            return []
        low = label.lower()
        verbs: list[str] = []
        for sent in _SENT_RE.split(text):
            if low not in sent.lower():
                continue
            root = self._sentence_root(sent)
            if root:
                verbs.append(root)
        return verbs

    def _sentence_root(self, sent: str) -> Optional[str]:
        nlp = self._nlp()
        doc = nlp(sent[:400])
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ in ("VERB", "AUX"):
                return tok.lemma_.lower()
        return None

    def _nlp(self):
        if self._ie is None:
            from ..extract.local_ie import LocalIE
            self._ie = LocalIE()
        return self._ie.nlp

    # ---- text-based prerequisite mining ----
    def text_prerequisites(self, graph: GraphStore, min_support: int = 1
                           ) -> dict[str, list[tuple[str, float, int]]]:
        """Mine prerequisite direction from triple predicates in the graph:
        (A, 'requires', B) means B is a prerequisite of A. Uses edge
        attributes (predicates) and provenance (span text) as evidence."""
        import json
        votes: dict[str, dict[str, list[tuple[float, int]]]] = {}
        rows = graph.conn.execute(
            "SELECT source, target, attributes, provenance FROM edges WHERE type='RELATED_TO'"
        ).fetchall()
        for src, tgt, attrs, prov in rows:
            preds: list[str] = []
            if attrs:
                try:
                    preds = [str(p) for p in json.loads(attrs).get("predicates", [])]
                except Exception:
                    pass
            if prov:
                try:
                    p = json.loads(prov)
                    if isinstance(p.get("span_text"), str):
                        words = set(p["span_text"].lower().split())
                        for v in TEXT_SPAN_VERBS:
                            if v in words and len(v) > 3:
                                preds.append(v)
                except Exception:
                    pass
            for pred in preds:
                pl = pred.lower().strip()
                if pl in PREREQ_VERBS:
                    # (source, pred, target) with a require-verb =>
                    # target is a prerequisite of source.
                    votes.setdefault(src, {}).setdefault(tgt, []).append((1.0, 1))
        return {
            s: sorted(((t, 1.0, 1) for t in v), key=lambda x: x[0])
            for s, v in votes.items()
        }

    # ---- page-order prerequisite mining ----
    def mine_prerequisites(self, labels: list[str], min_support: int = 2,
                           min_share: float = 0.6,
                           doc_ids: Optional[set[str]] = None) -> dict[str, list[tuple[str, float, int]]]:
        """For each label, return [(prereq, share, support), ...].

        A concept's *first* page per document defines its position in the
        curriculum; pairwise first-page order within each document produces
        directional votes (the ACE paper's document-based criterion).
        Uses the FTS index per label, then accumulates votes per document
        (near-linear in the number of concepts per document).

        Only pairs that co-occur in at least one chunk are considered: page
        order between unrelated concepts (e.g. 'Earth' and 'force') is
        meaningless, so every vote requires an actual co-occurrence link.

        doc_ids: restrict mining to a set of document ids (e.g. textbooks
        only); pairs from administrative/regulatory documents are noise.
        """
        doc_filter = ""
        params: list = []
        if doc_ids:
            doc_filter = " AND document_id IN (%s)" % ",".join("?" * len(doc_ids))
            params = list(doc_ids)
        first_page: dict[str, dict[str, int]] = {}
        # chunk_id -> set of labels seen in that chunk (co-occurrence evidence)
        chunk_labels: dict[str, set[str]] = {}
        for lb in labels:
            rows = self.index_conn.execute(
                "SELECT id, document_id, page, text FROM chunks WHERE chunks MATCH ?" + doc_filter,
                [_phrase(lb)] + params,
            ).fetchall()
            if not rows:
                continue
            fp = first_page.setdefault(lb, {})
            for chunk_id, doc_id, page, text in rows:
                if doc_id not in fp or page < fp[doc_id]:
                    fp[doc_id] = page
                low = text.lower()
                for other in labels:
                    if other != lb and other in low:
                        chunk_labels.setdefault(chunk_id, set()).update((lb, other))

        # pairs with at least one co-occurring chunk (semantic link)
        linked: set[tuple[str, str]] = set()
        for chunk_id, here in chunk_labels.items():
            here_list = sorted(here)
            for i in range(len(here_list)):
                for j in range(i + 1, len(here_list)):
                    linked.add((here_list[i], here_list[j]))

        # bucket labels by document (first page order)
        by_doc: dict[str, list[tuple[str, int]]] = {}
        for lb, fp in first_page.items():
            for doc_id, page in fp.items():
                by_doc.setdefault(doc_id, []).append((lb, page))

        ahead: dict[tuple[str, str], int] = {}
        total: dict[tuple[str, str], int] = {}
        for doc_items in by_doc.values():
            for i in range(len(doc_items)):
                la, pa = doc_items[i]
                for j in range(i + 1, len(doc_items)):
                    lb, pb = doc_items[j]
                    key = (la, lb) if pa < pb else (lb, la)
                    if key not in linked:
                        continue
                    if pa != pb:
                        ahead[key] = ahead.get(key, 0) + 1
                    total[key] = total.get(key, 0) + 1

        prereqs: dict[str, list[tuple[str, float, int]]] = {}
        for (earlier, later), t in total.items():
            if t < min_support:
                continue
            share = ahead.get((earlier, later), 0) / t
            if share >= min_share:
                prereqs.setdefault(later, []).append((earlier, share, t))
        for lb in prereqs:
            prereqs[lb].sort(key=lambda t: (-t[1], -t[2]))
        return prereqs

    # ---- cross-grade prerequisite mining ----
    def grade_prerequisites(self, labels: list[str], doc_grades: dict[str, int],
                            min_support: int = 1) -> dict[str, list[tuple[str, float, int]]]:
        """Cross-grade criterion: for the same subject across grades, a concept
        first appearing in an earlier grade is a prerequisite of one appearing
        in a later grade (NCERT books build concept depth grade by grade).

        doc_grades: {document_id: grade}. Returns {label: [(prereq, share, n)]}.
        """
        first_grade: dict[str, list[int]] = {}
        doc_filter = " AND document_id IN (%s)" % ",".join("?" * len(doc_grades))
        for lb in labels:
            rows = self.index_conn.execute(
                "SELECT document_id FROM chunks WHERE chunks MATCH ?" + doc_filter,
                [_phrase(lb)] + list(doc_grades.keys()),
            ).fetchall()
            grades = sorted({doc_grades[r[0]] for r in rows})
            if grades:
                first_grade[lb] = grades
        prereqs: dict[str, list[tuple[str, float, int]]] = {}
        for i, (later, g_l) in enumerate(first_grade.items()):
            for earlier, g_e in first_grade.items():
                if earlier == later:
                    continue
                # earlier must appear strictly in an earlier grade
                if min(g_e) >= min(g_l):
                    continue
                n = sum(1 for ge in g_e for gl in g_l if ge < gl)
                if n < min_support:
                    continue
                share = 1.0
                prereqs.setdefault(later, []).append((earlier, share, n))
        for lb in prereqs:
            prereqs[lb].sort(key=lambda t: (-t[1], -t[2]))
        return prereqs

    # ---- multi-criteria voting (Alatrash et al. 2025/26) ----
    def vote_prerequisites(self, labels: list[str],
                           doc_ids: Optional[set[str]] = None,
                           doc_grades: Optional[dict[str, int]] = None,
                           graph: Optional[GraphStore] = None,
                           theta: float = VOTE_THETA,
                           min_criteria: int = 2,
                           min_cooccur: int = 2,
                           prereq_candidates: Optional[set[str]] = None,
                           weights: Optional[dict[str, int]] = None,
                           strong_criteria: tuple[str, ...] = ("CMH", "IOLR", "Text", "HL-A"),
                           iolr_all_edges: bool = False,
                           iolr_min_degree: int = 0,
                           require_pair_evidence: bool = False,
                           criteria: Optional[tuple[str, ...]] = None,
                           vote_log: Optional[list] = None,
                           pair_detail: Optional[list] = None,
                           ) -> dict[str, list[tuple[str, float, int]]]:
        """Infer prerequisites with the paper's voting algorithm: five
        weighted binary criteria vote on the direction of each pair:

          TemO      first-page order within shared documents
          CMH       earlier grade in the same subject series
          IOLR      higher inbound/outbound prerequisite-link ratio
          BERTropy  higher Shannon entropy of co-occurrence spread
          Text      explicit graph triples (A requires B)
          HL-A      hyperlink containment: a is linked within b => a is
                    a prerequisite of b (pair-specific; resists the
                    hub-domination that biases IOLR on dense corpora)

        Each criterion contributes its weight (CRITERION_WEIGHTS, default
        equal CRITERION_WEIGHTS) to the direction it supports; the
        normalized difference (votes_ab - votes_ba) / (votes_ab + votes_ba)
        must reach theta (0.28) for a prerequisite edge, and at least
        min_criteria criteria must have voted. Two discriminative guards
        follow from the paper's Biology evaluation (Table 5): equal-weight
        voting scored F1 0.41 while CMH/IOLR alone reached 0.74 and BERTropy
        accuracy 0.75 — so we weight criteria by reliability and require a
        strong criterion (default CMH/IOLR/Text; pass a narrower set when a
        criterion's local signal is weak, e.g. ("CMH", "Text") on corpora
        where the graph-link signal is sparse):

          - the pair must co-occur in >= min_cooccur chunks (a shared
            document alone is not a semantic link), and
          - the winning direction must include a strong criterion vote.

        Returns {label: [(prereq, score, n_votes), ...]} with score in
        [-1, 1]. prereq_candidates: labels allowed to appear as
        prerequisites (e.g. concepts occurring in >= 2 documents); labels
        excluded there can still be profiled as the 'later' concept.
        """
        doc_filter, params = self._doc_filter(doc_ids)
        # token -> labels index: co-occurrence uses exact FTS token semantics
        # (a chunk matches a label when every token of the label appears),
        # not substring matching ("line" must not match "online").
        token_idx: dict[str, list[str]] = {}
        label_tokens: dict[str, frozenset] = {}
        for lb in labels:
            label_tokens[lb] = frozenset(lb.split())
            for tok in label_tokens[lb]:
                token_idx.setdefault(tok, []).append(lb)
        first_page: dict[str, dict[str, int]] = {}
        co_counts: dict[str, dict[str, int]] = {}
        headings: dict[str, dict[str, int]] = {}
        for lb in labels:
            rows = self.index_conn.execute(
                "SELECT id, document_id, page, heading, text FROM chunks"
                " WHERE chunks MATCH ?" + doc_filter,
                [_phrase(lb)] + params,
            ).fetchall()
            if not rows:
                continue
            fp = first_page.setdefault(lb, {})
            cc = co_counts.setdefault(lb, {})
            hd = headings.setdefault(lb, {})
            for chunk_id, doc_id, page, heading, text in rows:
                if page < FRONT_MATTER_PAGES:
                    continue
                if doc_id not in fp or page < fp[doc_id]:
                    fp[doc_id] = page
                hd[heading or "(none)"] = hd.get(heading or "(none)", 0) + 1
                toks = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
                if not toks:
                    continue
                for tok in toks:
                    for other in token_idx.get(tok, ()):
                        if other != lb and label_tokens[other] <= toks:
                            cc[other] = cc.get(other, 0) + 1

        grades: dict[str, int] = {}
        if doc_grades:
            for lb, fp in first_page.items():
                gs = {doc_grades[d] for d in fp if d in doc_grades}
                if gs:
                    grades[lb] = min(gs)

        iolr: dict[str, float] = {}
        if graph:
            stats = self.graph_link_stats(graph, any_predicate=iolr_all_edges)
            for k, (i, o) in stats.items():
                p = _plain_label(k)
                if p in labels:
                    # The +1 smoothing dominates on small graphs (a node with
                    # 1 in/0 out links gets 1.0 and outranks real hubs), so
                    # IOLR abstains unless both endpoints have enough links
                    # (the paper computed IOLR over Wikipedia's dense link
                    # graph where the smoothing was negligible).
                    if i + o >= iolr_min_degree:
                        iolr[p] = (i + 1) / (i + o + 1)

        # BERTropy over the *heading* distribution of each concept's chunks:
        # general concepts appear under many chapter/section headings, advanced
        # ones under few (the paper's topic-spread assumption, local analog of
        # their sentence-embedding topic clusters).
        entropy: dict[str, float] = {lb: _shannon(hd) for lb, hd in headings.items()}

        text_votes: dict[str, set[str]] = {}
        if graph:
            for k, v in self.text_prerequisites(graph).items():
                kp = _plain_label(k)
                text_votes.setdefault(kp, set()).update(_plain_label(t) for t, _, _ in v)

        # HL-A: directional hyperlink containment. Edge (X -> Y) means X's
        # article/page links to Y; if a is linked *within* b, a is a
        # prerequisite of b (the paper's pair-specific hub-resistant vote).
        links_out: dict[str, set[str]] = {}
        if graph:
            for src, tgt in graph.conn.execute(
                    "SELECT source, target FROM edges WHERE type='RELATED_TO'").fetchall():
                sp, tp = _plain_label(src), _plain_label(tgt)
                if sp in labels and tp in labels:
                    links_out.setdefault(sp, set()).add(tp)

        present = list(first_page)
        w = weights or CRITERION_WEIGHTS
        active = set(criteria) if criteria else set(CRITERIA)
        prereqs: dict[str, list[tuple[str, float, int]]] = {}
        for i, a in enumerate(present):
            if a.lower() in PREREQ_STOP:
                continue
            for b in present[i + 1:]:
                if b.lower() in PREREQ_STOP:
                    continue
                if prereq_candidates is not None and b not in prereq_candidates:
                    continue
                # HL-A containment is itself a semantic link: a pair linked
                # by hyperlink containment may vote even without chunk
                # co-occurrence (the paper's pair-specific evidence source).
                hl_pair = ("HL-A" in active
                           and ((a in links_out.get(b, ())) != (b in links_out.get(a, ()))))
                if co_counts.get(a, {}).get(b, 0) < min_cooccur and not hl_pair:
                    continue
                votes: dict[str, int] = {}
                fa, fb = first_page[a], first_page[b]
                common = fa.keys() & fb.keys()
                if "TemO" in active and common:
                    ahead = sum(1 for d in common if fa[d] < fb[d])
                    behind = len(common) - ahead
                    if ahead > behind:
                        votes["TemO"] = 1
                    elif behind > ahead:
                        votes["TemO"] = -1
                ga, gb = grades.get(a), grades.get(b)
                if "CMH" in active and ga is not None and gb is not None and ga != gb:
                    votes["CMH"] = 1 if ga < gb else -1
                ra, rb = iolr.get(a), iolr.get(b)
                if "IOLR" in active and ra is not None and rb is not None and ra != rb:
                    votes["IOLR"] = 1 if ra > rb else -1
                ea, eb = entropy.get(a), entropy.get(b)
                if "BERTropy" in active and ea != eb:
                    votes["BERTropy"] = 1 if ea > eb else -1
                if "Text" in active:
                    ta = a in text_votes.get(b, ())
                    tb = b in text_votes.get(a, ())
                    if ta != tb:
                        votes["Text"] = 1 if ta else -1
                if "HL-A" in active:
                    ha = a in links_out.get(b, ())
                    hb = b in links_out.get(a, ())
                    if ha != hb:
                        votes["HL-A"] = 1 if ha else -1
                n = len(votes)
                if n < min_criteria:
                    if vote_log is not None:
                        vote_log.append((a, b, dict(votes), None))
                    continue
                va = sum(w[c] for c, v in votes.items() if v > 0)
                vb = sum(w[c] for c, v in votes.items() if v < 0)
                s = (va - vb) / (va + vb)
                if vote_log is not None:
                    vote_log.append((a, b, dict(votes), s))
                # a wins => a is the prerequisite of b; a strong criterion
                # must back the winning side (paper Table 5: CMH/IOLR are
                # the top-F1 criteria; Text is the other reliable signal).
                # require_pair_evidence additionally demands a pair-specific
                # criterion (TemO order, a text triple, or an HL-A
                # containment): pairs supported only by global per-concept
                # scores (IOLR/BERTropy) are hub-dominated and mostly false
                # positives.
                if s >= theta:
                    if not any(votes.get(c, 0) > 0 for c in strong_criteria):
                        continue
                    if require_pair_evidence and not (votes.get("TemO", 0) > 0
                                                      or votes.get("Text", 0) > 0
                                                      or votes.get("HL-A", 0) > 0):
                        continue
                    prereqs.setdefault(b, []).append((a, s, va))
                    if pair_detail is not None:
                        pair_detail.append({"prereq": a, "later": b, "score": s,
                                            "va": va, "vb": vb, "votes": dict(votes)})
                elif s <= -theta:
                    if not any(votes.get(c, 0) < 0 for c in strong_criteria):
                        continue
                    if require_pair_evidence and not (votes.get("TemO", 0) < 0
                                                      or votes.get("Text", 0) < 0
                                                      or votes.get("HL-A", 0) < 0):
                        continue
                    prereqs.setdefault(a, []).append((b, -s, vb))
                    if pair_detail is not None:
                        pair_detail.append({"prereq": b, "later": a, "score": -s,
                                            "va": vb, "vb": va, "votes": dict(votes)})
        for lb in prereqs:
            prereqs[lb].sort(key=lambda t: (-t[1], -t[2]))
        return prereqs

    # ---- IOLR support: inbound/outbound prerequisite-verb link stats ----
    def graph_link_stats(self, graph: GraphStore, any_predicate: bool = False
                         ) -> dict[str, tuple[int, int]]:
        """Per node: (inbound, outbound) count of edges whose predicates are
        prerequisite verbs. A triple (A, requires, B) gives B one inbound
        (B is required) and A one outbound. Foundational concepts show a
        high inbound/(inbound+outbound) ratio. any_predicate=True counts
        every RELATED_TO edge regardless of predicate (for corpora where
        the link graph itself is the evidence, e.g. Wikipedia hyperlinks)."""
        import json
        stats: dict[str, list[int]] = {}
        rows = graph.conn.execute(
            "SELECT source, target, attributes FROM edges WHERE type='RELATED_TO'"
        ).fetchall()
        for src, tgt, attrs in rows:
            if any_predicate:
                s = stats.setdefault(src, [0, 0])
                t = stats.setdefault(tgt, [0, 0])
                s[1] += 1
                t[0] += 1
                continue
            if not attrs:
                continue
            try:
                preds = [str(p) for p in json.loads(attrs).get("predicates", [])]
            except Exception:
                continue
            if not any(p.lower() in PREREQ_VERBS for p in preds):
                continue
            s = stats.setdefault(src, [0, 0])
            t = stats.setdefault(tgt, [0, 0])
            s[1] += 1
            t[0] += 1
        return {k: (v[0], v[1]) for k, v in stats.items()}


def _phrase(label: str) -> str:
    """FTS5 phrase query for an exact concept label (safe against quotes)."""
    clean = label.replace('"', " ").strip()
    return '"' + clean + '"'


def _plain_label(node_id: str) -> str:
    """Strip the canonical graph id prefix ('cbse:concept:water' -> 'water')."""
    return node_id.split(":", 2)[-1]


def _shannon(dist: dict[str, int]) -> float:
    """Shannon entropy of a distribution; 0 for empty. General concepts
    co-occur with many different concepts (high entropy); advanced concepts
    are focused (low entropy)."""
    total = sum(dist.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in dist.values())


def transitive_prereqs(prereqs: dict[str, list[tuple[str, float, int]]],
                       max_hop: int = 2) -> dict[str, list[tuple[str, float, int]]]:
    """Transitivity assumption (Alatrash et al. 2025/26): if A <- B and
    B <- C then A <- C. Two-hop closure with score decay (product of edge
    scores) and the same theta threshold applied to the composed score.
    (longer chains are intentionally left out until evidence quality is
    verified against a gold standard)"""
    out: dict[str, dict[str, tuple[float, int]]] = {
        k: {p: (s, n) for p, s, n in v} for k, v in prereqs.items()
    }
    for _ in range(max_hop - 1):
        changed = False
        for a, edges in list(out.items()):
            for b, (sab, nab) in list(edges.items()):
                for c, (sbc, nbc) in out.get(b, {}).items():
                    if c == a:
                        continue
                    s = sab * sbc
                    if s < VOTE_THETA:
                        continue
                    if s > out[a].get(c, (0.0, 0))[0]:
                        out[a][c] = (s, nab + nbc)
                        changed = True
        if not changed:
            break
    return {k: [(p, s, n) for p, (s, n) in v.items()] for k, v in out.items()}


def _connect_index(index_db):
    import sqlite3
    conn = sqlite3.connect(index_db, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn
