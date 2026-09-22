"""Tag a question to its chapter, topic and subtopic in the taxonomy trees.

The trees (``academicos-data/syllabus/taxonomy/<Subject>_<grade>.json``) are
the NCERT textbooks' own headings, classes 6-10. This finds, for one question,
the chapter of its class that teaches it, the topic in that chapter and up to
two subtopics in that topic -- each with a confidence -- and says which of
those levels are sure enough to write onto the question.

How it matches. Deterministic TF-IDF, no model, no network. Every node is
represented by its heading plus the term counts of the book text printed under
it (``_node_text.json``, built by ``scripts/build_taxonomy_text.py``): a
heading alone ("A Lakh Varieties!") shares no word with the questions it
covers, the pages under it share many. The question is its stem, options and
answer (the answer at half weight: in the board bank the answer text is often
another question's). Each level is decided among siblings only -- chapters of
the class, topics of the chosen chapter, subtopics of the chosen topic -- with
IDF computed over those siblings, so words every topic of a chapter uses do not
decide between its topics.

The chapter prior. NCERT Exemplar questions carry the Exemplar chapter they
were printed in. All questions of one Exemplar chapter are pooled and matched
as one document, and that match is added at half weight to each question's
own. The Exemplar chapter names are the pre-2024 books' ("Integers"), and the
classes 6-9 trees are the new books ("The Other Side of Zero"), so the name
cannot be looked up; the pooled text can be matched.

A record with no Exemplar chapter gets the same kind of prior from the
``chapterIds`` it already carries (CBE and board banks). Those ids are of
three kinds -- old syllabus slugs ("light-reflection-refraction", which the
class 10 tree names almost word for word; "integers", which the new class 7
book does not), class 10 Maths unit names ("algebra") and CBE learning-outcome
codes ("6N1a", "10.3.11") -- so none can be looked up either. Each id is pooled
over the records of the bank that carry it, plus its name's words (from the
existing ``syllabus/<Subject>_<grade>.json`` where it is listed there, else the
slug's words), and matched like an Exemplar pool. An id seen on one record only
and with no name words is not a prior: its pool would be the question itself.
Many of these ids were inferred by an earlier tagger (``metadata.chapterTag``
says ``inferred``), so the prior is weighted, not obeyed, and the confidence
model sees that it came from ``chapterIds`` (feature ``idPrior``).

Topic or subtopic. Most questions are about a topic as a whole, not one
subtopic of it (236 of the 328 gold items have no subtopic a reviewer would
accept), so the subtopic decision has one more candidate than the topic has
subtopics: the topic's heading and own text, printed before its first
subtopic. When that
matches best no subtopic is proposed -- the question is tagged to the topic.

Confidence. Each level's raw scores (best, margin over the runner-up, whether
there was a runner-up) go through a logistic model fitted on the gold set
(``_tagger_model.json``, written by ``scripts/calibrate_topic_tagger.py``).
A topic's confidence is P(chapter right) x P(topic right | chapter right), a
subtopic's multiplies once more, so a sure subtopic under an unsure chapter is
not sure. A level is accepted when its confidence clears the threshold chosen
so that accepted tags are at least 93% right out-of-fold on the gold set -- a
margin over the 90% bar, which nested cross-validation holds them to; anything below
goes to the review queue, not onto the question. A confidently wrong tag is
worse than none: the blank shows in the coverage report, the wrong one does
not.
"""
from __future__ import annotations

import collections
import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from academicos.syllabus.tag_text import TOKENIZER_VERSION, tokens

REPO = Path(__file__).resolve().parents[3]
TAXONOMY_DIR = REPO / "academicos-data" / "syllabus" / "taxonomy"
# v2: the subtopic decision can end at the topic ("none of its subtopics")
TAG_METHOD = "tfidf_textbook_v2"
LEVELS = ("chapter", "topic", "subtopic")
# Fields `apply_tags` adds to a record; nothing else on it is touched.
TAG_FIELDS = ("taxonomyChapterId", "topicIds", "subtopicIds", "tagConfidence", "tagMethod")

SUBJECTS = {"Mathematics": "Mathematics", "Science": "Science",
            "Social Science": "Social_Science"}
GRADES = range(6, 11)

HEADING_WEIGHT = 3.0     # a heading word counts as three occurrences of it
ANSWER_WEIGHT = 0.5      # answer text: see the module docstring
PRIOR_WEIGHT = 0.5       # the pooled Exemplar chapter's match, added to the question's
# The pooled chapterIds' match, likewise, but at a quarter: on the gold set the
# ids already on CBE/board records are often wrong (an earlier embedding
# tagger's guesses), and a heavier prior moved more right picks to wrong than
# wrong to right (raw chapter picks on the 108 CBE+board gold items: 75 right
# without it, 74 at 0.25, 71 at 0.5, 65 at 2.0). At 0.25, with the confidence
# model told the prior came from chapterIds, accepted tags grew out-of-fold
# from 138/153 to 142/157 (chapter) and 55/61 to 60/66 (topic), under the old
# cut-at-90% rule. Tuned on the gold set, so _tag_audit*.json are the held-out checks.
ID_PRIOR_WEIGHT = 0.25
SECOND_SUBTOPIC = 0.9    # a second subtopic only when it scores within 10% of the first

# Board-paper records carry answer text that often belongs to another
# question: of the 24 board items in the gold set, at least 15 have an answer
# to a different question (cbse:q:src:5a4393f67e8e73f306439993:23 asks about
# Sri Lankan Tamils and "answers" with party reforms). Their answers are not read.
ANSWER_UNRELIABLE_SOURCES = frozenset({"cbse_board_paper"})

# An "Introduction" (or "A Pinch of History") previews the whole chapter, so it
# matches every question of the chapter a little and took topic picks that
# belonged to a named topic (exemplar:q:10:mathematics:5:5.3:35, a sum-of-an-AP
# word problem, went to "Introduction"). Its score is damped, not removed.
GENERIC_HEADING = re.compile(
    r"^(introduction|a pinch of history|conclusion|did you ever wonder)\b", re.I)
GENERIC_DAMPING = 0.8

# Topic headings are specific enough to vote: "Decentralisation in India" is
# the topic of a question that says "decentralisation", whatever the pages
# under "How is Federalism Practised?" also say. A topic's score adds 0.2 x
# its heading-only match. On the gold set this moved 5 of 96 topic picks from
# wrong to right and none the other way; chapter headings are too broad and
# subtopic headings too short for it to help (subtopics lost 2 of 41).
TOPIC_HEADING_BLEND = 0.2

# A question can be about its topic as a whole, not about any one subtopic of
# it: 236 of the 328 gold items have no subtopic a reviewer would accept.
# The subtopic decision therefore has one more candidate, the topic's own
# text (its span before its first subtopic) plus its heading; when that
# matches best the question stays at the topic and no subtopic is proposed.
# Measured on the gold set, out-of-fold: with no such candidate no subtopic
# threshold reached 90% on 10+ tags (0 accepted); with the span alone 10/11;
# span plus heading 11/12. The heading matters because most topics' own span
# is short (95 of 385 topics with subtopics have under 20 words of it).
TOPIC_LEVEL_ID = ""


def question_subject(rec: dict) -> str | None:
    """The tree subject for a record, or None when there is no tree for it.

    The board bank names paper variants ("Mathematics (Standard)"); its
    ``metadata.subjectFamily`` is the subject the trees are keyed on.
    """
    meta = rec.get("metadata") or {}
    subject = meta.get("subjectFamily") or rec.get("subject") or ""
    if subject not in SUBJECTS:
        subject = subject.split(" (")[0]
    return subject if subject in SUBJECTS else None


def question_grade(rec: dict) -> int | None:
    try:
        g = int(str(rec.get("grade")).strip())
    except ValueError:
        return None
    return g if g in GRADES else None


def question_terms(rec: dict) -> collections.Counter:
    c = collections.Counter(tokens(rec.get("stem") or ""))
    answer_weight = 0.0 if rec.get("source") in ANSWER_UNRELIABLE_SOURCES else ANSWER_WEIGHT
    answer = collections.Counter(tokens((rec.get("answerScheme") or {}).get("modelAnswer") or ""))
    for part in rec.get("parts") or []:
        c.update(tokens(part.get("text") or ""))
        for option in part.get("options") or []:
            c.update(tokens(option))
        answer.update(tokens(part.get("expectedAnswer") or ""))
    # CBE items name what they assess
    for key in ("topic", "contentReference"):
        if rec.get(key):
            c.update(tokens(rec[key]))
    for w, n in answer.items():
        if answer_weight:
            c[w] += answer_weight * n
    return c


def _heading_terms(name: str) -> collections.Counter:
    return collections.Counter({w: HEADING_WEIGHT * n
                                for w, n in collections.Counter(tokens(name)).items()})


def _weights(counts: dict) -> dict:
    return {w: 1.0 + math.log(n) for w, n in counts.items() if n > 0}


@dataclass
class Node:
    id: str
    name: str
    terms: collections.Counter
    children: list["Node"] = field(default_factory=list)


@dataclass
class _Siblings:
    """One decision: which of these nodes. IDF is over these nodes only."""
    nodes: list[Node]

    def __post_init__(self):
        n = len(self.nodes)
        df = collections.Counter()
        for node in self.nodes:
            df.update(node.terms.keys())
        self.idf = {w: math.log((n + 1) / (d + 1)) + 1.0 for w, d in df.items()}
        self.default_idf = math.log(n + 1) + 1.0
        self.vectors = [self._vector(node.terms) for node in self.nodes]
        self.damping = [GENERIC_DAMPING if GENERIC_HEADING.match(node.name) else 1.0
                        for node in self.nodes]

    def _vector(self, counts: dict) -> tuple[dict, float]:
        v = {w: x * self.idf.get(w, self.default_idf) for w, x in _weights(counts).items()}
        return v, math.sqrt(sum(x * x for x in v.values()))

    def scores(self, counts: dict) -> tuple[list[float], dict]:
        qv, qn = self._vector(counts)
        out = []
        for (dv, dn), damp in zip(self.vectors, self.damping):
            out.append(damp * sum(x * dv.get(w, 0.0) for w, x in qv.items()) / (qn * dn)
                       if qn and dn else 0.0)
        return out, qv

    def heading_scores(self, counts: dict) -> list[float]:
        """How well the question matches each node's heading words alone, IDF
        over the siblings' headings: a question that says "sum of n terms" and
        a topic headed "Sum of First n Terms of an AP" agree on more than the
        pages under it do."""
        if not hasattr(self, "_headings"):
            self._headings = _Siblings([Node(n.id, n.name, collections.Counter(tokens(n.name)))
                                        for n in self.nodes])
        return self._headings.scores(counts)[0]

    def shared_terms(self, qv: dict, index: int, k: int = 6) -> list[str]:
        dv, _ = self.vectors[index]
        ranked = sorted(((x * dv[w], w) for w, x in qv.items() if w in dv), reverse=True)
        return [w for _, w in ranked[:k]]


class Tree:
    def __init__(self, data: dict, node_text: dict):
        self.subject = data["subject"]
        self.grade = data["grade"]
        chapters = []
        self._own_span: dict[str, collections.Counter] = {}
        for ch in data["chapters"]:
            topics = []
            for tp in ch["topics"]:
                subs = [Node(s["id"], s["name"],
                             collections.Counter(node_text.get(s["id"], {})))
                        for s in tp["subtopics"]]
                own = collections.Counter(node_text.get(tp["id"], {}))
                self._own_span[tp["id"]] = collections.Counter(own)
                for s in subs:
                    own.update(s.terms)
                topics.append(Node(tp["id"], tp["name"], own, subs))
            own = collections.Counter(node_text.get(ch["id"], {}))
            for t in topics:
                own.update(t.terms)
            chapters.append(Node(ch["id"], ch["name"], own, topics))
        # headings are added after the spans are summed, so a topic's heading
        # counts in the topic, not again in its chapter
        for ch in chapters:
            for t in ch.children:
                for s in t.children:
                    s.terms.update(_heading_terms(s.name))
                t.terms.update(_heading_terms(t.name))
            ch.terms.update(_heading_terms(ch.name))
        self.chapters = _Siblings(chapters)
        self._below: dict[str, _Siblings] = {}

    def children(self, node: Node) -> _Siblings | None:
        if not node.children:
            return None
        if node.id not in self._below:
            self._below[node.id] = _Siblings(node.children)
        return self._below[node.id]

    def subtopic_choice(self, topic: Node) -> _Siblings | None:
        """The subtopics of ``topic`` plus a topic-level candidate (id
        ``TOPIC_LEVEL_ID``: the topic's heading and its own span, if any) that
        stands for "none of them"."""
        if not topic.children:
            return None
        key = "with-topic-level:" + topic.id
        if key not in self._below:
            nodes = list(topic.children)
            own = self._own_span.get(topic.id)
            terms = collections.Counter(own or {})
            terms.update(_heading_terms(topic.name))
            nodes.append(Node(TOPIC_LEVEL_ID, topic.name, terms))
            self._below[key] = _Siblings(nodes)
        return self._below[key]


@dataclass
class TagResult:
    chapter_id: str | None
    topic_id: str | None
    subtopic_ids: list[str]
    # raw features per level: {"best", "margin", "single", "heading"} (the
    # chapter also "prior", "hasPrior"); None when the level does not exist
    # below the chosen node (a chapter with no topics)
    features: dict
    confidence: dict
    accepted: dict
    evidence: dict

    def to_proposal(self) -> dict:
        return {"chapterId": self.chapter_id, "topicId": self.topic_id,
                "subtopicIds": list(self.subtopic_ids)}


def _features(scores: list[float]) -> dict:
    ranked = sorted(scores, reverse=True)
    second = ranked[1] if len(ranked) > 1 else 0.0
    return {"best": ranked[0], "margin": ranked[0] - second, "single": len(ranked) == 1}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -60 else 0.0


def level_probability(coef: dict, feats: dict) -> float:
    """P(this level's pick is right), from the fitted logistic model.

    A feature the model scores but ``feats`` lacks is a KeyError: counting it
    as 0 would let old coefficients go on scoring new tagger output."""
    z = coef["bias"] + sum(w * float(feats[k]) for k, w in coef.items() if k != "bias")
    return _sigmoid(z)


def _check_model(model: dict) -> None:
    """Refuse a confidence model fitted for other tagger output."""
    if model.get("method") != TAG_METHOD:
        raise RuntimeError(
            f"confidence model was fitted for tag method {model.get('method')!r}, this code "
            f"is {TAG_METHOD!r}: refit it with scripts/calibrate_topic_tagger.py")
    for level in LEVELS:
        declared = set((model.get("features") or {}).get(level) or ())
        coef = (model.get("coefficients") or {}).get(level)
        if coef is None or "bias" not in coef:
            raise RuntimeError(f"confidence model has no {level} coefficients: refit it")
        unknown = sorted(k for k in coef if k != "bias" and k not in declared)
        if unknown:
            raise RuntimeError(
                f"confidence model scores {level} features {unknown} that its "
                f"features[{level!r}] does not declare: refit it")


class TopicTagger:
    def __init__(self, taxonomy_dir: Path | None = None, model: dict | None = None,
                 thresholds: dict | None = None):
        self.dir = Path(taxonomy_dir) if taxonomy_dir else TAXONOMY_DIR
        index = json.loads((self.dir / "_node_text.json").read_text(encoding="utf-8"))
        if index.get("tokenizer") != TOKENIZER_VERSION:
            raise RuntimeError(
                f"_node_text.json was built by tokenizer {index.get('tokenizer')}, this code "
                f"is {TOKENIZER_VERSION}: rebuild it with scripts/build_taxonomy_text.py")
        self.node_text = index["nodes"]
        if model is None:
            path = self.dir / "_tagger_model.json"
            if not path.exists():
                # without it every confidence is None and nothing is accepted:
                # applying would empty every bank's tags into the review queue
                raise RuntimeError(
                    f"{path} is missing: fit it with scripts/calibrate_topic_tagger.py "
                    "(pass model={} only to get raw, unscored picks)")
            model = json.loads(path.read_text(encoding="utf-8"))
        if model:
            _check_model(model)
        self.model = model
        self.thresholds = dict(thresholds if thresholds is not None
                               else (model or {}).get("thresholds") or {})
        self._trees: dict[tuple[str, int], Tree | None] = {}
        self._id_names: dict[tuple[str, int], dict[str, str]] = {}

    # ------------------------------------------------------------------ trees
    def tree(self, subject: str, grade: int) -> Tree | None:
        key = (subject, grade)
        if key not in self._trees:
            path = self.dir / f"{SUBJECTS[subject]}_{grade}.json"
            self._trees[key] = (Tree(json.loads(path.read_text(encoding="utf-8")),
                                     self.node_text) if path.exists() else None)
        return self._trees[key]

    # ---------------------------------------------------------------- tagging
    def tag_records(self, records: list[dict]) -> list[TagResult | None]:
        """Tag every record; None for one with no tree (another subject or class).

        Questions are pooled per Exemplar chapter, or else per existing chapter
        id, across ``records``, so pass a whole bank, not one question at a
        time, to get the full prior.
        """
        pools: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
        members: collections.Counter = collections.Counter()
        for rec in records:
            for key in _pool_keys(rec):
                pools[key].update(question_terms(rec))
                pools[key].update(_heading_terms(self._prior_name(key)))
                members[key] += 1
        out = []
        for rec in records:
            subject, grade = question_subject(rec), question_grade(rec)
            tree = self.tree(subject, grade) if subject and grade else None
            if tree is None:
                out.append(None)
                continue
            keys = [k for k in _pool_keys(rec)
                    if k[2] == "exemplar" or members[k] > 1 or tokens(self._prior_name(k))]
            out.append(self._tag(tree, question_terms(rec), [pools[k] for k in keys],
                                 from_ids=bool(keys) and keys[0][2] == "chapterId"))
        return out

    def _prior_name(self, key: tuple) -> str:
        """The words a pool key names its chapter by: the Exemplar chapter
        title, or a chapter id's name in the existing syllabus file."""
        subject, grade, kind, value = key
        if kind == "exemplar":
            return value
        if (subject, grade) not in self._id_names:
            self._id_names[(subject, grade)] = _syllabus_names(subject, grade)
        name = self._id_names[(subject, grade)].get(value)
        if name is not None:
            return name
        # "decimals-6", "circles-9": the slug's words without the class
        return " ".join(w for w in value.split("-") if not w.isdigit())

    def _tag(self, tree: Tree, terms: collections.Counter,
             pools: list[collections.Counter], from_ids: bool = False) -> TagResult:
        feats: dict = {"chapter": None, "topic": None, "subtopic": None}
        evidence: dict = {"chapter": [], "topic": [], "subtopic": []}

        scores, qv = tree.chapters.scores(terms)
        prior = [0.0] * len(scores)
        if pools:
            # a record with two chapter ids gets the mean of their matches
            for pool in pools:
                prior = [a + b / len(pools) for a, b in zip(prior, tree.chapters.scores(pool)[0])]
            weight = ID_PRIOR_WEIGHT if from_ids else PRIOR_WEIGHT
            scores = [s + weight * p for s, p in zip(scores, prior)]
        ci = max(range(len(scores)), key=lambda i: (scores[i], -i))
        chapter = tree.chapters.nodes[ci]
        feats["chapter"] = _features(scores)
        feats["chapter"]["heading"] = tree.chapters.heading_scores(terms)[ci]
        # how well the whole Exemplar chapter matches this one: a chapter of the
        # old book that the new book dropped matches nothing well
        feats["chapter"]["prior"] = prior[ci]
        feats["chapter"]["hasPrior"] = bool(pools)
        feats["chapter"]["idPrior"] = bool(pools) and from_ids
        evidence["chapter"] = tree.chapters.shared_terms(qv, ci)

        topic, subtopics = None, []
        below = tree.children(chapter)
        if below is not None:
            scores, qv = below.scores(terms)
            heads = below.heading_scores(terms)
            scores = [s + TOPIC_HEADING_BLEND * h for s, h in zip(scores, heads)]
            ti = max(range(len(scores)), key=lambda i: (scores[i], -i))
            topic = below.nodes[ti]
            feats["topic"] = _features(scores)
            feats["topic"]["heading"] = heads[ti]
            evidence["topic"] = below.shared_terms(qv, ti)
            under = tree.subtopic_choice(topic)
            if under is not None:
                scores, qv = under.scores(terms)
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                if under.nodes[order[0]].id == TOPIC_LEVEL_ID:
                    # the topic's own text wins: the question is about the
                    # topic as a whole, and there is no subtopic to decide
                    evidence["subtopic"] = ["(topic as a whole)"]
                else:
                    subtopics = [under.nodes[order[0]]]
                    if (len(order) > 1 and scores[order[0]] > 0
                            and under.nodes[order[1]].id != TOPIC_LEVEL_ID
                            and scores[order[1]] >= SECOND_SUBTOPIC * scores[order[0]]):
                        subtopics.append(under.nodes[order[1]])
                    feats["subtopic"] = _features(scores)
                    feats["subtopic"]["heading"] = under.heading_scores(terms)[order[0]]
                    evidence["subtopic"] = under.shared_terms(qv, order[0])

        confidence = self._confidence(feats)
        accepted = self._accept(confidence)
        return TagResult(chapter.id, topic.id if topic else None, [s.id for s in subtopics],
                         feats, confidence, accepted, evidence)

    def _confidence(self, feats: dict) -> dict:
        conf = {"chapter": None, "topic": None, "subtopic": None}
        if not self.model:
            return conf
        running = 1.0
        for level in LEVELS:
            if feats[level] is None:
                break
            running *= level_probability(self.model["coefficients"][level], feats[level])
            conf[level] = round(running, 4)
        return conf

    def _accept(self, confidence: dict) -> dict:
        accepted = {}
        ok = True
        for level in LEVELS:
            c = confidence[level]
            ok = ok and c is not None and c >= self.thresholds.get(level, math.inf)
            accepted[level] = ok
        return accepted


def _pool_keys(rec: dict) -> list[tuple]:
    """The chapter prior's pools a record belongs to: its Exemplar chapter if
    it has one (the book it was printed in), else each of its chapter ids."""
    subject, grade = question_subject(rec), question_grade(rec)
    ex = (rec.get("metadata") or {}).get("exemplarChapter")
    if ex and ex.get("title"):
        return [(subject, grade, "exemplar", ex["title"])]
    ids = [c for c in rec.get("chapterIds") or [] if isinstance(c, str) and c.strip()]
    return [(subject, grade, "chapterId", c) for c in dict.fromkeys(ids)]


def _syllabus_names(subject: str | None, grade: int | None) -> dict[str, str]:
    """Chapter id -> name, and unit slug -> name, from the existing syllabus
    file (``academicos-data/syllabus/<Subject>_<grade>.json``, read only)."""
    if subject not in SUBJECTS or grade is None:
        return {}
    path = TAXONOMY_DIR.parent / f"{SUBJECTS[subject]}_{grade}.json"
    if not path.exists():
        return {}
    names = {}
    for unit in json.loads(path.read_text(encoding="utf-8")).get("units") or []:
        if unit.get("name"):
            names["-".join(re.findall(r"[a-z]+", unit["name"].lower()))] = unit["name"]
        for ch in unit.get("chapters") or []:
            if ch.get("id") and ch.get("name"):
                names[ch["id"]] = ch["name"]
    return names


# --------------------------------------------------------------------------- #
# writing tags onto records
# --------------------------------------------------------------------------- #

def apply_tags(records: list[dict], tagger: TopicTagger, bank: str) -> tuple[list[dict], list[dict]]:
    """Copies of ``records`` with the tag fields added, and the review queue.

    Only `TAG_FIELDS` are added (or replaced, on a rerun); every other field is
    left as it was. A record with no tree is returned unchanged. A record whose
    lowest available level was not accepted is queued, naming the first level
    that was not, with the proposal and its confidence.
    """
    tags = tagger.tag_records(records)
    out, review = [], []
    for rec, tag in zip(records, tags):
        new = copy.deepcopy(rec)
        if tag is None:
            out.append(new)
            continue
        acc = tag.accepted
        new["taxonomyChapterId"] = tag.chapter_id if acc["chapter"] else None
        new["topicIds"] = [tag.topic_id] if acc["topic"] and tag.topic_id else []
        new["subtopicIds"] = list(tag.subtopic_ids) if acc["subtopic"] else []
        new["tagConfidence"] = dict(tag.confidence)
        new["tagMethod"] = TAG_METHOD
        out.append(new)
        pending = next((lv for lv in LEVELS
                        if tag.features[lv] is not None and not acc[lv]), None)
        if pending:
            review.append({
                "id": rec.get("id"), "bank": bank, "subject": rec.get("subject"),
                "grade": rec.get("grade"), "level": pending,
                "stem": (rec.get("stem") or "")[:300],
                "proposed": tag.to_proposal(), "confidence": dict(tag.confidence),
                "evidence": tag.evidence,
            })
    return out, review


# --------------------------------------------------------------------------- #
# measuring against the gold set
# --------------------------------------------------------------------------- #

def judge(tag: TagResult, gold: dict) -> dict:
    """Whether each level of ``tag`` is right by the gold item's labels.

    A level counts only if every level above it is right too: a right topic
    name under a wrong chapter is not a right tag. A gold item with no chapter
    (its class's book does not teach it) makes any chapter wrong.
    """
    chapter = tag.chapter_id in gold["chapters"]
    topic = chapter and tag.topic_id is not None and tag.topic_id in gold["topics"]
    subs = [topic and s in gold["subtopics"] for s in tag.subtopic_ids]
    return {"chapter": chapter, "topic": topic, "subtopics": subs}


def gold_levels(item: dict) -> tuple[str, ...]:
    """The levels a gold item is counted at. An item drawn at random counts at
    every level; an audit item drawn *because* the tagger accepted a subtopic
    for it (``"levels": ["subtopic"]``) counts at the subtopic level only --
    its chapter and topic were accepted by construction, and counting them
    would flatter those levels' precision."""
    return tuple(item.get("levels") or LEVELS)


def score_against_gold(tagger: TopicTagger, items: list[dict], banks: dict) -> dict:
    """Accuracy of every pick, and precision of the accepted ones, on the gold set.

    ``banks`` maps each gold item's ``bank`` to that bank's full record list:
    the tagger sees the whole bank, as it does when it is applied, so the
    Exemplar chapter pools are the real ones.
    """
    tags: dict[str, TagResult] = {}
    wanted = {it["id"] for it in items}
    for name, records in banks.items():
        for rec, tag in zip(records, tagger.tag_records(records)):
            if rec.get("id") in wanted and tag is not None:
                tags[rec["id"]] = tag
    raw = {lv: [0, 0] for lv in LEVELS}
    acc = {lv: [0, 0] for lv in LEVELS}
    for it in items:
        tag = tags[it["id"]]
        j = judge(tag, it)
        levels = gold_levels(it)
        if "chapter" in levels:
            raw["chapter"][0] += j["chapter"]
            raw["chapter"][1] += 1
        if "topic" in levels and tag.topic_id is not None:
            raw["topic"][0] += j["topic"]
            raw["topic"][1] += 1
        if tag.subtopic_ids:
            raw["subtopic"][0] += j["subtopics"][0]
            raw["subtopic"][1] += 1
        if "chapter" in levels and tag.accepted["chapter"]:
            acc["chapter"][0] += j["chapter"]
            acc["chapter"][1] += 1
        if "topic" in levels and tag.accepted["topic"]:
            acc["topic"][0] += j["topic"]
            acc["topic"][1] += 1
        if tag.accepted["subtopic"]:
            acc["subtopic"][0] += sum(j["subtopics"])
            acc["subtopic"][1] += len(j["subtopics"])
    return {
        "items": len(items),
        "raw": {lv: {"right": r, "of": n, "accuracy": round(r / n, 4) if n else None}
                for lv, (r, n) in raw.items()},
        "accepted": {lv: {"right": r, "tags": n, "precision": round(r / n, 4) if n else None}
                     for lv, (r, n) in acc.items()},
    }
