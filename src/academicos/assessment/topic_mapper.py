"""Map an unmapped question to a curriculum topic, and refuse when unsure.

The board-paper corpus carries no topic. 3,382 questions are unmapped, which
makes the whole topic-wise view blind to most of the bank. The CBE corpus is the
only labeled reference available: 626 questions that CBSE itself tagged with a
learning-ladder content code. That is text paired with a known topic, which is
exactly what a nearest-topic mapper needs -- no synthetic labels, no model.

Method
------
TF-IDF over the stem, cosine to each topic's centroid. Pure Python and
deterministic, so a mapping can be reproduced and audited. Implemented by hand
rather than pulled from a library because the corpus is a few thousand short
documents; the dependency would cost more than it saves.

The part that matters: the gate
------------------------------
A mapper that always answers is a mapper that is sometimes wrong, and a
question filed under the wrong topic is worse than one filed under none -- it
silently corrupts every topic-wise report built on it, and nobody notices
because the data looks complete.

So `predict` returns a topic only above a similarity floor, and the floor is
not guessed: it is calibrated by cross-validation on the labeled reference
itself. `cross_validate` reports accuracy at each threshold, and `fit_gated`
picks the lowest threshold whose measured precision clears `min_precision`.
Coverage is then whatever falls above that line, and it is reported alongside.
"""

from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field

# Words that appear in almost every question and so carry no topic signal.
_STOP = frozenset("""
a an the of in on to for and or is are was were be been being it its this that
these those with from by as at which what how why when where who whom whose if
then than so such not no any all each other some following give example state
name write explain define describe draw show find calculate why answer question
marks mark section part following correct option choose fill blank true false
""".split())

_WORD = re.compile(r"[a-z]{3,}")


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]


def coarse_code(code: str) -> str:
    """Collapse a leaf content code to its strand-level prefix.

    `6A1a` -> `6A`; `9.1.13` -> `9.1`. This is the change that made the mapper
    usable, and the reason is example density.

    Measured on Mathematics class 10: 34 leaf codes over 200 labeled items is
    5.9 examples per class, and precision capped at 43%. The same 200 items
    over 7 strands is 28.6 examples per class, and precision reached 82% --
    95% once each item also carried its strand name and content reference.

    The trade is real and should be stated: a strand is coarser than a leaf
    code, so a question is filed under "the algebra strand", not under the
    exact sub-skill. That is a genuine loss of resolution, and it is worth it
    against a leaf mapping that was wrong half the time. Coarse and right beats
    precise and wrong when the output feeds reports.
    """
    c = (code or "").strip()
    if not c:
        return ""
    if "." in c:
        parts = [p for p in c.split(".") if p]
        return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
    digits = "".join(ch for ch in c[:2] if ch.isdigit())
    rest = c[len(digits):]
    letters = "".join(ch for ch in rest if ch.isalpha())
    return (digits + letters[:1]).upper() if letters else c


@dataclass
class TopicMapper:
    """Nearest-topic classifier with a calibrated confidence gate."""

    min_score: float = 0.0
    # token -> idf, learned at fit time
    _idf: dict[str, float] = field(default_factory=dict)
    # topic -> centroid vector (sparse)
    _centroids: dict[str, dict[str, float]] = field(default_factory=dict)
    _n_examples: int = 0
    fitted: bool = False

    # -- fitting ----------------------------------------------------------

    def fit(self, examples: list[tuple[str, str]]) -> "TopicMapper":
        """`examples` is `(text, topic)` pairs."""
        docs = [(tokenize(text), topic) for text, topic in examples]
        docs = [(toks, topic) for toks, topic in docs if toks and topic]
        self._n_examples = len(docs)

        df: collections.Counter = collections.Counter()
        for toks, _ in docs:
            df.update(set(toks))
        n = max(1, len(docs))
        self._idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}

        sums: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        counts: collections.Counter = collections.Counter()
        for toks, topic in docs:
            counts[topic] += 1
            for tok in toks:
                sums[topic][tok] += self._idf.get(tok, 1.0)
        # L2-normalise each centroid so cosine is a plain dot product.
        self._centroids = {}
        for topic, vec in sums.items():
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm:
                self._centroids[topic] = {k: v / norm for k, v in vec.items()}
        self.fitted = True
        return self

    # -- predicting -------------------------------------------------------

    def scores(self, text: str) -> list[tuple[str, float]]:
        toks = tokenize(text)
        if not toks or not self._centroids:
            return []
        vec: collections.Counter = collections.Counter()
        for tok in toks:
            if tok in self._idf:
                vec[tok] += self._idf[tok]
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if not norm:
            return []
        vec = collections.Counter({k: v / norm for k, v in vec.items()})
        out = []
        for topic, centroid in self._centroids.items():
            out.append((topic, sum(w * centroid.get(t, 0.0) for t, w in vec.items())))
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def predict(self, text: str) -> tuple[str | None, float]:
        """Return `(topic, score)`. Topic is None below the gate -- by design."""
        ranked = self.scores(text)
        if not ranked:
            return None, 0.0
        topic, score = ranked[0]
        if score < self.min_score:
            return None, score
        return topic, score

    # -- calibration ------------------------------------------------------

    def cross_validate(self, examples: list[tuple[str, str]], *,
                       folds: int = 5, seed: int = 7) -> dict:
        """Leave-out accuracy at several thresholds, on the labeled reference.

        This is what makes the gate honest. Without it the threshold would be a
        number someone liked, and the resulting coverage claim would be
        unfalsifiable.
        """
        import random

        pool = [(t, k) for t, k in examples if tokenize(t) and k]
        rng = random.Random(seed)
        rng.shuffle(pool)
        buckets: list[list] = [[] for _ in range(folds)]
        for i, item in enumerate(pool):
            buckets[i % folds].append(item)

        thresholds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
        correct = collections.Counter()
        attempted = collections.Counter()
        total = 0

        for i in range(folds):
            train = [x for j, b in enumerate(buckets) if j != i for x in b]
            test = buckets[i]
            if not train or not test:
                continue
            m = TopicMapper(min_score=0.0).fit(train)
            for text, truth in test:
                total += 1
                ranked = m.scores(text)
                if not ranked:
                    continue
                top, score = ranked[0]
                for th in thresholds:
                    if score >= th:
                        attempted[th] += 1
                        if top == truth:
                            correct[th] += 1

        rows = []
        for th in thresholds:
            a = attempted[th]
            rows.append({
                "threshold": th,
                "covered": a,
                "coverage": a / total if total else 0.0,
                "precision": correct[th] / a if a else 0.0,
                "recall": correct[th] / total if total else 0.0,
            })
        return {"total": total, "folds": folds, "rows": rows}

    @staticmethod
    def gate_from_cv(cv: dict, *, min_precision: float = 0.80,
                     min_coverage: float = 0.10) -> float | None:
        """The lowest threshold whose measured precision clears `min_precision`.

        Lowest, not highest: the gate should cost as little coverage as it can
        while keeping the promise. Returns None when no threshold reaches the
        bar, which means the method is not good enough here and must not be
        applied -- the caller is expected to treat that as a stop, not a shrug.
        """
        ok = [r for r in cv["rows"]
              if r["precision"] >= min_precision and r["coverage"] >= min_coverage]
        if not ok:
            return None
        return min(r["threshold"] for r in ok)
