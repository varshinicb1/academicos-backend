"""Deterministic local open information extraction (spaCy-based).

Fast, free, no API rate limits. Uses en_core_web_sm:
  - NER: noun chunks + named entities + key domain terms
  - RE:  subject-verb-object triples from dependency parses

Trade-off vs LLM OpenIE: lower recall on implicit relations, but zero cost,
~milliseconds per chunk, and fully deterministic (same input -> same graph).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import spacy

log = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its", "they",
    "there", "here", "we", "you", "i", "he", "she", "one", "two", "etc",
    "example", "examples", "figure", "fig", "question", "questions", "part",
    "parts", "chapter", "answer", "answers", "page", "pages", "section",
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "their", "our", "your", "my", "her", "him", "them", "us", "me", "his",
}

COPIULA = {"be", "am", "is", "are", "was", "were", "been", "being", "become", "became"}

VERB_STOP = {
    "be", "have", "do", "make", "use", "get", "take", "give", "show",
    "see", "find", "look", "know", "think", "say", "tell", "come", "go",
    "put", "keep", "let", "want", "need", "call", "refer", "mean", "read",
    "write", "draw", "mark", "match", "solve", "fill", "complete", "observe",
    "note", "discuss", "explain", "mention", "list", "identify", "guess",
    "choose", "copy", "learn", "remember", "imagine", "imagine",
}

_CONCEPT_RE = re.compile(r"^(the |a |an )?(.+?)(s|es)?$", re.I)


def _clean_concept(token_span) -> str:
    """Normalize a noun span to a canonical concept label."""
    label = " ".join(t.text for t in token_span if t.text.lower() not in STOPWORDS)
    label = re.sub(r"\s+", " ", label).strip(" .,;:()[]\"'")
    return label


GENERIC_NOUNS = {
    "process", "processes", "way", "ways", "thing", "things", "part", "parts",
    "whole", "example", "examples", "result", "results", "kind", "kinds",
    "type", "types", "number", "numbers", "amount", "amounts", "case", "cases",
    "side", "sides", "point", "points", "group", "groups", "set", "sets",
    "form", "forms", "method", "methods", "step", "steps", "value", "values",
    "unit", "units", "place", "places", "time", "times", "year", "years",
}


_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
_UNIT = {"cm", "mm", "km", "kg", "g", "mg", "ml", "l", "m", "s", "h", "min", "hrs", "sq"}


def _is_meaningful_concept(label: str) -> bool:
    low = label.lower().strip()
    if len(low) < 2 or len(low) > 60:
        return False
    if not re.search(r"[A-Za-z]", low):
        return False
    if low in STOPWORDS or low in GENERIC_NOUNS or low in _ROMAN or low in _UNIT:
        return False
    if low.startswith(("q.", "ans", "answer ", "fig ", "figure ", "exercise", "chapter", "section")):
        return False
    return True


@dataclass
class LocalIEResult:
    entities: list[str] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)


class LocalIE:
    """Deterministic NER + relation extraction via spaCy."""

    def __init__(self, model: str = "en_core_web_md"):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            log.error("spaCy model %s not installed; run: python -m spacy download %s", model, model)
            raise
        self._nlp_name = model
        self._seen: list[str] = []

    # ---- NER ----
    def entities(self, text: str, max_entities: int = 20) -> list[str]:
        self._seen = []
        doc = self.nlp(text[:2000])
        seen: list[str] = []
        for ent in doc.ents:
            label = _clean_concept(ent)
            if self._good_entity(label, ent):
                seen.append(label)
            if len(seen) >= max_entities:
                return seen
        for chunk in doc.noun_chunks:
            label = _clean_concept(chunk)
            if self._good_entity(label, chunk):
                seen.append(label)
            if len(seen) >= max_entities:
                return seen
        return seen

    def _good_entity(self, label: str, span) -> bool:
        if not label or not _is_meaningful_concept(label):
            return False
        if label.lower() in (s.lower() for s in self._seen):
            return False
        head = span.root if hasattr(span, "root") else span
        # A concept must be noun-headed; reject bare adjectives/pronouns/verbs.
        if head.pos_ not in ("NOUN", "PROPN", "NUM"):
            return False
        # Reject mixed-case noun phrases with leading possessive/determiner noise.
        first = span[0]
        if first.pos_ == "PRON":
            return False
        self._seen.append(label)
        return True

    # ---- RE: simple SVO extraction ----
    def triples(self, text: str, max_triples: int = 20) -> list[tuple[str, str, str]]:
        doc = self.nlp(text[:2000])
        out: list[tuple[str, str, str]] = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue
                if token.lemma_.lower() in VERB_STOP and token.lemma_.lower() not in COPIULA:
                    continue
                subj = self._find_child(sent, token, {"nsubj", "nsubjpass"})
                obj = self._find_child(sent, token, {"dobj", "attr", "oprd", "prep"})
                if subj is None or obj is None:
                    continue
                pred = token.lemma_.lower()
                if pred in COPIULA:
                    pred = "is"
                s = _clean_concept(subj)
                o = _clean_concept(obj)
                if not (_is_meaningful_concept(s) and _is_meaningful_concept(o)):
                    continue
                # Require noun-headed arguments: reject adjective objects like
                # "gravity is weak" -> (gravity, is, weak) should not be a node.
                s_head = subj.root if hasattr(subj, "root") else subj
                o_head = obj.root if hasattr(obj, "root") else obj
                if s_head.pos_ not in ("NOUN", "PROPN", "NUM") or o_head.pos_ not in ("NOUN", "PROPN", "NUM"):
                    continue
                if pred == "be" or pred == "have":
                    pred = {"be": "is", "have": "has"}.get(pred, pred)
                triple = (s, pred, o)
                if triple not in out:
                    out.append(triple)
                if len(out) >= max_triples:
                    return out
        return out

    def _find_child(self, sent, head, deps):
        for child in head.children:
            if child.dep_ in deps and child.dep_ != "prep":
                return self._extend_noun(child)
        for child in head.children:
            if child.dep_ in {"prep"} and "prep" in deps:
                for gc in child.children:
                    if gc.dep_ in {"pobj"}:
                        return self._extend_noun(gc)
        return None

    def _extend_noun(self, token):
        """Expand a token to its full noun phrase span."""
        if token.dep_ in {"attr", "oprd"}:
            return token.doc[token.i: token.i + 1]
        idx = token.i
        start = token.i
        while start - 1 >= 0:
            t = token.doc[start - 1]
            if t.dep_ in {"det", "amod", "compound", "nummod", "nmod"} and t.i < idx:
                start = t.i
            else:
                break
            idx = t.i
        return token.doc[start: token.i + 1]

    def extract(self, text: str) -> LocalIEResult:
        if not text or not text.strip():
            return LocalIEResult()
        ents = self.entities(text)
        trips = self.triples(text)
        return LocalIEResult(entities=ents, triples=trips)


def build_local_openie() -> LocalIE:
    return LocalIE()
