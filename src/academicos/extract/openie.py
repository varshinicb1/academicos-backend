"""Open information extraction via Sarvam LLM (HippoRAG-style).

Two-stage pipeline mirroring HippoRAG's prompts:
  1. NER: extract named entities from the passage (JSON list).
  2. RE:  NER-conditioned relation extraction -> [subject, predicate, object] triples.

The output feeds the graph builder as Concept nodes + RELATED_TO edges so
PPR-style subgraph retrieval has something to walk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.sarvam import SarvamLLM

log = logging.getLogger(__name__)

NER_SYSTEM = (
    "Your task is to extract named entities and important concepts from the given paragraph. "
    "Include proper nouns AND domain terms, processes, and key concepts (e.g. 'photosynthesis', "
    "'chlorophyll', 'electrolysis') — any noun phrase a student would need to know. "
    "Do NOT extract generic words like 'process', 'example', 'plant' unless they are the topic. "
    "Respond with a JSON object of the form {\"named_entities\": [\"...\", \"...\"]}. "
    "Output at most 20 entities."
)

NER_ONE_SHOT_PARAGRAPH = (
    "Photosynthesis is the process by which green plants make their own food using sunlight, "
    "water, and carbon dioxide. Chlorophyll in leaves captures sunlight. The plant stores the "
    "extra food in fruits and roots."
)

NER_ONE_SHOT_OUTPUT = (
    '{"named_entities": ["Photosynthesis", "green plants", "sunlight", "water", '
    '"carbon dioxide", "Chlorophyll", "leaves", "fruits", "roots"]}'
)

RE_SYSTEM = (
    "Your task is to construct an RDF graph from the given passage and named entity list. "
    "Respond with a JSON object of the form {\"triples\": [[\"subject\", \"predicate\", \"object\"], ...]}. "
    "Each triple should contain at least one, preferably two, of the named entities. "
    "Resolve pronouns to their specific names. Keep predicates short and factual. "
    "Output at most 20 triples."
)

RE_FRAME = "Passage:\n```\n{passage}\n```\n\nNamed entities: {entities}\nTriples (JSON):"


@dataclass
class OpenIEResult:
    entities: list[str] = field(default_factory=list)
    triples: list[tuple[str, str, str]] = field(default_factory=list)


class OpenIE:
    """NER + relation extraction over chunk text using a SarvamLLM."""

    def __init__(self, llm: SarvamLLM, max_chars: int = 1800):
        self.llm = llm
        self.max_chars = max_chars

    def extract(self, text: str) -> OpenIEResult:
        if not self.llm.available:
            return OpenIEResult()
        passage = text.strip()[: self.max_chars]
        if not passage:
            return OpenIEResult()
        # sarvam-105b spends ~1.8k tokens reasoning before answering and the
        # starter tier caps completions at 4096; keep passages short so the
        # JSON answer still fits.
        ner_passage = passage[:600]
        entities = self._ner(ner_passage)
        triples = self._relations(ner_passage, entities) if entities else []
        return OpenIEResult(entities=entities, triples=triples)

    def _ner(self, passage: str) -> list[str]:
        try:
            out = self.llm.chat_json(
                [
                    {"role": "system", "content": NER_SYSTEM},
                    {"role": "user", "content": NER_ONE_SHOT_PARAGRAPH},
                    {"role": "assistant", "content": NER_ONE_SHOT_OUTPUT},
                    {"role": "user", "content": passage},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception:
            try:
                out = self.llm.chat_json(
                    [
                        {"role": "system", "content": NER_SYSTEM + " Be very brief: output the JSON only."},
                        {"role": "user", "content": passage[:300]},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
            except Exception as e:
                log.warning("openie NER failed: %s", e)
                return []
        try:
            if isinstance(out, list):
                return [str(e).strip() for e in out if str(e).strip()]
            if isinstance(out, dict):
                raw = out.get("named_entities", out.get("entities", []))
                if isinstance(raw, list):
                    return [str(e).strip() for e in raw if str(e).strip()]
            return []
        except Exception as e:
            log.warning("openie NER failed: %s", e)
            return []

    def _relations(self, passage: str, entities: list[str]) -> list[tuple[str, str, str]]:
        out: Any = None
        try:
            out = self.llm.chat_json(
                [
                    {"role": "system", "content": RE_SYSTEM},
                    {"role": "user", "content": RE_FRAME.format(passage=passage, entities=json_dumps(entities))},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
        except Exception:
            try:
                out = self.llm.chat_json(
                    [
                        {"role": "system", "content": RE_SYSTEM + " Be very brief: output the JSON only."},
                        {"role": "user", "content": RE_FRAME.format(passage=passage[:300], entities=json_dumps(entities[:8]))},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
            except Exception as e:
                log.warning("openie RE failed: %s", e)
                return []
        raw = out.get("triples", out) if isinstance(out, dict) else out
        triples: list[tuple[str, str, str]] = []
        if isinstance(raw, list):
            for t in raw:
                if isinstance(t, list) and len(t) >= 3:
                    triples.append((clean(t[0]), clean(t[1]), clean(t[2])))
        return triples


def clean(s: str) -> str:
    return " ".join(str(s).split())


def json_dumps(v) -> str:
    import json
    return json.dumps(v, ensure_ascii=False)
