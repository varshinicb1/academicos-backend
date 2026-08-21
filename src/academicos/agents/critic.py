"""Self-RAG-style reflection critic over (query, answer, evidence).

Maps Self-RAG's reflection tokens onto prompt-based judgments from a Sarvam LLM:

  IsRel   [Retrieval]/[No Retrieval]  -> adaptive retrieval decision
  IsRel*  [Relevant]/[Irrelevant]     -> per-evidence relevance score
  IsSup   [Fully supported]/[Partially supported]/[No support] -> support score
  IsUse   [Utility:1..5]              -> utility score in [-1, +1]

Final per-path score: w_rel * relevance + w_sup * support + w_use * utility
(segment-wise best-path selection mirrors Self-RAG's beam over paths).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..llm.sarvam import SarvamLLM

log = logging.getLogger(__name__)

RETRIEVE_SYSTEM = (
    "Decide whether answering the user's question requires external reference "
    "material (documents, syllabus, rules) or can be answered from general knowledge alone. "
    "Respond with JSON: {\"retrieve\": true|false, \"reason\": \"...\"}"
)

CRITIC_SYSTEM = (
    "You are a strict evidence critic. Given a question, a candidate answer, and an "
    "evidence passage, judge three things and respond ONLY with JSON:\n"
    "{\"relevant\": true|false, \"support\": \"full\"|\"partial\"|\"none\", \"utility\": 1|2|3|4|5}\n"
    "relevant: does the passage bear on the question at all?\n"
    "support: full = answer is directly supported by the passage; partial = answer is "
    "consistent with but not fully stated in the passage; none = unsupported or contradictory.\n"
    "utility: how complete/useful the answer is (1 = useless, 5 = complete and direct)."
)


@dataclass
class ReflectionScores:
    retrieved: bool
    relevance: float = 0.0
    support: float = 0.0
    utility: float = 0.0
    detail: dict = field(default_factory=dict)

    def final(self, w_rel: float = 1.0, w_sup: float = 1.0, w_use: float = 0.5) -> float:
        return w_rel * self.relevance + w_sup * self.support + w_use * self.utility


class SarvamCritic:
    """Prompt-based Self-RAG reflection critic backed by a SarvamLLM."""

    def __init__(self, llm: SarvamLLM, *, w_rel: float = 1.0, w_sup: float = 1.0,
                 w_use: float = 0.5, retrieve_threshold: float = 0.5):
        self.llm = llm
        self.w_rel = w_rel
        self.w_sup = w_sup
        self.w_use = w_use
        self.retrieve_threshold = retrieve_threshold

    def decide_retrieve(self, query: str) -> bool:
        """IsRel: adaptive retrieval decision (Self-RAG [Retrieval] vs [No Retrieval])."""
        if not self.llm.available:
            return True
        try:
            out = self.llm.chat_json(
                [{"role": "system", "content": RETRIEVE_SYSTEM},
                 {"role": "user", "content": query}],
                temperature=0.0, max_tokens=128,
            )
            return bool(out.get("retrieve", True)) if isinstance(out, dict) else True
        except Exception as e:
            log.warning("critic.decide_retrieve failed (%s); defaulting to retrieve", e)
            return True

    def score(self, query: str, answer: str, evidence: str) -> ReflectionScores:
        """IsRel* + IsSup + IsUse for one (query, answer, evidence) path."""
        if not self.llm.available:
            return ReflectionScores(retrieved=True, relevance=1.0, support=0.5, utility=0.0)
        try:
            out = self.llm.chat_json(
                [
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": (
                        f"Question: {query}\n\nCandidate answer:\n{answer}\n\n"
                        f"Evidence passage:\n{evidence}"
                    )},
                ],
                temperature=0.0, max_tokens=128,
            )
            support_map = {"full": 1.0, "partial": 0.5, "none": 0.0}
            utility_map = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
            support = support_map.get(str(out.get("support", "")).lower(), 0.0)
            utility = utility_map.get(int(out.get("utility", 3)), 0.0)
            return ReflectionScores(
                retrieved=True,
                relevance=1.0 if out.get("relevant") else 0.0,
                support=support,
                utility=utility,
                detail={"raw": out},
            )
        except Exception as e:
            log.warning("critic.score failed (%s); neutral scores", e)
            return ReflectionScores(retrieved=True, relevance=0.5, support=0.5, utility=0.0, detail={"error": str(e)})
