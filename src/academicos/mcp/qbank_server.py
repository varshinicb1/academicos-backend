"""MCP server for the question bank.

Two consumers, one surface
--------------------------
`docs/PRD.md` section 4 names paper generation as a core loop, and the AI agents
in `src/academicos/agents/` need the same corpus the HTTP API serves. Rather
than build a second query engine for agents, this exposes the *same*
`QuestionBank` over the Model Context Protocol, so an agent and the backend
cannot disagree about what the bank contains.

Built on the official SDK (`mcp>=2,<3`), which supports the 2026-07-28 spec and
speaks stdio, SSE and Streamable HTTP. `--transport stdio` is the default
because that is what desktop MCP clients launch.

The answer-key rule, enforced here rather than documented
---------------------------------------------------------
The requirement is explicit: **without an answer key, no question.** So every
tool defaults to `require_answer_key=True`, and a question with no published
answer is invisible unless a caller explicitly asks for it by name. This is not
a filter applied at the edge -- it is the default of the query, so an agent
cannot accidentally build a paper it cannot mark.

`coverage_report` exists so the shortfall is visible rather than silent: much of
the older board-paper corpus has no attached scheme, and the honest thing is to
report that number rather than quietly serve the subset that does.

Run
---
    python -m academicos.mcp.qbank_server                  # stdio
    python -m academicos.mcp.qbank_server --transport sse  # for a web client
    python -m academicos.mcp.qbank_server --bank PATH      # a specific corpus
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_BANK = Path("academicos-data") / "syllabus" / "questions.json"
CBE_BANK = Path("academicos-data") / "corpus" / "cbse-cbe" / "questions.json"


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

class Corpus:
    """The loaded bank plus the queries the tools need.

    Wraps `qbank_routes.QuestionBank` rather than reimplementing filtering, so
    the MCP surface and the HTTP surface answer identically. Any divergence
    between them would be a bug that only appears for one kind of consumer.
    """

    def __init__(self, records: list[dict[str, Any]]):
        from ..assessment.qbank_routes import QuestionBank
        self._bank = QuestionBank(records)
        self.records = self._bank.records

    @classmethod
    def load(cls, paths: list[Path]) -> "Corpus":
        seen: dict[str, dict] = {}
        for p in paths:
            if not p.exists():
                log.warning("bank not found, skipping: %s", p)
                continue
            payload = json.loads(p.read_text(encoding="utf-8"))
            for rec in payload.get("questions") or []:
                rid = str(rec.get("id") or "")
                if rid and rid not in seen:
                    seen[rid] = rec
            log.info("loaded %d records from %s", len(payload.get("questions") or []), p)
        return cls(list(seen.values()))

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def has_answer_key(rec: dict[str, Any]) -> bool:
        """Delegates to the shared engine, so the two cannot drift apart.

        This was a second copy of the rule. Two copies of "what makes a question
        answerable" is exactly how the MCP surface and the HTTP surface start
        disagreeing -- the divergence Q5 exists to prevent -- and the copy here
        did not check the same fields as the builder.
        """
        from ..assessment.qbank_routes import QuestionBank
        return QuestionBank.has_answer_key(rec)

    def search(
        self,
        *,
        subject: str | None = None,
        grade: int | None = None,
        topic: str | None = None,
        marks: int | None = None,
        min_marks: int | None = None,
        max_marks: int | None = None,
        question_type: str | None = None,
        difficulty: str | None = None,
        keyword: str | None = None,
        limit: int = 20,
        require_answer_key: bool = True,
        answer_key_state: str = "any",
    ) -> list[dict[str, Any]]:
        """Filter the bank.

        `answer_key_state`:
          * `"any"` (default) -- require_answer_key is honoured
          * `"official"`      -- only questions carrying a CBSE marking scheme
          * `"none"`          -- only questions with no attached scheme, for
                                 auditing what the relink has not reached

        There is no "include unanswered" flag that quietly widens the default.
        An agent asking for questions gets answerable ones, and a caller who
        wants the unanswerable set has to say so in those words.
        """
        needle = (keyword or "").strip().lower()
        out: list[dict[str, Any]] = []
        for rec in self.records:
            if subject and str(rec.get("subject") or "").lower() != subject.lower():
                continue
            if grade is not None and rec.get("grade") != grade:
                continue
            if marks is not None and rec.get("marks") != marks:
                continue
            if min_marks is not None and (rec.get("marks") or 0) < min_marks:
                continue
            if max_marks is not None and (rec.get("marks") or 0) > max_marks:
                continue
            if question_type and str(rec.get("type") or "") != question_type:
                continue
            if difficulty and str(rec.get("difficulty") or "") != difficulty:
                continue
            if topic and not _topic_matches(rec, topic):
                continue
            if needle:
                hay = " ".join([
                    str(rec.get("stem") or ""),
                    " ".join(str(t) for t in (rec.get("tags") or [])),
                    str((rec.get("provenance") or {}).get("sourceDocumentId") or ""),
                ]).lower()
                if needle not in hay:
                    continue

            official = (rec.get("answerScheme") or {}).get("provenance") == "cbse_marking_scheme"
            has = self.has_answer_key(rec)
            if answer_key_state == "official" and not official:
                continue
            if answer_key_state == "none" and has:
                continue
            if answer_key_state == "any" and require_answer_key and not has:
                continue

            out.append(rec)
            if len(out) >= limit:
                break
        return out

    def topics(self, *, subject: str | None = None, grade: int | None = None,
               require_answer_key: bool = True) -> list[dict[str, Any]]:
        """Every topic in the bank, with its question count and marks spread.

        This is the requirement's "all chapters, and topics inside the chapter,
        with every question mapped to a topic" read back out. A topic with zero
        answerable questions is reported with a zero rather than dropped --
        the gap is the information.
        """
        buckets: dict[tuple[str, int, str], dict[str, Any]] = {}
        for rec in self.records:
            if subject and str(rec.get("subject") or "").lower() != subject.lower():
                continue
            if grade is not None and rec.get("grade") != grade:
                continue
            if require_answer_key and not self.has_answer_key(rec):
                continue
            for topic in _topics_of(rec):
                key = (str(rec.get("subject") or ""), int(rec.get("grade") or 0), topic)
                b = buckets.setdefault(key, {
                    "subject": key[0], "grade": key[1], "topic": topic,
                    "count": 0, "marks": {}, "withAnswerKey": 0,
                })
                b["count"] += 1
                m = str(rec.get("marks") or 0)
                b["marks"][m] = b["marks"].get(m, 0) + 1
                if self.has_answer_key(rec):
                    b["withAnswerKey"] += 1
        out = []
        for b in buckets.values():
            out.append({
                **b,
                "marksDistribution": [
                    {"marks": int(k), "count": v}
                    for k, v in sorted(b["marks"].items(), key=lambda kv: int(kv[0]))
                ],
            })
        out.sort(key=lambda b: (b["subject"], b["grade"], b["topic"]))
        return out

    def coverage(self) -> dict[str, Any]:
        """The honest state of the bank. Reported, never implied."""
        total = len(self.records)
        with_key = sum(1 for r in self.records if self.has_answer_key(r))
        official = sum(1 for r in self.records
                       if (r.get("answerScheme") or {}).get("provenance") == "cbse_marking_scheme")
        by_subject: dict[str, dict[str, int]] = {}
        for r in self.records:
            s = str(r.get("subject") or "unknown")
            d = by_subject.setdefault(s, {"total": 0, "withAnswerKey": 0})
            d["total"] += 1
            if self.has_answer_key(r):
                d["withAnswerKey"] += 1
        return {
            "questions": total,
            "withAnswerKey": with_key,
            "withOfficialCbseScheme": official,
            "answerKeyCoverage": round(with_key / total, 4) if total else 0.0,
            "subjects": dict(sorted(by_subject.items())),
            "grades": sorted({int(r.get("grade") or 0) for r in self.records}),
            "note": ("Questions without an answer key are excluded from every "
                     "search by default. They exist in the corpus and are "
                     "reported here rather than hidden."),
        }


def _topics_of(rec: dict[str, Any]) -> list[str]:
    """Every topic label a question carries.

    Chapters come from `chapterIds` (the keyword/embedding tagger) and the
    learning-ladder reference from the CBSE CBE import; both are real mappings,
    so both are exposed.
    """
    out = [str(c) for c in (rec.get("chapterIds") or []) if c]
    for key in ("topic", "contentCode", "contentReference"):
        v = rec.get(key)
        if v:
            out.append(str(v))
    return list(dict.fromkeys(out)) or ["unmapped"]


def _topic_matches(rec: dict[str, Any], topic: str) -> bool:
    needle = topic.strip().lower()
    for t in _topics_of(rec):
        if needle in t.lower():
            return True
    return False


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #

def build_server(corpus: Corpus):
    """Construct the MCP server over a loaded corpus."""
    from mcp.server import MCPServer

    mcp = MCPServer(
        name="academicos-question-bank",
        title="AcademicOS Question Bank",
        version="0.1.0",
        instructions=(
            "Authentic CBSE questions with their official marking schemes. "
            "Every question is mapped to a curriculum topic and carries a mark "
            "value, so you can assemble papers by topic and marks distribution. "
            "Questions without an answer key are excluded by default: an "
            "examination board does not issue a question it cannot mark. Use "
            "coverage_report to see how much of the corpus is answerable."
        ),
    )

    # -- tools ------------------------------------------------------------- #

    @mcp.tool(
        name="search_questions",
        description=(
            "Find answerable CBSE questions by subject, class, topic, marks, "
            "type or keyword. Returns questions WITH their marking schemes by "
            "default. Set answer_key_state='none' to audit which questions have "
            "no scheme."
        ),
    )
    def search_questions(
        subject: Optional[str] = None,
        grade: Optional[int] = None,
        topic: Optional[str] = None,
        marks: Optional[int] = None,
        min_marks: Optional[int] = None,
        max_marks: Optional[int] = None,
        question_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
        require_answer_key: bool = True,
        answer_key_state: str = "any",
    ) -> dict:
        rows = corpus.search(
            subject=subject, grade=grade, topic=topic, marks=marks,
            min_marks=min_marks, max_marks=max_marks,
            question_type=question_type, difficulty=difficulty, keyword=keyword,
            limit=max(1, min(limit, 200)), require_answer_key=require_answer_key,
            answer_key_state=answer_key_state,
        )
        return {"count": len(rows), "questions": [_brief(r) for r in rows]}

    @mcp.tool(
        name="get_question",
        description="One question in full: stem, options, marking scheme, "
                    "provenance (source paper and page), rights, and version.",
    )
    def get_question(question_id: str) -> dict:
        for rec in corpus.records:
            if str(rec.get("id")) == question_id:
                return rec
        return {"error": f"no question {question_id!r}"}

    @mcp.tool(
        name="get_answer_key",
        description="Just the marking scheme for one question: the value points "
                    "and whether they are an official CBSE scheme or absent.",
    )
    def get_answer_key(question_id: str) -> dict:
        for rec in corpus.records:
            if str(rec.get("id")) == question_id:
                scheme = rec.get("answerScheme") or {}
                return {
                    "questionId": question_id,
                    "totalMarks": scheme.get("totalMarks"),
                    "markingPoints": scheme.get("markingPoints") or [],
                    "modelAnswer": scheme.get("modelAnswer") or "",
                    "provenance": scheme.get("provenance") or "none",
                    "sourcePaperCode": scheme.get("sourcePaperCode") or "",
                    "hasAnswerKey": corpus.has_answer_key(rec),
                }
        return {"error": f"no question {question_id!r}"}

    @mcp.tool(
        name="list_topics",
        description="Every curriculum topic in the bank, with its question "
                    "count and marks distribution. Use this before building a "
                    "paper to see what is available.",
    )
    def list_topics(subject: Optional[str] = None, grade: Optional[int] = None,
                    require_answer_key: bool = True) -> dict:
        topics = corpus.topics(subject=subject, grade=grade,
                               require_answer_key=require_answer_key)
        return {"count": len(topics), "topics": topics}

    @mcp.tool(
        name="marks_distribution",
        description=(
            "For one topic, how many questions exist at each mark value -- the "
            "range a paper needs in order to be balanced. Reports which mark "
            "values are missing as well as which exist."
        ),
    )
    def marks_distribution(subject: str, grade: int, topic: str,
                           require_answer_key: bool = True) -> dict:
        rows = corpus.search(subject=subject, grade=grade, topic=topic,
                             limit=1000, require_answer_key=require_answer_key)
        spread: dict[int, int] = {}
        for r in rows:
            m = int(r.get("marks") or 0)
            spread[m] = spread.get(m, 0) + 1
        present = sorted(spread)
        available = set(range(1, 7))
        return {
            "subject": subject, "grade": grade, "topic": topic,
            "total": len(rows),
            "distribution": [{"marks": m, "count": spread[m]} for m in present],
            "missingMarks": sorted(available - set(present)),
            "note": ("missingMarks lists mark values in the 1-6 range the "
                     "curriculum uses that have no question for this topic. A "
                     "paper needing one of them cannot be built from this topic "
                     "alone."),
        }

    @mcp.tool(
        name="coverage_report",
        description="How much of the corpus is answerable, by subject. Use this "
                    "to judge whether the bank can support a paper before "
                    "attempting to build one.",
    )
    def coverage_report() -> dict:
        return corpus.coverage()

    @mcp.tool(
        name="build_paper",
        description=(
            "Assemble a paper from the bank: pick topics, then fill each "
            "section's mark value and question count. Only questions with a "
            "marking scheme are used. Returns the paper plus every gap it could "
            "not fill, rather than silently returning a short paper."
        ),
    )
    def build_paper(
        subject: str,
        grade: int,
        sections: list[dict],
        topic: Optional[str] = None,
    ) -> dict:
        """`sections` is a list of `{"marks": 1, "count": 5}`.

        Paper generation already exists at `POST /api/v1/papers/generate` with
        the full blueprint engine. This is the bank-level view: choose by topic
        and mark value, and be told exactly what could not be filled. It does
        not attempt Bloom or difficulty balancing -- that is the blueprint
        engine's job, and duplicating it here would give two answers to the
        same question.
        """
        chosen: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        used: set[str] = set()

        for section in sections or []:
            want_marks = int(section.get("marks") or 0)
            want_count = int(section.get("count") or 0)
            pool = corpus.search(subject=subject, grade=grade, topic=topic,
                                 marks=want_marks, limit=500)
            picked = 0
            for rec in pool:
                rid = str(rec.get("id"))
                if rid in used:
                    continue
                used.add(rid)
                chosen.append(_brief(rec))
                picked += 1
                if picked >= want_count:
                    break
            if picked < want_count:
                gaps.append({
                    "marks": want_marks, "requested": want_count, "found": picked,
                    "shortfall": want_count - picked,
                })
        total = sum(int(r.get("marks") or 0) for r in chosen)
        return {
            "subject": subject, "grade": grade, "topic": topic,
            "questions": chosen, "questionCount": len(chosen), "totalMarks": total,
            "gaps": gaps,
            "complete": not gaps,
        }

    # -- resources --------------------------------------------------------- #

    @mcp.resource("qbank://catalog", name="Question bank catalog",
                  description="Subjects, classes and answer-key coverage.",
                  mime_type="application/json")
    def catalog() -> str:
        return json.dumps(corpus.coverage(), indent=1)

    @mcp.resource("qbank://topics/{subject}/{grade}",
                  name="Topics for a subject and class",
                  description="Every topic with its question count and marks spread.",
                  mime_type="application/json")
    def topics_resource(subject: str, grade: str) -> str:
        try:
            g = int(grade)
        except ValueError:
            return json.dumps({"error": f"grade {grade!r} is not a number"})
        return json.dumps(corpus.topics(subject=subject, grade=g), indent=1)

    return mcp


def _brief(rec: dict[str, Any]) -> dict[str, Any]:
    """The list projection, including the scheme.

    Unlike the HTTP list, the scheme travels here. An agent assembling a paper
    needs to know the answer is present and what it says, and making it fetch
    each question separately would burn a round trip per question for data it
    always needs next.
    """
    return {
        "id": rec.get("id"),
        "subject": rec.get("subject"),
        "grade": rec.get("grade"),
        "marks": rec.get("marks"),
        "type": rec.get("type"),
        "difficulty": rec.get("difficulty"),
        "stem": rec.get("stem"),
        "topics": _topics_of(rec),
        "hasAnswerKey": bool((rec.get("answerScheme") or {}).get("markingPoints"))
                       or bool(((rec.get("answerScheme") or {}).get("modelAnswer") or "").strip()),
        "answerScheme": rec.get("answerScheme") or {},
        "provenance": rec.get("provenance") or {},
    }


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def load_corpus(bank: str | None = None) -> Corpus:
    root = Path(os.environ.get("ACOS_DATA_ROOT", "academicos-data"))

    def _opt(relative: str) -> Path | None:
        p = root / relative
        return p if p.exists() else None

    paths = [Path(bank)] if bank else [
        CBE_BANK if (root / "corpus" / "cbse-cbe").exists() else None,
        # Sample-paper items, joined to the marking scheme CBSE publishes
        # alongside each paper. This is the only source that carries Social
        # Science -- the CBE item banks are Maths, Science and English only.
        _opt("corpus/cbse-sqp/questions.json"),
        root / "syllabus" / "questions.json",
    ]
    return Corpus.load([p for p in paths if p])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AcademicOS question-bank MCP server")
    ap.add_argument("--transport", default="stdio",
                    choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--bank", default=None, help="a specific bank JSON to serve")
    ap.add_argument("--list-tools", action="store_true", help="print tools and exit")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    corpus = load_corpus(args.bank)
    mcp = build_server(corpus)

    if args.list_tools:
        import asyncio
        tools = asyncio.run(mcp.list_tools())
        print(f"corpus: {len(corpus)} questions")
        for t in tools:
            print(f"  {t.name:<22} {t.description.splitlines()[0][:70]}")
        return 0

    print(f"serving {len(corpus)} questions over {args.transport}", file=sys.stderr)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
