"""NCERT textbook exercises, with answers that were checked before they are
served (docs/textbook-exercise-format.md).

Classes 1-10 textbooks print their exercises and, apart from some Mathematics
answer pages, no answers. A transcriber writes each chapter's questions with an
answer that says where it comes from; this module keeps only what it can check:

  book       the book prints the answer                -> ncert_textbook_answer
  grounded   every value point quotes the chapter, and the quote is found in
             the chapter's own text layer               -> textbook_grounded
  solved     worked out (sums, grammar, meanings) and the final answer agrees
             with an independent second solve           -> two_model_solved
  open       no single answer (discuss, find out, draw) -> never keyed

and refuses the rest with the reason. The question itself must be found in the
chapter's text (`stem-not-in-book`): the questions are the book's, never the
transcriber's.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from .transcribed import SchemeText, numbers

# The marks a question of each type usually carries in a CBSE paper; flagged
# `marksInferred`, since a textbook prints none.
MARKS = {"mcq": 1, "true_false": 1, "fill_blank": 1, "very_short_answer": 1, "match": 2,
         "short_answer": 2, "long_answer": 5, "case_based": 4}
BANK_TYPE = {"mcq": "mcq", "true_false": "very_short_answer", "fill_blank": "very_short_answer",
             "very_short_answer": "very_short_answer", "match": "short_answer", "short_answer": "short_answer",
             "long_answer": "long_answer", "case_based": "case_study"}
PROVENANCE = {"book": "ncert_textbook_answer", "grounded": "textbook_grounded", "solved": "two_model_solved"}
LETTERS = ("A", "B", "C", "D")


class BookText(SchemeText):
    """A chapter's text layer, searched like a scheme's. Devanagari is kept:
    Hindi books are checked in their own script."""

    @classmethod
    def of(cls, raw: str) -> "BookText":
        t = _norm(raw)
        book = cls(t, frozenset(t.split()))
        book.compact = t.replace(" ", "")
        return book

    def verbatim(self, needle: str) -> bool:
        """`needle` word for word in the chapter. Spacing is ignored, so a word
        the PDF split across a line ("hemi-" / "spheres") still matches; a
        paraphrase does not."""
        n = _norm(needle)
        return bool(n) and (n in self.text or n.replace(" ", "") in self.compact)


def _evidence_ok(book: "BookText", quote: Any) -> bool:
    """A quote copied word for word from the chapter, and long enough to say
    something: a single word is found in any chapter (review of #54)."""
    q = _norm(quote)
    return len(q) >= 12 and len(q.split()) >= 3 and book.verbatim(str(quote))


def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFC", str(s or "")).lower()
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", s).strip()


def _has(book: BookText, needle: str) -> bool:
    n = _norm(needle)
    if not n or n in book.text:
        return True
    words = n.split()
    if len(words) <= 3:
        return all(w in book.words for w in words)
    return sum(1 for w in words if w in book.words) / len(words) >= 0.85 and _longest_run(n, book.text) >= min(
        len(n) * 0.5, 40)


def _longest_run(a: str, b: str) -> int:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b)).size


def final_answer_key(s: Any) -> str:
    """A final answer reduced to what two solves must agree on: its numbers
    when it has any ("x = 15 cm", "15km" and "15" agree; units and words are
    how a solver phrased it), else its words, case and spacing ignored."""
    raw = unicodedata.normalize("NFC", str(s or ""))
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", raw.replace(",", ""))
    if nums:
        return ",".join(nums)
    return _norm(raw).replace(" ", "")


# A stem that points at a picture the student cannot see.
_REFERS_TO_PICTURE = re.compile(
    r"\b(shown|given|marked|drawn)\b.{0,40}\b(map|figure|fig|picture|diagram|table|graph|chart|image|photo|illustration)s?\b"
    r"|\b(look at|see|observe|study|refer to|use|using) the (above |following |given )?(map|figure|fig|picture|diagram|table|graph|chart|image|photo|illustration)s?\b"
    r"|\bin the (above |following |given )?(figure|fig|picture|diagram|table|graph|chart|image|photo|illustration)s?\b"
    r"|\b(above|below|adjacent|alongside)\W+(map|figure|fig|picture|diagram|table|graph|chart|image|photo|illustration)s?\b"
    r"|\b(map|figure|fig|picture|diagram|table|graph|chart|image|photo|illustration)s?\W+(below|above)\b",
    re.I)


def refusal(q: dict, book: BookText, second: Optional[dict[str, str]]) -> Optional[str]:
    a = q.get("answer") or {}
    src = a.get("source")
    if q.get("needsFigure"):
        return "figure-unavailable"
    if src == "open":
        return "generated-open" if q.get("origin") == "generated" else "open-ended"
    # "the arrows shown in the map", "the table below": the book's wording
    # points at a picture the student cannot see. The merge's own figure rule
    # still runs on every record at the one gate (bank_merge.unservable_reason);
    # this builder does not take that rule for itself (test_qbank_q1_q5: only
    # the two gates leave bank_merge).
    stem = str(q.get("stem") or "")
    if _REFERS_TO_PICTURE.search(stem):
        return "figure-referenced"
    # A book question must be the book's; a generated one is new by design, and
    # stands on its answer alone (grounded or solved, checked below).
    if q.get("origin") != "generated" and not _has(book, str(q.get("stem") or "")):
        return "stem-not-in-book"
    if q.get("type") == "mcq":
        opts = q.get("options") or {}
        if sorted(opts) != list(LETTERS):
            return "options-missing"
        if a.get("correctOption") not in LETTERS:
            return "mcq-answer-unresolved"
    if src == "grounded":
        points = a.get("points") or []
        if not points:
            return "no-answer"
        if not all(_evidence_ok(book, p.get("evidence")) for p in points):
            return "evidence-not-found"
        if not all(_supported(p.get("text"), p.get("evidence")) for p in points):
            return "answer-not-in-quote"
        if q.get("type") == "mcq":
            if not _one_option_stated(q, a.get("correctOption"),
                                      " ".join(str(p.get("evidence") or "") for p in points)):
                return "answer-not-in-quote"
        return None
    if src == "solved":
        if not str(a.get("finalAnswer") or "").strip() and a.get("correctOption") is None:
            return "no-answer"
        if second is None or q.get("ref") not in second:
            return "solve-unchecked"
        mine = a.get("correctOption") or a.get("finalAnswer")
        if final_answer_key(mine) != final_answer_key(second[q["ref"]]):
            return "solve-disagrees"
        return None
    if src == "book":
        # The book prints the answer: it must be found, word for word, where the
        # book prints it -- never taken on the transcriber's word, and never
        # claimed for a question written for a topic (review of #54).
        if q.get("origin") == "generated":
            return "generated-unkeyed"
        if not (a.get("points") or a.get("finalAnswer") or a.get("correctOption")):
            return "no-answer"
        quotes = [p.get("evidence") for p in a.get("points") or []] + [a.get("evidence")]
        quotes = [e for e in quotes if str(e or "").strip()]
        if not quotes or not all(_evidence_ok(book, e) for e in quotes):
            return "evidence-not-found"
        # The answer served must be the one the quote prints: a point in its own
        # quote; a final answer, or the chosen option's text, in the answer's quote.
        for p in a.get("points") or []:
            if p.get("evidence") and not _supported(p.get("text"), p.get("evidence")):
                return "answer-not-in-quote"
        whole = a.get("evidence") or " ".join(str(p.get("evidence") or "") for p in a.get("points") or [])
        if q.get("type") == "mcq":
            if not _one_option_stated(q, a.get("correctOption"), whole):
                return "answer-not-in-quote"
        elif a.get("finalAnswer") and not _supported(a.get("finalAnswer"), whole):
            return "answer-not-in-quote"
        return None
    return "no-answer"


def _supported(answer: Any, quote: Any) -> bool:
    """True when the quote states `answer`: its numbers all appear in the quote,
    and it is in the quote whole or three in four of its words are (a point may
    shorten the sentence it quotes, never change it). Review of #54: a verified
    quote beside an unrelated answer served the answer at the published tier."""
    text, where = _norm(answer), _norm(quote)
    if not text or not where:
        return False
    if not numbers(answer, signed=True) <= numbers(quote, signed=True):
        return False
    have = set(where.split())
    mine = set(text.split())
    # A negation must agree both ways: "Plants do not need sunlight" is not
    # stated by "Plants need sunlight", nor the reverse (re-review of #54).
    if (mine & _NEGATIONS) - have:
        return False
    if f" {text} " in f" {where} ":          # whole words: "sun" is not in "sunlight"
        return True
    if (have & _NEGATIONS) - mine:
        return False                          # the quote negates what the answer leaves out
    words = [w for w in text.split() if len(w) > 2 or not w.isascii()]
    return bool(words) and sum(w in have for w in words) / len(words) >= 0.75


# Words that turn a statement into its opposite (English and Hindi).
_NEGATIONS = frozenset({"not", "no", "never", "nor", "neither", "none", "nothing", "nobody", "cannot",
                        "नहीं", "न", "मत", "कभी"})


def _one_option_stated(q: dict, chosen_letter: Any, quote: Any) -> bool:
    """The quote states the chosen option and no other: a quote naming Earth
    and Venus keys neither (re-review of #54)."""
    opts = q.get("options") or {}
    if not _supported(opts.get(chosen_letter), quote):
        return False
    where = f" {_norm(quote)} "
    return not any(f" {_norm(text)} " in where for letter, text in opts.items()
                   if letter != chosen_letter and _norm(text))


BLOOMS = ("remember", "understand", "apply", "analyze", "evaluate", "create")
DIFFICULTIES = ("easy", "medium", "hard")


def _slug(s: Any) -> str:
    """Letters, digits and combining marks of any script, joined by "-": a
    Hindi chapter keeps its Devanagari (the vowel signs are marks, not letters,
    so a letters-only test would split every word)."""
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else "-" for ch in str(s or "").lower())
    return re.sub(r"-+", "-", kept).strip("-")


# NCERT's name for the class 3-5 book's subject since 2024; schools, CBSE's
# primary scheme and requirements QB-1 call the subject EVS.
BANK_SUBJECT = {"The World Around Us": "EVS"}


def bank_subject(printed: Any) -> str:
    name = str(printed or "").strip()
    return BANK_SUBJECT.get(name, name)


def chapter_id(subject: Any, grade: Any, title: Any) -> str:
    """A book chapter's id in the taxonomy's form ("mathematics-6/number-play"),
    for a chapter no syllabus file names under an id of its own."""
    return f"{_slug(bank_subject(subject))}-{int(grade)}/{_slug(title)}"


def record(q: dict, tb: dict, *, rid: str, built_at: str, chapter_ref: Optional[str] = None) -> dict:
    book, chapter = tb.get("book") or {}, tb.get("chapter") or {}
    topics = {t.get("id"): t for t in tb.get("topics") or []}
    topic = topics.get(q.get("topic")) or {}
    generated = q.get("origin") == "generated"
    a = q.get("answer") or {}
    qtype = str(q.get("type") or "short_answer")
    marks = MARKS.get(qtype, 2)
    stem = " ".join(str(q.get("stem") or "").split())
    parts = [{"partNumber": i + 1, "text": f"{p.get('label') or ''} {p.get('text') or ''}".strip(), "marks": 0}
             for i, p in enumerate(q.get("parts") or [])]
    if qtype == "mcq":
        opts = q["options"]
        stem += " " + " ".join(f"({k}) {' '.join(str(opts[k]).split())}" for k in LETTERS)
        letter = a["correctOption"]
        points = [{"description": f"Correct option {letter}: {opts[letter]}", "marks": 1, "keyword": str(opts[letter]),
                   "isRequired": True, "synonyms": [letter]}]
        model, parts = f"({letter}) {opts[letter]}", []
    else:
        texts = [str(p.get("text") or "").strip() for p in a.get("points") or [] if str(p.get("text") or "").strip()]
        if a.get("source") == "solved" and str(a.get("finalAnswer") or "").strip():
            # Only the final answer was checked (by the second solve), so only it
            # is served as the key: a working that says "Area = 50" beside a
            # checked 48 must not print (review of #54).
            texts = [str(a["finalAnswer"]).strip()]
        if not texts and a.get("finalAnswer"):
            texts = [str(a["finalAnswer"]).strip()]
        per = max(1, marks // max(1, len(texts))) if texts else marks
        points = [{"description": t, "marks": per, "keyword": "", "isRequired": True, "synonyms": []}
                  for t in texts[:marks] or texts[:1]]
        if points:
            points[-1]["marks"] = marks - per * (len(points) - 1)
        model = "\n".join(texts)
        working = str(a.get("working") or "")
        # The working is shown only when it reaches the checked answer.
        if working and (a.get("source") != "solved"
                        or numbers(a.get("finalAnswer"), signed=True) <= numbers(working, signed=True)):
            model = f"{model}\n\nWorking: {working}".strip()
    evidence = [{"quote": p.get("evidence"), "page": p.get("page")} for p in a.get("points") or [] if p.get("evidence")]
    return {
        "id": rid,
        "questionBankId": f"ncert-textbook:{book.get('code')}",
        "subject": bank_subject(book.get("subject")),
        "grade": int(book.get("class")),
        # The chapter the question was taken from or written for: certain, so
        # it is also the record's tag.
        "chapterIds": [chapter_ref] if chapter_ref else [],
        **({"taxonomyChapterId": chapter_ref, "tagMethod": "source_chapter", "tagConfidence": 1.0}
           if chapter_ref else {}),
        "topic": str(topic.get("title") or ""),
        "difficulty": q.get("difficulty") if q.get("difficulty") in DIFFICULTIES else "medium",
        **({"bloomLevel": q["bloom"]} if q.get("bloom") in BLOOMS else {}),
        "type": BANK_TYPE.get(qtype, "short_answer"),
        "stem": stem,
        "stemLatex": "",
        "parts": parts,
        "answerScheme": {
            "totalMarks": marks, "markingPoints": points, "rubricLevels": [], "commonErrors": [],
            "alternativeAnswers": [], "modelAnswer": model, "modelAnswerLatex": "",
            "hasPartialCredit": len(points) > 1,
            "metadata": {"answerSource": a.get("source"), "evidence": evidence,
                         **({"objective": True, "options": q["options"], "correctOption": a["correctOption"]}
                            if qtype == "mcq" else {})},
            "provenance": PROVENANCE[a["source"]],
            "sourcePaperCode": "", "sourceDocumentId": f"ncert-textbook:{book.get('code')}",
        },
        "estimatedTimeMinutes": max(1, marks * 2),
        "marks": marks,
        "language": "hi" if book.get("language") == "hi" else "en",
        # A question written for the topic is the generator's, never passed off as the book's.
        "source": "ai_generated" if generated else "ncert_textbook",
        # Below every official source (board 0.6-0.77, selection.py): a paper
        # draws CBSE's and NCERT's own questions first, never ours ahead of them
        # (QB-4's order; review of #54).
        "qualityScore": 0.5 if generated else 0.55,
        "tags": ["ncert", "generated" if generated else "textbook", str(book.get("title") or "")],
        "createdAt": built_at,
        "updatedAt": built_at,
        "metadata": {"questionKey": q.get("ref"), "bookCode": book.get("code"), "bookTitle": book.get("title"),
                     "bookSubject": book.get("subject"),
                     "chapterNumber": chapter.get("number"), "chapterName": chapter.get("title"),
                     "page": q.get("page"), "marksInferred": True,
                     # The book prints no difficulty; a generated question's is its writer's.
                     "difficultyInferred": not (generated and q.get("difficulty") in DIFFICULTIES),
                     "bloomInferred": q.get("bloom") not in BLOOMS,
                     "topicId": q.get("topic"), "topicTitle": topic.get("title"),
                     "origin": "generated" if generated else "book",
                     "joinedFrom": "textbook-transcription",
                     **({"passage": str(q["passage"]).strip()} if q.get("passage") else {})},
    }


@dataclass
class Built:
    records: list[dict]
    refused: list[dict]


def build(tb: dict, book_text: str, *, second: Optional[dict[str, str]] = None,
          chapter_ref: Optional[str] = None, built_at: str = "2026-09-29T00:00:00+00:00") -> Built:
    """Every checked question of one chapter, and every refusal with its reason.
    `second` maps a question's `ref` to an independent second solve's answer.
    `chapter_ref` is the chapter's id in the class's syllabus when it names this
    chapter; otherwise the chapter's own id (`chapter_id`)."""
    book, chapter = tb.get("book") or {}, tb.get("chapter") or {}
    if chapter_ref is None and chapter.get("title") and book.get("class"):
        chapter_ref = chapter_id(book.get("subject"), book.get("class"), chapter.get("title"))
    text = BookText.of(book_text)
    records, refused, seen = [], [], set()
    for i, q in enumerate(tb.get("questions") or []):
        ref = re.sub(r"[^A-Za-z0-9.]+", "-", str(q.get("ref") or i + 1)).strip("-")
        rid = f"ncert:tb:{book.get('code')}:{int(chapter.get('number') or 0):02d}:{ref}"
        while rid in seen:
            rid += "b"
        seen.add(rid)
        reason = refusal(q, text, second)
        if reason:
            refused.append({"id": rid, "reason": reason})
            continue
        records.append(record(q, tb, rid=rid, built_at=built_at, chapter_ref=chapter_ref))
    return Built(records, refused)
