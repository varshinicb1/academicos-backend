"""Question papers transcribed with their official answers, checked against
the source, and turned into bank source records.

Why transcription
-----------------
The board papers are scans (no text layer), and the text extraction of the
sample papers carries page furniture into stems and keys ("... Page 27 Page of
16 SECTION C ..."). 3,267 board records sat withheld because no relink could
tie them to their scheme. A reader that sees the page -- a vision model run
through the Antigravity CLI (agy) -- writes each question and its official
answer into one JSON file per paper (the format is `docs/transcription-format.md`).

Why it can be trusted
---------------------
Nothing a transcription says is served on its word. Every value point and
every objective key is looked for in the official scheme's own text layer
(`verify`); a question whose answer is not found there is refused with the
reason, never repaired by guessing. Questions come from the paper; answers
from the scheme; the model only copies. The records then go through the same
merge gates as every other source (`bank_merge.compose`).

The pilot (2024 Social Science 32/1/1, 2026-09-29): 39 questions, 80 marks,
208 of 208 value points found word for word in the scheme text; the one
unkeyed MCQ is one CBSE itself withdrew ("marks to be given if attempted").
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

# The types a transcription may use, and the bank's type for each.
TYPES = {
    "mcq": "mcq",
    "assertion_reason": "assertion_reason",
    "very_short_answer": "very_short_answer",
    "short_answer": "short_answer",
    "long_answer": "long_answer",
    "source_based": "case_study",
    "case_based": "case_study",
    "map": "map_based",
    "diagram": "diagram_based",
}
OBJECTIVE = ("mcq", "assertion_reason")

# A paper's printed subject -> (the bank's subject, the course variant). The bank
# files one subject per class (the syllabus files' names); a variant is kept in
# metadata so a teacher can still ask for Basic or Course B.
_SUBJECTS = (
    (r"math.*basic", "Mathematics", "Basic"), (r"math.*stand", "Mathematics", "Standard"),
    (r"^math", "Mathematics", None), (r"social", "Social Science", None), (r"^science", "Science", None),
    (r"english.*comm", "English", "Communicative"), (r"^english", "English", "Language and Literature"),
    (r"hindi\W*(course\W*)?b\W*$", "Hindi", "Course B"), (r"hindi\W*(course\W*)?a\W*$", "Hindi", "Course A"),
    (r"^hindi", "Hindi", None),
)


def canonical_subject(printed: Any) -> tuple[str, Optional[str]]:
    """("Mathematics", "Standard") for "Mathematics Standard"; unknown names as printed."""
    name = " ".join(str(printed or "").split())
    for pattern, subject, variant in _SUBJECTS:
        if re.search(pattern, name.lower()):
            return subject, variant
    return name, None
LETTERS = ("A", "B", "C", "D")


def _norm(s: Any) -> str:
    """Letters, digits and combining marks of any script, lower-cased, one space
    between words. ASCII-only made every Devanagari value point "", which
    `has` then found everywhere (review of #54)."""
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else " " for ch in unicodedata.normalize("NFKC", str(s or "")).lower())
    return " ".join(kept.split())


@dataclass
class SchemeText:
    """The official scheme's text, ready to search."""
    text: str
    words: frozenset = field(default_factory=frozenset)

    @classmethod
    def of(cls, raw: str) -> "SchemeText":
        t = _norm(raw)
        return cls(t, frozenset(t.split()))

    def locate(self, needle: str) -> Optional[int]:
        """Where `needle` starts in the scheme text (its longest verbatim
        stretch when the scheme wraps it), or None when it is not there."""
        n = _norm(needle)
        if not n:
            return None
        at = self.text.find(n)
        if at >= 0:
            return at
        if not self.has(needle):
            return None
        m = SequenceMatcher(None, n, self.text, autojunk=False).find_longest_match(0, len(n), 0, len(self.text))
        return max(0, m.b - m.a)

    def option_at(self, letter: Any, option: Any) -> Optional[int]:
        """Where the scheme prints this option under its letter ("(b) Italy",
        "b. Italy", "Ans: b Italy"), or None. The option's text alone is not
        enough: a numeric option or an assertion-reason option recurs under
        other questions (review of #54)."""
        letter, opt = _norm(letter), _norm(option)
        if not letter or not opt:
            return None
        m = re.search(rf"(?<![^ ]){re.escape(letter)} {re.escape(opt)}(?![^ ])", self.text)
        return m.start() if m else None

    def has(self, needle: str) -> bool:
        """True when `needle` is in the scheme: whole, or -- for text the
        scheme wraps across lines or columns -- nearly all its words present
        and a long stretch of it verbatim."""
        n = _norm(needle)
        if not n:
            return False           # nothing to find is not "found"
        if n in self.text:
            return True
        words = n.split()
        if len(words) <= 3:
            return all(w in self.words for w in words)
        if sum(1 for w in words if w in self.words) / len(words) < 0.85:
            return False
        m = SequenceMatcher(None, n, self.text, autojunk=False).find_longest_match(0, len(n), 0, len(self.text))
        return m.size >= min(len(n) * 0.5, 40)


_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def numbers(text: Any, signed: bool = False) -> set[str]:
    """The numbers a piece of working states, fractions kept whole
    ("\\frac{67}{12}" and "67/12" are both "67/12"; "4.0" is "4")."""
    s = _FRAC.sub(r"\1/\2", str(text or "")).replace(",", "").replace(" / ", "/")
    out = set()
    for n in _NUM.findall(s):
        if not signed:
            n = n.lstrip("-")
        if "/" not in n and "." in n:
            n = n.rstrip("0").rstrip(".") or "0"
        out.add(n)
    return out


def verified_by(q: dict, scheme: Optional[SchemeText], second: Optional[str]) -> Optional[str]:
    """How a transcribed answer was checked against the official scheme, or
    None when it could not be:

      scheme-text   every value point (an MCQ: its correct option's text) is in
                    the scheme's own text layer;
      second-solve  the scheme sets its working as images (Mathematics, most
                    numericals), so no text can be matched: an independent
                    solve by another model agrees -- every number of its final
                    answer is in the transcribed official working, and an MCQ's
                    letter is the transcribed one."""
    s = q.get("scheme") or {}
    if q.get("type") in OBJECTIVE:
        letter = s.get("correctOption")
        option = str((q.get("options") or {}).get(letter) or "")
        if scheme is not None and scheme.option_at(letter, option) is not None:
            return "scheme-text"
        if second is not None and str(second).strip().strip("()").upper()[:1] == letter:
            return "second-solve"
        return None
    points = [p for p in s.get("points") or [] if str(p.get("text") or "").strip()]
    if scheme is not None and all(scheme.has(p["text"]) for p in points):
        return "scheme-text"
    if second is not None and points:
        # The final answer, not every number of the working: "12 x 5 = 60"
        # must not verify a solve of 12 (review of #54). Signs are kept.
        last = str(points[-1].get("text") or "")
        final = numbers(last.rsplit("=", 1)[-1], signed=True) or numbers(last, signed=True)
        theirs = numbers(second, signed=True)
        if theirs and theirs <= final:
            return "second-solve"
    return None


def unverified_reason(q: dict, scheme: Optional[SchemeText], second: Optional[str] = None) -> Optional[str]:
    """Why a transcribed question (or an internal choice) cannot be served
    as transcribed, or None. `second` is an independent second solve's final
    answer for it, when there is one (`verified_by`)."""
    s = q.get("scheme") or {}
    if q.get("unreadable"):
        return "unreadable"
    if q.get("needsFigure"):
        return "figure-unavailable"
    if q.get("type") in OBJECTIVE:
        opts = q.get("options") or {}
        if sorted(opts) != list(LETTERS) or not all(str(v).strip() for v in opts.values()):
            return "options-missing"
        if s.get("correctOption") not in LETTERS:
            return "mcq-answer-unresolved"
    else:
        points = [p for p in s.get("points") or [] if str(p.get("text") or "").strip()]
        if not points:
            return "no-answer"
    # Nothing to check against -- a scanned scheme and no second solve -- is
    # not verified: the answer is refused, never served on the transcriber's word.
    return None if verified_by(q, scheme, second) else "scheme-unverified"


def integer_points(points: list[dict]) -> list[tuple[str, int]]:
    """Value points with whole marks: consecutive half-mark points are joined
    ("a; b" for 1 mark) as the phone's scheme holds whole marks only; a lone
    fraction left at the end joins the point before it."""
    out: list[tuple[str, float]] = []
    text, marks = [], 0.0
    for p in points:
        label = f"{p['part']} " if p.get("part") else ""
        text.append(label + " ".join(str(p.get("text") or "").split()))
        marks += float(p.get("marks") or 0)
        if abs(marks - round(marks)) < 1e-6 and marks >= 1:
            out.append(("; ".join(text), marks))
            text, marks = [], 0.0
    if text:
        if out:
            last_text, last_marks = out.pop()
            out.append((last_text + "; " + "; ".join(text), last_marks + marks))
        else:
            out.append(("; ".join(text), marks))
    return [(t, int(round(m))) for t, m in out]


def _parts(q: dict) -> list[dict]:
    return [{"partNumber": i + 1, "text": f"{p.get('label') or ''} {p.get('text') or ''}".strip(),
             "marks": int(round(float(p.get("marks") or 0)))}
            for i, p in enumerate(q.get("parts") or [])]


def record(q: dict, paper: dict, *, rid: str, bank_id: str, source: str, scheme_file: str,
           provenance: str = "cbse_marking_scheme", alternative_of: Optional[str] = None,
           built_at: str = "2026-09-29T00:00:00+00:00") -> dict:
    """The raw source record for one transcribed question, in the shape the
    CBE and SQP sources use (`bank_merge.normalise` reads it)."""
    s = q.get("scheme") or {}
    marks = int(round(float(q.get("marks") or 0))) or 1
    qtype = TYPES.get(str(q.get("type") or ""), "short_answer")
    subject, variant = canonical_subject(paper.get("subject"))
    stem = " ".join(str(q.get("stem") or "").split())
    meta: dict[str, Any] = {
        "questionKey": str(q.get("number") or ""),
        "alternativeOf": alternative_of,
        "joinedFrom": "transcription",
        "section": q.get("section"),
        "paperCode": paper.get("code"),
        "year": paper.get("year"),
        "printedSubject": paper.get("subject"),
        "courseVariant": variant,
        "difficultyInferred": True,
    }
    if q.get("passage"):
        meta["passage"] = str(q["passage"]).strip()
    if q.get("visuallyImpairedVariant"):
        meta["visuallyImpairedVariant"] = True
    scheme_meta: dict[str, Any] = {"schemeSource": scheme_file, "schemeCode": paper.get("code") or "",
                                   "schemeQNo": q.get("number"), "msPage": s.get("msPage")}
    if qtype in OBJECTIVE:
        opts = q.get("options") or {}
        stem = stem + " " + " ".join(f"({k}) {' '.join(str(opts[k]).split())}" for k in LETTERS)
        letter = s.get("correctOption")
        points = [{"description": f"Correct option {letter}: {opts.get(letter, '')}".strip(), "marks": marks,
                   "keyword": str(opts.get(letter, "")), "isRequired": True, "synonyms": [letter]}]
        model = f"({letter}) {opts.get(letter, '')}".strip()
        scheme_meta.update({"objective": True, "options": opts, "correctOption": letter})
        parts: list[dict] = []
    else:
        whole = integer_points([p for p in s.get("points") or [] if str(p.get("text") or "").strip()])
        points = [{"description": t, "marks": m, "keyword": "", "isRequired": s.get("anyN") is None,
                   "synonyms": []} for t, m in whole]
        model = "\n".join(t for t, _ in whole)
        if s.get("anyN"):
            scheme_meta["anyN"] = s["anyN"]
        if s.get("notes"):
            scheme_meta["notes"] = s["notes"]
        parts = _parts(q)
    return {
        "id": rid,
        "questionBankId": bank_id,
        "subject": subject,
        "grade": int(paper.get("class")),
        "chapterIds": [],
        "topic": "",
        "difficulty": "medium",
        "type": qtype,
        "stem": stem,
        "stemLatex": "",
        "parts": parts,
        "answerScheme": {
            "totalMarks": marks,
            "markingPoints": points,
            "rubricLevels": [],
            "commonErrors": [],
            "alternativeAnswers": [],
            "modelAnswer": model,
            "modelAnswerLatex": "",
            "hasPartialCredit": len(points) > 1,
            "metadata": scheme_meta,
            "provenance": provenance,
            "sourcePaperCode": paper.get("code") or "",
            "sourceDocumentId": bank_id,
        },
        "estimatedTimeMinutes": max(1, marks * 2),
        "marks": marks,
        "language": "en",
        "source": source,
        "qualityScore": 0.95,
        "tags": ["cbse", source.replace("cbse_", ""), str(paper.get("year") or "")],
        "createdAt": built_at,
        "updatedAt": built_at,
        "metadata": meta,
    }


def misplaced(located: list[tuple[str, list[int]]], slack: int = 200) -> set[str]:
    """Records whose value points are not where their question's answer is.

    The scheme prints answers in question order, so the answers' positions in
    it rise with the question numbers. A question whose points sit out of that
    order has another question's answer copied in -- the run-on the length rule
    (`answer-bleed`) guards against in extracted text. A short point ("Italy")
    can also occur elsewhere by chance, so each record is placed by the median
    of its points' positions, and `slack` characters of disorder are allowed.
    The longest run of records in scheme order is taken as right (the tighter
    one on a tie); every record outside it is returned."""
    import statistics
    rows = [(rid, statistics.median(pos)) for rid, pos in located if pos]
    n = len(rows)
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if rows[j][1] > rows[i][1] + slack:
                continue
            # On a tie, the tighter chain: the predecessor that sits earlier in the
            # scheme. Two records found at the same place (one answer, copied into
            # another question) then keep the later question, whose place it is.
            if best[j] + 1 > best[i] or (best[j] + 1 == best[i] and prev[i] >= 0
                                          and rows[j][1] < rows[prev[i]][1]):
                best[i], prev[i] = best[j] + 1, j
    keep: set[str] = set()
    i = max(range(n), key=lambda k: best[k]) if n else -1
    while i >= 0:
        keep.add(rows[i][0])
        i = prev[i]
    return {rid for rid, _ in rows} - keep


@dataclass
class Built:
    records: list[dict]
    refused: list[dict]             # {"id", "reason"}


def solve_key(q: dict, alternative: bool = False) -> str:
    """The key a second solve answers a question under: its number, "-vi" for
    the visually-impaired variant, "-or" for its internal choice."""
    return f"{q.get('number')}{'-vi' if q.get('visuallyImpairedVariant') else ''}{'-or' if alternative else ''}"


_BOARD_CODE = re.compile(r"\d+(?:/\d+)+[A-Za-z]?")


def _slug(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")


def paper_key(paper: dict) -> str:
    """The part of a record id that names its paper. A board paper's code
    (30/1/1 -> 30-1-1) names it alone. A sample paper's or booklet's code
    ("SQP 2025-26") is the same for every subject that year, so the subject,
    with its course variant, is named too: two subjects' question 1 were one
    id (124 collisions, 2026-09-29)."""
    code = str(paper.get("code") or "").strip()
    if _BOARD_CODE.fullmatch(code):
        return code.replace("/", "-")
    return "-".join(part for part in (_slug(paper.get("subject")), _slug(code)) if part)


def build(transcription: dict, *, source: str, scheme: Optional[SchemeText], scheme_file: str,
          id_prefix: str, provenance: str = "cbse_marking_scheme",
          second: Optional[dict[str, str]] = None) -> Built:
    """Every servable question of one transcribed paper (and each internal
    choice as its own record), and every refusal with its reason. `second`
    maps `solve_key` to an independent second solve's final answer."""
    paper = transcription.get("paper") or {}
    code = paper_key(paper)
    bank_id = f"{id_prefix}:{paper.get('year')}:{paper.get('class')}:{paper.get('subject')}:{code}"
    records, refused = [], []
    located: list[tuple[str, list[int]]] = []
    placed: set[str] = set()
    seen: set[str] = set()
    for q in transcription.get("questions") or []:
        number = str(q.get("number") or "").strip()
        suffix = "-vi" if q.get("visuallyImpairedVariant") else ""
        # The number as an id part: a note the transcriber kept beside it
        # ("37 (Visually Impaired)") is dropped -- the suffix already says so.
        num = _slug(re.sub(r"\(.*?\)", "", number)) or _slug(number)
        rid = f"{id_prefix}:{paper.get('year')}:{code}:q{num}{suffix}"
        if rid in seen:
            rid = f"{rid}-{len(seen)}"
        seen.add(rid)
        for item, item_id, alt_of in ((q, rid, None), (q.get("alternative"), f"{rid}-or", rid)):
            if not item:
                continue
            key = solve_key(q, alternative=alt_of is not None)
            item = {**item, "number": number, "section": q.get("section"),
                    "marks": item.get("marks") or q.get("marks"), "type": item.get("type") or q.get("type"),
                    "passage": item.get("passage") or q.get("passage")}
            answer2 = (second or {}).get(key)
            reason = unverified_reason(item, scheme, answer2)
            if reason:
                refused.append({"id": item_id, "reason": reason})
                continue
            rec = record(item, paper, rid=item_id, bank_id=bank_id, source=source,
                         scheme_file=scheme_file, provenance=provenance, alternative_of=alt_of)
            how = verified_by(item, scheme, answer2) if (scheme is not None or answer2 is not None) else None
            rec["answerScheme"]["metadata"]["verifiedBy"] = how
            records.append(rec)
            if how == "second-solve":
                # Verified by agreement, not by reading the scheme: checked tier.
                rec["answerScheme"]["metadata"]["officialProvenance"] = rec["answerScheme"]["provenance"]
                rec["answerScheme"]["provenance"] = "two_model_solved"
            if how == "scheme-text":
                if item.get("type") in OBJECTIVE:
                    at = scheme.option_at((item.get("scheme") or {}).get("correctOption"),
                                          (item.get("options") or {}).get((item.get("scheme") or {}).get("correctOption")))
                    pos = [at]
                else:
                    pos = [scheme.locate(p.get("text", "")) for p in (item.get("scheme") or {}).get("points") or []]
                pos = [x for x in pos if x is not None]
                located.append((item_id, pos))
                if pos:
                    placed.add(item_id)
    bad = misplaced(located)
    for r in records:
        # Placed: every point verified in the scheme, where this question's
        # answer is -- the evidence `bank_merge`'s length rule accepts in
        # place of a length bound for a long "any N of these" answer.
        r["answerScheme"]["metadata"]["placeVerified"] = (
            r["answerScheme"]["metadata"].get("verifiedBy") == "scheme-text" and r["id"] in placed
            and r["id"] not in bad)
    for rid in bad:
        refused.append({"id": rid, "reason": "scheme-out-of-place"})
    records = [r for r in records if r["id"] not in bad]
    return Built(records, refused)
