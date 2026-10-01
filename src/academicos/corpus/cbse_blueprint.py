"""What a CBSE question paper looks like, as data.

Every other builder in `corpus/` answers "is this question well formed?". This one
answers the question that comes first: "is this the right question to ask in this
paper?". A question that is individually perfect and belongs to no section of the
paper is not a question the paper needs, and a paper assembled from individually
perfect questions that do not add up to the board's structure is not a CBSE paper.

So the format lives here, per grade band and subject, rather than in a prompt.
A prompt cannot be checked; this can.

What the board actually does, and where each rule below comes from:

* 80 marks of theory and 20 of internal assessment, for every main subject
  (CBSE circular on the class 10/12 theory-practical split, 2024-10-24).
* 50% of the questions competency-based, 20% MCQ, 30% short and long answer
  (CBSE class 10 exam pattern 2025-26, collegedunia mirror of the board PDF).
* Roughly 33% of the questions in each section carry an internal choice
  (PW, "CBSE Class 10 Exam Pattern 2025-26").
* Sections run A to E, and the section breakdown is per subject, not universal:
  Social Science adds an F for map questions; English has no Section A MCQ run at
  all but splits Reading / Grammar / Literature; Science spends four of its
  Section A marks on assertion-reason rather than plain MCQ.
* Competency questions are set on a real-life situation the student has to read
  into, not on a definition. That is why `case_based` and `source_based` are their
  own types here and not a flavour of the long answer.

The one thing this module will not do is invent a format. A grade band and
subject combination with no entry in `_BLUEPRINTS` raises, rather than falling
back to a default -- an unexamined fallback is how a school ends up printing a
paper in a shape no marker has ever seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The question shapes CBSE sets, in the words the marking schemes use. These are
# the values that land in a question's `type`, so they match what the bank already
# stores (very_short_answer, short_answer, long_answer, mcq, case_study).
MCQ = "mcq"
ASSERTION_REASON = "assertion_reason"
VSA = "very_short_answer"          # Section B, 2 marks
SA = "short_answer"                # Section C, 3 marks
LA = "long_answer"                 # Section D, 5 marks
CASE = "case_study"                # Section E, 4 marks
MAP = "map_based"                  # Social Science Section F, 5 marks

QUESTION_TYPES = (MCQ, ASSERTION_REASON, VSA, SA, LA, CASE, MAP)

# The cognitive levels the board's competency focus is written against. A
# question is only "competency-based" if it climbs past `understand`.
BLOOM_ORDER = ("remember", "understand", "apply", "analyze", "evaluate", "create")
COMPETENCY_FLOOR = "apply"

# The board's own 50 / 20 / 30 split of the theory paper.
COMPETENCY_SHARE = 0.50
MCQ_SHARE = 0.20
SUBJECTIVE_SHARE = 0.30

# "Around 33% of the questions in each section will have an internal choice."
INTERNAL_CHOICE_SHARE = 0.33

THEORY_MARKS = 80
INTERNAL_ASSESSMENT_MARKS = 20


@dataclass(frozen=True)
class Section:
    """One section of the paper: what it holds and what each question is worth."""

    key: str                      # "A", "B", ... as printed on the paper
    name: str
    qtype: str
    count: int
    marks: int
    # Word limits the board prints in the rubric, where it prints one at all.
    answer_words: int | None = None
    internal_choice: int = 0      # how many of `count` carry a choice
    # True where a section that exists only to hold source material -- a passage,
    # a map -- sits above the questions that use it.
    carries_source: bool = False

    @property
    def section_marks(self) -> int:
        return self.count * self.marks

    def json_schema(self) -> dict:
        """The answer this section wants, for a generation call."""
        out = {"type": {"type": "string", "enum": [self.qtype]},
               "marks": {"type": "integer", "const": self.marks}}
        return out


@dataclass(frozen=True)
class Blueprint:
    """A subject's paper for one grade band."""

    subject: str
    grade_band: str                # "primary" (1-5), "middle" (6-8), "board" (9-10)
    theory_marks: int
    sections: tuple[Section, ...]
    duration_minutes: int = 180
    notes: str = ""

    @property
    def total_marks(self) -> int:
        return sum(s.section_marks for s in self.sections)

    @property
    def total_questions(self) -> int:
        return sum(s.count for s in self.sections)

    def by_type(self) -> dict[str, list[Section]]:
        out: dict[str, list[Section]] = {}
        for s in self.sections:
            out.setdefault(s.qtype, []).append(s)
        return out

    def shares(self) -> dict[str, float]:
        """The board's 50/20/30, measured against this paper rather than assumed."""
        total = self.total_marks or 1
        mcq = sum(s.section_marks for s in self.sections
                  if s.qtype in (MCQ, ASSERTION_REASON))
        case = sum(s.section_marks for s in self.sections
                   if s.qtype in (CASE, MAP))
        return {"mcq": mcq / total,
                "competency": (mcq + case) / total,
                "subjective": (total - mcq - case) / total}

    def answer_words(self, qtype: str, marks: int) -> int:
        """How long an answer may run. The board's rule of thumb is about 1.5
        words a mark, floored at a line so a 1-mark MCQ is never asked to prose."""
        for s in self.sections:
            if s.qtype == qtype and s.marks == marks and s.answer_words:
                return s.answer_words
        return max(1, int(marks * 1.5))


def _primary(subject: str) -> Blueprint:
    """Classes 1-5. No board paper, but schools set to the same competency shape:
    a run of one-mark objective questions, then short answers, then a case."""
    # A grade-1 paper is not a board paper, but it is still not half a page of
    # recall: the board's competency framework applies from Class 1, so the
    # one-mark run is held to a third of the marks and the case work is real.
    return Blueprint(
        subject=subject, grade_band="primary", theory_marks=30,
        sections=(
            Section("A", "Objective", MCQ, 10, 1),
            Section("B", "Very Short Answer", VSA, 5, 2, answer_words=25,
                    internal_choice=2),
            Section("C", "Case Based", CASE, 2, 5, answer_words=60,
                    internal_choice=1),
        ),
        duration_minutes=60,
        notes="Classes 1-5 have no board paper; schools set to this competency shape.")


def _middle(subject: str) -> Blueprint:
    """Classes 6-8: the board structure, scaled to a 40-mark school paper."""
    if subject == "Social Science":
        return Blueprint(
            subject=subject, grade_band="middle", theory_marks=40,
            sections=(
                Section("A", "Objective", MCQ, 12, 1),
                Section("B", "Very Short Answer", VSA, 4, 2, answer_words=25,
                        internal_choice=2),
                Section("C", "Short Answer", SA, 4, 3, answer_words=45,
                        internal_choice=1),
                Section("D", "Case Based", CASE, 1, 4, answer_words=50),
            ),
            duration_minutes=90)
    if subject in ("English", "Hindi"):
        return Blueprint(
            subject=subject, grade_band="middle", theory_marks=40,
            sections=(
                Section("A", "Objective", MCQ, 10, 1),
                Section("B", "Very Short Answer", VSA, 5, 2, answer_words=25,
                        internal_choice=2),
                Section("C", "Short Answer", SA, 5, 3, answer_words=45,
                        internal_choice=2),
            ),
            duration_minutes=90)
    return Blueprint(
        subject=subject, grade_band="middle", theory_marks=40,
        sections=(
            Section("A", "Objective", MCQ, 12, 1),
            Section("B", "Very Short Answer", VSA, 4, 2, answer_words=25,
                    internal_choice=1),
            Section("C", "Short Answer", SA, 4, 3, answer_words=45,
                    internal_choice=1),
            Section("D", "Case Based", CASE, 1, 4, answer_words=50),
        ),
        duration_minutes=90)


def _board_maths() -> Blueprint:
    return Blueprint(
        subject="Mathematics", grade_band="board", theory_marks=80,
        sections=(
            Section("A", "Objective", MCQ, 20, 1),
            Section("B", "Very Short Answer", VSA, 5, 2, answer_words=25,
                    internal_choice=2),
            Section("C", "Short Answer", SA, 6, 3, answer_words=45,
                    internal_choice=2),
            Section("D", "Long Answer", LA, 4, 5, answer_words=75,
                    internal_choice=1),
            Section("E", "Case Based", CASE, 3, 4, answer_words=50,
                    internal_choice=1),
        ),
        notes="CBSE class 10 Mathematics 2025-26: 20+10+18+20+12.")


def _board_science() -> Blueprint:
    return Blueprint(
        subject="Science", grade_band="board", theory_marks=80,
        sections=(
            # Four of Section A's twenty marks go on assertion-reason, not plain MCQ.
            Section("A", "Objective", MCQ, 16, 1),
            Section("A", "Assertion-Reason", ASSERTION_REASON, 4, 1),
            Section("B", "Very Short Answer", VSA, 6, 2, answer_words=25,
                    internal_choice=2),
            Section("C", "Short Answer", SA, 7, 3, answer_words=45,
                    internal_choice=2),
            Section("D", "Long Answer", LA, 3, 5, answer_words=75,
                    internal_choice=1),
            Section("E", "Case Based", CASE, 3, 4, answer_words=50,
                    internal_choice=1),
        ),
        notes="CBSE class 10 Science 2025-26: 16+4 MCQ/AR, then 12+21+15+12.")


def _board_social_science() -> Blueprint:
    return Blueprint(
        subject="Social Science", grade_band="board", theory_marks=80,
        sections=(
            Section("A", "Objective", MCQ, 20, 1),
            Section("B", "Very Short Answer", VSA, 4, 2, answer_words=25,
                    internal_choice=1),
            Section("C", "Short Answer", SA, 5, 3, answer_words=45,
                    internal_choice=2),
            Section("D", "Long Answer", LA, 4, 5, answer_words=75,
                    internal_choice=2),
            Section("E", "Case Based", CASE, 3, 4, answer_words=50,
                    internal_choice=1),
            Section("F", "Map Based", MAP, 1, 5, answer_words=30,
                    carries_source=True),
        ),
        notes="CBSE class 10 Social Science 2025-26; Section F is the map work.")


def _board_language(subject: str) -> Blueprint:
    """English and Hindi do not run a Section A objective block at all. They read,
    then they write, then they answer on the text -- so a bank built on MCQ runs
    would be the wrong bank for these."""
    return Blueprint(
        subject=subject, grade_band="board", theory_marks=80,
        sections=(
            Section("A", "Reading", MCQ, 20, 1, carries_source=True),
            Section("B", "Grammar", MCQ, 10, 1),
            Section("C", "Creative Writing", SA, 1, 10, answer_words=150),
            Section("D", "Literature", SA, 4, 10, answer_words=150,
                    internal_choice=1),
        ),
        duration_minutes=180,
        notes="CBSE class 10 language paper: Reading 20, Grammar 10, "
              "Writing 10, Literature 40. No objective run in Section A.")


_BLUEPRINTS: dict[tuple[str, str], Blueprint] = {}


def _register(blueprints: list[Blueprint]) -> None:
    for b in blueprints:
        _BLUEPRINTS[(b.subject, b.grade_band)] = b


_register([
    # Primary: one shape per subject, scaled to a 30-mark school paper.
    _primary("Mathematics"), _primary("Science"), _primary("Social Science"),
    _primary("English"), _primary("Hindi"),
    # Middle: the board structure at 40 marks.
    _middle("Mathematics"), _middle("Science"), _middle("Social Science"),
    _middle("English"), _middle("Hindi"),
    # Board: 80 marks, per subject, as the 2025-26 pattern prints them.
    _board_maths(), _board_science(), _board_social_science(),
    _board_language("English"), _board_language("Hindi"),
])

GRADE_BANDS = {"primary": (1, 5), "middle": (6, 8), "board": (9, 10)}


def band_for(grade: int) -> str:
    for band, (low, high) in GRADE_BANDS.items():
        if low <= int(grade) <= high:
            return band
    raise KeyError(f"no CBSE band covers grade {grade}")


def for_paper(subject: str, grade: int) -> Blueprint:
    """The paper this subject sets in this grade. Raises rather than guessing --
    a subject with no registered format is a gap to fill, not to default."""
    band = band_for(grade)
    key = (subject, band)
    if key not in _BLUEPRINTS:
        raise KeyError(
            f"no CBSE format registered for {subject} in the {band} band "
            f"(grade {grade}); add one to _BLUEPRINTS")
    return _BLUEPRINTS[key]


def plan_chapter(blueprint: Blueprint, topics: list[str], mark_budget: int
                 ) -> list[tuple[Section, str]]:
    """Which question belongs in which section, for one chapter.

    A chapter does not carry a whole paper, so the paper's shape is spread across
    the book in proportion to the chapters' topics: every chapter gets its share of
    the objective run, and the longer sections are spread thinly enough that all of
    them are represented. Returns (section, topic) pairs in paper order, and the
    topic is empty where the paper's shape does not name a topic -- a Grammar
    section, a map.

    Deterministic: the same chapter and budget always plan the same paper, so a
    regenerated bank diffs cleanly against the last one.
    """
    if not topics:
        raise ValueError("a chapter with no topics cannot be planned for")

    # Weight the sections by marks, so the case-based and long-answer items -- the
    # ones that carry the board's competency focus -- are not crowded out by the
    # one-mark run.
    weights = [(s, s.marks) for s in blueprint.sections]
    total_weight = sum(w for _, w in weights) or 1

    plan: list[tuple[Section, str]] = []
    for section, weight in weights:
        share = weight / total_weight
        want = int(round(blueprint.total_marks * share * (mark_budget / blueprint.theory_marks)))
        want = min(want, section.count)
        if section.qtype in (MAP,) and len(topics) < 2:
            want = min(want, 1)
        for i in range(want):
            # Rotate topics so a chapter's questions spread across its own headings
            # rather than stacking on the first one.
            topic = topics[i % len(topics)] if section.qtype not in (MAP,) else ""
            plan.append((section, topic))
    return plan


def bloom_mix(blueprint: Blueprint) -> dict[str, int]:
    """How many questions sit at each cognitive level, for the whole paper.

    Derived from the section structure rather than chosen: the competency sections
    (case-based, long answer) have to sit at `apply` and above, because a one-mark
    recall item is not what a case-based section is for. The objective run is
    spread across the lower three levels, which is also where the board's
    difficulty curve actually sits.
    """
    mix = {level: 0 for level in BLOOM_ORDER}
    for section in blueprint.sections:
        if section.qtype in (CASE, MAP):
            levels = ("apply", "analyze", "evaluate")
        elif section.qtype in (LA, SA):
            levels = ("understand", "apply", "analyze")
        elif section.qtype == ASSERTION_REASON:
            levels = ("understand", "analyze")
        elif section.qtype == MCQ and section.carries_source:
            # A question on a passage is not a recall question. The board names the
            # Reading competencies as decoding, analysing, inferring, interpreting
            # and vocabulary -- so the run starts at `understand`, and the inference
            # and analysis items are what carry it.
            levels = ("understand", "apply", "analyze")
        else:
            levels = ("remember", "understand", "apply")
        for i in range(section.count):
            mix[levels[i % len(levels)]] += 1
    return mix


def competency_share(blueprint: Blueprint) -> float:
    """Share of the paper that is competency-based, by the board's own definition:
    anything at `apply` or above, plus the case-based and map work."""
    total = blueprint.total_questions or 1
    levels = bloom_mix(blueprint)
    high = sum(n for level, n in levels.items() if level in BLOOM_ORDER[2:])
    return high / total
