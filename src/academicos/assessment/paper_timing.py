"""Time saved per paper -- the renewal criterion, made measurable.

Decision 11 of the PRD: *"Renewal is measured as time saved per paper."* Until
this module there was nothing in the repository that measured it, which made the
criterion unfalsifiable -- and a renewal criterion nobody can compute is not a
criterion.

What is measured, and what is only declared
-------------------------------------------
This distinction is the whole design, and it is easy to get wrong in a way that
flatters the number:

  * **MEASURED**: how long the machine took to produce the paper. Recorded from
    a monotonic clock around the generation call, so it cannot be inflated and
    cannot be fabricated by the caller.
  * **MEASURED, by the web app**: how long the teacher spent making the paper
    with the product -- from opening the builder (or the quick dialog) to
    asking for the paper (`teacherSeconds` on the generation row). The client
    sends it with the generate request; a caller that does not measure it
    sends nothing.
  * **DECLARED**: how long the same paper takes a teacher to set by hand. This
    is NOT measured here. Nothing in this system watches a teacher set a paper
    by hand, and no honest implementation of it is possible without a study.

A paper's saving is the declared baseline less the teacher's measured time,
never below zero. A paper with no measured time counts the whole baseline. The
recorded server compute (0.02-0.15 s) is not the teacher's time and no longer
enters the saving: subtracting it made "time saved" the baseline times the
paper count (D49).

So the output is `estimatedMinutesSaved` -- every saving rests on the declared
baseline -- and the report carries the baseline's provenance and how many
papers had their teacher's time measured. An "estimated" saving built on a
declared baseline is a perfectly good management figure and a completely
unacceptable engineering claim, and the difference is only visible if it is
written down.

What counts as a paper
----------------------
A paper that still exists, once (`kept_papers`) -- not a press of Generate.
Every press used to count as a paper set and a full baseline saved (D42):
Generate pressed again on the same choices made the same questions under a
second paper id and counted twice, a regeneration on one assessment counted
the paper it replaced, and a deleted paper still counted. The median
generation time is over the same papers, one generation each.

Where it is stored, and why not a new store
-------------------------------------------
In the existing audit log, under the action paper generation already records.
That store is the system of record for "what happened", a generated paper is
already an audited action, and timing is part of what happened -- so this adds
no 20th SQLite store for a number that is already sitting next to the event.

The cost is honest: the audit log is compliance data with CERT-In retention, and
aggregating management figures out of it is a slight misuse. If this ever needs
to outlive that retention, or to be joined against things the audit log does not
carry, it should move to its own store and this docstring should be updated
rather than quietly left behind.
"""
from __future__ import annotations

import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Hashable, Iterable, Mapping, Optional

from .audit_log import AuditLog, get_audit_log

# Schools keep IST. Audit timestamps are UTC, and a paper set at 01:00 IST is
# on the school's next day, so a term window compares the school's date.
_IST = timezone(timedelta(hours=5, minutes=30))


def school_today() -> str:
    """Today on the school's calendar, as an ISO date."""
    return datetime.now(_IST).date().isoformat()


def school_date(timestamp: str) -> Optional[str]:
    """The school's calendar date (IST) of an audit timestamp, or None if the
    timestamp cannot be read -- such an entry is outside every window rather
    than guessed into one."""
    try:
        moment = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(_IST).date().isoformat()

# Where the web app has the controls the time-saved and coverage guidance
# names: the School Admin console's Calendar & Setup tab declares terms, and
# the console itself sets up a new school's year and classes. An instruction
# that names a screen without the control is worse than none (D43: the card
# told a school with its classes set up to set them up, "in calendar setup").
TERMS_PAGE = "/admin?tab=calendar-setup"
CLASSES_PAGE = "/admin"


def setup_step(*, classes_declared: bool, term_declared: bool) -> Optional[dict]:
    """The one thing a school must do before a figure can be per term or
    name its missing classes, with the web page that does it; None when
    nothing is missing. Classes come first: terms belong to a year."""
    if not classes_declared:
        return {"need": "classes", "label": "Set up this year's classes", "link": CLASSES_PAGE,
                "message": ("This school has no classes set up for the year covering today, so the "
                            "classes still without a paper cannot be listed. Set up this year's "
                            "classes on the School Admin console.")}
    if not term_declared:
        return {"need": "terms", "label": "Declare terms", "link": TERMS_PAGE,
                "message": ("Declare this year's terms on the School Admin console, under "
                            "Calendar & Setup, to see one term at a time.")}
    return None


# The action quick generation appends, and `record_generation` writes.
ACTION = "quick_paper_generated"

# Every action a generation route appends, one per way the product makes a
# paper, with the name the report gives that path. The report reads all of
# them: the web app makes its papers through the guided builder and the
# hand-picked path and never calls quick-generate, so reading ACTION alone
# reported zero papers for every school using the web app (audit 2026-09-26).
# tests/test_paper_timing.py scans src/ so a new path cannot go uncounted.
GENERATION_ACTIONS = {
    ACTION: "quick",
    "template_paper_generated": "builder",
    "id_curated_paper_generated": "hand-picked",
}

# The declared manual baseline, in minutes.
#
# 45 is a starting figure, not a finding: it is what a teacher typically spends
# assembling, formatting and answer-keying a full paper by hand, and it is what
# this module will happily report against until somebody measures the real one.
#
# It is deliberately overridable per deployment (`ACOS_PAPER_MANUAL_BASELINE_MINUTES`)
# so a school can use its OWN figure. A baseline the school supplied is worth
# far more in a renewal conversation than one the vendor asserted -- which is
# why the report echoes back where the number came from.
_DEFAULT_BASELINE_MINUTES = 45.0
_ENV_KEY = "ACOS_PAPER_MANUAL_BASELINE_MINUTES"
# Shown to the principal on the term report, so it names no server setting
# (it printed the environment variable's name; E2E run, 2026-10-01).
_DEFAULT_PROVENANCE = (
    "declared default, not measured -- a teacher assembling, formatting and "
    "answer-keying a full paper by hand; set the school's own figure for the "
    "term to use it instead"
)


@dataclass(frozen=True)
class SavedTimeReport:
    """What the numbers are, and how far each one can be trusted."""

    papers: int
    median_generation_seconds: float
    total_generation_seconds: float
    baseline_minutes_per_paper: float
    baseline_provenance: str
    estimated_minutes_saved_per_paper: float
    estimated_minutes_saved_total: float
    # How many of `papers` each generation path made ("quick", "builder",
    # "hand-picked"); a path with none is absent.
    papers_by_path: dict[str, int] = field(default_factory=dict)
    # How many of `papers` were set for each (class, subject). A paper whose
    # class and subject cannot be established is in `unattributed`, so the
    # two always add up to `papers`.
    papers_by_class_subject: dict[tuple[int, str], int] = field(default_factory=dict)
    unattributed: int = 0
    # The minutes saved on those papers, split the same way, so a report
    # filtered to some classes adds up their own papers' savings -- each paper
    # is measured against the baseline it was made under, so a per-paper
    # average times a count is not their sum.
    saved_by_class_subject: dict[tuple[int, str], float] = field(default_factory=dict)
    saved_unattributed: float = 0.0
    # How many of `papers` carry the baseline in force when they were made
    # (the rest predate rows carrying one and use `baseline_minutes_per_paper`).
    papers_at_recorded_baseline: int = 0
    # How many of `papers` carry the teacher's measured time; the rest count
    # the whole baseline. Split by (class, subject) like the savings.
    papers_with_teacher_time: int = 0
    teacher_time_by_class_subject: dict[tuple[int, str], int] = field(default_factory=dict)
    teacher_time_unattributed: int = 0
    # The median of those measured times, in minutes; None when none was.
    median_teacher_minutes: Optional[float] = None

    @property
    def caveat(self) -> str:
        measured = (f"{self.papers_with_teacher_time} of {self.papers} paper(s) count the baseline "
                    f"less the time their teacher spent making them, as the web app measured it; "
                    f"the rest count the whole baseline"
                    if self.papers else "a paper counts the baseline less the time its teacher "
                    "spent making it, where the web app measured that, else the whole baseline")
        return ("the baseline is DECLARED, not measured -- nothing here watches a teacher set "
                f"a paper by hand. {measured[0].upper()}{measured[1:]}. Treat the saving as a "
                "management estimate, not an engineering claim.")

    def as_dict(self) -> dict:
        return {
            "papers": self.papers,
            "papersByPath": dict(self.papers_by_path),
            "medianGenerationSeconds": round(self.median_generation_seconds, 2),
            "totalGenerationSeconds": round(self.total_generation_seconds, 2),
            # The teacher's time with the product, where the web app measured
            # it (D49): what each paper's saving is the baseline less.
            "papersWithTeacherTime": self.papers_with_teacher_time,
            "medianTeacherMinutes": (round(self.median_teacher_minutes, 1)
                                     if self.median_teacher_minutes is not None else None),
            # The baseline a paper made now is measured against. Each paper
            # already made keeps the one in force when it was made.
            "baselineMinutesPerPaper": self.baseline_minutes_per_paper,
            "baselineProvenance": self.baseline_provenance,
            "papersAtRecordedBaseline": self.papers_at_recorded_baseline,
            "estimatedMinutesSavedPerPaper": round(
                self.estimated_minutes_saved_per_paper, 1),
            "estimatedMinutesSavedTotal": round(
                self.estimated_minutes_saved_total, 1),
            # Every saving rests on the declared baseline.
            "estimated": True,
            # Stated on every response rather than only in the docs, because
            # this is the number that will be quoted at somebody.
            "measured": ["generationSeconds", "teacherSeconds"],
            "declared": ["baselineMinutesPerPaper"],
            "caveat": self.caveat,
        }


def baseline_minutes() -> tuple[float, str]:
    """The declared manual baseline and where it came from."""
    raw = os.environ.get(_ENV_KEY, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return _DEFAULT_BASELINE_MINUTES, (
                f"{_ENV_KEY}={raw!r} is not a number; fell back to the "
                f"declared default. " + _DEFAULT_PROVENANCE
            )
        if value <= 0:
            return _DEFAULT_BASELINE_MINUTES, (
                f"{_ENV_KEY}={value} is not positive; fell back to the declared "
                f"default. " + _DEFAULT_PROVENANCE
            )
        return value, f"configured by {_ENV_KEY}={value}"
    return _DEFAULT_BASELINE_MINUTES, _DEFAULT_PROVENANCE


def baseline_in_force(data_root, school_id: str) -> tuple[float, str]:
    """The baseline a paper made now is measured against, and where it came
    from: the principal's own for the term covering the school's today, else
    the declared one. `record_generation` stores it on the paper's row."""
    from ..curriculum.store import get_curriculum_store
    term = get_curriculum_store(data_root).term_for_date(school_id, school_today())
    if term is not None and term.manual_baseline_minutes is not None:
        return float(term.manual_baseline_minutes), f"set by the principal for {term.name}"
    return baseline_minutes()


def record_generation(action: str = ACTION, *, data_root, school_id: str, user_id: str,
                      paper_id: str, elapsed_seconds: float,
                      question_count: int = 0,
                      subject: str = "", grade: str | int = "",
                      assessment_id: str | None = None,
                      teacher_seconds: float | None = None,
                      details: Mapping | None = None) -> str:
    """Append a timestamped generation record to the audit log, under
    `action` (one of GENERATION_ACTIONS). Every generation route writes its
    row here, so each row carries the same measured and declared fields.

    The baseline in force for the school when the paper was made
    (`baseline_in_force`) is stored on the row, and the report reads it from
    there: a principal raising the term's baseline later moves only the papers
    made after the change, never the saving of papers already reported (D51:
    the routes appended their rows directly and none carried it, so moving
    Term 1's baseline from 60 to 120 rewrote the same 13 papers from 780 to
    1,560 minutes).

    `teacher_seconds` is the teacher's time making the paper, as the web app
    measured it (from opening the builder or the quick dialog to asking for
    the paper); None when the caller did not measure it. `details` adds
    route-specific fields (template id, set count, ...). Returns the audit
    entry id.
    """
    if action not in GENERATION_ACTIONS:
        raise ValueError(f"{action!r} is not a generation action the report reads")
    for name, value in (("elapsed_seconds", elapsed_seconds), ("teacher_seconds", teacher_seconds)):
        if value is not None and value < 0:
            raise ValueError(
                f"{name}={value} is negative; a duration cannot "
                "be. This is a caller bug, not a slow paper."
            )
    audit: AuditLog = get_audit_log(data_root)
    baseline, provenance = baseline_in_force(data_root, school_id)
    return audit.append(
        action,
        assessment_id=assessment_id,
        actor=user_id,
        details={
            **dict(details or {}),
            "paperId": paper_id,
            "userId": user_id,
            "schoolId": school_id,
            # The class and subject the paper was set for: PRD 12.6's exam
            # coverage counts papers per class and subject per term.
            "subject": subject,
            "grade": grade,
            "questionCount": question_count,
            # MEASURED.
            "generationSeconds": round(elapsed_seconds, 3),
            # MEASURED by the web app: the teacher's time (None: not measured).
            "teacherSeconds": round(teacher_seconds, 1) if teacher_seconds is not None else None,
            # DECLARED, recorded alongside so a later baseline change does not
            # silently rewrite the history of what was reported.
            "baselineMinutesAtRecordTime": baseline,
            "baselineProvenanceAtRecordTime": provenance,
        },
    )


def recorded_baseline(details: Mapping) -> Optional[float]:
    """The baseline stored on a generation row, or None for a row written
    before rows carried one (or carrying something that is not a positive
    number)."""
    value = details.get("baselineMinutesAtRecordTime")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def teacher_minutes(details: Mapping) -> Optional[float]:
    """The teacher's measured time on a generation row, in minutes, or None
    when the row carries none."""
    value = details.get("teacherSeconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value) / 60.0


def class_subject(grade, subject) -> Optional[tuple[int, str]]:
    """(class number, subject) from what a generation recorded, or None when
    either is missing or the class is not a number."""
    try:
        number = int(grade)
    except (TypeError, ValueError):
        return None
    name = str(subject or "").strip()
    return (number, name) if name else None


def kept_papers(assessments: Iterable) -> dict[str, Hashable]:
    """Paper id -> what makes it one paper, for each paper that still exists:
    the one an assessment holds now (`generated_paper_id`). The paper of a
    deleted assessment, and one a later generation replaced on the same
    assessment, are not here, so they no longer count.

    Papers the same teacher generated with the same questions for the same
    class and subject share a key and count once: the same paper made twice.
    Since 2026-10-01 pressing Generate again on the same choices prints
    other questions where the bank has them (QA P-03: a teacher's recent
    papers rank last), so like "Make another like this" it is another paper
    and counts.
    """
    out: dict[str, Hashable] = {}
    for a in assessments:
        paper_id = getattr(a, "generated_paper_id", None)
        if not paper_id:
            continue
        questions = frozenset(getattr(a, "selected_question_ids", None) or ())
        out[paper_id] = ((a.teacher_id, str(a.subject).strip().casefold(), str(a.grade), questions)
                         if questions else paper_id)
    return out


def report(data_root, *, school_id: str | None = None,
           limit: int = 5_000,
           start_date: str | None = None, end_date: str | None = None,
           attribute: Callable[[str], Optional[tuple[str, int]]] | None = None,
           baseline: tuple[float, str] | None = None,
           kept: Mapping[str, Hashable] | None = None) -> SavedTimeReport:
    """Aggregate the recorded generations into a saved-time figure.

    `school_id=None` aggregates everything, which is right for a single-tenant
    deployment and wrong for a multi-tenant one -- so it is explicit rather than
    defaulted silently.

    `start_date`/`end_date` (ISO, inclusive) keep only papers set on those
    days of the school's calendar -- a term. `attribute(assessment_id)`
    returns (subject, class) for an entry written before generation recorded
    them; an entry neither can place is counted as unattributed. `baseline`
    is (minutes, where it came from) and replaces the deployment's declared
    baseline, e.g. with the one the principal set for the term. A paper whose
    row carries the baseline in force when it was made is measured against
    that one (`record_generation`); `baseline` is for older rows, and is what
    the report says a paper made now is measured against.

    `kept` (`kept_papers`) is the papers that still exist: only their
    generations count, and papers sharing a key count once, at their first
    generation in the window. None counts every recorded generation, for
    a caller with no papers to check against; the routes always pass it.
    """
    audit = get_audit_log(data_root)
    entries = [(path, entry) for action, path in GENERATION_ACTIONS.items()
               for entry in audit.for_action(action)]
    # Oldest first, so a paper made twice counts its first generation.
    entries.sort(key=lambda pe: str(pe[1].get("timestamp") or ""))
    if limit and len(entries) > limit:
        # The most recent `limit` across every path, not the first path's.
        entries = entries[-limit:]

    baseline, provenance = baseline if baseline is not None else baseline_minutes()
    elapsed: list[float] = []
    by_path: Counter[str] = Counter()
    by_pair: Counter[tuple[int, str]] = Counter()
    saved_by_pair: dict[tuple[int, str], float] = {}
    unattributed = 0
    saved_unattributed = 0.0
    at_recorded = 0
    teacher: list[float] = []
    timed_by_pair: Counter[tuple[int, str]] = Counter()
    timed_unattributed = 0
    counted: set[Hashable] = set()
    for path, entry in entries:
        details = entry.get("details") or {}
        if school_id is not None and details.get("schoolId") != school_id:
            continue
        if start_date is not None or end_date is not None:
            day = school_date(entry.get("timestamp"))
            if day is None or (start_date and day < start_date) or (end_date and day > end_date):
                continue
        value = details.get("generationSeconds")
        if isinstance(value, (int, float)) and value >= 0:
            if kept is not None:
                key = kept.get(details.get("paperId"))
                if key is None or key in counted:
                    continue        # deleted, replaced, or a paper already counted
                counted.add(key)
            elapsed.append(float(value))
            by_path[path] += 1
            # The baseline the paper was made under, where its row carries
            # one (D51); a row from before rows did takes today's.
            recorded = recorded_baseline(details)
            at_recorded += recorded is not None
            # The baseline less the teacher's measured time (D49), or the
            # whole baseline where it was not measured. Clamped at zero: a
            # paper that took longer with the product than by hand is not a
            # saving, and reporting a negative one invites the reader to
            # distrust the figure that IS real.
            spent = teacher_minutes(details)
            if spent is not None:
                teacher.append(spent)
            saved = max(0.0, (recorded if recorded is not None else baseline) - (spent or 0.0))
            pair = class_subject(details.get("grade"), details.get("subject"))
            if pair is None and attribute is not None and entry.get("assessment_id"):
                found = attribute(entry["assessment_id"])
                pair = class_subject(found[1], found[0]) if found else None
            if pair is None:
                unattributed += 1
                saved_unattributed += saved
                timed_unattributed += spent is not None
            else:
                by_pair[pair] += 1
                saved_by_pair[pair] = saved_by_pair.get(pair, 0.0) + saved
                timed_by_pair[pair] += spent is not None

    if not elapsed:
        return SavedTimeReport(
            papers=0, median_generation_seconds=0.0, total_generation_seconds=0.0,
            baseline_minutes_per_paper=baseline, baseline_provenance=provenance,
            estimated_minutes_saved_per_paper=0.0,
            estimated_minutes_saved_total=0.0,
        )

    saved_total = saved_unattributed + sum(saved_by_pair.values())
    return SavedTimeReport(
        papers=len(elapsed),
        median_generation_seconds=statistics.median(elapsed),
        total_generation_seconds=sum(elapsed),
        baseline_minutes_per_paper=baseline,
        baseline_provenance=provenance,
        estimated_minutes_saved_per_paper=saved_total / len(elapsed),
        estimated_minutes_saved_total=saved_total,
        papers_by_path=dict(by_path),
        papers_by_class_subject=dict(by_pair),
        unattributed=unattributed,
        saved_by_class_subject=saved_by_pair,
        saved_unattributed=saved_unattributed,
        papers_at_recorded_baseline=at_recorded,
        papers_with_teacher_time=len(teacher),
        teacher_time_by_class_subject={p: n for p, n in timed_by_pair.items() if n},
        teacher_time_unattributed=timed_unattributed,
        median_teacher_minutes=statistics.median(teacher) if teacher else None,
    )


def exam_coverage(rep: SavedTimeReport,
                  universe: Iterable[tuple[int, str]] | None,
                  questions: Callable[[int, str], int] | None = None) -> dict:
    """PRD 12.6 (a): which classes and subjects have had a paper set, and which
    have not.

    `universe` is the (class, subject) pairs the school itself declared for the
    period -- None when it declared none, in which case what is missing cannot
    be listed and the response says so (`classesDeclared: false`) rather than
    showing an empty "not yet" list that reads as "nothing is missing".
    Subjects compare without case, because the curriculum and the papers are
    written by different screens.

    `questions(class, subject)` is the bank's question count for a pair (the
    catalog's). A pair with no paper and no question is `noQuestions`, not
    `notYet`: "no paper yet" said of Hindi 6-10, Social Science 6-9 and
    English 7-8, which the bank cannot serve at all, read as the teachers'
    gap when it is the bank's (D117).
    """
    declared = sorted({(g, s.strip()) for g, s in universe or () if s and s.strip()},
                      key=lambda p: (p[0], p[1].casefold()))
    school_spelling = {(g, s.casefold()): s for g, s in declared}
    # Papers per (class, subject) regardless of case; the label is the
    # school's own spelling where it declared the subject, else the spelling
    # most papers used.
    spellings: dict[tuple[int, str], Counter[str]] = {}
    for (grade, subject), n in rep.papers_by_class_subject.items():
        spellings.setdefault((grade, subject.casefold()), Counter())[subject] += n
    covered = {key: (school_spelling.get(key) or names.most_common(1)[0][0],
                     sum(names.values()))
               for key, names in spellings.items()}
    missing = [(g, s) for g, s in declared if (g, s.casefold()) not in covered]
    servable = {(g, s): questions is None or questions(g, s) > 0 for g, s in missing}
    return {
        "covered": [{"grade": g, "subject": name, "papers": n}
                    for (g, _), (name, n) in sorted(covered.items())],
        "notYet": [{"grade": g, "subject": s} for g, s in missing if servable[(g, s)]],
        "noQuestions": [{"grade": g, "subject": s} for g, s in missing if not servable[(g, s)]],
        "unattributed": rep.unattributed,
        "classesDeclared": bool(declared),
    }


class generation_timer:
    """Context manager measuring wall-clock generation time.

    `perf_counter`, not `time.time`: a clock adjustment during generation must
    not be able to change a reported duration.
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_seconds = 0.0

    def __enter__(self) -> "generation_timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start
