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
  * **DECLARED**: how long the same paper takes a teacher to set by hand. This
    is NOT measured here. Nothing in this system watches a teacher work, and no
    honest implementation of it is possible without a study.

So the output is `estimatedMinutesSaved`, and the report carries the baseline's
provenance with it. An "estimated" saving built on a declared baseline is a
perfectly good management figure and a completely unacceptable engineering
claim, and the difference is only visible if it is written down.

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

    def as_dict(self) -> dict:
        return {
            "papers": self.papers,
            "papersByPath": dict(self.papers_by_path),
            "medianGenerationSeconds": round(self.median_generation_seconds, 2),
            "totalGenerationSeconds": round(self.total_generation_seconds, 2),
            "baselineMinutesPerPaper": self.baseline_minutes_per_paper,
            "baselineProvenance": self.baseline_provenance,
            "estimatedMinutesSavedPerPaper": round(
                self.estimated_minutes_saved_per_paper, 1),
            "estimatedMinutesSavedTotal": round(
                self.estimated_minutes_saved_total, 1),
            # Stated on every response rather than only in the docs, because
            # this is the number that will be quoted at somebody.
            "measured": ["generationSeconds"],
            "declared": ["baselineMinutesPerPaper"],
            "caveat": (
                "the baseline is DECLARED, not measured -- nothing here watches "
                "a teacher work. Treat the saving as a management estimate, not "
                "an engineering claim."
            ),
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


def record_generation(*, data_root, school_id: str, user_id: str,
                      paper_id: str, elapsed_seconds: float,
                      question_count: int = 0,
                      subject: str = "", grade: str = "") -> str | None:
    """Append a timestamped generation record to the audit log.

    Returns the audit entry id, or None if the audit store is unavailable. A
    missing audit store must NOT fail generation -- the paper is the thing the
    user asked for, and measurement is secondary to it. The caller is expected
    to log the miss; see the routes.
    """
    if elapsed_seconds < 0:
        raise ValueError(
            f"elapsed_seconds={elapsed_seconds} is negative; a duration cannot "
            "be. This is a caller bug, not a slow paper."
        )
    audit: AuditLog = get_audit_log(data_root)
    baseline, _provenance = baseline_minutes()
    return audit.append(
        ACTION,
        actor=user_id,
        details={
            "paperId": paper_id,
            "schoolId": school_id,
            "subject": subject,
            "grade": grade,
            "questionCount": question_count,
            # MEASURED.
            "generationSeconds": round(elapsed_seconds, 3),
            # DECLARED, recorded alongside so a later baseline change does not
            # silently rewrite the history of what was reported.
            "baselineMinutesAtRecordTime": baseline,
        },
    )


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
    class and subject share a key and count once: that is Generate pressed
    again on the same choices (generation is deterministic). "Make another
    like this" prints other questions, so it is another paper.
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
    baseline, e.g. with the one the principal set for the term.

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

    elapsed: list[float] = []
    by_path: Counter[str] = Counter()
    by_pair: Counter[tuple[int, str]] = Counter()
    unattributed = 0
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
            pair = class_subject(details.get("grade"), details.get("subject"))
            if pair is None and attribute is not None and entry.get("assessment_id"):
                found = attribute(entry["assessment_id"])
                pair = class_subject(found[1], found[0]) if found else None
            if pair is None:
                unattributed += 1
            else:
                by_pair[pair] += 1

    baseline, provenance = baseline if baseline is not None else baseline_minutes()
    if not elapsed:
        return SavedTimeReport(
            papers=0, median_generation_seconds=0.0, total_generation_seconds=0.0,
            baseline_minutes_per_paper=baseline, baseline_provenance=provenance,
            estimated_minutes_saved_per_paper=0.0,
            estimated_minutes_saved_total=0.0,
        )

    total = sum(elapsed)
    median = statistics.median(elapsed)
    per_paper = baseline - (median / 60.0)
    return SavedTimeReport(
        papers=len(elapsed),
        median_generation_seconds=median,
        total_generation_seconds=total,
        baseline_minutes_per_paper=baseline,
        baseline_provenance=provenance,
        # Clamped at zero: a machine slower than a teacher by hand is not a
        # saving, and reporting a negative one invites the reader to distrust
        # the figure that IS real.
        estimated_minutes_saved_per_paper=max(0.0, per_paper),
        estimated_minutes_saved_total=max(0.0, per_paper) * len(elapsed),
        papers_by_path=dict(by_path),
        papers_by_class_subject=dict(by_pair),
        unattributed=unattributed,
    )


def exam_coverage(rep: SavedTimeReport,
                  universe: Iterable[tuple[int, str]] | None) -> dict:
    """PRD 12.6 (a): which classes and subjects have had a paper set, and which
    have not.

    `universe` is the (class, subject) pairs the school itself declared for the
    period -- None when it declared none, in which case what is missing cannot
    be listed and the response says so (`classesDeclared: false`) rather than
    showing an empty "not yet" list that reads as "nothing is missing".
    Subjects compare without case, because the curriculum and the papers are
    written by different screens.
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
    return {
        "covered": [{"grade": g, "subject": name, "papers": n}
                    for (g, _), (name, n) in sorted(covered.items())],
        "notYet": [{"grade": g, "subject": s} for g, s in declared
                   if (g, s.casefold()) not in covered],
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
