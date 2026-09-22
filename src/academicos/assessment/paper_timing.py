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
from dataclasses import dataclass

from .audit_log import AuditLog, get_audit_log

# The action the generation routes already append. Kept as one constant so the
# writer and the reader cannot drift apart.
ACTION = "quick_paper_generated"

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
_DEFAULT_PROVENANCE = (
    "declared default, not measured -- a teacher assembling, formatting and "
    "answer-keying a full paper by hand; override with "
    "ACOS_PAPER_MANUAL_BASELINE_MINUTES to use the school's own figure"
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

    def as_dict(self) -> dict:
        return {
            "papers": self.papers,
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


def report(data_root, *, school_id: str | None = None,
           limit: int = 5_000) -> SavedTimeReport:
    """Aggregate the recorded generations into a saved-time figure.

    `school_id=None` aggregates everything, which is right for a single-tenant
    deployment and wrong for a multi-tenant one -- so it is explicit rather than
    defaulted silently.
    """
    audit = get_audit_log(data_root)
    entries = audit.for_action(ACTION)
    if limit and len(entries) > limit:
        entries = entries[:limit]

    elapsed: list[float] = []
    for entry in entries:
        details = entry.get("details") or {}
        if school_id is not None and details.get("schoolId") != school_id:
            continue
        value = details.get("generationSeconds")
        if isinstance(value, (int, float)) and value >= 0:
            elapsed.append(float(value))

    baseline, provenance = baseline_minutes()
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
    )


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
