"""Pillars 5 & 6 — Teacher and Principal intelligence.

Teachers get actions, not marks:
    "76% of the class missed the same marking point on Photosynthesis Equation.
     Recommended: activity B, ~20 minutes."

Principals get system health, not questions: subject/grade rollups, teacher
comparisons, and where to intervene.

Class aggregation reuses `algorithms.teacher_assistance.TeacherAssistance`,
which already ranks concepts by a weakness/spread/at-risk-mass blend.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..algorithms.teacher_assistance import (
    ClassParams,
    ConceptSnapshot,
    LearnerSnapshot,
    TeacherAssistance,
)
from .chapters import chapter_name
from .evaluate import Evaluation
from .knowledge import KnowledgeStore
from .schemas import QuestionSchema

# Minutes of remediation to budget per weak concept, by how weak it is.
_INTERVENTION_MINUTES = ((0.30, 45), (0.45, 30), (0.60, 20), (1.01, 15))


def _minutes_for(accuracy: float) -> int:
    for ceiling, minutes in _INTERVENTION_MINUTES:
        if accuracy < ceiling:
            return minutes
    return 15


@dataclass
class SharedMistake:
    marking_point: str
    concept_id: str
    concept_name: str
    students_affected: int
    percentage: float
    example_gap: str


@dataclass
class ConceptInsight:
    concept_id: str
    concept_name: str
    class_accuracy: float
    students_rated: int
    at_risk_students: list[str]
    focus_priority: float
    recommendation: str
    estimated_minutes: int


@dataclass
class ClassInsights:
    assessment_id: str
    class_id: str
    students: int
    average_percentage: float
    hardest_concept: str | None
    concepts: list[ConceptInsight] = field(default_factory=list)
    shared_mistakes: list[SharedMistake] = field(default_factory=list)
    headline: str = ""


@dataclass
class SubjectRollup:
    subject: str
    grade: int
    average_mastery: float
    students: int
    weakest_concept: str | None
    curriculum_coverage: float


@dataclass
class SchoolInsights:
    school_id: str
    students: int
    assessments: int
    average_mastery: float
    subjects: list[SubjectRollup] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)


def class_insights(assessment_id: str, class_id: str,
                   per_student: dict[str, list[tuple[QuestionSchema, Evaluation]]],
                   *, weak_threshold: float = 0.6) -> ClassInsights:
    """Aggregate a marked class set into teaching actions."""
    snapshots: list[LearnerSnapshot] = []
    totals: list[float] = []
    # marking-point description -> {students}, and which concept it belongs to
    missed: dict[str, set[str]] = defaultdict(set)
    missed_concept: dict[str, str] = {}

    for student_id, graded in per_student.items():
        concept_hits: dict[str, list[float]] = defaultdict(list)
        awarded = maximum = 0
        for question, ev in graded:
            awarded += ev.awarded_marks
            maximum += ev.max_marks
            ratio = (ev.awarded_marks / ev.max_marks) if ev.max_marks else 0.0
            for cid in (question.chapter_ids or ["unmapped"]):
                concept_hits[cid].append(ratio)
                for gap in ev.gaps:
                    missed[gap].add(student_id)
                    missed_concept.setdefault(gap, cid)
        if maximum:
            totals.append(100.0 * awarded / maximum)
        snapshots.append(LearnerSnapshot(
            learner_id=student_id,
            concepts={cid: ConceptSnapshot(accuracy=sum(v) / len(v), attempts=len(v))
                      for cid, v in concept_hits.items()},
        ))

    engine = TeacherAssistance(ClassParams(weak_threshold=weak_threshold, min_attempts=1))
    report = engine.report(snapshots)

    concepts: list[ConceptInsight] = []
    for rank in report.concepts:
        name = chapter_name(rank.concept_id) or rank.concept_id.replace("-", " ").title()
        minutes = _minutes_for(rank.class_accuracy)
        if rank.class_accuracy < weak_threshold:
            rec = (f"Re-teach {name} to the whole class — "
                   f"{len(rank.at_risk)} of {rank.n_students_rated} students are below target.")
        elif rank.at_risk:
            rec = f"Small-group support on {name} for {len(rank.at_risk)} students."
        else:
            rec = f"{name} is secure — no class-wide action needed."
        concepts.append(ConceptInsight(
            concept_id=rank.concept_id, concept_name=name,
            class_accuracy=round(rank.class_accuracy, 4),
            students_rated=rank.n_students_rated,
            at_risk_students=list(rank.at_risk),
            focus_priority=round(rank.focus_priority, 4),
            recommendation=rec, estimated_minutes=minutes,
        ))

    n_students = len(per_student) or 1
    shared = [
        SharedMistake(
            marking_point=desc,
            concept_id=missed_concept.get(desc, "unmapped"),
            concept_name=chapter_name(missed_concept.get(desc, "")) or "General",
            students_affected=len(students),
            percentage=round(100.0 * len(students) / n_students, 1),
            example_gap=desc,
        )
        for desc, students in missed.items() if len(students) > 1
    ]
    shared.sort(key=lambda s: -s.students_affected)

    hardest = concepts[0].concept_name if concepts else None
    avg = round(sum(totals) / len(totals), 2) if totals else 0.0
    headline = (f"Class average {avg}%. Hardest concept: {hardest}." if hardest
                else f"Class average {avg}%.")
    if shared:
        top = shared[0]
        headline += (f" {top.percentage}% of students missed the same point: "
                     f"{top.marking_point}.")

    return ClassInsights(
        assessment_id=assessment_id, class_id=class_id, students=len(per_student),
        average_percentage=avg, hardest_concept=hardest,
        concepts=concepts[:10], shared_mistakes=shared[:10], headline=headline,
    )


@dataclass
class ClassEvidence:
    """One subject and class's graded sheets in one school."""
    subject: str
    grade: int
    students: set[str]
    # The chapter ids the graded questions carry: the only concepts this
    # group's rollup reads, so a class 6 Maths paper is not averaged into
    # class 10 Science through the same student's learner model.
    concepts: set[str]
    total_chapters: int = 0            # the syllabus's chapter count; 0 = unknown
    names: dict[str, str] = field(default_factory=dict)


def school_insights(school_id: str, store: KnowledgeStore, groups: list[ClassEvidence],
                    *, assessments: int = 0) -> SchoolInsights:
    """Subject and class rollups for the principal view, built only from the
    groups the caller passes (the school's own graded sheets)."""
    cache: dict[str, dict[str, float]] = {}

    def mastery_of(sid: str) -> dict[str, float]:
        if sid not in cache:
            cache[sid] = {v.concept_id: v.mastery for v in store.mastery(sid)}
        return cache[sid]

    rollups: list[SubjectRollup] = []
    interventions: list[str] = []
    every_mean: list[float] = []
    students: set[str] = set()
    for g in sorted(groups, key=lambda g: (g.grade, g.subject)):
        students |= g.students
        means: list[float] = []
        totals: Counter = Counter()
        counts: Counter = Counter()
        for sid in sorted(g.students):
            seen = {c: m for c, m in mastery_of(sid).items() if c in g.concepts}
            if not seen:
                continue
            means.append(sum(seen.values()) / len(seen))
            for c, m in seen.items():
                totals[c] += m
                counts[c] += 1
        every_mean += means
        avg = round(sum(means) / len(means), 4) if means else 0.0
        weakest = None
        if counts:
            weakest_id = min(sorted(counts), key=lambda c: totals[c] / counts[c])
            weakest = g.names.get(weakest_id) or chapter_name(weakest_id) or weakest_id
        coverage = (round(min(len(counts) / g.total_chapters, 1.0), 3)
                    if g.total_chapters else 0.0)
        label = f"{g.subject} class {g.grade}"
        if means and avg < 0.6:
            interventions.append(f"{label} mastery is {avg:.0%} — below the 60% action line.")
        if weakest:
            interventions.append(f"Weakest area in {label}: {weakest}. Prioritise for re-teaching.")
        if g.total_chapters and coverage < 0.5:
            interventions.append(
                f"Only {coverage:.0%} of {label}'s chapters have assessment evidence — "
                "widen chapter coverage in the next paper.")
        rollups.append(SubjectRollup(subject=g.subject, grade=g.grade, average_mastery=avg,
                                     students=len(means), weakest_concept=weakest,
                                     curriculum_coverage=coverage))

    return SchoolInsights(
        school_id=school_id, students=len(students), assessments=assessments,
        average_mastery=round(sum(every_mean) / len(every_mean), 4) if every_mean else 0.0,
        subjects=rollups, interventions=interventions,
    )
