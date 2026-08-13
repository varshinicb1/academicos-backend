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


def school_insights(school_id: str, store: KnowledgeStore, student_ids: list[str],
                    *, subject: str = "Science", grade: int = 10,
                    total_concepts: int = 13, assessments: int = 0) -> SchoolInsights:
    """Grade/subject rollup for the principal view."""
    masteries: list[float] = []
    concept_totals: Counter = Counter()
    concept_counts: Counter = Counter()

    for sid in student_ids:
        views = store.mastery(sid)
        if not views:
            continue
        masteries.append(sum(v.mastery for v in views) / len(views))
        for v in views:
            concept_totals[v.concept_id] += v.mastery
            concept_counts[v.concept_id] += 1

    avg = round(sum(masteries) / len(masteries), 4) if masteries else 0.0
    weakest = None
    if concept_counts:
        weakest_id = min(concept_counts, key=lambda c: concept_totals[c] / concept_counts[c])
        weakest = chapter_name(weakest_id) or weakest_id
    coverage = round(len(concept_counts) / total_concepts, 3) if total_concepts else 0.0

    interventions: list[str] = []
    if avg and avg < 0.6:
        interventions.append(
            f"{subject} Grade {grade} mastery is {avg:.0%} — below the 60% action line.")
    if weakest:
        interventions.append(f"Weakest area school-wide: {weakest}. Prioritise for re-teaching.")
    if coverage < 0.5:
        interventions.append(
            f"Only {coverage:.0%} of the curriculum has assessment evidence — "
            "widen chapter coverage in the next paper.")

    return SchoolInsights(
        school_id=school_id, students=len(student_ids), assessments=assessments,
        average_mastery=avg,
        subjects=[SubjectRollup(subject=subject, grade=grade, average_mastery=avg,
                                students=len(masteries), weakest_concept=weakest,
                                curriculum_coverage=coverage)],
        interventions=interventions,
    )
