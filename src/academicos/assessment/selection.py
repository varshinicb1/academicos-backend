"""Question Optimizer: deterministic rule-based selection.

Filters candidates against each section's constraints (marks, allowed Bloom
levels/difficulties, chapter weighting), greedily fills each section ranked
by quality score + chapter-weight match, and flags gaps where the candidate
pool can't fill a section. No LLM in this path — see chapters.py / pool.py
for why (deterministic-core design principle carried over from the rest of
this codebase).
"""
from __future__ import annotations

from collections import Counter

from .schemas import Blueprint, QuestionOptimizationResult, QuestionSchema, SectionBlueprint
from .templates import default_sections


def _score(q: QuestionSchema, chapter_weights: dict[str, float], used_chapters: Counter) -> float:
    score = q.quality_score
    if chapter_weights:
        for cid in q.chapter_ids:
            score += chapter_weights.get(cid, 0.0) * 0.5
    # mild penalty for repeating a chapter already picked, to spread coverage
    for cid in q.chapter_ids:
        score -= 0.05 * used_chapters.get(cid, 0)
    return score


def _fits_section(q: QuestionSchema, section: SectionBlueprint) -> bool:
    if q.marks != section.marks_per_question:
        return False
    if section.allowed_difficulties and q.difficulty not in section.allowed_difficulties:
        return False
    if section.allowed_bloom_levels and q.bloom_level not in section.allowed_bloom_levels:
        return False
    return True


def optimize(candidates: list[QuestionSchema], blueprint: Blueprint) -> QuestionOptimizationResult:
    sections = blueprint.sections or default_sections(blueprint.total_marks)
    chapter_weights = blueprint.chapter_weights.weights

    remaining = list(candidates)
    selected: list[QuestionSchema] = []
    used_chapters: Counter = Counter()
    warnings: list[str] = []
    gaps: list[str] = []

    for section in sections:
        pool = [q for q in remaining if _fits_section(q, section)]
        if len(pool) < section.question_count:
            gaps.append(
                f"Section {section.label} ({section.name}): needs {section.question_count} "
                f"questions worth {section.marks_per_question} marks each, only {len(pool)} available."
            )
        pool.sort(key=lambda q: _score(q, chapter_weights, used_chapters), reverse=True)
        take = pool[: section.question_count]
        for q in take:
            selected.append(q)
            remaining.remove(q)
            for cid in q.chapter_ids:
                used_chapters[cid] += 1

    selected_ids = {q.id for q in selected}
    rejected = [q for q in candidates if q.id not in selected_ids]

    chapter_coverage = Counter()
    bloom_hist: Counter = Counter()
    difficulty_hist: Counter = Counter()
    for q in selected:
        for cid in q.chapter_ids:
            chapter_coverage[cid] += 1
        bloom_hist[q.bloom_level] += 1
        difficulty_hist[q.difficulty] += 1

    if selected:
        distinct_chapters = len(chapter_coverage)
        if distinct_chapters == 1 and len(selected) > 3:
            warnings.append("Selected paper is concentrated in a single chapter — weak coverage.")
    total_marks_selected = sum(q.marks for q in selected)
    if total_marks_selected != blueprint.total_marks:
        warnings.append(
            f"Selected paper totals {total_marks_selected} marks, blueprint target is {blueprint.total_marks}."
        )

    return QuestionOptimizationResult(
        selected_questions=selected,
        rejected_questions=rejected,
        optimization_metrics={
            "totalMarksSelected": total_marks_selected,
            "questionCount": len(selected),
            "chapterCoverage": dict(chapter_coverage),
            "bloomDistribution": dict(bloom_hist),
            "difficultyDistribution": dict(difficulty_hist),
        },
        warnings=warnings,
        gaps=gaps,
    )
