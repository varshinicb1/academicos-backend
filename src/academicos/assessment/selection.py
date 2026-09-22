"""Question Optimizer: deterministic rule-based selection.

Filters candidates against each section's constraints (marks, allowed Bloom
levels/difficulties, chapter weighting), greedily fills each section ranked
by quality score + chapter-weight match, and flags gaps where the candidate
pool can't fill a section. No LLM in this path — see chapters.py / pool.py
for why (deterministic-core design principle carried over from the rest of
this codebase).

A section's marks-per-question is an exact match (`_fits_section`), by
design -- a CBSE section is "five 3-mark questions", not "five questions
around 3 marks". Confirmed live as a real bug: a teacher picked a handful of
chapters that, between them, had zero questions at some section's exact
marks value, and every affected section (in the worst case the whole paper)
came back with 0 questions -- silently, since an empty `selected_questions`
isn't an error, just an unusually short list the frontend generated a PDF
from anyway. `optimize()` now accepts an optional `fallback_candidates`
pool (the full subject+grade pool, unrestricted by chapter) that only kicks
in per-section when the chapter-scoped `candidates` can't fill it, so a
narrow chapter pick degrades to "some questions outside your chosen
chapters, clearly flagged" instead of "blank section." Questions pulled
from the fallback are recorded in `gaps` either way, so this is a rescue,
not a silent substitution.
"""
from __future__ import annotations

from collections import Counter

from .pool import near_duplicate
from .schemas import Blueprint, QuestionOptimizationResult, QuestionSchema, SectionBlueprint
from .templates import default_sections


def is_competency_question(q: QuestionSchema) -> bool:
    """Identify Competency-Based Questions (CBQs) per CBSE / EI criteria."""
    if q.type in ("case_study", "assertion_reason", "competency_based"):
        return True
    if q.bloom_level in ("apply", "analyze", "evaluate", "create"):
        return True
    if q.metadata.get("is_competency") or q.metadata.get("cbq"):
        return True
    stem_lower = q.stem.lower()
    if "assertion" in stem_lower and "reason" in stem_lower:
        return True
    if "read the following" in stem_lower or "based on the passage" in stem_lower or "case study" in stem_lower:
        return True
    return False


def competency_share(questions: list[QuestionSchema]) -> float:
    """The share of `questions` that are competency-based, rounded as reported."""
    if not questions:
        return 0.0
    return round(sum(1 for q in questions if is_competency_question(q)) / len(questions), 3)


# ---- tier signals -------------------------------------------------------------
#
# A tier can only re-rank questions *within* a section, and a section is one
# marks value. So a field is a tier signal only where it says something marks
# do not. Measured on the served bank (2026-09-22): no source authors a
# difficulty. Board records carry mapping._resolve_difficulty(marks, bloom);
# CBE records cbse_cbe._difficulty_for(marks); build_sqp_bank.py writes
# "medium" on every SQP item; bank_merge marks any value it supplies
# `difficultyInferred`/`bloomInferred`. Bloom is "understand" -- the bank's
# neutral default -- on 385/408 Science 10 and 634/645 Maths 10 board records,
# and on every CBE item. After the merge (--check --relink-in-memory) Maths 10
# has one Bloom level and a difficulty fixed by marks: a tier there cannot
# change the paper, and the paper must say so rather than silently doing nothing.

# Sources whose `difficulty` is a constant the builder wrote, not a judgement.
_PLACEHOLDER_DIFFICULTY_SOURCES = frozenset({"cbse_sample_paper"})


def difficulty_signal(q: QuestionSchema) -> str | None:
    """The question's difficulty, or None where the record's value was supplied
    rather than judged (so a tier must not act on it)."""
    if q.metadata.get("difficultyInferred") or q.source in _PLACEHOLDER_DIFFICULTY_SOURCES:
        return None
    return q.difficulty


def bloom_signal(q: QuestionSchema) -> str | None:
    """The question's Bloom level, or None where it was supplied rather than read."""
    if q.metadata.get("bloomInferred"):
        return None
    return q.bloom_level


def _live_signals(questions: list[QuestionSchema],
                  marks_values: set[int]) -> dict[int, tuple[bool, bool]]:
    """Per marks value: do its questions differ in (difficulty, Bloom level)?

    One known level among unknowns says nothing either: a CBE item's
    marks-derived "easy" next to SQP items with no signal would otherwise be
    the only thing a tier ranked on.
    """
    levels: dict[int, tuple[set, set]] = {}
    for q in questions:
        if q.marks in marks_values:
            d, b = levels.setdefault(q.marks, (set(), set()))
            if (ds := difficulty_signal(q)) is not None:
                d.add(ds)
            if (bs := bloom_signal(q)) is not None:
                b.add(bs)
    return {m: (len(d) > 1, len(b) > 1) for m, (d, b) in levels.items()}


def tier_signals(questions: list[QuestionSchema], marks_values: set[int]) -> dict:
    """Can a tier change which of `questions` a section of `marks_values` takes?

    Available when some marks value holds questions that differ in a real
    difficulty or Bloom signal; the level counts are what the questions carry.
    """
    live = _live_signals(questions, marks_values)
    return {
        "available": any(d or b for d, b in live.values()),
        "difficultyLevels": len({d for q in questions if (d := difficulty_signal(q))}),
        "bloomLevels": len({b for q in questions if (b := bloom_signal(q))}),
    }


def _score(
    q: QuestionSchema,
    chapter_weights: dict[str, float],
    used_chapters: Counter,
    tier: str = "standard",
    need_competency_boost: bool = False,
    live: dict[int, tuple[bool, bool]] | None = None,
) -> float:
    score = q.quality_score
    if chapter_weights:
        for cid in q.chapter_ids:
            score += chapter_weights.get(cid, 0.0) * 0.5
    # mild penalty for repeating a chapter already picked, to spread coverage
    for cid in q.chapter_ids:
        score -= 0.05 * used_chapters.get(cid, 0)

    # Student Level Tier adjustments (Differentiated assessment: Foundation vs
    # Standard vs Advanced), on real signals only: a supplied value, or one that
    # does not vary among questions of these marks (`live`), earns nothing.
    t = (tier or "standard").lower()
    difficulty, bloom = difficulty_signal(q), bloom_signal(q)
    if live is not None:
        d_live, b_live = live.get(q.marks, (False, False))
        difficulty = difficulty if d_live else None
        bloom = bloom if b_live else None
    if t == "foundation":
        if difficulty == "easy":
            score += 0.35
        elif difficulty == "hard":
            score -= 0.35
        if bloom in ("remember", "understand"):
            score += 0.20
    elif t == "advanced":
        if difficulty == "hard":
            score += 0.40
        elif difficulty == "easy":
            score -= 0.30
        if bloom in ("analyze", "evaluate", "create"):
            score += 0.30
    elif t == "standard":
        if difficulty == "medium":
            score += 0.15

    # CBSE Competency-Based Question (CBQ) target boost
    if need_competency_boost and is_competency_question(q):
        score += 0.30

    return score


def _fits_section(q: QuestionSchema, section: SectionBlueprint) -> bool:
    if q.marks != section.marks_per_question:
        return False
    if section.allowed_difficulties and q.difficulty not in section.allowed_difficulties:
        return False
    if section.allowed_bloom_levels and q.bloom_level not in section.allowed_bloom_levels:
        return False
    return True


def _repeats_paper(q: QuestionSchema, *on_paper: list[QuestionSchema]) -> bool:
    """Is `q` a near-duplicate of any question already printed, OR partners
    included? One implementation, `pool.near_duplicate`, for every path."""
    return any(near_duplicate(q.stem, p.stem) for group in on_paper for p in group)


def optimize(candidates: list[QuestionSchema], blueprint: Blueprint,
            fallback_candidates: list[QuestionSchema] | None = None) -> QuestionOptimizationResult:
    sections = blueprint.sections or default_sections(blueprint.total_marks)
    chapter_weights = blueprint.chapter_weights.weights
    tier = getattr(blueprint, "tier", "standard") or "standard"
    competency_target = getattr(blueprint, "competency_percentage", 0.50)
    if competency_target is None:
        competency_target = 0.50

    marks_values = {s.marks_per_question for s in sections}
    considered = list(candidates) + list(fallback_candidates or [])
    live = _live_signals(considered, marks_values)
    signals = tier_signals(considered, marks_values)

    remaining = list(candidates)
    selected_ids_seen: set[str] = {q.id for q in candidates}
    fallback_remaining = [q for q in (fallback_candidates or []) if q.id not in selected_ids_seen]
    selected: list[QuestionSchema] = []
    paired_choice_ids: set[str] = set()
    paired: list[QuestionSchema] = []   # the OR partners: printed too
    used_chapters: Counter = Counter()
    warnings: list[str] = []
    gaps: list[str] = []

    for section in sections:
        # Determine if competency boost is needed to hit the target quota (CBSE >= 50%)
        current_cbq = sum(1 for q in selected if is_competency_question(q))
        need_cbq = (current_cbq / len(selected) < competency_target) if selected else True

        def ranked(qs: list[QuestionSchema]) -> list[QuestionSchema]:
            return sorted(
                qs,
                key=lambda q: _score(q, chapter_weights, used_chapters, tier, need_cbq, live),
                reverse=True,
            )

        # Accept in rank order, skipping a question that restates one already
        # on the paper (`pool.near_duplicate`, the guard the template path and
        # swap/pick use). Without it the real pair of Class 10 Mathematics
        # MCQs (Task 903's test) filled two slots of one 1-mark section.
        # Returns how many it skipped as repeats, for the gap wording.
        take: list[QuestionSchema] = []

        def accept(qs: list[QuestionSchema]) -> int:
            skipped = 0
            for q in ranked(qs):
                if len(take) == section.question_count:
                    break
                if _repeats_paper(q, selected + take, paired):
                    skipped += 1
                else:
                    take.append(q)
            return skipped

        # The duplicate filter runs BEFORE the fallback is asked, and the
        # fallback is filtered the same way. Borrowing exactly the raw
        # shortfall first and filtering after (the order before this) left a
        # section short whenever the filter dropped a question -- while the
        # gap said the shortfall had been filled.
        repeats = accept([q for q in remaining if _fits_section(q, section)])
        if len(take) < section.question_count:
            gaps.append(
                f"Section {section.label} ({section.name}): needs {section.question_count} "
                f"questions worth {section.marks_per_question} marks each, only {len(take)} "
                f"available in the selected chapters."
            )
            before = len(take)
            repeats += accept([q for q in fallback_remaining if _fits_section(q, section)])
            if len(take) > before:
                gaps.append(
                    f"Section {section.label}: filled {len(take) - before} of the shortfall from "
                    f"outside the selected chapters so the section isn't left blank."
                )
        # Whatever the cause -- too few in the bank, or the rest repeating a
        # question already on the paper -- a section that prints short says so.
        if len(take) < section.question_count:
            why = (" after leaving out ones that repeat a question already on the paper"
                   if repeats else "")
            gaps.append(
                f"Section {section.label} ({section.name}): only {len(take)} of "
                f"{section.question_count} questions{why}; the section prints short."
            )
        for q in take:
            selected.append(q)
            if q in remaining:
                remaining.remove(q)
            elif q in fallback_remaining:
                fallback_remaining.remove(q)
            for cid in q.chapter_ids:
                used_chapters[cid] += 1

        # PARAKH Internal Choice Pairing ("OR")
        # Internal choice must be offered between questions of the same format and chapter
        if section.has_internal_choice and take:
            choice_quota = section.internal_choice_count if section.internal_choice_count > 0 else max(1, len(take) // 3)
            # Select target questions to receive an "OR" alternative
            eligible_for_or = take[-choice_quota:]
            offered = 0
            left_out_as_repeat = 0
            for primary_q in eligible_for_or:
                # Find best alternative from same chapter and section fit
                alt_pool = [
                    cand for cand in (remaining + fallback_remaining)
                    if _fits_section(cand, section) and cand.id != primary_q.id and cand.id not in paired_choice_ids
                ]
                if alt_pool:
                    # Intra-chapter choice prioritization (PARAKH guideline)
                    primary_chaps = set(primary_q.chapter_ids)
                    alt_pool.sort(
                        key=lambda cand: (
                            1 if primary_chaps and any(c in primary_chaps for c in cand.chapter_ids) else 0,
                            _score(cand, chapter_weights, used_chapters, tier, need_cbq, live),
                        ),
                        reverse=True,
                    )
                    # The partner is printed too, so it may not restate any
                    # question on the paper -- its own primary included. Checked
                    # down the ranking, not over the whole pool.
                    alt_q = next((c for c in alt_pool if not _repeats_paper(c, selected, paired)), None)
                    if alt_q is None:
                        left_out_as_repeat += 1
                        continue
                    primary_q.metadata["internal_choice_id"] = alt_q.id
                    primary_q.metadata["internal_choice_stem"] = alt_q.stem
                    primary_q.metadata["internal_choice_scheme"] = alt_q.answer_scheme.model_answer
                    primary_q.metadata["internal_choice_question"] = alt_q.model_dump(by_alias=False)
                    paired_choice_ids.add(alt_q.id)
                    paired.append(alt_q)
                    if alt_q in remaining:
                        remaining.remove(alt_q)
                    elif alt_q in fallback_remaining:
                        fallback_remaining.remove(alt_q)
                    offered += 1
            # A section offering fewer ORs than its blueprint asks for (the
            # Hindi Khand Gha rows need 2 and 1) says so: a silently missing
            # choice is a wrong paper the teacher would not notice.
            # Counted against the primaries actually printed: a section that
            # already prints short has its own gap above.
            if offered < len(eligible_for_or):
                why = ("the other alternatives repeat a question already on the paper"
                       if left_out_as_repeat else
                       "no other question of this section's marks and type is left to offer")
                gaps.append(
                    f"Section {section.label}: only {offered} of {len(eligible_for_or)} internal "
                    f"choices offered; {why}."
                )

    selected_ids = {q.id for q in selected}
    rejected = [q for q in candidates if q.id not in selected_ids and q.id not in paired_choice_ids]

    chapter_coverage = Counter()
    bloom_hist: Counter = Counter()
    difficulty_hist: Counter = Counter()
    cbq_count = 0
    for q in selected:
        for cid in q.chapter_ids:
            chapter_coverage[cid] += 1
        bloom_hist[q.bloom_level] += 1
        difficulty_hist[q.difficulty] += 1
        if is_competency_question(q):
            cbq_count += 1

    if selected:
        distinct_chapters = len(chapter_coverage)
        if distinct_chapters == 1 and len(selected) > 3:
            warnings.append("Selected paper is concentrated in a single chapter — weak coverage.")
    total_marks_selected = sum(q.marks for q in selected)
    if total_marks_selected != blueprint.total_marks:
        warnings.append(
            f"Selected paper totals {total_marks_selected} marks, blueprint target is {blueprint.total_marks}."
        )

    cbq_percentage = (cbq_count / len(selected)) if selected else 0.0
    if selected and cbq_percentage < competency_target:
        warnings.append(
            f"Selected paper has {round(cbq_percentage * 100, 1)}% competency-based questions "
            f"(CBSE target: {int(competency_target * 100)}%)."
        )

    if not signals["available"] and tier.lower() != "standard":
        warnings.append(
            f"Tiers unavailable for this subject: its questions of the same marks do not "
            f"differ in difficulty or Bloom level, so the {tier} tier cannot change "
            f"which questions are chosen."
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
            "competencyPercentage": round(cbq_percentage, 3),
            "tier": tier,
            "tierSignals": signals,
            "internalChoicesPaired": len(paired_choice_ids),
        },
        warnings=warnings,
        gaps=gaps,
    )


# ---- parallel sets ------------------------------------------------------------

_CHOICE_KEYS = ("internal_choice_id", "internal_choice_stem", "internal_choice_scheme",
                "internal_choice_question")


def _attach_choice(q: QuestionSchema, partner: QuestionSchema) -> None:
    """Make `partner` the printed OR alternative of `q`, as optimize() does."""
    q.metadata["internal_choice_id"] = partner.id
    q.metadata["internal_choice_stem"] = partner.stem
    q.metadata["internal_choice_scheme"] = partner.answer_scheme.model_answer
    q.metadata["internal_choice_question"] = partner.model_dump(by_alias=False)


def _sections_of(q: QuestionSchema, sections: list[SectionBlueprint]) -> list[SectionBlueprint]:
    """The sections `q` could sit in: those whose limits it meets (optimize()
    only places a question in such a section)."""
    return [s for s in sections if _fits_section(q, s)]


def _alternative(q: QuestionSchema, pool: list[QuestionSchema], used: set[str],
                 near_chapters: list[str],
                 sections: list[SectionBlueprint] | None = None) -> QuestionSchema | None:
    """The unused pool question that stands in for `q`: same marks and type are
    required, and so are the limits of `q`'s section (allowed difficulties and
    Bloom levels, as optimize() enforces for set A); then the closest in
    CBQ-ness, chapter, difficulty and Bloom level, then quality. None when the
    pool has no unused question of that shape."""
    fits = [c for c in pool if c.id not in used and c.marks == q.marks and c.type == q.type]
    homes = _sections_of(q, sections or [])
    if homes:
        fits = [c for c in fits if any(_fits_section(c, s) for s in homes)]
    if not fits:
        return None
    chapters, cbq = set(near_chapters), is_competency_question(q)
    return max(fits, key=lambda c: (
        is_competency_question(c) == cbq,
        bool(chapters & set(c.chapter_ids)),
        c.difficulty == q.difficulty,
        c.bloom_level == q.bloom_level,
        c.quality_score,
    ))


def alternative_sets(primary: list[QuestionSchema], pool: list[QuestionSchema],
                     set_count: int,
                     sections: list[SectionBlueprint] | None = None,
                     ) -> tuple[list[list[QuestionSchema]], list[int]]:
    """Set A is `primary`; each later set replaces every question -- and every
    OR alternative -- with a different pool question of the same marks and type
    that also meets its section's allowed difficulties and Bloom levels
    (`sections`, the blueprint's). Matching on marks and type alone let set B
    print a 'hard' or 'apply' question in a section set A kept easy/recall.

    Sets B and C used to be set A rotated: the same questions in another order
    (A and C identical when a section had two), which is no protection against
    copying. A question is reused only when the pool has no unused question of
    its shape left; the second list counts, per set, the printed questions it
    shares with an earlier set (0 for set A).
    """
    used = {q.id for q in primary}
    used |= {q.metadata["internal_choice_id"] for q in primary
             if q.metadata.get("internal_choice_id")}
    sets: list[list[QuestionSchema]] = [primary]
    overlaps = [0]
    for _ in range(1, set_count):
        variant: list[QuestionSchema] = []
        overlap = 0
        for q in primary:
            alt = _alternative(q, pool, used, q.chapter_ids, sections)
            if alt is None:
                # Out of alternatives: print set A's question, OR partner and all.
                variant.append(q.model_copy(deep=True))
                overlap += 2 if q.metadata.get("internal_choice_id") else 1
                continue
            used.add(alt.id)
            alt = alt.model_copy(deep=True)
            for key in _CHOICE_KEYS:
                alt.metadata.pop(key, None)
            if q.metadata.get("internal_choice_id"):
                # An OR choice is between questions of one format (PARAKH), so
                # the partner matches this set's question, not set A's partner:
                # set A's partner type is whatever optimize() found, and matching
                # it ran Science 10's pool out of 3-mark short_answers by set C.
                partner = _alternative(alt, pool, used, alt.chapter_ids, sections)
                if partner is None:
                    for key in _CHOICE_KEYS:
                        if key in q.metadata:
                            alt.metadata[key] = q.metadata[key]
                    overlap += 1
                else:
                    used.add(partner.id)
                    _attach_choice(alt, partner)
            variant.append(alt)
        sets.append(variant)
        overlaps.append(overlap)
    return sets, overlaps


def tier_changed_selection(candidates: list[QuestionSchema], blueprint: Blueprint,
                           fallback_candidates: list[QuestionSchema] | None,
                           selected: list[QuestionSchema]) -> bool:
    """Did the blueprint's tier pick other questions than the standard tier would?

    tier_signals only says the bank *could* rank differently. After the merge
    (--check --relink-in-memory) Science 10 has one "apply" question among 21
    at 1 mark: a signal, yet the section takes 20 of them and every tier
    prints the same paper. Runs on copies: optimize() writes OR pairings into
    the questions it is given.
    """
    if (blueprint.tier or "standard").lower() == "standard":
        return True
    standard = optimize([q.model_copy(deep=True) for q in candidates],
                        blueprint.model_copy(update={"tier": "standard"}),
                        fallback_candidates=[q.model_copy(deep=True)
                                             for q in (fallback_candidates or [])])
    return {q.id for q in standard.selected_questions} != {q.id for q in selected}
