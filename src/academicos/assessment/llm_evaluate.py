"""LLM-based answer evaluation via Sarvam's flagship model, for the mobile
scan-and-grade flow.

`evaluate.py`'s keyword matcher is fast and free, and stays the default for
the desktop review queue. The mobile capture flow asks for something closer
to how a human examiner actually marks a script — reading the full answer,
giving credit for a correct method with an arithmetic slip, not penalizing
untidy handwriting or phrasing — which a keyword matcher structurally cannot
do. This module sends the question, the official marking scheme, and the
OCR'd answer to `sarvam-105b` and asks it to mark the way a CBSE examiner is
instructed to: award every marking point independently, give the benefit of
the doubt on a legible alternate-but-correct approach, and say plainly when
the handwriting/OCR quality means a human should double-check.

Returns the same `Evaluation`/`MarkingPointOutcome` shapes as evaluate.py, so
it drops into the same knowledge-tracking and review-queue pipeline — the
teacher's swipe-review screen doesn't need to know which engine scored a
given answer.
"""
from __future__ import annotations

import json
import logging
import os
import re

import requests

from .evaluate import Evaluation, MarkingPointOutcome, REVIEW_THRESHOLD, ocr_risk_flags
from .schemas import AnswerSchemeSchema, QuestionSchema

log = logging.getLogger(__name__)

CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
MODEL = "sarvam-105b"
# sarvam-105b is a reasoning model — it spends anywhere from a few hundred to
# over a thousand tokens thinking before it writes the answer, and a tight
# budget truncates mid-JSON rather than mid-thought (confirmed empirically: a
# real CBSE question stem burned past 1800 tokens and cut off inside the
# marking_points array with no closing brace at all). 4000 leaves headroom for
# a verbose multi-point descriptive answer plus its reasoning.
MAX_TOKENS = 4000
# Observed a real timeout at 45s during testing on an ordinary 2-mark
# descriptive question — the reasoning-heavy completion can legitimately run
# past that. 90s trades a slower worst case for fewer answers falling back to
# "a teacher must mark this" when the model would have finished fine.
_TIMEOUT_SEC = 90

# Mirrors the CBSE spot-evaluation circular's own instructions to examiners:
# give credit for a correct alternate method, don't let an arithmetic slip
# zero out otherwise-correct working, and don't penalize language for a
# content-graded subject. Grounding the prompt in the board's own marking
# philosophy rather than a generic rubric is what "liberty of correction"
# means in this codebase.
_SYSTEM_PROMPT = """You are a CBSE Class X/XII examiner marking one answer from a scanned \
answer booklet. Follow CBSE spot-evaluation conventions:

- Award each marking point independently and on its own merits.
- Give the benefit of the doubt: if the method is correct but there is a small \
arithmetic slip, award marks for the correct steps and only deduct for the step \
that is actually wrong.
- A correct answer reached by a valid alternate method (not the one in the \
official scheme) earns full marks for that point — do not require the exact \
wording or steps of the scheme.
- Do not penalize spelling, grammar, or untidy phrasing in a content-graded \
subject; judge whether the required idea is present.
- The answer text came from OCR of handwriting, which can drop or garble \
characters. If a word looks like an OCR error of a plausible correct term \
(not a different, wrong term), give the student the benefit of the doubt but \
lower your confidence and say so.
- If the answer is blank, off-topic, or shows no relevant working, award zero \
honestly — liberty of correction is not liberty to invent credit that is not there.

Respond with ONLY a single JSON object, no markdown fences, no commentary, in \
exactly this shape:
{
  "marking_points": [
    {"id": "<point id from the scheme>", "awarded": true|false, "marks": <int>, "reason": "<one sentence, cite what the student wrote>"}
  ],
  "selected_option": "<A|B|C|D|null — objective questions only>",
  "confidence": <0.0-1.0, your confidence the transcription was clean enough to mark reliably>,
  "reasoning": "<one or two sentence overall verdict>",
  "strengths": ["<short phrase>", ...],
  "gaps": ["<short phrase>", ...],
  "misconceptions": ["<short phrase, only if a specific misconception is evident>", ...]
}"""


class LLMEvaluationError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise LLMEvaluationError("SARVAM_API_KEY is not set")
    return key


def _build_user_prompt(question: QuestionSchema, scheme: AnswerSchemeSchema,
                       answer: str, ocr_warnings: list[str]) -> str:
    objective = bool(scheme.metadata.get("objective"))
    lines = [
        f"Subject: {question.subject}, Class {question.grade}, {question.marks} mark(s)",
        f"Question: {question.stem}",
    ]
    if objective:
        options = scheme.metadata.get("options") or {}
        key = scheme.metadata.get("correctOption") or "(not recorded)"
        lines.append(f"Options: {options}")
        lines.append(f"Correct option per official key: {key}")
        lines.append('Marking points: [{"id": "%s", "description": "correct option selected"}]'
                     % (scheme.marking_points[0].id if scheme.marking_points else "mp1"))
    else:
        points = [{"id": p.id, "description": p.description, "marks": p.marks}
                 for p in scheme.marking_points]
        lines.append(f"Official marking scheme ({scheme.total_marks} marks total): "
                     f"{json.dumps(points, ensure_ascii=False)}")
        if scheme.model_answer:
            lines.append(f"Reference model answer: {scheme.model_answer}")
    lines.append(f"Student's answer (from OCR): {answer!r}")
    if ocr_warnings:
        lines.append(f"Known OCR risk signals for this answer: {'; '.join(ocr_warnings)}")
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise LLMEvaluationError(f"no JSON object in model response: {raw[:200]!r}")
    return json.loads(raw[start:end + 1])


def _call_sarvam(system: str, user: str) -> dict:
    key = _api_key()
    resp = requests.post(
        CHAT_URL,
        headers={"api-subscription-key": key, "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,   # marking should be consistent, not creative
            "max_tokens": MAX_TOKENS,
        },
        timeout=_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content")
    if not content:
        raise LLMEvaluationError(
            f"model returned no content (finish_reason={data['choices'][0].get('finish_reason')}); "
            "likely truncated mid-reasoning — raise max_tokens")
    return _extract_json(content)


def evaluate_answer_llm(question: QuestionSchema, scheme: AnswerSchemeSchema,
                        student_answer: str, *, concept_label: str | None = None) -> Evaluation:
    """LLM-scored equivalent of `evaluate.evaluate_answer`. Never raises — on
    any failure (network, malformed JSON, missing key) it returns a
    needs-review evaluation rather than breaking the scan-and-grade pipeline
    for the rest of the booklet.
    """
    answer = (student_answer or "").strip()
    max_marks = scheme.total_marks or question.marks or 1
    ocr_flags = ocr_risk_flags(answer)

    if not any(c.isalnum() for c in answer):
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks, verdict="blank",
            confidence=0.95, reasoning="No answer written for this question.",
            ocr_warnings=ocr_flags, needs_review=False,
        )

    try:
        result = _call_sarvam(_SYSTEM_PROMPT, _build_user_prompt(question, scheme, answer, ocr_flags))
    except Exception as exc:
        log.warning("LLM evaluation failed for %s: %s", question.id, exc)
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks, verdict="partialCredit",
            confidence=0.0,
            reasoning=f"AI evaluation unavailable ({exc}); a teacher must mark this answer.",
            ocr_warnings=ocr_flags, needs_review=True,
        )

    objective = bool(scheme.metadata.get("objective"))
    if objective:
        return _from_objective_result(question, scheme, result, max_marks, ocr_flags)
    return _from_descriptive_result(question, scheme, result, max_marks, ocr_flags, concept_label)


def _from_objective_result(question: QuestionSchema, scheme: AnswerSchemeSchema, result: dict,
                           max_marks: int, ocr_flags: list[str]) -> Evaluation:
    key = str(scheme.metadata.get("correctOption") or "").upper()
    chosen = str(result.get("selected_option") or "").upper().strip("()")
    confidence = round(max(0.0, min(1.0, float(result.get("confidence", 0.5)))), 2)
    if ocr_flags:
        confidence = min(confidence, 0.6)

    if not key:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks,
            verdict="partialCredit", confidence=0.0,
            reasoning="No correct option recorded in the marking scheme — teacher must set the key.",
            ocr_warnings=ocr_flags, needs_review=True,
        )
    if not chosen:
        return Evaluation(
            question_id=question.id, awarded_marks=0, max_marks=max_marks,
            verdict="noCredit", confidence=min(confidence, 0.5),
            reasoning=result.get("reasoning") or "Could not identify which option the student selected.",
            ocr_warnings=ocr_flags, needs_review=True,
        )

    correct = chosen == key
    point = scheme.marking_points[0] if scheme.marking_points else None
    outcomes = []
    if point:
        outcomes.append(MarkingPointOutcome(
            marking_point_id=point.id, description=point.description, awarded=correct,
            marks=max_marks if correct else 0,
            reason=f"student selected {chosen}; key is {key}", similarity=1.0 if correct else 0.0,
        ))
    return Evaluation(
        question_id=question.id, awarded_marks=max_marks if correct else 0, max_marks=max_marks,
        verdict="fullCredit" if correct else "noCredit", confidence=confidence,
        reasoning=result.get("reasoning") or (
            f"Selected option {chosen}, which matches the key." if correct else
            f"Selected option {chosen}; the correct option is {key}."),
        marking_points=outcomes,
        strengths=([point.description] if correct and point else []),
        gaps=([] if correct else [f"Correct option was {key}"]),
        ocr_warnings=ocr_flags,
        needs_review=confidence < REVIEW_THRESHOLD,
    )


def _from_descriptive_result(question: QuestionSchema, scheme: AnswerSchemeSchema, result: dict,
                             max_marks: int, ocr_flags: list[str],
                             concept_label: str | None) -> Evaluation:
    by_id = {p.id: p for p in scheme.marking_points}
    outcomes: list[MarkingPointOutcome] = []
    for raw_point in result.get("marking_points", []):
        pid = raw_point.get("id")
        scheme_point = by_id.get(pid)
        max_for_point = scheme_point.marks if scheme_point else 1
        awarded = bool(raw_point.get("awarded"))
        marks = raw_point.get("marks")
        marks = max(0, min(int(marks) if marks is not None else (max_for_point if awarded else 0),
                           max_for_point))
        outcomes.append(MarkingPointOutcome(
            marking_point_id=pid or "", description=(scheme_point.description if scheme_point
                                                      else str(raw_point.get("reason", ""))),
            awarded=awarded, marks=marks,
            reason=str(raw_point.get("reason", "")), similarity=1.0 if awarded else 0.0,
        ))
    # A point in the scheme the model silently dropped counts as not awarded,
    # rather than vanishing from the total.
    seen = {o.marking_point_id for o in outcomes}
    for pid, sp in by_id.items():
        if pid not in seen:
            outcomes.append(MarkingPointOutcome(
                marking_point_id=pid, description=sp.description, awarded=False, marks=0,
                reason="not addressed in the model's response", similarity=0.0,
            ))

    awarded = min(sum(o.marks for o in outcomes), max_marks)
    verdict = "fullCredit" if awarded >= max_marks else "partialCredit" if awarded > 0 else "noCredit"
    confidence = round(max(0.0, min(1.0, float(result.get("confidence", 0.5)))), 2)
    if ocr_flags:
        confidence = min(confidence, 0.65)

    return Evaluation(
        question_id=question.id, awarded_marks=awarded, max_marks=max_marks, verdict=verdict,
        confidence=confidence,
        reasoning=result.get("reasoning") or f"{awarded}/{max_marks} marks awarded.",
        marking_points=outcomes,
        strengths=list(result.get("strengths", [])) or [o.description for o in outcomes if o.awarded],
        gaps=list(result.get("gaps", [])) or [o.description for o in outcomes if not o.awarded],
        misconceptions=list(result.get("misconceptions", [])),
        ocr_warnings=ocr_flags,
        needs_review=confidence < REVIEW_THRESHOLD or bool(ocr_flags),
    )
