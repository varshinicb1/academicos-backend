# The Algorithms Library

> We are not building an AI tutor. We are discovering the algorithms of human
> learning.

Every algorithm here is:
- **Evidence-grounded** — parameters and formulas traceable to research
  (see `docs/research/learning-science.md`) or to corpus-derived data.
- **Versioned** — each algorithm declares a `version`; behavior changes are
  additive or explicitly versioned.
- **Eval-able** — each has a `metrics` entry point against held-out data.

Planned library (Phase 4 of roadmap):

| Algorithm | Purpose | Status |
|---|---|---|
| `knowledge_mastery` | Mastery % instead of marks (confidence + retention + application) | implementing |
| `forgetting_prediction` | When will this child forget? → auto-revise (FSRS-6 DSR engine, v2) | implemented |
| `attention_prediction` | Engagement/fatigue estimation from session signals | planned |
| `motivation_prediction` | Detect low motivation → adapt difficulty/breaks | planned |
| `habit_formation` | Consistency tracking + streaks | planned |
| `emotional_state` | Frustration/confidence signals from interactions | planned |
| `curiosity_engine` | Route to next concept by curiosity signals | planned |
| `parent_guidance` | "Just say this today" prompts | planned |
| `teacher_assistance` | "Today's priority: 12 students need fractions" | planned |
| `career_discovery` | Behavior → career graph | planned |
| `revision_scheduler` | Spaced-repetition schedule per learner (FSRS-6 backed) | implemented |
| `reward_engine` | Feedback/celebration policy | planned |
| `study_planner` | When/how long/what to study (prereq plan + spacing) | implemented |
| `confidence_model` | Per-concept confidence dynamics | implemented |
| `lifelong_learner_model` | Cross-subject, cross-year model | implemented |

Implemented in this module set:

| Module | Model | Status |
|---|---|---|
| `forgetting.py` | `model="fsrs"` (default, v2) — FSRS-6 DSR; `model="exponential"` (v1, legacy) | implemented |
| `fsrs.py` | FSRS-6 (Open Spaced Repetition, MIT) — 21 params, power-law forgetting curve, difficulty/stability/retrievability | implemented |
| `mastery.py` | Weighted mastery + recency-decayed retention | implemented |
| `learner_model.py` | Interaction log → ConceptState per concept | implemented |
| `study_planner.py` | A12 v2 — prereq topological order (P1.8) + FSRS retention action (learn/retrieve/skip) + daily capacity window | implemented |
| `misconceptions.py` | P1.10 — research-shaped template catalog (FCI, fraction/decimals reviews), offline answer diagnosis (`match_answer`), profile merge | implemented |
| `confidence_model.py` | A13 — Bandura 4-source self-efficacy per concept + calibration (over/under/calibrated), deterministic v1 | implemented |
| `lifelong_model.py` | A14 — CBSE-year cross-year portraits, recurring-concept continuity (accuracy chains), subject breakdown, growth | implemented |

Evidence: see `docs/research/fsrs.md` (FSRS-6 math, defaults, validation) and
`docs/research/learning-science.md`. Validation: `scripts/simulate_spaced_repetition.py`
reproduces the adaptive-spacing result on our own learner model (85-92% fewer
reviews than a fixed 7-day cadence at equal retention). A13/A14 evidence:
`docs/research/confidence.md`, `docs/research/lifelong.md`.

## Pipeline position

```
interactions (answers, time, affect)
  → learner model (mastery/confidence/retention state)
  → algorithms (schedule, adapt, guide)
  → actions (study plan, revision, encouragement)
  → update learner model        ← the loop
```
