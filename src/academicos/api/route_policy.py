"""Who may call each route: one table, and a test that holds every route to it.

Why this exists (audit item 8.4, reproduced 2026-09-21): of 131 operations,
107 depended on `get_current_user` and about 30 added `require_principal`.
`get_current_user` proves identity, not permission, so a student-role
account got 200 on every classmate's answers
(GET /evaluations/assessment/{aid}/question/{qid}), changed another student's
marks (POST .../finalize, .../award), read an unreleased paper and exported
its answer key. Nothing forced a new route to choose who could call it, so
each one inherited "anyone with a login".

Why a module-level table rather than an attribute on each route: the policy
of the whole API is reviewable in one screen, a route in a module this
stream must not edit (curriculum/routes.py, owned by the calendar stream)
can still be classified, and tests/test_role_gates.py can fail on any
(method, path) the app serves that is missing here -- and on any entry here
that no longer exists. The same test checks each entry against the route's
real dependency chain (a `staff` route must depend on `require_staff` or
`require_principal`, and so on), so the table cannot drift into describing
gates that are not there.

Levels, loosest to strictest:

- `public`: no identity. Health probes, login/register, OpenAPI docs, and
  reads of official, school-independent CBSE reference data.
- `api_key`: the /v1 question-bank API (qbank_routes.py). A scoped machine
  key, never a user; api_keys.py explains why student data is unexpressible.
- `any_user`: any logged-in account of any role, the caller's own school.
- `self_or_staff`: a student may call it only about themselves
  (authz.require_school_owns_student); staff for any student of their school.
- `staff`: teacher or principal (auth_routes.require_staff).
- `principal`: principal only (auth_routes.require_principal).

When unsure, a route is `staff`: denying a student a teacher screen costs a
support message, letting them in costs another child's marks.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple

PUBLIC = "public"
API_KEY = "api_key"
ANY_USER = "any_user"
SELF_OR_STAFF = "self_or_staff"
STAFF = "staff"
PRINCIPAL = "principal"

LEVELS = (PUBLIC, API_KEY, ANY_USER, SELF_OR_STAFF, STAFF, PRINCIPAL)


class Policy(NamedTuple):
    level: str
    why: str = ""


def _p(level: str, why: str = "") -> Policy:
    if level not in LEVELS:
        raise ValueError(f"unknown route policy level {level!r}; one of {LEVELS}")
    return Policy(level, why)


ROUTE_POLICY: dict[tuple[str, str], Policy] = {
    # ---- OpenAPI docs (FastAPI) ----
    ("GET", "/openapi.json"): _p(PUBLIC),
    ("GET", "/docs"): _p(PUBLIC),
    ("GET", "/docs/oauth2-redirect"): _p(PUBLIC),
    ("GET", "/redoc"): _p(PUBLIC),

    # ---- api/main.py ----
    ("GET", "/health"): _p(PUBLIC),
    ("GET", "/health/llm"): _p(PUBLIC, "counts and timings only, never prompt content"),
    ("GET", "/health/storage"): _p(PUBLIC, "backend name and per-table reachability, no rows"),
    ("GET", "/v1/registry/stats"): _p(ANY_USER, "one corpus-wide document count"),
    ("POST", "/v1/search"): _p(ANY_USER, "shared NCERT corpus, no school data"),
    ("POST", "/v1/agent/doubt"): _p(ANY_USER, "a student asking a doubt is the intended user"),
    ("POST", "/v1/agent/score"): _p(ANY_USER, "scores text in the body; stores nothing about a student"),
    ("POST", "/v1/agent/analyze"): _p(ANY_USER, "analyses a corpus paper id, not a school's paper"),
    ("POST", "/v1/graph/neighbors"): _p(ANY_USER, "concept graph, no learner data"),
    ("POST", "/v1/graph/paths"): _p(ANY_USER, "concept graph, no learner data"),
    ("POST", "/v1/revise"): _p(SELF_OR_STAFF, "body.learner is a student id; a school_B teacher read a school_A student's retention until 2026-09-21"),
    ("POST", "/v1/plan"): _p(SELF_OR_STAFF, "optional body.learner, same check as /v1/revise"),
    ("POST", "/v1/question/solve"): _p(ANY_USER, "maps question text to concepts"),

    # ---- the question-bank API (assessment/qbank_routes.py) ----
    ("GET", "/v1/questions"): _p(API_KEY),
    ("GET", "/v1/questions/{question_id}"): _p(API_KEY),
    ("GET", "/v1/questions/{question_id}/scheme"): _p(API_KEY),
    ("GET", "/v1/subtopics/{subtopic_id}/questions"): _p(API_KEY),
    ("GET", "/v1/facets"): _p(API_KEY),

    # ---- assessment/auth_routes.py ----
    ("POST", "/api/v1/auth/register"): _p(PUBLIC, "needs an invite code or the bootstrap key (Task 301)"),
    ("POST", "/api/v1/auth/login"): _p(PUBLIC),
    ("POST", "/api/v1/auth/logout"): _p(PUBLIC, "deletes the bearer's own session if any; nothing to leak"),
    ("GET", "/api/v1/auth/me"): _p(ANY_USER),
    ("GET", "/api/v1/auth/users"): _p(PRINCIPAL),
    ("POST", "/api/v1/auth/invites"): _p(PRINCIPAL),
    ("GET", "/api/v1/auth/invites"): _p(PRINCIPAL),
    ("DELETE", "/api/v1/auth/invites/{code}"): _p(PRINCIPAL),

    # ---- assessment/routes.py (assessment designer) ----
    ("POST", "/api/v1/blueprints/generate"): _p(STAFF),
    ("GET", "/api/v1/schools/{school_id}/templates"): _p(STAFF, "paper-design input"),
    ("GET", "/api/v1/schools/{school_id}/paper-templates"): _p(STAFF, "paper-design input"),
    ("POST", "/api/v1/questions/search"): _p(STAFF, "builds papers; a student browsing it is previewing the exam pool"),
    ("POST", "/api/v1/questions/optimize"): _p(STAFF),
    ("POST", "/api/v1/papers/generate"): _p(STAFF),
    ("POST", "/api/v1/papers/quick-generate"): _p(STAFF),
    ("POST", "/api/v1/papers/generate-from-ids"): _p(STAFF),
    ("GET", "/api/v1/paper-timing"): _p(PRINCIPAL, "renewal metric; Task 201 sets the same"),
    ("GET", "/api/v1/papers/{paper_id}"): _p(STAFF, "an unreleased paper's stems"),
    ("POST", "/api/v1/papers/{paper_id}/export/{fmt}"): _p(STAFF, "includes the answer key"),
    ("GET", "/api/v1/papers/{paper_id}/file"): _p(STAFF, "includes the answer-key PDF"),
    ("POST", "/api/v1/assessments"): _p(STAFF),
    ("GET", "/api/v1/assessments"): _p(STAFF, "assessments carry the blueprint of an unreleased exam"),
    ("GET", "/api/v1/assessments/{assessment_id}"): _p(STAFF, "same as the list"),
    ("PUT", "/api/v1/assessments/{assessment_id}"): _p(STAFF),
    ("DELETE", "/api/v1/assessments/{assessment_id}"): _p(STAFF),
    ("PATCH", "/api/v1/assessments/{assessment_id}/status"): _p(STAFF),
    ("PATCH", "/api/v1/assessments/{assessment_id}/approve"): _p(PRINCIPAL),

    # ---- assessment/pillar_routes.py ----
    ("GET", "/api/v1/schools/{school_id}/school-templates"): _p(STAFF),
    ("POST", "/api/v1/schools/{school_id}/school-templates"): _p(STAFF),
    ("GET", "/api/v1/schools/{school_id}/school-templates/{template_id}/sections"): _p(STAFF),
    ("DELETE", "/api/v1/schools/{school_id}/school-templates/{template_id}"): _p(STAFF),
    ("POST", "/api/v1/assessments/{assessment_id}/answer-key"): _p(STAFF),
    ("GET", "/api/v1/catalog"): _p(PUBLIC, "official CBSE subject/grade catalogue"),
    ("GET", "/api/v1/catalog/{subject}/{grade}/chapters"): _p(PUBLIC, "official CBSE chapter list"),
    ("GET", "/api/v1/syllabus/{subject}/{grade}"): _p(PUBLIC, "official CBSE syllabus"),
    ("GET", "/api/v1/syllabus/{subject}/{grade}/timetable"): _p(PUBLIC, "computed from the public syllabus"),
    ("GET", "/api/v1/mail/status"): _p(PUBLIC, "configured yes/no per backend, no addresses or keys"),
    ("POST", "/api/v1/mail/send-paper"): _p(STAFF),
    ("POST", "/api/v1/evaluations/answer"): _p(STAFF, "writes a mark for any student id in the body"),
    ("POST", "/api/v1/evaluations/sheet"): _p(STAFF, "writes a whole sheet's marks"),
    ("POST", "/api/v1/evaluations/sheet/{assessment_id}/{student_id}/review/{question_id}"): _p(STAFF),
    ("POST", "/api/v1/evaluations/sheet/{assessment_id}/{student_id}/finalize"): _p(STAFF, "a student finalized a classmate's sheet until 2026-09-21"),
    ("GET", "/api/v1/knowledge/{student_id}"): _p(SELF_OR_STAFF),
    ("GET", "/api/v1/knowledge/{student_id}/report"): _p(SELF_OR_STAFF),
    ("POST", "/api/v1/practice/generate"): _p(SELF_OR_STAFF),
    ("POST", "/api/v1/practice/submit"): _p(SELF_OR_STAFF, "checked against the practice set's own student"),
    ("GET", "/api/v1/insights/class/{assessment_id}"): _p(STAFF, "names at-risk students"),
    ("GET", "/api/v1/insights/school/{school_id}"): _p(PRINCIPAL),

    # ---- assessment/grade_by_question.py ----
    ("GET", "/api/v1/evaluations/assessment/{assessment_id}/question/{question_id}"): _p(STAFF, "every student's answer to one question"),
    ("POST", "/api/v1/evaluations/assessment/{assessment_id}/question/{question_id}/award"): _p(STAFF, "writes marks for many students"),

    # ---- assessment/mobile_routes.py (scan-and-grade; all teacher work) ----
    ("POST", "/api/v1/scan/sessions"): _p(STAFF),
    ("POST", "/api/v1/scan/sessions/{session_id}/pages"): _p(STAFF),
    ("POST", "/api/v1/scan/sessions/{session_id}/process"): _p(STAFF),
    ("GET", "/api/v1/scan/sessions/{session_id}/review"): _p(STAFF),
    ("POST", "/api/v1/scan/sessions/{session_id}/review/{question_id}"): _p(STAFF),
    ("POST", "/api/v1/scan/sessions/{session_id}/finalize"): _p(STAFF),
    ("GET", "/api/v1/scan/sessions/{session_id}/pages/{page_no}/image"): _p(STAFF, "a photographed answer sheet"),
    ("GET", "/api/v1/scan/sessions/{session_id}/raw-pdf"): _p(STAFF),
    ("GET", "/api/v1/scan/sessions/{session_id}/corrected-pdf"): _p(STAFF),

    # ---- assessment/ingest_routes.py ----
    ("POST", "/api/v1/ingest/school-documents"): _p(STAFF, "files a document against the school"),

    # ---- assessment/consent_routes.py (DPDP Act 2023) ----
    ("GET", "/api/v1/consent"): _p(STAFF, "every guardian's name at the school"),
    ("POST", "/api/v1/consent"): _p(STAFF, "a student recording their own parent's consent is not consent"),
    ("GET", "/api/v1/consent/missing"): _p(STAFF, "which students at the school have no parental consent"),
    ("GET", "/api/v1/consent/students/{student_id}"): _p(STAFF, "takes any student id and returns the guardian record"),
    ("DELETE", "/api/v1/consent/students/{student_id}"): _p(STAFF),

    # ---- assessment/paper_template_routes.py (paper-builder Tasks 901-906) ----
    ("GET", "/api/v1/schools/{school_id}/teacher-templates"): _p(STAFF, "presets + mine + school-shared"),
    ("POST", "/api/v1/schools/{school_id}/teacher-templates"): _p(STAFF),
    ("GET", "/api/v1/schools/{school_id}/teacher-templates/{template_id}"): _p(STAFF, "owner or shared; private templates stay private"),
    ("PUT", "/api/v1/schools/{school_id}/teacher-templates/{template_id}"): _p(STAFF, "owner only (handler)"),
    ("DELETE", "/api/v1/schools/{school_id}/teacher-templates/{template_id}"): _p(STAFF, "owner or principal (handler)"),
    ("POST", "/api/v1/schools/{school_id}/teacher-templates/{template_id}/share"): _p(STAFF, "owner or principal (handler)"),
    ("POST", "/api/v1/schools/{school_id}/teacher-templates/{template_id}/duplicate"): _p(STAFF),
    ("POST", "/api/v1/schools/{school_id}/teacher-templates/{template_id}/availability"): _p(STAFF),
    ("POST", "/api/v1/papers/generate-from-template"): _p(STAFF),
    ("POST", "/api/v1/papers/{paper_id}/questions/{slot}/swap"): _p(STAFF, "edits the teacher's own saved paper"),
    ("POST", "/api/v1/papers/{paper_id}/questions/{slot}"): _p(STAFF, "edits the teacher's own saved paper"),
    ("DELETE", "/api/v1/papers/{paper_id}/questions/{slot}"): _p(STAFF, "removes a question from the teacher's own saved paper"),
    # ---- curriculum/routes.py ----
    # Classified from the dependencies the file has today; this stream does
    # not edit it (the calendar stream owns it). Unauthenticated reads are
    # `public` because that is what they are, not what they should be.
    ("GET", "/api/v1/curriculum/boards"): _p(ANY_USER, "board reference data; login required since calendar Task 101"),
    ("GET", "/api/v1/curriculum/academic-years"): _p(ANY_USER, "?school_id= must be the caller's own school (calendar Task 101)"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/grades"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/grades/{grade_id}/subjects"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/subjects/{subject_id}/books"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/books/{book_id}/units"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/books/{book_id}/chapters"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/chapters/{chapter_id}/topics"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("POST", "/api/v1/curriculum/books/{book_id}/toc/ingest"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/seed/cbse10"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/extract"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/extraction-runs/{run_id}"): _p(ANY_USER, "school-scoped; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/extraction-runs/{run_id}/approve"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/topics"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/topics/{topic_id}/subtopics"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/topics/{topic_id}"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/subtopics/{subtopic_id}"): _p(PRINCIPAL),
    ("DELETE", "/api/v1/curriculum/subtopics/{subtopic_id}"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/units/{unit_id}/sequence"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/chapters/{chapter_id}/sequence"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/topics/{topic_id}/sequence"): _p(PRINCIPAL),
    ("PATCH", "/api/v1/curriculum/subtopics/{subtopic_id}/sequence"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/questions/tag"): _p(ANY_USER, "a write any role can make; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/questions/by-subtopics"): _p(ANY_USER, "returns question ids only; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(ANY_USER, "the school's own calendar"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays"): _p(PRINCIPAL),
    ("PUT", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(PRINCIPAL, "correct the calendar (calendar Task 103)"),
    ("DELETE", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays/{holiday_id}"): _p(PRINCIPAL, "remove a mistyped holiday (calendar Task 103)"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays"): _p(ANY_USER, "the school's own holidays"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/period-configuration"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/subject-period-allocations"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/subject-period-allocations"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable-slots"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable-slots"): _p(ANY_USER),
    ("DELETE", "/api/v1/curriculum/timetable-slots/{slot_id}"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/working-days"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/books/{book_id}/teaching-time-estimates"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/subtopics/{subtopic_id}/teaching-time-estimate"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/books/{book_id}/schedule"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/books/{book_id}/schedule"): _p(ANY_USER, "includes teacher notes; should be staff (reported, not edited)"),
    ("GET", "/api/v1/curriculum/schedule"): _p(ANY_USER, "includes teacher notes; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/teacher-assignments"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/my-schedule"): _p(ANY_USER, "only the caller's own assigned books; empty for a student"),
    ("PATCH", "/api/v1/curriculum/scheduled-lessons/{lesson_id}"): _p(ANY_USER, "handler admits only the assigned teacher or a principal"),
    ("POST", "/api/v1/curriculum/scheduled-lessons/{lesson_id}/reschedule"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/books/{book_id}/schedule/push"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/scheduled-lessons/{lesson_id}/history"): _p(ANY_USER, "handler admits only the assigned teacher or a principal"),
    ("POST", "/api/v1/curriculum/student-enrollments"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/my-class-schedule"): _p(ANY_USER, "handler admits students only, their own grade"),
    ("GET", "/api/v1/curriculum/my-progress"): _p(ANY_USER, "handler admits students only, their own grade"),
    ("GET", "/api/v1/curriculum/reporting/coverage"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/reporting/delayed-topics"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/export"): _p(PRINCIPAL),
}


def app_operations(app) -> frozenset[tuple[str, str]]:
    """Every (method, path) the app serves, HEAD excluded. Walks
    `_IncludedRouter` wrappers the same way tests/test_route_inventory.py
    does (newer FastAPI nests included routers instead of flattening them)."""
    return frozenset((m, path) for m, path, _ in iter_routes(app))


def iter_routes(app) -> Iterable[tuple[str, str, object]]:
    """(method, path, route) for every served operation, HEAD excluded."""
    def walk(routes):
        for r in routes:
            if hasattr(r, "original_router"):
                yield from walk(r.original_router.routes)
            elif hasattr(r, "methods") and r.methods:
                for m in sorted(r.methods):
                    if m != "HEAD":
                        yield m, r.path, r

    yield from walk(app.routes)
