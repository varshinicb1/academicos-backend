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
- `owner`: an owner account only (auth_routes.require_owner, ROLE-4): its
  schools, switching between them, and the per-school overview. Acting in a
  school, an owner passes every other level as that school's principal.

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
OWNER = "owner"
# Delegable principal actions (M1.4): the principal, or a teacher granted
# that capability (auth_routes.require_admin). ADMIN["leave"] == "admin:leave".
ADMIN = {c: f"admin:{c}" for c in ("users", "calendar", "timetable", "leave", "exams", "qbank_review", "reports")}

LEVELS = (PUBLIC, API_KEY, ANY_USER, SELF_OR_STAFF, STAFF, PRINCIPAL, OWNER, *ADMIN.values())


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
    ("GET", "/health/errors"): _p(PUBLIC, "NFR-7: error counts only, never a message"),
    ("POST", "/api/v1/client-errors"): _p(PUBLIC, "NFR-7: the apps report what they caught, signed in or not; scrubbed, rate-limited"),
    ("GET", "/api/v1/ops/errors"): _p(PUBLIC, "NFR-7: operator key checked in the handler; 404 unless ACOS_OPS_KEY is set"),
    ("GET", "/api/v1/ops/usage"): _p(PUBLIC, "NFR-7: per-school counts; operator key checked in the handler; 404 unless ACOS_OPS_KEY is set"),
    # INT-2: a person's own number and SMS/WhatsApp consent; the handler admits
    # staff and parents only (a student is a minor: their parent is messaged).
    ("GET", "/api/v1/me/contact"): _p(ANY_USER, "the caller's own number only; students refused in the handler"),
    ("PUT", "/api/v1/me/contact"): _p(ANY_USER, "the caller's own number only; students refused in the handler"),
    # EX-8 report cards: the student, their parent, the principal or a reports
    # admin, and the section's class teacher -- checked in the handler; logged.
    ("GET", "/api/v1/report-cards/students/{student_id}"): _p(ANY_USER, "the student, their linked parent, a reports admin or the class teacher; logged"),
    ("GET", "/api/v1/report-cards/students/{student_id}/pdf"): _p(ANY_USER, "as above; logged"),
    ("GET", "/api/v1/report-cards/sections/{section_id}/pdf"): _p(ANY_USER, "a reports admin or the section's class teacher; logged"),
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
    ("GET", "/v1/subtopics/{subtopic_id:path}/questions"): _p(API_KEY, "a subtopic id carries slashes, so the route matches a path (D52)"),
    ("GET", "/v1/facets"): _p(API_KEY),
    ("GET", "/v1/coverage"): _p(API_KEY, "aggregate answer-key coverage; no question text crosses it"),

    # ---- assessment/auth_routes.py ----
    ("POST", "/api/v1/auth/operator/school-invites"): _p(PUBLIC, "the operator key is checked in the route; rate-limited"),
    ("POST", "/api/v1/auth/register"): _p(PUBLIC, "needs an invite code or the bootstrap key (Task 301)"),
    ("POST", "/api/v1/auth/login"): _p(PUBLIC),
    ("POST", "/api/v1/auth/logout"): _p(PUBLIC, "deletes the bearer's own session if any; nothing to leak"),
    ("GET", "/api/v1/auth/me"): _p(ANY_USER),
    ("GET", "/api/v1/auth/users"): _p(ADMIN["users"]),
    ("POST", "/api/v1/auth/users/{user_id}/close"): _p(ADMIN["users"]),
    ("POST", "/api/v1/auth/users/{user_id}/reopen"): _p(ADMIN["users"]),
    ("POST", "/api/v1/auth/users/{user_id}/password-reset"): _p(
        ADMIN["users"], "a one-time password for a school account; never the principal's (QA S-05)"),
    ("POST", "/api/v1/auth/password"): _p(ANY_USER, "the caller's own password; needs the current one"),
    # NFR-1: never delegated -- an erasure cannot be undone (assessment/erasure.py).
    ("GET", "/api/v1/auth/users/{user_id}/erasure"): _p(PRINCIPAL, "what erasing a student or parent deletes"),
    ("POST", "/api/v1/auth/users/{user_id}/erasure"): _p(PRINCIPAL, "DPDP erasure on the school's request"),
    ("POST", "/api/v1/auth/invites"): _p(ADMIN["users"]),
    ("GET", "/api/v1/auth/invites"): _p(ADMIN["users"]),
    ("DELETE", "/api/v1/auth/invites/{code}"): _p(ADMIN["users"]),

    # ---- assessment/routes.py (assessment designer) ----
    ("POST", "/api/v1/blueprints/generate"): _p(STAFF),
    ("GET", "/api/v1/schools/{school_id}/templates"): _p(STAFF, "paper-design input"),
    ("GET", "/api/v1/schools/{school_id}/paper-templates"): _p(STAFF, "paper-design input"),
    ("POST", "/api/v1/questions/search"): _p(STAFF, "builds papers; a student browsing it is previewing the exam pool"),
    ("POST", "/api/v1/questions/optimize"): _p(STAFF),
    ("POST", "/api/v1/papers/generate"): _p(STAFF),
    ("POST", "/api/v1/papers/quick-generate"): _p(STAFF),
    ("POST", "/api/v1/papers/generate-from-ids"): _p(STAFF),
    ("GET", "/api/v1/paper-timing"): _p(ADMIN["reports"], "renewal metric; Task 201 sets the same"),
    ("GET", "/api/v1/papers/{paper_id}"): _p(STAFF, "an unreleased paper's stems"),
    ("POST", "/api/v1/papers/{paper_id}/export/{fmt}"): _p(STAFF, "includes the answer key"),
    ("GET", "/api/v1/papers/{paper_id}/file"): _p(STAFF, "includes the answer-key PDF"),
    ("POST", "/api/v1/assessments"): _p(STAFF),
    ("GET", "/api/v1/assessments"): _p(STAFF, "assessments carry the blueprint of an unreleased exam"),
    ("GET", "/api/v1/assessments/{assessment_id}"): _p(STAFF, "same as the list"),
    ("PUT", "/api/v1/assessments/{assessment_id}"): _p(STAFF),
    ("DELETE", "/api/v1/assessments/{assessment_id}"): _p(STAFF),
    ("PATCH", "/api/v1/assessments/{assessment_id}/status"): _p(STAFF),
    ("PATCH", "/api/v1/assessments/{assessment_id}/approve"): _p(ADMIN["exams"]),

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
    ("GET", "/api/v1/mail/status"): _p(ANY_USER, "configured yes/no per backend; signed in only, it describes the deployment (NFR-2)"),
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
    ("GET", "/api/v1/insights/school/{school_id}"): _p(ADMIN["reports"]),

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
    # Sections (M1.1): a section carries its class teacher, so reads are staff;
    # who is enrolled where is the principal's roster.
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/sections"): _p(STAFF, "carries each class teacher"),
    ("POST", "/api/v1/curriculum/grades/{grade_id}/sections"): _p(ADMIN["timetable"]),
    ("PATCH", "/api/v1/curriculum/sections/{section_id}"): _p(ADMIN["timetable"]),
    ("DELETE", "/api/v1/curriculum/sections/{section_id}"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/sections/{section_id}/students"): _p(ADMIN["users"], "the roster of one class"),
    # M1.2-M1.3 school model (curriculum/school_model_routes.py): reads staff, writes principal.
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/allocations"): _p(STAFF, "the allocation grid"),
    ("PUT", "/api/v1/curriculum/sections/{section_id}/allocations/{subject_id}"): _p(ADMIN["timetable"]),
    ("DELETE", "/api/v1/curriculum/sections/{section_id}/allocations/{subject_id}"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/teacher-load"): _p(ADMIN["timetable"], "teacher-level data is for the principal (ADM-5)"),
    ("GET", "/api/v1/curriculum/rooms"): _p(STAFF),
    ("POST", "/api/v1/curriculum/rooms"): _p(ADMIN["timetable"]),
    ("PATCH", "/api/v1/curriculum/rooms/{room_id}"): _p(ADMIN["timetable"]),
    ("DELETE", "/api/v1/curriculum/rooms/{room_id}"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/bell-schedules"): _p(ANY_USER, "a student day view needs the period times"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/bell-schedules"): _p(ADMIN["timetable"]),
    ("PUT", "/api/v1/curriculum/bell-schedules/{bell_schedule_id}"): _p(ADMIN["timetable"]),
    ("DELETE", "/api/v1/curriculum/bell-schedules/{bell_schedule_id}"): _p(ADMIN["timetable"]),
    ("PUT", "/api/v1/curriculum/sections/{section_id}/bell-schedule"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable"): _p(STAFF, "names teachers"),
    ("GET", "/api/v1/curriculum/sections/{section_id}/timetable"): _p(STAFF, "names teachers"),
    ("PUT", "/api/v1/curriculum/sections/{section_id}/timetable"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/my-timetable"): _p(ANY_USER, "the week of the caller"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable/solve"): _p(ADMIN["timetable"]),
    # N-3-6: publishing exactly the previewed week; N-3-20: elective groups,
    # combined classes, and splitting or merging a section.
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable/previews/{preview_id}/publish"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/teaching-groups"): _p(STAFF, "names teachers"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/teaching-groups"): _p(ADMIN["timetable"]),
    ("PUT", "/api/v1/curriculum/teaching-groups/{group_id}"): _p(ADMIN["timetable"]),
    ("DELETE", "/api/v1/curriculum/teaching-groups/{group_id}"): _p(ADMIN["timetable"]),
    ("POST", "/api/v1/curriculum/sections/{section_id}/split"): _p(ADMIN["timetable"], "scoped to the admin's classes"),
    ("POST", "/api/v1/curriculum/sections/{section_id}/merge"): _p(ADMIN["timetable"], "scoped to the admin's classes"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/teacher-unavailability"): _p(ADMIN["timetable"], "teacher-level data is for the principal (ADM-5)"),
    ("PUT", "/api/v1/curriculum/academic-years/{academic_year_id}/teacher-unavailability/{teacher_id}"): _p(ADMIN["timetable"]),
    # SCH-8: a room out of use (a lab's maintenance slot). Rooms are no one's
    # personal data, so staff may read them, as they read /rooms.
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/room-unavailability"): _p(STAFF),
    ("PUT", "/api/v1/curriculum/academic-years/{academic_year_id}/room-unavailability/{room_id}"): _p(ADMIN["timetable"]),
    # EX-5 / D41: the school's own name and logo, printed on its papers.
    ("GET", "/api/v1/curriculum/school-profile"): _p(ANY_USER, "the school's name; the apps show it"),
    ("PUT", "/api/v1/curriculum/school-profile"): _p(PRINCIPAL),
    ("GET", "/api/v1/curriculum/school-profile/logo"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/school-profile/logo"): _p(PRINCIPAL),
    ("DELETE", "/api/v1/curriculum/school-profile/logo"): _p(PRINCIPAL),
    # SCH-5..7 leave, substitution, compensation (curriculum/cover_routes.py).
    ("POST", "/api/v1/curriculum/leave-requests"): _p(STAFF, "a teacher applies for own leave; the principal for anyone"),
    ("GET", "/api/v1/curriculum/leave-requests"): _p(STAFF, "the principal sees all; a teacher their own"),
    ("POST", "/api/v1/curriculum/leave-requests/{leave_id}/decision"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/leave-requests/{leave_id}/cancel"): _p(STAFF, "the teacher or the principal"),
    # SCH-8: a temporary replacement teacher for a long leave.
    ("PUT", "/api/v1/curriculum/leave-requests/{leave_id}/replacement"): _p(ADMIN["leave"]),
    # N-3-17: a leave's supporting document; the handler admits only the
    # teacher on leave, whoever applied for them and the school's leave admin.
    ("POST", "/api/v1/curriculum/leave-requests/{leave_id}/document"): _p(STAFF, "the teacher on leave or the principal"),
    ("GET", "/api/v1/curriculum/leave-requests/{leave_id}/document"): _p(STAFF, "the teacher on leave or the leave admin; logged"),
    ("GET", "/api/v1/curriculum/substitutions"): _p(STAFF, "the principal sees all; a teacher their duties"),
    ("GET", "/api/v1/curriculum/substitutions/{sub_id}/candidates"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/substitutions/{sub_id}/assign"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/substitutions/{sub_id}/respond"): _p(STAFF, "only the proposed substitute"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/lost-periods"): _p(STAFF),
    ("GET", "/api/v1/curriculum/lost-periods/{lost_id}/compensation-options"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/lost-periods/{lost_id}/compensate"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/lost-periods/{lost_id}/waive"): _p(ADMIN["leave"]),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/closures"): _p(ADMIN["calendar"]),
    ("GET", "/api/v1/curriculum/day"): _p(ANY_USER, "a student sees only their own section"),
    ("GET", "/api/v1/curriculum/my-day"): _p(ANY_USER, "the day of the caller"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/cover-summary"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/curriculum/teacher-attendance"): _p(ADMIN["leave"]),
    ("PUT", "/api/v1/curriculum/teacher-attendance"): _p(ADMIN["leave"]),
    ("GET", "/api/v1/curriculum/my-attendance"): _p(STAFF, "the caller's own row of today's register (QA S-12)"),
    ("POST", "/api/v1/curriculum/my-attendance"): _p(STAFF, "the caller checks in; late after the first bell and the grace"),
    # Staff rules and leave balances (curriculum/staff_policy_routes.py).
    ("GET", "/api/v1/curriculum/staff-policy"): _p(STAFF, "the callers own schools leave and check-in rules"),
    ("PUT", "/api/v1/curriculum/staff-policy"): _p(ADMIN["leave"], "the principal or a leave admin; audited"),
    ("GET", "/api/v1/curriculum/my-leave-balance"): _p(STAFF, "the callers own leave this year"),
    ("GET", "/api/v1/curriculum/leave-balances"): _p(ADMIN["leave"], "every teacher of the callers school"),
    # M3 notifications (operations/routes.py): every route is the caller's own.
    ("GET", "/api/v1/notifications"): _p(ANY_USER, "the callers own inbox"),
    ("POST", "/api/v1/notifications/{notification_id}/read"): _p(ANY_USER, "only the recipient"),
    ("POST", "/api/v1/notifications/read-all"): _p(ANY_USER),
    ("GET", "/api/v1/notification-preferences"): _p(ANY_USER),
    ("PUT", "/api/v1/notification-preferences"): _p(ANY_USER),
    ("POST", "/api/v1/devices"): _p(ANY_USER, "the callers own push token"),
    ("DELETE", "/api/v1/devices/{token}"): _p(ANY_USER, "the callers own push token"),
    # NTF-2 scheduled automations (operations/routes.py, automations.py).
    ("POST", "/api/v1/automations/run"): _p(PUBLIC, "operator cron key checked in the handler; 404 unless configured"),
    ("GET", "/api/v1/automations/status"): _p(ADMIN["reports"]),
    # M1.4 delegated admin (operations/grant_routes.py): granting stays the principal's.
    ("GET", "/api/v1/admin/grants"): _p(PRINCIPAL),
    ("POST", "/api/v1/admin/grants"): _p(PRINCIPAL),
    ("DELETE", "/api/v1/admin/grants/{grant_id}"): _p(PRINCIPAL),
    ("GET", "/api/v1/my-grants"): _p(ANY_USER, "the callers own capabilities"),
    ("GET", "/api/v1/staff"): _p(STAFF, "names only, for pickers"),
    # ADM-2 dashboard, INT-2 calendar feed, API-2 homework sets, SA-2 photos.
    ("GET", "/api/v1/dashboard/summary"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/dashboard/term"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/dashboard/term/export"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/calendar-feed"): _p(ANY_USER, "the callers own feed; staff and students"),
    ("POST", "/api/v1/calendar-feed/reset"): _p(ANY_USER, "the callers own feed"),
    ("GET", "/api/v1/ics/{token}.ics"): _p(PUBLIC, "the unguessable feed token is the key; the owners own dates only"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/lesson-plans"): _p(PRINCIPAL, "approves proposed topics and plans every section (N-67-5)"),
    ("GET", "/api/v1/my-calendar"): _p(ANY_USER, "the callers own dated events; staff and students"),
    ("GET", "/api/v1/my-load"): _p(ANY_USER, "the callers own teaching load; staff only"),
    ("POST", "/v1/homework-sets"): _p(API_KEY, "questions:read"),
    ("POST", "/api/v1/my-homework/{homework_id}/photos"): _p(ANY_USER, "students only, own submission; needs consent"),
    ("GET", "/api/v1/my-homework/{homework_id}/photos"): _p(ANY_USER, "students only, own submission"),
    ("GET", "/api/v1/my-homework/{homework_id}/photos/{photo_id}"): _p(ANY_USER, "students only, own photo"),
    ("DELETE", "/api/v1/my-homework/{homework_id}/photos/{photo_id}"): _p(ANY_USER, "students only, own photo"),
    ("GET", "/api/v1/homework/{homework_id}/submissions/{student_id}/photos/{photo_id}"): _p(STAFF, "the teacher of the homework or the principal; logged"),
    # Saved progress and the setup level (operations/progress_routes.py).
    ("GET", "/api/v1/me/progress"): _p(ANY_USER, "the callers own saved progress"),
    ("PUT", "/api/v1/me/progress/{key}"): _p(ANY_USER, "the callers own saved progress"),
    ("GET", "/api/v1/setup-status"): _p(PRINCIPAL, "the principals own school, computed from its data"),
    # SA-4 self-practice (operations/practice_routes.py).
    ("POST", "/api/v1/my-practice"): _p(ANY_USER, "students only, their own sets"),
    ("GET", "/api/v1/my-practice"): _p(ANY_USER, "students only, their own sets"),
    ("GET", "/api/v1/my-practice/{practice_id}"): _p(ANY_USER, "students only, their own sets"),
    ("POST", "/api/v1/my-practice/{practice_id}/submit"): _p(ANY_USER, "students only, their own sets"),
    ("GET", "/api/v1/sections/{section_id}/progress"): _p(STAFF, "the sections teachers or a reports admin; logged"),
    ("POST", "/api/v1/auth/code/request"): _p(PUBLIC, "sign-in: same answer whether or not the email exists; rate-limited"),
    ("POST", "/api/v1/auth/code/verify"): _p(PUBLIC, "sign-in: hashed six-digit code, 10 minutes, 5 tries, 30 a day; rate-limited"),
    # Phase 1 pilot agreement (operations/pilot_agreement_routes.py).
    ("GET", "/api/v1/pilot-agreement"): _p(STAFF, "the agreement text and the callers own schools acceptance"),
    ("POST", "/api/v1/pilot-agreement/accept"): _p(PRINCIPAL, "the principal accepts for their own school; the texts hash recorded; audited"),
    ("GET", "/api/v1/qbank/coverage-map"): _p(STAFF),
    ("GET", "/api/v1/qbank/review-queue"): _p(ADMIN["qbank_review"]),
    ("POST", "/api/v1/qbank/review/{question_id}"): _p(ADMIN["qbank_review"]),
    ("GET", "/api/v1/auth/providers"): _p(PUBLIC, "sign-in page: which ways in are set up; the Google client id is public"),
    ("POST", "/api/v1/auth/google"): _p(PUBLIC, "sign-in: a Google ID token checked against Google's keys, for an existing account; rate-limited"),
    # M5 homework (operations/homework_routes.py).
    ("GET", "/api/v1/homework/suggestion"): _p(STAFF, "the teacher of the subject in the section, or the principal"),
    ("POST", "/api/v1/homework"): _p(STAFF, "a teacher only for sections they teach the subject in"),
    ("GET", "/api/v1/homework"): _p(STAFF, "the principal sees all; a teacher what they set or teach"),
    ("GET", "/api/v1/homework/{homework_id}"): _p(STAFF, "the teacher of the subject in its sections, or the principal"),
    ("POST", "/api/v1/homework/{homework_id}/publish"): _p(STAFF, "the teacher of the subject in its sections, or the principal"),
    ("POST", "/api/v1/homework/{homework_id}/close"): _p(STAFF, "the teacher of the subject in its sections, or the principal"),
    ("GET", "/api/v1/homework/{homework_id}/submissions"): _p(STAFF, "the teacher of the subject in its sections, or the principal"),
    ("POST", "/api/v1/homework/{homework_id}/submissions/{student_id}/grade"): _p(STAFF, "the teacher of the subject in its sections, or the principal; needs consent"),
    ("POST", "/api/v1/homework/{homework_id}/remind"): _p(STAFF, "the teacher of the subject in its sections, or the principal"),
    ("GET", "/api/v1/my-homework"): _p(ANY_USER, "handler admits students only, their own section"),
    ("GET", "/api/v1/my-homework/{homework_id}"): _p(ANY_USER, "handler admits students only, their own section; no answers before marking"),
    ("POST", "/api/v1/my-homework/{homework_id}/submit"): _p(ANY_USER, "handler admits students only, their own section; needs consent"),
    # SA-3 learning progress (operations/learning_routes.py).
    ("GET", "/api/v1/my-learning"): _p(ANY_USER, "handler admits students only, their own progress"),
    ("GET", "/api/v1/students/{student_id}/learning"): _p(SELF_OR_STAFF),
    # M1.5 parents (operations/guardian_routes.py).
    ("POST", "/api/v1/students/{student_id}/guardian-invites"): _p(ADMIN["users"]),
    ("GET", "/api/v1/students/{student_id}/guardians"): _p(STAFF),
    ("POST", "/api/v1/guardianships"): _p(ADMIN["users"]),
    ("DELETE", "/api/v1/guardianships/{parent_id}/{student_id}"): _p(ADMIN["users"]),
    ("GET", "/api/v1/my-children"): _p(ANY_USER, "handler admits parent accounts only"),
    ("GET", "/api/v1/children/{student_id}/timetable"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("GET", "/api/v1/children/{student_id}/day"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("GET", "/api/v1/children/{student_id}/calendar"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("GET", "/api/v1/children/{student_id}/homework"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("GET", "/api/v1/children/{student_id}/homework/{homework_id}"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("POST", "/api/v1/children/{student_id}/homework/{homework_id}/submit"): _p(ANY_USER, "a linked guardian of a class 1-5 child only (SA-7); audited"),
    ("GET", "/api/v1/children/{student_id}/learning"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("GET", "/api/v1/children/{student_id}/consent"): _p(ANY_USER, "handler admits the childs linked guardians only"),
    ("POST", "/api/v1/children/{student_id}/consent"): _p(ANY_USER, "the linked guardians own consent"),
    ("DELETE", "/api/v1/children/{student_id}/consent"): _p(ANY_USER, "the linked guardians own consent"),
    ("POST", "/api/v1/admin/import/{kind}"): _p(ADMIN["users"]),
    ("GET", "/api/v1/admin/import/{kind}/template"): _p(ADMIN["users"]),
    ("GET", "/api/v1/admin/export/{kind}"): _p(ADMIN["users"]),
    # D38: issuing keys for the question-bank API (assessment/api_key_routes.py).
    ("POST", "/api/v1/api-keys"): _p(PRINCIPAL, "own school keys only"),
    ("GET", "/api/v1/api-keys"): _p(PRINCIPAL, "own school keys only"),
    ("DELETE", "/api/v1/api-keys/{key_id}"): _p(PRINCIPAL, "own school keys only"),
    ("GET", "/api/v1/api-keys/{key_id}/usage"): _p(PRINCIPAL, "own school keys only"),
    # API-6: a key's webhooks (assessment/api_key_routes.py, webhooks.py).
    ("POST", "/api/v1/api-keys/{key_id}/webhooks"): _p(PRINCIPAL, "own school keys only; public HTTPS only; secret shown once"),
    ("GET", "/api/v1/api-keys/{key_id}/webhooks"): _p(PRINCIPAL, "own school keys only; never the secret"),
    ("DELETE", "/api/v1/api-keys/{key_id}/webhooks/{webhook_id}"): _p(PRINCIPAL, "own school keys only"),
    ("GET", "/api/v1/api-keys/{key_id}/webhooks/{webhook_id}/deliveries"): _p(PRINCIPAL, "own school keys only; ids and counts, no question text"),
    # EX-6 paper review (assessment/paper_review_routes.py).
    ("POST", "/api/v1/assessments/{assessment_id}/review-request"): _p(STAFF, "the author or the principal"),
    ("POST", "/api/v1/assessments/{assessment_id}/review-decision"): _p(STAFF, "the named reviewer or the principal"),
    ("GET", "/api/v1/assessments/{assessment_id}/reviews"): _p(STAFF),
    ("GET", "/api/v1/paper-reviews"): _p(STAFF, "the callers reviews; the principal the schools"),
    # EX-8 marks entry and analysis (assessment/marks_routes.py).
    ("PUT", "/api/v1/papers/{paper_id}/marks"): _p(STAFF, "own school paper; needs consent"),
    ("GET", "/api/v1/papers/{paper_id}/marks"): _p(STAFF, "own school paper; logged as a read"),
    ("GET", "/api/v1/papers/{paper_id}/analysis"): _p(STAFF),
    # EX-7 exam planning (operations/exam_routes.py).
    ("POST", "/api/v1/exams"): _p(ADMIN["exams"]),
    ("GET", "/api/v1/exams"): _p(ADMIN["exams"]),
    ("GET", "/api/v1/exams/{exam_id}"): _p(ADMIN["exams"]),
    ("PUT", "/api/v1/exams/{exam_id}/datesheet"): _p(ADMIN["exams"]),
    ("POST", "/api/v1/exams/{exam_id}/invigilation/auto"): _p(ADMIN["exams"]),
    ("PUT", "/api/v1/exams/{exam_id}/invigilation"): _p(ADMIN["exams"]),
    ("POST", "/api/v1/exams/{exam_id}/publish"): _p(ADMIN["exams"]),
    ("GET", "/api/v1/my-exams"): _p(ANY_USER, "a student their class papers; a teacher their duties"),
    ("GET", "/api/v1/curriculum/subjects/{subject_id}/books"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/books/{book_id}/units"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/books/{book_id}/chapters"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("GET", "/api/v1/curriculum/chapters/{chapter_id}/topics"): _p(ANY_USER, "school-scoped since calendar Task 101"),
    ("POST", "/api/v1/curriculum/books/{book_id}/toc/ingest"): _p(ADMIN["qbank_review"]),
    ("POST", "/api/v1/curriculum/seed/cbse10"): _p(PRINCIPAL),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/extract"): _p(ADMIN["qbank_review"]),
    ("GET", "/api/v1/curriculum/extraction-runs/{run_id}"): _p(ANY_USER, "school-scoped; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/extraction-runs/{run_id}/approve"): _p(ADMIN["qbank_review"]),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/topics"): _p(ADMIN["qbank_review"]),
    ("POST", "/api/v1/curriculum/topics/{topic_id}/subtopics"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/topics/{topic_id}"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/subtopics/{subtopic_id}"): _p(ADMIN["qbank_review"]),
    ("DELETE", "/api/v1/curriculum/subtopics/{subtopic_id}"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/units/{unit_id}/sequence"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/chapters/{chapter_id}/sequence"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/topics/{topic_id}/sequence"): _p(ADMIN["qbank_review"]),
    ("PATCH", "/api/v1/curriculum/subtopics/{subtopic_id}/sequence"): _p(ADMIN["qbank_review"]),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/questions/tag"): _p(ANY_USER, "a write any role can make; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/questions/by-subtopics"): _p(ANY_USER, "returns question ids only; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(ADMIN["calendar"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(ANY_USER, "the school's own calendar"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays"): _p(ADMIN["calendar"]),
    ("PUT", "/api/v1/curriculum/academic-years/{academic_year_id}/calendar"): _p(ADMIN["calendar"], "correct the calendar (calendar Task 103)"),
    ("DELETE", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays/{holiday_id}"): _p(ADMIN["calendar"], "remove a mistyped holiday (calendar Task 103)"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/holidays"): _p(ANY_USER, "the school's own holidays"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/period-configuration"): _p(ADMIN["timetable"]),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/subject-period-allocations"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/subject-period-allocations"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable-slots"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/timetable-slots"): _p(ANY_USER),
    ("DELETE", "/api/v1/curriculum/timetable-slots/{slot_id}"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/working-days"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/books/{book_id}/teaching-time-estimates"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/subtopics/{subtopic_id}/teaching-time-estimate"): _p(ANY_USER),
    ("POST", "/api/v1/curriculum/books/{book_id}/schedule"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/books/{book_id}/schedule"): _p(ANY_USER, "includes teacher notes; should be staff (reported, not edited)"),
    ("GET", "/api/v1/curriculum/schedule"): _p(ANY_USER, "includes teacher notes; should be staff (reported, not edited)"),
    ("POST", "/api/v1/curriculum/teacher-assignments"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/my-schedule"): _p(ANY_USER, "only the caller's own assigned books; empty for a student"),
    ("PATCH", "/api/v1/curriculum/scheduled-lessons/{lesson_id}"): _p(ANY_USER, "handler admits only the assigned teacher or a principal"),
    ("POST", "/api/v1/curriculum/scheduled-lessons/{lesson_id}/reschedule"): _p(ADMIN["timetable"]),
    ("POST", "/api/v1/curriculum/books/{book_id}/schedule/push"): _p(ADMIN["timetable"]),
    ("GET", "/api/v1/curriculum/scheduled-lessons/{lesson_id}/history"): _p(ANY_USER, "handler admits only the assigned teacher or a principal"),
    ("POST", "/api/v1/curriculum/student-enrollments"): _p(ADMIN["users"]),
    ("GET", "/api/v1/curriculum/my-class-schedule"): _p(ANY_USER, "handler admits students only, their own grade"),
    ("GET", "/api/v1/curriculum/my-progress"): _p(ANY_USER, "handler admits students only, their own grade"),
    ("GET", "/api/v1/curriculum/reporting/coverage"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/curriculum/reporting/delayed-topics"): _p(ADMIN["reports"]),
    ("GET", "/api/v1/curriculum/export"): _p(ADMIN["reports"]),

    # ---- curriculum/routes.py: book editions, seeding, terms ----
    # feat/product-calendar branched before this table existed, so its twelve
    # routes arrived ungoverned. Each level below was read off the route's own
    # `Depends(...)` in curriculum/routes.py at merge time, not inferred from
    # the path: `require_principal` -> PRINCIPAL, `get_current_user` ->
    # ANY_USER. test_role_gates.py re-checks every one against the real
    # dependency chain, so a row that flatters a route fails.
    ("POST", "/api/v1/curriculum/subjects/{subject_id}/books"): _p(
        PRINCIPAL, "adds a book edition to a subject; Depends(require_principal)"),
    ("GET", "/api/v1/curriculum/subjects/{subject_id}/selected-book"): _p(
        ANY_USER, "the subject's chosen edition; _require_school_owns_subject scopes it to the caller's school"),
    ("PUT", "/api/v1/curriculum/subjects/{subject_id}/selected-book"): _p(
        PRINCIPAL, "choosing the year's edition is the school's decision; Depends(require_principal)"),
    ("POST", "/api/v1/curriculum/seed/cbse"): _p(
        PRINCIPAL, "seeds any class 6-12; same posture as /seed/cbse10, which delegates here"),
    # The setup wizard: a new school's year, classes, calendar, terms and bell
    # in one call (curriculum/school_setup.py), for the caller's own school.
    ("POST", "/api/v1/curriculum/school-setup"): _p(
        PRINCIPAL, "sets up the caller's own school; idempotent; Depends(require_principal)"),
    ("GET", "/api/v1/curriculum/chapters/by-slug/{subject}/{grade}/{slug}/topics"): _p(
        ANY_USER, "read-only; resolves a syllabus slug to the caller's own school's chapter, so a teacher's subtopic picker can find one"),
    ("GET", "/api/v1/curriculum/chapters/{chapter_id}/proposed-topics"): _p(
        ANY_USER, "pending, unapproved topics of the caller's own school (_require_school_owns_chapter)"),
    ("POST", "/api/v1/curriculum/chapters/{chapter_id}/topics/approve-all"): _p(
        PRINCIPAL, "approval is the principal's act; Depends(require_principal)"),
    ("POST", "/api/v1/curriculum/academic-years/{academic_year_id}/terms"): _p(
        PRINCIPAL, "defines a term of the school's year; Depends(require_principal)"),
    ("GET", "/api/v1/curriculum/academic-years/{academic_year_id}/terms"): _p(
        ANY_USER, "the school's own terms; _require_school_owns_academic_year scopes it"),
    ("GET", "/api/v1/curriculum/terms/current"): _p(
        ANY_USER, "which term today falls in, for the caller's own school"),
    ("PUT", "/api/v1/curriculum/terms/{term_id}"): _p(
        PRINCIPAL, "correct a term's dates; Depends(require_principal)"),
    ("DELETE", "/api/v1/curriculum/terms/{term_id}"): _p(
        PRINCIPAL, "remove a term; Depends(require_principal)"),
    ("PUT", "/api/v1/curriculum/terms/{term_id}/baseline"): _p(
        PRINCIPAL, "the term's manual paper baseline, which time saved is measured "
        "against; Depends(require_principal)"),

    # ---- ROLE-4 multi-campus (operations/owner_routes.py) ----
    # A school's own principal links an owner. An owner acting as the school's
    # principal is refused here in the handler: it must not mint or withdraw
    # owner access, or a revoked owner could keep a second account linked.
    ("POST", "/api/v1/owner-links"): _p(PRINCIPAL, "own school only; the school's own principal account; code shown once"),
    ("GET", "/api/v1/owner-links"): _p(PRINCIPAL, "own school only; who holds owner access"),
    ("DELETE", "/api/v1/owner-links/{link_id}"): _p(PRINCIPAL, "own school only; ends the owner's access at once"),
    ("POST", "/api/v1/owner/register"): _p(PUBLIC, "needs an owner link code a principal issued; rate-limited"),
    ("POST", "/api/v1/owner/links/redeem"): _p(OWNER, "links one more school with its principal's code"),
    ("GET", "/api/v1/owner/schools"): _p(OWNER, "the owner's linked schools, names only"),
    ("POST", "/api/v1/owner/active-school"): _p(OWNER, "a linked school only; the session then acts as its principal"),
    ("DELETE", "/api/v1/owner/active-school"): _p(OWNER, "back to no school"),
    ("GET", "/api/v1/owner/overview"): _p(OWNER, "per-school aggregates only: no student, teacher or record named"),
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


# The only routes a parent account may call (auth_routes.get_current_user
# refuses every other one with 403). Their own inbox, settings and devices,
# and their linked children's pages, each of which checks the link itself.
PARENT_PATHS = frozenset({
    "/api/v1/auth/me",
    # Their own password (the school hands them a temporary one and Settings offers the change) and their
    # own Guide progress: both were refused with "can see only their children's pages" (audit 2026-10-04).
    "/api/v1/auth/password",
    "/api/v1/me/progress",
    "/api/v1/me/progress/{key}",
    # EX-8: a linked parent reads their child's report card (checked in the handler).
    "/api/v1/report-cards/students/{student_id}",
    "/api/v1/report-cards/students/{student_id}/pdf",
    "/api/v1/notifications",
    "/api/v1/notifications/{notification_id}/read",
    "/api/v1/notifications/read-all",
    "/api/v1/notification-preferences",
    "/api/v1/devices",
    "/api/v1/devices/{token}",
    # INT-2: a parent's own number and SMS/WhatsApp consent.
    "/api/v1/me/contact",
    "/api/v1/my-children",
    "/api/v1/children/{student_id}/timetable",
    "/api/v1/children/{student_id}/day",
    "/api/v1/children/{student_id}/calendar",
    "/api/v1/children/{student_id}/homework",
    "/api/v1/children/{student_id}/homework/{homework_id}",
    "/api/v1/children/{student_id}/homework/{homework_id}/submit",
    "/api/v1/children/{student_id}/learning",
    "/api/v1/children/{student_id}/consent",
})


# The only get_current_user routes an owner account may call before it has
# chosen one of its schools (auth_routes.get_current_user refuses every other
# with 403). Its own routes take auth_routes.require_owner instead and need no
# entry here. Once a school is chosen the owner is that school's principal.
OWNER_PATHS = frozenset({
    "/api/v1/auth/me",
})
