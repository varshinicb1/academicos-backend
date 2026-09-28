"""CurriculumStore: SQLite persistence for the operational curriculum/
schedule hierarchy -- see docs/ACADEMIC_DATA_MODEL.md.

Durability, 2026-09-15: this was the last store the AGENTS.md inventory
listed as "Deferred (Target for Path B)" -- the deferral note used to say
it was safe because "nothing writes real school data into it yet." That's
no longer true: the School Admin Console (Master Calendar, Sequence
Reorder, Reschedule PUSH/ADJUST, Calendar Setup) now writes real
admin-entered operational data here, so losing it on every Render
redeploy/restart is a real problem, not a hypothetical one.

Unlike the flat single-object stores (assessment/store.py,
assessment/paper_store.py, ...), this store owns 18 relational tables with
real FK relationships (`PRAGMA foreign_keys=ON` above) -- wrapping each
table individually in SupabaseTable the way those stores do would mean 18
new REST call sites and would lose that FK integrity across the two
storage layers. Instead this store snapshots/restores the *whole SQLite
file* as one blob via SupabaseStorage (the same class already used for
scan-session photos/PDFs, just a different bucket):
  - `_commit()` (replaces every direct `self.conn.commit()` call in this
    file, ~35 call sites) checkpoints the WAL into the main file and
    uploads it, debounced to once per `_SNAPSHOT_DEBOUNCE_SECONDS` so a
    burst of edits doesn't re-upload the whole file on every single write.
  - `__init__` downloads the last snapshot and writes it to `db_path`
    *before* opening the connection, but only if `db_path` doesn't already
    exist locally -- a fresh container with no prior local state restores
    the last known-good state instead of starting empty; a container that
    already has local data (e.g. mid-request restart within the same
    disk lifetime) never overwrites it with a possibly-older remote copy.
  - Requires a "curriculum-snapshots" bucket to exist in the Supabase
    project (see docs/deployment.md) -- `.enabled` is False without
    SUPABASE_KNOWLEDGE_URL/SUPABASE_KNOWLEDGE_ANON_KEY set, in which case
    this whole mechanism is a no-op and behavior is unchanged from before.

Fixed 2026-09-15 (was the "known gap" below): a Render stress test
caught this live -- concurrent curriculum reads returned rows whose fields
all read back as None (then 500'd in pydantic validation). Every statement
now funnels through the _exec/_fetchone/_fetchall helpers, which serialize
on an RLock and materialize rows to plain dicts inside it; _commit's own
commit() is serialized too. The other SQLite stores in this codebase
(UserStore, AssessmentStore, ...) still share one unlocked connection each
-- same latent hazard, still open, still flagged rather than silently left
undocumented.

IDs are short opaque strings (`{prefix}_{uuid4 hex[:12]}`), matching
users.py's `user_{...}` convention -- NOT the graph's descriptive
`board:grade:subject:...` style, because these rows are meant to be
edited/reordered by an admin (§13), and a human-readable-but-positional id
would break the moment something gets reordered. `canonical_id` (assigned
once, immutable) is the separate, stable field other systems (qmap.py)
key against -- see models.py's docstrings on Unit/Chapter/Subtopic.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

from ..storage.snapshot_sync import SnapshotSync

from .models import (
    AcademicYear,
    Board,
    Book,
    BookSelection,
    Calendar,
    Chapter,
    CHAPTER_SLUG_REASONS,
    ChapterSlugMatch,
    CurriculumExtractionProposal,
    CurriculumExtractionRun,
    Grade,
    Holiday,
    OtherEdition,
    PeriodConfiguration,
    SubjectPeriodAllocation,
    SubjectTimetableSlot,
    QuestionSubtopicLink,
    RECORDED_STATUSES,
    STATUS_VALUES,
    ScheduledLesson,
    DEFAULT_SECTION_NAME,
    SECTION_NAME_MAX,
    Section,
    StudentEnrollment,
    Subject,
    Subtopic,
    TeacherAssignment,
    TeachingTimeEstimate,
    Term,
    Topic,
    Unit,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS boards (
  id           TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  code         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS academic_years (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  label        TEXT NOT NULL,
  start_date   TEXT NOT NULL,
  end_date     TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'draft'
);
CREATE INDEX IF NOT EXISTS idx_years_school ON academic_years(school_id);

-- Terms (Task 106): a named date range inside one academic year. Added after
-- school_1's real database existed; CREATE TABLE IF NOT EXISTS is enough
-- because it is a new table, not a new column. The no-overlap / inside-the-
-- year rules live in CurriculumStore._check_term (SQLite cannot express
-- them); the name rule is also a constraint, case-insensitive like the check.
CREATE TABLE IF NOT EXISTS terms (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  name         TEXT NOT NULL COLLATE NOCASE,
  start_date   TEXT NOT NULL,
  end_date     TEXT NOT NULL,
  manual_baseline_minutes REAL,
  UNIQUE(academic_year_id, name)
);
CREATE INDEX IF NOT EXISTS idx_terms_year ON terms(academic_year_id);
CREATE INDEX IF NOT EXISTS idx_terms_school_dates ON terms(school_id, start_date);

CREATE TABLE IF NOT EXISTS grades (
  id           TEXT PRIMARY KEY,
  academic_year_id TEXT NOT NULL,
  number       INTEGER NOT NULL,
  section      TEXT
);
CREATE INDEX IF NOT EXISTS idx_grades_year ON grades(academic_year_id);

-- Sections (M1.1, docs/plans/m1-school-data-model.md): the class of a grade a
-- student is enrolled in. Until 2026-09-28 a section was only the optional
-- grades.section label. Every grade has at least one: create_grade makes the
-- first, and _migrate gives an existing grade without one a section named
-- after its old label, or "A". New table, so CREATE TABLE IF NOT EXISTS is
-- enough; the name rule is also a constraint, case-insensitive like the check.
CREATE TABLE IF NOT EXISTS sections (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  grade_id     TEXT NOT NULL,
  name         TEXT NOT NULL COLLATE NOCASE,
  class_teacher_id TEXT,
  created_at   TEXT NOT NULL,
  UNIQUE(grade_id, name)
);
CREATE INDEX IF NOT EXISTS idx_sections_year ON sections(academic_year_id);
CREATE INDEX IF NOT EXISTS idx_sections_grade ON sections(grade_id);

CREATE TABLE IF NOT EXISTS subjects (
  id           TEXT PRIMARY KEY,
  grade_id     TEXT NOT NULL,
  name         TEXT NOT NULL,
  code         TEXT
);
CREATE INDEX IF NOT EXISTS idx_subjects_grade ON subjects(grade_id);

CREATE TABLE IF NOT EXISTS books (
  id           TEXT PRIMARY KEY,
  subject_id   TEXT NOT NULL,
  board_id     TEXT NOT NULL,
  title        TEXT NOT NULL,
  publisher    TEXT,
  source_doc_ids TEXT,   -- json list
  status       TEXT NOT NULL DEFAULT 'selected'
);
CREATE INDEX IF NOT EXISTS idx_books_subject ON books(subject_id);

-- The school's chosen edition (Task 107, PRD 12.7: "one per subject per
-- year, chosen by the school"). The primary key IS the rule: a second row for
-- the same subject and year is refused by SQLite itself, not only by a route.
-- A subject already belongs to one year (subject -> grade -> year), so the
-- year column is redundant with subject_id today; it is kept so the rule
-- reads as PRD 12.7 states it. New table, so CREATE TABLE IF NOT EXISTS
-- is enough for school_1's existing database. No row for a subject with
-- exactly one book means that book (selected_book_for_subject).
CREATE TABLE IF NOT EXISTS book_selections (
  academic_year_id TEXT NOT NULL,
  subject_id   TEXT NOT NULL,
  book_id      TEXT NOT NULL,
  selected_at  TEXT NOT NULL,
  PRIMARY KEY (academic_year_id, subject_id)
);

CREATE TABLE IF NOT EXISTS units (
  id           TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL UNIQUE,
  book_id      TEXT NOT NULL,
  unit_no      TEXT NOT NULL,
  name         TEXT NOT NULL,
  marks        INTEGER,
  seq          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_units_book ON units(book_id);

CREATE TABLE IF NOT EXISTS chapters (
  id           TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL UNIQUE,
  unit_id      TEXT NOT NULL,
  name         TEXT NOT NULL,
  seq          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chapters_unit ON chapters(unit_id);

CREATE TABLE IF NOT EXISTS topics (
  id           TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL UNIQUE,
  chapter_id   TEXT NOT NULL,
  name         TEXT NOT NULL,
  seq          INTEGER NOT NULL DEFAULT 0,
  description  TEXT,
  source_type  TEXT NOT NULL DEFAULT 'manual',
  source_reference TEXT,
  approved_by  TEXT,
  approved_at  TEXT,
  model_used   TEXT,
  generation_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_topics_chapter ON topics(chapter_id);

CREATE TABLE IF NOT EXISTS subtopics (
  id           TEXT PRIMARY KEY,
  canonical_id TEXT NOT NULL UNIQUE,
  topic_id     TEXT NOT NULL,
  name         TEXT NOT NULL,
  seq          INTEGER NOT NULL DEFAULT 0,
  description  TEXT,
  source_type  TEXT NOT NULL DEFAULT 'manual',
  source_reference TEXT,
  approved_by  TEXT,
  approved_at  TEXT,
  model_used   TEXT,
  generation_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_subtopics_topic ON subtopics(topic_id);

CREATE TABLE IF NOT EXISTS curriculum_extraction_runs (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  book_id      TEXT NOT NULL,
  chapter_id   TEXT NOT NULL,
  source_hash  TEXT,
  model        TEXT,
  prompt_version TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_extruns_chapter ON curriculum_extraction_runs(chapter_id);

CREATE TABLE IF NOT EXISTS curriculum_extraction_proposals (
  id           TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL,
  entity_type  TEXT NOT NULL,
  proposed_name TEXT NOT NULL,
  proposed_description TEXT,
  proposed_parent TEXT,
  sequence     INTEGER NOT NULL DEFAULT 0,
  confidence   REAL NOT NULL DEFAULT 0.5,
  status       TEXT NOT NULL DEFAULT 'pending',
  edited_name  TEXT,
  materialized_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_extprops_run ON curriculum_extraction_proposals(run_id);

CREATE TABLE IF NOT EXISTS question_subtopic_links (
  id           TEXT PRIMARY KEY,
  question_id  TEXT NOT NULL,
  subtopic_id  TEXT NOT NULL,
  method       TEXT NOT NULL DEFAULT 'lexical',
  confidence   REAL NOT NULL DEFAULT 0.5,
  UNIQUE(question_id, subtopic_id)
);
CREATE INDEX IF NOT EXISTS idx_qsl_question ON question_subtopic_links(question_id);
CREATE INDEX IF NOT EXISTS idx_qsl_subtopic ON question_subtopic_links(subtopic_id);

CREATE TABLE IF NOT EXISTS period_configurations (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  period_minutes INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_periodcfg_year ON period_configurations(academic_year_id);

CREATE TABLE IF NOT EXISTS subject_period_allocations (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  subject      TEXT NOT NULL,
  periods_per_week INTEGER NOT NULL,
  UNIQUE(academic_year_id, subject)
);
CREATE INDEX IF NOT EXISTS idx_spa_year_subject ON subject_period_allocations(academic_year_id, subject);

CREATE TABLE IF NOT EXISTS subject_timetable_slots (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  subject      TEXT NOT NULL,
  day_of_week  INTEGER NOT NULL,
  period_number INTEGER NOT NULL,
  UNIQUE(academic_year_id, subject, day_of_week, period_number)
);
CREATE INDEX IF NOT EXISTS idx_sts_year_subject ON subject_timetable_slots(academic_year_id, subject);

CREATE TABLE IF NOT EXISTS calendars (
  id           TEXT PRIMARY KEY,
  academic_year_id TEXT NOT NULL UNIQUE,
  weekly_off_days TEXT NOT NULL,   -- json list
  alternate_saturday_rule TEXT NOT NULL DEFAULT 'none'
);

CREATE TABLE IF NOT EXISTS holidays (
  id           TEXT PRIMARY KEY,
  calendar_id  TEXT NOT NULL,
  date         TEXT NOT NULL,
  label        TEXT NOT NULL,
  kind         TEXT NOT NULL DEFAULT 'holiday',
  end_date     TEXT
);
CREATE INDEX IF NOT EXISTS idx_holidays_calendar ON holidays(calendar_id);

CREATE TABLE IF NOT EXISTS teaching_time_estimates (
  id           TEXT PRIMARY KEY,
  subtopic_id  TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  estimated_minutes INTEGER NOT NULL,
  estimated_periods INTEGER,
  method       TEXT NOT NULL DEFAULT 'admin_override',
  approved_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_tte_subtopic ON teaching_time_estimates(subtopic_id);

CREATE TABLE IF NOT EXISTS scheduled_lessons (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  academic_year_id TEXT NOT NULL,
  book_id      TEXT NOT NULL,
  subtopic_id  TEXT NOT NULL,
  date         TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'scheduled',
  note         TEXT,
  completed_by TEXT,
  completed_at TEXT,
  created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sl_book_year ON scheduled_lessons(academic_year_id, book_id);
CREATE INDEX IF NOT EXISTS idx_sl_subtopic ON scheduled_lessons(subtopic_id, academic_year_id);
CREATE INDEX IF NOT EXISTS idx_sl_school_date ON scheduled_lessons(school_id, date);

-- The cadence a book's schedule was placed with (Task 102 review): the
-- schedule route takes periods_per_week explicitly and may disagree with the
-- stored SubjectPeriodAllocation, so PUSH used to fall back to the allocation
-- and reschedule at a different cadence than the schedule was made with.
CREATE TABLE IF NOT EXISTS book_schedule_cadences (
  academic_year_id TEXT NOT NULL,
  book_id      TEXT NOT NULL,
  periods_per_week INTEGER NOT NULL,
  PRIMARY KEY (academic_year_id, book_id)
);

CREATE TABLE IF NOT EXISTS teacher_assignments (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  teacher_id   TEXT NOT NULL,
  book_id      TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  UNIQUE(teacher_id, book_id)
);
CREATE INDEX IF NOT EXISTS idx_ta_teacher ON teacher_assignments(teacher_id);

CREATE TABLE IF NOT EXISTS student_enrollments (
  id           TEXT PRIMARY KEY,
  school_id    TEXT NOT NULL,
  student_id   TEXT NOT NULL UNIQUE,
  grade_id     TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  section_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_se_grade ON student_enrollments(grade_id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


from .cover import COVER_SCHEMA, CoverMixin
from .school_model import SCHOOL_MODEL_SCHEMA, SchoolModelMixin


class SectionInUse(Exception):
    """A section cannot be removed: students are enrolled in it, or it is
    its grade's last. The route's 409."""


class CurriculumStore(SchoolModelMixin, CoverMixin):
    _SNAPSHOT_KEY = "curriculum.sqlite"
    _SNAPSHOT_DEBOUNCE_SECONDS = 30.0

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # RLock, not Lock: helpers below take it per-call, and _commit/
        # the snapshot sync nest a raw connection use inside their own
        # acquisition -- a plain Lock would deadlock those paths.
        self._conn_lock = threading.RLock()
        # Restores the last snapshot into db_path before the connect below;
        # upload, conflicts and flush live there too (storage/snapshot_sync.py).
        self._snapshots = SnapshotSync("curriculum-snapshots", self._SNAPSHOT_KEY, db_path,
                                       self._conn_lock, debounce_seconds=self._SNAPSHOT_DEBOUNCE_SECONDS,
                                       on_reload=self._bring_up_to_date)

        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._exec("PRAGMA journal_mode=WAL")
        self._exec("PRAGMA busy_timeout=60000")
        self._exec("PRAGMA foreign_keys=ON")
        self._bring_up_to_date()
        # The migration's rows (M1.1's first sections) are the same on every
        # instance that starts from this snapshot, so starting does not upload
        # them: that would turn every rollout into a snapshot conflict with
        # the instance being replaced (SnapshotSync.commit_derived).
        self._snapshots.commit_derived(self.conn)

    def _bring_up_to_date(self) -> None:
        """This release's schema and migrations over whatever the file holds:
        on start, and after a snapshot conflict reloads one an older release
        wrote. Executes only; the caller commits."""
        with self._conn_lock:
            self.conn.executescript(SCHEMA + SCHOOL_MODEL_SCHEMA + COVER_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """CREATE TABLE IF NOT EXISTS never adds a column to a table that
        already exists on disk -- unlike scheduled_lessons' note/completed_by
        columns (added when zero real rows existed yet, so no migration was
        needed), school_1 already has real seeded holiday rows by the time
        `end_date` was added, so a real ALTER TABLE is required here."""
        cols = {r["name"] for r in self._fetchall("PRAGMA table_info(holidays)")}
        if "end_date" not in cols:
            self._exec("ALTER TABLE holidays ADD COLUMN end_date TEXT")
        # Terms went live before a term carried its own paper baseline, so a
        # school's existing terms table needs the column added, not assumed.
        term_cols = {r["name"] for r in self._fetchall("PRAGMA table_info(terms)")}
        if term_cols and "manual_baseline_minutes" not in term_cols:
            self._exec("ALTER TABLE terms ADD COLUMN manual_baseline_minutes REAL")
        # Sections (M1.1). The index lives here, not in SCHEMA: SCHEMA runs
        # before this, while an existing enrollments table has no such column.
        enrol_cols = {r["name"] for r in self._fetchall("PRAGMA table_info(student_enrollments)")}
        if "section_id" not in enrol_cols:
            self._exec("ALTER TABLE student_enrollments ADD COLUMN section_id TEXT")
        self._exec("CREATE INDEX IF NOT EXISTS idx_se_section ON student_enrollments(section_id)")
        self._migrate_sections()
        # M1.3: a section may keep its own bell (a junior wing's shorter day);
        # None means the year's default schedule.
        section_cols = {r["name"] for r in self._fetchall("PRAGMA table_info(sections)")}
        if "bell_schedule_id" not in section_cols:
            self._exec("ALTER TABLE sections ADD COLUMN bell_schedule_id TEXT")
        # SCH-4: a lesson plan is per (book, section). A NULL section is the
        # school-wide plan every lesson made before this change belongs to.
        lesson_cols = {r["name"] for r in self._fetchall("PRAGMA table_info(scheduled_lessons)")}
        if "section_id" not in lesson_cols:
            self._exec("ALTER TABLE scheduled_lessons ADD COLUMN section_id TEXT")
        self._exec("CREATE INDEX IF NOT EXISTS idx_sl_plan "
                   "ON scheduled_lessons(academic_year_id, book_id, section_id)")
        # SCH-3: a locked period is kept by the timetable solver.
        entry_cols = {r["name"] for r in self._fetchall("PRAGMA table_info(timetable_entries)")}
        if "locked" not in entry_cols:
            self._exec("ALTER TABLE timetable_entries ADD COLUMN locked INTEGER NOT NULL DEFAULT 0")

    def _migrate_sections(self) -> None:
        """Give every grade that has no section its first one, and place
        every enrollment that has no section in its grade's. Idempotent: a
        grade that already has a section and an enrollment that already has
        one are left alone, so this runs on every start at no cost.

        A grade's first section is named after its old `grades.section`
        label when it had one, else DEFAULT_SECTION_NAME; so a school that
        never named sections has "10-A", and the students enrolled in grade
        10 are in 10-A. An enrollment is placed only in a grade with exactly
        one section: after the first pass every migrated grade has one, and
        a grade whose principal has since added more is never guessed at.

        The section's id is derived from its grade's, so every instance that
        migrates the same snapshot writes the same rows (a rollout runs two);
        which is why the start does not upload them (commit_derived)."""
        import hashlib
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with self._conn_lock:
            unsectioned = self._fetchall(
                "SELECT g.id, g.academic_year_id, g.section, y.school_id FROM grades g "
                "JOIN academic_years y ON g.academic_year_id = y.id "
                "WHERE NOT EXISTS (SELECT 1 FROM sections s WHERE s.grade_id = g.id)")
            for g in unsectioned:
                name = (g["section"] or "").strip()[:SECTION_NAME_MAX] or DEFAULT_SECTION_NAME
                sid = "section_" + hashlib.sha1(f"first-section:{g['id']}".encode()).hexdigest()[:12]
                self._exec(
                    "INSERT INTO sections (id, school_id, academic_year_id, grade_id, name, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (sid, g["school_id"], g["academic_year_id"], g["id"], name, now))
            self._exec(
                "UPDATE student_enrollments SET section_id = "
                "(SELECT s.id FROM sections s WHERE s.grade_id = student_enrollments.grade_id) "
                "WHERE section_id IS NULL AND "
                "(SELECT COUNT(*) FROM sections s WHERE s.grade_id = student_enrollments.grade_id) = 1")
        if unsectioned:
            logger.info("sections migration: gave %d grade(s) their first section", len(unsectioned))

    # ------------------------------------------------------------------
    # Serialized SQLite access. The store holds ONE shared connection with
    # check_same_thread=False, and FastAPI serves every request on a
    # threadpool thread -- two threads inside self.conn at once corrupt
    # each other's cursors. The 2026-09-15 Render stress test caught this
    # live: concurrent GET /curriculum/boards reads returned rows whose
    # fields all read back as None (then 500'd in pydantic validation).
    # Every statement below funnels through these three helpers so the
    # materialized result -- never a live cursor -- is what escapes the
    # lock. Callers keep their `dict(r)` wrappers; dict(dict) is a no-op
    # copy, so those lines are untouched on purpose.
    # ------------------------------------------------------------------

    def _exec(self, sql: str, params: tuple = ()) -> None:
        """Serialized write (INSERT/UPDATE/DELETE/DDL one-shot)."""
        with self._conn_lock:
            self.conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        """Serialized single-row read, materialized to a plain dict INSIDE
        the lock -- sqlite3.Row values can read back as NULL once another
        thread interleaves an execute, exactly the corruption above."""
        with self._conn_lock:
            r = self.conn.execute(sql, params).fetchone()
            return dict(r) if r is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        """Serialized multi-row read, materialized to plain dicts INSIDE
        the lock, for the same reason as _fetchone."""
        with self._conn_lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def _commit(self) -> None:
        # The commit itself is a shared-connection use and must be
        # serialized like every other one -- two threads committing at
        # once is the same race as two threads executing at once.
        # SnapshotSync.commit takes self._conn_lock around the commit.
        self._snapshots.commit(self.conn)

    def close(self) -> None:
        self._snapshots.close()
        self.conn.close()

    # ---------------- boards ----------------

    def create_board(self, name: str, code: str) -> Board:
        b = Board(id=new_id("board"), name=name, code=code)
        self._exec("INSERT INTO boards (id, name, code) VALUES (?,?,?)",
                          (b.id, b.name, b.code))
        self._commit()
        return b

    def get_board(self, board_id: str) -> Optional[Board]:
        r = self._fetchone("SELECT * FROM boards WHERE id=?", (board_id,))
        return Board(**dict(r)) if r else None

    def get_board_by_code(self, code: str) -> Optional[Board]:
        r = self._fetchone("SELECT * FROM boards WHERE code=?", (code,))
        return Board(**dict(r)) if r else None

    def list_boards(self) -> list[Board]:
        rows = self._fetchall("SELECT * FROM boards")
        return [Board(**dict(r)) for r in rows]

    # ---------------- academic years ----------------

    def create_academic_year(self, *, school_id: str, label: str, start_date: str,
                             end_date: str, status: str = "draft") -> AcademicYear:
        y = AcademicYear(id=new_id("year"), school_id=school_id, label=label,
                         start_date=start_date, end_date=end_date, status=status)
        self._exec(
            "INSERT INTO academic_years (id, school_id, label, start_date, end_date, status) "
            "VALUES (?,?,?,?,?,?)",
            (y.id, y.school_id, y.label, y.start_date, y.end_date, y.status))
        self._commit()
        return y

    def get_academic_year(self, year_id: str) -> Optional[AcademicYear]:
        r = self._fetchone("SELECT * FROM academic_years WHERE id=?", (year_id,))
        return AcademicYear(**dict(r)) if r else None

    def academic_years_for_school(self, school_id: str) -> list[AcademicYear]:
        rows = self._fetchall(
            "SELECT * FROM academic_years WHERE school_id=? ORDER BY start_date", (school_id,))
        return [AcademicYear(**dict(r)) for r in rows]

    def get_academic_year_by_label(self, school_id: str, label: str) -> Optional[AcademicYear]:
        r = self._fetchone(
            "SELECT * FROM academic_years WHERE school_id=? AND label=?", (school_id, label))
        return AcademicYear(**dict(r)) if r else None

    # ---------------- terms ----------------
    # PRD 12.6 / section 0 decision 11 measure per term; see models.Term.
    # create/update raise ValueError (the route's 422) for a range outside
    # the year, an overlap with another term of the year, or a repeated name.
    # The check and the write run under one _conn_lock hold: two principals
    # saving overlapping terms at once must not both pass the check. The
    # statements inside still go through _exec/_fetch* (the RLock nests),
    # and _commit runs after the hold so a snapshot upload never happens
    # while other requests wait on the lock.

    @staticmethod
    def _iso_date(value: str, field_name: str) -> str:
        from datetime import date as _date
        try:
            parsed = _date.fromisoformat(value)
        except (TypeError, ValueError):
            raise ValueError(f"term {field_name} {value!r} is not a YYYY-MM-DD date") from None
        # fromisoformat also takes '20260401'; stored dates are compared as
        # strings, so only the canonical form may be stored (the same trap
        # calendar.validate_holiday documents for holidays).
        if parsed.isoformat() != value:
            raise ValueError(f"term {field_name} {value!r} is not a YYYY-MM-DD date")
        return value

    def _check_term(self, *, year: AcademicYear, name: str, start_date: str, end_date: str,
                    exclude_term_id: Optional[str] = None) -> str:
        """Returns the cleaned name, or raises ValueError. Caller holds
        _conn_lock."""
        name = (name or "").strip()
        if not name:
            raise ValueError("a term needs a name")
        self._iso_date(start_date, "start_date")
        self._iso_date(end_date, "end_date")
        if end_date < start_date:
            raise ValueError(f"term end_date {end_date} is before its start_date {start_date}")
        if start_date < year.start_date or end_date > year.end_date:
            raise ValueError(
                f"term {start_date} .. {end_date} falls outside the academic year "
                f"({year.start_date} .. {year.end_date})")
        for other in self.terms_for_year(year.id):
            if other.id == exclude_term_id:
                continue
            if other.name.casefold() == name.casefold():
                raise ValueError(f"this academic year already has a term named {other.name!r}")
            # Inclusive ranges: sharing even one day is an overlap, or that
            # day would belong to two terms and term_for_date would guess.
            if start_date <= other.end_date and other.start_date <= end_date:
                raise ValueError(
                    f"term {start_date} .. {end_date} overlaps {other.name!r} "
                    f"({other.start_date} .. {other.end_date})")
        return name

    def create_term(self, *, academic_year_id: str, name: str, start_date: str,
                    end_date: str) -> Term:
        year = self.get_academic_year(academic_year_id)
        if year is None:
            raise KeyError(academic_year_id)
        with self._conn_lock:
            name = self._check_term(year=year, name=name, start_date=start_date,
                                    end_date=end_date)
            t = Term(id=new_id("term"), school_id=year.school_id, academic_year_id=year.id,
                     name=name, start_date=start_date, end_date=end_date)
            self._exec(
                "INSERT INTO terms (id, school_id, academic_year_id, name, start_date, end_date) "
                "VALUES (?,?,?,?,?,?)",
                (t.id, t.school_id, t.academic_year_id, t.name, t.start_date, t.end_date))
        self._commit()
        return t

    def update_term(self, term_id: str, *, name: str, start_date: str, end_date: str) -> Term:
        with self._conn_lock:
            current = self.get_term(term_id)
            if current is None:
                raise KeyError(term_id)
            year = self.get_academic_year(current.academic_year_id)
            name = self._check_term(year=year, name=name, start_date=start_date,
                                    end_date=end_date, exclude_term_id=term_id)
            self._exec("UPDATE terms SET name=?, start_date=?, end_date=? WHERE id=?",
                       (name, start_date, end_date, term_id))
        self._commit()
        return Term(id=current.id, school_id=current.school_id,
                    academic_year_id=current.academic_year_id,
                    name=name, start_date=start_date, end_date=end_date,
                    manual_baseline_minutes=current.manual_baseline_minutes)

    def set_term_baseline(self, term_id: str, minutes: Optional[float]) -> Term:
        """Set, or clear with None, the minutes a teacher takes to set a paper
        by hand for this term. Kept apart from update_term, whose callers send
        name and dates only and must not be able to wipe it."""
        if minutes is not None and not (0 < minutes <= 600):
            raise ValueError(f"a baseline of {minutes} minutes is not a paper set by hand")
        with self._conn_lock:
            if self.get_term(term_id) is None:
                raise KeyError(term_id)
            self._exec("UPDATE terms SET manual_baseline_minutes=? WHERE id=?",
                       (minutes, term_id))
        self._commit()
        return self.get_term(term_id)

    def delete_term(self, term_id: str) -> None:
        self._exec("DELETE FROM terms WHERE id=?", (term_id,))
        self._commit()

    def get_term(self, term_id: str) -> Optional[Term]:
        r = self._fetchone("SELECT * FROM terms WHERE id=?", (term_id,))
        return Term(**dict(r)) if r else None

    def terms_for_year(self, academic_year_id: str) -> list[Term]:
        rows = self._fetchall(
            "SELECT * FROM terms WHERE academic_year_id=? ORDER BY start_date", (academic_year_id,))
        return [Term(**dict(r)) for r in rows]

    def term_for_date(self, school_id: str, date: str) -> Optional[Term]:
        """The school's term containing `date` (inclusive), or None. Never a
        guessed term: no term covering the date is None, not the nearest
        one. Terms of one year cannot overlap; if two of the school's years
        overlap and each has a term on this date, that is ambiguous too, and
        also None rather than an arbitrary pick."""
        rows = self._fetchall(
            "SELECT * FROM terms WHERE school_id=? AND start_date<=? AND end_date>=? "
            "ORDER BY start_date", (school_id, date, date))
        return Term(**dict(rows[0])) if len(rows) == 1 else None

    # ---------------- grades ----------------

    def create_grade(self, *, academic_year_id: str, number: int,
                     section: Optional[str] = None) -> Grade:
        """The grade and its first Section, named `section` when given, else
        DEFAULT_SECTION_NAME: a grade is never without a class to enrol in."""
        g = Grade(id=new_id("grade"), academic_year_id=academic_year_id, number=number, section=section)
        # Checked before anything is written: a refused name must not leave a
        # grade row behind for the next commit to save.
        name = self._clean_section_name(section) if section else DEFAULT_SECTION_NAME
        with self._conn_lock:
            self._exec(
                "INSERT INTO grades (id, academic_year_id, number, section) VALUES (?,?,?,?)",
                (g.id, g.academic_year_id, g.number, g.section))
            year = self.get_academic_year(academic_year_id)
            if year is not None:
                self._insert_section(school_id=year.school_id, academic_year_id=academic_year_id,
                                     grade_id=g.id, name=name)
        self._commit()
        return g

    def get_grade(self, grade_id: str) -> Optional[Grade]:
        r = self._fetchone("SELECT * FROM grades WHERE id=?", (grade_id,))
        return Grade(**dict(r)) if r else None

    def grades_for_year(self, academic_year_id: str) -> list[Grade]:
        rows = self._fetchall(
            "SELECT * FROM grades WHERE academic_year_id=? ORDER BY number", (academic_year_id,))
        return [Grade(**dict(r)) for r in rows]

    def get_grade_by_number(self, academic_year_id: str, number: int,
                            section: Optional[str] = None) -> Optional[Grade]:
        # SQLite's IS operator handles a bound NULL parameter correctly
        # (unlike =, which never matches NULL) -- one clause covers both
        # "no section" (section is None) and a real section value.
        r = self._fetchone(
            "SELECT * FROM grades WHERE academic_year_id=? AND number=? AND section IS ?",
            (academic_year_id, number, section))
        return Grade(**dict(r)) if r else None

    # ---------------- sections (M1.1) ----------------
    # create/update raise ValueError (the route's 422) for a bad or repeated
    # name, KeyError for an unknown id, and SectionInUse (409) for a delete
    # that would strand students or leave the grade with no section. The
    # check and the write share one _conn_lock hold, as the terms' do.

    @staticmethod
    def _clean_section_name(name: Optional[str]) -> str:
        cleaned = " ".join((name or "").split())
        if not cleaned:
            raise ValueError("a section needs a name, such as A or Rose")
        if len(cleaned) > SECTION_NAME_MAX:
            raise ValueError(f"a section name is at most {SECTION_NAME_MAX} characters")
        return cleaned

    def _insert_section(self, *, school_id: str, academic_year_id: str, grade_id: str,
                        name: str, class_teacher_id: Optional[str] = None) -> Section:
        """Caller holds _conn_lock, has cleaned the name, and commits."""
        from datetime import datetime, timezone
        sec = Section(id=new_id("section"), school_id=school_id,
                      academic_year_id=academic_year_id, grade_id=grade_id, name=name,
                      class_teacher_id=class_teacher_id,
                      created_at=datetime.now(timezone.utc).isoformat())
        self._exec(
            "INSERT INTO sections (id, school_id, academic_year_id, grade_id, name, "
            "class_teacher_id, created_at) VALUES (?,?,?,?,?,?,?)",
            (sec.id, sec.school_id, sec.academic_year_id, sec.grade_id, sec.name,
             sec.class_teacher_id, sec.created_at))
        return sec

    def _check_section_name_free(self, grade_id: str, name: str,
                                 exclude_section_id: Optional[str] = None) -> None:
        for other in self.sections_for_grade(grade_id):
            if other.id != exclude_section_id and other.name.casefold() == name.casefold():
                raise ValueError(f"this class already has a section named {other.name!r}")

    def create_section(self, *, grade_id: str, name: str,
                       class_teacher_id: Optional[str] = None) -> Section:
        name = self._clean_section_name(name)
        with self._conn_lock:
            grade = self.get_grade(grade_id)
            year = self.get_academic_year(grade.academic_year_id) if grade else None
            if grade is None or year is None:
                raise KeyError(grade_id)
            self._check_section_name_free(grade_id, name)
            sec = self._insert_section(school_id=year.school_id, academic_year_id=year.id,
                                       grade_id=grade_id, name=name,
                                       class_teacher_id=class_teacher_id)
        self._commit()
        return sec

    def get_section(self, section_id: str) -> Optional[Section]:
        r = self._fetchone("SELECT * FROM sections WHERE id=?", (section_id,))
        return Section(**dict(r)) if r else None

    def school_id_for_section(self, section_id: str) -> Optional[str]:
        r = self._fetchone("SELECT school_id FROM sections WHERE id=?", (section_id,))
        return r["school_id"] if r else None

    def sections_for_grade(self, grade_id: str) -> list[Section]:
        rows = self._fetchall(
            "SELECT * FROM sections WHERE grade_id=? ORDER BY name COLLATE NOCASE", (grade_id,))
        return [Section(**dict(r)) for r in rows]

    def sections_for_year(self, academic_year_id: str) -> list[Section]:
        """In class order: grade number, then section name."""
        rows = self._fetchall(
            "SELECT s.* FROM sections s JOIN grades g ON s.grade_id = g.id "
            "WHERE s.academic_year_id=? ORDER BY g.number, s.name COLLATE NOCASE",
            (academic_year_id,))
        return [Section(**dict(r)) for r in rows]

    _UNCHANGED: Any = object()

    def update_section(self, section_id: str, *, name: Any = _UNCHANGED,
                       class_teacher_id: Any = _UNCHANGED) -> tuple[Section, Section]:
        """Rename a section and/or set its class teacher (None clears it).
        An argument left out is not changed. Returns (before, after), for
        the audit entry."""
        with self._conn_lock:
            before = self.get_section(section_id)
            if before is None:
                raise KeyError(section_id)
            new_name = before.name
            if name is not self._UNCHANGED:
                new_name = self._clean_section_name(name)
                self._check_section_name_free(before.grade_id, new_name,
                                              exclude_section_id=section_id)
            new_teacher = (before.class_teacher_id if class_teacher_id is self._UNCHANGED
                           else class_teacher_id)
            self._exec("UPDATE sections SET name=?, class_teacher_id=? WHERE id=?",
                       (new_name, new_teacher, section_id))
        self._commit()
        after = Section(id=before.id, school_id=before.school_id,
                        academic_year_id=before.academic_year_id, grade_id=before.grade_id,
                        name=new_name, class_teacher_id=new_teacher, created_at=before.created_at)
        return before, after

    def delete_section(self, section_id: str) -> Section:
        """Refused while a student is enrolled in it (move them first: their
        class would silently disappear), and for a grade's last section (a
        grade always has a class to enrol in)."""
        with self._conn_lock:
            sec = self.get_section(section_id)
            if sec is None:
                raise KeyError(section_id)
            if len(self.sections_for_grade(sec.grade_id)) <= 1:
                raise SectionInUse("a class keeps at least one section; rename this one instead")
            enrolled = self.enrollments_for_section(section_id)
            if enrolled:
                raise SectionInUse(
                    f"{len(enrolled)} student(s) are enrolled in this section; "
                    "enroll them in another section first")
            self._exec("DELETE FROM sections WHERE id=?", (section_id,))
        self._commit()
        return sec

    def enrollments_for_section(self, section_id: str) -> list[StudentEnrollment]:
        rows = self._fetchall(
            "SELECT * FROM student_enrollments WHERE section_id=? ORDER BY created_at",
            (section_id,))
        return [StudentEnrollment(**dict(r)) for r in rows]

    def enrollment_counts_for_year(self, academic_year_id: str) -> dict[str, int]:
        """Students enrolled per section of the year, by section id; a
        section with none is absent."""
        rows = self._fetchall(
            "SELECT e.section_id, COUNT(*) AS n FROM student_enrollments e "
            "JOIN sections s ON e.section_id = s.id WHERE s.academic_year_id=? "
            "GROUP BY e.section_id", (academic_year_id,))
        return {r["section_id"]: r["n"] for r in rows}

    # ---------------- subjects ----------------

    def create_subject(self, *, grade_id: str, name: str, code: Optional[str] = None) -> Subject:
        s = Subject(id=new_id("subj"), grade_id=grade_id, name=name, code=code)
        self._exec("INSERT INTO subjects (id, grade_id, name, code) VALUES (?,?,?,?)",
                          (s.id, s.grade_id, s.name, s.code))
        self._commit()
        return s

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        r = self._fetchone("SELECT * FROM subjects WHERE id=?", (subject_id,))
        return Subject(**dict(r)) if r else None

    def subjects_for_grade(self, grade_id: str) -> list[Subject]:
        rows = self._fetchall("SELECT * FROM subjects WHERE grade_id=? ORDER BY name", (grade_id,))
        return [Subject(**dict(r)) for r in rows]

    def get_subject_by_name(self, grade_id: str, name: str) -> Optional[Subject]:
        r = self._fetchone(
            "SELECT * FROM subjects WHERE grade_id=? AND name=?", (grade_id, name))
        return Subject(**dict(r)) if r else None

    # ---------------- books ----------------

    def create_book(self, *, subject_id: str, board_id: str, title: str,
                    publisher: Optional[str] = None,
                    source_doc_ids: Optional[list[str]] = None,
                    status: str = "selected") -> "Any":
        b = Book(id=new_id("book"), subject_id=subject_id, board_id=board_id, title=title,
                publisher=publisher, source_doc_ids=source_doc_ids or [], status=status)
        with self._conn_lock:
            # A subject's one book is its edition only implicitly (no
            # book_selections row). Before a second book arrives, write that
            # choice down: otherwise adding a book would leave the subject
            # with two and no edition, and every screen using it would stop.
            existing = self.books_for_subject(subject_id)
            year_id = self.academic_year_id_for_subject(subject_id)
            if (len(existing) == 1 and year_id is not None
                    and self._selection_row(year_id, subject_id) is None):
                self._write_selection(year_id, subject_id, existing[0].id)
            self._exec(
                "INSERT INTO books (id, subject_id, board_id, title, publisher, source_doc_ids, status) "
                "VALUES (?,?,?,?,?,?,?)",
                (b.id, b.subject_id, b.board_id, b.title, b.publisher,
                 json.dumps(b.source_doc_ids), b.status))
        self._commit()
        return b

    # ---------------- book selection (PRD 12.7) ----------------

    def academic_year_id_for_subject(self, subject_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT g.academic_year_id FROM subjects s JOIN grades g ON s.grade_id = g.id "
            "WHERE s.id=?", (subject_id,))
        return r["academic_year_id"] if r else None

    def _selection_row(self, academic_year_id: str, subject_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM book_selections WHERE academic_year_id=? AND subject_id=?",
            (academic_year_id, subject_id))

    def _write_selection(self, academic_year_id: str, subject_id: str, book_id: str) -> None:
        from datetime import datetime, timezone
        # An upsert on the primary key: replacing the choice, never adding a
        # second one for the same subject and year.
        self._exec(
            "INSERT INTO book_selections (academic_year_id, subject_id, book_id, selected_at) "
            "VALUES (?,?,?,?) ON CONFLICT(academic_year_id, subject_id) "
            "DO UPDATE SET book_id=excluded.book_id, selected_at=excluded.selected_at",
            (academic_year_id, subject_id, book_id, datetime.now(timezone.utc).isoformat()))

    def selected_book_for_subject(self, subject_id: str) -> Optional[Book]:
        """The subject's edition for its academic year -- what every caller
        uses instead of "the first book" (the web admin used books.first,
        i.e. SQLite row order). The recorded choice if there is one; else the
        subject's only book, which is every seeded subject today; else None:
        no book, or several and no choice, is reported, never guessed."""
        year_id = self.academic_year_id_for_subject(subject_id)
        if year_id is None:
            return None
        row = self._selection_row(year_id, subject_id)
        if row is not None:
            return self.get_book(row["book_id"])
        books = self.books_for_subject(subject_id)
        return books[0] if len(books) == 1 else None

    def select_book(self, book_id: str) -> BookSelection:
        """Makes `book_id` its subject's edition for the subject's academic
        year. Raises KeyError for an unknown book. Lessons already scheduled
        from other editions, and teachers assigned to them, are left exactly
        where they are -- the lessons were placed from those books' chapters
        -- and counted per book in the result so the caller can say so.

        Counted over every other book of the subject, not just the edition
        chosen before this one: after A -> B -> C, A's lessons still exist,
        and a first choice among several books has no previous edition."""
        book = self.get_book(book_id)
        if book is None:
            raise KeyError(book_id)
        year_id = self.academic_year_id_for_subject(book.subject_id)
        if year_id is None:
            raise KeyError(book.subject_id)
        with self._conn_lock:
            previous = self.selected_book_for_subject(book.subject_id)
            self._write_selection(year_id, book.subject_id, book.id)
        self._commit()
        previous_id = previous.id if previous is not None else None
        others = [b for b in self.books_for_subject(book.subject_id) if b.id != book.id]
        lessons: dict[str, int] = {}
        assignments: dict[str, int] = {}
        if others:
            ids = [b.id for b in others]
            marks = ",".join("?" * len(ids))
            for r in self._fetchall(
                    f"SELECT book_id, COUNT(*) AS n FROM scheduled_lessons "
                    f"WHERE academic_year_id=? AND book_id IN ({marks}) GROUP BY book_id",
                    (year_id, *ids)):
                lessons[r["book_id"]] = r["n"]
            # Books already belong to one subject of one year, so these rows
            # need no year filter.
            for r in self._fetchall(
                    f"SELECT book_id, COUNT(*) AS n FROM teacher_assignments "
                    f"WHERE book_id IN ({marks}) GROUP BY book_id", tuple(ids)):
                assignments[r["book_id"]] = r["n"]
        editions = [OtherEdition(book_id=b.id, title=b.title,
                                 scheduled_lessons=lessons.get(b.id, 0),
                                 teacher_assignments=assignments.get(b.id, 0))
                    for b in others]
        return BookSelection(academic_year_id=year_id, subject_id=book.subject_id,
                             book_id=book.id, previous_book_id=previous_id,
                             previous_book_scheduled_lessons=(
                                 lessons.get(previous_id, 0) if previous_id != book.id else 0),
                             other_editions=editions)

    def get_book(self, book_id: str):
        from .models import Book
        r = self._fetchone("SELECT * FROM books WHERE id=?", (book_id,))
        if not r:
            return None
        d = dict(r)
        d["source_doc_ids"] = json.loads(d["source_doc_ids"] or "[]")
        return Book(**d)

    def books_for_subject(self, subject_id: str) -> list["Any"]:
        rows = self._fetchall("SELECT * FROM books WHERE subject_id=?", (subject_id,))
        out = []
        from .models import Book
        for r in rows:
            d = dict(r)
            d["source_doc_ids"] = json.loads(d["source_doc_ids"] or "[]")
            out.append(Book(**d))
        return out

    def get_book_by_title(self, subject_id: str, title: str) -> Optional["Any"]:
        from .models import Book
        r = self._fetchone(
            "SELECT * FROM books WHERE subject_id=? AND title=?", (subject_id, title))
        if not r:
            return None
        d = dict(r)
        d["source_doc_ids"] = json.loads(d["source_doc_ids"] or "[]")
        return Book(**d)

    # ---------------- units ----------------

    def create_unit(self, *, canonical_id: str, book_id: str, unit_no: str, name: str,
                    marks: Optional[int] = None, seq: int = 0) -> Unit:
        u = Unit(id=new_id("unit"), canonical_id=canonical_id, book_id=book_id,
                 unit_no=unit_no, name=name, marks=marks, seq=seq)
        self._exec(
            "INSERT INTO units (id, canonical_id, book_id, unit_no, name, marks, seq) "
            "VALUES (?,?,?,?,?,?,?)",
            (u.id, u.canonical_id, u.book_id, u.unit_no, u.name, u.marks, u.seq))
        self._commit()
        return u

    def get_unit(self, unit_id: str) -> Optional[Unit]:
        r = self._fetchone("SELECT * FROM units WHERE id=?", (unit_id,))
        return Unit(**dict(r)) if r else None

    def get_unit_by_canonical_id(self, canonical_id: str) -> Optional[Unit]:
        r = self._fetchone("SELECT * FROM units WHERE canonical_id=?", (canonical_id,))
        return Unit(**dict(r)) if r else None

    def units_for_book(self, book_id: str) -> list[Unit]:
        rows = self._fetchall("SELECT * FROM units WHERE book_id=? ORDER BY seq", (book_id,))
        return [Unit(**dict(r)) for r in rows]

    # ---------------- chapters ----------------

    def create_chapter(self, *, canonical_id: str, unit_id: str, name: str, seq: int = 0) -> Chapter:
        c = Chapter(id=new_id("chap"), canonical_id=canonical_id, unit_id=unit_id, name=name, seq=seq)
        self._exec(
            "INSERT INTO chapters (id, canonical_id, unit_id, name, seq) VALUES (?,?,?,?,?)",
            (c.id, c.canonical_id, c.unit_id, c.name, c.seq))
        self._commit()
        return c

    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        r = self._fetchone("SELECT * FROM chapters WHERE id=?", (chapter_id,))
        return Chapter(**dict(r)) if r else None

    def get_chapter_by_canonical_id(self, canonical_id: str) -> Optional[Chapter]:
        r = self._fetchone("SELECT * FROM chapters WHERE canonical_id=?", (canonical_id,))
        return Chapter(**dict(r)) if r else None

    def chapter_for_syllabus_slug(self, *, school_id: str, grade_number: int,
                                  subject_name: str, slug: str,
                                  title_slug: Optional[str] = None,
                                  on_date: Optional[str] = None) -> ChapterSlugMatch:
        """One school's own Chapter row for a syllabus/catalog chapter slug.

        The slug (`chemical-reactions-equations`) is the syllabus id, from
        academicos-data/syllabus/*.json by way of
        `GET /api/v1/catalog/{subject}/{grade}/chapters`; a school's own
        chapters are keyed `chap_<uuid>`. The two meet on canonical_id,
        which seed_cbse10.py writes as `{book_id}:chapter:{that same slug}`
        -- a STORED mapping recorded when the curriculum was seeded from the
        syllabus files, not a name match guessed at read time, so a chapter
        the school has since renamed still resolves.

        `title_slug` is the second try, for a book whose chapters came from
        an ingested Table of Contents instead: extraction.py keys those by
        the slug of the chapter TITLE ("chemical-reactions-and-equations"),
        which is a different string from the syllabus id.

        Searched inside the subject's CHOSEN edition only (PRD 12.7 / the
        one-edition-per-subject-per-year rule): two editions have different
        chapters, and a filter that silently read the unchosen one would
        offer subtopics nobody in that class is being taught.

        Never crosses schools: every lookup starts from this school's own
        academic years, so the same official slug resolves to each school's
        own rows and to nothing else.
        """
        # The year that contains `on_date` first, then the most recent --
        # list.sort is stable, so the second sort keeps that order within
        # each group.
        years = sorted(self.academic_years_for_school(school_id),
                       key=lambda y: y.start_date, reverse=True)
        if on_date:
            years.sort(key=lambda y: not (y.start_date <= on_date <= y.end_date))
        # The year the school is actually in, if it has one -- the sort above
        # has already put it first.
        current = (years[0] if years and on_date
                   and years[0].start_date <= on_date <= years[0].end_date else None)

        # Any year may ANSWER: a half-set-up 2026-27 must never hide a fully
        # set-up 2025-26 behind "not set up", so the loop runs on past the
        # current year looking for an `ok`.
        #
        # But a REASON is advice, and advice belongs to the year the teacher
        # is in. Ranking every year's reason and reporting the furthest one
        # (what this did until 2026-09-23) tells a school whose current year
        # has two editions and no choice -- the state the real
        # academicos-data/curriculum/curriculum.sqlite is in -- that "your
        # Class 10 Science book has no chapter 'X'", because last year's
        # finished, fully-chosen edition ranks higher. That is a true
        # sentence about a year nobody is teaching, and it never mentions
        # the one action that would fix the year she is in. So when a year
        # contains `on_date` and it did not resolve, its reason is the
        # answer; the ranking is the fallback for a school between sessions,
        # where there is no current year to prefer.
        best = ChapterSlugMatch(reason="no_year")
        current_found: Optional[ChapterSlugMatch] = None
        for year in years:
            found = self._chapter_for_slug_in_year(
                year.id, grade_number=grade_number, subject_name=subject_name,
                slug=slug, title_slug=title_slug)
            if found.reason == "ok":
                return found
            if current is not None and year.id == current.id:
                current_found = found
            elif (CHAPTER_SLUG_REASONS.index(found.reason)
                    > CHAPTER_SLUG_REASONS.index(best.reason)):
                best = found
        return current_found if current_found is not None else best

    def _chapter_for_slug_in_year(self, academic_year_id: str, *, grade_number: int,
                                  subject_name: str, slug: str,
                                  title_slug: Optional[str]) -> ChapterSlugMatch:
        grade = self.get_grade_by_number(academic_year_id, grade_number)
        if grade is None:
            return ChapterSlugMatch(reason="no_grade")
        subject = self.get_subject_by_name(grade.id, subject_name)
        if subject is None:
            return ChapterSlugMatch(reason="no_subject")
        if not self.books_for_subject(subject.id):
            return ChapterSlugMatch(reason="no_book")
        book = self.selected_book_for_subject(subject.id)
        if book is None:
            # Several editions and no recorded choice: guessing one here is
            # exactly what select_book() exists to stop.
            return ChapterSlugMatch(reason="no_edition_chosen")
        for key in (slug, title_slug):
            if not key:
                continue
            chapter = self.get_chapter_by_canonical_id(f"{book.id}:chapter:{key}")
            if chapter is not None:
                return ChapterSlugMatch(reason="ok", chapter=chapter, book=book)
        return ChapterSlugMatch(reason="no_chapter", book=book)

    def school_id_for_book(self, book_id: str) -> Optional[str]:
        """Ownership-chain lookup (book -> subject -> grade -> academic_year
        -> school_id) -- what every write endpoint in routes.py uses to
        verify a caller's school actually owns the book/chapter they're
        trying to touch, the same school-scoping pattern this session
        already applied to pillar_routes.py's authorization fixes."""
        r = self._fetchone(
            "SELECT y.school_id FROM books b "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE b.id=?", (book_id,))
        return r["school_id"] if r else None

    def school_id_for_grade(self, grade_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT y.school_id FROM grades g JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE g.id=?", (grade_id,))
        return r["school_id"] if r else None

    def grade_id_for_book(self, book_id: str) -> Optional[str]:
        """book -> subject -> grade -- what a student's schedule filters
        by: every book taught to their real enrolled grade, not just one
        subject (unlike a teacher, who is scoped to specific books via
        TeacherAssignment)."""
        r = self._fetchone(
            "SELECT s.grade_id FROM books b JOIN subjects s ON b.subject_id = s.id WHERE b.id=?",
            (book_id,))
        return r["grade_id"] if r else None

    def book_ids_for_grade(self, grade_id: str) -> list[str]:
        """Every book of every subject of the grade, abandoned editions
        included. Reads that answer "what is this class taught?" want
        selected_book_ids_for_grade instead."""
        rows = self._fetchall(
            "SELECT b.id FROM books b JOIN subjects s ON b.subject_id = s.id WHERE s.grade_id=?",
            (grade_id,))
        return [r["id"] for r in rows]

    def selected_book_ids_for_grade(self, grade_id: str) -> list[str]:
        """One book per subject of the grade: the edition the school chose
        (PRD 12.7), or the subject's only book when no choice was recorded
        -- selected_book_for_subject's rule, applied to a whole class.

        Every read of "this class's books" goes through this rather than
        book_ids_for_grade. The review of fc279a4 drove a student through
        the real routes after a principal added a second edition: my-progress
        returned two 'Mathematics' rows (2 lessons from the abandoned book,
        3 from the chosen one) with nothing to tell them apart, and the class
        calendar showed both editions' lessons on the same days. A subject
        with several books and no recorded choice contributes nothing, the
        same "report, never guess" answer selected_book_for_subject gives --
        create_book records the implicit choice before a second book lands,
        so that case needs a hand-written row to reach."""
        out: list[str] = []
        for s in self.subjects_for_grade(grade_id):
            b = self.selected_book_for_subject(s.id)
            if b is not None:
                out.append(b.id)
        return out

    def school_id_for_unit(self, unit_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT y.school_id FROM units u "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE u.id=?", (unit_id,))
        return r["school_id"] if r else None

    def school_id_for_chapter(self, chapter_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT y.school_id FROM chapters c "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE c.id=?", (chapter_id,))
        return r["school_id"] if r else None

    def school_id_for_topic(self, topic_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT y.school_id FROM topics t "
            "JOIN chapters c ON t.chapter_id = c.id "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE t.id=?", (topic_id,))
        return r["school_id"] if r else None

    def school_id_for_subtopic(self, subtopic_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT y.school_id FROM subtopics st "
            "JOIN topics t ON st.topic_id = t.id "
            "JOIN chapters c ON t.chapter_id = c.id "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE st.id=?", (subtopic_id,))
        return r["school_id"] if r else None

    def school_ids_for_subtopics(self, subtopic_ids: list[str]) -> dict[str, str]:
        """school_id_for_subtopic for a whole request's ids in one query:
        {subtopic_id: school_id}. The two subtopic-keyed question routes
        (curriculum questions/by-subtopics and assessment questions/search)
        take a caller-picked list, and the per-id lookup is a seven-join query
        per subtopic. An id that resolves to no school is absent from the result
        rather than mapped to None -- the callers treat "another school's" and
        "no longer exists" differently (403 vs. ignore)."""
        if not subtopic_ids:
            return {}
        unique = list(dict.fromkeys(subtopic_ids))
        placeholders = ",".join("?" * len(unique))
        rows = self._fetchall(
            "SELECT st.id AS subtopic_id, y.school_id FROM subtopics st "
            "JOIN topics t ON st.topic_id = t.id "
            "JOIN chapters c ON t.chapter_id = c.id "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            f"WHERE st.id IN ({placeholders})", tuple(unique))
        return {r["subtopic_id"]: r["school_id"] for r in rows}

    _SEQUENCE_TABLES = {"unit": "units", "chapter": "chapters", "topic": "topics", "subtopic": "subtopics"}
    _SEQUENCE_PARENTS = {"unit": "book_id", "chapter": "unit_id", "topic": "chapter_id",
                         "subtopic": "topic_id"}

    def set_sequence(self, entity_type: str, entity_id: str, seq: int) -> None:
        """Moves one Unit/Chapter/Topic/Subtopic to delivery position `seq`
        among its siblings (same book/unit/chapter/topic) and renumbers the
        whole sibling list 0..n-1 in one transaction: the others shift to
        make room, so sequence numbers stay unique and contiguous. `seq` past
        the end means last; below 0 means first.

        Until 2026-09-22 this updated the one row and never touched its
        siblings: moving 'Carbon and its Compounds' to seq 0 left two
        chapters of that unit at 0, and the regenerated schedule did not
        start with the moved chapter (ORDER BY seq broke the tie by storage
        order). Siblings are read ORDER BY seq, rowid, so a list that
        already holds duplicates is repaired deterministically by the move.

        The `seq` field is what units_for_book/chapters_for_unit/
        topics_for_chapter/subtopics_for_topic ORDER BY and what
        scheduling.schedule_book() walks. A change takes effect on the next
        schedule_book() call; it does not re-date lessons already scheduled.

        Unlike rename_topic/rename_subtopic's silent no-op on an unknown id,
        this raises: a reorder is a deliberate admin action, and reordering
        something that does not exist deserves a real error."""
        table = self._SEQUENCE_TABLES.get(entity_type)
        if table is None:
            raise ValueError(f"unknown sequence entity_type: {entity_type!r}")
        parent_col = self._SEQUENCE_PARENTS[entity_type]
        with self._conn_lock:
            row = self.conn.execute(
                f"SELECT {parent_col} FROM {table} WHERE id=?", (entity_id,)).fetchone()
            if row is None:
                raise ValueError(f"no {entity_type} with id {entity_id!r}")
            ordered = [r[0] for r in self.conn.execute(
                f"SELECT id FROM {table} WHERE {parent_col}=? AND id<>? ORDER BY seq, rowid",
                (row[0], entity_id)).fetchall()]
            ordered.insert(min(max(seq, 0), len(ordered)), entity_id)
            try:
                for position, sibling_id in enumerate(ordered):
                    self.conn.execute(f"UPDATE {table} SET seq=? WHERE id=?", (position, sibling_id))
            except sqlite3.Error:
                self.conn.rollback()
                raise
        self._commit()

    def chapters_for_unit(self, unit_id: str) -> list[Chapter]:
        rows = self._fetchall("SELECT * FROM chapters WHERE unit_id=? ORDER BY seq", (unit_id,))
        return [Chapter(**dict(r)) for r in rows]

    def chapters_for_book(self, book_id: str) -> list[Chapter]:
        """Convenience join: every chapter across every unit of a book, in
        unit-then-chapter seq order -- what an admin's curriculum review
        screen (§29) actually wants to show."""
        rows = self._fetchall(
            "SELECT c.* FROM chapters c JOIN units u ON c.unit_id = u.id "
            "WHERE u.book_id=? ORDER BY u.seq, c.seq", (book_id,))
        return [Chapter(**dict(r)) for r in rows]

    # ---------------- topics ----------------

    def create_topic(self, *, canonical_id: str, chapter_id: str, name: str, seq: int = 0,
                     description: Optional[str] = None, source_type: str = "manual",
                     source_reference: Optional[str] = None, approved_by: Optional[str] = None,
                     approved_at: Optional[str] = None, model_used: Optional[str] = None,
                     generation_version: Optional[str] = None) -> Topic:
        t = Topic(id=new_id("topic"), canonical_id=canonical_id, chapter_id=chapter_id, name=name,
                  seq=seq, description=description, source_type=source_type,
                  source_reference=source_reference, approved_by=approved_by,
                  approved_at=approved_at, model_used=model_used,
                  generation_version=generation_version)
        self._exec(
            "INSERT INTO topics (id, canonical_id, chapter_id, name, seq, description, "
            "source_type, source_reference, approved_by, approved_at, model_used, "
            "generation_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.id, t.canonical_id, t.chapter_id, t.name, t.seq, t.description, t.source_type,
             t.source_reference, t.approved_by, t.approved_at, t.model_used, t.generation_version))
        self._commit()
        return t

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        r = self._fetchone("SELECT * FROM topics WHERE id=?", (topic_id,))
        return Topic(**dict(r)) if r else None

    def topics_for_chapter(self, chapter_id: str) -> list[Topic]:
        rows = self._fetchall("SELECT * FROM topics WHERE chapter_id=? ORDER BY seq", (chapter_id,))
        return [Topic(**dict(r)) for r in rows]

    def rename_topic(self, topic_id: str, new_name: str) -> None:
        """§ acceptance criteria: renaming a topic must not break question
        references -- canonical_id (what qmap.py/QuestionSubtopicLink key
        against) is untouched by this; only the display name changes."""
        self._exec("UPDATE topics SET name=? WHERE id=?", (new_name, topic_id))
        self._commit()

    # ---------------- subtopics ----------------

    def create_subtopic(self, *, canonical_id: str, topic_id: str, name: str, seq: int = 0,
                        description: Optional[str] = None, source_type: str = "manual",
                        source_reference: Optional[str] = None, approved_by: Optional[str] = None,
                        approved_at: Optional[str] = None, model_used: Optional[str] = None,
                        generation_version: Optional[str] = None) -> Subtopic:
        s = Subtopic(id=new_id("subtopic"), canonical_id=canonical_id, topic_id=topic_id, name=name,
                     seq=seq, description=description, source_type=source_type,
                     source_reference=source_reference, approved_by=approved_by,
                     approved_at=approved_at, model_used=model_used,
                     generation_version=generation_version)
        self._exec(
            "INSERT INTO subtopics (id, canonical_id, topic_id, name, seq, description, "
            "source_type, source_reference, approved_by, approved_at, model_used, "
            "generation_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.id, s.canonical_id, s.topic_id, s.name, s.seq, s.description, s.source_type,
             s.source_reference, s.approved_by, s.approved_at, s.model_used, s.generation_version))
        self._commit()
        return s

    def get_subtopic(self, subtopic_id: str) -> Optional[Subtopic]:
        r = self._fetchone("SELECT * FROM subtopics WHERE id=?", (subtopic_id,))
        return Subtopic(**dict(r)) if r else None

    def get_subtopic_by_canonical_id(self, canonical_id: str) -> Optional[Subtopic]:
        r = self._fetchone("SELECT * FROM subtopics WHERE canonical_id=?", (canonical_id,))
        return Subtopic(**dict(r)) if r else None

    def subtopics_for_topic(self, topic_id: str) -> list[Subtopic]:
        rows = self._fetchall("SELECT * FROM subtopics WHERE topic_id=? ORDER BY seq", (topic_id,))
        return [Subtopic(**dict(r)) for r in rows]

    def rename_subtopic(self, subtopic_id: str, new_name: str) -> None:
        self._exec("UPDATE subtopics SET name=? WHERE id=?", (new_name, subtopic_id))
        self._commit()

    class SubtopicHasLinkedQuestions(Exception):
        """Raised by delete_subtopic when real question tags exist --
        acceptance criteria: deleting a subtopic with linked questions
        must be prevented or require migration, never silently orphan
        those links."""

    def delete_subtopic(self, subtopic_id: str, *, force: bool = False) -> None:
        linked = self.question_links_for_subtopic(subtopic_id)
        if linked and not force:
            raise CurriculumStore.SubtopicHasLinkedQuestions(
                f"subtopic {subtopic_id} has {len(linked)} linked question(s); "
                "pass force=True to delete anyway (also deletes those links) "
                "or re-tag the questions to a different subtopic first")
        if force:
            self._exec("DELETE FROM question_subtopic_links WHERE subtopic_id=?", (subtopic_id,))
        self._exec("DELETE FROM subtopics WHERE id=?", (subtopic_id,))
        self._commit()

    # ---------------- question <-> subtopic links ----------------

    def link_question_to_subtopic(self, *, question_id: str, subtopic_id: str,
                                  method: str = "lexical", confidence: float = 0.5) -> QuestionSubtopicLink:
        link = QuestionSubtopicLink(id=new_id("qsl"), question_id=question_id, subtopic_id=subtopic_id,
                                    method=method, confidence=confidence)
        # A question can be re-tagged (new method/confidence) without
        # accumulating duplicate rows for the same (question, subtopic) pair.
        self._exec(
            "INSERT INTO question_subtopic_links (id, question_id, subtopic_id, method, confidence) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(question_id, subtopic_id) DO UPDATE SET method=excluded.method, "
            "confidence=excluded.confidence",
            (link.id, link.question_id, link.subtopic_id, link.method, link.confidence))
        self._commit()
        return link

    def subtopics_for_question(self, question_id: str) -> list[QuestionSubtopicLink]:
        rows = self._fetchall(
            "SELECT * FROM question_subtopic_links WHERE question_id=?", (question_id,))
        return [QuestionSubtopicLink(**dict(r)) for r in rows]

    def question_links_for_subtopic(self, subtopic_id: str) -> list[QuestionSubtopicLink]:
        rows = self._fetchall(
            "SELECT * FROM question_subtopic_links WHERE subtopic_id=?", (subtopic_id,))
        return [QuestionSubtopicLink(**dict(r)) for r in rows]

    def tagged_question_counts(self, subtopic_ids: list[str], *,
                               eligible_question_ids: Optional[set[str]] = None,
                               ) -> dict[str, int]:
        """How many distinct questions are tagged to each of these subtopics.

        One query for a whole chapter rather than a query per subtopic --
        the picker asks for every subtopic of every selected chapter at once.
        Subtopics with nothing tagged are absent from the SQL result and are
        filled in as 0 here, because "0" is the answer the caller needs to
        show, not a missing key.

        `eligible_question_ids` is the set of questions the caller could
        actually return, and it is why this is not a bare COUNT(*). A link
        row is (question_id, subtopic_id) and nothing deletes a link when
        the question id it names stops existing: the 2026-09-21 bank rebuild
        re-tagged Science 10 chapters and churned question ids, and a
        rebuild is exactly what strands links. assessment/routes.py's
        `POST /questions/search` intersects these links with the pool
        candidates for the subject/grade/chapter, so a stranded link can
        never contribute a question to a paper. Counting it anyway would
        print "7 questions tagged" next to a subtopic that yields an empty
        paper -- the same silent wrongness this count was added to remove.
        Pass None only when the caller genuinely means "every link row"."""
        counts = {sid: 0 for sid in subtopic_ids}
        if not subtopic_ids:
            return counts
        placeholders = ",".join("?" * len(subtopic_ids))
        rows = self._fetchall(
            f"SELECT DISTINCT subtopic_id, question_id FROM question_subtopic_links "
            f"WHERE subtopic_id IN ({placeholders})", subtopic_ids)
        for r in rows:
            if eligible_question_ids is not None and r["question_id"] not in eligible_question_ids:
                continue
            counts[r["subtopic_id"]] += 1
        return counts

    def question_ids_for_subtopics(self, subtopic_ids: list[str]) -> list[str]:
        """The real "generate a paper from these subtopics" query: every
        question tagged (via qmap.py's resolution -> link_question_to_subtopic)
        to any of the given subtopics."""
        if not subtopic_ids:
            return []
        placeholders = ",".join("?" * len(subtopic_ids))
        rows = self._fetchall(
            f"SELECT DISTINCT question_id FROM question_subtopic_links "
            f"WHERE subtopic_id IN ({placeholders})", subtopic_ids)
        return [r["question_id"] for r in rows]

    # ---------------- curriculum extraction runs / proposals ----------------

    def create_extraction_run(self, *, school_id: str, book_id: str, chapter_id: str,
                              prompt_version: str, source_hash: Optional[str] = None,
                              model: Optional[str] = None) -> CurriculumExtractionRun:
        from datetime import datetime, timezone
        run = CurriculumExtractionRun(
            id=new_id("extrun"), school_id=school_id, book_id=book_id, chapter_id=chapter_id,
            source_hash=source_hash, model=model, prompt_version=prompt_version,
            created_at=datetime.now(timezone.utc).isoformat(), status="pending")
        self._exec(
            "INSERT INTO curriculum_extraction_runs (id, school_id, book_id, chapter_id, "
            "source_hash, model, prompt_version, created_at, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (run.id, run.school_id, run.book_id, run.chapter_id, run.source_hash, run.model,
             run.prompt_version, run.created_at, run.status))
        self._commit()
        return run

    def get_extraction_run(self, run_id: str) -> Optional[CurriculumExtractionRun]:
        r = self._fetchone(
            "SELECT * FROM curriculum_extraction_runs WHERE id=?", (run_id,))
        return CurriculumExtractionRun(**dict(r)) if r else None

    def extraction_runs_for_chapter(self, chapter_id: str) -> list[CurriculumExtractionRun]:
        rows = self._fetchall(
            "SELECT * FROM curriculum_extraction_runs WHERE chapter_id=? ORDER BY created_at DESC",
            (chapter_id,))
        return [CurriculumExtractionRun(**dict(r)) for r in rows]

    def update_run_status(self, run_id: str, status: str) -> None:
        self._exec(
            "UPDATE curriculum_extraction_runs SET status=? WHERE id=?", (status, run_id))
        self._commit()

    def add_extraction_proposal(self, *, run_id: str, entity_type: str, proposed_name: str,
                                proposed_description: Optional[str] = None,
                                proposed_parent: Optional[str] = None, sequence: int = 0,
                                confidence: float = 0.5) -> CurriculumExtractionProposal:
        p = CurriculumExtractionProposal(
            id=new_id("extprop"), run_id=run_id, entity_type=entity_type,
            proposed_name=proposed_name, proposed_description=proposed_description,
            proposed_parent=proposed_parent, sequence=sequence, confidence=confidence)
        self._exec(
            "INSERT INTO curriculum_extraction_proposals (id, run_id, entity_type, proposed_name, "
            "proposed_description, proposed_parent, sequence, confidence, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (p.id, p.run_id, p.entity_type, p.proposed_name, p.proposed_description,
             p.proposed_parent, p.sequence, p.confidence, p.status))
        self._commit()
        return p

    def proposals_for_run(self, run_id: str) -> list[CurriculumExtractionProposal]:
        rows = self._fetchall(
            "SELECT * FROM curriculum_extraction_proposals WHERE run_id=? ORDER BY sequence",
            (run_id,))
        return [CurriculumExtractionProposal(**dict(r)) for r in rows]

    def get_proposal(self, proposal_id: str) -> Optional[CurriculumExtractionProposal]:
        r = self._fetchone(
            "SELECT * FROM curriculum_extraction_proposals WHERE id=?", (proposal_id,))
        return CurriculumExtractionProposal(**dict(r)) if r else None

    def set_proposal_status(self, proposal_id: str, status: str, *,
                            edited_name: Optional[str] = None) -> None:
        """status: 'approved' | 'edited' | 'rejected'. edited_name is the
        admin's replacement text when status='edited' -- the proposal keeps
        proposed_name as the original AI suggestion (traceability) and
        edited_name as what actually gets materialized."""
        self._exec(
            "UPDATE curriculum_extraction_proposals SET status=?, edited_name=? WHERE id=?",
            (status, edited_name, proposal_id))
        self._commit()

    def set_proposal_materialized_id(self, proposal_id: str, materialized_id: str) -> None:
        self._exec(
            "UPDATE curriculum_extraction_proposals SET materialized_id=? WHERE id=?",
            (materialized_id, proposal_id))
        self._commit()

    # ---------------- period configuration ----------------

    def create_period_configuration(self, *, school_id: str, academic_year_id: str,
                                    period_minutes: int) -> PeriodConfiguration:
        p = PeriodConfiguration(id=new_id("periodcfg"), school_id=school_id,
                                academic_year_id=academic_year_id, period_minutes=period_minutes)
        self._exec(
            "INSERT INTO period_configurations (id, school_id, academic_year_id, period_minutes) "
            "VALUES (?,?,?,?)",
            (p.id, p.school_id, p.academic_year_id, p.period_minutes))
        self._commit()
        return p

    def period_configuration_for_year(self, academic_year_id: str) -> Optional[PeriodConfiguration]:
        r = self._fetchone(
            "SELECT * FROM period_configurations WHERE academic_year_id=?", (academic_year_id,))
        return PeriodConfiguration(**dict(r)) if r else None

    def set_subject_period_allocation(self, *, school_id: str, academic_year_id: str,
                                      subject: str, periods_per_week: int) -> SubjectPeriodAllocation:
        """Upsert, not create-once-then-409 (see SubjectPeriodAllocation's
        own docstring for why): a school setting Science to 6 periods/week
        this term and 7 next term should not need a distinct row per term,
        and re-POSTing must update, not IntegrityError."""
        existing = self._fetchone(
            "SELECT id FROM subject_period_allocations WHERE academic_year_id=? AND subject=?",
            (academic_year_id, subject))
        alloc_id = existing["id"] if existing else new_id("spa")
        self._exec(
            "INSERT INTO subject_period_allocations "
            "(id, school_id, academic_year_id, subject, periods_per_week) VALUES (?,?,?,?,?) "
            "ON CONFLICT(academic_year_id, subject) DO UPDATE SET periods_per_week=excluded.periods_per_week",
            (alloc_id, school_id, academic_year_id, subject, periods_per_week))
        self._commit()
        return SubjectPeriodAllocation(id=alloc_id, school_id=school_id, academic_year_id=academic_year_id,
                                       subject=subject, periods_per_week=periods_per_week)

    def subject_period_allocation(self, academic_year_id: str, subject: str) -> Optional[SubjectPeriodAllocation]:
        r = self._fetchone(
            "SELECT * FROM subject_period_allocations WHERE academic_year_id=? AND subject=?",
            (academic_year_id, subject))
        return SubjectPeriodAllocation(**dict(r)) if r else None

    def subject_period_allocations_for_year(self, academic_year_id: str) -> list[SubjectPeriodAllocation]:
        rows = self._fetchall(
            "SELECT * FROM subject_period_allocations WHERE academic_year_id=? ORDER BY subject",
            (academic_year_id,))
        return [SubjectPeriodAllocation(**dict(r)) for r in rows]

    def subject_name_for_book(self, book_id: str) -> Optional[str]:
        r = self._fetchone(
            "SELECT s.name FROM books b JOIN subjects s ON b.subject_id = s.id WHERE b.id=?",
            (book_id,))
        return r["name"] if r else None

    def add_timetable_slot(self, *, school_id: str, academic_year_id: str, subject: str,
                           day_of_week: int, period_number: int) -> SubjectTimetableSlot:
        if not (0 <= day_of_week <= 6):
            raise ValueError(f"day_of_week must be 0 (Monday) .. 6 (Sunday), got {day_of_week}")
        slot = SubjectTimetableSlot(id=new_id("slot"), school_id=school_id,
                                    academic_year_id=academic_year_id, subject=subject,
                                    day_of_week=day_of_week, period_number=period_number)
        self._exec(
            "INSERT INTO subject_timetable_slots "
            "(id, school_id, academic_year_id, subject, day_of_week, period_number) VALUES (?,?,?,?,?,?)",
            (slot.id, slot.school_id, slot.academic_year_id, slot.subject,
             slot.day_of_week, slot.period_number))
        self._commit()
        return slot

    def timetable_slots_for_subject(self, academic_year_id: str, subject: str) -> list[SubjectTimetableSlot]:
        rows = self._fetchall(
            "SELECT * FROM subject_timetable_slots WHERE academic_year_id=? AND subject=? "
            "ORDER BY day_of_week, period_number",
            (academic_year_id, subject))
        return [SubjectTimetableSlot(**dict(r)) for r in rows]

    def get_timetable_slot(self, slot_id: str) -> Optional[SubjectTimetableSlot]:
        r = self._fetchone(
            "SELECT * FROM subject_timetable_slots WHERE id=?", (slot_id,))
        return SubjectTimetableSlot(**dict(r)) if r else None

    def remove_timetable_slot(self, slot_id: str) -> None:
        with self._conn_lock:
            cur = self.conn.execute("DELETE FROM subject_timetable_slots WHERE id=?", (slot_id,))
            rowcount = cur.rowcount
        if rowcount == 0:
            raise ValueError(f"no timetable slot with id {slot_id!r}")
        self._commit()

    # ---------------- calendar / holidays ----------------

    def create_calendar(self, *, academic_year_id: str,
                        weekly_off_days: Optional[list[str]] = None,
                        alternate_saturday_rule: str = "none") -> Calendar:
        # `is not None`, not `or`: an explicit [] (a school with no weekly off)
        # used to become ["sunday"] here while update_calendar stored [] --
        # the same request body gave two different calendars by verb.
        c = Calendar(id=new_id("cal"), academic_year_id=academic_year_id,
                    weekly_off_days=weekly_off_days if weekly_off_days is not None else ["sunday"],
                    alternate_saturday_rule=alternate_saturday_rule)
        self._exec(
            "INSERT INTO calendars (id, academic_year_id, weekly_off_days, alternate_saturday_rule) "
            "VALUES (?,?,?,?)",
            (c.id, c.academic_year_id, json.dumps(c.weekly_off_days), c.alternate_saturday_rule))
        self._commit()
        return c

    def get_calendar_for_year(self, academic_year_id: str) -> Optional[Calendar]:
        r = self._fetchone(
            "SELECT * FROM calendars WHERE academic_year_id=?", (academic_year_id,))
        if not r:
            return None
        d = dict(r)
        d["weekly_off_days"] = json.loads(d["weekly_off_days"])
        return Calendar(**d)

    def update_calendar(self, *, academic_year_id: str, weekly_off_days: list[str],
                        alternate_saturday_rule: str) -> Calendar:
        """The correction path: a calendar is one row per academic year
        (create is a 409 the second time), so before this existed a wrong
        weekly off day entered once was stuck for the whole year."""
        with self._conn_lock:
            cur = self.conn.execute(
                "UPDATE calendars SET weekly_off_days=?, alternate_saturday_rule=? "
                "WHERE academic_year_id=?",
                (json.dumps(weekly_off_days), alternate_saturday_rule, academic_year_id))
            rowcount = cur.rowcount
        if rowcount == 0:
            raise ValueError(f"academic year {academic_year_id!r} has no calendar to update")
        self._commit()
        return self.get_calendar_for_year(academic_year_id)

    def get_holiday(self, holiday_id: str) -> Optional[Holiday]:
        r = self._fetchone("SELECT * FROM holidays WHERE id=?", (holiday_id,))
        return Holiday(**dict(r)) if r else None

    def remove_holiday(self, holiday_id: str) -> None:
        with self._conn_lock:
            cur = self.conn.execute("DELETE FROM holidays WHERE id=?", (holiday_id,))
            rowcount = cur.rowcount
        if rowcount == 0:
            raise ValueError(f"no holiday with id {holiday_id!r}")
        self._commit()

    def add_holiday(self, *, calendar_id: str, date: str, label: str, kind: str = "holiday",
                    end_date: Optional[str] = None) -> Holiday:
        if end_date is not None and end_date < date:
            raise ValueError(f"holiday end_date {end_date} is before its date {date}")
        h = Holiday(id=new_id("holiday"), calendar_id=calendar_id, date=date, label=label,
                   kind=kind, end_date=end_date)
        self._exec(
            "INSERT INTO holidays (id, calendar_id, date, label, kind, end_date) VALUES (?,?,?,?,?,?)",
            (h.id, h.calendar_id, h.date, h.label, h.kind, h.end_date))
        self._commit()
        return h

    def holidays_for_calendar(self, calendar_id: str) -> list[Holiday]:
        rows = self._fetchall(
            "SELECT * FROM holidays WHERE calendar_id=? ORDER BY date", (calendar_id,))
        return [Holiday(**dict(r)) for r in rows]

    # ---------------- teaching time estimates ----------------

    def create_teaching_time_estimate(self, *, subtopic_id: str, academic_year_id: str,
                                      estimated_minutes: int,
                                      estimated_periods: Optional[int] = None,
                                      method: str = "admin_override",
                                      approved_by: Optional[str] = None) -> TeachingTimeEstimate:
        t = TeachingTimeEstimate(
            id=new_id("tte"), subtopic_id=subtopic_id, academic_year_id=academic_year_id,
            estimated_minutes=estimated_minutes, estimated_periods=estimated_periods,
            method=method, approved_by=approved_by)
        self._exec(
            "INSERT INTO teaching_time_estimates (id, subtopic_id, academic_year_id, "
            "estimated_minutes, estimated_periods, method, approved_by) VALUES (?,?,?,?,?,?,?)",
            (t.id, t.subtopic_id, t.academic_year_id, t.estimated_minutes,
             t.estimated_periods, t.method, t.approved_by))
        self._commit()
        return t

    def update_teaching_time_estimate(self, estimate_id: str, *, estimated_minutes: int,
                                      estimated_periods: Optional[int], method: str,
                                      approved_by: Optional[str]) -> TeachingTimeEstimate:
        """calendar.compute_teaching_time_estimates(recompute=True) replacing
        its own earlier estimate in place (same id), never an admin's."""
        self._exec(
            "UPDATE teaching_time_estimates SET estimated_minutes=?, estimated_periods=?, method=?, "
            "approved_by=? WHERE id=?",
            (estimated_minutes, estimated_periods, method, approved_by, estimate_id))
        self._commit()
        r = self._fetchone("SELECT * FROM teaching_time_estimates WHERE id=?", (estimate_id,))
        return TeachingTimeEstimate(**dict(r))

    def teaching_time_estimate_for_subtopic(self, subtopic_id: str,
                                            academic_year_id: str) -> Optional[TeachingTimeEstimate]:
        r = self._fetchone(
            "SELECT * FROM teaching_time_estimates WHERE subtopic_id=? AND academic_year_id=?",
            (subtopic_id, academic_year_id))
        return TeachingTimeEstimate(**dict(r)) if r else None

    # ---------------- scheduled lessons (§11-14) ----------------

    def create_scheduled_lesson(self, *, school_id: str, academic_year_id: str, book_id: str,
                                subtopic_id: str, date: str, status: str = "scheduled",
                                section_id: Optional[str] = None) -> ScheduledLesson:
        from datetime import datetime, timezone
        lesson = ScheduledLesson(
            id=new_id("lesson"), school_id=school_id, academic_year_id=academic_year_id,
            book_id=book_id, subtopic_id=subtopic_id, date=date, status=status,
            created_at=datetime.now(timezone.utc).isoformat(), section_id=section_id)
        self._exec(
            "INSERT INTO scheduled_lessons (id, school_id, academic_year_id, book_id, subtopic_id, "
            "date, status, created_at, section_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (lesson.id, lesson.school_id, lesson.academic_year_id, lesson.book_id,
             lesson.subtopic_id, lesson.date, lesson.status, lesson.created_at, lesson.section_id))
        self._commit()
        return lesson

    def get_scheduled_lesson(self, lesson_id: str) -> Optional[ScheduledLesson]:
        r = self._fetchone("SELECT * FROM scheduled_lessons WHERE id=?", (lesson_id,))
        return ScheduledLesson(**dict(r)) if r else None

    def mark_lesson(self, lesson_id: str, *, status: str, note: Optional[str],
                    completed_by: str) -> Optional[ScheduledLesson]:
        """§15: the one write that ever changes a lesson's status/note
        after the scheduler creates it. `completed_by`/`completed_at` are
        set on every call (even a status="scheduled" one, i.e. "undo") so
        there's always a real record of who last touched it -- matching
        approved_by/approved_at's shape elsewhere in this module."""
        if status not in STATUS_VALUES:
            raise ValueError(f"invalid status: {status!r} (must be one of {STATUS_VALUES})")
        from datetime import datetime, timezone
        completed_at = datetime.now(timezone.utc).isoformat()
        self._exec(
            "UPDATE scheduled_lessons SET status=?, note=?, completed_by=?, completed_at=? WHERE id=?",
            (status, note, completed_by, completed_at, lesson_id))
        self._commit()
        return self.get_scheduled_lesson(lesson_id)

    def reschedule_lesson_date(self, lesson_id: str, *, new_date: str,
                               status: Optional[str] = None) -> Optional[ScheduledLesson]:
        """§14: the one write PUSH/ADJUST ever make -- just the date.
        Never touches status/note/completed_by, so a lesson already marked
        completed and then legitimately moved (e.g. corrected after the
        fact) keeps its completion record intact. The real audit trail
        (old date, new date, reason, changed-by, timestamp) is the
        caller's job (scheduling.py), via the existing assessment audit
        log -- this store method only ever changes the one column, plus
        `status` when ADJUST puts an 'unscheduled' lesson back on a day."""
        if status is not None:
            self._exec("UPDATE scheduled_lessons SET date=?, status=? WHERE id=?",
                       (new_date, status, lesson_id))
        else:
            self._exec("UPDATE scheduled_lessons SET date=? WHERE id=?", (new_date, lesson_id))
        self._commit()
        return self.get_scheduled_lesson(lesson_id)

    def mark_lesson_unscheduled(self, lesson_id: str) -> None:
        """A still-to-teach lesson a PUSH could not fit
        before the year ends: it stops holding its day (see models.py's
        STATUS_VALUES note). Its date is left as the last one planned."""
        self._exec("UPDATE scheduled_lessons SET status='unscheduled' WHERE id=? AND status='scheduled'",
                   (lesson_id,))
        self._commit()

    def scheduled_lessons_for_book(self, academic_year_id: str, book_id: str,
                                   section_id: Optional[str] = None) -> list[ScheduledLesson]:
        """One plan's lessons: the book's for `section_id`, or with None the
        school-wide plan made before plans were per section (SCH-4). Date
        order; lessons sharing a day (a double period) in the order they were
        created, which is the delivery order schedule_book() placed them in."""
        rows = self._fetchall(
            "SELECT * FROM scheduled_lessons WHERE academic_year_id=? AND book_id=? AND section_id IS ? "
            "ORDER BY date, rowid",
            (academic_year_id, book_id, section_id))
        return [ScheduledLesson(**dict(r)) for r in rows]

    def scheduled_lessons_for_subtopic(self, subtopic_id: str, academic_year_id: str) -> list[ScheduledLesson]:
        rows = self._fetchall(
            "SELECT * FROM scheduled_lessons WHERE subtopic_id=? AND academic_year_id=? ORDER BY date",
            (subtopic_id, academic_year_id))
        return [ScheduledLesson(**dict(r)) for r in rows]

    def scheduled_lessons_for_date_range(self, school_id: str, start_date: str,
                                         end_date: str) -> list[ScheduledLesson]:
        """The real access pattern a yearly/monthly/weekly/daily view (the
        next milestone) needs -- date-range scoped to one school, not one
        book, since a real day mixes lessons from every subject. An
        'unscheduled' lesson holds no day, so no calendar view shows it on
        the date it last had."""
        rows = self._fetchall(
            "SELECT * FROM scheduled_lessons WHERE school_id=? AND date>=? AND date<=? "
            "AND status<>'unscheduled' ORDER BY date, rowid",
            (school_id, start_date, end_date))
        return [ScheduledLesson(**dict(r)) for r in rows]

    def delete_unrecorded_lessons_for_book(self, academic_year_id: str, book_id: str,
                                           from_date: Optional[str] = None,
                                           section_id: Optional[str] = None) -> int:
        """A force regenerate's delete: only lessons still to be taught
        ('scheduled'/'unscheduled'). Completed and skipped lessons are the
        teaching record and stay; until 2026-09-22 a regenerate deleted
        them with every other row.

        With `from_date`, a 'scheduled' lesson dated before it stays too: its
        day has passed, so it was taught or is overdue, and either way it is
        what delayed-topics, planned-to-date and pace report on. Deleting it
        (the first version of this delete) erased the overdue record of a
        school whose teachers taught but did not tick. Every 'unscheduled'
        lesson goes: a PUSH found it no day, so it is still to be placed."""
        placeholders = ",".join("?" for _ in RECORDED_STATUSES)
        sql = (f"DELETE FROM scheduled_lessons WHERE academic_year_id=? AND book_id=? "
               f"AND section_id IS ? AND status NOT IN ({placeholders})")
        params: tuple = (academic_year_id, book_id, section_id, *RECORDED_STATUSES)
        if from_date is not None:
            sql += " AND (status='unscheduled' OR date >= ?)"
            params += (from_date,)
        with self._conn_lock:
            cur = self.conn.execute(sql, params)
            rowcount = cur.rowcount
        self._commit()
        return rowcount

    def set_book_schedule_cadence(self, academic_year_id: str, book_id: str,
                                  periods_per_week: int, section_id: Optional[str] = None) -> None:
        if section_id is not None:
            self._exec(
                "INSERT INTO section_plan_cadences (academic_year_id, book_id, section_id, "
                "periods_per_week) VALUES (?,?,?,?) ON CONFLICT(academic_year_id, book_id, section_id) "
                "DO UPDATE SET periods_per_week=excluded.periods_per_week",
                (academic_year_id, book_id, section_id, periods_per_week))
            self._commit()
            return
        self._exec(
            "INSERT INTO book_schedule_cadences (academic_year_id, book_id, periods_per_week) "
            "VALUES (?,?,?) ON CONFLICT(academic_year_id, book_id) "
            "DO UPDATE SET periods_per_week=excluded.periods_per_week",
            (academic_year_id, book_id, periods_per_week))
        self._commit()

    def book_schedule_cadence(self, academic_year_id: str, book_id: str,
                              section_id: Optional[str] = None) -> Optional[int]:
        """The periods_per_week this plan was placed with; None for a schedule
        made before 2026-09-22, when it was not recorded."""
        if section_id is not None:
            r = self._fetchone(
                "SELECT periods_per_week FROM section_plan_cadences WHERE academic_year_id=? "
                "AND book_id=? AND section_id=?", (academic_year_id, book_id, section_id))
            return r["periods_per_week"] if r else None
        r = self._fetchone(
            "SELECT periods_per_week FROM book_schedule_cadences WHERE academic_year_id=? AND book_id=?",
            (academic_year_id, book_id))
        return r["periods_per_week"] if r else None

    # ---------------- teacher assignments (§15 -- "what do I teach today") ----------------

    def assign_teacher(self, *, school_id: str, teacher_id: str, book_id: str) -> TeacherAssignment:
        """Idempotent -- assigning the same (teacher, book) pair twice
        returns the existing row rather than erroring or duplicating,
        matching this module's established idempotency posture
        elsewhere (create_teacher_assignment is safe to call from an
        admin UI's "Save" button without a prior existence check)."""
        existing = self._fetchone(
            "SELECT * FROM teacher_assignments WHERE teacher_id=? AND book_id=?",
            (teacher_id, book_id))
        if existing:
            return TeacherAssignment(**dict(existing))
        from datetime import datetime, timezone
        a = TeacherAssignment(id=new_id("ta"), school_id=school_id, teacher_id=teacher_id,
                              book_id=book_id, created_at=datetime.now(timezone.utc).isoformat())
        self._exec(
            "INSERT INTO teacher_assignments (id, school_id, teacher_id, book_id, created_at) "
            "VALUES (?,?,?,?,?)",
            (a.id, a.school_id, a.teacher_id, a.book_id, a.created_at))
        self._commit()
        return a

    def assignments_for_teacher(self, teacher_id: str) -> list[TeacherAssignment]:
        rows = self._fetchall(
            "SELECT * FROM teacher_assignments WHERE teacher_id=? ORDER BY created_at",
            (teacher_id,))
        return [TeacherAssignment(**dict(r)) for r in rows]

    def unassign_teacher(self, *, teacher_id: str, book_id: str) -> None:
        self._exec("DELETE FROM teacher_assignments WHERE teacher_id=? AND book_id=?",
                          (teacher_id, book_id))
        self._commit()

    # ---------------- student enrollment (§18 -- student visibility) ----------------

    def enroll_student(self, *, school_id: str, student_id: str,
                       grade_id: Optional[str] = None,
                       section_id: Optional[str] = None) -> StudentEnrollment:
        """One enrollment per student -- re-enrolling (e.g. a promotion to
        a new grade, or a move from 10-A to 10-B) replaces the existing row
        rather than erroring or leaving two, since a real student is only
        ever in one class at a time.

        The enrollment is in a section (M1.1); the grade is the section's.
        `grade_id` alone still works for a grade with one section, so the
        callers that predate sections keep working; for a grade with
        several it is a ValueError naming them, never a guess. Both given
        must agree. KeyError for an unknown grade or section."""
        from datetime import datetime, timezone
        with self._conn_lock:
            section = self._section_for_enrollment(grade_id, section_id)
            existing = self._fetchone(
                "SELECT * FROM student_enrollments WHERE student_id=?", (student_id,))
            if existing:
                self._exec("UPDATE student_enrollments SET grade_id=?, section_id=?, school_id=? "
                           "WHERE student_id=?",
                           (section.grade_id, section.id, school_id, student_id))
                e = StudentEnrollment(id=existing["id"], school_id=school_id,
                                      student_id=student_id, grade_id=section.grade_id,
                                      created_at=existing["created_at"], section_id=section.id)
            else:
                e = StudentEnrollment(id=new_id("enroll"), school_id=school_id,
                                      student_id=student_id, grade_id=section.grade_id,
                                      created_at=datetime.now(timezone.utc).isoformat(),
                                      section_id=section.id)
                self._exec(
                    "INSERT INTO student_enrollments (id, school_id, student_id, grade_id, "
                    "created_at, section_id) VALUES (?,?,?,?,?,?)",
                    (e.id, e.school_id, e.student_id, e.grade_id, e.created_at, e.section_id))
        self._commit()
        return e

    def _section_for_enrollment(self, grade_id: Optional[str],
                                section_id: Optional[str]) -> Section:
        if section_id is not None:
            section = self.get_section(section_id)
            if section is None:
                raise KeyError(section_id)
            if grade_id is not None and grade_id != section.grade_id:
                raise ValueError("that section is not in that grade")
            return section
        if grade_id is None:
            raise ValueError("say which section the student is in")
        if self.get_grade(grade_id) is None:
            raise KeyError(grade_id)
        sections = self.sections_for_grade(grade_id)
        if len(sections) != 1:
            names = ", ".join(s.name for s in sections) or "none"
            raise ValueError(f"this class has {len(sections)} sections ({names}); "
                             "say which one the student is in")
        return sections[0]

    def enrollment_for_student(self, student_id: str) -> Optional[StudentEnrollment]:
        r = self._fetchone(
            "SELECT * FROM student_enrollments WHERE student_id=?", (student_id,))
        return StudentEnrollment(**dict(r)) if r else None

    # ---------------- management reporting & variance (§17, §32) ----------------

    def get_coverage_report(self, *, school_id: str, academic_year_id: str,
                            as_of_date: Optional[str] = None) -> dict[str, Any]:
        """Planned vs. actually-taught coverage, variance, and completion %
        aggregated by Subject and Chapter for school management (§17, §32).

        Only the subject's chosen edition counts (PRD 12.7). The review of
        fc279a4 ran this against a subject with an old and a new edition and
        got ONE 'Mathematics' row titled after the chosen book whose chapter
        list was ['Chapter new', 'Chapter old'] -- the abandoned edition's
        work attributed to the chosen book on the principal's own Coverage &
        Pace screen. A subject with no recorded choice keeps every book
        (that is the seeded single-book case), and the per-subject key below
        carries the book so two editions can never merge into one row."""
        from datetime import datetime, timezone
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).date().isoformat()

        rows = self._fetchall(
            """
            SELECT l.id, l.date, l.status, l.subtopic_id,
                   st.name as subtopic_name, tp.id as topic_id, tp.name as topic_name,
                   ch.id as chapter_id, ch.name as chapter_name,
                   b.id as book_id, b.title as book_title,
                   s.id as subject_id, s.name as subject_name,
                   g.id as grade_id, g.number as grade_number,
                   l.section_id as section_id, sec.name as section_name
            FROM scheduled_lessons l
            LEFT JOIN sections sec ON l.section_id = sec.id
            JOIN subtopics st ON l.subtopic_id = st.id
            JOIN topics tp ON st.topic_id = tp.id
            JOIN chapters ch ON tp.chapter_id = ch.id
            JOIN books b ON l.book_id = b.id
            JOIN subjects s ON b.subject_id = s.id
            JOIN grades g ON s.grade_id = g.id
            WHERE l.school_id = ? AND l.academic_year_id = ?
              AND (l.book_id = (SELECT sel.book_id FROM book_selections sel
                                 WHERE sel.academic_year_id = l.academic_year_id
                                   AND sel.subject_id = s.id)
                   OR NOT EXISTS (SELECT 1 FROM book_selections sel
                                   WHERE sel.academic_year_id = l.academic_year_id
                                     AND sel.subject_id = s.id))
            ORDER BY g.number, s.name, ch.name, l.date
            """,
            (school_id, academic_year_id),
        )

        assignment_rows = self._fetchall(
            "SELECT teacher_id, book_id FROM teacher_assignments WHERE school_id=?",
            (school_id,),
        )
        teacher_for_book = {r["book_id"]: r["teacher_id"] for r in assignment_rows}
        teacher_for_cell = {(a["section_id"], a["subject_id"]): a["teacher_id"] for a in self._fetchall(
            "SELECT section_id, subject_id, teacher_id FROM teaching_allocations WHERE academic_year_id=?",
            (academic_year_id,))}

        # A section's own plan (SCH-4) is its own row, keyed with the section:
        # otherwise two sections' lessons would merge and count twice.
        # Keyed on (subject, book), not the subject alone: an edition the
        # school did not choose -- possible only with a hand-written books
        # row, since the filter above keeps just the chosen one -- reports
        # under its own title instead of being folded into another book's.
        subjects_map: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            sid = (r["subject_id"], r["book_id"], r["section_id"])
            if sid not in subjects_map:
                subjects_map[sid] = {
                    "subject_id": r["subject_id"],
                    "subject_name": r["subject_name"],
                    "grade_number": r["grade_number"],
                    "book_id": r["book_id"],
                    "book_title": r["book_title"],
                    "section_id": r["section_id"],
                    "section_name": r["section_name"],
                    "teacher_id": (teacher_for_cell.get((r["section_id"], r["subject_id"]))
                                   if r["section_id"] else teacher_for_book.get(r["book_id"])),
                    "lessons": [],
                    "chapters": {},
                }
            cid = r["chapter_id"]
            if cid not in subjects_map[sid]["chapters"]:
                subjects_map[sid]["chapters"][cid] = {
                    "chapter_id": cid,
                    "chapter_name": r["chapter_name"],
                    "lessons": [],
                }
            subjects_map[sid]["lessons"].append(r)
            subjects_map[sid]["chapters"][cid]["lessons"].append(r)

        total_lessons = len(rows)
        completed_lessons = sum(1 for r in rows if r["status"] == "completed")
        skipped_lessons = sum(1 for r in rows if r["status"] == "skipped")
        # An 'unscheduled' lesson has no day in the plan, so it is never "planned to date".
        planned_to_date = sum(1 for r in rows if r["date"] <= as_of_date and r["status"] != "unscheduled")
        completed_to_date = sum(1 for r in rows if r["date"] <= as_of_date and r["status"] == "completed")

        overall_coverage_pct = round((completed_lessons / total_lessons * 100), 2) if total_lessons > 0 else 0.0
        overall_pace_pct = round((completed_to_date / planned_to_date * 100), 2) if planned_to_date > 0 else 100.0
        overall_variance = completed_to_date - planned_to_date

        subjects_out = []
        for sdata in subjects_map.values():
            s_lessons = sdata["lessons"]
            s_total = len(s_lessons)
            s_completed = sum(1 for l in s_lessons if l["status"] == "completed")
            s_skipped = sum(1 for l in s_lessons if l["status"] == "skipped")
            s_planned_to_date = sum(1 for l in s_lessons
                                    if l["date"] <= as_of_date and l["status"] != "unscheduled")
            s_completed_to_date = sum(1 for l in s_lessons if l["date"] <= as_of_date and l["status"] == "completed")
            s_coverage_pct = round((s_completed / s_total * 100), 2) if s_total > 0 else 0.0
            s_pace_pct = round((s_completed_to_date / s_planned_to_date * 100), 2) if s_planned_to_date > 0 else 100.0
            s_variance = s_completed_to_date - s_planned_to_date

            chapters_out = []
            for cid, cdata in sdata["chapters"].items():
                c_lessons = cdata["lessons"]
                c_total = len(c_lessons)
                c_completed = sum(1 for l in c_lessons if l["status"] == "completed")
                c_skipped = sum(1 for l in c_lessons if l["status"] == "skipped")
                c_cov = round((c_completed / c_total * 100), 2) if c_total > 0 else 0.0
                chapters_out.append({
                    "chapter_id": cid,
                    "chapter_name": cdata["chapter_name"],
                    "total_lessons": c_total,
                    "completed_lessons": c_completed,
                    "skipped_lessons": c_skipped,
                    "coverage_pct": c_cov,
                })

            subjects_out.append({
                "subject_id": sdata["subject_id"],
                "subject_name": sdata["subject_name"],
                "grade_number": sdata["grade_number"],
                "book_id": sdata["book_id"],
                "book_title": sdata["book_title"],
                "section_id": sdata["section_id"],
                "section_name": sdata["section_name"],
                "teacher_id": sdata["teacher_id"],
                "teacher_name": None,
                "total_lessons": s_total,
                "completed_lessons": s_completed,
                "skipped_lessons": s_skipped,
                "planned_to_date": s_planned_to_date,
                "completed_to_date": s_completed_to_date,
                "coverage_pct": s_coverage_pct,
                "pace_pct": s_pace_pct,
                "variance": s_variance,
                "chapters": chapters_out,
            })

        return {
            "school_id": school_id,
            "academic_year_id": academic_year_id,
            "as_of_date": as_of_date,
            "total_lessons": total_lessons,
            "completed_lessons": completed_lessons,
            "skipped_lessons": skipped_lessons,
            "planned_to_date": planned_to_date,
            "completed_to_date": completed_to_date,
            "overall_coverage_pct": overall_coverage_pct,
            "overall_pace_pct": overall_pace_pct,
            "overall_variance": overall_variance,
            "subjects": subjects_out,
        }

    def get_delayed_topics(self, *, school_id: str, academic_year_id: str,
                           as_of_date: Optional[str] = None) -> dict[str, Any]:
        """All scheduled lessons past due (date < as_of_date) still in
        'scheduled' status (§17, §32), and separately every 'unscheduled'
        lesson (a PUSH found it no period before the year ends), whatever
        its stale date: a lesson that lost its day is the worst delay, and
        until the review of 2026-09-22 the `status = 'scheduled'` filter
        kept it off this report entirely."""
        from datetime import date, datetime, timezone
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).date().isoformat()
        as_of = date.fromisoformat(as_of_date)

        select = """
            SELECT l.id as lesson_id, l.date as scheduled_date, l.subtopic_id,
                   st.name as subtopic_name, tp.name as topic_name,
                   ch.name as chapter_name, s.name as subject_name,
                   g.number as grade_number, b.id as book_id, b.title as book_title,
                   s.id as subject_id, l.section_id as section_id, sec.name as section_name
            FROM scheduled_lessons l
            LEFT JOIN sections sec ON l.section_id = sec.id
            JOIN subtopics st ON l.subtopic_id = st.id
            JOIN topics tp ON st.topic_id = tp.id
            JOIN chapters ch ON tp.chapter_id = ch.id
            JOIN books b ON l.book_id = b.id
            JOIN subjects s ON b.subject_id = s.id
            JOIN grades g ON s.grade_id = g.id
            WHERE l.school_id = ? AND l.academic_year_id = ? AND {cond}
            ORDER BY l.date, g.number, s.name
            """
        rows = self._fetchall(select.format(cond="l.date < ? AND l.status = 'scheduled'"),
                              (school_id, academic_year_id, as_of_date))
        unscheduled_rows = self._fetchall(select.format(cond="l.status = 'unscheduled'"),
                                          (school_id, academic_year_id))

        assignment_rows = self._fetchall(
            "SELECT teacher_id, book_id FROM teacher_assignments WHERE school_id=?",
            (school_id,),
        )
        teacher_for_book = {r["book_id"]: r["teacher_id"] for r in assignment_rows}
        teacher_for_cell = {(a["section_id"], a["subject_id"]): a["teacher_id"] for a in self._fetchall(
            "SELECT section_id, subject_id, teacher_id FROM teaching_allocations WHERE academic_year_id=?",
            (academic_year_id,))}

        def _teacher(r) -> Optional[str]:
            if r["section_id"]:
                return teacher_for_cell.get((r["section_id"], r["subject_id"]))
            return teacher_for_book.get(r["book_id"])

        # An edition the school stopped teaching keeps its overdue lessons
        # (nothing deletes them), and the row said nothing about which book it
        # came from: a principal chased work on a book nobody teaches. Coverage
        # drops the old edition; this report names it instead, because an
        # overdue lesson is a fact about the past either way.
        chosen_cache: dict[str, Optional[str]] = {}

        def _chosen(subject_id: str) -> Optional[str]:
            if subject_id not in chosen_cache:
                book = self.selected_book_for_subject(subject_id)
                chosen_cache[subject_id] = book.id if book else None
            return chosen_cache[subject_id]

        delayed = []
        for r in rows:
            sched_d = date.fromisoformat(r["scheduled_date"])
            days_overdue = (as_of - sched_d).days
            delayed.append({
                "lesson_id": r["lesson_id"],
                "scheduled_date": r["scheduled_date"],
                "days_overdue": days_overdue,
                "grade_number": r["grade_number"],
                "subject_name": r["subject_name"],
                "chapter_name": r["chapter_name"],
                "topic_name": r["topic_name"],
                "subtopic_name": r["subtopic_name"],
                "book_id": r["book_id"],
                "book_title": r["book_title"],
                "is_chosen_edition": _chosen(r["subject_id"]) == r["book_id"],
                "section_id": r["section_id"],
                "section_name": r["section_name"],
                "teacher_id": _teacher(r),
                "teacher_name": None,
            })

        unscheduled = [{
            "lesson_id": r["lesson_id"],
            "last_planned_date": r["scheduled_date"],
            "book_id": r["book_id"],
            "book_title": r["book_title"],
            "is_chosen_edition": _chosen(r["subject_id"]) == r["book_id"],
            "grade_number": r["grade_number"],
            "subject_name": r["subject_name"],
            "chapter_name": r["chapter_name"],
            "topic_name": r["topic_name"],
            "subtopic_name": r["subtopic_name"],
            "section_id": r["section_id"],
            "section_name": r["section_name"],
            "teacher_id": _teacher(r),
            "teacher_name": None,
        } for r in unscheduled_rows]

        return {
            "school_id": school_id,
            "academic_year_id": academic_year_id,
            "as_of_date": as_of_date,
            "delayed_count": len(delayed),
            "delayed_lessons": delayed,
            "unscheduled_count": len(unscheduled),
            "unscheduled_lessons": unscheduled,
        }


_INSTANCE: Optional[dict] = {}


def get_curriculum_store(data_root: Path) -> CurriculumStore:
    # Keyed on the RESOLVED PATH, not a bare global.
    #
    # This was `if _INSTANCE is None`, so the FIRST data_root ever passed won
    # for the life of the process and every later call with a different root got
    # that first instance. In production only one root is used, which is why it
    # survived unspotted -- but anywhere a process touches more than one root
    # (tests, a CLI run beside a server, tooling) the store silently reads and
    # writes the WRONG database. For the audit log that means a compliance trail
    # filed against another data root.
    # `_INSTANCE` is a PATH-KEYED cache, not one store. Tests reset it with
    # `monkeypatch.setattr(..., "_INSTANCE", None)`, which is kept working,
    # but the reset is no longer load-bearing: two data roots now get two
    # stores by construction. See git history for the single-global version.
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = {}                      # a caller cleared the cache
    key = str(Path(data_root) / "curriculum" / "curriculum.sqlite")
    instance = _INSTANCE.get(key)
    if instance is None:
        instance = CurriculumStore(Path(key))
        # A service that started empty saves that empty curriculum now, so the
        # next deploy restores it instead of refusing to start.
        instance._snapshots.save_empty_boot(instance.conn)
        _INSTANCE[key] = instance
    return instance
