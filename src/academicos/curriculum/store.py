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

Known gap this pass does NOT fix: like every other SQLite store in this
codebase before EventStore's fix earlier in this pass, CurriculumStore's
methods call `self.conn.execute(...)` without any lock protecting the
shared `sqlite3.Connection` from concurrent access by two threads at once
-- a real latent risk under true multi-teacher concurrent writes, not
introduced by this change but not fixed by it either. `_commit()` below
takes `self._conn_lock` around its own checkpoint+read (so a snapshot
can't race with itself), but the other 30+ pre-existing call sites in this
file are unguarded, same as before this pass. Flagged here rather than
silently left undocumented; hardening every call site is a larger,
separate change.

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
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

from ..assessment.supabase_kv import SupabaseStorage

from .models import (
    AcademicYear,
    Board,
    Calendar,
    Chapter,
    CurriculumExtractionProposal,
    CurriculumExtractionRun,
    Grade,
    Holiday,
    PeriodConfiguration,
    SubjectPeriodAllocation,
    SubjectTimetableSlot,
    QuestionSubtopicLink,
    STATUS_VALUES,
    ScheduledLesson,
    StudentEnrollment,
    Subject,
    Subtopic,
    TeacherAssignment,
    TeachingTimeEstimate,
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

CREATE TABLE IF NOT EXISTS grades (
  id           TEXT PRIMARY KEY,
  academic_year_id TEXT NOT NULL,
  number       INTEGER NOT NULL,
  section      TEXT
);
CREATE INDEX IF NOT EXISTS idx_grades_year ON grades(academic_year_id);

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
  created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_se_grade ON student_enrollments(grade_id);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class CurriculumStore:
    _SNAPSHOT_KEY = "curriculum.sqlite"
    _SNAPSHOT_DEBOUNCE_SECONDS = 30.0

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._remote_storage = SupabaseStorage("curriculum-snapshots")
        self._conn_lock = threading.Lock()
        self._last_snapshot_at = 0.0

        db_path.parent.mkdir(parents=True, exist_ok=True)
        if not db_path.exists() and self._remote_storage.enabled:
            self._restore_from_remote()

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self._commit()

    def _migrate(self) -> None:
        """CREATE TABLE IF NOT EXISTS never adds a column to a table that
        already exists on disk -- unlike scheduled_lessons' note/completed_by
        columns (added when zero real rows existed yet, so no migration was
        needed), school_1 already has real seeded holiday rows by the time
        `end_date` was added, so a real ALTER TABLE is required here."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(holidays)").fetchall()}
        if "end_date" not in cols:
            self.conn.execute("ALTER TABLE holidays ADD COLUMN end_date TEXT")

    def _restore_from_remote(self) -> None:
        """Called only when db_path doesn't exist locally yet and Supabase
        is configured -- a fresh container restoring last known state
        instead of starting empty. No snapshot yet (first-ever boot) is the
        normal, expected case, not an error: falls through to a fresh
        CREATE TABLE against an empty file exactly like before this pass."""
        try:
            data = self._remote_storage.download(self._SNAPSHOT_KEY)
        except requests.exceptions.RequestException:
            logger.info(
                "No curriculum snapshot restored from Supabase (none uploaded yet, "
                "or unreachable) -- starting with a fresh local database",
                exc_info=True,
            )
            return
        self.db_path.write_bytes(data)
        logger.info("Restored CurriculumStore from Supabase snapshot (%d bytes)", len(data))

    def _commit(self) -> None:
        self.conn.commit()
        self._maybe_snapshot()

    def _maybe_snapshot(self) -> None:
        if not self._remote_storage.enabled:
            return
        now = time.monotonic()
        with self._conn_lock:
            if now - self._last_snapshot_at < self._SNAPSHOT_DEBOUNCE_SECONDS:
                return
            self._last_snapshot_at = now
            try:
                # WAL mode means recent commits can still live only in the
                # -wal sidecar file -- checkpoint first so db_path itself is
                # a complete, self-contained snapshot, not a stale main file
                # missing whatever hasn't been checkpointed back into it yet.
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                data = self.db_path.read_bytes()
            except OSError:
                logger.warning("Could not read CurriculumStore db file to snapshot", exc_info=True)
                return
        try:
            self._remote_storage.upload(self._SNAPSHOT_KEY, data, "application/x-sqlite3")
        except requests.exceptions.RequestException:
            logger.warning("Failed to upload CurriculumStore snapshot to Supabase", exc_info=True)

    def close(self) -> None:
        self.conn.close()

    # ---------------- boards ----------------

    def create_board(self, name: str, code: str) -> Board:
        b = Board(id=new_id("board"), name=name, code=code)
        self.conn.execute("INSERT INTO boards (id, name, code) VALUES (?,?,?)",
                          (b.id, b.name, b.code))
        self._commit()
        return b

    def get_board(self, board_id: str) -> Optional[Board]:
        r = self.conn.execute("SELECT * FROM boards WHERE id=?", (board_id,)).fetchone()
        return Board(**dict(r)) if r else None

    def get_board_by_code(self, code: str) -> Optional[Board]:
        r = self.conn.execute("SELECT * FROM boards WHERE code=?", (code,)).fetchone()
        return Board(**dict(r)) if r else None

    def list_boards(self) -> list[Board]:
        rows = self.conn.execute("SELECT * FROM boards").fetchall()
        return [Board(**dict(r)) for r in rows]

    # ---------------- academic years ----------------

    def create_academic_year(self, *, school_id: str, label: str, start_date: str,
                             end_date: str, status: str = "draft") -> AcademicYear:
        y = AcademicYear(id=new_id("year"), school_id=school_id, label=label,
                         start_date=start_date, end_date=end_date, status=status)
        self.conn.execute(
            "INSERT INTO academic_years (id, school_id, label, start_date, end_date, status) "
            "VALUES (?,?,?,?,?,?)",
            (y.id, y.school_id, y.label, y.start_date, y.end_date, y.status))
        self._commit()
        return y

    def get_academic_year(self, year_id: str) -> Optional[AcademicYear]:
        r = self.conn.execute("SELECT * FROM academic_years WHERE id=?", (year_id,)).fetchone()
        return AcademicYear(**dict(r)) if r else None

    def academic_years_for_school(self, school_id: str) -> list[AcademicYear]:
        rows = self.conn.execute(
            "SELECT * FROM academic_years WHERE school_id=? ORDER BY start_date", (school_id,)).fetchall()
        return [AcademicYear(**dict(r)) for r in rows]

    def get_academic_year_by_label(self, school_id: str, label: str) -> Optional[AcademicYear]:
        r = self.conn.execute(
            "SELECT * FROM academic_years WHERE school_id=? AND label=?", (school_id, label)).fetchone()
        return AcademicYear(**dict(r)) if r else None

    # ---------------- grades ----------------

    def create_grade(self, *, academic_year_id: str, number: int,
                     section: Optional[str] = None) -> Grade:
        g = Grade(id=new_id("grade"), academic_year_id=academic_year_id, number=number, section=section)
        self.conn.execute(
            "INSERT INTO grades (id, academic_year_id, number, section) VALUES (?,?,?,?)",
            (g.id, g.academic_year_id, g.number, g.section))
        self._commit()
        return g

    def get_grade(self, grade_id: str) -> Optional[Grade]:
        r = self.conn.execute("SELECT * FROM grades WHERE id=?", (grade_id,)).fetchone()
        return Grade(**dict(r)) if r else None

    def grades_for_year(self, academic_year_id: str) -> list[Grade]:
        rows = self.conn.execute(
            "SELECT * FROM grades WHERE academic_year_id=? ORDER BY number", (academic_year_id,)).fetchall()
        return [Grade(**dict(r)) for r in rows]

    def get_grade_by_number(self, academic_year_id: str, number: int,
                            section: Optional[str] = None) -> Optional[Grade]:
        # SQLite's IS operator handles a bound NULL parameter correctly
        # (unlike =, which never matches NULL) -- one clause covers both
        # "no section" (section is None) and a real section value.
        r = self.conn.execute(
            "SELECT * FROM grades WHERE academic_year_id=? AND number=? AND section IS ?",
            (academic_year_id, number, section)).fetchone()
        return Grade(**dict(r)) if r else None

    # ---------------- subjects ----------------

    def create_subject(self, *, grade_id: str, name: str, code: Optional[str] = None) -> Subject:
        s = Subject(id=new_id("subj"), grade_id=grade_id, name=name, code=code)
        self.conn.execute("INSERT INTO subjects (id, grade_id, name, code) VALUES (?,?,?,?)",
                          (s.id, s.grade_id, s.name, s.code))
        self._commit()
        return s

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        r = self.conn.execute("SELECT * FROM subjects WHERE id=?", (subject_id,)).fetchone()
        return Subject(**dict(r)) if r else None

    def subjects_for_grade(self, grade_id: str) -> list[Subject]:
        rows = self.conn.execute("SELECT * FROM subjects WHERE grade_id=? ORDER BY name", (grade_id,)).fetchall()
        return [Subject(**dict(r)) for r in rows]

    def get_subject_by_name(self, grade_id: str, name: str) -> Optional[Subject]:
        r = self.conn.execute(
            "SELECT * FROM subjects WHERE grade_id=? AND name=?", (grade_id, name)).fetchone()
        return Subject(**dict(r)) if r else None

    # ---------------- books ----------------

    def create_book(self, *, subject_id: str, board_id: str, title: str,
                    publisher: Optional[str] = None,
                    source_doc_ids: Optional[list[str]] = None,
                    status: str = "selected") -> "Any":
        from .models import Book
        b = Book(id=new_id("book"), subject_id=subject_id, board_id=board_id, title=title,
                publisher=publisher, source_doc_ids=source_doc_ids or [], status=status)
        self.conn.execute(
            "INSERT INTO books (id, subject_id, board_id, title, publisher, source_doc_ids, status) "
            "VALUES (?,?,?,?,?,?,?)",
            (b.id, b.subject_id, b.board_id, b.title, b.publisher,
             json.dumps(b.source_doc_ids), b.status))
        self._commit()
        return b

    def get_book(self, book_id: str):
        from .models import Book
        r = self.conn.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["source_doc_ids"] = json.loads(d["source_doc_ids"] or "[]")
        return Book(**d)

    def books_for_subject(self, subject_id: str) -> list["Any"]:
        rows = self.conn.execute("SELECT * FROM books WHERE subject_id=?", (subject_id,)).fetchall()
        out = []
        from .models import Book
        for r in rows:
            d = dict(r)
            d["source_doc_ids"] = json.loads(d["source_doc_ids"] or "[]")
            out.append(Book(**d))
        return out

    def get_book_by_title(self, subject_id: str, title: str) -> Optional["Any"]:
        from .models import Book
        r = self.conn.execute(
            "SELECT * FROM books WHERE subject_id=? AND title=?", (subject_id, title)).fetchone()
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
        self.conn.execute(
            "INSERT INTO units (id, canonical_id, book_id, unit_no, name, marks, seq) "
            "VALUES (?,?,?,?,?,?,?)",
            (u.id, u.canonical_id, u.book_id, u.unit_no, u.name, u.marks, u.seq))
        self._commit()
        return u

    def get_unit(self, unit_id: str) -> Optional[Unit]:
        r = self.conn.execute("SELECT * FROM units WHERE id=?", (unit_id,)).fetchone()
        return Unit(**dict(r)) if r else None

    def get_unit_by_canonical_id(self, canonical_id: str) -> Optional[Unit]:
        r = self.conn.execute("SELECT * FROM units WHERE canonical_id=?", (canonical_id,)).fetchone()
        return Unit(**dict(r)) if r else None

    def units_for_book(self, book_id: str) -> list[Unit]:
        rows = self.conn.execute("SELECT * FROM units WHERE book_id=? ORDER BY seq", (book_id,)).fetchall()
        return [Unit(**dict(r)) for r in rows]

    # ---------------- chapters ----------------

    def create_chapter(self, *, canonical_id: str, unit_id: str, name: str, seq: int = 0) -> Chapter:
        c = Chapter(id=new_id("chap"), canonical_id=canonical_id, unit_id=unit_id, name=name, seq=seq)
        self.conn.execute(
            "INSERT INTO chapters (id, canonical_id, unit_id, name, seq) VALUES (?,?,?,?,?)",
            (c.id, c.canonical_id, c.unit_id, c.name, c.seq))
        self._commit()
        return c

    def get_chapter(self, chapter_id: str) -> Optional[Chapter]:
        r = self.conn.execute("SELECT * FROM chapters WHERE id=?", (chapter_id,)).fetchone()
        return Chapter(**dict(r)) if r else None

    def get_chapter_by_canonical_id(self, canonical_id: str) -> Optional[Chapter]:
        r = self.conn.execute("SELECT * FROM chapters WHERE canonical_id=?", (canonical_id,)).fetchone()
        return Chapter(**dict(r)) if r else None

    def school_id_for_book(self, book_id: str) -> Optional[str]:
        """Ownership-chain lookup (book -> subject -> grade -> academic_year
        -> school_id) -- what every write endpoint in routes.py uses to
        verify a caller's school actually owns the book/chapter they're
        trying to touch, the same school-scoping pattern this session
        already applied to pillar_routes.py's authorization fixes."""
        r = self.conn.execute(
            "SELECT y.school_id FROM books b "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE b.id=?", (book_id,)).fetchone()
        return r["school_id"] if r else None

    def school_id_for_grade(self, grade_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT y.school_id FROM grades g JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE g.id=?", (grade_id,)).fetchone()
        return r["school_id"] if r else None

    def grade_id_for_book(self, book_id: str) -> Optional[str]:
        """book -> subject -> grade -- what a student's schedule filters
        by: every book taught to their real enrolled grade, not just one
        subject (unlike a teacher, who is scoped to specific books via
        TeacherAssignment)."""
        r = self.conn.execute(
            "SELECT s.grade_id FROM books b JOIN subjects s ON b.subject_id = s.id WHERE b.id=?",
            (book_id,)).fetchone()
        return r["grade_id"] if r else None

    def book_ids_for_grade(self, grade_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT b.id FROM books b JOIN subjects s ON b.subject_id = s.id WHERE s.grade_id=?",
            (grade_id,)).fetchall()
        return [r["id"] for r in rows]

    def school_id_for_unit(self, unit_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT y.school_id FROM units u "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE u.id=?", (unit_id,)).fetchone()
        return r["school_id"] if r else None

    def school_id_for_chapter(self, chapter_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT y.school_id FROM chapters c "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE c.id=?", (chapter_id,)).fetchone()
        return r["school_id"] if r else None

    def school_id_for_topic(self, topic_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT y.school_id FROM topics t "
            "JOIN chapters c ON t.chapter_id = c.id "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE t.id=?", (topic_id,)).fetchone()
        return r["school_id"] if r else None

    def school_id_for_subtopic(self, subtopic_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT y.school_id FROM subtopics st "
            "JOIN topics t ON st.topic_id = t.id "
            "JOIN chapters c ON t.chapter_id = c.id "
            "JOIN units u ON c.unit_id = u.id "
            "JOIN books b ON u.book_id = b.id "
            "JOIN subjects s ON b.subject_id = s.id "
            "JOIN grades g ON s.grade_id = g.id "
            "JOIN academic_years y ON g.academic_year_id = y.id "
            "WHERE st.id=?", (subtopic_id,)).fetchone()
        return r["school_id"] if r else None

    _SEQUENCE_TABLES = {"unit": "units", "chapter": "chapters", "topic": "topics", "subtopic": "subtopics"}

    def set_sequence(self, entity_type: str, entity_id: str, seq: int) -> None:
        """Changes one Unit/Chapter/Topic/Subtopic's delivery-order `seq` --
        the field `chapters_for_unit`/`topics_for_chapter`/
        `subtopics_for_topic`/`units_for_book` all `ORDER BY`, and the field
        `scheduling.py::schedule_book()` walks in that same order (unit ->
        chapter -> topic -> subtopic) to decide what gets taught when. A
        change here takes effect on the *next* `schedule_book()` call --
        it does not retroactively touch any `ScheduledLesson` rows a
        previous run already created (see the matrix's own open item on
        re-running/flagging schedules stale after a reorder).

        Unlike rename_topic/rename_subtopic's silent no-op on an unknown id,
        this raises: a reorder is a deliberate, meaningful admin action, and
        a caller reordering something that doesn't exist deserves a real
        error, not quiet success."""
        table = self._SEQUENCE_TABLES.get(entity_type)
        if table is None:
            raise ValueError(f"unknown sequence entity_type: {entity_type!r}")
        cur = self.conn.execute(f"UPDATE {table} SET seq=? WHERE id=?", (seq, entity_id))
        if cur.rowcount == 0:
            raise ValueError(f"no {entity_type} with id {entity_id!r}")
        self._commit()

    def chapters_for_unit(self, unit_id: str) -> list[Chapter]:
        rows = self.conn.execute("SELECT * FROM chapters WHERE unit_id=? ORDER BY seq", (unit_id,)).fetchall()
        return [Chapter(**dict(r)) for r in rows]

    def chapters_for_book(self, book_id: str) -> list[Chapter]:
        """Convenience join: every chapter across every unit of a book, in
        unit-then-chapter seq order -- what an admin's curriculum review
        screen (§29) actually wants to show."""
        rows = self.conn.execute(
            "SELECT c.* FROM chapters c JOIN units u ON c.unit_id = u.id "
            "WHERE u.book_id=? ORDER BY u.seq, c.seq", (book_id,)).fetchall()
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
        self.conn.execute(
            "INSERT INTO topics (id, canonical_id, chapter_id, name, seq, description, "
            "source_type, source_reference, approved_by, approved_at, model_used, "
            "generation_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.id, t.canonical_id, t.chapter_id, t.name, t.seq, t.description, t.source_type,
             t.source_reference, t.approved_by, t.approved_at, t.model_used, t.generation_version))
        self._commit()
        return t

    def get_topic(self, topic_id: str) -> Optional[Topic]:
        r = self.conn.execute("SELECT * FROM topics WHERE id=?", (topic_id,)).fetchone()
        return Topic(**dict(r)) if r else None

    def topics_for_chapter(self, chapter_id: str) -> list[Topic]:
        rows = self.conn.execute("SELECT * FROM topics WHERE chapter_id=? ORDER BY seq", (chapter_id,)).fetchall()
        return [Topic(**dict(r)) for r in rows]

    def rename_topic(self, topic_id: str, new_name: str) -> None:
        """§ acceptance criteria: renaming a topic must not break question
        references -- canonical_id (what qmap.py/QuestionSubtopicLink key
        against) is untouched by this; only the display name changes."""
        self.conn.execute("UPDATE topics SET name=? WHERE id=?", (new_name, topic_id))
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
        self.conn.execute(
            "INSERT INTO subtopics (id, canonical_id, topic_id, name, seq, description, "
            "source_type, source_reference, approved_by, approved_at, model_used, "
            "generation_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.id, s.canonical_id, s.topic_id, s.name, s.seq, s.description, s.source_type,
             s.source_reference, s.approved_by, s.approved_at, s.model_used, s.generation_version))
        self._commit()
        return s

    def get_subtopic(self, subtopic_id: str) -> Optional[Subtopic]:
        r = self.conn.execute("SELECT * FROM subtopics WHERE id=?", (subtopic_id,)).fetchone()
        return Subtopic(**dict(r)) if r else None

    def get_subtopic_by_canonical_id(self, canonical_id: str) -> Optional[Subtopic]:
        r = self.conn.execute("SELECT * FROM subtopics WHERE canonical_id=?", (canonical_id,)).fetchone()
        return Subtopic(**dict(r)) if r else None

    def subtopics_for_topic(self, topic_id: str) -> list[Subtopic]:
        rows = self.conn.execute("SELECT * FROM subtopics WHERE topic_id=? ORDER BY seq", (topic_id,)).fetchall()
        return [Subtopic(**dict(r)) for r in rows]

    def rename_subtopic(self, subtopic_id: str, new_name: str) -> None:
        self.conn.execute("UPDATE subtopics SET name=? WHERE id=?", (new_name, subtopic_id))
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
            self.conn.execute("DELETE FROM question_subtopic_links WHERE subtopic_id=?", (subtopic_id,))
        self.conn.execute("DELETE FROM subtopics WHERE id=?", (subtopic_id,))
        self._commit()

    # ---------------- question <-> subtopic links ----------------

    def link_question_to_subtopic(self, *, question_id: str, subtopic_id: str,
                                  method: str = "lexical", confidence: float = 0.5) -> QuestionSubtopicLink:
        link = QuestionSubtopicLink(id=new_id("qsl"), question_id=question_id, subtopic_id=subtopic_id,
                                    method=method, confidence=confidence)
        # A question can be re-tagged (new method/confidence) without
        # accumulating duplicate rows for the same (question, subtopic) pair.
        self.conn.execute(
            "INSERT INTO question_subtopic_links (id, question_id, subtopic_id, method, confidence) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(question_id, subtopic_id) DO UPDATE SET method=excluded.method, "
            "confidence=excluded.confidence",
            (link.id, link.question_id, link.subtopic_id, link.method, link.confidence))
        self._commit()
        return link

    def subtopics_for_question(self, question_id: str) -> list[QuestionSubtopicLink]:
        rows = self.conn.execute(
            "SELECT * FROM question_subtopic_links WHERE question_id=?", (question_id,)).fetchall()
        return [QuestionSubtopicLink(**dict(r)) for r in rows]

    def question_links_for_subtopic(self, subtopic_id: str) -> list[QuestionSubtopicLink]:
        rows = self.conn.execute(
            "SELECT * FROM question_subtopic_links WHERE subtopic_id=?", (subtopic_id,)).fetchall()
        return [QuestionSubtopicLink(**dict(r)) for r in rows]

    def question_ids_for_subtopics(self, subtopic_ids: list[str]) -> list[str]:
        """The real "generate a paper from these subtopics" query: every
        question tagged (via qmap.py's resolution -> link_question_to_subtopic)
        to any of the given subtopics."""
        if not subtopic_ids:
            return []
        placeholders = ",".join("?" * len(subtopic_ids))
        rows = self.conn.execute(
            f"SELECT DISTINCT question_id FROM question_subtopic_links "
            f"WHERE subtopic_id IN ({placeholders})", subtopic_ids).fetchall()
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
        self.conn.execute(
            "INSERT INTO curriculum_extraction_runs (id, school_id, book_id, chapter_id, "
            "source_hash, model, prompt_version, created_at, status) VALUES (?,?,?,?,?,?,?,?,?)",
            (run.id, run.school_id, run.book_id, run.chapter_id, run.source_hash, run.model,
             run.prompt_version, run.created_at, run.status))
        self._commit()
        return run

    def get_extraction_run(self, run_id: str) -> Optional[CurriculumExtractionRun]:
        r = self.conn.execute(
            "SELECT * FROM curriculum_extraction_runs WHERE id=?", (run_id,)).fetchone()
        return CurriculumExtractionRun(**dict(r)) if r else None

    def extraction_runs_for_chapter(self, chapter_id: str) -> list[CurriculumExtractionRun]:
        rows = self.conn.execute(
            "SELECT * FROM curriculum_extraction_runs WHERE chapter_id=? ORDER BY created_at DESC",
            (chapter_id,)).fetchall()
        return [CurriculumExtractionRun(**dict(r)) for r in rows]

    def update_run_status(self, run_id: str, status: str) -> None:
        self.conn.execute(
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
        self.conn.execute(
            "INSERT INTO curriculum_extraction_proposals (id, run_id, entity_type, proposed_name, "
            "proposed_description, proposed_parent, sequence, confidence, status) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (p.id, p.run_id, p.entity_type, p.proposed_name, p.proposed_description,
             p.proposed_parent, p.sequence, p.confidence, p.status))
        self._commit()
        return p

    def proposals_for_run(self, run_id: str) -> list[CurriculumExtractionProposal]:
        rows = self.conn.execute(
            "SELECT * FROM curriculum_extraction_proposals WHERE run_id=? ORDER BY sequence",
            (run_id,)).fetchall()
        return [CurriculumExtractionProposal(**dict(r)) for r in rows]

    def get_proposal(self, proposal_id: str) -> Optional[CurriculumExtractionProposal]:
        r = self.conn.execute(
            "SELECT * FROM curriculum_extraction_proposals WHERE id=?", (proposal_id,)).fetchone()
        return CurriculumExtractionProposal(**dict(r)) if r else None

    def set_proposal_status(self, proposal_id: str, status: str, *,
                            edited_name: Optional[str] = None) -> None:
        """status: 'approved' | 'edited' | 'rejected'. edited_name is the
        admin's replacement text when status='edited' -- the proposal keeps
        proposed_name as the original AI suggestion (traceability) and
        edited_name as what actually gets materialized."""
        self.conn.execute(
            "UPDATE curriculum_extraction_proposals SET status=?, edited_name=? WHERE id=?",
            (status, edited_name, proposal_id))
        self._commit()

    def set_proposal_materialized_id(self, proposal_id: str, materialized_id: str) -> None:
        self.conn.execute(
            "UPDATE curriculum_extraction_proposals SET materialized_id=? WHERE id=?",
            (materialized_id, proposal_id))
        self._commit()

    # ---------------- period configuration ----------------

    def create_period_configuration(self, *, school_id: str, academic_year_id: str,
                                    period_minutes: int) -> PeriodConfiguration:
        p = PeriodConfiguration(id=new_id("periodcfg"), school_id=school_id,
                                academic_year_id=academic_year_id, period_minutes=period_minutes)
        self.conn.execute(
            "INSERT INTO period_configurations (id, school_id, academic_year_id, period_minutes) "
            "VALUES (?,?,?,?)",
            (p.id, p.school_id, p.academic_year_id, p.period_minutes))
        self._commit()
        return p

    def period_configuration_for_year(self, academic_year_id: str) -> Optional[PeriodConfiguration]:
        r = self.conn.execute(
            "SELECT * FROM period_configurations WHERE academic_year_id=?", (academic_year_id,)).fetchone()
        return PeriodConfiguration(**dict(r)) if r else None

    def set_subject_period_allocation(self, *, school_id: str, academic_year_id: str,
                                      subject: str, periods_per_week: int) -> SubjectPeriodAllocation:
        """Upsert, not create-once-then-409 (see SubjectPeriodAllocation's
        own docstring for why): a school setting Science to 6 periods/week
        this term and 7 next term should not need a distinct row per term,
        and re-POSTing must update, not IntegrityError."""
        existing = self.conn.execute(
            "SELECT id FROM subject_period_allocations WHERE academic_year_id=? AND subject=?",
            (academic_year_id, subject)).fetchone()
        alloc_id = existing["id"] if existing else new_id("spa")
        self.conn.execute(
            "INSERT INTO subject_period_allocations "
            "(id, school_id, academic_year_id, subject, periods_per_week) VALUES (?,?,?,?,?) "
            "ON CONFLICT(academic_year_id, subject) DO UPDATE SET periods_per_week=excluded.periods_per_week",
            (alloc_id, school_id, academic_year_id, subject, periods_per_week))
        self._commit()
        return SubjectPeriodAllocation(id=alloc_id, school_id=school_id, academic_year_id=academic_year_id,
                                       subject=subject, periods_per_week=periods_per_week)

    def subject_period_allocation(self, academic_year_id: str, subject: str) -> Optional[SubjectPeriodAllocation]:
        r = self.conn.execute(
            "SELECT * FROM subject_period_allocations WHERE academic_year_id=? AND subject=?",
            (academic_year_id, subject)).fetchone()
        return SubjectPeriodAllocation(**dict(r)) if r else None

    def subject_period_allocations_for_year(self, academic_year_id: str) -> list[SubjectPeriodAllocation]:
        rows = self.conn.execute(
            "SELECT * FROM subject_period_allocations WHERE academic_year_id=? ORDER BY subject",
            (academic_year_id,)).fetchall()
        return [SubjectPeriodAllocation(**dict(r)) for r in rows]

    def subject_name_for_book(self, book_id: str) -> Optional[str]:
        r = self.conn.execute(
            "SELECT s.name FROM books b JOIN subjects s ON b.subject_id = s.id WHERE b.id=?",
            (book_id,)).fetchone()
        return r["name"] if r else None

    def add_timetable_slot(self, *, school_id: str, academic_year_id: str, subject: str,
                           day_of_week: int, period_number: int) -> SubjectTimetableSlot:
        if not (0 <= day_of_week <= 6):
            raise ValueError(f"day_of_week must be 0 (Monday) .. 6 (Sunday), got {day_of_week}")
        slot = SubjectTimetableSlot(id=new_id("slot"), school_id=school_id,
                                    academic_year_id=academic_year_id, subject=subject,
                                    day_of_week=day_of_week, period_number=period_number)
        self.conn.execute(
            "INSERT INTO subject_timetable_slots "
            "(id, school_id, academic_year_id, subject, day_of_week, period_number) VALUES (?,?,?,?,?,?)",
            (slot.id, slot.school_id, slot.academic_year_id, slot.subject,
             slot.day_of_week, slot.period_number))
        self._commit()
        return slot

    def timetable_slots_for_subject(self, academic_year_id: str, subject: str) -> list[SubjectTimetableSlot]:
        rows = self.conn.execute(
            "SELECT * FROM subject_timetable_slots WHERE academic_year_id=? AND subject=? "
            "ORDER BY day_of_week, period_number",
            (academic_year_id, subject)).fetchall()
        return [SubjectTimetableSlot(**dict(r)) for r in rows]

    def get_timetable_slot(self, slot_id: str) -> Optional[SubjectTimetableSlot]:
        r = self.conn.execute(
            "SELECT * FROM subject_timetable_slots WHERE id=?", (slot_id,)).fetchone()
        return SubjectTimetableSlot(**dict(r)) if r else None

    def remove_timetable_slot(self, slot_id: str) -> None:
        cur = self.conn.execute("DELETE FROM subject_timetable_slots WHERE id=?", (slot_id,))
        if cur.rowcount == 0:
            raise ValueError(f"no timetable slot with id {slot_id!r}")
        self._commit()

    # ---------------- calendar / holidays ----------------

    def create_calendar(self, *, academic_year_id: str,
                        weekly_off_days: Optional[list[str]] = None,
                        alternate_saturday_rule: str = "none") -> Calendar:
        c = Calendar(id=new_id("cal"), academic_year_id=academic_year_id,
                    weekly_off_days=weekly_off_days or ["sunday"],
                    alternate_saturday_rule=alternate_saturday_rule)
        self.conn.execute(
            "INSERT INTO calendars (id, academic_year_id, weekly_off_days, alternate_saturday_rule) "
            "VALUES (?,?,?,?)",
            (c.id, c.academic_year_id, json.dumps(c.weekly_off_days), c.alternate_saturday_rule))
        self._commit()
        return c

    def get_calendar_for_year(self, academic_year_id: str) -> Optional[Calendar]:
        r = self.conn.execute(
            "SELECT * FROM calendars WHERE academic_year_id=?", (academic_year_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["weekly_off_days"] = json.loads(d["weekly_off_days"])
        return Calendar(**d)

    def add_holiday(self, *, calendar_id: str, date: str, label: str, kind: str = "holiday",
                    end_date: Optional[str] = None) -> Holiday:
        if end_date is not None and end_date < date:
            raise ValueError(f"holiday end_date {end_date} is before its date {date}")
        h = Holiday(id=new_id("holiday"), calendar_id=calendar_id, date=date, label=label,
                   kind=kind, end_date=end_date)
        self.conn.execute(
            "INSERT INTO holidays (id, calendar_id, date, label, kind, end_date) VALUES (?,?,?,?,?,?)",
            (h.id, h.calendar_id, h.date, h.label, h.kind, h.end_date))
        self._commit()
        return h

    def holidays_for_calendar(self, calendar_id: str) -> list[Holiday]:
        rows = self.conn.execute(
            "SELECT * FROM holidays WHERE calendar_id=? ORDER BY date", (calendar_id,)).fetchall()
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
        self.conn.execute(
            "INSERT INTO teaching_time_estimates (id, subtopic_id, academic_year_id, "
            "estimated_minutes, estimated_periods, method, approved_by) VALUES (?,?,?,?,?,?,?)",
            (t.id, t.subtopic_id, t.academic_year_id, t.estimated_minutes,
             t.estimated_periods, t.method, t.approved_by))
        self._commit()
        return t

    def teaching_time_estimate_for_subtopic(self, subtopic_id: str,
                                            academic_year_id: str) -> Optional[TeachingTimeEstimate]:
        r = self.conn.execute(
            "SELECT * FROM teaching_time_estimates WHERE subtopic_id=? AND academic_year_id=?",
            (subtopic_id, academic_year_id)).fetchone()
        return TeachingTimeEstimate(**dict(r)) if r else None

    # ---------------- scheduled lessons (§11-14) ----------------

    def create_scheduled_lesson(self, *, school_id: str, academic_year_id: str, book_id: str,
                                subtopic_id: str, date: str, status: str = "scheduled") -> ScheduledLesson:
        from datetime import datetime, timezone
        lesson = ScheduledLesson(
            id=new_id("lesson"), school_id=school_id, academic_year_id=academic_year_id,
            book_id=book_id, subtopic_id=subtopic_id, date=date, status=status,
            created_at=datetime.now(timezone.utc).isoformat())
        self.conn.execute(
            "INSERT INTO scheduled_lessons (id, school_id, academic_year_id, book_id, subtopic_id, "
            "date, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (lesson.id, lesson.school_id, lesson.academic_year_id, lesson.book_id,
             lesson.subtopic_id, lesson.date, lesson.status, lesson.created_at))
        self._commit()
        return lesson

    def get_scheduled_lesson(self, lesson_id: str) -> Optional[ScheduledLesson]:
        r = self.conn.execute("SELECT * FROM scheduled_lessons WHERE id=?", (lesson_id,)).fetchone()
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
        self.conn.execute(
            "UPDATE scheduled_lessons SET status=?, note=?, completed_by=?, completed_at=? WHERE id=?",
            (status, note, completed_by, completed_at, lesson_id))
        self._commit()
        return self.get_scheduled_lesson(lesson_id)

    def reschedule_lesson_date(self, lesson_id: str, *, new_date: str) -> Optional[ScheduledLesson]:
        """§14: the one write PUSH/ADJUST ever make -- just the date.
        Never touches status/note/completed_by, so a lesson already marked
        completed and then legitimately moved (e.g. corrected after the
        fact) keeps its completion record intact. The real audit trail
        (old date, new date, reason, changed-by, timestamp) is the
        caller's job (scheduling.py), via the existing assessment audit
        log -- this store method only ever changes the one column."""
        self.conn.execute("UPDATE scheduled_lessons SET date=? WHERE id=?", (new_date, lesson_id))
        self._commit()
        return self.get_scheduled_lesson(lesson_id)

    def scheduled_lessons_for_book(self, academic_year_id: str, book_id: str) -> list[ScheduledLesson]:
        rows = self.conn.execute(
            "SELECT * FROM scheduled_lessons WHERE academic_year_id=? AND book_id=? ORDER BY date",
            (academic_year_id, book_id)).fetchall()
        return [ScheduledLesson(**dict(r)) for r in rows]

    def scheduled_lessons_for_subtopic(self, subtopic_id: str, academic_year_id: str) -> list[ScheduledLesson]:
        rows = self.conn.execute(
            "SELECT * FROM scheduled_lessons WHERE subtopic_id=? AND academic_year_id=? ORDER BY date",
            (subtopic_id, academic_year_id)).fetchall()
        return [ScheduledLesson(**dict(r)) for r in rows]

    def scheduled_lessons_for_date_range(self, school_id: str, start_date: str,
                                         end_date: str) -> list[ScheduledLesson]:
        """The real access pattern a yearly/monthly/weekly/daily view (the
        next milestone) needs -- date-range scoped to one school, not one
        book, since a real day mixes lessons from every subject."""
        rows = self.conn.execute(
            "SELECT * FROM scheduled_lessons WHERE school_id=? AND date>=? AND date<=? ORDER BY date",
            (school_id, start_date, end_date)).fetchall()
        return [ScheduledLesson(**dict(r)) for r in rows]

    def delete_scheduled_lessons_for_book(self, academic_year_id: str, book_id: str) -> int:
        cur = self.conn.execute(
            "DELETE FROM scheduled_lessons WHERE academic_year_id=? AND book_id=?",
            (academic_year_id, book_id))
        self._commit()
        return cur.rowcount

    # ---------------- teacher assignments (§15 -- "what do I teach today") ----------------

    def assign_teacher(self, *, school_id: str, teacher_id: str, book_id: str) -> TeacherAssignment:
        """Idempotent -- assigning the same (teacher, book) pair twice
        returns the existing row rather than erroring or duplicating,
        matching this module's established idempotency posture
        elsewhere (create_teacher_assignment is safe to call from an
        admin UI's "Save" button without a prior existence check)."""
        existing = self.conn.execute(
            "SELECT * FROM teacher_assignments WHERE teacher_id=? AND book_id=?",
            (teacher_id, book_id)).fetchone()
        if existing:
            return TeacherAssignment(**dict(existing))
        from datetime import datetime, timezone
        a = TeacherAssignment(id=new_id("ta"), school_id=school_id, teacher_id=teacher_id,
                              book_id=book_id, created_at=datetime.now(timezone.utc).isoformat())
        self.conn.execute(
            "INSERT INTO teacher_assignments (id, school_id, teacher_id, book_id, created_at) "
            "VALUES (?,?,?,?,?)",
            (a.id, a.school_id, a.teacher_id, a.book_id, a.created_at))
        self._commit()
        return a

    def assignments_for_teacher(self, teacher_id: str) -> list[TeacherAssignment]:
        rows = self.conn.execute(
            "SELECT * FROM teacher_assignments WHERE teacher_id=? ORDER BY created_at",
            (teacher_id,)).fetchall()
        return [TeacherAssignment(**dict(r)) for r in rows]

    def unassign_teacher(self, *, teacher_id: str, book_id: str) -> None:
        self.conn.execute("DELETE FROM teacher_assignments WHERE teacher_id=? AND book_id=?",
                          (teacher_id, book_id))
        self._commit()

    # ---------------- student enrollment (§18 -- student visibility) ----------------

    def enroll_student(self, *, school_id: str, student_id: str, grade_id: str) -> StudentEnrollment:
        """One enrollment per student -- re-enrolling (e.g. a promotion to
        a new grade) replaces the existing row rather than erroring or
        leaving two, since a real student is only ever in one class at a
        time."""
        existing = self.conn.execute(
            "SELECT * FROM student_enrollments WHERE student_id=?", (student_id,)).fetchone()
        from datetime import datetime, timezone
        if existing:
            self.conn.execute("UPDATE student_enrollments SET grade_id=?, school_id=? WHERE student_id=?",
                              (grade_id, school_id, student_id))
            self._commit()
            return StudentEnrollment(id=existing["id"], school_id=school_id, student_id=student_id,
                                     grade_id=grade_id, created_at=existing["created_at"])
        e = StudentEnrollment(id=new_id("enroll"), school_id=school_id, student_id=student_id,
                              grade_id=grade_id, created_at=datetime.now(timezone.utc).isoformat())
        self.conn.execute(
            "INSERT INTO student_enrollments (id, school_id, student_id, grade_id, created_at) "
            "VALUES (?,?,?,?,?)",
            (e.id, e.school_id, e.student_id, e.grade_id, e.created_at))
        self._commit()
        return e

    def enrollment_for_student(self, student_id: str) -> Optional[StudentEnrollment]:
        r = self.conn.execute(
            "SELECT * FROM student_enrollments WHERE student_id=?", (student_id,)).fetchone()
        return StudentEnrollment(**dict(r)) if r else None

    # ---------------- management reporting & variance (§17, §32) ----------------

    def get_coverage_report(self, *, school_id: str, academic_year_id: str,
                            as_of_date: Optional[str] = None) -> dict[str, Any]:
        """Planned vs. actually-taught coverage, variance, and completion %
        aggregated by Subject and Chapter for school management (§17, §32)."""
        from datetime import datetime, timezone
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).date().isoformat()

        rows = self.conn.execute(
            """
            SELECT l.id, l.date, l.status, l.subtopic_id,
                   st.name as subtopic_name, tp.id as topic_id, tp.name as topic_name,
                   ch.id as chapter_id, ch.name as chapter_name,
                   b.id as book_id, b.title as book_title,
                   s.id as subject_id, s.name as subject_name,
                   g.id as grade_id, g.number as grade_number
            FROM scheduled_lessons l
            JOIN subtopics st ON l.subtopic_id = st.id
            JOIN topics tp ON st.topic_id = tp.id
            JOIN chapters ch ON tp.chapter_id = ch.id
            JOIN books b ON l.book_id = b.id
            JOIN subjects s ON b.subject_id = s.id
            JOIN grades g ON s.grade_id = g.id
            WHERE l.school_id = ? AND l.academic_year_id = ?
            ORDER BY g.number, s.name, ch.name, l.date
            """,
            (school_id, academic_year_id),
        ).fetchall()

        assignment_rows = self.conn.execute(
            "SELECT teacher_id, book_id FROM teacher_assignments WHERE school_id=?",
            (school_id,),
        ).fetchall()
        teacher_for_book = {r["book_id"]: r["teacher_id"] for r in assignment_rows}

        subjects_map: dict[str, dict[str, Any]] = {}
        for r in rows:
            sid = r["subject_id"]
            if sid not in subjects_map:
                subjects_map[sid] = {
                    "subject_id": sid,
                    "subject_name": r["subject_name"],
                    "grade_number": r["grade_number"],
                    "book_id": r["book_id"],
                    "book_title": r["book_title"],
                    "teacher_id": teacher_for_book.get(r["book_id"]),
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
        planned_to_date = sum(1 for r in rows if r["date"] <= as_of_date)
        completed_to_date = sum(1 for r in rows if r["date"] <= as_of_date and r["status"] == "completed")

        overall_coverage_pct = round((completed_lessons / total_lessons * 100), 2) if total_lessons > 0 else 0.0
        overall_pace_pct = round((completed_to_date / planned_to_date * 100), 2) if planned_to_date > 0 else 100.0
        overall_variance = completed_to_date - planned_to_date

        subjects_out = []
        for sid, sdata in subjects_map.items():
            s_lessons = sdata["lessons"]
            s_total = len(s_lessons)
            s_completed = sum(1 for l in s_lessons if l["status"] == "completed")
            s_skipped = sum(1 for l in s_lessons if l["status"] == "skipped")
            s_planned_to_date = sum(1 for l in s_lessons if l["date"] <= as_of_date)
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
                "subject_id": sid,
                "subject_name": sdata["subject_name"],
                "grade_number": sdata["grade_number"],
                "book_id": sdata["book_id"],
                "book_title": sdata["book_title"],
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
        'scheduled' status (§17, §32)."""
        from datetime import date, datetime, timezone
        if not as_of_date:
            as_of_date = datetime.now(timezone.utc).date().isoformat()
        as_of = date.fromisoformat(as_of_date)

        rows = self.conn.execute(
            """
            SELECT l.id as lesson_id, l.date as scheduled_date, l.subtopic_id,
                   st.name as subtopic_name, tp.name as topic_name,
                   ch.name as chapter_name, s.name as subject_name,
                   g.number as grade_number, b.id as book_id
            FROM scheduled_lessons l
            JOIN subtopics st ON l.subtopic_id = st.id
            JOIN topics tp ON st.topic_id = tp.id
            JOIN chapters ch ON tp.chapter_id = ch.id
            JOIN books b ON l.book_id = b.id
            JOIN subjects s ON b.subject_id = s.id
            JOIN grades g ON s.grade_id = g.id
            WHERE l.school_id = ? AND l.academic_year_id = ? AND l.date < ? AND l.status = 'scheduled'
            ORDER BY l.date, g.number, s.name
            """,
            (school_id, academic_year_id, as_of_date),
        ).fetchall()

        assignment_rows = self.conn.execute(
            "SELECT teacher_id, book_id FROM teacher_assignments WHERE school_id=?",
            (school_id,),
        ).fetchall()
        teacher_for_book = {r["book_id"]: r["teacher_id"] for r in assignment_rows}

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
                "teacher_id": teacher_for_book.get(r["book_id"]),
                "teacher_name": None,
            })

        return {
            "school_id": school_id,
            "academic_year_id": academic_year_id,
            "as_of_date": as_of_date,
            "delayed_count": len(delayed),
            "delayed_lessons": delayed,
        }


_INSTANCE: Optional[CurriculumStore] = None


def get_curriculum_store(data_root: Path) -> CurriculumStore:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = CurriculumStore(data_root / "curriculum" / "curriculum.sqlite")
    return _INSTANCE
