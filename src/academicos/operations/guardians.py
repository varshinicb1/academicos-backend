"""Guardians (M1.5, SA-6): which parent accounts may see which students.

A parent joins through an invite the school makes for one student (the
invite code is linked to the student here, since the invites table itself
is Cloud SQL and has no student column). Registering with that code links
the new account to the child. The principal can link an existing parent
account to a second child and can end a link; both are audited by the
routes. A link, not the parent role, is what lets a parent see a child: a
parent account with no active link sees nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

GUARDIANS_SCHEMA = """
CREATE TABLE IF NOT EXISTS guardianships (
    parent_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    school_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    revoked_by TEXT,
    PRIMARY KEY (parent_id, student_id)
);
CREATE INDEX IF NOT EXISTS idx_guardianships_student ON guardianships(student_id);
CREATE TABLE IF NOT EXISTS guardian_invites (
    code TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    student_id TEXT,
    student_email TEXT,
    relation TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_guardian_links (
    parent_id TEXT NOT NULL,
    school_id TEXT NOT NULL,
    student_email TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (parent_id, student_email)
);
CREATE TABLE IF NOT EXISTS pending_enrollments (
    code TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

RELATIONS = ("mother", "father", "guardian", "grandparent", "other")


@dataclass
class Guardianship:
    parent_id: str
    student_id: str
    school_id: str
    relation: str
    created_by: str
    created_at: str
    revoked_at: Optional[str] = None
    revoked_by: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _relation(value: str) -> str:
    v = (value or "").strip().lower()
    if v not in RELATIONS:
        raise ValueError(f"relation must be one of {', '.join(RELATIONS)}")
    return v


class GuardiansMixin:
    """Added to OperationsStore; uses its _fetchone/_fetchall/_commit."""

    def remember_guardian_invite(self, *, code: str, school_id: str, relation: str, created_by: str,
                                 student_id: Optional[str] = None, student_email: Optional[str] = None) -> None:
        """A parent invite for one child: by account id, or (bulk import,
        before the child has joined) by the child's email."""
        if not (student_id or student_email):
            raise ValueError("a guardian invite needs the student")
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO guardian_invites (code, school_id, student_id, student_email, relation, created_by,"
                " created_at) VALUES (?,?,?,?,?,?,?)",
                (code, school_id, student_id, (student_email or "").strip().lower() or None, _relation(relation),
                 created_by, _now()))
            self._commit()

    def remember_pending_link(self, *, parent_id: str, school_id: str, student_email: str, relation: str,
                              created_by: str) -> None:
        with self._conn_lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO pending_guardian_links (parent_id, school_id, student_email, relation,"
                " created_by, created_at) VALUES (?,?,?,?,?,?)",
                (parent_id, school_id, student_email.strip().lower(), _relation(relation), created_by, _now()))
            self._commit()

    def take_pending_links(self, school_id: str, student_email: str) -> list[dict]:
        """The parents waiting for this student to join, removed as read."""
        email = student_email.strip().lower()
        with self._conn_lock:
            rows = [dict(r) for r in self.conn.execute(
                "SELECT * FROM pending_guardian_links WHERE school_id=? AND student_email=?", (school_id, email))]
            self.conn.execute("DELETE FROM pending_guardian_links WHERE school_id=? AND student_email=?",
                              (school_id, email))
            self._commit()
        return rows

    def remember_pending_enrollment(self, *, code: str, school_id: str, section_id: str, created_by: str) -> None:
        with self._conn_lock:
            self.conn.execute("INSERT OR REPLACE INTO pending_enrollments (code, school_id, section_id, created_by,"
                              " created_at) VALUES (?,?,?,?,?)", (code, school_id, section_id, created_by, _now()))
            self._commit()

    def pending_enrollment(self, code: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM pending_enrollments WHERE code=?", (code,))

    def guardian_invite(self, code: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM guardian_invites WHERE code=?", (code,))

    def link_guardian(self, *, parent_id: str, student_id: str, school_id: str, relation: str,
                      created_by: str) -> Guardianship:
        """Link, or re-link a link that was ended. Linking an active link
        again only updates the relation."""
        relation = _relation(relation)
        with self._conn_lock:
            self.conn.execute(
                "INSERT INTO guardianships (parent_id, student_id, school_id, relation, created_by, created_at)"
                " VALUES (?,?,?,?,?,?) ON CONFLICT(parent_id, student_id) DO UPDATE SET relation=excluded.relation,"
                " revoked_at=NULL, revoked_by=NULL, school_id=excluded.school_id",
                (parent_id, student_id, school_id, relation, created_by, _now()))
            self._commit()
        return self.guardianship(parent_id, student_id)

    def guardianship(self, parent_id: str, student_id: str) -> Optional[Guardianship]:
        r = self._fetchone("SELECT * FROM guardianships WHERE parent_id=? AND student_id=?", (parent_id, student_id))
        return Guardianship(**r) if r else None

    def end_guardianship(self, parent_id: str, student_id: str, *, revoked_by: str) -> Optional[Guardianship]:
        g = self.guardianship(parent_id, student_id)
        if g is None or not g.active:
            return None
        with self._conn_lock:
            self.conn.execute("UPDATE guardianships SET revoked_at=?, revoked_by=? WHERE parent_id=? AND student_id=?",
                              (_now(), revoked_by, parent_id, student_id))
            self._commit()
        return self.guardianship(parent_id, student_id)

    def children_of(self, parent_id: str) -> list[Guardianship]:
        return [Guardianship(**r) for r in self._fetchall(
            "SELECT * FROM guardianships WHERE parent_id=? AND revoked_at IS NULL ORDER BY created_at", (parent_id,))]

    def guardians_of(self, student_id: str, *, include_ended: bool = False) -> list[Guardianship]:
        sql = "SELECT * FROM guardianships WHERE student_id=?" + ("" if include_ended else " AND revoked_at IS NULL")
        return [Guardianship(**r) for r in self._fetchall(sql + " ORDER BY created_at", (student_id,))]

    def parents_of(self, student_ids, *, school_id: str) -> list[str]:
        """The parent accounts with an active link, at this school, to any of
        these students: who hears a school notice about them (N-8-7)."""
        ids, out = sorted(set(student_ids)), set()
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            out |= {r["parent_id"] for r in self._fetchall(
                "SELECT parent_id FROM guardianships WHERE school_id=? AND revoked_at IS NULL AND student_id IN ("
                + ",".join("?" * len(chunk)) + ")", (school_id, *chunk))}
        return sorted(out)

    def is_guardian(self, parent_id: str, student_id: str) -> bool:
        g = self.guardianship(parent_id, student_id)
        return g is not None and g.active
