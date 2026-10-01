"""Erasing one student's or parent's rows from the operations store (NFR-1:
DPDP deletion on the school's request; assessment/erasure.py runs the whole
erasure, of which this is one store).

Every table in OperationsStore.schema() is named below, either in ERASED (what
erasure deletes from it) or in KEPT (why it holds nothing of the person).
tests/test_erasure.py fails on a table that is in neither, so a new table that
holds a student's data cannot be added without deciding how it is erased.

Deletes run with SQLite's secure_delete on, so the freed pages are zeroed
rather than left in the file; the store's snapshot (operations.sqlite in the
blob store) is a copy of that file.
"""
from __future__ import annotations

import json
from typing import Iterable

from ..storage.secure_delete import secure_delete, truncate_wal

ERASED = {
    "notifications": "the person's inbox, and other people's notices about the student "
                     "(a parent's marks notice, a teacher's 'handed in' notice naming them)",
    "message_deliveries": "SMS and WhatsApp deliveries of those notices",
    "notification_prefs": "the person's channel choices",
    "notification_settings": "the person's quiet hours and language",
    "notification_digests": "the person's digest time",
    "device_tokens": "the person's phones registered for push",
    "device_token_failures": "push failures of those phones",
    "contact_numbers": "the person's mobile number and messaging consent",
    "calendar_feeds": "the person's calendar link",
    "sign_in_codes": "an emailed sign-in code waiting for the person",
    "homework_submissions": "the student's answers, marks and the teacher's feedback",
    "homework_photos": "the student's photos of written work (the files are deleted by the caller)",
    "learning_evidence": "the student's learning progress",
    "practice_sets": "the student's self-practice",
    "paper_marks": "the student's marks per question",
    "paper_absent": "the student marked absent for a paper",
    "guardianships": "the links between the student and their parents",
    "guardian_invites": "parent invites made for the student, or used by the parent",
    "pending_guardian_links": "parent links waiting for the student to join",
    "pending_enrollments": "a class place waiting for the person's invite",
}

KEPT = {
    "homework": "set by a teacher; holds no student's data",
    "homework_sections": "which sections a homework is for",
    "exams": "the school's exams",
    "exam_papers": "the school's datesheet",
    "invigilation": "staff duties",
    "admin_grants": "staff permissions",
    "paper_reviews": "staff review of a paper",
    "question_reviews": "staff review of a bank question",
    "automation_runs": "when a scheduled job last ran",
    "automation_school_runs": "when a scheduled job last ran for a school",
}


class ForgetMixin:
    """Added to OperationsStore."""

    def photos_of_student(self, student_id: str) -> list[dict]:
        """Every homework photo the student uploaded, for the caller to delete
        the files before the rows."""
        return self._fetchall("SELECT * FROM homework_photos WHERE student_id=?", (student_id,))

    def forget_student(self, student_id: str, *, school_id: str, email: str, name: str,
                       invite_codes: Iterable[str] = (), dry_run: bool = False) -> dict[str, int]:
        """Delete everything this store holds of one student. Returns the rows
        per table (the rows that would go, with dry_run)."""
        handed_in = {r["homework_id"] for r in self._fetchall(
            "SELECT homework_id FROM homework_submissions WHERE student_id=?", (student_id,))}
        about = self._notices_about_student(student_id, school_id=school_id, name=name, handed_in=handed_in)
        codes = list(invite_codes)
        # An account with no email must not match every row whose email is
        # blank (an unbound invite, another blank account): "0" matches none.
        by_email = "lower(student_email)=?" if email else "0"
        steps = [
            *self._own_steps(student_id, email),
            ("notifications", f"id IN ({_marks(about)})", tuple(about)),
            ("message_deliveries", f"notification_id IN ({_marks(about)})", tuple(about)),
            ("homework_submissions", "student_id=?", (student_id,)),
            ("homework_photos", "student_id=?", (student_id,)),
            ("learning_evidence", "student_id=?", (student_id,)),
            ("practice_sets", "student_id=?", (student_id,)),
            ("paper_marks", "student_id=?", (student_id,)),
            ("paper_absent", "student_id=?", (student_id,)),
            ("guardianships", "student_id=?", (student_id,)),
            ("guardian_invites", f"school_id=? AND (student_id=? OR {by_email})",
             (school_id, student_id, *((email,) if email else ()))),
            ("pending_guardian_links", f"school_id=? AND {by_email}", (school_id, *((email,) if email else ()))),
            ("pending_enrollments", f"code IN ({_marks(codes)})", tuple(codes)),
        ]
        return self._forget(steps, dry_run)

    def forget_parent(self, parent_id: str, *, email: str, invite_codes: Iterable[str] = (),
                      dry_run: bool = False) -> dict[str, int]:
        """Delete everything this store holds of one parent. The consent they
        gave for a child is not here: it is the child's record (ConsentStore)."""
        codes = list(invite_codes)
        steps = [
            *self._own_steps(parent_id, email),
            ("guardianships", "parent_id=?", (parent_id,)),
            ("guardian_invites", f"code IN ({_marks(codes)})", tuple(codes)),
            ("pending_guardian_links", "parent_id=?", (parent_id,)),
            ("pending_enrollments", f"code IN ({_marks(codes)})", tuple(codes)),
        ]
        return self._forget(steps, dry_run)

    # ---------------- internals ----------------

    @staticmethod
    def _own_steps(user_id: str, email: str) -> list[tuple[str, str, tuple]]:
        """What any account holds of its own: inbox, settings, phones, number,
        calendar link, a pending sign-in code. Failures before tokens: a
        failure row is found through its token."""
        return [
            ("message_deliveries", "user_id=?", (user_id,)),
            ("notifications", "user_id=?", (user_id,)),
            ("notification_prefs", "user_id=?", (user_id,)),
            ("notification_settings", "user_id=?", (user_id,)),
            ("notification_digests", "user_id=?", (user_id,)),
            ("device_token_failures", "token IN (SELECT token FROM device_tokens WHERE user_id=?)", (user_id,)),
            ("device_tokens", "user_id=?", (user_id,)),
            ("contact_numbers", "user_id=?", (user_id,)),
            ("calendar_feeds", "user_id=?", (user_id,)),
            ("sign_in_codes", "lower(email)=?" if email else "0", (email,) if email else ()),
        ]

    def _notices_about_student(self, student_id: str, *, school_id: str, name: str,
                               handed_in: set[str]) -> list[str]:
        """Other people's notifications that carry this student's data: the
        ones keyed on the student (a parent's homework and test marks,
        `hwgraded:<hw>:<student>:...`, `testmarks-parent:<paper>:<student>`)
        and a teacher's "<name> handed in" notice (`hwsubmitted:<hw>:<day>`)
        for a homework the student handed in -- it names the student only by
        name, so another student of that name who handed in the same homework
        cannot be told apart. Their own are deleted by user_id."""
        rows = self._fetchall(
            "SELECT id, kind, dedupe_key, data_json FROM notifications WHERE school_id=? AND user_id<>? "
            "AND (instr(COALESCE(dedupe_key, ''), ?) > 0 OR kind='homework_submitted')",
            (school_id, student_id, student_id))
        out = []
        for r in rows:
            key = (r["dedupe_key"] or "").split(":")
            if student_id in key:
                out.append(r["id"])
            elif (r["kind"] == "homework_submitted" and name and len(key) > 1 and key[1] in handed_in
                  and _param(r["data_json"], "student") == name):
                out.append(r["id"])
        return out

    def _forget(self, steps: list[tuple[str, str, tuple]], dry_run: bool) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._conn_lock:
            with secure_delete(self.conn):
                for table, where, params in steps:
                    if where.endswith("IN ()"):
                        counts.setdefault(table, 0)
                        continue
                    if dry_run:
                        n = self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]
                    else:
                        n = self.conn.execute(f"DELETE FROM {table} WHERE {where}", params).rowcount
                    counts[table] = counts.get(table, 0) + n
            if not dry_run:
                self._commit()
                truncate_wal(self.conn)
        return counts


def _marks(values: list) -> str:
    return ",".join("?" * len(values))


def _param(data_json: str, key: str):
    try:
        return json.loads(data_json or "{}").get(key)
    except (ValueError, AttributeError):
        return None
