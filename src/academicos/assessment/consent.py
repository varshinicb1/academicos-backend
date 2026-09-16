"""Parental consent tracking — DPDP Act 2023 compliance.

Every CBSE student in standard K-12 schooling is a minor (<18 years old).
Under India's Digital Personal Data Protection (DPDP) Act 2023 §9, processing
personal data of minors requires verifiable parental/guardian consent.

Consent records here represent verified external consent (e.g. signed admission
forms, physical consent slips on file, verified parent portal assertions),
bound to (school_id, student_id) and audited server-side.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field

from .postgres_kv import durable_table

logger = logging.getLogger(__name__)


class ParentalConsentRecord(BaseModel):
    id: str
    school_id: str = Field(alias="schoolId")
    student_id: str = Field(alias="studentId")
    guardian_name: str = Field(alias="guardianName")
    guardian_relationship: str = Field(default="parent", alias="guardianRelationship")
    method: str  # e.g. "Signed admission form", "Physical consent slip", "Parent portal confirmation"
    status: str = "granted"  # "granted" | "revoked"
    purpose: str = "assessment_and_grading"
    notes: Optional[str] = None
    recorded_by: str = Field(alias="recordedBy")
    recorded_at: str = Field(alias="recordedAt")
    revoked_at: Optional[str] = Field(default=None, alias="revokedAt")
    revoked_by: Optional[str] = Field(default=None, alias="revokedBy")

    model_config = {"populate_by_name": True}

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "school_id": self.school_id,
            "student_id": self.student_id,
            "guardian_name": self.guardian_name,
            "guardian_relationship": self.guardian_relationship,
            "method": self.method,
            "status": self.status,
            "purpose": self.purpose,
            "notes": self.notes,
            "recorded_by": self.recorded_by,
            "recorded_at": self.recorded_at,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ParentalConsentRecord:
        return cls(
            id=row["id"],
            schoolId=row["school_id"],
            studentId=row["student_id"],
            guardianName=row["guardian_name"],
            guardianRelationship=row.get("guardian_relationship") or "parent",
            method=row["method"],
            status=row.get("status") or "granted",
            purpose=row.get("purpose") or "assessment_and_grading",
            notes=row.get("notes"),
            recordedBy=row["recorded_by"],
            recordedAt=row["recorded_at"],
            revokedAt=row.get("revoked_at"),
            revokedBy=row.get("revoked_by"),
        )


class ConsentStore:
    def __init__(self, db_path: Path):
        self._remote = durable_table("parental_consents")
        self._conn_lock = threading.Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn_lock:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS parental_consents (
                    id TEXT PRIMARY KEY,
                    school_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    guardian_name TEXT NOT NULL,
                    guardian_relationship TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'granted',
                    purpose TEXT NOT NULL DEFAULT 'assessment_and_grading',
                    notes TEXT,
                    recorded_by TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    revoked_at TEXT,
                    revoked_by TEXT,
                    UNIQUE(school_id, student_id)
                );
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_parental_consents_school
                ON parental_consents (school_id);
            """)
            self.conn.commit()

    def record_consent(
        self,
        *,
        school_id: str,
        student_id: str,
        guardian_name: str,
        guardian_relationship: str = "parent",
        method: str,
        recorded_by: str,
        notes: Optional[str] = None,
        purpose: str = "assessment_and_grading",
    ) -> ParentalConsentRecord:
        now = datetime.now(timezone.utc).isoformat()
        record_id = f"consent_{uuid.uuid4().hex[:12]}"

        record = ParentalConsentRecord(
            id=record_id,
            schoolId=school_id,
            studentId=student_id,
            guardianName=guardian_name,
            guardianRelationship=guardian_relationship,
            method=method,
            status="granted",
            purpose=purpose,
            notes=notes,
            recordedBy=recorded_by,
            recordedAt=now,
        )

        row = record.to_row()

        if self._remote.enabled:
            try:
                self._remote.upsert(row, on_conflict="school_id,student_id")
                return record
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for consent (%s, %s)",
                    school_id, student_id, exc_info=True,
                )

        with self._conn_lock:
            self.conn.execute("""
                INSERT INTO parental_consents (
                    id, school_id, student_id, guardian_name, guardian_relationship,
                    method, status, purpose, notes, recorded_by, recorded_at, revoked_at, revoked_by
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(school_id, student_id) DO UPDATE SET
                    guardian_name=excluded.guardian_name,
                    guardian_relationship=excluded.guardian_relationship,
                    method=excluded.method,
                    status='granted',
                    purpose=excluded.purpose,
                    notes=excluded.notes,
                    recorded_by=excluded.recorded_by,
                    recorded_at=excluded.recorded_at,
                    revoked_at=NULL,
                    revoked_by=NULL
            """, (
                record.id, record.school_id, record.student_id, record.guardian_name,
                record.guardian_relationship, record.method, record.status, record.purpose,
                record.notes, record.recorded_by, record.recorded_at, None, None
            ))
            self.conn.commit()

        return record

    def get_consent(self, school_id: str, student_id: str) -> Optional[ParentalConsentRecord]:
        if self._remote.enabled:
            try:
                rows = self._remote.select(school_id=school_id, student_id=student_id)
                if rows:
                    return ParentalConsentRecord.from_row(rows[0])
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for get_consent (%s, %s)",
                    school_id, student_id, exc_info=True,
                )

        with self._conn_lock:
            r = self.conn.execute(
                "SELECT * FROM parental_consents WHERE school_id=? AND student_id=?",
                (school_id, student_id)
            ).fetchone()

        return ParentalConsentRecord.from_row(dict(r)) if r else None

    def has_consent(self, school_id: str, student_id: str) -> bool:
        """True if verifiable parental consent is on file and active."""
        c = self.get_consent(school_id, student_id)
        return c is not None and c.status == "granted"

    def revoke_consent(
        self,
        school_id: str,
        student_id: str,
        revoked_by: str,
    ) -> Optional[ParentalConsentRecord]:
        existing = self.get_consent(school_id, student_id)
        if not existing:
            return None

        now = datetime.now(timezone.utc).isoformat()
        updated = existing.model_copy(update={
            "status": "revoked",
            "revoked_at": now,
            "revoked_by": revoked_by,
        })
        row = updated.to_row()

        if self._remote.enabled:
            try:
                self._remote.upsert(row, on_conflict="school_id,student_id")
                return updated
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for revoke_consent (%s, %s)",
                    school_id, student_id, exc_info=True,
                )

        with self._conn_lock:
            self.conn.execute("""
                UPDATE parental_consents
                SET status='revoked', revoked_at=?, revoked_by=?
                WHERE school_id=? AND student_id=?
            """, (now, revoked_by, school_id, student_id))
            self.conn.commit()

        return updated

    def list_for_school(self, school_id: str) -> list[ParentalConsentRecord]:
        if self._remote.enabled:
            try:
                rows = self._remote.select(school_id=school_id)
                return [ParentalConsentRecord.from_row(r) for r in rows]
            except requests.exceptions.RequestException:
                logger.warning(
                    "Supabase unreachable, falling back to local SQLite for list_for_school (%s)",
                    school_id, exc_info=True,
                )

        with self._conn_lock:
            rows = self.conn.execute(
                "SELECT * FROM parental_consents WHERE school_id=? ORDER BY recorded_at DESC",
                (school_id,)
            ).fetchall()

        return [ParentalConsentRecord.from_row(dict(r)) for r in rows]


_INSTANCE: Optional[ConsentStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_consent_store(data_root: Optional[Path] = None) -> ConsentStore:
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                root = data_root or Path(os.environ.get("ACOS_DATA_ROOT", "academicos-data"))
                _INSTANCE = ConsentStore(root / "assessment" / "consents.db")
    return _INSTANCE
