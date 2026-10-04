"""The Phase 1 pilot agreement (2026-10-04): pilot terms, mutual
non-disclosure and data sharing, accepted once per school by its principal in
the web console before the school is used. Routes: pilot_agreement_routes.py.

The text is `pilot_agreement_phase1.md`, served as it is. An acceptance
records the version, a SHA-256 of the exact text the principal was shown,
the school's legal name, who accepted it (name, designation, account,
email), when, and from which address. A changed text is a new version: the
old acceptance stays on record and the principal is asked again. The first
acceptance of a version stands; accepting again returns it unchanged.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..assessment.users import User

VERSION = "phase1-v1"
TITLE = "AcademicOS Phase 1 Pilot Agreement"
_TEXT_FILE = Path(__file__).with_name("pilot_agreement_phase1.md")

PILOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_agreements (
    school_id TEXT NOT NULL,
    version TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    school_legal_name TEXT NOT NULL,
    signatory_name TEXT NOT NULL,
    designation TEXT NOT NULL,
    accepted_by TEXT NOT NULL,
    accepted_email TEXT NOT NULL,
    accepted_at TEXT NOT NULL,
    client_ip TEXT,
    user_agent TEXT,
    PRIMARY KEY (school_id, version)
);
"""


@lru_cache(maxsize=1)
def text() -> str:
    return _TEXT_FILE.read_text(encoding="utf-8")


def text_sha256() -> str:
    return hashlib.sha256(text().encode("utf-8")).hexdigest()


class PilotMixin:
    """Added to OperationsStore."""

    def pilot_acceptance(self, school_id: str, version: str = VERSION) -> Optional[dict]:
        return self._fetchone("SELECT * FROM pilot_agreements WHERE school_id=? AND version=?",
                              (school_id, version))

    def accept_pilot(self, *, school_id: str, version: str, sha256: str, school_legal_name: str,
                     signatory_name: str, designation: str, user: User, client_ip: Optional[str],
                     user_agent: Optional[str]) -> tuple[dict, bool]:
        """Returns (the acceptance on record, whether this call made it)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn_lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO pilot_agreements (school_id, version, text_sha256, school_legal_name,"
                " signatory_name, designation, accepted_by, accepted_email, accepted_at, client_ip, user_agent)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (school_id, version, sha256, school_legal_name, signatory_name, designation, user.id,
                 user.email, now, client_ip, (user_agent or "")[:300] or None))
            self._commit()
            made = cur.rowcount == 1
        return self.pilot_acceptance(school_id, version), made
