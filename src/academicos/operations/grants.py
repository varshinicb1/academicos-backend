"""Delegated admin (M1.4, ROLE-1): the principal grants a teacher one admin
capability, optionally limited to some classes, sections or subjects (an
HOD or a coordinator), and revokes it. A grant is what lets a teacher
through auth_routes.require_admin(capability); the principal always passes.
Every grant and revoke is audited by the routes."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

GRANTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_grants (
    id TEXT PRIMARY KEY,
    school_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    capability TEXT NOT NULL,
    scope_json TEXT NOT NULL DEFAULT '{}',
    granted_by TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    revoked_at TEXT,
    revoked_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_admin_grants_user ON admin_grants(user_id, capability);
"""

SCOPE_KEYS = ("grades", "sectionIds", "subjectIds")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_scope(scope: Optional[dict]) -> dict[str, list]:
    """{grades?, sectionIds?, subjectIds?}; an empty or missing list means
    the whole school on that axis."""
    out: dict[str, list] = {}
    for k in SCOPE_KEYS:
        v = (scope or {}).get(k) or []
        if not isinstance(v, list):
            raise ValueError(f"{k} must be a list")
        if v:
            out[k] = sorted(set(int(x) if k == "grades" else str(x) for x in v))
    return out


def scope_admits(scope: dict, *, grade: Optional[int] = None, section_id: Optional[str] = None,
                 subject_id: Optional[str] = None) -> bool:
    """Whether a grant with this scope covers the target. Each axis the scope
    limits must be named by the target and be in the list: a grant for
    classes 6 and 7 does not cover an action that names no class (a
    whole-school solve, the school's user list), and a Science HOD's grant
    does not cover a change to a whole section. Until 2026-09-30 an axis the
    target left out passed, and no route named one, so every limited grant
    acted as a whole-school grant (N-67-1)."""
    for key, value in (("grades", grade), ("sectionIds", section_id), ("subjectIds", subject_id)):
        if scope.get(key) and (value is None or value not in scope[key]):
            return False
    return True


class GrantsMixin:
    """Added to OperationsStore."""

    def _grant(self, r: Optional[dict]) -> Optional[dict]:
        if r is None:
            return None
        r = dict(r)
        r["scope"] = json.loads(r.pop("scope_json") or "{}")
        r["active"] = r["revoked_at"] is None
        return r

    def grant(self, *, school_id: str, user_id: str, capability: str, scope: Optional[dict],
              granted_by: str) -> dict:
        """One active grant per (user, capability): granting again replaces
        its scope."""
        clean = clean_scope(scope)
        with self._conn_lock:
            self.conn.execute("UPDATE admin_grants SET revoked_at=?, revoked_by=? WHERE user_id=? AND capability=?"
                              " AND revoked_at IS NULL", (_now(), granted_by, user_id, capability))
            gid = f"grant_{uuid.uuid4().hex[:12]}"
            self.conn.execute("INSERT INTO admin_grants (id, school_id, user_id, capability, scope_json, granted_by,"
                              " granted_at) VALUES (?,?,?,?,?,?,?)",
                              (gid, school_id, user_id, capability, json.dumps(clean), granted_by, _now()))
            self._commit()
        return self.get_grant(gid)

    def get_grant(self, grant_id: str) -> Optional[dict]:
        return self._grant(self._fetchone("SELECT * FROM admin_grants WHERE id=?", (grant_id,)))

    def revoke_grant(self, grant_id: str, *, revoked_by: str) -> Optional[dict]:
        g = self.get_grant(grant_id)
        if g is None or not g["active"]:
            return g
        with self._conn_lock:
            self.conn.execute("UPDATE admin_grants SET revoked_at=?, revoked_by=? WHERE id=?",
                              (_now(), revoked_by, grant_id))
            self._commit()
        return self.get_grant(grant_id)

    def grants_for_school(self, school_id: str, *, include_revoked: bool = False) -> list[dict]:
        sql = "SELECT * FROM admin_grants WHERE school_id=?" + ("" if include_revoked else " AND revoked_at IS NULL")
        return [self._grant(r) for r in self._fetchall(sql + " ORDER BY granted_at DESC", (school_id,))]

    def active_grants(self, user_id: str, capability: Optional[str] = None) -> list[dict]:
        sql, params = "SELECT * FROM admin_grants WHERE user_id=? AND revoked_at IS NULL", [user_id]
        if capability:
            sql += " AND capability=?"
            params.append(capability)
        return [self._grant(r) for r in self._fetchall(sql, tuple(params))]

    def holds(self, user_id: str, capability: str, **target: Any) -> bool:
        return any(scope_admits(g["scope"], **target) for g in self.active_grants(user_id, capability))

    def grant_scopes(self, user_id: str, capability: str) -> list[dict]:
        """The scopes of the user's active grants for the capability ({} is
        the whole school); [] when they hold none."""
        return [g["scope"] for g in self.active_grants(user_id, capability)]
