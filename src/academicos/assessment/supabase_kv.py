"""Shared Supabase PostgREST helper for simple key/JSONB-payload stores.

Used by school_templates.py, graded_store.py and practice_store.py so each
doesn't reimplement the same REST plumbing. See knowledge.py's module
docstring for why this matters: Render's free tier wipes local disk
(SQLite files included) on every restart/spin-down/redeploy, not just
full redeploys, so anything meant to outlive a single request needs to
live in Postgres instead. Reuses the same SUPABASE_KNOWLEDGE_URL/
SUPABASE_KNOWLEDGE_ANON_KEY env vars as KnowledgeStore -- one Supabase
project, multiple tables.
"""
from __future__ import annotations

from typing import Any, Optional

import requests

from ..config import credential_or_none


def _supabase_credentials() -> tuple[Optional[str], Optional[str]]:
    """The (url, key) pair, with the deployment placeholder treated as absent.

    Shared by SupabaseTable and SupabaseStorage so the two can never disagree
    about whether Supabase is configured -- they used to duplicate the raw
    `os.environ.get(...)` read, and `bool(url and key)` reported True for the
    GCP bootstrap's `REPLACE_ME` placeholder, which sent blob uploads at a host
    that does not exist. See config.credential_or_none.
    """
    url = (credential_or_none("SUPABASE_KNOWLEDGE_URL") or "").rstrip("/") or None
    return url, credential_or_none("SUPABASE_KNOWLEDGE_ANON_KEY")


class SupabaseUnavailable(requests.exceptions.RequestException):
    """A live Supabase/PostgREST call failed: a non-2xx response (most
    commonly a missing/mis-shaped table, which PostgREST reports as a 404
    with a `PGRST205` body) or no connection at all. Raised instead of a
    bare `HTTPError` so callers get the table, operation, status, and a
    body excerpt in one message.

    Deliberately a RequestException subclass: the stores that already
    catch a live failure to fall back to local SQLite (AssessmentStore,
    EventStore) keep working unchanged. api/main.py also registers one
    app-level handler for this exact type, so the stores that DON'T fall
    back answer a real 503 with this message instead of a bare 500 -- the
    2026-09-15 production incident where /auth/login, /auth/register, and
    /auth/me all 500'd was exactly this: the Supabase `users`/`sessions`
    tables were absent, and nothing surfaced any of it.
    """

    def __init__(self, *, table: str, op: str, status: int | None = None,
                 body: str = "", cause: BaseException | None = None):
        detail = f"Supabase {op} on {table!r} failed"
        if status is not None:
            detail += f" (HTTP {status})"
        if body:
            detail += f": {body}"
        super().__init__(detail)
        self.table = table
        self.op = op
        self.status = status
        self.body = body


class SupabaseTable:
    """Minimal PostgREST wrapper: upsert/select/delete against one table
    with a JSONB `payload` column. Not a general ORM."""

    def __init__(self, table: str):
        self.table = table
        self._url, self._key = _supabase_credentials()

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        h = {"apikey": self._key, "Authorization": f"Bearer {self._key}",
             "Content-Type": "application/json"}
        if prefer:
            h["Prefer"] = prefer
        return h

    def _request(self, method: str, op: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("headers", self._headers())
        try:
            r = requests.request(method, f"{self._url}/rest/v1/{self.table}",
                                 timeout=10, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise SupabaseUnavailable(table=self.table, op=op,
                                      body=str(exc)[:200], cause=exc) from exc
        if not r.ok:
            body = " ".join((r.text or "").split())[:200]
            raise SupabaseUnavailable(table=self.table, op=op,
                                      status=r.status_code, body=body)
        return r

    def select(self, order: str | None = None, limit: int | None = None,
               gt: dict[str, Any] | None = None, **eq_filters: str) -> list[dict[str, Any]]:
        params: dict[str, str] = {k: f"eq.{v}" for k, v in eq_filters.items()}
        if gt:
            # Separate from eq_filters because a column can't appear twice in
            # one **kwargs dict -- needed by EventStore's `seq > since_seq`
            # pagination, which an eq-only filter set can't express.
            params.update({k: f"gt.{v}" for k, v in gt.items()})
        params["select"] = "*"
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        r = self._request("GET", "select", params=params)
        return r.json()

    def upsert(self, row: dict[str, Any], on_conflict: str) -> None:
        self._request("POST", "upsert", params={"on_conflict": on_conflict},
                      json=row,
                      headers=self._headers(prefer="resolution=merge-duplicates"))

    def update(self, values: dict[str, Any], **eq_filters: str) -> None:
        params = {k: f"eq.{v}" for k, v in eq_filters.items()}
        self._request("PATCH", "update", params=params, json=values)

    def delete(self, **eq_filters: str) -> None:
        params = {k: f"eq.{v}" for k, v in eq_filters.items()}
        self._request("DELETE", "delete", params=params)


class SupabaseStorage:
    """Thin wrapper over Supabase Storage's REST API for one bucket -- used
    for scan-session photos/PDFs, which are too large/binary for a JSONB
    column. Same enable/disable gating and env vars as SupabaseTable (one
    Supabase project, Postgres for structured data, Storage for blobs).
    Private bucket: the anon key is a server-side-only credential here (see
    knowledge.py's module docstring), never shipped to the Flutter client --
    objects are always proxied through our own FastAPI endpoints."""

    def __init__(self, bucket: str):
        self.bucket = bucket
        self._url, self._key = _supabase_credentials()

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    def _headers(self) -> dict[str, str]:
        return {"apikey": self._key, "Authorization": f"Bearer {self._key}"}

    def upload(self, key: str, data: bytes, content_type: str) -> None:
        r = requests.post(
            f"{self._url}/storage/v1/object/{self.bucket}/{key}",
            data=data, headers={**self._headers(), "Content-Type": content_type,
                                "x-upsert": "true"},
            timeout=30,
        )
        r.raise_for_status()

    def download(self, key: str) -> bytes:
        r = requests.get(f"{self._url}/storage/v1/object/{self.bucket}/{key}",
                         headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.content
