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

import os
from typing import Any

import requests


class SupabaseTable:
    """Minimal PostgREST wrapper: upsert/select/delete against one table
    with a JSONB `payload` column. Not a general ORM."""

    def __init__(self, table: str):
        self.table = table
        self._url = os.environ.get("SUPABASE_KNOWLEDGE_URL", "").strip().rstrip("/") or None
        self._key = os.environ.get("SUPABASE_KNOWLEDGE_ANON_KEY", "").strip() or None

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._key)

    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        h = {"apikey": self._key, "Authorization": f"Bearer {self._key}",
             "Content-Type": "application/json"}
        if prefer:
            h["Prefer"] = prefer
        return h

    def select(self, order: str | None = None, **eq_filters: str) -> list[dict[str, Any]]:
        params: dict[str, str] = {k: f"eq.{v}" for k, v in eq_filters.items()}
        params["select"] = "*"
        if order:
            params["order"] = order
        r = requests.get(f"{self._url}/rest/v1/{self.table}", params=params,
                         headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json()

    def upsert(self, row: dict[str, Any], on_conflict: str) -> None:
        r = requests.post(
            f"{self._url}/rest/v1/{self.table}", params={"on_conflict": on_conflict},
            json=row, headers=self._headers(prefer="resolution=merge-duplicates"), timeout=10,
        )
        r.raise_for_status()

    def update(self, values: dict[str, Any], **eq_filters: str) -> None:
        params = {k: f"eq.{v}" for k, v in eq_filters.items()}
        r = requests.patch(f"{self._url}/rest/v1/{self.table}", params=params, json=values,
                           headers=self._headers(), timeout=10)
        r.raise_for_status()

    def delete(self, **eq_filters: str) -> None:
        params = {k: f"eq.{v}" for k, v in eq_filters.items()}
        r = requests.delete(f"{self._url}/rest/v1/{self.table}", params=params,
                            headers=self._headers(), timeout=10)
        r.raise_for_status()
