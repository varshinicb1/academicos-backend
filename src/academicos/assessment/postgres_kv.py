"""Cloud SQL (PostgreSQL) durability backend for the JSONB-payload stores.

Why this exists
---------------
Every operational store in ``assessment/`` and ``storage/`` was built against
``SupabaseTable`` (see ``supabase_kv.py``): a table with a few real columns
plus one JSONB column holding the full payload, addressed by SQL-ish
operations (``select``/``upsert``/``update``/``delete``). That was the right
shape for the Render deployment, whose local disk is wiped on every restart.

``PostgresTable`` implements the *same four operations with the same
signatures and return shapes*, but talks to a Google Cloud SQL Postgres
instance over the Cloud Run database socket instead of Supabase's PostgREST
HTTP API. That means every store body stays byte-for-byte unchanged -- only
the object it constructs differs (via ``durable_table``), and Supabase
remains a working fallback for as long as its env vars are set.

Migration path, deliberately low-risk
-------------------------------------
``durable_table`` prefers Postgres when Cloud SQL is configured and falls
back to Supabase otherwise. So the two can coexist: provision Cloud SQL,
prove it in staging, then remove the Supabase env vars. Nothing about the
existing Supabase path is modified or removed here.

Compatibility notes learned from the real call sites
----------------------------------------------------
* ``upsert`` is called with a composite conflict target
  (``"assessment_id,student_id"``, ``"school_id,student_id"``) as well as
  single columns, so the conflict target is parsed, not assumed.
* ``select`` filters are keyword equality (``school_id=...``) plus an
  optional ``gt=`` mapping (EventStore's ``seq > since_seq`` pagination).
* ``order`` uses PostgREST's ``"column.dir"`` syntax, sometimes multi-column.
* ``school_templates`` filters on the literal string ``"true"`` for
  ``is_default``, so booleans are stored and compared as ``'true'``/``'false'``
  text -- that also makes ``ORDER BY is_default DESC`` put the default first,
  which is what the caller expects.
* ``seq`` (learner_events) must be a real integer column or numeric
  ``gt``/``ORDER BY`` would sort ``"10"`` before ``"9"``.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

from .supabase_kv import SupabaseUnavailable

logger = logging.getLogger(__name__)

# Only ever built from hardcoded store code, but validated anyway: these
# names are interpolated into SQL text (they cannot be bound as parameters),
# so a typo or a future dynamic name must fail loudly rather than silently
# become an injection point.
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_DEFAULT_POOL_MAX = 10


class PostgresUnavailable(SupabaseUnavailable):
    """A live Cloud SQL call failed.

    Subclasses ``SupabaseUnavailable`` on purpose rather than introducing a
    sibling type: every existing call site catches ``SupabaseUnavailable``
    (or its ``requests.RequestException`` ancestor) to fall back to local
    SQLite, and ``api/main.py`` registers an app-level handler for it. By
    inheriting, all of that keeps working with no edits -- the backend
    changed, the failure contract did not.
    """

    def __init__(self, *, table: str, op: str, status: int | None = None,
                 body: str = "", cause: BaseException | None = None):
        Exception.__init__(self, "")  # bypass SupabaseUnavailable's wording
        detail = f"Cloud SQL {op} on {table!r} failed"
        if status is not None:
            detail += f" (status {status})"
        if body:
            detail += f": {body}"
        self.args = (detail,)
        self.table = table
        self.op = op
        self.status = status
        self.body = body


def _coerce(value: Any) -> Any:
    """Map a Python value onto something Postgres can compare/insert.

    Booleans become the text ``'true'``/``'false'`` (see the module
    docstring), ``dict``/``list`` become JSONB, everything else passes
    through for psycopg to adapt natively.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        from psycopg.types.json import Jsonb
        return Jsonb(value)
    return value


def _check_ident(name: str, what: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"unsafe {what} name for Cloud SQL: {name!r}")
    return name


def _parse_order(order: str) -> str:
    """``"created_at.desc"`` / ``"is_default.desc"`` / ``"seq.asc"`` ->
    ``ORDER BY`` body. Unknown directions are ignored (defaulting to ASC),
    matching PostgREST's tolerant behaviour."""
    parts: list[str] = []
    for chunk in order.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        col, _, direction = chunk.partition(".")
        col = _check_ident(col.strip(), "column")
        parts.append(f"{col} {'DESC' if direction.strip().lower() == 'desc' else 'ASC'}")
    return ", ".join(parts)


# --- connection pool --------------------------------------------------------
# One pool per distinct DSN, shared across all table objects and threads.
# Bounded because Cloud Run's concurrency (40) far exceeds a small Cloud SQL
# instance's connection limit -- an unbounded per-thread scheme would exhaust
# the server under exactly the load this is meant to survive.
_pools: dict[str, Any] = {}
_pools_lock = threading.Lock()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _dsn_parts() -> dict[str, str] | None:
    """Resolve the Cloud SQL connection settings, or ``None`` if unconfigured.

    ``ACOS_POSTGRES_HOST`` is either a Unix-socket directory (Cloud Run's
    ``/cloudsql/PROJECT:REGION:INSTANCE`` mount) or a hostname for local
    development. ``ACOS_POSTGRES_PASSWORD`` is injected from Secret Manager.
    """
    host = _env("ACOS_POSTGRES_HOST")
    db = _env("ACOS_POSTGRES_DB")
    user = _env("ACOS_POSTGRES_USER")
    password = os.environ.get("ACOS_POSTGRES_PASSWORD", "")
    if not (host and db and user):
        return None
    return {
        "host": host,
        "port": _env("ACOS_POSTGRES_PORT", "5432"),
        "dbname": db,
        "user": user,
        "password": password,
        "connect_timeout": "10",
    }


def postgres_configured() -> bool:
    return _dsn_parts() is not None


def _get_pool():
    cfg = _dsn_parts()
    if cfg is None:
        return None
    key = f"{cfg['host']}|{cfg['port']}|{cfg['dbname']}|{cfg['user']}"
    with _pools_lock:
        pool = _pools.get(key)
        if pool is None:
            try:
                from psycopg_pool import ConnectionPool
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise PostgresUnavailable(
                    table="*", op="connect",
                    body="psycopg pool driver not installed (pip install "
                         "academicos[postgres])",
                    cause=exc,
                ) from exc
            pool = ConnectionPool(
                conninfo="",
                kwargs=cfg,
                min_size=0,
                max_size=int(_env("ACOS_POSTGRES_POOL_MAX", str(_DEFAULT_POOL_MAX))),
                open=True,
                timeout=10,
                name="academicos",
            )
            _pools[key] = pool
        return pool


def reset_pools() -> None:
    """Drop every cached pool. Test-support: lets a test point the adapter at
    a different database without leaking connections from the previous one."""
    with _pools_lock:
        pools = dict(_pools)
        _pools.clear()
    for pool in pools.values():
        try:
            pool.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class PostgresTable:
    """Cloud SQL equivalent of :class:`SupabaseTable`.

    Same public surface -- ``enabled``, ``select``, ``upsert``, ``update``,
    ``delete`` -- so a store can be switched between backends by changing
    only the constructor call.
    """

    def __init__(self, table: str):
        self.table = _check_ident(table, "table")

    @property
    def enabled(self) -> bool:
        if not postgres_configured():
            return False
        try:
            return _get_pool() is not None
        except PostgresUnavailable:
            logger.warning("Cloud SQL configured but driver unavailable", exc_info=True)
            return False

    # -- internals ----------------------------------------------------------
    def _execute(self, op: str, sql: str, params: tuple = (), *,
                 fetch: bool = False) -> list[dict[str, Any]]:
        pool = _get_pool()
        if pool is None:
            raise PostgresUnavailable(
                table=self.table, op=op,
                body="Cloud SQL is not configured (ACOS_POSTGRES_HOST/DB/USER unset)",
            )
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    if fetch:
                        cols = [d.name for d in (cur.description or [])]
                        return [dict(zip(cols, row)) for row in cur.fetchall()]
            return []
        except PostgresUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised for fallback
            raise PostgresUnavailable(
                table=self.table, op=op, body=str(exc)[:300], cause=exc,
            ) from exc

    def _where(self, eq_filters: dict[str, Any],
               gt: dict[str, Any] | None = None) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for col, value in eq_filters.items():
            clauses.append(f"{_check_ident(col, 'column')} = %s")
            params.append(_coerce(value))
        for col, value in (gt or {}).items():
            clauses.append(f"{_check_ident(col, 'column')} > %s")
            params.append(_coerce(value))
        return (" AND ".join(clauses) if clauses else "TRUE"), params

    # -- public API ---------------------------------------------------------
    def select(self, order: str | None = None, limit: int | None = None,
               gt: dict[str, Any] | None = None, **eq_filters: str) -> list[dict[str, Any]]:
        where, params = self._where(eq_filters, gt)
        sql = f'SELECT * FROM "{self.table}" WHERE {where}'
        if order:
            sql += f" ORDER BY {_parse_order(order)}"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        return self._execute("select", sql, tuple(params), fetch=True)

    def upsert(self, row: dict[str, Any], on_conflict: str) -> None:
        cols = [_check_ident(c, "column") for c in row]
        if not cols:
            raise ValueError("upsert() requires at least one column")
        conflict_cols = [_check_ident(c.strip(), "column")
                         for c in on_conflict.split(",") if c.strip()]
        if not conflict_cols:
            raise ValueError("upsert() requires an on_conflict column")

        placeholders = ", ".join(["%s"] * len(cols))
        quoted = ", ".join(f'"{c}"' for c in cols)
        updates = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in cols if c not in conflict_cols
        )
        target = ", ".join(f'"{c}"' for c in conflict_cols)

        if updates:
            sql = (f'INSERT INTO "{self.table}" ({quoted}) VALUES ({placeholders}) '
                   f"ON CONFLICT ({target}) DO UPDATE SET {updates}")
        else:
            # Only the conflict columns were supplied -- nothing to update.
            sql = (f'INSERT INTO "{self.table}" ({quoted}) VALUES ({placeholders}) '
                   f"ON CONFLICT ({target}) DO NOTHING")

        self._execute("upsert", sql, tuple(_coerce(v) for v in row.values()))

    def update(self, values: dict[str, Any], **eq_filters: str) -> None:
        if not values:
            return
        sets = ", ".join(f'"{_check_ident(c, "column")}" = %s' for c in values)
        where, params = self._where(eq_filters)
        sql = f'UPDATE "{self.table}" SET {sets} WHERE {where}'
        self._execute("update", sql,
                      tuple(_coerce(v) for v in values.values()) + tuple(params))

    def delete(self, **eq_filters: str) -> None:
        where, params = self._where(eq_filters)
        sql = f'DELETE FROM "{self.table}" WHERE {where}'
        self._execute("delete", sql, tuple(params))


def durable_table(name: str):
    """The durability backend for one table.

    Cloud SQL when configured, otherwise Supabase (which is itself disabled
    -- and therefore efficiently a no-op the stores fall through -- when its
    env vars are unset). Centralising the choice here means a store never
    needs to know which backend it is talking to.
    """
    if postgres_configured():
        return PostgresTable(name)
    from .supabase_kv import SupabaseTable
    return SupabaseTable(name)
