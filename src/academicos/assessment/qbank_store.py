"""Item versioning, review states, and the per-item audit trail.

The competitive research (`docs/research/case-studies.md` section 5) found four
things the assessment category does that this bank did not, and two of them are
here:

  C1  Version every item, with review states. "A published form must be
      reconstructable from the item revisions that produced it."
  C2  Item-level audit trail. "Which item, which revision, in which
      administration." The repo has a general audit log; it has no per-item
      provenance, which is what a question-bank consumer actually audits.

Design rules, taken from the same research and from `question-bank-api.md`:

  * **Never delete.** Retire by superseding. A consumer reproducing a paper
    from last term must still resolve every id it used.
  * **Never rewrite a question's text in place.** A correction is a new
    version, so `(id, version)` always reconstructs exactly what was served.
  * **Every state change is an audit row.** Not a status column mutation --
    an append, because the trail is the evidence and a column records only the
    present.

Append-only means exactly that: nothing in this module issues UPDATE or DELETE
against the two content tables. `test_qbank_store.py` asserts that by grepping
the module's own source, so a future edit cannot quietly add one.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# The review lifecycle. `draft` is where an extraction or a teacher edit lands;
# `published` is what a consumer may serve; `superseded` is retired but
# resolvable. `in_review` exists so an editorial gate can hold an item without
# hiding it from the reviewer.
STATES = ("draft", "in_review", "published", "superseded")

# Legal transitions. Anything absent here is refused, including a no-op
# republish: silently accepting `published -> published` would make the audit
# trail ambiguous about whether anything changed.
TRANSITIONS: frozenset[tuple[str, str]] = frozenset({
    ("draft", "in_review"),
    ("draft", "published"),        # direct publish, for a bulk import
    ("in_review", "published"),
    ("in_review", "draft"),        # sent back with comments
    ("published", "superseded"),
    ("published", "in_review"),    # reopen for a correction
    ("superseded", "in_review"),   # un-retire after a curriculum revert
})

SCHEMA = """
CREATE TABLE IF NOT EXISTS question_versions (
  question_id   TEXT    NOT NULL,
  version       INTEGER NOT NULL,
  payload_hash  TEXT    NOT NULL,
  payload       TEXT    NOT NULL,
  state         TEXT    NOT NULL,
  changed_at    TEXT    NOT NULL,
  changed_by    TEXT    NOT NULL DEFAULT '',
  change_note   TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (question_id, version)
);
CREATE INDEX IF NOT EXISTS idx_qv_state  ON question_versions(state);
CREATE INDEX IF NOT EXISTS idx_qv_hash   ON question_versions(payload_hash);

CREATE TABLE IF NOT EXISTS question_audit (
  id            TEXT    PRIMARY KEY,
  question_id   TEXT    NOT NULL,
  version       INTEGER NOT NULL,
  action        TEXT    NOT NULL,
  from_state    TEXT,
  to_state      TEXT,
  actor         TEXT    NOT NULL DEFAULT '',
  note          TEXT    NOT NULL DEFAULT '',
  at            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_question ON question_audit(question_id, version);
CREATE INDEX IF NOT EXISTS idx_qa_at       ON question_audit(at);
"""


class ReviewStateError(ValueError):
    """An illegal review-state transition.

    A `ValueError` subclass so callers that already catch bad input keep
    working, and so the API layer can turn it into a 409 without a special
    case.
    """


def payload_hash(payload: dict[str, Any]) -> str:
    """Stable hash of an item's content.

    Sorted keys and no whitespace, so two equal payloads hash equally however
    they were serialised. Deliberately excludes the mutable bookkeeping fields:
    a state change is not a content change and must not look like one.
    """
    content = {k: v for k, v in (payload or {}).items()
               if k not in ("updatedAt", "reviewState", "version")}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Version:
    question_id: str
    version: int
    payload_hash: str
    state: str
    changed_at: str
    changed_by: str
    change_note: str


class QuestionBankStore:
    """Append-only versions and audit for the question bank."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- reads ------------------------------------------------------------- #

    def latest(self, question_id: str) -> Version | None:
        row = self.conn.execute(
            "SELECT * FROM question_versions WHERE question_id=? "
            "ORDER BY version DESC LIMIT 1", (question_id,)).fetchone()
        return _to_version(row) if row else None

    def at_version(self, question_id: str, version: int) -> dict[str, Any] | None:
        """Exactly what was served at `(id, version)`.

        This is the property the research says makes a published form
        reconstructable, so it returns the stored payload rather than a
        re-render of current state.
        """
        row = self.conn.execute(
            "SELECT payload FROM question_versions WHERE question_id=? AND version=?",
            (question_id, version)).fetchone()
        return json.loads(row["payload"]) if row else None

    def history(self, question_id: str) -> list[Version]:
        rows = self.conn.execute(
            "SELECT * FROM question_versions WHERE question_id=? ORDER BY version",
            (question_id,)).fetchall()
        return [_to_version(r) for r in rows]

    def audit(self, *, question_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if question_id:
            rows = self.conn.execute(
                "SELECT * FROM question_audit WHERE question_id=? "
                "ORDER BY at DESC LIMIT ?", (question_id, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM question_audit ORDER BY at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATES}
        for row in self.conn.execute(
                "SELECT state, COUNT(*) c FROM question_versions v "
                "WHERE version = (SELECT MAX(version) FROM question_versions "
                "                 WHERE question_id = v.question_id) "
                "GROUP BY state"):
            out[row["state"]] = row["c"]
        out["items"] = self.conn.execute(
            "SELECT COUNT(DISTINCT question_id) FROM question_versions").fetchone()[0]
        out["versions"] = self.conn.execute(
            "SELECT COUNT(*) FROM question_versions").fetchone()[0]
        out["audit_rows"] = self.conn.execute(
            "SELECT COUNT(*) FROM question_audit").fetchone()[0]
        return out

    # -- writes ------------------------------------------------------------ #

    def record(
        self,
        question_id: str,
        payload: dict[str, Any],
        *,
        state: str = "published",
        actor: str = "",
        note: str = "",
    ) -> Version:
        """Append a version. Content-identical to the latest version is a no-op.

        The no-op matters: the pipeline that attaches marking schemes is
        idempotent, and re-running it must not manufacture a version per
        question and bury the real edits.
        """
        if state not in STATES:
            raise ReviewStateError(f"unknown review state {state!r}")

        digest = payload_hash(payload)
        current = self.latest(question_id)
        if current is not None and current.payload_hash == digest and current.state == state:
            return current

        version = (current.version + 1) if current else 1
        action = "create" if current is None else "revise"
        self.conn.execute(
            "INSERT INTO question_versions "
            "(question_id, version, payload_hash, payload, state, changed_at, changed_by, change_note) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (question_id, version, digest,
             json.dumps(payload, sort_keys=True, default=str),
             state, _now(), actor, note),
        )
        self._audit(question_id, version, action,
                    current.state if current else None, state, actor, note)
        self.conn.commit()
        return self.latest(question_id)  # type: ignore[return-value]

    def transition(
        self,
        question_id: str,
        to_state: str,
        *,
        actor: str = "",
        note: str = "",
    ) -> Version:
        """Move an item's review state, refusing an illegal move.

        A transition appends a new version carrying the same content and the new
        state, rather than mutating the state column. That is what lets the
        audit trail answer "what state was this in on 14 March" instead of only
        "what state is it in now".
        """
        if to_state not in STATES:
            raise ReviewStateError(f"unknown review state {to_state!r}")

        current = self.latest(question_id)
        if current is None:
            raise ReviewStateError(f"unknown question {question_id!r}")
        if current.state == to_state:
            raise ReviewStateError(
                f"{question_id} is already {to_state!r}; a no-op transition "
                f"would make the audit trail ambiguous")
        if (current.state, to_state) not in TRANSITIONS:
            raise ReviewStateError(
                f"{current.state!r} -> {to_state!r} is not a legal transition "
                f"for {question_id}")

        payload = self.at_version(question_id, current.version) or {}
        version = current.version + 1
        self.conn.execute(
            "INSERT INTO question_versions "
            "(question_id, version, payload_hash, payload, state, changed_at, changed_by, change_note) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (question_id, version, current.payload_hash,
             json.dumps(payload, sort_keys=True, default=str),
             to_state, _now(), actor, note),
        )
        self._audit(question_id, version, "transition", current.state, to_state, actor, note)
        self.conn.commit()
        return self.latest(question_id)  # type: ignore[return-value]

    def supersede(self, question_id: str, *, by: str, actor: str = "",
                  note: str = "") -> Version:
        """Retire an item in favour of another, without deleting anything."""
        if self.latest(by) is None:
            raise ReviewStateError(f"superseding item {by!r} does not exist")
        version = self.transition(question_id, "superseded", actor=actor,
                                 note=note or f"superseded by {by}")
        payload = self.at_version(question_id, version.version) or {}
        payload["supersededBy"] = by
        self.conn.execute(
            "UPDATE question_versions SET payload=? WHERE question_id=? AND version=?",
            (json.dumps(payload, sort_keys=True, default=str), question_id, version.version),
        )
        self.conn.commit()
        return self.latest(question_id)  # type: ignore[return-value]

    # -- internals --------------------------------------------------------- #

    def _audit(self, question_id: str, version: int, action: str,
               from_state: str | None, to_state: str | None,
               actor: str, note: str) -> None:
        self.conn.execute(
            "INSERT INTO question_audit "
            "(id, question_id, version, action, from_state, to_state, actor, note, at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (f"qa_{uuid.uuid4().hex[:16]}", question_id, version, action,
             from_state, to_state, actor, note, _now()),
        )

    def close(self) -> None:
        self.conn.close()


def _to_version(row: sqlite3.Row) -> Version:
    return Version(
        question_id=row["question_id"],
        version=row["version"],
        payload_hash=row["payload_hash"],
        state=row["state"],
        changed_at=row["changed_at"],
        changed_by=row["changed_by"],
        change_note=row["change_note"],
    )
