"""Syncs scheduled assessments to the school's real Google Calendar via Composio.

Runs server-side ("internally", per the request that prompted this) rather
than from the Flutter client — the client has no Composio credentials and
never will; only this backend does. Best-effort: a Calendar sync failure
never blocks saving the assessment, it's just logged.

Requires three env vars:
  COMPOSIO_API_KEY                    from app.composio.dev -> API Keys
  COMPOSIO_CALENDAR_CONNECTED_ACCOUNT_ID   the connected Google Calendar
                                       account's id (ca_...), obtained once
                                       via the OAuth link Composio issues.
  COMPOSIO_CALENDAR_USER_ID           the user_id passed when creating that
                                       connection (see connected_accounts/link)
                                       -- Composio's execute endpoint rejects
                                       connected_account_id without it
                                       (confirmed live: error code 1811,
                                       "ActionExecute_ConnectedAccountEntityIdRequired").
Without all three set, sync_to_google_calendar is a no-op.
"""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from ..assessment.schemas import Assessment
from .composio_provider import ComposioProvider

log = logging.getLogger(__name__)

_CREATE_EVENT_TOOL = "GOOGLECALENDAR_CREATE_EVENT"


def _api_key() -> str | None:
    return os.environ.get("COMPOSIO_API_KEY", "").strip() or None


def _connected_account_id() -> str | None:
    return os.environ.get("COMPOSIO_CALENDAR_CONNECTED_ACCOUNT_ID", "").strip() or None


def _user_id() -> str | None:
    return os.environ.get("COMPOSIO_CALENDAR_USER_ID", "").strip() or None


def sync_to_google_calendar(assessment: Assessment) -> None:
    """Creates (once) a Google Calendar event for a scheduled assessment.

    Idempotent per-assessment: the created event's id is stashed in
    assessment.metadata['googleCalendarEventId'] by the caller so a second
    call (e.g. re-saving the same assessment) does not duplicate the event.
    """
    if assessment.scheduled_at is None:
        return
    if assessment.metadata.get("googleCalendarEventId"):
        return  # already synced

    key = _api_key()
    account_id = _connected_account_id()
    user_id = _user_id()
    if not key or not account_id or not user_id:
        log.info("Composio Calendar sync skipped for %s: not configured "
                 "(COMPOSIO_API_KEY / COMPOSIO_CALENDAR_CONNECTED_ACCOUNT_ID / "
                 "COMPOSIO_CALENDAR_USER_ID)", assessment.id)
        return

    start = assessment.scheduled_at
    duration_min = assessment.blueprint.duration_minutes or 60
    end = start + timedelta(minutes=duration_min)

    arguments: dict[str, Any] = {
        "summary": f"{assessment.title} — {assessment.subject} (Grade {assessment.grade})",
        "description": (
            f"AcademicOS assessment: {assessment.title}\n"
            f"Subject: {assessment.subject} · Grade {assessment.grade}\n"
            f"Total marks: {assessment.blueprint.total_marks}"
        ),
        "start_datetime": start.isoformat(),
        "end_datetime": end.isoformat(),
        "calendar_id": "primary",
    }
    # Not event_duration_hour/event_duration_minutes: confirmed live that
    # Composio's GOOGLECALENDAR_CREATE_EVENT rejects event_duration_minutes
    # above 59 ("Input should be less than or equal to 59"), which a 3-hour
    # board exam's duration_minutes (180) always exceeds. start/end_datetime
    # together avoid the field entirely and were confirmed working directly
    # against the live API.

    result = ComposioProvider(api_key=key).execute(
        _CREATE_EVENT_TOOL, arguments, user_id=user_id, connected_account_id=account_id,
    )
    # response_data shape confirmed live: {"id": "...", "htmlLink": "...", ...}.
    # A non-dict response_data or a missing "id" is treated as "no event id",
    # same as any other transport failure -- logged, never raised.
    event_id = result.payload.get("id") if result.ok and isinstance(result.payload, dict) else None
    if event_id:
        assessment.metadata["googleCalendarEventId"] = event_id
        log.info("synced assessment %s to Google Calendar as event %s",
                 assessment.id, event_id)
    elif result.ok:
        log.warning("Composio Calendar sync for %s returned no event id: %s",
                   assessment.id, result.payload)
    else:
        log.warning("Composio Calendar sync failed for %s: %s", assessment.id, result.error)
