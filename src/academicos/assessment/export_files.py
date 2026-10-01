"""Exported paper files: one file per export, and the paper's latest copy.

Every export rendered into one shared file per paper (`papers/<id>.pdf`)
while GET /papers/{id}/file streamed that same file. Two exports of one paper
at once -- a teacher and the principal, or a double click -- rewrote it
mid-download: downloads ended short of, or ran past, their Content-Length, and
a download could carry another export's traceable-copy id, so the audit could
name the wrong export for a leaked copy (stress test on production's commit,
2026-10-01: 26 torn downloads in 716 exports at 5-60 concurrent users).

Now each export renders into its own directory, named by its watermark id;
the route's URL names that export, so a download is exactly the file that
export produced. The paper's latest copy (what a client fetching
`/papers/{id}/file` without an export id gets) is published by an atomic
replace, and the file route reads a file whole before answering, so no
download is ever a mix of two exports.
"""
from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from pathlib import Path

EXPORT_ID = re.compile(r"^exp_[0-9a-f]{10}$")
SUFFIX = {"pdf": ".pdf", "answer-key": "_answer_key.pdf", "docx": ".docx"}
KEEP_SECONDS = 3600            # an export's own file stays fetchable for an hour


def run_dir(papers_dir: Path, export_id: str) -> Path:
    """The directory one export renders into."""
    d = papers_dir / "exports" / export_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def exported_file(papers_dir: Path, paper_id: str, fmt: str, export_id: str | None) -> Path:
    """The file a download names: that export's own, or the latest copy."""
    name = f"{paper_id}{SUFFIX[fmt]}"
    return papers_dir / "exports" / export_id / name if export_id else papers_dir / name


def publish(papers_dir: Path, produced: Path) -> Path:
    """Make `produced` the paper's latest copy, replacing the old one in one
    step: a reader sees the old file or the new one, never half of each."""
    target = papers_dir / produced.name
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(produced, tmp)
    for _ in range(40):
        try:
            os.replace(tmp, target)
            return target
        except PermissionError:            # Windows: a reader holds the old file open
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)
    raise PermissionError(f"could not publish {target.name}")


def read_whole(path: Path) -> bytes:
    """The file's bytes in one read (exports are tens to hundreds of KB), so
    the response's length is the length of what it sends."""
    for _ in range(40):
        try:
            return path.read_bytes()
        except PermissionError:            # Windows: mid-replace
            time.sleep(0.05)
    return path.read_bytes()


_LAST_SWEEP: dict[str, float] = {}
SWEEP_EVERY = 60.0


def sweep(papers_dir: Path, now: float | None = None) -> None:
    """Drop exports older than KEEP_SECONDS (each export leaves a directory),
    at most once a minute: listing every export directory on every export
    grows with the exports kept, and a busy hour keeps hundreds."""
    now = now or time.time()
    last = _LAST_SWEEP.get(str(papers_dir))
    if last is not None and now - last < SWEEP_EVERY:
        return
    _LAST_SWEEP[str(papers_dir)] = now
    root = papers_dir / "exports"
    if not root.is_dir():
        return
    cutoff = now - KEEP_SECONDS
    for d in root.iterdir():
        try:
            expired = d.is_dir() and d.stat().st_mtime < cutoff
        except FileNotFoundError:          # another sweep removed it: nothing left to do
            continue
        if expired:
            shutil.rmtree(d, ignore_errors=True)
