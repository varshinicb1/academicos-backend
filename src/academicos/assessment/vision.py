"""Sarvam Vision (doc-digitization) client — handwriting OCR for answer sheets.

Job pipeline: create -> request upload URLs -> PUT file -> start -> poll -> download ZIP.

Two things learned from running this against real evaluated CBSE scripts, both
handled here:

1. On a region containing no prose (a QR code / barcode margin) the VLM
   hallucinates — it returned an essay about "historical Indic scripts,
   particularly Malayalam" instead of empty output. `crop_margins` removes the
   answer-booklet furniture before OCR, and `looks_hallucinated` catches what
   slips through.
2. Struck-through working (a cancelled `220` with `44` written above) gets
   merged into a single token like `44220`. That is flagged, not silently
   trusted — see `evaluate.py`, which routes low-confidence answers to the
   teacher rather than auto-scoring them.
"""
from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.sarvam.ai/doc-digitization/job/v1"
_POLL_INTERVAL_SEC = 5
_POLL_MAX_ATTEMPTS = 60

# Phrases the VLM emits when it is describing the image instead of transcribing it.
_HALLUCINATION_MARKERS = (
    "i am an expert ocr engine",
    "i cannot perform the requested task",
    "does not contain any human-readable text",
    "the provided image is a qr code",
    "as an ai language model",
)


class VisionError(RuntimeError):
    pass


@dataclass
class PageOCR:
    page_no: int
    text: str
    hallucinated: bool = False
    warnings: list[str] = field(default_factory=list)


def api_key() -> str:
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not key:
        raise VisionError("SARVAM_API_KEY is not set")
    return key


def looks_hallucinated(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _HALLUCINATION_MARKERS)


def strip_hallucinations(text: str) -> tuple[str, bool]:
    """Drop paragraphs where the model narrated instead of transcribing."""
    blocks = [b for b in text.split("\n\n")]
    kept, found = [], False
    for b in blocks:
        if looks_hallucinated(b):
            found = True
            continue
        kept.append(b)
    return "\n\n".join(kept).strip(), found


def crop_margins(image_path: Path, out_path: Path,
                 left: float = 0.10, right: float = 0.02,
                 top: float = 0.12, bottom: float = 0.02) -> Path:
    """Trim answer-booklet furniture (QR code, 'Space for writing Question
    Number' rail, punch holes) so the VLM only sees the written answer."""
    try:
        from PIL import Image
    except ImportError:  # Pillow ships with reportlab; degrade gracefully
        log.warning("Pillow unavailable, skipping margin crop")
        return image_path
    with Image.open(image_path) as im:
        w, h = im.size
        box = (int(w * left), int(h * top), int(w * (1 - right)), int(h * (1 - bottom)))
        im.crop(box).save(out_path)
    return out_path


def extract(path: Path, *, language: str = "en-IN", output_format: str = "md",
            timeout_sec: int = _POLL_INTERVAL_SEC * _POLL_MAX_ATTEMPTS) -> str:
    """Run one document (PDF or image, <=10 pages) through Sarvam Vision."""
    key = api_key()
    headers = {"api-subscription-key": key}
    json_headers = {**headers, "Content-Type": "application/json"}
    fname = path.name

    r = requests.post(BASE_URL, headers=json_headers,
                      json={"job_parameters": {"language": language,
                                               "output_format": output_format}})
    r.raise_for_status()
    job_id = r.json()["job_id"]

    r = requests.post(f"{BASE_URL}/upload-files", headers=json_headers,
                      json={"job_id": job_id, "files": [fname]})
    r.raise_for_status()
    put_url = _first_upload_url(r.json(), fname)

    with path.open("rb") as f:
        pr = requests.put(put_url, data=f, headers={"x-ms-blob-type": "BlockBlob"})
    pr.raise_for_status()

    r = requests.post(f"{BASE_URL}/{job_id}/start", headers=json_headers, json={})
    r.raise_for_status()

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL_SEC)
        st = requests.get(f"{BASE_URL}/{job_id}/status", headers=headers).json()
        state = (st.get("job_state") or "").lower()
        if state in ("completed", "succeeded", "success"):
            return _download_text(job_id, headers, json_headers)
        if state in ("failed", "error"):
            raise VisionError(f"vision job {job_id} failed: {st.get('error_message')}")
    raise VisionError(f"vision job {job_id} timed out after {timeout_sec}s")


def _first_upload_url(payload: dict, fname: str) -> str:
    urls = payload.get("upload_urls") or payload.get("files") or {}
    entry = None
    if isinstance(urls, dict):
        entry = urls.get(fname) or (next(iter(urls.values())) if urls else None)
    elif isinstance(urls, list) and urls:
        entry = urls[0]
    if entry is None:
        raise VisionError(f"no upload url in response: {list(payload)}")
    if isinstance(entry, str):
        return entry
    url = entry.get("file_url") or entry.get("url") or entry.get("upload_url")
    if not url:
        raise VisionError(f"could not read upload url from {entry}")
    return url


def _download_text(job_id: str, headers: dict, json_headers: dict) -> str:
    r = requests.post(f"{BASE_URL}/{job_id}/download-files", headers=json_headers,
                      json={"job_id": job_id})
    r.raise_for_status()
    downloads = r.json().get("download_urls", {})
    if not downloads:
        raise VisionError("no download urls returned")
    entry = next(iter(downloads.values()))
    url = entry["file_url"] if isinstance(entry, dict) else entry
    blob = requests.get(url).content

    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith((".md", ".txt", ".html")):
                parts.append(zf.read(name).decode("utf-8", "replace"))
    return "\n\n".join(parts).strip()


_VOTE_RUNS = 3


def _normalize_for_vote(text: str) -> str:
    return " ".join(text.split()).lower()


def _vote(candidates: list[str]) -> tuple[str, bool]:
    """Picks the OCR result to trust from repeated passes over the same image.

    Handwriting OCR rarely reproduces byte-identical text across passes, so
    "majority vote" means: if >=2 passes normalize to the exact same text,
    that's the answer — a real, reproducible read, not a fluke. Otherwise no
    two passes agreed; fall back to the longest result (a truncated/garbled
    pass is usually the shortest) and flag the disagreement so the teacher
    reviewing this answer knows the transcription is unconfirmed.
    """
    if not candidates:
        return "", False
    if len(candidates) == 1:
        return candidates[0], False
    normalized = [_normalize_for_vote(c) for c in candidates]
    for i, norm_i in enumerate(normalized):
        if not norm_i:
            continue
        agreeing = [candidates[j] for j, norm_j in enumerate(normalized) if norm_j == norm_i]
        if len(agreeing) >= 2:
            return agreeing[0], False
    return max(candidates, key=len), True


def ocr_page(image_path: Path, page_no: int, *, workdir: Path,
             language: str = "en-IN", crop_furniture: bool = True) -> PageOCR:
    """OCR one answer-sheet page, with margin cropping + hallucination scrubbing.

    Runs Sarvam Vision `_VOTE_RUNS` times over the same image and votes on the
    result (see `_vote`) rather than trusting a single pass — handwriting OCR
    is exactly the kind of task where one bad read (a skipped line, a merged
    struck-through digit) silently costs a student their marks, and running it
    once gives no way to notice that happened.

    `crop_furniture` trims a fixed percentage off each edge to remove the
    *official CBSE booklet's* fixed layout — QR code, "Space for writing
    Question Number" rail — confirmed against real evaluated scripts in
    `scan.py`'s render-from-PDF flow. A mobile-captured photo has no such
    fixed layout, and by the time it reaches here `imaging.py`'s document-edge
    detection has already cropped it to the actual page content; applying
    this fixed percentage crop on top trims real handwriting near the edges
    instead of furniture — confirmed by testing a photo whose first five
    answers, all in the top 12% of the frame, were silently cut before OCR
    ever ran. Callers on the mobile capture path must pass `False`.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    source = image_path
    if crop_furniture:
        source = crop_margins(image_path, workdir / f"crop_{page_no}.png")

    warnings: list[str] = []
    hallucinated = False
    passes: list[str] = []
    for attempt in range(_VOTE_RUNS):
        try:
            raw = extract(source, language=language)
        except VisionError as exc:
            log.warning("vision pass %d/%d failed for page %s: %s",
                       attempt + 1, _VOTE_RUNS, page_no, exc)
            continue
        cleaned, was_hallucinated = strip_hallucinations(raw)
        hallucinated = hallucinated or was_hallucinated
        if cleaned:
            passes.append(cleaned)

    if not passes:
        warnings.append("no text recovered from this page")
        return PageOCR(page_no=page_no, text="", hallucinated=hallucinated, warnings=warnings)

    text, disagreed = _vote(passes)
    if len(passes) < _VOTE_RUNS:
        warnings.append(f"only {len(passes)}/{_VOTE_RUNS} vision passes succeeded for this page")
    if disagreed:
        warnings.append("vision passes disagreed on this page's text — transcription unconfirmed, please check")
    if hallucinated:
        warnings.append("model narrated a non-text region; that output was discarded")
    return PageOCR(page_no=page_no, text=text, hallucinated=hallucinated, warnings=warnings)
