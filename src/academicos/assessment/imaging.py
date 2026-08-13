"""Mobile capture preprocessing: auto-crop + lighting enhancement.

A phone photo of an answer sheet is never a clean scan — the page is at an
angle, the desk shows around the edges, and lighting is uneven across the
frame. Both problems measurably hurt OCR accuracy if left uncorrected, so
this runs before every captured page reaches Sarvam Vision:

1. `find_document_contour` / `crop_to_document` — standard OpenCV document-
   scanner technique: edge detection, find the largest four-sided contour,
   perspective-warp it flat. Falls back to a plain center-crop when no
   confident quadrilateral is found (a photo taken very close to the page,
   or a low-contrast desk background) rather than distorting the image.
2. `enhance_lighting` — CLAHE (contrast-limited adaptive histogram
   equalization) on the luminance channel, which evens out a shadow falling
   across half the page without blowing out the lit half the way a naive
   global contrast stretch would.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_MIN_DOCUMENT_AREA_FRACTION = 0.20  # a found quad smaller than this is probably noise


@dataclass
class PreprocessResult:
    path: Path
    cropped: bool
    warnings: list


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sorts 4 points into top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],       # top-left: smallest x+y
        pts[np.argmin(diff)],    # top-right: smallest y-x
        pts[np.argmax(s)],       # bottom-right: largest x+y
        pts[np.argmax(diff)],    # bottom-left: largest y-x
    ], dtype="float32")


def find_document_contour(gray) -> np.ndarray | None:
    """Largest plausible 4-point page contour in a grayscale image, or None."""
    import cv2

    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = h * w
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) / frame_area >= _MIN_DOCUMENT_AREA_FRACTION:
            best = approx.reshape(4, 2).astype("float32")
            break
    return best


def crop_to_document(image_path: Path, out_path: Path) -> PreprocessResult:
    """Perspective-corrects a photographed page to a flat, cropped rectangle."""
    import cv2

    warnings: list[str] = []
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = find_document_contour(gray)

    if quad is None:
        # No confident page edge found — safer to keep the full frame than to
        # guess a crop that might cut off written answers.
        warnings.append("page edges not detected; used the full photo uncropped")
        cv2.imwrite(str(out_path), img)
        return PreprocessResult(path=out_path, cropped=False, warnings=warnings)

    corners = _order_corners(quad)
    (tl, tr, br, bl) = corners
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    width, height = max(width, 100), max(height, 100)

    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    cv2.imwrite(str(out_path), warped)
    return PreprocessResult(path=out_path, cropped=True, warnings=warnings)


def enhance_lighting(image_path: Path, out_path: Path) -> Path:
    """CLAHE on the L channel of LAB color space — evens out shadows/glare
    without washing out ink contrast the way global brightness/contrast would."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(out_path), result)
    return out_path


def preprocess_page(image_path: Path, workdir: Path, page_no: int) -> PreprocessResult:
    """Full pipeline for one captured page: crop to document, then even lighting."""
    workdir.mkdir(parents=True, exist_ok=True)
    cropped_path = workdir / f"cropped_{page_no:03d}.jpg"
    result = crop_to_document(image_path, cropped_path)
    enhanced_path = workdir / f"enhanced_{page_no:03d}.jpg"
    enhance_lighting(result.path, enhanced_path)
    return PreprocessResult(path=enhanced_path, cropped=result.cropped, warnings=result.warnings)
