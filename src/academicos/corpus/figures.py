"""Extract the figures a CBSE CBE item refers to, so the item can be served whole.

The problem this solves
-----------------------
194 of the 722 answerable CBE items (26.9%) say something like "in the given
figure" or "study the map". None of them carried a figure. Serving those as
text alone is worse than not serving them: the question as printed is
unanswerable, and a student sees a question that refers to nothing.

512 images are embedded in the 15 source PDFs. They are recoverable.

Why crop the page rather than extract the embedded object
--------------------------------------------------------
`doc.extract_image(xref)` returns the raw stored object, which may be a mask, a
soft mask, a palette image, or oriented by a page transform. `get_pixmap(clip=)`
renders exactly the rectangle the reader sees, with the page's own rotation and
scaling already applied. For a figure that must look right to a student, the
rendered crop is the correct choice.

Why so much of this module is about refusing to attach an image
---------------------------------------------------------------
A wrong diagram is worse than a missing one. A student shown the wrong figure
will answer the wrong question confidently. So attribution is deliberately
conservative:

  * A logo or a bullet is not a figure. Anything below `_MIN_SIDE_PX` on either
    side, or below `_MIN_AREA_PX`, is dropped before attribution.
  * If a page carries several surviving images and the item spans them all, the
    choice is ambiguous. It is recorded as ambiguous and flagged, not guessed.
  * If an item refers to a figure and nothing is found, that is recorded as
    missing. `figureState` always says which of the three happened.

The three states are `attached`, `ambiguous` and `missing`. There is no state
for "probably this one".
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Iterable, Iterator, Sequence

try:                                       # PyMuPDF renamed the module
    import pymupdf as _fitz
except ImportError:                        # pragma: no cover - older installs
    import fitz as _fitz

# A figure is at least this many pixels on a side, and this many square pixels.
# A CBSE running header logo lands around 40x40; the smallest real figure seen
# in these banks (a number line) is about 200x60.
_MIN_SIDE_PX = 90
_MIN_AREA_PX = 9_000

# Render at this resolution. Figures are line art and small text, so 150 dpi is
# legible without making the asset large; a full-page Science diagram lands
# around 60 KB.
_RENDER_DPI = 150

# Phrasings that make an item depend on a figure it does not contain. This list
# is intentionally the same family the CBE parser uses, extended with the ones
# that only appear alongside an actual image ("the given map", "the graph
# shows").
_FIGURE_PHRASES = (
    "given figure", "given diagram", "given graph", "given map", "given picture",
    "given image", "given table", "figure below", "diagram below", "graph below",
    "map below", "table below", "shown below", "given below", "shown in the",
    "observe the", "study the", "look at the", "refer to the", "from the figure",
    "in the figure", "in the diagram", "as shown", "the figure shows",
    "the diagram shows", "the graph shows", "the map shows", "the table shows",
    "the picture shows",
)

_FIGURE_RE = re.compile("|".join(re.escape(p) for p in _FIGURE_PHRASES), re.I)

# A page footer/header rule, a bullet glyph, or a divider. These are wide but
# very short, or square and tiny; the side and area gates cover most, this
# catches a long thin rule that sneaks past the area gate.
_ASPECT_RULE_MAX = 0.06


@dataclasses.dataclass(frozen=True)
class Figure:
    """One rendered figure, positioned on one page of one PDF."""

    page: int                       # 1-based page number
    bbox: tuple[float, float, float, float]
    png: bytes
    width_px: int
    height_px: int

    @property
    def area_px(self) -> int:
        return self.width_px * self.height_px


@dataclasses.dataclass
class Attribution:
    """What was decided for one item, and on what basis."""

    state: str                      # "attached" | "ambiguous" | "missing" | "none"
    figure: Figure | None = None
    candidates: int = 0
    reason: str = ""


def _plausibly_a_figure(info: dict) -> bool:
    """Reject headers, logos, bullets and rules before they can be attributed."""
    w, h = info.get("width", 0), info.get("height", 0)
    x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
    # Prefer the rendered size, falling back to the stored pixel size.
    pw, ph = abs(x1 - x0), abs(y1 - y0)
    if w < _MIN_SIDE_PX or h < _MIN_SIDE_PX:
        return False
    if w * h < _MIN_AREA_PX:
        return False
    # A near-flat band is a horizontal rule, not a figure.
    if pw > 0 and ph / pw < _ASPECT_RULE_MAX:
        return False
    return True


# --- vector figures -------------------------------------------------------
#
# Many CBSE figures are not images. A geometry diagram -- a labelled triangle,
# a number line, a ray diagram -- is drawn as vector paths, and
# `page.get_image_info()` cannot see it at all. On the real banks this is why
# 39 items referred to a figure whose page appeared to contain no image.
#
# `page.get_drawings()` returns every path. The work is separating a figure
# from everything else that is also vector: the page frame, table rules, the
# answer-lines a question paper prints, and bullet markers.

# A figure is drawn from many segments. A single rule is one or two paths.
_MIN_PATHS = 5

# A cluster this wide AND this tall is the page frame, a full-page table, or
# the ruled answer area -- not a figure.
_MAX_PAGE_FRACTION = 0.92

# Paths closer than this vertically belong to the same figure (PDF points).
_CLUSTER_GAP = 26.0

# A figure must occupy a real area, matching the raster gate's spirit.
_MIN_AREA_PT = 5_000


def _vector_regions(page) -> list[Figure]:
    """Cluster vector paths into candidate figure regions and render them."""
    try:
        drawings = page.get_drawings()
    except Exception:                              # pragma: no cover
        return []
    rects = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.is_empty or rect.is_infinite:
            continue
        if rect.width <= 1 and rect.height <= 1:   # degenerate point
            continue
        rects.append(rect)
    if len(rects) < _MIN_PATHS:
        return []

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    # Cluster by vertical proximity. Sort by top, grow a cluster while the next
    # rect starts before the current cluster's bottom plus the gap.
    rects.sort(key=lambda r: (round(r.y0, 1), round(r.x0, 1)))
    clusters: list[list] = []
    for rect in rects:
        if clusters and rect.y0 <= max(r.y1 for r in clusters[-1]) + _CLUSTER_GAP:
            clusters[-1].append(rect)
        else:
            clusters.append([rect])

    out: list[Figure] = []
    for cluster in clusters:
        if len(cluster) < _MIN_PATHS:
            continue
        region = cluster[0]
        for rect in cluster[1:]:
            region = region | rect                    # bounding union
        if region.width * region.height < _MIN_AREA_PT:
            continue
        if (region.width * region.height) / max(page_area, 1) > _MAX_PAGE_FRACTION:
            continue
        if (region.width >= page_rect.width * 0.9
                and region.height >= page_rect.height * 0.9):
            continue
        # Leave a small margin so labels sitting just outside the paths are
        # included; the figure is being shown to a student, not measured.
        padded = _fitz.Rect(region.x0 - 6, region.y0 - 6,
                            region.x1 + 6, region.y1 + 6) & page_rect
        if padded.is_empty:
            continue
        try:
            pix = page.get_pixmap(clip=padded, dpi=_RENDER_DPI)
            png = pix.tobytes("png")
        except Exception:                          # pragma: no cover
            continue
        out.append(Figure(page=page.number + 1,
                          bbox=(padded.x0, padded.y0, padded.x1, padded.y1),
                          png=png, width_px=pix.width, height_px=pix.height))
    return out


def page_figures(pdf_path: pathlib.Path,
                 *, include_vector: bool = False) -> dict[int, list[Figure]]:
    """Render every plausible figure, keyed by 1-based page number.

    Deterministic: figures on a page are returned top-to-bottom, left-to-right,
    so an ambiguous choice is reproducible.

    `include_vector` is OFF by default, and that is a measured decision, not a
    preference. See `_vector_regions` for what happens when it is turned on:
    naive clustering is too permissive, every page ends up with many candidate
    regions, nearly every item becomes contested, and attribution gets WORSE
    than raster-only (59 attached fall to 2). It is kept in the module because
    the code is sound and the failure is a tuning failure, but it must not be
    enabled until a region can be identified by the item's own text position
    rather than by shape alone.
    """
    out: dict[int, list[Figure]] = {}
    doc = _fitz.open(pdf_path)
    try:
        for index in range(doc.page_count):
            page = doc[index]
            found: list[Figure] = []
            try:
                infos = page.get_image_info()
            except Exception:                     # pragma: no cover - bad page
                infos = []
            for info in infos:
                if not _plausibly_a_figure(info):
                    continue
                rect = _fitz.Rect(info["bbox"])
                if rect.is_empty or rect.is_infinite:
                    continue
                try:
                    pix = page.get_pixmap(clip=rect, dpi=_RENDER_DPI)
                    png = pix.tobytes("png")
                except Exception:                 # pragma: no cover
                    continue
                found.append(Figure(
                    page=index + 1, bbox=tuple(info["bbox"]), png=png,
                    width_px=pix.width, height_px=pix.height,
                ))
            if include_vector:
                found.extend(_vector_regions(page))
            if found:
                found.sort(key=lambda f: (round(f.bbox[1], 1), round(f.bbox[0], 1)))
                # A region found both ways (a raster inside a drawn frame) would
                # otherwise be offered twice and made to look contested.
                deduped: list[Figure] = []
                for fig in found:
                    dup = any(
                        abs(fig.bbox[0] - k.bbox[0]) < 12 and abs(fig.bbox[1] - k.bbox[1]) < 12
                        and abs(fig.bbox[2] - k.bbox[2]) < 12 and abs(fig.bbox[3] - k.bbox[3]) < 12
                        for k in deduped
                    )
                    if not dup:
                        deduped.append(fig)
                out[index + 1] = deduped
    finally:
        doc.close()
    return out


def needs_figure(text: str, has_options: bool = False) -> bool:
    """Does this item's text depend on a figure it does not itself contain?

    `has_options` is accepted and deliberately NOT used to suppress the signal.
    The CBE parser treats an item with options as self-contained; that is wrong
    for an MCQ whose stem says "in the given figure". An option list does not
    supply the figure.
    """
    return bool(_FIGURE_RE.search(text or ""))


def _candidates(pages: Sequence[int],
                figures_by_page: dict[int, list[Figure]]) -> list[Figure]:
    out: list[Figure] = []
    for page in pages:
        out.extend(figures_by_page.get(page, ()))
    return out


def assign_figures(items: Iterable,
                   figures_by_page: dict[int, list[Figure]],
                   ) -> dict[str, Attribution]:
    """Assign figures to items across a whole document, with mutual exclusion.

    Attribution cannot be decided one item at a time. The earlier per-item
    version treated "exactly one image on this page" as a confident 1-to-1
    match, but two figure-referencing items routinely share a page -- and both
    then claimed the same image. Measured on the real banks, that produced
    pairs like `Maths9LK6` and `Maths9CN3`, distinct items with distinct
    figures, both handed the same one at bbox (48, 442, 353, 616). One of them
    was necessarily wrong.

    So the unit of decision is the document, not the item. An image is attached
    only when exactly one item can claim it. An image claimed by two or more
    items is attached to none of them, because a wrong figure teaches wrong
    geometry, and the honest failure is to flag the item rather than guess.

    Returns `item_id -> Attribution`. `none` means the item never referenced a
    figure; `missing` means it did and no image could be assigned.
    """
    items = list(items)
    wants: dict[str, list[Figure]] = {}
    for item in items:
        text = getattr(item, "question", "") or ""
        if not needs_figure(text):
            wants[item.item_id] = []
            continue
        wants[item.item_id] = _candidates(list(getattr(item, "pages", []) or []),
                                          figures_by_page)

    # How many distinct items can claim each image?
    claimants: dict[tuple[int, tuple], int] = {}
    for figures in wants.values():
        for figure in set(figures):
            key = (figure.page, figure.bbox)
            claimants[key] = claimants.get(key, 0) + 1

    results: dict[str, Attribution] = {}
    for item in items:
        figures = wants[item.item_id]
        if not figures:
            text = getattr(item, "question", "") or ""
            results[item.item_id] = (
                Attribution("missing", reason="no image on the item's pages")
                if needs_figure(text) else
                Attribution("none", reason="no figure reference")
            )
            continue
        unique = sorted({f for f in figures}, key=lambda f: (f.page, round(f.bbox[1], 1)))
        uncontested = [f for f in unique if claimants[(f.page, f.bbox)] == 1]
        if len(unique) == 1 and uncontested:
            results[item.item_id] = Attribution(
                "attached", figure=unique[0], candidates=1,
                reason="the only image on the item's pages, claimed by no other item")
        elif len(uncontested) == 1:
            results[item.item_id] = Attribution(
                "attached", figure=uncontested[0], candidates=len(unique),
                reason="one unclaimed image among "
                       f"{len(unique)} on the item's pages")
        else:
            results[item.item_id] = Attribution(
                "ambiguous", candidates=len(unique),
                reason=f"no image is uniquely assignable "
                       f"({len(uncontested)} uncontested of {len(unique)})")
    return results


def extract_document(pdf_path: pathlib.Path,
                     items: Iterable,
                     out_dir: pathlib.Path) -> dict[str, Attribution]:
    """Assign and write figures for `items` from one PDF.

    `items` must expose `item_id`, `pages` and `question` (the `CbeItem` shape).
    Returns `item_id -> Attribution`. Files are written as
    `<out_dir>/<item_id>.png` only when the state is `attached`.
    """
    figures_by_page = page_figures(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = assign_figures(items, figures_by_page)
    for item_id, verdict in results.items():
        target = out_dir / f"{_safe(item_id)}.png"
        if verdict.state == "attached" and verdict.figure is not None:
            target.write_bytes(verdict.figure.png)
        elif target.exists():
            # An item that no longer has an assignable figure must not keep the
            # previous run's image. Measured once at 63 files for 59
            # attachments: the stale file would have been served as this item's
            # figure, which is exactly the wrong-figure failure this module
            # exists to prevent.
            target.unlink()
    return results


_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(name: str) -> str:
    """A filesystem-safe asset name. Item ids are alphanumeric but be certain."""
    return _SAFE_RE.sub("_", name)[:120]


# --- position-based attribution ------------------------------------------
#
# Shape alone could not identify a figure. `_vector_regions` clusters paths by
# proximity, which is also how it clusters a text box, a table and the ruled
# answer area, so every page produced many candidates, everything became
# contested, and attribution measured WORSE than raster-only (59 attached fell
# to 2).
#
# The signal that was missing is the item's own position on the page. A figure
# sits near the question that refers to it, so the nearest region to the item's
# text is a far better candidate than any region selected on shape. This is
# what a reader does without thinking about it.
#
# Two filters keep the vector noise from winning:
#   * a region containing a sentence of text is a text box, not a figure --
#     figures carry only short labels, so a character count separates them
#   * a region overlapping the item's own anchor is the text, not a figure

# A figure's own labels are short. A text box holds sentences. This is the line
# between them, measured in characters inside the region.
#
# 60 was too high and let a wrong figure through: the item metadata table that
# heads every CBE item reads "Subject | Class | Question reference/Filename"
# plus its values, which is about 55 characters -- just under the line. Verified
# by eye against the extracted asset, which was the table and not a figure.
_MAX_LABEL_CHARS = 45

# Every CBE item is headed by the same metadata table. It is page furniture, and
# it sits right beside the item's own text, so proximity prefers it. Caught by
# its content rather than its shape: it names the item, and it always says
# "Question reference".
_FURNITURE_RE = re.compile(r"question\s*reference|subject\s*class", re.I)

# Beyond this vertical gap a region is not "the figure for this item".
_MAX_GAP_PT = 420.0


def _page_words(page) -> list:
    """Word boxes for the page, used to detect text inside a region."""
    try:
        return page.get_text("words")
    except Exception:                              # pragma: no cover
        return []


def _text_inside(rect, words) -> str:
    """The text whose centre falls inside `rect`.

    A figure with axis labels holds a handful of characters; a paragraph holds
    hundreds; the metadata table holds a sentence and names the item.
    """
    parts = []
    for x0, y0, x1, y1, word, *_rest in words:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1:
            parts.append(word)
    return " ".join(parts)


def _is_furniture(text: str, item_id: str) -> bool:
    """Is this region page furniture rather than a figure?

    The metadata table heads every item, so no page can be parsed by rejecting
    it globally -- it must be rejected per region, which is why this takes the
    item id.
    """
    if not text:
        return False
    if _FURNITURE_RE.search(text):
        return True
    return bool(item_id) and item_id in text


def _find_anchor(page, text: str):
    """Where the item's own text sits on this page, or None.

    Uses the longest token in the stem as the probe. The longest word is the
    most distinctive one, and `search_for` on a single token survives the
    line-breaks and ligatures that break a multi-word phrase search.
    """
    tokens = sorted({w for w in re.findall(r"[A-Za-z][A-Za-z0-9]{4,}", text or "")},
                    key=lambda w: (-len(w), w))
    for token in tokens[:6]:
        try:
            hits = page.search_for(token)
        except Exception:                          # pragma: no cover
            continue
        if hits:
            return hits[0]
    return None


def _gap_from(anchor, rect) -> float:
    """Vertical gap between the item's text and a region, in points.

    Zero when they overlap vertically; otherwise the distance between them.
    """
    if rect.y1 < anchor.y0:
        return anchor.y0 - rect.y1
    if rect.y0 > anchor.y1:
        return rect.y0 - anchor.y1
    return 0.0


def attribute_by_position(pdf_path: pathlib.Path,
                          items: Iterable,
                          out_dir: pathlib.Path,
                          *, include_vector: bool = True) -> dict[str, Attribution]:
    """Assign figures by proximity to the item that refers to them.

    Greedy over `(item, region)` pairs sorted by gap, with mutual exclusion: an
    image two items can claim is attached to neither unless proximity
    distinguishes them. Ties are broken deterministically so a build is
    reproducible.

    `include_vector` defaults to True here, unlike `page_figures`, because the
    nearest-region rule fixes what made vector clustering useless on its own --
    38 items referred to a figure whose page held no raster image at all,
    because CBSE draws most mathematics figures as vector paths.
    """
    items = list(items)
    doc = _fitz.open(pdf_path)
    try:
        needs = []
        for item in items:
            text = getattr(item, "question", "") or ""
            if not needs_figure(text):
                continue
            pages = [p for p in (getattr(item, "pages", []) or []) if 1 <= p <= doc.page_count]
            if pages:
                needs.append((item, pages))

        # Anchor each item, and gather the candidate regions on its pages.
        words_by_page: dict[int, list] = {}
        regions_by_page: dict[int, list] = {}
        for _item, pages in needs:
            for page_no in pages:
                if page_no not in regions_by_page:
                    page = doc[page_no - 1]
                    regions = _raster_regions(page)
                    if include_vector:
                        regions.extend(_vector_regions(page))
                    regions_by_page[page_no] = regions
                    words_by_page[page_no] = _page_words(page)

        pairs = []
        for item, pages in needs:
            anchor = None
            for page_no in pages:
                anchor = _find_anchor(doc[page_no - 1], getattr(item, "question", "") or "")
                if anchor is not None:
                    anchor_page = page_no
                    break
            if anchor is None:
                continue
            for page_no in pages:
                words = words_by_page.get(page_no, [])
                for index, region in enumerate(regions_by_page.get(page_no, [])):
                    rect = _fitz.Rect(region.bbox)
                    gap = _gap_from(anchor, rect) + (0.0 if page_no == anchor_page else 60.0)
                    if gap > _MAX_GAP_PT:
                        continue
                    text = _text_inside(rect, words)
                    # A region holding a sentence is a text box.
                    if len(text) > _MAX_LABEL_CHARS:
                        continue
                    # The item metadata table heads every item and sits beside
                    # it, so proximity prefers it. Verified by eye as a wrong
                    # figure before this filter existed.
                    if _is_furniture(text, item.item_id):
                        continue
                    # A region overlapping the item's own text is the text.
                    if anchor.intersects(rect) and gap == 0.0:
                        continue
                    pairs.append((gap, item.item_id, page_no, index))

        # Greedy nearest-first with mutual exclusion.
        pairs.sort(key=lambda p: (round(p[0], 2), p[1], p[2], p[3]))
        taken_items: set[str] = set()
        taken_regions: set[tuple[int, int]] = set()
        assigned: dict[str, tuple[int, int, float]] = {}
        for gap, item_id, page_no, index in pairs:
            if item_id in taken_items or (page_no, index) in taken_regions:
                continue
            taken_items.add(item_id)
            taken_regions.add((page_no, index))
            assigned[item_id] = (page_no, index, gap)

        results: dict[str, Attribution] = {}
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            text = getattr(item, "question", "") or ""
            item_id = item.item_id
            target = out_dir / f"{_safe(item_id)}.png"
            if not needs_figure(text):
                results[item_id] = Attribution("none", reason="no figure reference")
                if target.exists():
                    target.unlink()
                continue
            if item_id in assigned:
                page_no, index, gap = assigned[item_id]
                region = regions_by_page[page_no][index]
                target.write_bytes(region.png)
                results[item_id] = Attribution(
                    "attached", figure=region, candidates=1,
                    reason=f"nearest region, {gap:.0f}pt from the item's text")
            else:
                results[item_id] = Attribution(
                    "ambiguous" if regions_by_page else "missing",
                    candidates=sum(len(v) for v in regions_by_page.values()),
                    reason="no region within reach, or all were claimed")
                if target.exists():
                    target.unlink()
        return results
    finally:
        doc.close()


def _raster_regions(page) -> list[Figure]:
    """The embedded images on one page that could be a figure."""
    out: list[Figure] = []
    try:
        infos = page.get_image_info()
    except Exception:                              # pragma: no cover
        return out
    for info in infos:
        if not _plausibly_a_figure(info):
            continue
        rect = _fitz.Rect(info["bbox"])
        if rect.is_empty or rect.is_infinite:
            continue
        try:
            pix = page.get_pixmap(clip=rect, dpi=_RENDER_DPI)
        except Exception:                          # pragma: no cover
            continue
        out.append(Figure(page=page.number + 1, bbox=tuple(info["bbox"]),
                          png=pix.tobytes("png"),
                          width_px=pix.width, height_px=pix.height))
    return out
