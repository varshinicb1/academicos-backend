"""CBSE-format question paper renderer.

Lays a generated paper out the way a board paper actually prints, because a
school judges the product on whether the paper looks like one of theirs:

  * school header band (name, logo, exam title) from the school's template
  * Roll No. grid + Q.P. Code box, as on a real answer booklet
  * Time Allowed / Maximum Marks rule
  * numbered General Instructions
  * section banners with their own instruction line
  * question number in the left gutter, marks right-aligned in the margin
  * MCQ options laid out (A)-(D), two per row, rather than run into the stem

ReportLab (pure Python) rather than a HTML engine: no native GTK dependency,
and the layout is deterministic across machines.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .schemas import GeneratedPaper, GeneratedSectionSchema, SchoolTemplate

log = logging.getLogger(__name__)

# ReportLab's built-in Helvetica is Latin-1 only: subscripts (₂), superscripts
# (⁸), Greek (Ω) and arrows (→) render as black boxes. Register a Unicode TTF.
#
# Arial (the first candidate this used to be) turned out to be missing the
# subscript-digit block entirely (U+2080-2089, U+2212 subscript minus) even
# though it covers superscripts and other symbols fine — confirmed with
# fontTools against the actual installed TTF. A chemistry equation like
# "Al2O3" restored to "Al₂O₃" by notation.py then printed as "Al▯O▯" on a real
# exported paper. Segoe UI has full coverage of both blocks and ships with
# every supported Windows version, so it goes first.
_BODY_FONT = "Helvetica"
_BODY_FONT_BOLD = "Helvetica-Bold"
_BODY_FONT_PATH = ""  # the TTF actually registered, for the startup log
_FONT_CANDIDATES = (
    ("AcademicSans", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("AcademicSans", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("AcademicSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)

_OPTION_RE = re.compile(r"\(([A-Da-d])\)\s*")
# The word before a label that names an option instead of starting one:
# "Both (A) and (B)", "neither (a) nor (b)", "explanation of (A)",
# "Assertion (A) is correct".
_MENTION_BEFORE = re.compile(
    r"(?:\b(?:both|and|or|nor|either|neither|of|but|than|except|assertion|reason)|\bi\.e\.)\s*$",
    re.I)
# What joins the items of a mention list: "(A), (B) and (C)", "(A) & (B)".
_LIST_JOINER = re.compile(r"\s*[,&]\s*")
# An option that is only a joining word is the tail of a mis-split mention
# list ([',', 'and', ''] from "All of (A), (B) and (C)"), never an option.
# "both" / "neither" alone are real one-word options and are not here.
_JUNK_OPTION = re.compile(r"(?:and|or|nor|of|but|than|except)?", re.I)

DEFAULT_INSTRUCTIONS = (
    "This question paper contains {n} questions. <b>All questions are compulsory.</b>",
    "This question paper is divided into {sections} sections — {labels}.",
    "There is no overall choice. However, an internal choice has been provided in some questions.",
    "Use of calculators is <b>not</b> permitted.",
    "Draw neat diagrams wherever necessary.",
)

_SECTION_NOTE = {
    "A": (1, "consists of multiple choice questions carrying 1 mark each."),
    "B": (2, "consists of very short answer questions carrying 2 marks each."),
    "C": (3, "consists of short answer questions carrying 3 marks each."),
    "D": (5, "consists of long answer questions carrying 5 marks each."),
    "E": (4, "consists of case-study based questions carrying 4 marks each."),
}


def find_unicode_font(candidates=_FONT_CANDIDATES) -> tuple[str, str, str] | None:
    """The first candidate whose regular face exists on disk, or None.

    On the production image that is DejaVu, installed by the Dockerfile's
    `fonts-dejavu-core` (audit 1.2: the image had no Unicode font at all, so
    subscripts and symbols -- in 4.6% of stems -- printed as boxes).
    """
    for name, regular, bold in candidates:
        if Path(regular).exists():
            return name, regular, bold
    return None


def register_unicode_font() -> str:
    """Register the body font now and say which one it is, for the startup
    log: a missing font is otherwise invisible until a paper prints boxes."""
    _register_unicode_font()
    if _BODY_FONT == "Helvetica":
        return "Helvetica (built-in, Latin-1 only -- no Unicode font found)"
    return f"{_BODY_FONT} ({_BODY_FONT_PATH})"


def _register_unicode_font() -> None:
    global _BODY_FONT, _BODY_FONT_BOLD, _BODY_FONT_PATH
    if _BODY_FONT != "Helvetica":
        return
    for name, regular, bold in _FONT_CANDIDATES:
        if find_unicode_font([(name, regular, bold)]) is None:
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, regular))
            if Path(bold).exists():
                pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold))
                _BODY_FONT_BOLD = f"{name}-Bold"
            else:
                _BODY_FONT_BOLD = name
            _BODY_FONT = name
            _BODY_FONT_PATH = regular
            return
        except Exception as e:
            log.warning("could not register font %s: %s", regular, e)


# A line-for-line port of frontend/lib/core/local_engine/pdf_text_safety.dart,
# so a question prints the same on the web as on the phone. The web had no
# equivalent: CBE Maths writes variables as Mathematical Alphanumeric letters
# (U+1D434-) and symbol-font private-use characters, and those vanished --
# "12x^2 + 11x - 15" printed as "122 + 11-15". In production it is worse: the
# container falls back to Helvetica (WinAnsi only) when DejaVu is missing.
_PDF_PUNCTUATION = (("—", "-"), ("–", "-"), ("•", "-"),
                    ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'))
# Subscript digits U+2080-2089. A subscript names a thing -- H2O, a1 -- and
# never changes a value, so a plain digit reads as the ASCII spelling already
# in use.
_SUB_DIGITS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Superscripts are an exponent, and a plain digit is a DIFFERENT quantity:
# cbe:q:Maths8BS2 is "(A) 4y³ (B) 9y³ (C) 13y³ (D) 36y³" and printed
# "(A) 4y3 (B) 9y3 ..." -- the whole point of the question, gone (re-audit,
# 2026-09-23). Printed with the caret a teacher writes on a board, over the
# whole run, so "10⁻³" is "10^-3" and not "10^-^3".
_SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
                 "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-",
                 "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n"}
_SUPER_RUN = re.compile(f"[{''.join(_SUPERSCRIPTS)}]+")


def _exponent(match: re.Match) -> str:
    return "^" + "".join(_SUPERSCRIPTS[c] for c in match.group(0))
_PDF_MATH = (
    ("−", "-"), ("√", "sqrt"), ("∴", "therefore"), ("∵", "because"),
    ("∠", "angle "), ("⇒", "=>"), ("⟹", "=>"), ("≠", "!="),
    ("≥", ">="), ("≤", "<="), ("∈", "in"), ("∞", "infinity"),
    ("∫", "integral"),
    # DOT OPERATOR (U+22C5): Segoe UI has no glyph for it, so it printed as a
    # notdef box, while MIDDLE DOT (U+00B7) draws the same mark and is Latin-1.
    # The papers use it for both a decimal point and a product -- Biology XII
    # 784bd959 Q27's "2⋅4 g/litre" -- so the mark is kept, not read.
    ("⋅", "·"),
    ("′", "'"), ("θ", "theta"), ("π", "pi"), ("⃗", ""),
    ("̂", ""), ("₹", "Rs."), ("…", "..."),
    ("ଶ", "2"),        # an OCR artifact standing in for a superscript 2
    ("𝛼", "alpha"), ("𝜃", "theta"), ("𝜋", "pi"),
    # Every remaining value of `symbol_font.SYMBOL` that the registered font
    # draws as a notdef box, measured by rendering the whole table and reading
    # it back (tests/test_assessment_pdf.py::
    # test_every_symbol_the_repair_can_restore_reaches_the_page). The repair
    # covers these codes, so it WILL hand the renderer these characters the
    # day a source uses one -- 0x40 alone is already 18 occurrences across 7
    # served records -- and a repaired symbol that prints as a box is the hole
    # the repair set out to close, moved one stage later.
    ("≅", "congruent"), ("∼", "~"), ("∪", "union"), ("∀", "for all "),
    ("∃", "there exists "), ("∋", "contains"), ("∉", "not in"), ("∅", "empty set"),
    ("⊥", "perpendicular"), ("∝", "proportional to"), ("∗", "*"), ("∇", "nabla"),
    ("∧", "and"), ("∨", "or"), ("⊂", "subset of"), ("⊃", "superset of"),
    ("⊄", "not a subset of"), ("⊆", "subset of or equal to"),
    ("⊇", "superset of or equal to"), ("⊕", "(+)"), ("⊗", "(x)"),
    ("〈", "<"), ("〉", ">"),   # Symbol 0xE1/0xF1 angle brackets
    ("⇐", "<=="), ("⇔", "<=>"), ("⇑", "up"), ("⇓", "down"),
    ("ℵ", "aleph"), ("ℑ", "Im"), ("ℜ", "Re"), ("℘", "P"),
    ("↵", ""),         # Symbol 0xBF: a line break in the source, not content
    # And every other non-ASCII character the served bank holds that the font
    # cannot draw, from the same measurement over the bank itself.
    ("ℎ", "h"),        # U+210E PLANCK CONSTANT, an italic h in "V = πr2ℎ"
    # U+2218 RING OPERATOR after a number is a degree sign, not composition:
    # SQP Mathematics X (Basic) 2024-25 Q13 prints "cos 60∘".
    ("∘", "°"),
    ("∛", "cbrt"), ("∜", "4th root "),
    ("∥", "||"),
    # U+2551 BOX DRAWINGS DOUBLE VERTICAL for "is parallel to": cbe:q:Maths9IM7
    # prints "TS║QR". U+27D8 LARGE UP TACK for perpendicular: cbe:q:Maths10ASR11
    # prints "QS ⟘ PR". U+2A6D CONGRUENT WITH DOT ABOVE for congruent: SQP
    # Mathematics X (Basic) Q29 prints "𝛥𝑂𝐴𝑃⩭𝛥𝑂𝐵𝑃".
    ("║", "||"), ("⟘", "perpendicular"), ("⩭", "congruent"),
    ("△", "triangle "),
    # Styled Greek outside the Latin-only Mathematical Alphanumeric range
    # `_demathify` covers. The capital delta keeps its plain code point so the
    # triangle/change-in rule at the end of `pdf_safe` still reads it.
    ("𝛥", "Δ"), ("𝛴", "Σ"), ("𝛽", "β"), ("𝜀", "ε"), ("𝜇", "μ"),
    ("𝜎", "σ"), ("𝜔", "ω"), ("𝝅", "pi"),
    # Fullwidth forms, from a CJK font substitution in the SQP Mathematics X
    # 2024-25 marking schemes: "（𝑥-10)(𝑥-8）=0" (Standard VIC Q27), "FBD ～ DEF".
    ("（", "("), ("）", ")"), ("～", "~"),
    # U+571F, the CJK ideograph for "earth", from the same substitution: it
    # stands where a plus-minus sign belongs -- "8 - x =土4 => x = 4, 12"
    # (Standard Q23) and "= 土 1/(cosθ - sinθ)" (Q29), both plus-minus in
    # context and nowhere near any CJK text.
    ("土", "±"),
)
# Symbol-font glyph references left by the source PDFs' OCR: no defined
# meaning, so dropped rather than guessed.
_PRIVATE_USE = re.compile("[-]")
# U+2206 INCREMENT and U+0394 GREEK CAPITAL DELTA. Geometry writes a triangle
# as "∆ABC" (often with italic math letters, so this runs after _demathify),
# but Economics, Physics and Chemistry use the same glyph for "change in":
# "Increase in Income (ΔY)", "Energy released = ∆m x 931.5 MeV",
# "∆E_I > ∆E_II". Mapping every delta to "triangle" printed "triangle Y".
# So "triangle" only before a three-capital vertex name that is not an
# energy-level subscript; everything else is "delta".
_TRIANGLE = re.compile(r"[∆Δ] ?(?=(?!E[IVX]{2}\b)[A-Z]{3}(?![A-Za-z]))")
_DELTA = re.compile(r"[∆Δ] ?")
_MATH_LETTER_BLOCKS = (0x1D400, 0x1D434, 0x1D468, 0x1D49C, 0x1D4D0, 0x1D504, 0x1D538,
                       0x1D56C, 0x1D5A0, 0x1D5D4, 0x1D608, 0x1D63C, 0x1D670)


def _demathify(cp: int) -> int | None:
    """A styled Mathematical Alphanumeric letter or digit, back to plain ASCII."""
    for start in _MATH_LETTER_BLOCKS:
        offset = cp - start
        if 0 <= offset < 52:
            return 0x41 + offset if offset < 26 else 0x61 + offset - 26
    if 0x1D7CE <= cp <= 0x1D7FF:
        return 0x30 + (cp - 0x1D7CE) % 10
    return None


def pdf_safe(text: str) -> str:
    """Text a base PDF font can print, keeping every symbol's meaning."""
    for a, b in _PDF_PUNCTUATION:
        text = text.replace(a, b)
    text = _SUPER_RUN.sub(_exponent, text.translate(_SUB_DIGITS))
    for a, b in _PDF_MATH:
        text = text.replace(a, b)
    text = _PRIVATE_USE.sub("", text)
    text = "".join(chr(_demathify(ord(c)) or ord(c)) for c in text)
    return _DELTA.sub("delta ", _TRIANGLE.sub("triangle ", text))


def escape(text: str) -> str:
    text = pdf_safe(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class _Styles:
    school: ParagraphStyle
    exam: ParagraphStyle
    meta: ParagraphStyle
    instr_head: ParagraphStyle
    instr: ParagraphStyle
    section: ParagraphStyle
    section_note: ParagraphStyle
    question: ParagraphStyle
    option: ParagraphStyle
    marks: ParagraphStyle
    choice: ParagraphStyle


def _build_styles(brand_color: str = "#000000") -> _Styles:
    ss = getSampleStyleSheet()
    return _Styles(
        school=ParagraphStyle("School", parent=ss["Title"], fontName=_BODY_FONT_BOLD,
                              fontSize=15, leading=18, spaceAfter=0,
                              textColor=colors.HexColor(brand_color)),
        exam=ParagraphStyle("Exam", parent=ss["Normal"], fontName=_BODY_FONT_BOLD,
                            fontSize=11, alignment=TA_CENTER, spaceBefore=2),
        meta=ParagraphStyle("Meta", parent=ss["Normal"], fontName=_BODY_FONT_BOLD,
                            fontSize=9.5, alignment=TA_CENTER),
        instr_head=ParagraphStyle("InstrHead", parent=ss["Normal"], fontName=_BODY_FONT_BOLD,
                                  fontSize=10, alignment=TA_CENTER, spaceBefore=6, spaceAfter=4),
        instr=ParagraphStyle("Instr", parent=ss["Normal"], fontName=_BODY_FONT,
                             fontSize=9, leading=12.5, leftIndent=14, firstLineIndent=-14,
                             spaceAfter=2, alignment=TA_JUSTIFY),
        section=ParagraphStyle("Section", parent=ss["Normal"], fontName=_BODY_FONT_BOLD,
                               fontSize=11.5, alignment=TA_CENTER, spaceBefore=12, spaceAfter=2),
        section_note=ParagraphStyle("SectionNote", parent=ss["Normal"], fontName=_BODY_FONT,
                                    fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                                    spaceAfter=8),
        question=ParagraphStyle("Question", parent=ss["Normal"], fontName=_BODY_FONT,
                                fontSize=10, leading=14, alignment=TA_JUSTIFY),
        option=ParagraphStyle("Option", parent=ss["Normal"], fontName=_BODY_FONT,
                              fontSize=9.5, leading=13),
        marks=ParagraphStyle("Marks", parent=ss["Normal"], fontName=_BODY_FONT,
                             fontSize=9.5, alignment=2),
        choice=ParagraphStyle("Choice", parent=ss["Normal"], fontName=_BODY_FONT_BOLD,
                              fontSize=9.5, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2),
    )


def _follows_label_in_list(stem: str, matches: list[re.Match], i: int) -> bool:
    """matches[i] is joined to the label before it by ',' or '&' -- the next
    item of a mention list, as the (B) in "(A), (B) and (C)"."""
    if i == 0:
        return False
    prev_close = matches[i - 1].start() + 3  # a label is always "(X)"
    return bool(_LIST_JOINER.fullmatch(stem, prev_close, matches[i].start()))


def _option_run(stem: str, matches: list[re.Match], first: int) -> list[re.Match] | None:
    """The option labels from matches[first] (an A) to the end of the stem.

    Each next label must be the next letter. A label already used is text when
    it names an option -- a mention word before it, it opens the option's text
    ("(C) (A) is correct"), or it follows a mention through ',' or '&'
    ("All of (A), (B) and (C)"). Anything else -- a gap, a label out of order,
    a label repeated with nothing naming it -- means this is not an option list.
    """
    run = [matches[first]]
    prev_mention = False
    for i in range(first + 1, len(matches)):
        m = matches[i]
        k = "ABCD".find(m.group(1).upper())
        if k == len(run):
            run.append(m)
            prev_mention = False
        elif k < len(run) and (_MENTION_BEFORE.search(stem, 0, m.start())
                               or m.start() == run[-1].end()
                               or (prev_mention and _follows_label_in_list(stem, matches, i))):
            prev_mention = True
        else:
            return None
    return run if len(run) >= 2 else None


def _options_of(stem: str, run: list[re.Match]) -> list[str]:
    options: list[str] = []
    for i, m in enumerate(run):
        end = run[i + 1].start() if i + 1 < len(run) else len(stem)
        options.append(stem[m.end():end].strip(" .;"))
    return options


def split_stem_and_options(stem: str) -> tuple[str, list[str]]:
    """Separate an MCQ stem from its (A)-(D) options so they can be laid out.

    The options are the longest run of labels A, B, C[, D] that reaches the end
    of the stem, the first such run if two are as long. A label is printed in
    three other places, and none of them is an option:

    - before the options: CBE Science numbers its sub-question "(a) Which
      conclusions are correct? (A) ... (D)" (5 served items, e.g.
      cbe:q:Science10TM2) and assertion-reason items say "Assertion (A) and
      Reason (R) ... (A) ... (D)". Taking every label printed the question as
      option (A) and shifted each real option down a letter.
    - inside an option, naming another one: "(D) Both (A) and (B)"
      (cbse:q:src:e268efcf2e142b7e0591b8ed:2), "(d) neither (a) nor (b)", and
      the assertion-reason options "(C) (A) is correct but (R) is not correct"
      (cbse:sqp:ClassXII_2025_26:History:10). Taking the run from the last
      "(A)" printed head "... (D) Both" and options ['and', '11-'].

    A run that does not reach the end, or none at all, splits nothing and the
    stem prints verbatim. So does a run with an option that is empty or only a
    joiner ("and", ",") -- the tail of a mention list like "All of (A), (B)
    and (C)" -- and no run starts at an (A) that is itself a mention. Over the
    5840 served + CBE + SQP stems this changed 43 splits (vs 0708f8d), every
    one from junk options ('', 'and', ', ,') to the stem as written. Of the 4475 served stems, 3 split differently from
    the trailing-run rule this replaced (measured at 1e3e30f), all three the
    mention shape above.
    """
    matches = list(_OPTION_RE.finditer(stem))
    best: list[re.Match] | None = None
    for i, m in enumerate(matches):
        # An (A) that is itself a mention ("of (A)", ", (A)") starts no run.
        if (m.group(1).upper() != "A" or _MENTION_BEFORE.search(stem, 0, m.start())
                or _follows_label_in_list(stem, matches, i)):
            continue
        run = _option_run(stem, matches, i)
        if not run or (best is not None and len(run) <= len(best)):
            continue
        # A run with an empty or joiner-only option is a mis-split: print the
        # stem as written instead of junk options.
        if any(_JUNK_OPTION.fullmatch(o.strip(" .;,&")) for o in _options_of(stem, run)):
            continue
        best = run
    if best is None:
        return stem.strip(), []
    head = stem[: best[0].start()].strip()
    if not head:
        return stem.strip(), []
    return head, _options_of(stem, best)


def _roll_no_grid(styles: _Styles, boxes: int = 11) -> Table:
    """The Roll No. grid printed on every CBSE paper."""
    cells = [[Paragraph("<b>Roll No.</b>", styles.meta)] + [""] * boxes]
    widths = [22 * mm] + [7 * mm] * boxes
    t = Table(cells, colWidths=widths, rowHeights=[8 * mm])
    t.setStyle(TableStyle([
        ("GRID", (1, 0), (-1, -1), 0.6, colors.black),
        ("BOX", (1, 0), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "RIGHT"),
        ("RIGHTPADDING", (0, 0), (0, 0), 6),
    ]))
    return t


def _header(paper: GeneratedPaper, template: Optional[SchoolTemplate],
            styles: _Styles, content_width: float) -> list:
    m = paper.metadata
    # The teacher's header wins over the school's branding name: a template
    # paper may be set for a named branch or a joint exam.
    school_name = (m.school_name or (template.name if template and template.name else "")
                   or "AcademicOS School")
    brand = colors.HexColor(template.brand_color) if template and template.brand_color else colors.black
    story: list = []

    exam_title = escape(m.assessment_title)
    if paper.set_label:
        exam_title += f" &nbsp;&nbsp;|&nbsp;&nbsp; <b>SET {escape(paper.set_label)}</b>"

    logo_path = Path(template.logo_url) if template and template.logo_url else None
    title_block = [Paragraph(escape(school_name), styles.school)]
    # The exam name is normally the title; a teacher who gave the paper its
    # own title still gets the exam name their template set.
    if m.exam_name and m.exam_name != m.assessment_title:
        title_block.append(Paragraph(escape(m.exam_name), styles.exam))
    title_block += [
        Paragraph(exam_title, styles.exam),
        Paragraph(f"Subject: {escape(m.subject)} &nbsp;&nbsp;|&nbsp;&nbsp; Class: {m.grade}",
                  styles.meta),
    ]
    if template and template.tagline:
        title_block.append(Paragraph(escape(template.tagline), styles.meta))
    if m.date_line:
        title_block.append(Paragraph(escape(m.date_line), styles.meta))
    if logo_path and logo_path.exists():
        try:
            band = Table([[Image(str(logo_path), width=18 * mm, height=18 * mm), title_block]],
                         colWidths=[22 * mm, content_width - 22 * mm])
            band.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
            story.append(band)
        except Exception as e:                       # a bad logo must not kill the export
            log.warning("could not place logo %s: %s", logo_path, e)
            story.extend(title_block)
    else:
        story.extend(title_block)

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.4, color=brand, spaceAfter=6))
    story.append(_roll_no_grid(styles))
    story.append(Spacer(1, 6))

    marks_str = f"<b>Maximum Marks: {m.total_marks}</b>"
    if paper.set_label:
        marks_str += f" &nbsp;&nbsp; [<b>SET {escape(paper.set_label)}</b>]"

    rule = Table(
        [[Paragraph(f"<b>Time Allowed: {m.duration_minutes // 60} hours "
                    f"{m.duration_minutes % 60:02d} minutes</b>", styles.question),
          Paragraph(marks_str, styles.marks)]],
        colWidths=[content_width * 0.6, content_width * 0.4],
    )
    rule.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(rule)
    return story


def _instructions(paper: GeneratedPaper, styles: _Styles) -> list:
    total_q = sum(len(s.questions) for s in paper.sections)
    labels = ", ".join(s.label for s in paper.sections)
    story = [Paragraph("General Instructions:", styles.instr_head)]
    teacher = [line.strip() for line in (paper.metadata.instructions or "").splitlines()
               if line.strip()]
    if teacher:
        # The teacher's own instructions replace the canned list: printing
        # both would put "internal choice has been provided" on a paper whose
        # template may offer none.
        for i, line in enumerate(teacher, start=1):
            story.append(Paragraph(f"({i})&nbsp;&nbsp;{escape(line)}", styles.instr))
        return story
    for i, tmpl in enumerate(DEFAULT_INSTRUCTIONS, start=1):
        text = tmpl.format(n=total_q, sections=len(paper.sections), labels=labels)
        story.append(Paragraph(f"({i})&nbsp;&nbsp;{text}", styles.instr))
    return story


def _question_flowables(gq, styles: _Styles, content_width: float,
                        gutter: float, marks_col: float) -> list:
    """One question: number in the gutter, marks in the right margin, options gridded.

    Options are split out only for an objective question: a descriptive
    question's sub-parts "(a) Find ... (b) Explain ..." read exactly like
    options and were gridded as (A)-(D). An empty type is a paper generated
    before GeneratedQuestionSchema carried one, and keeps the old behaviour.

    The board records type most 1-mark MCQs very_short_answer (538 of them
    split into exactly four options, against 46 typed mcq), so a 1-mark item
    whose stem splits into four ordered options is objective too. The
    descriptive sub-part stems in the bank are all multi-mark.
    """
    split_head, split_options = split_stem_and_options(gq.stem)
    objective = (gq.type in ("", "mcq", "assertion_reason")
                 or (gq.marks == 1 and len(split_options) == 4))
    head, options = (split_head, split_options) if objective else (gq.stem.strip(), [])
    body: list = [Paragraph(escape(head), styles.question)]

    if options:
        rows: list[list] = []
        pairs = [options[i:i + 2] for i in range(0, len(options), 2)]
        idx = 0
        for pair in pairs:
            row = []
            for opt in pair:
                # split_stem_and_options returns at most (A)-(D); chr() is
                # only a guard so a future fifth label cannot IndexError.
                row.append(Paragraph(f"({chr(ord('A') + idx)})&nbsp; {escape(opt)}", styles.option))
                idx += 1
            if len(row) == 1:
                row.append("")
            rows.append(row)
        opt_width = (content_width - gutter - marks_col) / 2
        opt_table = Table(rows, colWidths=[opt_width, opt_width])
        opt_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ]))
        body.append(Spacer(1, 3))
        body.append(opt_table)

    if gq.internal_choice_text:
        body.append(Paragraph("OR", styles.choice))
        body.append(Paragraph(escape(gq.internal_choice_text), styles.question))

    # splitInRow: the whole question is one table row, and a row that cannot
    # split raised LayoutError (HTTP 500) on any question taller than a page.
    row = Table(
        [[Paragraph(f"<b>{gq.display_number}.</b>", styles.question), body,
          Paragraph(str(gq.marks), styles.marks)]],
        colWidths=[gutter, content_width - gutter - marks_col, marks_col],
        splitInRow=1,
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (2, 0), (2, 0), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [row]


def _section_note(section: GeneratedSectionSchema) -> str:
    """The CBSE-standard descriptions in _SECTION_NOTE assume a section
    *labeled* A/B/C/D/E also follows the CBSE per-section mark convention
    (A=1, B=2, C=3, D=5, E=4). That's true for a paper generated from a
    standard CBSE blueprint, but a custom blueprint can label a section "A"
    while putting 2-mark questions in it -- found live 2026-09-18 by actually
    reading a generated PDF: it printed "carrying 1 mark each" over a section
    of 2-mark questions. Only trust the canned description when the section's
    real, uniform per-question marks actually match what it claims; otherwise
    fall back to a description built from the real data."""
    marks_seen = {q.marks for q in section.questions}
    uniform_marks = marks_seen.pop() if len(marks_seen) == 1 else None
    canned = _SECTION_NOTE.get(section.label.upper())
    if canned is not None and canned[0] == uniform_marks:
        return canned[1]
    if uniform_marks is not None:
        plural = "" if uniform_marks == 1 else "s"
        return f"consists of questions carrying {uniform_marks} mark{plural} each."
    return "consists of questions carrying marks as indicated."


def _section_block(section: GeneratedSectionSchema, styles: _Styles,
                   content_width: float, gutter: float, marks_col: float,
                   frame_height: float) -> list:
    note = _section_note(section)
    story: list = [
        Paragraph(f"SECTION {section.label}", styles.section),
        Paragraph(f"({section.name} — {len(section.questions)} questions, "
                  f"{section.total_marks} marks. This section {note})", styles.section_note),
    ]
    for gq in section.questions:
        flowables = _question_flowables(gq, styles, content_width, gutter, marks_col)
        # Keep a question on one page when it fits on one; one that cannot
        # fit anywhere is left free to split across pages instead.
        height = sum(f.wrap(content_width, frame_height)[1] for f in flowables)
        if height <= frame_height:
            story.append(KeepTogether(flowables))
        else:
            story.extend(flowables)
    return story


def _page_furniture(paper: GeneratedPaper, template: Optional[SchoolTemplate],
                    watermark_id: Optional[str] = None):
    """Footer drawn on every page: paper code left, page number centred.

    watermark_id (when provided) is a per-export identifier stamped in the
    margin -- small and legible-but-unobtrusive, not a bold diagonal stamp
    that would make a real exam paper harder to read. The point isn't to
    deface the paper; it's that if a copy leaks, the specific export that
    produced it is traceable via the audit log (see audit_log.py and the
    /papers/{id}/export/{fmt} route), the same way a real print run is
    traceable to a specific press job.
    """
    code = paper.id.replace("paper_", "").upper()[:10]
    school = template.name if template and template.name else "AcademicOS"

    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(_BODY_FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(doc.leftMargin, 12 * mm, f"{school}  ·  Q.P. Code {code}")
        canvas.drawCentredString(A4[0] / 2.0, 12 * mm, f"Page {doc.page}")
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, "P.T.O.")
        if watermark_id:
            canvas.setFont(_BODY_FONT, 6)
            canvas.setFillColor(colors.HexColor("#AAAAAA"))
            canvas.drawCentredString(A4[0] / 2.0, 7 * mm,
                                     f"Confidential — traceable copy {watermark_id}")
        canvas.restoreState()

    return draw


def export_pdf(paper: GeneratedPaper, output_dir: Path,
               template: Optional[SchoolTemplate] = None,
               watermark_id: Optional[str] = None) -> Path:
    _register_unicode_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper.id}.pdf"
    styles = _build_styles(template.brand_color if template and template.brand_color else "#000000")

    left = (template.margin_left if template else 18) * mm
    right = (template.margin_right if template else 18) * mm
    top = (template.margin_top if template else 16) * mm
    bottom = (template.margin_bottom if template else 20) * mm

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        topMargin=top, bottomMargin=bottom, leftMargin=left, rightMargin=right,
        title=paper.metadata.assessment_title, author="AcademicOS",
    )
    content_width = A4[0] - left - right
    gutter, marks_col = 10 * mm, 12 * mm

    story: list = []
    story.extend(_header(paper, template, styles, content_width))
    story.extend(_instructions(paper, styles))
    for section in paper.sections:
        story.extend(_section_block(section, styles, content_width, gutter, marks_col,
                                    doc.height))

    footer = _page_furniture(paper, template, watermark_id=watermark_id)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


def export_answer_key_pdf(paper: GeneratedPaper, output_dir: Path,
                          template: Optional[SchoolTemplate] = None) -> Path:
    """Companion marking sheet: question number, marks, expected answer / value points."""
    _register_unicode_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper.id}_answer_key.pdf"
    styles = _build_styles(template.brand_color if template and template.brand_color else "#000000")

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=16 * mm,
                            bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm)
    set_suffix = f" — SET {paper.set_label}" if paper.set_label else ""
    story: list = [
        Paragraph(escape(paper.metadata.assessment_title) + set_suffix, styles.school),
        Paragraph("MARKING SCHEME / VALUE POINTS", styles.exam),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=8),
    ]
    rows = [[Paragraph("<b>Q.No.</b>", styles.option),
             Paragraph("<b>Marks</b>", styles.option),
             Paragraph("<b>Expected answer / Value points</b>", styles.option)]]
    for section in paper.sections:
        for gq in section.questions:
            answer = paper.answer_key.get(str(gq.display_number), "")
            ans_formatted = escape(str(answer)).replace("\n", "<br/>") if answer else "<i>(pending teacher entry)</i>"
            if gq.internal_choice_text:
                or_answer = paper.answer_key.get(f"{gq.display_number}_OR", "")
                if or_answer:
                    or_formatted = escape(str(or_answer)).replace("\n", "<br/>")
                    ans_formatted += f"<br/><br/><b>[OR CHOICE]:</b><br/>{or_formatted}"
            rows.append([
                Paragraph(str(gq.display_number), styles.option),
                Paragraph(str(gq.marks), styles.option),
                Paragraph(ans_formatted, styles.option),
            ])
    # splitInRow: a model answer longer than a page is one row that must split.
    table = Table(rows, colWidths=[14 * mm, 14 * mm, None], repeatRows=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    doc.build(story)
    return out_path
