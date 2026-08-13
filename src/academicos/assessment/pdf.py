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
_FONT_CANDIDATES = (
    ("AcademicSans", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("AcademicSans", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("AcademicSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
)

_OPTION_RE = re.compile(r"\(([A-Da-d])\)\s*")

DEFAULT_INSTRUCTIONS = (
    "This question paper contains {n} questions. <b>All questions are compulsory.</b>",
    "This question paper is divided into {sections} sections — {labels}.",
    "There is no overall choice. However, an internal choice has been provided in some questions.",
    "Use of calculators is <b>not</b> permitted.",
    "Draw neat diagrams wherever necessary.",
)

_SECTION_NOTE = {
    "A": "consists of multiple choice questions carrying 1 mark each.",
    "B": "consists of very short answer questions carrying 2 marks each.",
    "C": "consists of short answer questions carrying 3 marks each.",
    "D": "consists of long answer questions carrying 5 marks each.",
    "E": "consists of case-study based questions carrying 4 marks each.",
}


def _register_unicode_font() -> None:
    global _BODY_FONT, _BODY_FONT_BOLD
    if _BODY_FONT != "Helvetica":
        return
    for name, regular, bold in _FONT_CANDIDATES:
        if not Path(regular).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, regular))
            if Path(bold).exists():
                pdfmetrics.registerFont(TTFont(f"{name}-Bold", bold))
                _BODY_FONT_BOLD = f"{name}-Bold"
            else:
                _BODY_FONT_BOLD = name
            _BODY_FONT = name
            return
        except Exception as e:
            log.warning("could not register font %s: %s", regular, e)


def escape(text: str) -> str:
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


def split_stem_and_options(stem: str) -> tuple[str, list[str]]:
    """Separate an MCQ stem from its (A)-(D) options so they can be laid out."""
    matches = list(_OPTION_RE.finditer(stem))
    if len(matches) < 2:
        return stem.strip(), []
    # Options must be in order and near the end; otherwise "(a)" is a part label.
    labels = [m.group(1).upper() for m in matches]
    if labels != sorted(labels) or labels[0] != "A":
        return stem.strip(), []
    head = stem[: matches[0].start()].strip()
    options: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stem)
        options.append(stem[m.end():end].strip(" .;"))
    if not head:
        return stem.strip(), []
    return head, options


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
    school_name = (template.name if template and template.name else "AcademicOS School")
    brand = colors.HexColor(template.brand_color) if template and template.brand_color else colors.black
    story: list = []

    logo_path = Path(template.logo_url) if template and template.logo_url else None
    title_block = [
        Paragraph(escape(school_name), styles.school),
        Paragraph(escape(m.assessment_title), styles.exam),
        Paragraph(f"Subject: {escape(m.subject)} &nbsp;&nbsp;|&nbsp;&nbsp; Class: {m.grade}",
                  styles.meta),
    ]
    if template and template.tagline:
        title_block.append(Paragraph(escape(template.tagline), styles.meta))
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

    rule = Table(
        [[Paragraph(f"<b>Time Allowed: {m.duration_minutes // 60} hours "
                    f"{m.duration_minutes % 60:02d} minutes</b>", styles.question),
          Paragraph(f"<b>Maximum Marks: {m.total_marks}</b>", styles.marks)]],
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
    for i, tmpl in enumerate(DEFAULT_INSTRUCTIONS, start=1):
        text = tmpl.format(n=total_q, sections=len(paper.sections), labels=labels)
        story.append(Paragraph(f"({i})&nbsp;&nbsp;{text}", styles.instr))
    return story


def _question_flowables(gq, styles: _Styles, content_width: float,
                        gutter: float, marks_col: float) -> list:
    """One question: number in the gutter, marks in the right margin, options gridded."""
    head, options = split_stem_and_options(gq.stem)
    body: list = [Paragraph(escape(head), styles.question)]

    if options:
        rows: list[list] = []
        labels = "ABCD"
        pairs = [options[i:i + 2] for i in range(0, len(options), 2)]
        idx = 0
        for pair in pairs:
            row = []
            for opt in pair:
                row.append(Paragraph(f"({labels[idx]})&nbsp; {escape(opt)}", styles.option))
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

    row = Table(
        [[Paragraph(f"<b>{gq.display_number}.</b>", styles.question), body,
          Paragraph(str(gq.marks), styles.marks)]],
        colWidths=[gutter, content_width - gutter - marks_col, marks_col],
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (2, 0), (2, 0), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return [row]


def _section_block(section: GeneratedSectionSchema, styles: _Styles,
                   content_width: float, gutter: float, marks_col: float) -> list:
    note = _SECTION_NOTE.get(section.label.upper(),
                             f"consists of questions carrying marks as indicated.")
    story: list = [
        Paragraph(f"SECTION {section.label}", styles.section),
        Paragraph(f"({section.name} — {len(section.questions)} questions, "
                  f"{section.total_marks} marks. This section {note})", styles.section_note),
    ]
    for gq in section.questions:
        story.append(KeepTogether(
            _question_flowables(gq, styles, content_width, gutter, marks_col)))
    return story


def _page_furniture(paper: GeneratedPaper, template: Optional[SchoolTemplate]):
    """Footer drawn on every page: paper code left, page number centred."""
    code = paper.id.replace("paper_", "").upper()[:10]
    school = template.name if template and template.name else "AcademicOS"

    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(_BODY_FONT, 7.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(doc.leftMargin, 12 * mm, f"{school}  ·  Q.P. Code {code}")
        canvas.drawCentredString(A4[0] / 2.0, 12 * mm, f"Page {doc.page}")
        canvas.drawRightString(A4[0] - doc.rightMargin, 12 * mm, "P.T.O.")
        canvas.restoreState()

    return draw


def export_pdf(paper: GeneratedPaper, output_dir: Path,
               template: Optional[SchoolTemplate] = None) -> Path:
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
        title=paper.metadata.assessment_title, author="AssessmentOS",
    )
    content_width = A4[0] - left - right
    gutter, marks_col = 10 * mm, 12 * mm

    story: list = []
    story.extend(_header(paper, template, styles, content_width))
    story.extend(_instructions(paper, styles))
    for section in paper.sections:
        story.extend(_section_block(section, styles, content_width, gutter, marks_col))

    footer = _page_furniture(paper, template)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_path


def export_answer_key_pdf(paper: GeneratedPaper, output_dir: Path,
                          template: Optional[SchoolTemplate] = None) -> Path:
    """Companion marking sheet: question number, marks, expected answer."""
    _register_unicode_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper.id}_answer_key.pdf"
    styles = _build_styles(template.brand_color if template and template.brand_color else "#000000")

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=16 * mm,
                            bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm)
    story: list = [
        Paragraph(escape(paper.metadata.assessment_title), styles.school),
        Paragraph("MARKING SCHEME / ANSWER KEY", styles.exam),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=8),
    ]
    rows = [[Paragraph("<b>Q.No.</b>", styles.option),
             Paragraph("<b>Marks</b>", styles.option),
             Paragraph("<b>Expected answer / marking points</b>", styles.option)]]
    for section in paper.sections:
        for gq in section.questions:
            answer = paper.answer_key.get(str(gq.display_number), "")
            rows.append([
                Paragraph(str(gq.display_number), styles.option),
                Paragraph(str(gq.marks), styles.option),
                Paragraph(escape(str(answer)) or "<i>(pending teacher entry)</i>", styles.option),
            ])
    table = Table(rows, colWidths=[14 * mm, 14 * mm, None], repeatRows=1)
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
