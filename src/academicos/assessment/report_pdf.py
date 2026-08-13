"""Branded student progress reports — Pillar 3/4's mastery data rendered as
something a parent-teacher meeting can actually hand over.

A mastery score by itself ("62%") tells a parent nothing actionable. This
renderer turns `ConceptMasteryView` records into three things a teacher can
use in the room: which concepts are solid, which are genuinely weak (not just
low-confidence from too little evidence), and a concrete next step per weak
concept — not "study harder," but "revise Chemical Reactions and Equations;
recent answers show the balancing-equations misconception."
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import pdf as pdf_export
from .knowledge import ConceptMasteryView
from .schemas import SchoolTemplate

# `pdf_export._BODY_FONT` is only correct at call time, *after*
# `pdf_export._register_unicode_font()` runs — it starts as "Helvetica" and gets
# reassigned to the registered Unicode font the first time any PDF export
# runs. `from .pdf import _BODY_FONT` would have frozen the pre-registration
# value forever, since a plain import copies the binding rather than reading
# the module attribute live.
escape = pdf_export.escape

_STATUS_COLOR = {
    "mastered": colors.HexColor("#1B7A3D"),
    "proficient": colors.HexColor("#2E7D32"),
    "developing": colors.HexColor("#B8860B"),
    "needsReview": colors.HexColor("#C0392B"),
    "learning": colors.HexColor("#607D8B"),
}
_STATUS_LABEL = {
    "mastered": "Mastered", "proficient": "Proficient", "developing": "Developing",
    "needsReview": "Needs review", "learning": "Still gathering evidence",
}

# One concrete study action per misconception name. Anything not in this map
# falls back to a generic-but-still-specific action built from the concept
# name itself, never a bare "practice more."
_MISCONCEPTION_ACTIONS = {
    "sign_error": "Redo problems from this concept slowly, checking the sign at every step before moving on.",
    "unit_confusion": "Drill unit conversions for this concept in isolation before mixing them back into full problems.",
    "formula_misuse": "Re-derive the formula from first principles rather than memorizing it, then re-attempt.",
}


def _build_report_styles(brand_color: str = "#000000"):
    ss = getSampleStyleSheet()
    brand = colors.HexColor(brand_color)
    return {
        "school": ParagraphStyle("RSchool", parent=ss["Title"], fontName=pdf_export._BODY_FONT_BOLD,
                                 fontSize=16, leading=19, textColor=brand, alignment=TA_CENTER),
        "title": ParagraphStyle("RTitle", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD,
                                fontSize=12, alignment=TA_CENTER, spaceBefore=2),
        "meta": ParagraphStyle("RMeta", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                               fontSize=9.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
        "h2": ParagraphStyle("RH2", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD,
                             fontSize=12, spaceBefore=14, spaceAfter=6, textColor=brand),
        "body": ParagraphStyle("RBody", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                               fontSize=9.5, leading=13.5, alignment=TA_LEFT),
        "concept": ParagraphStyle("RConcept", parent=ss["Normal"], fontName=pdf_export._BODY_FONT_BOLD,
                                  fontSize=9.5, leading=13),
        "small": ParagraphStyle("RSmall", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                fontSize=8.5, leading=12, textColor=colors.HexColor("#555555")),
        "action": ParagraphStyle("RAction", parent=ss["Normal"], fontName=pdf_export._BODY_FONT,
                                 fontSize=9, leading=13, leftIndent=10,
                                 textColor=colors.HexColor("#1a1a1a")),
    }


def _action_for(view: ConceptMasteryView) -> str:
    for name in view.misconceptions:
        if name in _MISCONCEPTION_ACTIONS:
            return _MISCONCEPTION_ACTIONS[name]
    if view.evidence_count < 3:
        return (f"Too few graded answers on {escape(view.concept_name)} to be sure yet — "
                f"assign one more short practice set before drawing conclusions.")
    if view.accuracy < view.mastery:
        return (f"Accuracy on {escape(view.concept_name)} is inconsistent — right on some "
                f"attempts, wrong on others of the same type. Points to a gap in one specific "
                f"step rather than the whole concept; review recent answer sheets together.")
    return (f"Revise {escape(view.concept_name)} from the textbook explanation first, then "
            f"attempt a fresh practice set — accuracy is low and consistent, which usually "
            f"means the concept itself needs re-teaching, not just more repetition.")


def _header(student_name: str, student_id: str, subject: str,
           template: Optional[SchoolTemplate], styles: dict, content_width: float) -> list:
    school_name = template.name if template and template.name else "AcademicOS School"
    brand = colors.HexColor(template.brand_color) if template and template.brand_color else colors.black
    story: list = []

    title_block = [
        Paragraph(escape(school_name), styles["school"]),
        Paragraph("Student Progress Report", styles["title"]),
        Paragraph(f"{escape(student_name)} &nbsp;|&nbsp; Roll/ID: {escape(student_id)} "
                  f"&nbsp;|&nbsp; Subject: {escape(subject)}", styles["meta"]),
        Paragraph(f"Generated {datetime.now().strftime('%d %b %Y')}", styles["small"]),
    ]
    logo_path = Path(template.logo_url) if template and template.logo_url else None
    if logo_path and logo_path.exists():
        band = Table([[Image(str(logo_path), width=16 * mm, height=16 * mm), title_block]],
                     colWidths=[20 * mm, content_width - 20 * mm])
        band.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(band)
    else:
        story.extend(title_block)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.4, color=brand, spaceAfter=8))
    return story


def _summary_row(views: list[ConceptMasteryView]) -> Table:
    mastered = sum(1 for v in views if v.status in ("mastered", "proficient"))
    weak = sum(1 for v in views if v.is_weak)
    avg = round(100 * sum(v.mastery for v in views) / len(views)) if views else 0
    cells = [[f"{avg}%", f"{mastered}/{len(views)}", f"{weak}"]]
    headers = [["Overall mastery", "Concepts on track", "Concepts needing attention"]]
    t = Table(headers + cells, colWidths=[None, None, None])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), pdf_export._BODY_FONT),
        ("FONTSIZE", (0, 0), (-1, 0), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#666666")),
        ("FONTNAME", (0, 1), (-1, 1), pdf_export._BODY_FONT_BOLD),
        ("FONTSIZE", (0, 1), (-1, 1), 15),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#dddddd")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#dddddd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#eeeeee")),
    ]))
    return t


def export_progress_report_pdf(student_name: str, student_id: str, subject: str,
                               views: list[ConceptMasteryView], output_dir: Path,
                               template: Optional[SchoolTemplate] = None) -> Path:
    """Renders one student's concept-mastery breakdown as a branded PDF:
    summary numbers, a per-concept table (status, accuracy, evidence), and a
    concrete next-step recommendation for every concept below the mastery bar.
    """
    pdf_export._register_unicode_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c for c in student_id if c.isalnum() or c in "-_") or "student"
    out_path = output_dir / f"progress_{safe_id}_{subject.lower()}.pdf"
    styles = _build_report_styles(template.brand_color if template and template.brand_color else "#000000")

    left = right = 18 * mm
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=16 * mm, bottomMargin=18 * mm,
                            leftMargin=left, rightMargin=right,
                            title=f"Progress Report — {student_name}", author="AssessmentOS")
    content_width = A4[0] - left - right

    story: list = _header(student_name, student_id, subject, template, styles, content_width)
    story.append(Spacer(1, 4))

    ranked = sorted(views, key=lambda v: v.mastery)
    if ranked:
        story.append(_summary_row(ranked))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Concept-by-concept mastery", styles["h2"]))
    rows = [["Concept", "Status", "Accuracy", "Evidence"]]
    for v in ranked:
        rows.append([
            Paragraph(escape(v.concept_name), styles["concept"]),
            Paragraph(_STATUS_LABEL.get(v.status, v.status), styles["body"]),
            Paragraph(f"{round(v.accuracy * 100)}%", styles["body"]),
            Paragraph(f"{v.evidence_count} answer(s)", styles["small"]),
        ])
    if len(rows) > 1:
        t = Table(rows, colWidths=[content_width * 0.42, content_width * 0.22,
                                   content_width * 0.18, content_width * 0.18])
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), pdf_export._BODY_FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]
        for i, v in enumerate(ranked, start=1):
            style_cmds.append(("TEXTCOLOR", (1, i), (1, i), _STATUS_COLOR.get(v.status, colors.black)))
        t.setStyle(TableStyle(style_cmds))
        story.append(t)
    else:
        story.append(Paragraph("No graded answers recorded yet for this subject.", styles["body"]))

    weak = [v for v in ranked if v.is_weak]
    if weak:
        story.append(Paragraph("Recommended next steps", styles["h2"]))
        for v in weak:
            story.append(Paragraph(f"<b>{escape(v.concept_name)}</b>", styles["concept"]))
            story.append(Paragraph(_action_for(v), styles["action"]))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("Recommended next steps", styles["h2"]))
        story.append(Paragraph(
            "No concept is currently below the mastery threshold — maintain "
            "the current pace and revisit periodically to keep retention up.",
            styles["body"]))

    doc.build(story)
    return out_path
