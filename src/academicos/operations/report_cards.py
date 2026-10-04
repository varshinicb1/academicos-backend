"""Report cards (REQUIREMENTS EX-8: "CBSE-format report cards").

Marks entry (EX-8) records each student's marks per question for a paper
sat on paper, and its analysis gives a class's grade bands -- but nothing
turned a term's marks into the one page a school hands a parent. This does:
for one student and one term, every paper of their class set in the term
whose marks were entered, by subject and exam type, with each subject's
total, percentage and CBSE grade band (the 8-point scale the marks analysis
uses), and the term overall. A student marked absent for a paper reads "AB"
and that paper counts for neither side; so does a paper whose marks were
only partly entered, which reads "incomplete" (v3 audit N-2-10: such a paper
was totalled over the whole paper and graded).

It says only what was measured: marks entered in AcademicOS for papers set
this term. Co-scholastic areas, attendance and teachers' remarks are not
recorded here, and the card says so. No rank and no class position (the
product shows no leaderboards to children, REQUIREMENTS section 0).

Who reads a card: the student, their linked parent, the principal or a
teacher given the `reports` permission, and the section's class teacher.
Every read by someone other than the student is logged as a read of student
data.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from ..assessment.auth_routes import get_current_user, holds
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel

router = APIRouter(prefix="/api/v1/report-cards")

EXAM_TYPE_LABELS = {
    "unit_test": "Unit test", "periodic": "Periodic test", "half_yearly": "Half-yearly", "annual": "Annual",
    "class_test": "Class test", "weekly_test": "Weekly test", "board": "Board", "custom": "Test",
}
EXAM_TYPE_ORDER = ["unit_test", "weekly_test", "class_test", "periodic", "half_yearly", "annual", "board", "custom"]
NOT_RECORDED = ("Marks entered in AcademicOS for papers set this term. Co-scholastic areas, attendance and "
                "teachers' remarks are not recorded here.")


@dataclass
class PaperMark:
    paper_id: str
    title: str
    exam_type: str
    set_on: str
    obtained: Optional[float]      # None: absent, or incomplete
    maximum: int
    incomplete: bool = False       # some of the paper's questions have marks, not all

    @property
    def absent(self) -> bool:
        return self.obtained is None and not self.incomplete


@dataclass
class SubjectLine:
    subject: str
    papers: list[PaperMark] = field(default_factory=list)

    @property
    def obtained(self) -> float:
        return sum(p.obtained for p in self.papers if p.obtained is not None)

    @property
    def maximum(self) -> int:
        return sum(p.maximum for p in self.papers if p.obtained is not None)


def _percent(obtained: float, maximum: int) -> Optional[float]:
    return round(100 * obtained / maximum, 1) if maximum else None


def _band(percent: Optional[float]) -> Optional[str]:
    from ..assessment.marks_routes import band
    return band(percent) if percent is not None else None


def _paper_total(paper) -> int:
    from ..assessment.marks_routes import paper_total
    return paper_total(paper)


def build_card(*, student_id: str, grade: int, school_id: str, start: str, end: str,
               papers: list, marks_for, absent_for) -> list[SubjectLine]:
    """The card's subjects, from `papers` (the school's generated papers):
    those of the student's class set within [start, end] for which the
    student has marks or was marked absent."""
    from ..assessment.marks_routes import paper_score
    lines: dict[str, SubjectLine] = {}
    for paper in papers:
        m = paper.metadata
        set_on = m.generated_at.date().isoformat() if m.generated_at else ""
        if m.grade != grade or not (start <= set_on <= end):
            continue
        marks = marks_for(paper.id).get(student_id)
        absent = student_id in absent_for(paper.id)
        if marks is None and not absent:
            continue
        got = None if absent else paper_score(paper, marks)
        complete = got is not None
        line = lines.setdefault(m.subject, SubjectLine(subject=m.subject))
        line.papers.append(PaperMark(paper_id=paper.id, title=m.assessment_title,
                                     exam_type=m.exam_type or "custom", set_on=set_on,
                                     obtained=float(got) if complete else None,
                                     maximum=_paper_total(paper), incomplete=not absent and not complete))
    for line in lines.values():
        line.papers.sort(key=lambda p: (EXAM_TYPE_ORDER.index(p.exam_type) if p.exam_type in EXAM_TYPE_ORDER
                                        else len(EXAM_TYPE_ORDER), p.set_on))
    return sorted(lines.values(), key=lambda s: s.subject.lower())


# ---------------------------------------------------------------- access

@dataclass
class _Context:
    student: Any
    section: Any
    grade: int
    term: Any
    year: Any


def _store():
    from .routes import store
    return store()


def _term(term_id: Optional[str], school_id: str):
    cs = cr._require()
    if term_id:
        term = cs.get_term(term_id)
        if term is None:
            raise HTTPException(404, "term not found")
        if term.school_id != school_id:
            raise HTTPException(403, "this term belongs to a different school")
        return term
    term = cs.term_for_date(school_id, cr._school_today().isoformat())
    if term is None:
        raise HTTPException(409, "today is in no term: choose one (termId), or declare the school's terms")
    return term


def _may_read_section(current: User, section) -> bool:
    return holds(current, "reports") or (current.role == "teacher" and section.class_teacher_id == current.id)


def _context(student_id: str, current: User, term_id: Optional[str]) -> _Context:
    cs = cr._require()
    student = cr._require_users().get(student_id)
    if student is None or student.role != "student":
        raise HTTPException(404, "student not found")
    if student.school_id != current.school_id:
        raise HTTPException(403, "that student belongs to a different school")
    enrollment = cs.enrollment_for_student(student_id)
    section = cs.get_section(enrollment.section_id) if enrollment and enrollment.section_id else None
    if section is None:
        raise HTTPException(409, "the student is not enrolled in a section")
    if current.id == student_id:
        pass
    elif current.role == "parent":
        if not _store().is_guardian(current.id, student_id):
            raise HTTPException(403, "a parent reads their own child's report card")
    elif current.role in ("teacher", "principal"):
        if not _may_read_section(current, section):
            raise HTTPException(403, "report cards are for the principal, a teacher given reports, "
                                     "and the section's class teacher")
    else:
        raise HTTPException(403, "you may read only your own report card")
    grade = cs.get_grade(section.grade_id)
    term = _term(term_id, current.school_id)
    return _Context(student=student, section=section, grade=grade.number if grade else 0, term=term,
                    year=cs.get_academic_year(term.academic_year_id))


def _school_papers(school_id: str) -> list:
    from ..assessment import routes as ar
    return ar._require_papers().list_by_school(school_id)


def _lines(ctx: _Context, papers: Optional[list] = None) -> list[SubjectLine]:
    papers = papers if papers is not None else _school_papers(ctx.student.school_id)
    return build_card(student_id=ctx.student.id, grade=ctx.grade, school_id=ctx.student.school_id,
                      start=ctx.term.start_date, end=ctx.term.end_date, papers=papers,
                      marks_for=_store().marks_for, absent_for=_store().absent_for)


def _log_read(current: User, what: str, student_ids: list[str], **details: Any) -> None:
    from ..assessment.audit_log import get_audit_log, record_pii_read
    others = [s for s in student_ids if s != current.id]
    if others:
        record_pii_read(get_audit_log(cr._cfg.data_root), actor=current.id, what=what, student_ids=others,
                        **details)


# ---------------------------------------------------------------- JSON

class PaperMarkResponse(Camel):
    paper_id: str
    title: str
    exam_type: str
    exam_type_label: str
    set_on: str
    obtained: Optional[float] = None
    maximum: int
    absent: bool
    # Marks entered for some of the paper's questions, not all: no total,
    # and it counts for neither side, as an absence does.
    incomplete: bool = False


class SubjectLineResponse(Camel):
    subject: str
    papers: list[PaperMarkResponse]
    obtained: float
    maximum: int
    percent: Optional[float] = None
    band: Optional[str] = None


class ReportCardResponse(Camel):
    student_id: str
    student_name: str
    section: str
    term_id: str
    term_name: str
    year_label: str
    subjects: list[SubjectLineResponse]
    obtained: float
    maximum: int
    percent: Optional[float] = None
    band: Optional[str] = None
    note: str = NOT_RECORDED


def _response(ctx: _Context, lines: list[SubjectLine]) -> ReportCardResponse:
    obtained, maximum = sum(l.obtained for l in lines), sum(l.maximum for l in lines)
    pct = _percent(obtained, maximum)
    return ReportCardResponse(
        student_id=ctx.student.id, student_name=ctx.student.name,
        section=cr._require()._section_label(ctx.section), term_id=ctx.term.id, term_name=ctx.term.name,
        year_label=ctx.year.label if ctx.year else "",
        subjects=[SubjectLineResponse(
            subject=l.subject, obtained=l.obtained, maximum=l.maximum,
            percent=_percent(l.obtained, l.maximum), band=_band(_percent(l.obtained, l.maximum)),
            papers=[PaperMarkResponse(paper_id=p.paper_id, title=p.title, exam_type=p.exam_type,
                                      exam_type_label=EXAM_TYPE_LABELS.get(p.exam_type, "Test"), set_on=p.set_on,
                                      obtained=p.obtained, maximum=p.maximum, absent=p.absent,
                                      incomplete=p.incomplete)
                    for p in l.papers]) for l in lines],
        obtained=obtained, maximum=maximum, percent=pct, band=_band(pct))


@router.get("/students/{student_id}", response_model=ReportCardResponse)
def report_card(student_id: str, term_id: Optional[str] = Query(default=None, alias="termId"),
                current: User = Depends(get_current_user)) -> ReportCardResponse:
    """One student's card for a term (default: the term today falls in)."""
    ctx = _context(student_id, current, term_id)
    card = _response(ctx, _lines(ctx))
    _log_read(current, "report_card", [student_id], termId=ctx.term.id)
    return card


# ---------------------------------------------------------------- PDF

def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return str(int(x)) if float(x).is_integer() else f"{x:.1f}"


def render_pdf(cards: list[ReportCardResponse], school_id: str) -> bytes:
    """One A4 page per card, headed with the school's own name and logo
    (school_profile.py) -- never a template's name."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    from ..assessment import pdf as pdf_export
    from ..curriculum.school_profile import local_logo

    pdf_export._register_unicode_font()
    body, bold = pdf_export._BODY_FONT, pdf_export._BODY_FONT_BOLD
    ss = getSampleStyleSheet()
    school_style = ParagraphStyle("RCSchool", parent=ss["Title"], fontName=bold, fontSize=15, spaceAfter=2)
    meta_style = ParagraphStyle("RCMeta", parent=ss["Normal"], fontName=body, fontSize=9, alignment=1)
    title_style = ParagraphStyle("RCTitle", parent=ss["Normal"], fontName=bold, fontSize=12, alignment=1,
                                 spaceBefore=4, spaceAfter=6)
    cell = ParagraphStyle("RCCell", parent=ss["Normal"], fontName=body, fontSize=9, leading=11)
    note_style = ParagraphStyle("RCNote", parent=ss["Normal"], fontName=body, fontSize=8, textColor=colors.grey)

    profile = cr._require().get_school_profile(school_id)
    logo = local_logo(cr._cfg.data_root, profile)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=14 * mm, bottomMargin=14 * mm,
                            leftMargin=16 * mm, rightMargin=16 * mm, title="Report card", author="AcademicOS")
    width = A4[0] - 32 * mm
    story: list = []
    for i, c in enumerate(cards):
        if i:
            story.append(PageBreak())
        head = []
        if profile is not None:
            head.append(Paragraph(pdf_export.escape(profile.name), school_style))
            extra = " | ".join(x for x in (profile.address, profile.affiliation) if x)
            if extra:
                head.append(Paragraph(pdf_export.escape(extra), meta_style))
        if logo is not None:
            story.append(Table([[Image(str(logo), width=16 * mm, height=16 * mm), head]],
                               colWidths=[20 * mm, width - 20 * mm],
                               style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")])))
        else:
            story.extend(head)
        story.append(Paragraph(pdf_export.escape(f"Report card: {c.term_name}"
                                                 f"{', ' + c.year_label if c.year_label else ''}"), title_style))
        story.append(Paragraph(pdf_export.escape(f"{c.student_name}  |  Class {c.section}"), meta_style))
        story.append(Spacer(1, 6))
        rows = [[Paragraph(f"<b>{h}</b>", cell) for h in ("Subject", "Papers", "Marks", "%", "Grade")]]
        for s in c.subjects:
            papers = "<br/>".join(
                pdf_export.escape(f"{p.exam_type_label}: "
                                  f"{'AB' if p.absent else 'incomplete' if p.incomplete else _fmt(p.obtained)}"
                                  f"/{p.maximum}")
                for p in s.papers)
            rows.append([Paragraph(pdf_export.escape(s.subject), cell), Paragraph(papers, cell),
                         Paragraph(f"{_fmt(s.obtained)}/{s.maximum}", cell),
                         Paragraph(_fmt(s.percent), cell), Paragraph(s.band or "-", cell)])
        rows.append([Paragraph("<b>Overall</b>", cell), Paragraph("", cell),
                     Paragraph(f"<b>{_fmt(c.obtained)}/{c.maximum}</b>", cell),
                     Paragraph(f"<b>{_fmt(c.percent)}</b>", cell), Paragraph(f"<b>{c.band or '-'}</b>", cell)])
        table = Table(rows, colWidths=[width * 0.24, width * 0.36, width * 0.16, width * 0.1, width * 0.14],
                      repeatRows=1)
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(table if c.subjects else Paragraph("No marks have been entered for this term yet.", cell))
        story.append(Spacer(1, 8))
        story.append(Paragraph(pdf_export.escape(NOT_RECORDED) + " Grades: CBSE 8-point scale "
                               "(A1 91-100, A2 81-90, B1 71-80, B2 61-70, C1 51-60, C2 41-50, D 33-40, E below 33).",
                               note_style))
        story.append(Spacer(1, 28))
        story.append(Table([["Class teacher", "Principal", "Parent"]], colWidths=[width / 3] * 3,
                           style=TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black),
                                             ("FONTNAME", (0, 0), (-1, -1), body),
                                             ("FONTSIZE", (0, 0), (-1, -1), 9)])))
    doc.build(story)
    return buf.getvalue()


def _pdf(content: bytes, name: str) -> Response:
    return Response(content=content, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


@router.get("/students/{student_id}/pdf")
def report_card_pdf(student_id: str, term_id: Optional[str] = Query(default=None, alias="termId"),
                    current: User = Depends(get_current_user)) -> Response:
    ctx = _context(student_id, current, term_id)
    card = _response(ctx, _lines(ctx))
    _log_read(current, "report_card_pdf", [student_id], termId=ctx.term.id)
    return _pdf(render_pdf([card], current.school_id), f"report-card-{ctx.term.name}.pdf".replace(" ", "-"))


@router.get("/sections/{section_id}/pdf")
def section_report_cards_pdf(section_id: str, term_id: Optional[str] = Query(default=None, alias="termId"),
                             current: User = Depends(get_current_user)) -> Response:
    """Every student of the section, one page each, in name order. For the
    principal, a teacher given reports, and the section's class teacher."""
    section = cr._require_school_owns_section(section_id, current)
    if current.role not in ("teacher", "principal") or not _may_read_section(current, section):
        raise HTTPException(403, "report cards are for the principal, a teacher given reports, "
                                 "and the section's class teacher")
    cs = cr._require()
    users = cr._require_users()
    ids = [e.student_id for e in cs.enrollments_for_section(section_id)]
    students = sorted((u for u in (users.get(i) for i in ids) if u is not None), key=lambda u: u.name.lower())
    if not students:
        raise HTTPException(409, "no students are enrolled in this section")
    papers = _school_papers(current.school_id)          # once, not once a student
    cards = []
    for s in students:
        ctx = _context(s.id, current, term_id)
        cards.append(_response(ctx, _lines(ctx, papers)))
    _log_read(current, "report_cards_section_pdf", [s.id for s in students], sectionId=section_id,
              termId=cards[0].term_id)
    label = cs._section_label(section)
    return _pdf(render_pdf(cards, current.school_id), f"report-cards-{label}-{cards[0].term_name}.pdf".replace(" ", "-"))
