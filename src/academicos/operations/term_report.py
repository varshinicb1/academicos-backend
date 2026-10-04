"""The principal's term report (ADM-2, ADM-4): one term, filterable by grade,
section and subject.

It covers the syllabus against its plan (and where it is behind), exams
scheduled and papers set, time saved (estimated, and labelled so), homework
issued and handed in, cover (substitutions, lost and made-up periods), the
question bank for the school's classes with its gaps named, and how students
are doing chapter by chapter. The same report downloads as an Excel workbook
or a PDF.

Aggregates only: no student is named, and no teacher either, so an exported
file passed round a staff room is not a leaderboard (ADM-5).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from ..assessment.auth_routes import require_admin
from ..assessment.users import User
from ..curriculum import routes as cr
from ..curriculum.schemas import Camel
from .routes import store

router = APIRouter(prefix="/api/v1")


class Scope(Camel):
    kind: Literal["term", "year"]
    name: str
    start_date: str
    end_date: str
    as_of: str
    term_id: Optional[str] = None
    note: Optional[str] = None
    # The web page where what `note` asks for is done (D43), or None.
    note_link: Optional[str] = None


class Filters(Camel):
    grade: Optional[int] = None
    section_id: Optional[str] = None
    section_name: Optional[str] = None
    subject: Optional[str] = None


class SyllabusRow(Camel):
    grade: int
    section_name: Optional[str] = None      # None: one plan for every section of the class
    subject: str
    planned_to_date: int
    taught_to_date: int
    variance: int
    pace_pct: float
    total_lessons: int
    coverage_pct: float
    behind: bool
    # Subtopics of the book with no day in this year's plan: the periods the
    # timetable gives cannot hold the syllabus (D35). The plan is not behind
    # yet, but it cannot finish; 0 when it fits.
    undated_subtopics: int = 0


class Syllabus(Camel):
    planned_to_date: int
    taught_to_date: int
    behind: int
    rows: list[SyllabusRow]
    # Class-subjects whose plan cannot finish this year (undated subtopics).
    cannot_finish: int = 0


class ExamRow(Camel):
    name: str
    start_date: str
    end_date: str
    status: str
    papers: int


class ClassSubject(Camel):
    grade: int
    subject: str
    papers: int = 0


class Exams(Camel):
    scheduled: list[ExamRow]
    papers_set: int
    covered: list[ClassSubject]
    # No paper yet, and the question bank can serve the class.
    not_yet: list[ClassSubject]
    unattributed: int
    # No paper yet because the question bank has no question for the class
    # yet: the bank's gap, not the teachers' (D117).
    no_questions: list[ClassSubject] = []


class TimeSaved(Camel):
    papers: int
    minutes_per_paper: float
    minutes_total: float
    # Always: every saving rests on the declared baseline.
    estimated: bool = True
    baseline_minutes_per_paper: float
    baseline_provenance: str
    # Of `papers`, how many count the baseline less their teacher's time as
    # the web app measured it; the rest count the whole baseline (D49).
    papers_with_teacher_time: int = 0


class HomeworkRow(Camel):
    grade: int
    subject: str
    issued: int
    expected: int
    handed_in: int
    marked: int


class Homework(Camel):
    issued: int
    expected: int
    handed_in: int
    marked: int
    rows: list[HomeworkRow]


class Cover(Camel):
    requested: int
    filled: int
    supervised: int
    unfilled: int
    lost: int
    compensated: int
    owed: int


class BankRow(Camel):
    grade: int
    subject: str
    questions: int
    chapters: int
    chapters_without_questions: list[str]


class ProgressRow(Camel):
    grade: int
    subject: str
    students_with_evidence: int
    mastered: int
    developing: int
    needs_work: int
    early: int


class TermReport(Camel):
    scope: Scope
    filters: Filters
    syllabus: Syllabus
    exams: Exams
    time_saved: TimeSaved
    homework: Homework
    cover: Cover
    bank: list[BankRow]
    progress: list[ProgressRow]


# ---------------------------------------------------------------- helpers

class _School:
    """The year's classes, sections and subjects, looked up once."""

    def __init__(self, cs, year_id: str):
        self.grade_of = {g.id: g.number for g in cs.grades_for_year(year_id)}
        self.sections = {s.id: s for s in cs.sections_for_year(year_id)}
        self.label = {sid: cs._section_label(s) for sid, s in self.sections.items()}
        self.section_grade = {sid: self.grade_of.get(s.grade_id) for sid, s in self.sections.items()}
        self.subjects = {s.id: (self.grade_of[g_id], s.name)
                         for g_id in self.grade_of for s in cs.subjects_for_grade(g_id)}


class _Filter:
    def __init__(self, grade: Optional[int], section_id: Optional[str], subject: Optional[str]):
        self.grade, self.section_id = grade, section_id
        self.subject = subject.strip().casefold() if subject and subject.strip() else None

    def keep(self, grade: Optional[int], subject: Optional[str], section_id: Optional[str] = None,
             sections: Optional[list[str]] = None) -> bool:
        if self.grade is not None and grade != self.grade:
            return False
        if self.subject is not None and (subject or "").strip().casefold() != self.subject:
            return False
        if self.section_id is not None:
            if section_id is not None and section_id != self.section_id:
                return False
            if sections is not None and self.section_id not in sections:
                return False
        return True


def _resolve(current: User, term_id: Optional[str]) -> tuple[str, Scope]:
    cs = cr._require()
    today = cr._school_today().isoformat()
    if term_id:
        term = cs.get_term(term_id)
        if term is None:
            raise HTTPException(404, "term not found")
        if term.school_id != current.school_id:
            raise HTTPException(403, "this term belongs to a different school")
    else:
        term = cs.term_for_date(current.school_id, today)
    if term is not None:
        return term.academic_year_id, Scope(kind="term", name=term.name, start_date=term.start_date,
                                            end_date=term.end_date, as_of=min(today, term.end_date),
                                            term_id=term.id)
    from ..assessment import paper_timing
    year = next((y for y in cs.academic_years_for_school(current.school_id)
                 if y.start_date <= today <= y.end_date), None)
    if year is None:
        raise HTTPException(404, "no academic year covers today; set up this year's classes on the "
                                 f"School Admin console ({paper_timing.CLASSES_PAGE})")
    step = paper_timing.setup_step(classes_declared=True, term_declared=False)
    return year.id, Scope(kind="year", name=year.label, start_date=year.start_date, end_date=year.end_date,
                          as_of=today, note=f"No term covers today, so this is the year so far. {step['message']}",
                          note_link=step["link"])


def _syllabus(current: User, year_id: str, scope: Scope, f: _Filter) -> Syllabus:
    # A term's syllabus is its own lessons, not the year's so far (N-67-6).
    term = (scope.start_date, scope.end_date) if scope.kind == "term" else (None, None)
    cs = cr._require()
    data = cs.get_coverage_report(school_id=current.school_id, academic_year_id=year_id,
                                  as_of_date=scope.as_of, from_date=term[0], to_date=term[1])
    rows = []
    for s in data["subjects"]:
        if not f.keep(s["grade_number"], s["subject_name"], s.get("section_id")):
            continue
        rows.append(SyllabusRow(
            grade=s["grade_number"], section_name=s.get("section_name") and f"{s['grade_number']}-{s['section_name']}",
            subject=s["subject_name"], planned_to_date=s["planned_to_date"], taught_to_date=s["completed_to_date"],
            variance=s["variance"], pace_pct=s["pace_pct"], total_lessons=s["total_lessons"],
            coverage_pct=s["coverage_pct"], behind=s["variance"] < 0,
            # The whole year's plan, whatever the term: a subtopic with no day
            # anywhere in it is one the class will not be taught (D35).
            undated_subtopics=cs.undated_subtopic_count(year_id, s["book_id"], s.get("section_id"))))
    rows.sort(key=lambda r: (not (r.behind or r.undated_subtopics), r.variance, -r.undated_subtopics, r.grade,
                             r.section_name or "", r.subject))
    return Syllabus(planned_to_date=sum(r.planned_to_date for r in rows),
                    taught_to_date=sum(r.taught_to_date for r in rows),
                    behind=sum(1 for r in rows if r.behind), rows=rows,
                    cannot_finish=sum(1 for r in rows if r.undated_subtopics))


def _exams_and_time(current: User, year_id: str, scope: Scope, school: _School,
                    f: _Filter) -> tuple[Exams, TimeSaved]:
    from ..assessment import paper_timing
    from ..assessment import routes as ar
    ops = store()
    scheduled = []
    for e in ops.exams_for_school(current.school_id):
        if e["academic_year_id"] != year_id or e["end_date"] < scope.start_date or e["start_date"] > scope.end_date:
            continue
        papers = [p for p in ops.exam_papers(e["id"]) if f.keep(p["grade"], p["subject_name"])]
        if (f.grade is not None or f.subject is not None or f.section_id is not None) and not papers:
            continue
        scheduled.append(ExamRow(name=e["name"], start_date=e["start_date"], end_date=e["end_date"],
                                 status=e["status"], papers=len(papers)))
    scheduled.sort(key=lambda x: x.start_date)

    baseline = None
    if scope.term_id:
        term = cr._require().get_term(scope.term_id)
        if term is not None and term.manual_baseline_minutes is not None:
            baseline = (float(term.manual_baseline_minutes), f"set by the principal for {term.name}")
    # The papers that still exist, each once, as /paper-timing counts them (D42).
    kept = paper_timing.kept_papers(ar._require()[1].list_by_school(current.school_id))
    rep = paper_timing.report(cr._cfg.data_root, school_id=current.school_id, baseline=baseline,
                              start_date=scope.start_date, end_date=scope.as_of, kept=kept)
    # The same classes /paper-timing measures "not yet" against, so the two
    # screens cannot disagree about one term.
    universe = [(g, name) for g, name in school.subjects.values() if 1 <= g <= 10 and f.keep(g, name)]
    from . import homework_routes as hr
    cov = paper_timing.exam_coverage(rep, universe, questions=lambda g, s: hr._bank_question_count(s, g))
    covered = [ClassSubject(grade=c["grade"], subject=c["subject"], papers=c["papers"])
               for c in cov["covered"] if f.keep(c["grade"], c["subject"])]
    not_yet = [ClassSubject(grade=c["grade"], subject=c["subject"]) for c in cov["notYet"]]
    no_questions = [ClassSubject(grade=c["grade"], subject=c["subject"]) for c in cov["noQuestions"]]
    papers = sum(c.papers for c in covered)
    filtered = f.grade is not None or f.subject is not None
    # Each paper's own saving, against the baseline it was made under (D51),
    # summed over the papers the filter keeps.
    saved = sum(m for (g, s), m in rep.saved_by_class_subject.items() if f.keep(g, s))
    timed = sum(n for (g, s), n in rep.teacher_time_by_class_subject.items() if f.keep(g, s))
    if not filtered:
        papers += cov["unattributed"]
        saved += rep.saved_unattributed
        timed += rep.teacher_time_unattributed
    exams = Exams(scheduled=scheduled, papers_set=papers, covered=covered, not_yet=not_yet,
                  unattributed=0 if filtered else cov["unattributed"], no_questions=no_questions)
    per = saved / papers if papers else 0.0
    time = TimeSaved(papers=papers, minutes_per_paper=round(per, 1), minutes_total=round(saved, 1),
                     baseline_minutes_per_paper=rep.baseline_minutes_per_paper,
                     baseline_provenance=rep.baseline_provenance, papers_with_teacher_time=timed)
    return exams, time


def _students_by_section(school: _School, f: _Filter) -> dict[str, str]:
    """Student id -> section id, for the sections the filter keeps."""
    cs = cr._require()
    out: dict[str, str] = {}
    for sid in school.sections:
        if f.section_id is not None and sid != f.section_id:
            continue
        if f.grade is not None and school.section_grade.get(sid) != f.grade:
            continue
        for e in cs.enrollments_for_section(sid):
            out[e.student_id] = sid
    return out


def _homework(current: User, scope: Scope, f: _Filter, roster: dict[str, str]) -> Homework:
    ops = store()
    by_section: dict[str, int] = {}
    for sec in roster.values():
        by_section[sec] = by_section.get(sec, 0) + 1
    rows: dict[tuple[int, str], HomeworkRow] = {}
    for h in ops.homework_for_school(current.school_id):
        day = (h.published_at or "")[:10]
        if not day or day < scope.start_date or day > scope.as_of:
            continue
        if not f.keep(h.grade, h.subject_name, sections=h.section_ids):
            continue
        sections = [s for s in h.section_ids if f.section_id is None or s == f.section_id]
        expected = sum(by_section.get(s, 0) for s in sections)
        subs = [s for s in ops.submissions_for(h.id) if roster.get(s.student_id) in sections]
        row = rows.setdefault((h.grade, h.subject_name), HomeworkRow(
            grade=h.grade, subject=h.subject_name, issued=0, expected=0, handed_in=0, marked=0))
        row.issued += 1
        row.expected += expected
        row.handed_in += len(subs)
        row.marked += sum(1 for s in subs if s.status == "graded")
    out = sorted(rows.values(), key=lambda r: (r.grade, r.subject))
    return Homework(issued=sum(r.issued for r in out), expected=sum(r.expected for r in out),
                    handed_in=sum(r.handed_in for r in out), marked=sum(r.marked for r in out), rows=out)


def _cover(year_id: str, scope: Scope, school: _School, f: _Filter) -> Cover:
    from ..curriculum.cover import count_cover
    cs = cr._require()

    def kept(section_id: str, subject_id: str) -> bool:
        grade, name = school.subjects.get(subject_id, (school.section_grade.get(section_id), ""))
        return f.keep(school.section_grade.get(section_id, grade), name, section_id)

    subs = [s for s in cs.substitutions_between(year_id, scope.start_date, scope.as_of)
            if s.status != "cancelled" and kept(s.section_id, s.subject_id)]
    lost = [l for l in cs.lost_periods_for_year(year_id)
            if scope.start_date <= l.date <= scope.as_of and kept(l.section_id, l.subject_id)]
    # The same counting as the principal's cover summary (cover.count_cover).
    c = count_cover(subs, lost)
    return Cover(requested=c["substitutionsRequested"], filled=c["filled"], supervised=c["supervised"],
                 unfilled=c["unfilled"], lost=c["lost"], compensated=c["compensated"], owed=c["owed"])


def _bank(school: _School, f: _Filter) -> list[BankRow]:
    from . import homework_routes as hr
    out = []
    for grade, name in sorted(set(school.subjects.values())):
        if not f.keep(grade, name):
            continue
        chapters = hr._bank_chapters(name, grade)
        out.append(BankRow(grade=grade, subject=name, questions=sum(c["question_count"] for c in chapters),
                           chapters=len(chapters),
                           chapters_without_questions=[c["chapter_name"] for c in chapters if not c["question_count"]]))
    return out


def _progress(school: _School, f: _Filter, roster: dict[str, str]) -> list[ProgressRow]:
    from . import homework_routes as hr
    from .learning import score
    ops = store()
    ids = list(roster)
    groups: dict[tuple[int, str], dict[tuple[str, str], list[dict]]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for r in ops._fetchall(f"SELECT * FROM learning_evidence WHERE student_id IN ({marks})", tuple(chunk)):
            if r["chapter_id"] == "unmapped":
                continue
            grade = int(r["grade"]) if str(r["grade"]).isdigit() else None
            if f.grade is not None and grade != f.grade:
                continue
            if f.subject is not None and r["subject"].casefold() not in (f.subject, hr.bank_subject(f.subject).casefold()):
                continue
            groups.setdefault((grade or 0, r["subject"]), {}).setdefault((r["student_id"], r["chapter_id"]), []).append(r)
    now = datetime.now(timezone.utc)
    out = []
    for (grade, subject), pairs in sorted(groups.items()):
        counts = {"mastered": 0, "developing": 0, "needs_work": 0, "early": 0}
        for rows in pairs.values():
            status = score(rows, now)["status"]
            if status in counts:
                counts[status] += 1
        out.append(ProgressRow(grade=grade, subject=subject,
                               students_with_evidence=len({sid for sid, _ in pairs}), **counts))
    return out


def build(current: User, *, term_id: Optional[str], grade: Optional[int], section_id: Optional[str],
          subject: Optional[str]) -> TermReport:
    year_id, scope = _resolve(current, term_id)
    school = _School(cr._require(), year_id)
    section_name = None
    if section_id is not None:
        if section_id not in school.sections:
            raise HTTPException(404, "that section is not in this school's year")
        section_name = school.label[section_id]
        if grade is not None and school.section_grade[section_id] != grade:
            raise HTTPException(422, f"{section_name} is not in class {grade}")
        grade = school.section_grade[section_id]
    f = _Filter(grade, section_id, subject)
    roster = _students_by_section(school, f)
    exams, time = _exams_and_time(current, year_id, scope, school, f)
    return TermReport(
        scope=scope, filters=Filters(grade=grade, section_id=section_id, section_name=section_name,
                                     subject=subject.strip() if subject and subject.strip() else None),
        syllabus=_syllabus(current, year_id, scope, f), exams=exams, time_saved=time,
        homework=_homework(current, scope, f, roster), cover=_cover(year_id, scope, school, f),
        bank=_bank(school, f), progress=_progress(school, f, roster))


# ---------------------------------------------------------------- files

def _title(r: TermReport) -> tuple[str, str]:
    parts = [p for p in (f"class {r.filters.grade}" if r.filters.grade and not r.filters.section_name else None,
                         r.filters.section_name, r.filters.subject) if p]
    head = f"Term report: {r.scope.name}" + (f" ({', '.join(parts)})" if parts else "")
    sub = f"{r.scope.start_date} to {r.scope.end_date}, as of {r.scope.as_of}"
    return head, sub


def _tables(r: TermReport) -> list[tuple[str, list[list[Any]]]]:
    t = r.time_saved
    summary = [
        ["Measure", "Value"],
        ["Lessons planned to date", r.syllabus.planned_to_date],
        ["Lessons taught to date", r.syllabus.taught_to_date],
        ["Class-subjects behind plan", r.syllabus.behind],
        ["Class-subjects whose plan cannot finish this year", r.syllabus.cannot_finish],
        ["Papers set", r.exams.papers_set],
        ["Class-subjects with no paper yet", len(r.exams.not_yet)],
        ["Class-subjects the question bank has no questions for yet", len(r.exams.no_questions)],
        ["Time saved, minutes (estimated)", t.minutes_total],
        ["Homework issued", r.homework.issued],
        ["Homework handed in / expected", f"{r.homework.handed_in} / {r.homework.expected}"],
        ["Substitutions requested", r.cover.requested],
        ["Substitutions unfilled", r.cover.unfilled],
        ["Periods lost / made up / still owed", f"{r.cover.lost} / {r.cover.compensated} / {r.cover.owed}"],
    ]
    syllabus = [["Class", "Section", "Subject", "Planned to date", "Taught to date", "Variance", "Pace %",
                 "Coverage %", "Behind", "Subtopics with no date this year"]] + [
        [x.grade, x.section_name or "all", x.subject, x.planned_to_date, x.taught_to_date, x.variance,
         x.pace_pct, x.coverage_pct, x.behind, x.undated_subtopics] for x in r.syllabus.rows]
    exams = ([["Exam", "From", "To", "Status", "Papers"]]
             + [[e.name, e.start_date, e.end_date, e.status, e.papers] for e in r.exams.scheduled]
             + [[""], ["Class", "Subject", "Papers set"]]
             + [[c.grade, c.subject, c.papers] for c in r.exams.covered]
             + [[c.grade, c.subject, 0] for c in r.exams.not_yet]
             + [[c.grade, c.subject, "0 -- the question bank has no questions for this class yet"]
                for c in r.exams.no_questions]
             + [[""], ["Time saved is an estimate", f"{t.minutes_per_paper} min per paper",
                       f"baseline {t.baseline_minutes_per_paper} min: {t.baseline_provenance}",
                       f"{t.papers_with_teacher_time} of {t.papers} paper(s) less their teacher's "
                       f"measured time; the rest the whole baseline"]])
    homework = [["Class", "Subject", "Issued", "Expected", "Handed in", "Marked"]] + [
        [h.grade, h.subject, h.issued, h.expected, h.handed_in, h.marked] for h in r.homework.rows]
    bank = [["Class", "Subject", "Questions", "Chapters", "Chapters with no questions"]] + [
        [b.grade, b.subject, b.questions, b.chapters, "; ".join(b.chapters_without_questions)
         or ("no chapters for this class yet" if not b.chapters else "")] for b in r.bank]
    progress = [["Class", "Subject", "Students with evidence", "Mastered", "Developing", "Needs work", "Early"]] + [
        [p.grade, p.subject, p.students_with_evidence, p.mastered, p.developing, p.needs_work, p.early]
        for p in r.progress]
    return [("Summary", summary), ("Syllabus", syllabus), ("Exams", exams), ("Homework", homework),
            ("Question bank", bank), ("Progress", progress)]


class Letterhead:
    """The school's name, address line and logo, as its papers and report
    cards print them. The term report carried none of them (QA S-11), though
    the profile promises them "at the top of every paper, answer key and
    report"."""

    def __init__(self, name: str = "", address_line: str = "", logo=None):
        self.name, self.address_line, self.logo = name, address_line, logo

    @classmethod
    def for_school(cls, school_id: str) -> "Letterhead":
        from ..curriculum.school_profile import local_logo
        try:
            profile = cr._require().get_school_profile(school_id)
        except HTTPException:
            return cls()
        if profile is None:
            return cls()
        line = " | ".join(x for x in (profile.address, profile.affiliation) if x)
        logo = local_logo(cr._cfg.data_root, profile) if cr._cfg is not None else None
        return cls(profile.name, line, logo)


def to_xlsx(r: TermReport, letterhead: Optional[Letterhead] = None) -> bytes:
    from .xlsx import workbook
    head, sub = _title(r)
    sheets = _tables(r)
    name, rows = sheets[0]
    school = [[x] for x in ((letterhead.name, letterhead.address_line) if letterhead else ()) if x]
    sheets[0] = (name, school + [[head], [sub], [r.scope.note or ""], []] + rows)
    return workbook(sheets)


def to_pdf(r: TermReport, letterhead: Optional[Letterhead] = None) -> bytes:
    import io
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    cell = styles["BodyText"].clone("cell", fontSize=8, leading=10)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm, title="Term report")
    head, sub = _title(r)
    story: list[Any] = []
    if letterhead is not None and letterhead.name:
        block = [Paragraph(escape(letterhead.name), styles["Heading1"])]
        if letterhead.address_line:
            block.append(Paragraph(escape(letterhead.address_line), styles["Normal"]))
        if letterhead.logo is not None:
            from reportlab.platypus import Image
            story.append(Table([[Image(str(letterhead.logo), width=16 * mm, height=16 * mm), block]],
                               colWidths=[20 * mm, None], hAlign="LEFT",
                               style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")])))
        else:
            story.extend(block)
    story += [Paragraph(escape(head), styles["Title"]), Paragraph(escape(sub), styles["Normal"])]
    if r.scope.note:
        story.append(Paragraph(escape(r.scope.note), styles["Italic"]))
    for name, rows in _tables(r):
        story += [Spacer(1, 5 * mm), Paragraph(escape(name), styles["Heading2"])]
        rows = [row for row in rows if row and any(v not in ("", None) for v in row)]
        if len(rows) < 2:
            story.append(Paragraph("Nothing to show for this scope yet.", styles["Normal"]))
            continue
        width = max(len(row) for row in rows)
        data = [[Paragraph(escape("" if v is None else ("yes" if v is True else "no" if v is False else str(v))), cell)
                 for v in row] + [""] * (width - len(row)) for row in rows]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#B0B8C4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(table)
    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------- routes

def _audit(user: User, what: str, r: TermReport) -> None:
    from ..assessment.audit_log import get_audit_log
    get_audit_log(cr._cfg.data_root).append(what, actor=user.id, details={
        "schoolId": user.school_id, "scope": r.scope.name, "grade": r.filters.grade,
        "sectionId": r.filters.section_id, "subject": r.filters.subject})


@router.get("/dashboard/term", response_model=TermReport)
def term_report(term_id: Optional[str] = Query(None, alias="termId"), grade: Optional[int] = Query(None, ge=1, le=12),
                section_id: Optional[str] = Query(None, alias="sectionId"), subject: Optional[str] = None,
                principal: User = Depends(require_admin("reports"))) -> TermReport:
    """The term at a glance, for the principal or a reports admin, filtered
    by class, section or subject. By default the term covering today; with no
    term declared, the year so far (and the response says so)."""
    return build(principal, term_id=term_id, grade=grade, section_id=section_id, subject=subject)


@router.get("/dashboard/term/export")
def export_term_report(format: Literal["xlsx", "pdf"] = "xlsx",
                       term_id: Optional[str] = Query(None, alias="termId"),
                       grade: Optional[int] = Query(None, ge=1, le=12),
                       section_id: Optional[str] = Query(None, alias="sectionId"), subject: Optional[str] = None,
                       principal: User = Depends(require_admin("reports"))) -> Response:
    """The same report as an Excel workbook or a PDF (ADM-4). Every download
    is recorded in the audit log with who took it."""
    r = build(principal, term_id=term_id, grade=grade, section_id=section_id, subject=subject)
    _audit(principal, "term_report_exported", r)
    stem = "".join(ch if ch.isalnum() else "-" for ch in f"term-report-{r.scope.name}-{r.scope.as_of}").strip("-")
    letterhead = Letterhead.for_school(principal.school_id)
    if format == "pdf":
        return Response(to_pdf(r, letterhead), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{stem}.pdf"'})
    return Response(to_xlsx(r, letterhead), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{stem}.xlsx"'})
