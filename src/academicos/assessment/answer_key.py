"""Official answer keys, lifted from CBSE marking schemes.

The question papers say what was asked; only the marking schemes say what a
correct answer looks like. Without them every MCQ evaluates to "no correct
option recorded" and every descriptive answer is scored against an empty
rubric, so the teacher review queue is noise.

A marking scheme PDF is laid out as a table:

    Q.No.   EXPECTED ANSWERS / VALUE POINTS                      Marks  Total
    1.      D /1: 8                                                  1      1
    2.      B / Al2O3 and MgO                                        1      1
    ...
    21.     * Evolution of gas                                       1
            * Change / Rise in temperature                           1      2

Section A (objective) gives a letter and the option text. Sections B onwards
give value points — the individual scoring steps an examiner ticks — each with
its own mark. Both are worth capturing: the letter drives automatic MCQ
scoring, the value points become the marking points a teacher sees.

One PDF often covers several paper variants (31/1/1 through 31/1/3) with the
question numbering restarting at each. Every page carries its own paper code in
the header, so pages are grouped by code before parsing rather than treating
the file as one document.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config

log = logging.getLogger(__name__)

# "31/1/1", "X_086_ 31/1/1", "Paper Code: 30-1-1". Two-to-three digit series,
# then set, then variant.
_PAPER_CODE = re.compile(r"\b(\d{2,3})\s*[/-]\s*(\d)\s*[/-]\s*(\d)\b")

# Subject codes as they appear in CBSE marking-scheme filenames and headers.
_SUBJECT_BY_CODE = {
    "086": "Science",
    "041": "Mathematics",
    "241": "Mathematics",
    "087": "Social Science",
    "184": "English",
    "101": "English",
    "042": "Physics",
    "043": "Chemistry",
    "044": "Biology",
    "055": "Accountancy",
    "054": "Business Studies",
    "030": "Economics",
    "027": "History",
    "029": "Geography",
    "028": "Political Science",
}

# Marking-scheme boilerplate: the examiner instructions that precede the actual
# answers. A "1." inside these pages is an instruction number, not a question.
_PREAMBLE_END = re.compile(r"SECTION\s*[-–—]?\s*A\b", re.I)

# Section A objective answer, two layouts seen across subjects:
#   Science: "1.\nD / 1: 8"                  — letter and text on one line
#   Math:    "1.\n...\nSol.\n(d) answer"      — letter in parens after "Sol."
# Both end the answer at the mark column (a line holding only a mark).
_MCQ_SLASH = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*\n"          # 1.
    r"[ \t]*([A-D])[ \t]*/[ \t]*"                  # D /
    r"(.*?)"                                       # answer text (may wrap)
    r"(?=\n[ \t]*(?:1|2|½|\d{1,2}\s*\.[ \t]*\n))",  # up to marks or next Q
    re.S,
)
_MCQ_SOL = re.compile(
    r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*\n"          # 1.
    r"(?:[ \t]*\n)*"                               # blank lines before Sol.
    r"[ \t]*Sol\.?[ \t]*\n"                        # Sol.
    r"[ \t]*\(([a-dA-D])\)[ \t]*"                  # (d)
    r"(.*?)"                                       # answer text
    r"(?=\n[ \t]*(?:1|2|½|\d{1,2}\s*\.[ \t]*\n))",
    re.S,
)

# A value point: a bulleted/dashed scoring step in sections B onwards.
_VALUE_POINT = re.compile(r"(?m)^[ \t]*[•▪●➢*•][ \t]*(.+)$")

# A mark token standing alone in the marks column.
_MARK_TOKEN = re.compile(r"^\s*(½|\d+(?:\s*½)?)\s*$")


def normalize_paper_code(raw: str) -> str:
    """Renders any of 31/1/1, 31-1-1, '31 / 1 / 1' as the canonical 31/1/1."""
    m = _PAPER_CODE.search(raw)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def subject_from_path(path: Path) -> str | None:
    """Infers the subject from a marking-scheme path.

    CBSE nests these as `.../Science/086 Science (English Medium)/X_086_...pdf`,
    so the subject code in the filename is the most reliable signal; the folder
    name is the fallback for files that omit it.
    """
    text = str(path)
    for code, subject in _SUBJECT_BY_CODE.items():
        if re.search(rf"[_\s/\\]{code}[_\s]", text):
            return subject
    lowered = text.lower()
    for subject in ("social science", "science", "mathematics", "math", "english"):
        if subject in lowered:
            return "Mathematics" if subject == "math" else subject.title()
    return None


def grade_from_path(path: Path) -> str | None:
    """Class X vs XII, from the `/2025/X/` segment CBSE uses."""
    for part in path.parts:
        if part in ("X", "XII"):
            return part
    if re.search(r"[\\/]XII[_\s]", str(path)):
        return "XII"
    if re.search(r"[\\/]X[_\s]", str(path)):
        return "X"
    return None


@dataclass
class KeyedAnswer:
    """The official answer to one question of one paper variant."""
    q_no: int
    correct_option: str | None          # "D" for objective questions
    answer_text: str                    # option text, or the joined value points
    value_points: list[tuple[str, float]] = field(default_factory=list)
    marks: float = 0.0


@dataclass
class PaperKey:
    paper_code: str
    subject: str | None
    grade: str | None
    answers: dict[int, KeyedAnswer] = field(default_factory=dict)


def _clean(text: str) -> str:
    return " ".join(text.split()).strip(" .:/")


def _pages_by_paper_code(pdf_path: Path) -> dict[str, str]:
    """Groups a marking scheme's pages by the paper code in each page header.

    A single PDF routinely covers 31/1/1 through 31/1/3 with question numbers
    restarting at 1 for each, so parsing the file as one blob would collide the
    variants' answers.
    """
    import fitz

    grouped: dict[str, list[str]] = {}
    current = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text()
            # The code lives in the running header, so only look at the top.
            code = normalize_paper_code(text[:300])
            if code:
                current = code
            if not current:
                continue
            grouped.setdefault(current, []).append(text)
    return {code: "\n".join(pages) for code, pages in grouped.items()}


def _parse_objective(body: str) -> dict[int, KeyedAnswer]:
    """Pulls Section A letter answers out of one paper variant's text."""
    answers: dict[int, KeyedAnswer] = {}
    for pattern in (_MCQ_SLASH, _MCQ_SOL):
        for m in pattern.finditer(body):
            q_no = int(m.group(1))
            # Section A is questions 1-20; a later match with a low number means
            # the regex wandered into a different section, so ignore it.
            if not 1 <= q_no <= 20 or q_no in answers:
                continue
            answers[q_no] = KeyedAnswer(
                q_no=q_no,
                correct_option=m.group(2).upper(),
                answer_text=_clean(m.group(3)),
                marks=1.0,
            )
    return answers


def _parse_value_points(body: str) -> dict[int, KeyedAnswer]:
    """Pulls sections B+ value points, keyed by question number.

    Splits on question-number lines and collects the bulleted scoring steps
    inside each block. Marks per point are not reliably alignable to their text
    in a flattened PDF (the mark column is emitted separately), so each point
    carries 0.0 and the teacher allocates — better than inventing a number.
    """
    answers: dict[int, KeyedAnswer] = {}
    blocks = re.split(r"(?m)^[ \t]*(\d{1,2})\s*\.[ \t]*$", body)
    # re.split with one group yields [pre, qno, block, qno, block, ...]
    for i in range(1, len(blocks) - 1, 2):
        try:
            q_no = int(blocks[i])
        except ValueError:
            continue
        if q_no <= 20 or q_no in answers:  # Section A handled separately
            continue
        block = blocks[i + 1]
        points = [_clean(p) for p in _VALUE_POINT.findall(block)]
        points = [p for p in points if len(p) > 8][:8]
        if not points:
            # Unbulleted prose answer: keep the first substantial lines.
            lines = [_clean(l) for l in block.splitlines()]
            points = [l for l in lines if len(l) > 25][:4]
        if not points:
            continue
        answers[q_no] = KeyedAnswer(
            q_no=q_no,
            correct_option=None,
            answer_text=" ".join(points)[:1200],
            value_points=[(p, 0.0) for p in points],
        )
    return answers


def parse_marking_scheme(pdf_path: Path) -> list[PaperKey]:
    """Extracts every paper variant's answer key from one marking-scheme PDF."""
    subject = subject_from_path(pdf_path)
    grade = grade_from_path(pdf_path)
    keys: list[PaperKey] = []
    try:
        grouped = _pages_by_paper_code(pdf_path)
    except Exception as exc:  # a corrupt or image-only PDF must not abort the run
        log.warning("cannot read marking scheme %s: %s", pdf_path.name, exc)
        return keys

    for code, body in grouped.items():
        # Drop the examiner-instruction preamble so its numbered list is not
        # mistaken for answers.
        cut = _PREAMBLE_END.search(body)
        answers_body = body[cut.start():] if cut else body
        answers = _parse_objective(answers_body)
        answers.update(_parse_value_points(answers_body))
        if answers:
            keys.append(PaperKey(paper_code=code, subject=subject,
                                 grade=grade, answers=answers))
    return keys


def discover_marking_schemes(root: Path) -> list[Path]:
    """Every marking-scheme PDF under a corpus root, archives already expanded."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


class AnswerKeyStore:
    """SQLite-backed lookup of official answers by (subject, grade, paper, q_no)."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        # Paper codes ("31/1/1") are only unique WITHIN a subject/grade — CBSE
        # reuses the same numeric code space across subjects (Science and
        # Social Science both had "32/1/1" papers), so a key of paper_code+q_no
        # alone silently overwrites one subject's answers with another's.
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS answer_keys (
                subject        TEXT NOT NULL DEFAULT '',
                grade          TEXT NOT NULL DEFAULT '',
                paper_code     TEXT NOT NULL,
                q_no           INTEGER NOT NULL,
                correct_option TEXT,
                answer_text    TEXT NOT NULL,
                value_points   TEXT NOT NULL DEFAULT '[]',
                marks          REAL NOT NULL DEFAULT 0,
                source         TEXT,
                PRIMARY KEY (subject, grade, paper_code, q_no)
            );
            CREATE INDEX IF NOT EXISTS idx_keys_subject
                ON answer_keys (subject, grade);
            """
        )
        self.conn.commit()

    def put(self, key: PaperKey, source: str) -> int:
        import json

        subject = key.subject or ""
        grade = key.grade or ""
        rows = [
            (subject, grade, key.paper_code, a.q_no, a.correct_option,
             a.answer_text, json.dumps(a.value_points), a.marks, source)
            for a in key.answers.values()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO answer_keys (subject, grade, paper_code, q_no, "
            "correct_option, answer_text, value_points, marks, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def lookup(self, paper_code: str, q_no: int, *,
               subject: str | None = None, grade: str | None = None) -> KeyedAnswer | None:
        import json

        if subject is not None:
            row = self.conn.execute(
                "SELECT * FROM answer_keys WHERE paper_code=? AND q_no=? "
                "AND subject=? AND (grade=? OR grade='')",
                (paper_code, q_no, subject, grade or ""),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM answer_keys WHERE paper_code=? AND q_no=?",
                (paper_code, q_no),
            ).fetchone()
        if row is None:
            return None
        return KeyedAnswer(
            q_no=row["q_no"],
            correct_option=row["correct_option"],
            answer_text=row["answer_text"],
            value_points=[tuple(p) for p in json.loads(row["value_points"])],
            marks=row["marks"] or 0.0,
        )

    def stats(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) total, "
            "SUM(CASE WHEN correct_option IS NOT NULL THEN 1 ELSE 0 END) objective, "
            "COUNT(DISTINCT paper_code) papers FROM answer_keys"
        ).fetchone()
        return {"total": row["total"] or 0,
                "objective": row["objective"] or 0,
                "papers": row["papers"] or 0}

    def close(self) -> None:
        self.conn.close()


def default_db(cfg: Config) -> Path:
    return cfg.data_root / "answer_keys.db"


def build_answer_keys(cfg: Config, roots: list[Path]) -> dict[str, int]:
    """Parses every marking scheme under `roots` into the answer-key store."""
    store = AnswerKeyStore(default_db(cfg))
    files = [p for root in roots for p in discover_marking_schemes(root)]
    written = papers = 0
    for path in files:
        for key in parse_marking_scheme(path):
            written += store.put(key, source=path.name)
            papers += 1
    stats = store.stats()
    store.close()
    log.info("answer keys: %d answers across %d paper variants from %d files",
             stats["total"], stats["papers"], len(files))
    return {"files": len(files), "written": written, "papers": papers, **stats}
