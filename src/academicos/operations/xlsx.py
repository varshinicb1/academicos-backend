"""A small Excel (.xlsx) writer: named sheets of rows, the first row bold.

The reports need a file a principal can open in Excel, and nothing more
than cells: text and numbers, no formulas, no formatting beyond a header.
openpyxl is not a dependency of the deployed image, and an .xlsx is a zip of
a few XML parts, so this writes them directly. Text goes in as inline
strings, escaped; a cell that would start a formula (=, +, -, @) is kept as
text, so a name like "=HYPERLINK(...)" typed into the school's data cannot
run when the file is opened.
"""
from __future__ import annotations

import io
import zipfile
from typing import Any, Iterable, Sequence
from xml.sax.saxutils import escape

_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{sheets}
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>
</styleSheet>"""


def _col(i: int) -> str:
    name = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        name = chr(65 + r) + name
    return name


def _cell(ref: str, value: Any, bold: bool) -> str:
    style = ' s="1"' if bold else ""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f'<c r="{ref}"{style}><v>{value}</v></c>'
    text = str(value)
    if text.startswith(_FORMULA_START):
        text = "'" + text
    return f'<c r="{ref}"{style} t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'


def _sheet(rows: Iterable[Sequence[Any]]) -> str:
    out = []
    for r, row in enumerate(rows, start=1):
        cells = "".join(_cell(f"{_col(c)}{r}", v, r == 1) for c, v in enumerate(row))
        out.append(f'<row r="{r}">{cells}</row>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(out)}</sheetData></worksheet>')


def _sheet_name(name: str, taken: set[str]) -> str:
    clean = "".join(ch for ch in name if ch not in '[]:*?/\\')[:31] or "Sheet"
    base, n = clean, 2
    while clean.casefold() in taken:
        suffix = f" ({n})"
        clean, n = base[:31 - len(suffix)] + suffix, n + 1
    taken.add(clean.casefold())
    return clean


def workbook(sheets: Sequence[tuple[str, Sequence[Sequence[Any]]]]) -> bytes:
    """An .xlsx with one sheet per (name, rows); each first row is bold."""
    if not sheets:
        sheets = [("Sheet", [])]
    taken: set[str] = set()
    names = [_sheet_name(n, taken) for n, _ in sheets]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES.format(sheets="".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheets) + 1))))
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("xl/workbook.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                   'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                   + "".join(f'<sheet name="{escape(n, {chr(34): "&quot;"})}" sheetId="{i}" r:id="rId{i}"/>'
                             for i, n in enumerate(names, start=1))
                   + "</sheets></workbook>")
        z.writestr("xl/_rels/workbook.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(f'<Relationship Id="rId{i}" '
                             'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                             f'Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets) + 1))
                   + f'<Relationship Id="rId{len(sheets) + 1}" '
                     'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
                     'Target="styles.xml"/></Relationships>')
        z.writestr("xl/styles.xml", _STYLES)
        for i, (_, rows) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet(rows))
    return buf.getvalue()


# ---------------------------------------------------------------- reading

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
MAX_UNPACKED = 20_000_000          # bytes, all parts together: a real school sheet is far smaller
_BUILTIN_DATE_FORMATS = set(range(14, 23)) | {45, 46, 47}


class XlsxError(ValueError):
    """A file that is not a workbook this reader will open; the message is
    for the person who chose it."""


def _cell_index(ref: str) -> int:
    col = 0
    for ch in ref:
        if not ch.isalpha():
            break
        col = col * 26 + (ord(ch.upper()) - 64)
    return col - 1


def _date_styles(z: zipfile.ZipFile, read) -> set[int]:
    """Indexes of the cell styles that show a number as a date."""
    import xml.etree.ElementTree as ET
    if "xl/styles.xml" not in z.namelist():
        return set()
    root = ET.fromstring(read("xl/styles.xml"))
    custom = {int(f.get("numFmtId")): (f.get("formatCode") or "").lower()
              for f in root.iterfind("m:numFmts/m:numFmt", _NS)}

    def is_date(fmt_id: int) -> bool:
        if fmt_id in _BUILTIN_DATE_FORMATS:
            return True
        code = custom.get(fmt_id, "")
        # A date format shows d, m or y outside quoted text and [colour] tags.
        import re
        bare = re.sub(r'"[^"]*"|\[[^\]]*\]', "", code)
        return bool(re.search(r"[dy]", bare)) or ("m" in bare and "h" not in bare and "s" not in bare)

    return {i for i, xf in enumerate(root.iterfind("m:cellXfs/m:xf", _NS)) if is_date(int(xf.get("numFmtId", "0")))}


def read_rows(data: bytes) -> list[list[str]]:
    """The first sheet of an .xlsx as rows of text: numbers without a
    trailing ".0", dates as YYYY-MM-DD, blanks as "". Formulas give their
    saved value. Raises XlsxError for anything else."""
    import xml.etree.ElementTree as ET
    from datetime import date, timedelta
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise XlsxError("this is not an Excel .xlsx file (save it as .xlsx or .csv and try again)") from None
    with z:
        if sum(i.file_size for i in z.infolist()) > MAX_UNPACKED or len(z.infolist()) > 500:
            raise XlsxError("this workbook is too large to import; keep one sheet of the rows to import")

        def read(name: str) -> bytes:
            raw = z.read(name)
            if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
                raise XlsxError("this workbook contains XML this importer does not open")
            return raw

        try:
            book = ET.fromstring(read("xl/workbook.xml"))
            first = book.find("m:sheets/m:sheet", _NS)
            if first is None:
                raise XlsxError("this workbook has no sheets")
            rels = ET.fromstring(read("xl/_rels/workbook.xml.rels"))
            target = next((r.get("Target") for r in rels if r.get("Id") == first.get(_REL)), None)
            if not target:
                raise XlsxError("this workbook's first sheet could not be found")
            path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                for si in ET.fromstring(read("xl/sharedStrings.xml")).iterfind("m:si", _NS):
                    shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
            dates = _date_styles(z, read)
            sheet = ET.fromstring(read(path))
        except KeyError as e:
            raise XlsxError(f"this workbook is missing a part ({e}); save it again from Excel") from None
        except ET.ParseError:
            raise XlsxError("this workbook could not be read; save it again from Excel or as .csv") from None

        rows: list[list[str]] = []
        for row in sheet.iterfind("m:sheetData/m:row", _NS):
            r_index = int(row.get("r", len(rows) + 1)) - 1
            while len(rows) < r_index:
                rows.append([])
            cells: list[str] = []
            for c in row.iterfind("m:c", _NS):
                col = _cell_index(c.get("r", "")) if c.get("r") else len(cells)
                kind, style = c.get("t"), int(c.get("s", "0"))
                v = c.find("m:v", _NS)
                if kind == "inlineStr":
                    text = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                elif v is None or v.text is None:
                    text = ""
                elif kind == "s":
                    text = shared[int(v.text)] if int(v.text) < len(shared) else ""
                elif kind in ("str", "e"):
                    text = v.text
                elif kind == "b":
                    text = "TRUE" if v.text == "1" else "FALSE"
                else:
                    num = float(v.text)
                    if style in dates and 0 < num < 2958466:
                        text = (date(1899, 12, 30) + timedelta(days=int(num))).isoformat()
                    else:
                        text = str(int(num)) if num == int(num) else repr(num)
                while len(cells) < col:
                    cells.append("")
                cells.append(text.strip())
            rows.append(cells)
        return rows
