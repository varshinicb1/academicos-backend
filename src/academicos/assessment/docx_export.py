"""A question paper as an editable Word document (EX-5, DOCX export).

The same paper the PDF prints -- the school's name, the exam, subject and
class, time and marks, the general instructions, each section with its
note, each question numbered with its marks at the right margin, options
(A)-(D), and an internal choice as "OR" -- written as a .docx a teacher can
open and change in Word or Google Docs. It uses the PDF's own helpers for
options, instructions and section notes, so the two cannot say different
things. Every page's footer carries the Q.P. code and the export's
traceable-copy id, as the PDF's does.

A .docx is a zip of a few XML parts; this writes them directly, with no new
dependency (python-docx is not in the image).
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from xml.sax.saxutils import escape

from .duration import format_duration
from .pdf import DEFAULT_INSTRUCTIONS, _section_note, instruction_lines, split_stem_and_options

# A4 with 18 mm margins, in twentieths of a point.
_PAGE_W, _PAGE_H, _MARGIN = 11906, 16838, 1020
_TEXT_W = _PAGE_W - 2 * _MARGIN
_NUMBER_W = 567                      # 1 cm for the question number

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' \
     'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
# For the logo's inline picture.
_DRAWING_NS = ('xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
               'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
               'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')


_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _run(text: str, *, bold: bool = False, size: Optional[int] = None, italic: bool = False) -> str:
    text = _XML_INVALID.sub("", text)   # a control character from a scanned stem would corrupt the file
    props = ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "") + (f'<w:sz w:val="{size}"/>' if size else "")
    parts = text.split("\t")
    body = '<w:tab/>'.join(f'<w:t xml:space="preserve">{escape(p)}</w:t>' for p in parts)
    return f"<w:r>{f'<w:rPr>{props}</w:rPr>' if props else ''}{body}</w:r>"


def _p(*runs: str, align: Optional[str] = None, space_after: int = 80, space_before: int = 0,
       indent: Optional[tuple[int, int]] = None, tabs: Optional[list[tuple[str, int]]] = None,
       keep_next: bool = False, border_bottom: bool = False) -> str:
    props = ""
    if keep_next:
        props += "<w:keepNext/>"
    if border_bottom:
        props += '<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="1" w:color="000000"/></w:pBdr>'
    if tabs:
        props += "<w:tabs>" + "".join(f'<w:tab w:val="{k}" w:pos="{pos}"/>' for k, pos in tabs) + "</w:tabs>"
    props += f'<w:spacing w:before="{space_before}" w:after="{space_after}"/>'
    if indent:
        props += f'<w:ind w:left="{indent[0]}" w:hanging="{indent[1]}"/>'
    if align:
        props += f'<w:jc w:val="{align}"/>'
    return f"<w:p><w:pPr>{props}</w:pPr>{''.join(runs)}</w:p>"


def _question(gq) -> list[str]:
    """One question: number, stem, marks at the right margin; options and
    an OR choice below, indented under the stem."""
    split_head, split_options = split_stem_and_options(gq.stem)
    objective = gq.type in ("", "mcq", "assertion_reason") or (gq.marks == 1 and len(split_options) == 4)
    head, options = (split_head, split_options) if objective else (gq.stem.strip(), [])
    tabs = [("left", _NUMBER_W), ("right", _TEXT_W)]
    lines = head.splitlines() or [""]
    out = [_p(_run(f"{gq.display_number}.", bold=True), _run("\t" + lines[0] + f"\t[{gq.marks}]"),
              tabs=tabs, indent=(_NUMBER_W, _NUMBER_W), space_after=60, space_before=160)]
    out += [_p(_run(line), indent=(_NUMBER_W, 0), space_after=60) for line in lines[1:]]
    for i, opt in enumerate(options):
        out.append(_p(_run(f"({chr(ord('A') + i)})  {opt}"), indent=(_NUMBER_W * 2, _NUMBER_W), space_after=40))
    if gq.internal_choice_text:
        out.append(_p(_run("OR", bold=True), align="center", space_after=60))
        out.append(_p(_run(gq.internal_choice_text), indent=(_NUMBER_W, 0), space_after=60))
    return out


@dataclass(frozen=True)
class _Logo:
    data: bytes
    ext: str             # "png" or "jpeg": the part's name and its content type
    cx: int              # EMU
    cy: int


_EMU_PER_MM = 36000
_TWIPS_PER_MM = 56.6929
_LOGO_MM = 18            # the PDF draws the logo in an 18 mm box beside the title
_LOGO_CELL = round(22 * _TWIPS_PER_MM)


def _logo(template) -> Optional[_Logo]:
    """The school's logo, read from the same file the PDF draws
    (`template.logo_url`, which the export route sets from the school
    profile: `school_profile.branding_for_school`), sized to fit the PDF's
    18 mm box without stretching it. None when there is no logo, the file is
    not on this server, or it is not a PNG or JPEG image that can be read --
    the paper still prints, as the PDF does, without it."""
    from ..curriculum.school_profile import printable_logo
    # A header-only read passed a logo whose data is broken, and Word showed a
    # broken picture where the PDF printed none: both use the one full decode.
    path = printable_logo(Path(template.logo_url) if template is not None and template.logo_url else None)
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    ext = "png" if data.startswith(b"\x89PNG") else "jpeg" if data.startswith(b"\xff\xd8") else None
    if ext is None:
        return None
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
    except Exception:                    # a bad logo must not kill the export
        return None
    if width <= 0 or height <= 0:
        return None
    box = _LOGO_MM * _EMU_PER_MM
    scale = box / max(width, height)
    return _Logo(data, ext, max(1, round(width * scale)), max(1, round(height * scale)))


def _drawing(logo: _Logo) -> str:
    """An inline picture run, the image part being `rIdLogo`."""
    return (
        '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{logo.cx}" cy="{logo.cy}"/><wp:docPr id="1" name="School logo"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="logo.{logo.ext}"/><pic:cNvPicPr/></pic:nvPicPr>'
        '<pic:blipFill><a:blip r:embed="rIdLogo"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{logo.cx}" cy="{logo.cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


def _beside_logo(title: list[str], logo: Optional[_Logo]) -> list[str]:
    """The title block with the logo to its left, in a borderless two-cell
    table -- the band the PDF prints (`pdf._beside_logo`)."""
    if logo is None:
        return title
    none = "".join(f'<w:{side} w:val="nil"/>' for side in
                   ("top", "left", "bottom", "right", "insideH", "insideV"))
    rest = _TEXT_W - _LOGO_CELL
    cell = '<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>{body}</w:tc>'
    return [
        f'<w:tbl><w:tblPr><w:tblW w:w="{_TEXT_W}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        f'<w:tblBorders>{none}</w:tblBorders></w:tblPr>'
        f'<w:tblGrid><w:gridCol w:w="{_LOGO_CELL}"/><w:gridCol w:w="{rest}"/></w:tblGrid><w:tr>'
        + cell.format(w=_LOGO_CELL, body=f"<w:p>{_drawing(logo)}</w:p>")
        + cell.format(w=rest, body="".join(title) or "<w:p/>")
        + "</w:tr></w:tbl>"]


def _body(paper, template, logo: Optional[_Logo] = None) -> str:
    m = paper.metadata
    from .pdf import printable_address, printable_school_name
    school = printable_school_name(m.school_name, template)
    title_block = [_p(_run(school, bold=True, size=32), align="center", space_after=40)] if school else []
    if printable_address(school, template):
        title_block.append(_p(_run(printable_address(school, template)), align="center", space_after=40))
    if m.exam_name and m.exam_name != m.assessment_title:
        title_block.append(_p(_run(m.exam_name, bold=True, size=26), align="center", space_after=40))
    title = m.assessment_title + (f"  |  SET {paper.set_label}" if paper.set_label else "")
    title_block.append(_p(_run(title, bold=True, size=26), align="center", space_after=40))
    title_block.append(_p(_run(f"Subject: {m.subject}  |  Class: {m.grade}"), align="center", space_after=40))
    if template and template.tagline:
        title_block.append(_p(_run(template.tagline, italic=True), align="center", space_after=40))
    if m.date_line:
        title_block.append(_p(_run(m.date_line), align="center", space_after=40))
    paras = _beside_logo(title_block, logo)
    paras.append(_p(_run("Roll No.: ____________        Name: ______________________________"), space_after=120))
    marks = f"Maximum Marks: {m.total_marks}" + (f"   [SET {paper.set_label}]" if paper.set_label else "")
    paras.append(_p(_run(f"Time Allowed: {format_duration(m.duration_minutes)}",
                         bold=True), _run("\t" + marks, bold=True),
                    tabs=[("right", _TEXT_W)], border_bottom=True, space_after=160))

    paras.append(_p(_run("General Instructions:", bold=True), space_after=60, keep_next=True))
    teacher = instruction_lines(paper)
    if teacher:
        lines = teacher
    else:
        total_q = sum(len(s.questions) for s in paper.sections)
        labels = ", ".join(s.label for s in paper.sections)
        lines = [t.format(n=total_q, sections=len(paper.sections), labels=labels) for t in DEFAULT_INSTRUCTIONS]
    for i, line in enumerate(lines, start=1):
        paras.append(_p(_run(f"({i})\t{_plain(line)}"), tabs=[("left", _NUMBER_W)], indent=(_NUMBER_W, _NUMBER_W),
                        space_after=40))

    for section in paper.sections:
        paras.append(_p(_run(f"SECTION {section.label}", bold=True, size=26), align="center", space_after=40,
                        keep_next=True))
        paras.append(_p(_run(f"({section.name} — {len(section.questions)} questions, {section.total_marks} marks. "
                             f"This section {_section_note(section)})", italic=True), align="center",
                        space_after=160, keep_next=True))
        for gq in section.questions:
            paras += _question(gq)
    sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
            f'<w:pgSz w:w="{_PAGE_W}" w:h="{_PAGE_H}"/>'
            f'<w:pgMar w:top="{_MARGIN}" w:right="{_MARGIN}" w:bottom="{_MARGIN}" w:left="{_MARGIN}" '
            f'w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {_W} {_DRAWING_NS}>'
            "<w:body>" + "".join(paras) + sect + "</w:body></w:document>")


def _plain(markup: str) -> str:
    """DEFAULT_INSTRUCTIONS carry the PDF's inline tags; Word gets the text."""
    import re
    return re.sub(r"<[^>]+>", "", markup).replace("&nbsp;", " ").replace("&amp;", "&")


def _footer(paper, template, watermark_id: Optional[str]) -> str:
    code = paper.id.replace("paper_", "").upper()[:10]
    from .pdf import printable_school_name
    # The PDF footer's rule: the school's printable name or none, never a
    # placeholder or the product's name (audit D41).
    school = printable_school_name(paper.metadata.school_name, template)
    left = f"{school}  ·  Q.P. Code {code}" if school else f"Q.P. Code {code}"
    lines = [_p(_run(left, size=15), align="center", space_after=0)]
    if watermark_id:
        lines.append(_p(_run(f"Confidential — traceable copy {watermark_id}", size=12), align="center", space_after=0))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {_W}>' + "".join(lines) + "</w:ftr>")


def export_docx(paper, output_dir: Path, template=None, watermark_id: Optional[str] = None) -> Path:
    """The paper as a .docx. With a logo on the school's branding (the file
    the PDF draws), the Word file carries it too, beside the title as on the
    PDF: it had none (v3 audit N-2-9, 0 files in word/media on 42 of 42
    exported after a logo upload, while the PDFs printed it)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{paper.id}.docx"
    logo = _logo(template)
    image_types = ""
    image_rel = ""
    if logo is not None:
        image_types = f'<Default Extension="{logo.ext}" ContentType="image/{logo.ext}"/>'
        image_rel = ('<Relationship Id="rIdLogo" Type="http://schemas.openxmlformats.org/officeDocument/'
                     f'2006/relationships/image" Target="media/logo.{logo.ext}"/>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   + image_types +
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                   '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                   '</Relationships>')
        z.writestr("docProps/core.xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                   'xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:title>{escape(paper.metadata.assessment_title)}</dc:title><dc:creator>AcademicOS</dc:creator>'
                   '</cp:coreProperties>')
        z.writestr("word/_rels/document.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
                   + image_rel +
                   '</Relationships>')
        z.writestr("word/document.xml", _body(paper, template, logo))
        z.writestr("word/footer1.xml", _footer(paper, template, watermark_id))
        if logo is not None:
            z.writestr(f"word/media/logo.{logo.ext}", logo.data)
    out_path.write_bytes(buf.getvalue())
    return out_path
