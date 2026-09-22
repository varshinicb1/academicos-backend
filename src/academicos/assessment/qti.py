"""QTI 2.1 export -- the interoperability format the assessment sector speaks.

Why this is here
----------------
`docs/research/case-studies.md` section 5 surveyed the assessment category and
found four capabilities this bank lacked. This is the fourth:

> **QTI export.** It is the interoperability format the whole sector speaks. If
> the question bank is to be an asset, QTI is a distribution channel we would be
> foolish to ignore.

Two audiences, and the difference matters:

  * **Outward** -- a school migrating to Moodle, Canvas or Blackboard can take
    its bank with it. That is a de-risking promise, not a feature.
  * **Inward** -- the same shape lets a third-party item bank arrive here
    without a bespoke importer.

Scope, stated honestly
----------------------
This implements the subset that carries the questions we actually hold:

  * `assessmentItem` for single-choice (MCQ) and extended-text (everything
    else), which is every question type in the corpus.
  * `assessmentTest` with one `testPart` and one `assessmentSection`, plus the
    `SCORE` outcome the spec requires.
  * A package: `imsmanifest.xml` plus the item files plus the test file, zipped.

Not implemented, and not claimed: adaptive items, response processing rules
beyond a plain correct/incorrect, media, per-item rubrics, and QTI 1.2 (legacy,
but some Indian LMS deployments still accept only 1.2 -- a real follow-up, not
an oversight we can hide).

The value points travel in `rubricBlock`, which is where QTI puts marking
guidance, so the marking scheme survives the round trip rather than being lost
at the boundary.
"""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any, Iterable
from xml.sax.saxutils import escape, quoteattr

QTI_NS = "http://www.imsglobal.org/xsd/imsqti_v2p1"
IMS_NS = "http://www.imsglobal.org/xsd/imscp_v1p1"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# The four options a CBSE MCQ carries, in order. `a`-`d` is what the extractor
# produces; `A`-`D` is what is printed.
_OPTION_LETTERS = "abcd"

# "(a) option text", "(A) option text", "a) text" -- the shapes the corpus has.
_OPTION_SPLIT = re.compile(r"\((?:a|A)\)|\((?:b|B)\)|\((?:c|C)\)|\((?:d|D)\)")


def _safe_identifier(raw: str) -> str:
    """QTI identifiers are XML NCNames: no spaces, no colons, no leading digit.

    A canonical question id is `cbse:q:src:<hash>:<n>` -- every one of those is
    illegal here, so it is rewritten rather than passed through and rejected by
    the importing LMS with an unhelpful error.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", raw or "item")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "q-" + cleaned
    # Preserve the original so a round trip can map back.
    return cleaned[:220]


def split_options(stem: str) -> tuple[str, list[str]]:
    """Split an extracted MCQ stem into its prompt and its options.

    The corpus stores options inline in the stem because that is how the PDF
    laid them out. Returns `(prompt, options)`; an empty list means this is not
    really an MCQ and should be exported as free text instead.
    """
    matches = list(_OPTION_SPLIT.finditer(stem or ""))
    if len(matches) < 2:
        return (stem or "").strip(), []
    prompt = stem[:matches[0].start()].strip()
    options: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(stem)
        text = stem[m.end():end].strip()
        if text:
            options.append(text)
    # A single surviving option is not a choice question.
    return prompt, options if len(options) >= 2 else []


def _rubric_block(question: dict[str, Any]) -> str:
    """Marking guidance, as QTI's rubricBlock.

    This is the part worth getting right: the official CBSE value points are the
    scarce half of the bank, and a format that drops them exports the question
    and loses the reason it is valuable.
    """
    scheme = question.get("answerScheme") or {}
    points = scheme.get("markingPoints") or []
    if not points:
        model = (scheme.get("modelAnswer") or "").strip()
        if not model:
            return ""
        body = escape(model)
    else:
        body = "\n".join(
            "• {}{}".format(
                escape(str(p.get("description") or "")),
                " ({} mark{})".format(p["marks"], "" if p["marks"] == 1 else "s")
                if p.get("marks") else "",
            )
            for p in points
        )
    origin = scheme.get("provenance")
    label = "Official CBSE marking scheme" if origin == "cbse_marking_scheme" \
        else "Marking guidance"
    code = scheme.get("sourcePaperCode")
    if code:
        label += " — paper {}".format(code)
    return (
        '<rubricBlock view="scorer">'
        "<p><strong>{}</strong></p><pre>{}</pre>"
        "</rubricBlock>"
    ).format(escape(label), body)


def to_qti_item(question: dict[str, Any], *, identifier: str | None = None) -> str:
    """One `assessmentItem`. Single-choice when options parse, else free text."""
    qid = str(question.get("id") or "item")
    ident = _safe_identifier(identifier or qid)
    prompt, options = split_options(str(question.get("stem") or ""))
    title = (prompt or qid)[:120]
    marks = int(question.get("marks") or 0)

    scheme = question.get("answerScheme") or {}
    correct = ""
    for i, opt in enumerate(options):
        if scheme.get("modelAnswer") and str(scheme["modelAnswer"]).strip() == opt.strip():
            correct = _OPTION_LETTERS[i].upper()
            break
    if not correct and options:
        # `correctOption` is stored per question in some paths; accept it.
        raw = str(question.get("correctOption") or "").strip().upper()
        if raw and raw[0] in _OPTION_LETTERS.upper():
            correct = raw[0]

    if options:
        response_decl = (
            '<responseDeclaration identifier="RESPONSE" cardinality="single" '
            'baseType="identifier">'
            + ("<correctResponse><value>{}</value></correctResponse>".format(correct)
               if correct else "")
            + "</responseDeclaration>"
        )
        choices = "".join(
            '<simpleChoice identifier="{}">{}</simpleChoice>'.format(
                _OPTION_LETTERS[i].upper(), escape(opt))
            for i, opt in enumerate(options)
        )
        body = (
            '<choiceInteraction responseIdentifier="RESPONSE" shuffle="false" '
            'maxChoices="1"><prompt>{}</prompt>{}</choiceInteraction>'
        ).format(escape(prompt), choices)
    else:
        response_decl = (
            '<responseDeclaration identifier="RESPONSE" cardinality="single" '
            'baseType="string"/>'
        )
        body = (
            '<extendedTextInteraction responseIdentifier="RESPONSE" '
            'expectedLength="{}"><prompt>{}</prompt></extendedTextInteraction>'
        ).format(max(80, marks * 120), escape(prompt or ""))

    province = question.get("provenance") or {}
    source_note = ""
    if province.get("sourceDocumentId"):
        source_note = (
            '<itemMetadata><qtimetadata>'
            '<qtimetadatafield><fieldlabel>sourceDocument</fieldlabel>'
            "<fieldentry>{}</fieldentry></qtimetadatafield>"
            '<qtimetadatafield><fieldlabel>pageNumber</fieldlabel>'
            "<fieldentry>{}</fieldentry></qtimetadatafield>"
            '<qtimetadatafield><fieldlabel>canonicalId</fieldlabel>'
            "<fieldentry>{}</fieldentry></qtimetadatafield>"
            "</qtimetadata></itemMetadata>"
        ).format(escape(str(province.get("sourceDocumentId"))),
                 escape(str(province.get("pageNumber") or "")),
                 escape(qid))

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<assessmentItem xmlns="{ns}" xmlns:xsi="{xsi}" '
        'xsi:schemaLocation="{ns} imsQTI_v2p1.xsd" '
        "identifier={ident} title={title} adaptive=\"false\" timeDependent=\"false\">\n"
        "  {response}\n"
        '  <outcomeDeclaration identifier="SCORE" cardinality="single" baseType="float"/>\n'
        "  <itemBody>{body}</itemBody>\n"
        "{rubric}\n"
        "{meta}\n"
        "</assessmentItem>\n"
    ).format(
        ns=QTI_NS, xsi=XSI_NS, ident=quoteattr(ident), title=quoteattr(title),
        response=response_decl, body=body,
        rubric=("  " + _rubric_block(question) if _rubric_block(question) else ""),
        meta=("  " + source_note if source_note else ""),
    )


def to_qti_test(questions: Iterable[dict[str, Any]], *, title: str = "AcademicOS export",
                identifier: str = "academicos-test") -> str:
    """One `assessmentTest` referencing every item."""
    items = list(questions)
    refs = "".join(
        '      <itemRef identifierref="{}"/>\n'.format(
            escape(_safe_identifier(str(q.get("id") or "item"))))
        for q in items
    )
    total = sum(int(q.get("marks") or 0) for q in items)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<assessmentTest xmlns="{ns}" xmlns:xsi="{xsi}" '
        'xsi:schemaLocation="{ns} imsQTI_v2p1.xsd" '
        "identifier={ident} title={title}>\n"
        '  <outcomeDeclaration identifier="SCORE" cardinality="single" baseType="float"/>\n'
        '  <testPart identifier="part1" navigationMode="linear" '
        'submissionMode="individual">\n'
        '    <itemSessionControl maxAttempts="0" allowReview="true"/>\n'
        '    <assessmentSection identifier="section1" title="Section 1" visible="true">\n'
        "{refs}"
        "    </assessmentSection>\n"
        "  </testPart>\n"
        "</assessmentTest>\n"
    ).format(ns=QTI_NS, xsi=XSI_NS, ident=quoteattr(identifier),
             title=quoteattr("{} ({} marks)".format(title, total)), refs=refs)


def to_qti_package(questions: Iterable[dict[str, Any]], *,
                   title: str = "AcademicOS export") -> bytes:
    """A zipped QTI 2.1 package: manifest + items + test.

    The zip is what an LMS actually accepts; returning loose XML would push the
    packaging problem onto every consumer.
    """
    items = list(questions)
    manifest_resources = []
    files: dict[str, str] = {}

    for q in items:
        ident = _safe_identifier(str(q.get("id") or "item"))
        fname = "items/{}.xml".format(ident)
        files[fname] = to_qti_item(q, identifier=ident)
        manifest_resources.append(
            '    <resource identifier={ident} type="imsqti_item_xmlv2p1" '
            "href={href}>\n"
            '      <file href={href}/>\n'
            "    </resource>".format(ident=quoteattr(ident), href=quoteattr(fname))
        )

    files["test.xml"] = to_qti_test(items, title=title)
    manifest_resources.append(
        '    <resource identifier="test" type="imsqti_test_xmlv2p1" '
        'href="test.xml">\n      <file href="test.xml"/>\n    </resource>'
    )

    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<manifest xmlns="{ims}" xmlns:xsi="{xsi}" '
        'xsi:schemaLocation="{ims} imscp_v1p1.xsd" identifier="academicos-package">\n'
        '  <metadata><schema>QTIv2.1</schema><schemaversion>1.0.0</schemaversion></metadata>\n'
        "  <organizations/>\n"
        "  <resources>\n{res}\n  </resources>\n"
        "</manifest>\n"
    ).format(ims=IMS_NS, xsi=XSI_NS, res="\n".join(manifest_resources))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()
