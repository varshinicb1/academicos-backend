"""The question bank's health per class, subject, chapter and topic.

"How solid is the bank?" -- tagged to chapter, topic and subtopic, answer key
mapped, marks distributed, classes 6-10 -- used to take an ad-hoc script to
answer, and an ad-hoc script reports what exists. PRD rules Q3/Q4 ask for the
opposite as well: coverage is reported, never implied, and every chapter and
topic reports its gaps. A chapter with no 5-mark question is found here, not
by a teacher holding a board paper that came out short.

Pure: records and taxonomy trees in, a dict out (`build_report`), and that
dict rendered as Markdown (`render_markdown`). `scripts/bank_health.py` reads
the files and writes ``docs/bank-health.md`` and
``academicos-data/syllabus/bank_health.json``.

What counts
-----------
* Scope: classes 6-10 x Mathematics, Science, Social Science, English, Hindi.
  A paper variant ("Mathematics (Standard)", "Hindi Course A") counts under its
  subject; everything else is out of scope and not counted anywhere.
* Q1, "without an answer key, no question": a question counts as *keyed* by
  the served bank's own gate, `QuestionBank.has_answer_key` (content, not a
  provenance label). Marks, types, chapter/topic counts and the paper verdicts
  count keyed questions only, since nothing else can be served. Question
  counts and tag percentages count every in-scope question, keyed or not, so
  the unkeyed ones stay visible.
* Tags are the taxonomy tags the topic tagger writes (``taxonomyChapterId``,
  ``topicIds``, ``subtopicIds``). The older ``chapterIds`` are shown beside
  them but are a different, coarser thing: old syllabus slugs and unit names,
  many inferred by an earlier tagger.
* A paper is possible when every section of the preset can be filled the way
  `assessment.selection.optimize` fills it: exact marks, an allowed
  difficulty, and ``question_count`` + ``internal_choice_count`` distinct
  questions (the OR alternatives come out of the same pool).
"""
from __future__ import annotations

import collections
from typing import Iterable

from ..assessment.qbank_routes import QuestionBank
from ..assessment.templates import EXAM_PRESETS
from .tagger import SUBJECTS as TAGGER_SUBJECTS

SUBJECTS = ("Mathematics", "Science", "Social Science", "English", "Hindi")
GRADES = tuple(range(6, 11))
MARK_BUCKETS = ("1", "2", "3", "4", "5+")
LEVEL_FIELDS = {"chapter": "taxonomyChapterId", "topic": "topicIds", "subtopic": "subtopicIds"}


def subject_family(rec: dict) -> str | None:
    """The report subject of a record, or None when it is out of scope."""
    meta = rec.get("metadata") or {}
    name = (meta.get("subjectFamily") or rec.get("subject") or "").strip()
    if name in SUBJECTS:
        return name
    for subject in SUBJECTS:
        # "Mathematics (Standard)", "Hindi Course A", "English Language and Literature"
        if name.startswith(subject + " ") or name.startswith(subject + "("):
            return subject
    return None


def grade_of(rec: dict) -> int | None:
    try:
        g = int(str(rec.get("grade")).strip())
    except ValueError:
        return None
    return g if g in GRADES else None


def mark_bucket(marks) -> str:
    m = int(marks or 0)
    return "5+" if m >= 5 else str(m)


def has_answer_key(rec: dict) -> bool:
    return QuestionBank.has_answer_key(rec)


def key_verified(rec: dict) -> bool | None:
    """Whether the record's key passed the answer-key verifier; None when it
    has no key to check.

    The verifier is the relink's (main-stream Task 15), and it writes no
    per-record verdict field, so the verdict is read the way the pipeline
    reads it: a board record's key is verified when the relink stamped the
    official row it accepted (`bank_merge.has_verified_key`); a CBE, SQP or
    Exemplar record's own key when the rule the relink and the merge share
    accepts it (`question_bank.builder_key_reason`). An explicit
    ``answerScheme.verification`` (``{"passed": bool}`` or ``{"status": ...}``)
    or ``answerScheme.verified`` still wins, for a verdict recorded by hand.
    """
    scheme = rec.get("answerScheme") or {}
    v = scheme.get("verification", scheme.get("verified"))
    if isinstance(v, bool):
        return v
    if isinstance(v, dict):
        if "passed" in v:
            return bool(v["passed"])
        if "status" in v:
            return str(v["status"]).lower() in ("passed", "verified", "ok")
    if not has_answer_key(rec):
        return None
    # Imported here: bank_merge is heavy, and the rest of this module is not.
    from ..assessment.question_bank import builder_key_reason
    from ..corpus.bank_merge import BOARD_SOURCE, has_verified_key

    if rec.get("source") == BOARD_SOURCE:
        return has_verified_key(rec)
    return builder_key_reason(rec) is None


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def _buckets(records: Iterable[dict]) -> dict:
    c = collections.Counter(mark_bucket(r.get("marks")) for r in records)
    return {b: c.get(b, 0) for b in MARK_BUCKETS}


def _paper_verdicts(keyed: list[dict], presets: dict) -> dict:
    out = {}
    for name, preset in presets.items():
        reasons = []
        for label, sec_name, marks, count, difficulties, _choice, choice_count in preset["sections"]:
            at_marks = [r for r in keyed if int(r.get("marks") or 0) == marks]
            fit = [r for r in at_marks if not difficulties or r.get("difficulty") in difficulties]
            need = count + choice_count
            if len(fit) >= need:
                continue
            split = f" ({count} + {choice_count} OR)" if choice_count else ""
            reason = (f"Section {label} ({sec_name}): needs {need} keyed {marks}-mark "
                      f"questions{split}, has {len(fit)}")
            if len(at_marks) > len(fit):
                reason += (f" ({len(at_marks) - len(fit)} more at {marks} marks are outside "
                           f"difficulty {'/'.join(difficulties)})")
            reasons.append(reason)
        out[name] = {"possible": not reasons, "reasons": reasons}
    return out


def _chapter_table(tree: dict, keyed: list[dict]) -> list[dict]:
    by_chapter: dict[str, list[dict]] = collections.defaultdict(list)
    by_topic: dict[str, list[dict]] = collections.defaultdict(list)
    for r in keyed:
        if r.get("taxonomyChapterId"):
            by_chapter[r["taxonomyChapterId"]].append(r)
        for t in r.get("topicIds") or []:
            by_topic[t].append(r)

    def entry(node, recs):
        return {"id": node["id"], "name": node["name"], "questions": len(recs),
                "marks": sum(int(r.get("marks") or 0) for r in recs), "byMarks": _buckets(recs)}

    chapters = []
    for ch in tree.get("chapters", []):
        row = entry(ch, by_chapter.get(ch["id"], []))
        row["number"] = ch.get("number")
        row["book"] = ch.get("book")
        row["topics"] = [entry(tp, by_topic.get(tp["id"], [])) for tp in ch.get("topics", [])]
        chapters.append(row)
    return chapters


def _cell(subject: str, grade: int, recs: list[tuple[str, dict]], tree: dict | None,
          presets: dict, served: str | None) -> dict:
    records = [r for _, r in recs]
    keyed = [r for r in records if has_answer_key(r)]
    n = len(records)
    tagged = {}
    for level, field in LEVEL_FIELDS.items():
        k = sum(1 for r in records if r.get(field))
        tagged[level] = {"n": k, "pct": _pct(k, n)}
    old = sum(1 for r in records if r.get("chapterIds"))
    tagged["oldChapterIds"] = {"n": old, "pct": _pct(old, n)}

    verdicts = [key_verified(r) for r in keyed]
    checked = [v for v in verdicts if v is not None]
    if checked:
        verified = {"state": "verified", "passed": sum(checked), "of": len(checked),
                    "unchecked": len(keyed) - len(checked),
                    "pct": _pct(sum(checked), len(keyed))}
    else:
        verified = {"state": "not verified"}

    cell = {
        "subject": subject, "grade": grade, "questions": n,
        "byBank": dict(collections.Counter(label for label, _ in recs)),
        "bySource": dict(collections.Counter(r.get("source") or "unknown" for r in records)),
        "keyed": len(keyed), "noKey": n - len(keyed), "keyedPct": _pct(len(keyed), n),
        "keyProvenance": dict(collections.Counter(
            (r.get("answerScheme") or {}).get("provenance") or "unknown" for r in keyed)),
        "verified": verified,
        "tagged": tagged,
        "taggerCovers": subject in TAGGER_SUBJECTS,
        # never run through the tagger (no tagMethod): a blank that is not the
        # tagger declining -- the served bank is tagged in the main stream
        "neverTagged": sum(1 for r in records if "tagMethod" not in r),
        "marks": _buckets(keyed),
        "types": dict(collections.Counter(r.get("type") or "unknown" for r in keyed)),
        "tree": tree is not None,
        "untaggedKeyed": sum(1 for r in keyed if not r.get("taxonomyChapterId")),
    }
    chapters = _chapter_table(tree, keyed) if tree else []
    topics = [t for ch in chapters for t in ch["topics"]]
    cell.update(
        chapters=chapters,
        emptyChapters=[c["id"] for c in chapters if not c["questions"]],
        chaptersWithout3Mark=[c["id"] for c in chapters if not c["byMarks"]["3"]],
        chaptersWithout5Mark=[c["id"] for c in chapters if not c["byMarks"]["5+"]],
        emptyTopics=[t["id"] for t in topics if not t["questions"]],
        topicsWithout3Mark=[t["id"] for t in topics if not t["byMarks"]["3"]],
        topicsWithout5Mark=[t["id"] for t in topics if not t["byMarks"]["5+"]],
    )
    papers = {"all": _paper_verdicts(keyed, presets)}
    if served is not None:
        served_keyed = [r for label, r in recs if label == served and has_answer_key(r)]
        cell["servedKeyed"] = len(served_keyed)
        papers["served"] = _paper_verdicts(served_keyed, presets)
    cell["papers"] = papers
    return cell


def build_report(banks: dict[str, list[dict]], trees: dict[tuple[str, int], dict], *,
                 presets: dict | None = None, served: str | None = None) -> dict:
    """The health of ``banks`` ({label: records}) against the taxonomy ``trees``
    ({(subject, grade): tree}). ``served`` names the bank that is served today;
    its paper verdicts are given on their own as well as for all banks together.
    A record id seen in an earlier bank is not counted again."""
    presets = EXAM_PRESETS if presets is None else presets
    per_cell: dict[tuple[str, int], list[tuple[str, dict]]] = collections.defaultdict(list)
    seen: set[str] = set()
    duplicates = 0
    for label, records in banks.items():
        for r in records:
            subject, grade = subject_family(r), grade_of(r)
            if subject is None or grade is None:
                continue
            rid = str(r.get("id"))
            if rid in seen:
                duplicates += 1
                continue
            seen.add(rid)
            per_cell[(subject, grade)].append((label, r))
    cells = [_cell(s, g, per_cell.get((s, g), []), trees.get((s, g)), presets, served)
             for s in SUBJECTS for g in GRADES]
    return {"scope": {"subjects": list(SUBJECTS), "grades": list(GRADES)},
            "served": served, "banks": list(banks), "duplicateIds": duplicates,
            "presets": {k: v.get("name", k) for k, v in presets.items()},
            "cells": cells}


# --------------------------------------------------------------------------- #
# tag accuracy
# --------------------------------------------------------------------------- #

def tagging_accuracy(model: dict, audit_subtopic: dict, audit_chapter_topic: dict) -> dict:
    """Tag precision per level: nested cross-validation on the gold set (from
    ``_tagger_model.json``) and the held-out audits of the tags actually
    written (``_tag_audit*.json``). The audits are the honest figure -- the
    gold set chose the thresholds -- so both are quoted, never one for the
    other. A wrong topic tag is split by whether the question is taught in
    another section of the class's book or in none of it."""
    nested = {lv: dict(v) for lv, v in model["gold"]["nested"]["accepted"].items()}

    def held(m):
        return {"right": m["right"], "tags": m["tags"], "precision": round(m["right"] / m["tags"], 4)}

    measured = audit_chapter_topic["measured"]
    held_out = {"chapter": held(measured["chapter"]), "topic": held(measured["topic"]),
                "subtopic": held(audit_subtopic["measured"])}
    wrong = [it for it in audit_chapter_topic["items"]
             if "topic" in it.get("levels", []) and not it["right"]]
    return {
        "thresholds": model.get("thresholds", {}),
        "goldItems": model["gold"].get("items"),
        "nested": nested,
        "heldOut": held_out,
        "wrongTopics": {
            "otherSection": sum(1 for it in wrong if it.get("topics")),
            "noSection": sum(1 for it in wrong if not it.get("topics")),
            "items": [{"id": it["id"], "note": it.get("note", "")} for it in wrong],
        },
    }


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

def _names(cell: dict) -> dict[str, str]:
    names = {}
    for ch in cell["chapters"]:
        names[ch["id"]] = ch["name"]
        for t in ch["topics"]:
            names[t["id"]] = f"{ch['name']} > {t['name']}"
    return names


def _yes_no(v: dict | None) -> str:
    if v is None:
        return "-"
    return "yes" if v["possible"] else "**no**"


def render_markdown(report: dict) -> str:
    cells = report["cells"]
    presets = report["presets"]
    served = report.get("served")
    out = ["# Question bank health, classes 6-10", ""]
    out.append(f"Generated by `scripts/bank_health.py` on {report.get('builtAt', '?')}. "
               "Do not edit by hand: rerun the script. The same numbers are in "
               "`academicos-data/syllabus/bank_health.json`.")
    out += ["", "## Inputs", "", "| bank | file | records | sha256 |", "|---|---|---:|---|"]
    for inp in report.get("inputs", []):
        out.append(f"| {inp['label']} | `{inp['path']}` | {inp['records']:,} | `{inp['sha256']}` |")
    extra = [inp for inp in report.get("taxonomyInputs", [])]
    if extra:
        out += ["", "Tag accuracy is read from:", ""]
        out += [f"- `{inp['path']}` (sha256 `{inp['sha256']}`)" for inp in extra]
    out += ["", "## How to read this", "",
            "- Scope: classes 6-10 x Mathematics, Science, Social Science, English, Hindi. "
            "Paper variants count under their subject; other subjects and classes are not counted.",
            "- **Keyed** = has an answer key by the served bank's own gate (PRD rule Q1: without "
            "an answer key, no question). Marks, types, chapter and topic counts and the paper "
            "verdicts count keyed questions only. Question counts and tag percentages count every "
            "question.",
            "- **Chapter / topic / subtopic** = the taxonomy tags the topic tagger writes "
            "(NCERT textbook headings). *Old chapterIds* are the earlier, coarser chapter "
            "labels, shown for comparison.",
            "- A **paper is possible** when every section of the preset can be filled at its "
            "exact marks and allowed difficulty, with its OR alternatives, as the paper builder "
            "fills it.",
            f"- Record ids repeated across banks are counted once ({report.get('duplicateIds', 0)} "
            "repeats skipped)."]
    if served:
        out.append(f"- *Served* is the `{served}` bank alone -- what a teacher can use today. "
                   "*All* is every input bank together.")

    out += ["", "## Summary", "",
            "| subject | class | questions | keyed | chapter | topic | subtopic | old chapterIds "
            "| 1m | 2m | 3m | 4m | 5m+ | board paper (served / all) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in cells:
        t, m = c["tagged"], c["marks"]
        board = c["papers"]["all"].get("board")
        sboard = c["papers"].get("served", {}).get("board")
        out.append(
            f"| {c['subject']} | {c['grade']} | {c['questions']:,} | {c['keyedPct']}% "
            f"| {t['chapter']['pct']}% | {t['topic']['pct']}% | {t['subtopic']['pct']}% "
            f"| {t['oldChapterIds']['pct']}% | {m['1']} | {m['2']} | {m['3']} | {m['4']} "
            f"| {m['5+']} | {_yes_no(sboard)} / {_yes_no(board)} |")
    no_q = [f"{c['subject']} {c['grade']}" for c in cells if not c["questions"]]
    if no_q:
        out += ["", f"**No question at all ({len(no_q)}):** " + ", ".join(no_q) + "."]
    never = [f"{c['subject']} {c['grade']} {c['neverTagged']:,}" for c in cells
             if c["taggerCovers"] and c["neverTagged"]]
    if never:
        out += ["", "**Never run through the topic tagger** (no `tagMethod`; their 0% is not "
                "the tagger's verdict): " + ", ".join(never) + " questions."]
    not_tagged = sorted({c["subject"] for c in cells if not c["taggerCovers"]})
    if not_tagged:
        out += ["", f"The topic tagger covers {', '.join(sorted(TAGGER_SUBJECTS))} only, so "
                f"{', '.join(not_tagged)} carry no taxonomy tag whatever their tree."]
    no_tree = [f"{c['subject']} {c['grade']}" for c in cells if not c["tree"]]
    if no_tree:
        out += ["", "**No taxonomy tree** (no chapter list to count against): "
                + ", ".join(no_tree) + "."]

    out += ["", "## Answer keys", "",
            "Keyed questions by where the key came from, and whether the key passed the "
            "answer-key verifier.", "",
            "| subject | class | questions | keyed | no key | key provenance | verifier |",
            "|---|---:|---:|---:|---:|---|---|"]
    for c in cells:
        if not c["questions"]:
            continue
        prov = ", ".join(f"{k} {v:,}" for k, v in sorted(c["keyProvenance"].items())) or "-"
        v = c["verified"]
        ver = ("not verified" if v["state"] == "not verified"
               else f"{v['passed']}/{v['of']} passed ({v['pct']}% of keyed; {v['unchecked']} unchecked)")
        out.append(f"| {c['subject']} | {c['grade']} | {c['questions']:,} | {c['keyed']:,} "
                   f"| {c['noKey']:,} | {prov} | {ver} |")

    tagging = report.get("tagging")
    if tagging:
        out += _tagging_md(tagging)

    out += ["", "## Can a full paper be made?", "",
            "Presets from `academicos.assessment.templates.EXAM_PRESETS`: "
            + ", ".join(f"`{k}` ({v})" for k, v in presets.items()) + "."]
    for c in cells:
        if not c["questions"]:
            continue
        out += ["", f"### {c['subject']} class {c['grade']}", ""]
        for scope in ("served", "all"):
            verdicts = c["papers"].get(scope)
            if verdicts is None:
                continue
            if scope == "served" and not c.get("servedKeyed"):
                out.append("- served: **no** preset -- the served bank has no keyed question "
                           "for this class and subject")
                continue
            for name, v in verdicts.items():
                if v["possible"]:
                    out.append(f"- {scope}, `{name}`: yes")
                else:
                    out.append(f"- {scope}, `{name}`: **no** -- " + "; ".join(v["reasons"]))

    out += ["", "## Chapters and topics", "",
            "Keyed questions and the marks they carry, per textbook chapter of the taxonomy "
            "tree. *Topics with a question* counts the chapter's topics that have at least one."]
    for c in cells:
        if not c["tree"]:
            continue
        names = _names(c)
        never = (f" {c['neverTagged']:,} questions of this class and subject were never run "
                 "through the tagger." if c["taggerCovers"] and c["neverTagged"] else "")
        # Social Science is several books, each numbering its chapters from 1
        multi = len({str(ch.get("book")) for ch in c["chapters"]}) > 1
        book_head, book_rule = ("book | ", "---|") if multi else ("", "")
        out += ["", f"### {c['subject']} class {c['grade']}", "",
                f"{c['keyed']:,} keyed questions; {c['untaggedKeyed']:,} of them carry no "
                "chapter tag and are counted in no chapter below." + never, "",
                f"| {book_head}# | chapter | questions | marks | 1m | 2m | 3m | 4m | 5m+ "
                "| topics with a question |",
                f"|{book_rule}---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for ch in c["chapters"]:
            b = ch["byMarks"]
            with_q = sum(1 for t in ch["topics"] if t["questions"])
            book = f"{ch.get('book')} | " if multi else ""
            out.append(f"| {book}{ch['number'] if ch['number'] is not None else ''} | {ch['name']} "
                       f"| {ch['questions']} | {ch['marks']} | {b['1']} | {b['2']} | {b['3']} "
                       f"| {b['4']} | {b['5+']} | {with_q}/{len(ch['topics'])} |")

        def listing(title, ids):
            if ids:
                out.extend(["", f"**{title} ({len(ids)}):** " + "; ".join(names[i] for i in ids) + "."])

        listing("Chapters with no question", c["emptyChapters"])
        listing("Chapters with no 3-mark question", c["chaptersWithout3Mark"])
        listing("Chapters with no 5-mark question", c["chaptersWithout5Mark"])
        listing("Topics with no question", c["emptyTopics"])
        empty = set(c["emptyTopics"])
        listing("Topics with questions but no 3-mark question",
                [t for t in c["topicsWithout3Mark"] if t not in empty])
        listing("Topics with questions but no 5-mark question",
                [t for t in c["topicsWithout5Mark"] if t not in empty])
    out.append("")
    return "\n".join(out)


def _tagging_md(t: dict) -> list[str]:
    ho, ne = t["heldOut"], t["nested"]
    topic = ho["topic"]
    wrong = t["wrongTopics"]
    n_wrong = wrong["otherSection"] + wrong["noSection"]
    out = ["", "## How right are the tags?", "",
           "Two measurements, both labelled by an AI reviewer (Claude), not by a teacher. "
           "*Nested* is cross-validation on the gold set, with the choice of threshold inside "
           f"each fold ({t.get('goldItems', '?')} gold items). *Held-out* audits random samples "
           "of the tags actually written onto the banks, drawn after the thresholds were fixed "
           "and never used to fit anything -- it is the figure to quote.", "",
           "| level | threshold | nested (gold set) | Wilson 95% lower | held-out audit |",
           "|---|---:|---:|---:|---:|"]
    for lv in ("chapter", "topic", "subtopic"):
        n, h = ne.get(lv, {}), ho[lv]
        out.append(f"| {lv} | {t['thresholds'].get(lv, '-')} | {n.get('right')}/{n.get('tags')} "
                   f"({100 * n.get('precision', 0):.1f}%) | {100 * n.get('wilsonLower95', 0):.1f}% "
                   f"| {h['right']}/{h['tags']} ({100 * h['precision']:.1f}%) |")
    side = "under" if topic["precision"] < 0.9 else "at or over"
    line = (f"Topic tags are about {100 * topic['precision']:.0f}% right on the held-out audit "
            f"({topic['right']}/{topic['tags']}), {side} the 90% the tagger aims for; the nested "
            f"gold-set figure is {ne['topic']['right']}/{ne['topic']['tags']}.")
    if n_wrong:
        line += (f" All {n_wrong} wrong topic tags put the question in a section of its class's "
                 f"book that does not teach it: in {wrong['otherSection']} another section of "
                 f"the same chapter does, and {wrong['noSection']} are taught in no section of "
                 "the class's book at all.")
    out += ["", line, ""]
    out += [f"- `{it['id']}`: {it['note']}" for it in wrong["items"]]
    return out
