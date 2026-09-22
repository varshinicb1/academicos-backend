"""Does this official answer answer this question?

Pure functions: a question record and one candidate answer in, a verdict out.
The index (`assessment.question_bank.AnswerKeyIndex`) ranks every row of a
question's paper series against it (the content join, at the end of this
module) and labels only a row `join_verdict` accepts `cbse_marking_scheme`.

Why it exists
-------------
answer_keys.db holds 15,605 rows under 11,787 (paper code, q_no) keys: 3,117
keys carry answers from more than one source document -- the English- and
Hindi-medium schemes of one paper, two editions of the same scheme, a question
paper parsed as if it were a scheme -- and within one source the question
numbers are often misaligned. The index used to keep whichever row loaded last
and the relink labelled it official unseen. The audit of 2026-09-21 measured
895 of 2,930 served "official" schemes detectably wrong and about 75% of the
subjective ones answering a different question.

A verdict names one reason, checked in this order:

  document-id         the answer is a document identifier, not an answer
  legacy-font         Krutidev/legacy-font text, unreadable as served
  script-mismatch     the answer is in another script than the question
  unreadable-options  an option letter, for a stem whose options cannot be read
  unverifiable-letter a bare option letter: nothing ties it to this question
  option-mismatch     an MCQ key that resolves to none of the stem's options
  content-mismatch    a subjective answer that shares too little with the stem

The content rule was calibrated on 120 hand-judged pairs, saved with their seed
in tests/fixtures/scheme_verify_sample.json. The judgements are an AI
reviewer's, not a teacher's. The rule accepts none of the 76 pairs judged wrong
and 17 of the 32 judged correct; the constants below say where each number
comes from. The same fixture holds 75 judged objective pairs (every objective
key accepted at 960455e, and 20 drawn at random): the option rules accept none
of the 29 judged wrong and 24 of the 32 judged correct.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..assessment.marking import parse_options
from .mcq_shape import (
    correct_letter,
    has_option_labels,
    options_from_parts,
    options_from_stem,
    tokens,
)

REASONS = ("document-id", "legacy-font", "script-mismatch", "unreadable-options",
           "unverifiable-letter", "option-mismatch", "content-mismatch")

# --- calibration (tests/fixtures/scheme_verify_sample.json) ---------------- #
# An answer is judged only on the words it shares with the stem; with fewer
# than three shared, judged-correct and judged-wrong pairs are not separable
# (two shared words is what any two questions of one chapter have in common).
MIN_SHARED_WORDS = 3
# Share of the ANSWER's content words the stem also has. Judged-wrong pairs
# reached 0.50 (another set's case study: same verbs, different company);
# judged-correct pairs accepted on this share sit at 0.56 and above. The
# margin is thin, and the fixture test pins it.
ANSWER_SHARE = 0.55
# Share of the STEM's content words the answer also has: a long answer that
# addresses every term of a short question. Judged-wrong pairs reached 0.57
# (the parallel question of another set); the 5 judged-correct pairs accepted
# on this share alone start at 0.62.
STEM_SHARE = 0.6
# Mathematics shares few words and many numbers. Three of the stem's own
# numbers (two digits or more) in the answer: no judged-wrong pair had more
# than two -- a word limit ("120-150 words") and an LCM question repeated back
# ("576 and 512"). Of the 25 maths pairs, 2 of the 3 judged correct are
# accepted (one on three numbers and one word) and all 20 judged wrong are
# rejected. At least one content word must be shared too: on the served bank
# a mean-and-median question (class intervals 0-10, 10-20, 20-30) took a
# speed-of-a-boat answer that happened to say 10, 20 and 30.
MIN_SHARED_NUMBERS = 3
# An answer whose tokens are 80% runs copied from the stem repeats the
# question; 4 judged-wrong pairs were exactly that, no judged-correct one.
ECHO_SHARE = 0.8
ECHO_RUN = 3

# Words any question or scheme uses, whatever it is about. Sharing them says
# nothing: an assertion-reason verdict shares "assertion", "reason", "true"
# with every assertion-reason stem, and "sin", "cos", "tan" are in every
# trigonometry question -- 1 judged-wrong maths pair matched on those alone.
_STOPWORDS = frozenset("""
the and for are was were this that with from have has had not but can will shall would
should could may might been being into onto than then them they their there these those
which what when where who whom whose why how its his her our your you any all each some
such only also very more most other about above below after before between both during
over under again further once here same own too just because while until upon per via
given following find show write state explain give name mention describe discuss define
two three four five one marks mark answer answers question questions correct option
options value point points award awarded either reason reasons assertion true false
explanation statement attempt words word note ans sol candidates candidate section part
parts extract passage read based information using use used case study hence thus
therefore let also
sin cos tan sec cosec cot log
का के की में है हैं और से को पर एक यह वह इस उस लिए द्वारा तथा या भी नहीं होता होती होते
किया करते कीजिए लिखिए क्या किस कौन
""".split())

# --- document ids ---------------------------------------------------------- #
# "MS_X_Mathematics_041_30/2/1_2022-23", "XII_042_Physics_MS_55_1_1,2,3.pdf":
# an underscore-joined run with the scheme's code and year, or a filename.
_DOC_ID = re.compile(r"\b(?:MS_)?(?:X|XII)_[\w/,.\-]*\d[\w/,.\-]*|\S+(?i:\.pdf)\b")

# --- legacy font ----------------------------------------------------------- #
# Krutidev maps Devanagari glyphs onto Latin code points, so Hindi typed in it
# extracts as "fdUgha rhu dkj.kksa dk o.kZu dhft,". Its commonest words are
# fixed strings. Across the served bank's 3,286 stems and their 4,176
# candidates, >= 3 of them at >= 4% of tokens flags 127 Krutidev answers and
# no English text; one English stem has 2 hits in 238 tokens (0.8%).
_KRUTIDEV = frozenset("""
gS gSa esa dk ds dh dks vkSj rFkk ls fd ;g og Hkh ugha fy, tks bl mUgsa gks x;k fn;k dj
djsa djrs djus gksrk gksrh gksus ,d vFkok iz'u mÙkj
""".split())
_KRUTIDEV_HITS = 3
_KRUTIDEV_SHARE = 0.04

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CAPITAL_LABEL = re.compile(r"\(([A-D])\)")
_COMPANY = re.compile(r"\b([A-Z][A-Za-z]+)\s+(?:Ltd|Limited)\b")
_PARTNERS = re.compile(
    r"((?:[A-Z][a-z]+\s*,\s*)*[A-Z][a-z]+\s+and\s+[A-Z][a-z]+)\s+(?:were|are)\s+partners")
_ASSERTION_REASON = re.compile(r"\bAssertion\s*\(A\).*\bReason\s*\(R\)", re.I | re.S)
# "(B) / ", "Ans. (c)", "C." -- the letter a key opens with -- and the mark
# note a scheme closes one with ("1 mark", "1 Mark", "½").
_KEY_LABEL = re.compile(r"^\s*(?:ans\.?\s*)?\(?\s*[A-Da-d]\s*\)?\s*(?:[/.):\-]\s*)*(?=\s|$)", re.I)
_MARK_NOTE = re.compile(r"(?:\s*(?:\d+|½)\s*marks?\b)+\s*$", re.I)


@dataclass(frozen=True)
class Verdict:
    """`reason` is None when the answer is accepted. `score` ranks accepted
    candidates against each other; it means nothing on a rejection. `option`
    is the letter an accepted MCQ key resolved to, whether it was keyed by
    letter or by the option's words."""

    reason: str | None = None
    score: float = 0.0
    option: str | None = None

    @property
    def accepted(self) -> bool:
        return self.reason is None


# --------------------------------------------------------------------------- #
# script
# --------------------------------------------------------------------------- #

def _garble():
    """bank_merge's garble detector and its script table, imported late.

    One detector, not two: the composer drops CBE/SQP rows with it and the
    verifier rejects answers with it. The import is deferred because
    bank_merge imports assessment.question_bank, which imports this module.
    """
    from .bank_merge import _ARTS, _GARBLE_RUNS, _SUBJECT_SCRIPTS, _vowel_sign_runs
    return _ARTS, _GARBLE_RUNS, _SUBJECT_SCRIPTS, _vowel_sign_runs


def _script_of(ch: str) -> str:
    return unicodedata.name(ch, "").split(" ", 1)[0]


def dominant_script(text: str, *, minimum: int = 4) -> str | None:
    """LATIN, DEVANAGARI, ... for the script most letters are in.

    Only scripts a paper is written in count: Greek and the mathematical
    alphanumerics (θ, 𝑥) appear in every script's maths. Fewer than `minimum`
    letters is no evidence either way.
    """
    _, _, scripts, _ = _garble()
    writing = {"LATIN"}.union(*scripts.values())
    counts: dict[str, int] = {}
    for ch in text or "":
        if ch.isalpha():
            s = _script_of(ch)
            if s in writing:
                counts[s] = counts.get(s, 0) + 1
    if sum(counts.values()) < minimum:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def medium_of_text(text: str) -> str:
    """'hi' for Devanagari text, 'en' for Latin, '' when it cannot tell."""
    script = dominant_script(text)
    return {"DEVANAGARI": "hi", "LATIN": "en"}.get(script or "", "")


def question_medium(record: dict) -> str:
    return medium_of_text(str(record.get("stem") or "")) or "en"


def _expected_script(record: dict) -> str | None:
    """The script an answer to this question is written in.

    A language paper's own script (Hindi and Sanskrit are Devanagari on both
    sides whatever the stem's instructions are in); otherwise the stem's.
    """
    arts, _, scripts, _ = _garble()
    subject = str(record.get("subject") or "").strip()
    if not arts.search(subject):
        first = re.split(r"[\s(]", subject.lower(), maxsplit=1)[0]
        own = scripts.get(first)
        if own is not None:
            return sorted(own)[0] if len(own) == 1 else None
    return dominant_script(str(record.get("stem") or ""))


def is_legacy_font(text: str) -> bool:
    _, runs_needed, _, vowel_sign_runs = _garble()
    if vowel_sign_runs(text) >= runs_needed:
        return True
    toks = [t.strip(".,()[]") for t in (text or "").split()]
    hits = sum(t in _KRUTIDEV for t in toks)
    return hits >= _KRUTIDEV_HITS and hits / max(len(toks), 1) >= _KRUTIDEV_SHARE


def strip_document_ids(text: str) -> str:
    return " ".join(_DOC_ID.sub(" ", text or "").split())


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #

def _words(text: str) -> list[str]:
    """Runs of letters and combining marks, casefolded.

    Letters AND marks, because a Devanagari word without its vowel signs is
    not the word (\\w drops them). Digits split a word: "6V" is "v".
    """
    out, cur = [], []
    for ch in unicodedata.normalize("NFC", text or "").casefold():
        if unicodedata.category(ch)[0] in "LM":
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def content_words(text: str) -> set[str]:
    """Words of 3+ characters that are not generic, cut to 6 characters so
    "reflect" and "reflection" meet."""
    return {w[:6] for w in _words(text) if len(w) >= 3 and w not in _STOPWORDS}


def numbers(text: str) -> set[str]:
    """Numbers of two significant characters or more, commas dropped:
    "30,00,000" and "3000000" meet; single digits are in every question."""
    out = set()
    for m in _NUMBER.findall(unicodedata.normalize("NFKC", text or "")):
        v = m.replace(",", "").rstrip(".")
        if len(v.replace(".", "")) >= 2:
            out.add(v)
    return out


def _echo_share(stem: str, answer: str) -> float:
    """Share of the answer's tokens that sit in runs copied from the stem."""
    a = [t for t in re.findall(r"\w+", answer.casefold())]
    s = [t for t in re.findall(r"\w+", stem.casefold())]
    if not a:
        return 0.0
    blocks = SequenceMatcher(None, a, s, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= ECHO_RUN) / len(a)


def case_names(text: str) -> set[str]:
    """The firms and partners a case study names: "Saket and Suveni were
    partners", "Shivalik Ltd.". Lower-cased."""
    names = {m.group(1).casefold() for m in _COMPANY.finditer(text or "")}
    for m in _PARTNERS.finditer(text or ""):
        names |= {n.casefold() for n in re.findall(r"[A-Z][a-z]+", m.group(1))}
    return names


def _content_verdict(stem: str, answer: str) -> Verdict:
    if _echo_share(stem, answer) >= ECHO_SHARE:
        return Verdict("content-mismatch")
    named = case_names(answer)
    if named and not any(n in stem.casefold() for n in named):
        # Another set's case study: the same verbs and ledger words, other
        # people. 67/4/2 Q26 asks about Bhumi and Chavi; its candidate
        # answers for "Saket and Suveni were partners" and shares enough
        # accounting words to pass the overlap rule.
        return Verdict("content-mismatch")
    a, s = content_words(answer), content_words(stem)
    shared = len(a & s)
    answer_share = shared / len(a) if a else 0.0
    stem_share = shared / len(s) if s else 0.0
    if shared >= MIN_SHARED_WORDS and (answer_share >= ANSWER_SHARE or stem_share >= STEM_SHARE):
        return Verdict(score=max(answer_share, stem_share))
    shared_numbers = len(numbers(answer) & numbers(stem))
    if shared_numbers >= MIN_SHARED_NUMBERS and shared:
        return Verdict(score=min(1.0, shared_numbers / max(len(numbers(stem)), 1)))
    return Verdict("content-mismatch")


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #

def question_options(record: dict) -> dict[str, str] | None:
    """The options an MCQ prints, or None when this is not one.

    `options_from_stem` also reads the (a)-(d) sub-parts of a 5-mark question
    as options, and a subjective answer judged against them would fail for the
    wrong reason, so above 1 mark only all four CAPITAL labels (A)-(D) make an
    MCQ: 67/1/1 Q29 is filed at 3 marks with a 1-mark MCQ fused into its stem,
    and its candidate -- the question's own opening plus the next part's --
    shares enough words to pass as a subjective answer. Capital labels that
    `options_from_stem` refuses ("(a) Analysis ... (A) Labour Unions (B) ...",
    where a part label comes first) are read by `parse_options`.
    """
    parts = record.get("parts") or []
    if parts:
        # CBE/SQP items carry their options as parts, not in the stem.
        from_parts = options_from_parts(parts)
        if from_parts:
            return from_parts
    stem = str(record.get("stem") or "")
    marks = record.get("marks")
    one_mark = (record.get("type") == "mcq" or not isinstance(marks, (int, float))
                or marks <= 1)
    if one_mark:
        opts = options_from_stem(stem)
        if not opts and has_option_labels(stem):
            opts = parse_options(stem) or None
        return opts
    if set("ABCD") <= set(_CAPITAL_LABEL.findall(stem)):
        return parse_options(stem) or None
    return None


def _core(option: str) -> list[str]:
    """An option's tokens without the page footer extraction glued on:
    "Public relations 11-" is option D of 66/1/1 Q16.

    Only a worded option has a footer to cut. A numeric option IS its
    numbers: cutting them left "2550" (430/5/2 Q17, option B) with no core, so
    no key could ever name it in words.
    """
    toks = tokens(option)
    if not any(re.search(r"[^\W\d_]", t) for t in toks):
        return toks
    while toks and re.fullmatch(r"\d+|-", toks[-1]):
        toks.pop()
    return toks


def _named_options(text: str, options: dict[str, str], *,
                   labelled_words: bool = False) -> list[str]:
    """The options whose words open `text`, once any "(B) /" label is cut.

    An option with no letter or digit left names nothing: 30/3/2 Q6 extracts
    its four options as "-, -" and "-,", and "– 3x2 – x – 6" (another
    question's polynomial) opens with a dash.

    `labelled_words` also reads `text` with its opening kept, for an option
    that opens with a word the label rule cuts: the article in "a rational
    number" (cbe:q:Maths9SM2) and "A shoe box" (exemplar:q:6:science:11:-:6),
    the statement letter in "B and G" or "A is true but R is false". Cut, they
    named no option: 45 of the 67 served CBE, SQP and Exemplar keys the relink
    removed at 8e92ed9 (`question_bank.builder_key_reason` has the rest).
    It is for a key from the item's own builder only (`trust_letter`); a
    board row that now named an option this way would be a pick no one has
    judged."""
    body = tokens(_KEY_LABEL.sub("", text))
    bodies = [body, tokens(text)] if labelled_words else [body]
    return [k for k, v in options.items()
            if (core := _core(v)) and any(re.search(r"[^\W_]", t) for t in core)
            and any(b[:len(core)] == core for b in bodies)]


def _bare_letter(answer: str) -> bool:
    """True when a key is its option letter and nothing else ("", "(b)",
    "B 1 mark")."""
    return not re.search(r"\w", _MARK_NOTE.sub("", _KEY_LABEL.sub("", answer)).strip())


def _option_verdict(options: dict[str, str], answer: str, option: str | None, *,
                    trust_letter: bool) -> Verdict:
    """A key resolves to an option by its words -- and when it also has a
    letter, the two must agree.

    A letter alone is always "among the options", so it is only as good as
    the row it came from, and rows misalign: editions number questions
    differently. Of the served bank's 200 MCQ candidates with a letter, 26
    are the bare letter, 18 carry words naming the letter's option and 156
    words naming none: "Noise For Visually Impaired Candidates" keyed C for
    "doing the task correctly and with minimum cost" (66/1/1 Q4), "59" keyed A
    among four coordinate pairs. Words that name no option mean the row
    answers another question. A bare letter is rejected as
    unverifiable-letter: of the 24 the verifier used to accept, 12 were judged
    wrong and 8 correct (fixture stratum objective-accepted) -- 430/5/2 Q17
    keys 2550 for the sum of the first 100 even numbers, 55/1/1 Q1 a non-zero
    field at the centre of a charged ring. Keys whose words name their option
    were right in every case judged.

    `trust_letter` is for a letter from the item's own builder (CBE/SQP):
    there is no row to misalign, so the letter alone stands.
    """
    bare = _bare_letter(answer)
    if option:
        letter = str(option).strip().upper()
        if letter not in options:
            return Verdict("option-mismatch")
        if bare:
            return Verdict(score=1.0, option=letter) if trust_letter else Verdict(
                "unverifiable-letter")
        if letter in _named_options(answer, options, labelled_words=trust_letter):
            return Verdict(score=1.0, option=letter)
        return Verdict("option-mismatch")
    named = _named_options(answer, options, labelled_words=trust_letter)
    letter = correct_letter([answer], options) or (named[0] if len(named) == 1 else None)
    if not letter:
        return Verdict("option-mismatch")
    if bare and not trust_letter:
        return Verdict("unverifiable-letter")
    return Verdict(score=1.0, option=letter)


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #

def verify(record: dict, answer: str, option: str | None = None, *,
           judge_content: bool = True, options: dict[str, str] | None = None) -> Verdict:
    """Accept `answer` (and its option letter, if the key has one) as the
    answer to `record`, or reject it with one of `REASONS`.

    `record` is the wire-shaped question: `stem`, `subject`, `marks`, `type`,
    `parts`. `judge_content=False` skips the content-mismatch rule, for a
    scheme its builder attached from the same item (CBE/SQP): the rule
    measures how board marking-scheme rows misalign with their questions, the
    calibration sample holds no CBE/SQP pair, and their one-word and numeric
    answers would all fail it. For the same reason such a scheme's bare
    option letter is trusted, where a board row's is unverifiable-letter.
    Every other check still applies.

    `options`, when given, are the options the item's own builder resolved
    (`question_bank.builder_key_reason`), read instead of the record: {} for
    an item its builder did not make an MCQ.
    """
    answer = (answer or "").strip()
    if options is None:
        options = question_options(record)
    if not options and (option or _ASSERTION_REASON.search(str(record.get("stem") or ""))):
        # A letter for a stem with no readable options: nothing ties it to
        # this question. 2 of the 3 checked by hand keyed the wrong verdict
        # of an assertion-reason item. An assertion-reason item's answer is a
        # verdict among four fixed alternatives, so its words are the same
        # for every such item and prove nothing either: "(B)/ The reaction of
        # a reactive metal with dilute acid" shares enough words with 31/3/1
        # Q17 (hydrogen and nitric acid) to pass as a subjective answer.
        return Verdict("unreadable-options")
    if options and option:
        return _option_verdict(options, answer, option, trust_letter=not judge_content)

    if answer and not strip_document_ids(answer):
        return Verdict("document-id")
    answer = strip_document_ids(answer)
    if is_legacy_font(answer):
        return Verdict("legacy-font")
    expected = _expected_script(record)
    got = dominant_script(answer)
    if expected and got and got != expected:
        return Verdict("script-mismatch")
    if options:
        return _option_verdict(options, answer, None, trust_letter=not judge_content)
    if not judge_content:
        return Verdict(score=0.0)
    return _content_verdict(str(record.get("stem") or ""), answer)


# --------------------------------------------------------------------------- #
# the content join
# --------------------------------------------------------------------------- #
#
# One CBSE marking scheme covers sets 1-3 of a paper series and is keyed under
# one code, while sets 2 and 3 print the same questions in another order. A
# set-2 question joined by (code, q_no) gets another question's answer. The
# index (`AnswerKeyIndex.choose`) therefore ranks every row of the scheme
# FAMILY -- all sets of the series, in the question's medium, whatever their
# question number -- against the stem, and `join_verdict` decides the best.
#
# Ranking alone is not enough. On the served bank, of 50 content-best picks
# the verifier accepted, drawn at random with seed 20260922 and judged by
# reading (an AI reviewer's judgements, not a teacher's), 20 answer this
# question and 21 another: a family holds every chapter's answers, and an
# answer to the neighbouring question on the same topic shares as many words
# as the right one. What separated them, in that sample and the census of
# the 89 picks the rule below accepts (fixture strata join-random and
# join-accepted; the census counts follow each rule):
#
#   * an objective key whose words ARE one option, with no key of the series
#     naming another. "(B) / Human Chorionic Gonadotropin", "30°". A key that
#     only opens with an option's words is a sentence from another answer:
#     "Sangama dynasty, the first dynasty, exercised control till 1485"
#     (61/3/2 Q54) named option (a) of "Krishna Deva Raya belonged to which
#     dynasty" (Tuluva). Census: 37 accepted, 33 correct, 1 wrong (30/1/1
#     Q3 took 30/1/1 Q8's "x-axis"), 3 cannot-tell.
#   * a subjective pick that sits at the question's own (code, q_no), or at
#     the number of the same question served from another set -- two
#     independent signals agreeing. Census: 46, 41 correct, 2 wrong, 3
#     cannot-tell. Both wrong ones are rows every set files under the
#     question's number: 67/4/1 and 67/4/3 both print Q26's OR part
#     (debentures) where the question asks for equity shares, and 64/4/1
#     Q29 takes another set's Visually Impaired alternative;
#   * or one that opens by restating most of the question and then answers
#     it: "(a) Explain any five challenges faced by political parties in
#     India (i) Lack of internal democracy ..." for set 1's "Analyse any three
#     challenges faced by political parties in India" (32/5/2 Q33 for 32/5/1
#     Q27). A restatement of instructions alone ("Read the given passage
#     carefully and answer", "On the given political outline map of India")
#     covers too little of a long stem to count, and a restatement with no
#     answer after it ("Mahabharata is considered as a dynamic text. Explain
#     the") is no answer. Census: 6, all correct.

# The best candidate must beat the runner-up -- the best-scoring candidate
# that does not agree with it -- by this share of its score. Only rows the
# gates above would accept compete (see `AnswerKeyIndex.choose`), and on the
# served bank two such rows rarely meet: of the 89 subjective and objective
# picks accepted at f34680f, 1 has a subjective runner-up at all (430/5/2 Q28, 0.42 of
# its pick's score). So the margin is a guard for the day two evidenced rows
# disagree, set where one row's score is plainly not the other's; it is not
# what the precision rests on -- the gates are, and the objective rule that
# any dissenting key is a tie. See test_the_combined_fixture_is_precise.
JOIN_MARGIN = 0.1
# Two subjective candidates agree -- two editions of one answer, which split
# and trim value points differently -- when this share of their join terms is
# common. A judgement, not a calibration: the fixture holds no pair that
# turns on it.
AGREE_SHARE = 0.5
# Two served stems are the same question printed in two sets. Also a
# judgement; set above AGREE_SHARE because a stem is the whole question and
# sets reword little (the 2 wrong anchored picks above are not siblings).
SIBLING_SHARE = 0.6
# Restatement: a run of at least RESTATE_RUN tokens copied from the stem,
# starting within the pick's first RESTATE_HEAD tokens, covering at least
# RESTATE_COVER of the stem, with at least RESTATE_REST tokens of answer after.
RESTATE_RUN = 6
RESTATE_HEAD = 4
RESTATE_COVER = 0.5
RESTATE_REST = 5

NEAR_TIE = "near-tie"
CONFLICTING_KEYS = "conflicting-keys"
UNANCHORED = "unanchored"
NOT_AN_OPTION = "not-an-option"
# A row under the question's own number the verifier accepts on its own, which
# the join did not pick: another row answers the question better, or none
# could be told apart from its runner-up.
OUTRANKED = "outranked"
# The pick is filed under the question's own paper code at another number.
# One set's scheme numbers as that set's paper does, so such a row answers
# another question of the same paper: of the 89 picks the census accepted, 4
# were such rows, and 3 of them wrong or a fused stem's (30/1/1 Q3 took Q8's
# "x-axis"; 64/4/1 Q29 Q22's Visually Impaired alternative; 430/3/3 Q7 Q9's
# "6.4 cm"). Taken only when anchored: the 4th, 66/1/1 Q10, took Q3's "Both
# the Statements are true", and its own row names the same option.
OTHER_NUMBER = "other-number"
JOIN_REASONS = (NEAR_TIE, CONFLICTING_KEYS, UNANCHORED, NOT_AN_OPTION, OUTRANKED,
                OTHER_NUMBER)

# "Q.", "Q. No. 12", "(a)", "(iii)", "2." -- the labels a scheme row opens with
# before it restates its question.
_ROW_LABEL = re.compile(
    r"^\s*(?:q\.?\s*(?:no\.?)?\s*\d*\s*[.:)]?\s*)?(?:\(?[a-z0-9ivx]{1,4}\)\s*|[a-z0-9]{1,3}\.\s+)*",
    re.I)


# "Sales promotion 1 mark 7 Q. Beenu had a bookstore ..." (66_1_3..pdf Q6): a
# row runs on past its mark note into the next question, whose words then
# rank the row for THAT question. The join reads a row up to the cut.
_NEXT_QUESTION = re.compile(r"\s(?:\d+|½)\s*marks?\s+(?:\d+\s+)*Q\b.*$", re.I | re.S)


def row_text(text: str) -> str:
    """A scheme row's text without the next question it ran on into."""
    return _NEXT_QUESTION.sub("", text or "").strip()


def join_terms(text: str) -> frozenset[str]:
    """What the join compares: content words (as `content_words`) and numbers
    of two significant characters or more, marked so "12" never meets a word."""
    return frozenset(content_words(text)) | {"#" + n for n in numbers(text)}


def idf_weights(pool: list[frozenset[str]]) -> dict[str, float]:
    """Inverse document frequency fitted on the candidate pool itself: a word
    every answer of the family uses ("figure", "diagram") weighs little."""
    n = len(pool)
    df: dict[str, int] = {}
    for terms in pool:
        for t in terms:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (k + 1)) + 1.0 for t, k in df.items()}


def join_score(query: frozenset[str], doc: frozenset[str], idf: dict[str, float],
               doc_norm: float) -> float:
    """Cosine of binary TF-IDF vectors. A query word no candidate uses takes
    the rarest weight; it lowers every candidate's score alike."""
    shared = query & doc
    if not shared or not doc_norm:
        return 0.0
    unseen = max(idf.values(), default=1.0)
    q_norm = math.sqrt(sum(idf.get(t, unseen) ** 2 for t in query))
    return sum(idf[t] ** 2 for t in shared) / (q_norm * doc_norm)


def norm_of(doc: frozenset[str], idf: dict[str, float]) -> float:
    return math.sqrt(sum(idf[t] ** 2 for t in doc))


def share(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# A key read as an option ends at its mark note, whatever follows: "Efficiency
# 1 mark 5 Q. 'Dovex' was a large company ..." (66_1_1..pdf Q4).
_RUNS_ON = re.compile(r"\s(?:\d+|½)\s*marks?\b.*$", re.I | re.S)


def outright_option(answer: str, options: dict[str, str]) -> str | None:
    """The option a key IS: its words, once the "(B) /" label, the mark note
    and whatever runs on after it are cut, are exactly one option's. None for
    a key that names no option, several, or one followed by anything else."""
    body = _MARK_NOTE.sub("", _RUNS_ON.sub("", _KEY_LABEL.sub("", answer or ""))).strip()
    words = [t for t in tokens(body) if t != "/"]
    hits = [k for k, v in options.items()
            if (core := _core(v)) and any(re.search(r"[^\W_]", t) for t in core)
            and words == core]
    return hits[0] if len(hits) == 1 else None


def restates(stem: str, answer: str) -> bool:
    """True when `answer` opens by restating most of `stem` and then answers."""
    pick = re.findall(r"\w+", _ROW_LABEL.sub("", answer or "").casefold())
    words = re.findall(r"\w+", (stem or "").casefold())
    if not pick or not words:
        return False
    blocks = SequenceMatcher(None, pick[:40], words, autojunk=False).get_matching_blocks()
    for b in blocks:
        if (b.a <= RESTATE_HEAD and b.size >= RESTATE_RUN
                and b.size >= RESTATE_COVER * len(words)
                and len(pick) - (b.a + b.size) >= RESTATE_REST):
            return True
    return False


def join_verdict(record: dict, answer: str, *, ratio: float, anchored: bool,
                 other_number: bool = False) -> Verdict:
    """Accept the content join's best candidate for `record`, or say why not.

    `ratio` is the runner-up's score over the best's (0 when nothing
    disagrees with the best); `anchored` is True when the best sits at the
    question's own (code, q_no) or at that of the same question served from
    another set. `other_number` is True when the best is filed under the
    question's own code at another number (OTHER_NUMBER), refused unless
    anchored. All three come from the pool, which only the index sees.

    An objective key is judged on its words alone: a letter from another
    set's row may name another option, and a bare letter is not evidence
    (43da0f4). The letter returned is the option the words are.
    """
    verdict = verify(record, answer, option=None)
    if not verdict.accepted:
        return verdict
    options = question_options(record)
    if options:
        letter = outright_option(answer, options)
        if letter is None:
            return Verdict(NOT_AN_OPTION)
        if other_number and not anchored:
            return Verdict(OTHER_NUMBER)
        if ratio > 1.0 - JOIN_MARGIN:
            return Verdict(CONFLICTING_KEYS)
        return Verdict(score=verdict.score, option=letter)
    if other_number and not anchored:
        return Verdict(OTHER_NUMBER)
    if not anchored and not restates(str(record.get("stem") or ""), answer):
        return Verdict(UNANCHORED)
    if ratio > 1.0 - JOIN_MARGIN:
        return Verdict(NEAR_TIE)
    return verdict
