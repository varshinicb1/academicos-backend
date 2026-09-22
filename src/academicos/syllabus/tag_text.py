"""The one tokenizer the topic tagger and its textbook index share.

The textbook index (``taxonomy/_node_text.json``) stores term counts, not
text, so the terms it was built with and the terms a question is cut into must
come from the same function. `TOKENIZER_VERSION` is written into the index and
checked on load: an index built by a different tokenizer is refused rather
than silently matched against the wrong terms.

Deliberately plain: lower-case words, a stop list, and a light suffix strip.
No model, no dictionary download -- the tagger must run offline and give the
same tags on every machine.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD = re.compile(r"[a-z]+")

# Function words, plus the words every textbook page and every question
# carries whatever its subject ("activity", "figure", "explain", "following").
# Left in, they make every chapter look alike to every question. "make" is
# here because Exemplar stems open "Fill in the blanks to make the statements
# true": left in, it matched "25.4 x 1000 = ___" to the class 7 topic "Making
# an Adjustment".
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can cannot could did do does doing done down
during each either else etc even ever every few for from further get gets got had has
have having he her here hers him his how however i if in into is it its itself just
let lets like may me might more most much must my neither no nor not now of off on
once one only or other our ours out over own per same shall she should so some such
than that the their them then there these they this those through thus to too two
under until up upon us very was we were what when where whether which while who whom
whose why will with would yes yet you your
activity activities answer answers chapter choose class correct explain example
examples exercise exercises fig figure figures fill find following give given
incorrect kindly let mark marks match name note option options page question
questions reason reasons select show state statement statements true false write
think discuss look observe see tell ask shown table column blank blanks
make many use using used way among want
""".split())

# Changing either the stop list or the stemming rules changes every term.
TOKENIZER_VERSION = hashlib.sha256(
    (" ".join(sorted(STOPWORDS)) + "|stem-v1").encode("utf-8")).hexdigest()[:12]


def stem(word: str) -> str:
    """Fold the plural and verb endings that split one idea into two terms
    ("fractions"/"fraction", "reflected"/"reflection" stay apart; that is fine).
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("sses"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    return word


def tokens(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    out = []
    for w in _WORD.findall(folded.lower()):
        if len(w) < 3 or w in STOPWORDS:
            continue
        s = stem(w)
        if s not in STOPWORDS:
            out.append(s)
    return out
