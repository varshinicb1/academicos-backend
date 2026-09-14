"""Embedding-based chapter classifier: replaces the earlier LLM-batch-JSON
approach (llm_chapter_tagger.py, retired this session) for every subject
beyond chapters.py's Science-only keyword tagger.

Why embeddings instead of asking an LLM to emit JSON: a live validation run
against a local qwen2.5:7b (Ollama) tagged Mathematics (7 candidate chapters)
cleanly, but Social Science (21 candidates -- three times the list, one name
containing a comma) failed roughly 40-50% of batches with malformed JSON
("Expecting property name enclosed in double quotes"). That failure mode is
inherent to generation: the model has to produce syntactically valid
structured text, and a longer prompt/candidate list makes that harder. A
sentence-embedding cosine-similarity match has no generation step at all --
it's pure vector math over a fixed candidate set, so there is nothing to
parse and nothing that can come back malformed. It is also fully local (no
API key, no per-call cost) and still genuinely GPU-accelerable, unlike the
earlier hosted-LLM path.

Ground truth chapters: same source as before, syllabus/cbse_syllabus.py's
SyllabusDocument (hand-verified CBSE curriculum data) -- this module only
changes how a question's text gets matched against that list, not where the
list comes from.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Below this cosine similarity, decline to tag rather than force a weak
# match -- chapters.py's keyword tagger has the same "no confident match ->
# None" behavior, this preserves it rather than always picking *some*
# chapter just because it scored highest of a bad lot.
_MIN_SIMILARITY = 0.30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chapter_tags (
  text_hash  TEXT NOT NULL,
  subject    TEXT NOT NULL,
  grade      TEXT NOT NULL,
  chapter_id TEXT,
  confidence REAL NOT NULL,
  tagged_at  TEXT NOT NULL,
  PRIMARY KEY (text_hash, subject, grade)
);
"""


class ChapterTagCache:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def get_many(self, subject: str, grade: str,
                 text_hashes: list[str]) -> dict[str, tuple[Optional[str], float]]:
        if not text_hashes:
            return {}
        placeholders = ",".join("?" * len(text_hashes))
        rows = self.conn.execute(
            f"SELECT text_hash, chapter_id, confidence FROM chapter_tags "
            f"WHERE subject=? AND grade=? AND text_hash IN ({placeholders})",
            (subject, grade, *text_hashes),
        ).fetchall()
        return {r["text_hash"]: (r["chapter_id"], r["confidence"]) for r in rows}

    def put_many(self, subject: str, grade: str,
                 tags: dict[str, tuple[Optional[str], float]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany(
            "INSERT INTO chapter_tags (text_hash, subject, grade, chapter_id, confidence, tagged_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(text_hash, subject, grade) DO UPDATE SET "
            "chapter_id=excluded.chapter_id, confidence=excluded.confidence, tagged_at=excluded.tagged_at",
            [(h, subject, grade, cid, conf, now) for h, (cid, conf) in tags.items()],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# A bare CBSE unit *name* ("Geometry", "Mensuration") is often too short for
# the embedding model to tell semantically-adjacent units apart -- a live run
# showed "Geometry" swallowing nearly all of Mensuration (0 questions tagged)
# and a chunk of Coordinate Geometry, because circles/areas/angles vocabulary
# overlaps across all three. Enriching the anchor text with the topic
# keywords a real question in that unit would actually use gives the model
# something to differentiate on. Mathematics/X is the only subject with a
# confirmed problem this fixes (Social Science's 21 real chapter names, e.g.
# "Federalism", "Consumer Rights", already carry enough distinct vocabulary
# on their own); keyed by (subject, grade_int) so other subjects are free to
# get their own entry later without this staying Math-only by accident.
_ANCHOR_ENRICHMENT: dict[tuple[str, int], dict[str, str]] = {
    ("Mathematics", 10): {
        "number-systems": "real numbers, rational and irrational numbers, Euclid's division lemma, HCF LCM, decimal expansion",
        "algebra": "polynomials, zeroes of a polynomial, linear equations in two variables, quadratic equations, arithmetic progressions, nth term, sum of AP",
        "coordinate-geometry": "distance formula, section formula, area of a triangle using coordinates, points on the x-axis and y-axis",
        "geometry": "similar triangles, Pythagoras theorem, tangent to a circle, chord, congruence, angle properties of triangles and circles",
        "trigonometry": "trigonometric ratios, sin cos tan, trigonometric identities, heights and distances, angle of elevation and depression",
        "mensuration": "area of a circle, sector and segment, surface area and volume of a cone cylinder sphere and cuboid, perimeter",
        "statistics-and-probability": "mean median mode of grouped data, cumulative frequency, probability of an event, sample space",
    },
    # SST's 21 real chapter names already carry more distinguishing
    # vocabulary than Math's bare unit names did, and a live run confirmed
    # decent results with names alone -- this enrichment is a precision pass
    # on top of that, not a fix for a confirmed failure the way Math's was.
    # Aimed mainly at the pairs within the same unit that share the most
    # surface vocabulary (Power-sharing/Federalism; Political Parties/
    # Outcomes of Democracy; Development/Sectors of the Indian Economy/
    # Globalisation and the Indian Economy).
    ("Social Science", 10): {
        "rise-of-nationalism-europe": "unification of Germany and Italy, Treaty of Versailles, Napoleon, nation-state in 19th century Europe",
        "nationalism-india": "Non-Cooperation Movement, Civil Disobedience Movement, Rowlatt Act, Gandhi, Salt March, Indian National Congress",
        "making-global-world": "trade routes, indentured labour, Bretton Woods, silk route, colonialism and world economy before World War I",
        "age-of-industrialisation": "factories, industrial revolution in Britain and India, spinning jenny, handicrafts and mills",
        "print-culture-modern-world": "printing press, Gutenberg, newspapers and books, censorship, spread of literacy",
        "resources-development": "land use, soil types, resource planning, land degradation, sustainable development",
        "forest-wildlife-resources": "biodiversity, flora and fauna, conservation, Project Tiger, deforestation",
        "water-resources": "dams, multipurpose river projects, water scarcity, rainwater harvesting",
        "agriculture": "cropping pattern, kharif and rabi crops, irrigation, green revolution, food security",
        "minerals-energy-resources": "iron ore, coal, petroleum, mineral conservation, non-conventional energy sources",
        "manufacturing-industries": "iron and steel industry, textile industry, industrial pollution, agro-based industries",
        "lifelines-national-economy": "roads, railways, waterways, trade and tourism, means of transportation and communication",
        "power-sharing": "power sharing among social groups, community government, horizontal and vertical division of power in Belgium and Sri Lanka",
        "federalism": "federal government, union list state list concurrent list, local self-government, decentralisation",
        "gender-religion-caste": "gender division of labour, communalism, secular state, caste-based inequality and politics",
        "political-parties": "national and regional parties, multi-party system, challenges of political parties, party symbols",
        "outcomes-of-democracy": "accountable and transparent government, economic growth under democracy, dignity and freedom of citizens",
        "development": "national income, per capita income, human development index, sustainability of development",
        "sectors-indian-economy": "primary secondary tertiary sector, organised and unorganised sector, public and private sector, underemployment",
        "money-credit": "loans, banks, formal and informal credit, self-help groups, terms of credit",
        "globalisation-indian-economy": "multinational company, foreign trade, liberalisation, special economic zones, fair globalisation",
        "consumer-rights": "consumer protection act, right to information as a consumer, exploitation in the marketplace, COPRA",
    },
    # English's "chapters" (see syllabus JSON) are skill categories, not
    # topics -- a comprehension passage about any subject is still "Reading
    # Comprehension". Bare topic-style keywords wouldn't help here the way
    # they helped Math; what actually distinguishes these is the *format* of
    # the question, which does have real, consistent vocabulary.
    ("English", 10): {
        "reading-comprehension": "unseen passage, read the passage and answer the following questions, factual passage, discursive passage, note-making",
        "grammar": "fill in the blanks, error correction, editing exercise, tenses, modals, reported speech, determiners, prepositions",
        "writing-skills": "write a letter, write a notice, write an article, analytical paragraph, formal letter, informal letter",
        "literature-textbook": "First Flight, Footprints without Feet, extract based questions, poem, short story, chapter-based question from the textbook",
    },
    ("Hindi", 10): {
        "apthit-gadyansh": "apthit gadyansh kavyansh, unseen passage in hindi, padhkar prashno ke uttar dijiye, gadyansh par aadharit prashna",
        "rachna-ke-aadhar-par-vakya-bhed": "rachna ke aadhar par vakya bhed, saral vakya, sanyukt vakya, mishra vakya, vakya rupantaran",
        "vachya": "vachya, kartrivachya, karmavachya, bhavavachya, vachya parivartan",
        "pad-parichay": "pad parichay, rekhankit pad ka parichay, sangya sarvanam visheshan kriya",
        "alankar": "alankar, shlesh, utpreksha, atishayokti, manavikaran alankar",
        "kshitij-gadya-khand": "kshitij bhag 2 gadya khand, netaji ka chashma, balgobin bhagat, lakhnavi andaz, ek kahani yeh bhi",
        "kshitij-kavya-khand": "kshitij bhag 2 kavya khand, surdas ke pad, ram-lakshman-parashuram samvad, aatmakathya, utsah, sangatkar",
        "kritika-bhag-2": "kritika bhag 2, mata ka aanchal, sana sana hath jodi, main kyon likhta hoon",
        "anuchhed-lekhan": "anuchhed lekhan, vishay par anuchhed, sanket bindu ke aadhar par",
        "patra-lekhan": "patra lekhan, aupcharik patra, anaupcharik patra, pradhanacharya ya sampadak ko patra",
        "vigyapan-lekhan": "vigyapan lekhan, sandesh lekhan, vigyapan taiyar kijiye, shubhkaamna sandesh",
        "email-lekhan": "email lekhan, swavratt lekhan, aupcharik e-mail, resume bio data in hindi",
    },
}


def _anchor_text(subject: str, grade_int: int, chapter_id: str, chapter_name: str) -> str:
    keywords = _ANCHOR_ENRICHMENT.get((subject, grade_int), {}).get(chapter_id)
    return f"{chapter_name}: {keywords}" if keywords else chapter_name


_model = None  # process-wide singleton -- loading it is the expensive part


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def tag_questions(cache: ChapterTagCache, *, subject: str, grade_label: str,
                   grade_int: int, items: list[tuple[str, str]],
                   model: object = None) -> dict[str, tuple[Optional[str], float]]:
    """Tag every (text_hash, text) pair in `items` with a chapter id +
    cosine-similarity confidence. Checks the persistent cache first; only
    cache misses get embedded. `model` lets a caller (tests) inject a
    pre-loaded/fake model; production code leaves it None and the real
    SentenceTransformer loads lazily (and only once per process).

    Returns a dict covering every text_hash in `items`; entries this
    function couldn't classify (no syllabus data for this subject/grade, or
    every candidate scored below _MIN_SIMILARITY) are `(None, 0.0)`, never
    omitted.
    """
    from ..syllabus.cbse_syllabus import load_syllabus

    syllabus = load_syllabus(subject, grade_int)
    if syllabus is None:
        return {h: (None, 0.0) for h, _ in items}
    candidates = [(c.id, c.name) for _, c in syllabus.all_chapters()]
    if not candidates:
        return {h: (None, 0.0) for h, _ in items}

    all_hashes = [h for h, _ in items]
    cached = cache.get_many(subject, grade_label, all_hashes)
    missing = [(h, t) for h, t in items if h not in cached]
    if not missing:
        return cached

    m = model if model is not None else _get_model()
    anchors = [_anchor_text(subject, grade_int, cid, name) for cid, name in candidates]
    chapter_embeds = m.encode(anchors, normalize_embeddings=True)
    question_embeds = m.encode([t for _, t in missing], normalize_embeddings=True)
    sims = question_embeds @ chapter_embeds.T  # cosine similarity, both sides normalized

    new_tags: dict[str, tuple[Optional[str], float]] = {}
    for idx, (h, _) in enumerate(missing):
        row = sims[idx]
        best = int(row.argmax())
        score = float(row[best])
        new_tags[h] = (candidates[best][0], score) if score >= _MIN_SIMILARITY else (None, 0.0)

    cache.put_many(subject, grade_label, new_tags)
    cached.update(new_tags)
    return cached
