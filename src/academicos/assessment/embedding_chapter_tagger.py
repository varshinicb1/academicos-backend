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
    ("Physics", 12): {
        "electric-charges-fields": "Coulomb's law, electric field, electric dipole, Gauss's law, flux, charge distribution, electric lines of force",
        "electrostatic-potential-capacitance": "electric potential, potential difference, equipotential surface, capacitor, capacitance, dielectric, parallel plate capacitor, energy stored",
        "current-electricity": "electric current, drift velocity, Ohm's law, resistance, resistivity, Kirchhoff's rules, Wheatstone bridge, internal resistance, EMF, cell combination",
        "moving-charges-magnetism": "Biot-Savart law, Ampere's law, magnetic field, Lorentz force, motion in magnetic field, solenoid, galvanometer, torque on magnetic dipole",
        "magnetism-matter": "bar magnet, magnetic field lines, magnetic dipole moment, magnetic susceptibility, permeability, diamagnetic, paramagnetic, ferromagnetic",
        "electromagnetic-induction": "magnetic flux, Faraday's law of induction, Lenz's law, eddy currents, self inductance, mutual inductance, motional EMF",
        "alternating-current": "AC generator, peak and RMS current, LCR circuit, phasor, impedance, resonance in AC, power in AC circuit, transformer",
        "electromagnetic-waves": "displacement current, characteristics of EM waves, electromagnetic spectrum, radio waves, microwaves, infrared, ultraviolet, X-rays, gamma rays",
        "ray-optics": "reflection of light, spherical mirrors, refraction, total internal reflection, lens maker's formula, prism, dispersion, optical instruments, microscope, telescope",
        "wave-optics": "wavefront, Huygens' principle, interference of light, Young's double slit experiment, fringe width, diffraction, central maximum, resolution",
        "dual-nature-radiation": "photoelectric effect, work function, threshold frequency, Einstein's equation, stopping potential, de Broglie wavelength, matter waves",
        "atoms": "Rutherford alpha particle scattering, Bohr model of hydrogen atom, energy levels, spectral series, Lyman, Balmer, Paschen, radius of orbit",
        "nuclei": "atomic mass unit, mass defect, binding energy per nucleon, nuclear forces, nuclear fission, nuclear fusion, nuclear stability",
        "semiconductor-electronics": "energy bands, intrinsic extrinsic semiconductor, p-n junction, forward reverse bias, I-V characteristics, rectifier, photodiode, solar cell, LED",
    },
    ("Chemistry", 12): {
        "solutions": "Raoult's law, Henry's law, molarity molality mole fraction, colligative properties, boiling point elevation, freezing point depression, osmotic pressure, van't Hoff factor",
        "electrochemistry": "redox reactions, galvanic cell, Nernst equation, electrode potential, standard hydrogen electrode, Kohlrausch's law, molar conductivity, electrolysis, fuel cell, corrosion",
        "chemical-kinetics": "rate of chemical reaction, average and instantaneous rate, order and molecularity, zero first order reaction, rate constant, half life, activation energy, Arrhenius equation",
        "d-and-f-block-elements": "transition metals, electronic configuration, variable oxidation states, lanthanoid contraction, actinoids, magnetic properties, catalytic property, potassium dichromate, permanganate",
        "coordination-compounds": "coordination entities, ligands, coordination number, Werner's coordination theory, IUPAC naming, isomerism, valence bond theory, crystal field theory",
        "haloalkanes-and-haloarenes": "nomenclature, preparation, nucleophilic substitution SN1 SN2 mechanism, haloarenes, electrophilic substitution, organometallic compounds, chlorobenzene",
        "alcohols-phenols-ethers": "preparation of alcohols, dehydration, acidity of phenols, Kolbe's reaction, Reimer-Tiemann reaction, Williamson ether synthesis, electrophilic substitution of phenols",
        "aldehydes-ketones-carboxylic-acids": "carbonyl group, nucleophilic addition reactions, Tollens Fehling test, aldol condensation, Cannizzaro reaction, oxidation, acidity of carboxylic acids, esterification",
        "amines": "classification of amines, basicity of amines, Gabriel phthalimide synthesis, Hoffmann bromamide reaction, carbylamine reaction, Hinsberg test, diazonium salts",
        "biomolecules": "carbohydrates, monosaccharides, glucose, fructose, glycosidic bond, proteins, amino acids, peptide linkage, primary secondary structure, nucleic acids, DNA, RNA, vitamins",
    },
    ("Biology", 12): {
        "sexual-reproduction-flowering-plants": "flower structure, microsporogenesis, megasporogenesis, pollen grain, embryo sac, pollination, double fertilization, endosperm embryo development, seed, apomixis",
        "human-reproduction": "male and female reproductive system, spermatogenesis, oogenesis, menstrual cycle, fertilization, cleavage, blastocyst, implantation, placenta, parturition, lactation",
        "reproductive-health": "reproductive health awareness, population explosion, contraception, birth control methods, medical termination of pregnancy MTP, amniocentesis, STD, infertility, IVF, ART",
        "principles-inheritance-variation": "Mendel's laws of inheritance, monohybrid dihybrid cross, incomplete dominance, codominance, linkage and crossing over, sex determination, mutation, genetic disorders, pedigree",
        "molecular-basis-inheritance": "structure of DNA RNA, packaging of DNA, transforming principle, replication, transcription, genetic code, translation, lac operon, human genome project, DNA fingerprinting",
        "evolution": "origin of life, evidence for evolution, Darwin's theory of natural selection, Lamarckism, adaptive radiation, Hardy-Weinberg equilibrium, human evolution",
        "human-health-disease": "common infectious diseases, typhoid pneumonia malaria amoebiasis, immune system, innate acquired immunity, allergy, autoimmunity, AIDS HIV, cancer, drug alcohol abuse",
        "microbes-human-welfare": "microbes in household products, curd cheese, industrial production, sewage treatment plant STP, biogas production, biocontrol agents, biofertilizers",
        "biotechnology-principles-processes": "recombinant DNA technology, restriction endonucleases, cloning vectors, plasmids, gel electrophoresis, PCR polymerase chain reaction, bioreactors, downstream processing",
        "biotechnology-applications": "applications of biotechnology in agriculture, Bt crops, RNA interference, gene therapy, genetically engineered insulin, transgenic animals, biosafety issues",
        "organisms-populations": "organism and its environment, major abiotic factors, adaptations, population attributes, population growth curves, logistic exponential, species interactions, mutualism",
        "ecosystem": "ecosystem structure function, productivity, decomposition, energy flow 10 percent law, ecological pyramids, primary secondary succession, nutrient cycling",
        "biodiversity-conservation": "genetic species ecological diversity, patterns of biodiversity, loss of biodiversity, conservation of biodiversity, in situ ex situ conservation, endangered species",
    },
    ("Mathematics", 12): {
        "relations-and-functions": "types of relations, reflexive symmetric transitive equivalence relations, one-one onto bijective functions, composite functions, domain and range",
        "inverse-trigonometric-functions": "inverse trigonometric functions, domain range, principal value branch, graphs of sin inverse cos inverse tan inverse",
        "matrices": "matrix types, row column square zero identity, matrix addition multiplication, transpose, symmetric and skew symmetric matrices, elementary operations, invertible matrix",
        "determinants": "determinant of a square matrix, properties of determinants, minors and cofactors, adjoint and inverse of matrix, system of linear equations, Cramer's rule",
        "continuity-and-differentiability": "continuity at a point, differentiability, derivative of composite functions, chain rule, implicit differentiation, logarithmic differentiation, parametric form, second order derivative",
        "applications-of-derivatives": "rate of change of quantities, increasing and decreasing functions, tangents and normals, maxima and minima, critical points, first and second derivative test",
        "integrals": "integration as inverse process of differentiation, indefinite integral, substitution, partial fractions, integration by parts, definite integrals, fundamental theorem of calculus",
        "applications-of-integrals": "area under simple curves, area of region bounded by curve and line, area between two curves, parabolas, circles, ellipses",
        "differential-equations": "order and degree of differential equation, general and particular solutions, solution by method of separation of variables, homogeneous differential equations, linear differential equations",
        "vectors": "vector algebra, position vector, direction cosines and direction ratios, scalar dot product, vector cross product, projection of vector",
        "three-dimensional-geometry": "direction cosines and ratios, line in 3D space, angle between two lines, shortest distance between skew lines, coplanar lines, Cartesian and vector equations",
        "linear-programming": "linear programming problem, constraints, objective function, feasible region, corner point method, optimal solution, bounded and unbounded",
        "probability": "conditional probability, multiplication theorem, independent events, total probability theorem, Bayes' theorem, random variable, probability distribution",
    },
    ("Accountancy", 12): {
        "partnership-fundamentals": "partnership deed, profit and loss appropriation account, capital accounts, interest on capital, interest on drawings, salary, past adjustments, guarantee of profit",
        "change-in-profit-sharing-ratio": "sacrificing ratio, gaining ratio, valuation of goodwill, average profit super profit, revaluation of assets and liabilities, accumulated profits and reserves",
        "admission-of-partner": "admission of a new partner, new profit sharing ratio, treatment of goodwill premium, revaluation account, capital adjustments of partners",
        "retirement-death-of-partner": "retirement of partner, gaining ratio, treatment of goodwill on retirement, deceased partner share of profit, executor's account, settlement of retiring partner",
        "dissolution-of-partnership-firm": "dissolution of partnership firm, realization account, treatment of unrecorded assets liabilities, payment of liabilities, partners loan, final settlement",
        "accounting-for-share-capital": "issue of shares, equity preference shares, share application allotment calls, pro-rata allotment, calls in arrears, forfeiture of shares, reissue of forfeited shares",
        "accounting-for-debentures": "issue of debentures, debentures as collateral security, discount and premium on issue of debentures, writing off loss on issue of debentures",
        "financial-statements-of-company": "balance sheet of company, schedule III Companies Act, statement of profit and loss, non-current assets, current assets, equity and liabilities",
        "financial-statement-analysis": "comparative balance sheet, common size statement, significance limitations of financial analysis",
        "accounting-ratios": "liquidity ratios current quick, solvency ratios debt equity, proprietary ratio, activity ratios inventory turnover debtors turnover, profitability ratios gross net profit ROI",
        "cash-flow-statement": "cash flow statement, operating activities, investing activities, financing activities, non-cash items, indirect method, cash and cash equivalents",
    },
    ("Business Studies", 12): {
        "nature-significance-management": "concept of management, effectiveness efficiency, characteristics objectives importance of management, management as science art profession, levels of management, coordination",
        "principles-of-management": "principles of management, Henri Fayol principles division of work unity of command scalar chain, scientific management FW Taylor techniques, mental revolution",
        "business-environment": "business environment meaning importance, dimensions of business environment economic social technological political legal, demonetization impact",
        "planning": "concept of planning, importance limitations of planning, planning process, types of plans objective strategy policy procedure rule budget method",
        "organising": "organising process, organisational structure functional divisional, formal and informal organisation, delegation elements authority responsibility accountability, decentralisation",
        "staffing": "staffing concept importance, staffing process, recruitment internal external sources, selection process tests interview, training development on the job off the job",
        "directing": "directing elements, supervision, motivation Maslow's hierarchy of needs, financial non-financial incentives, leadership styles autocratic democratic laissez faire, communication barriers",
        "controlling": "controlling meaning importance, relationship between planning and controlling, controlling process, critical point control, management by exception",
        "financial-management": "financial management objectives, financial decisions investment financing dividend, financial planning, capital structure factors, fixed and working capital factors",
        "financial-markets": "financial markets concept functions, money market instruments treasury bill commercial paper, capital market primary secondary, stock exchange trading procedure, SEBI",
        "marketing-management": "marketing concept marketing philosophies, marketing mix 4 Ps, product branding packaging labelling, pricing factors, physical distribution channels, promotion advertising",
        "consumer-protection": "consumer protection importance, Consumer Protection Act 2019, consumer rights responsibilities, remedies available, consumer redressal commissions district state national",
    },
    ("Economics", 12): {
        "national-income-aggregates": "macroeconomics concepts, gross domestic product GDP, GNP, NNP, NDP at market price and factor cost, circular flow of income, value added method, income expenditure method",
        "money-and-banking": "money meaning functions, commercial banks money credit creation, central bank Reserve Bank of India RBI functions, repo rate reverse repo CRR SLR bank rate",
        "determination-income-employment": "aggregate demand aggregate supply, consumption function saving function, propensity to consume MPC MPS, investment multiplier, excess demand deficient demand",
        "government-budget-economy": "government budget objectives, revenue budget capital budget, tax non-tax revenue, revenue deficit fiscal deficit primary deficit, measures to reduce deficits",
        "balance-of-payments": "balance of payments meaning accounts, current account capital account, balance of trade, foreign exchange rate fixed flexible floating, appreciation depreciation",
        "development-experience-1947-90": "Indian economy on the eve of independence, five year plans common goals, agriculture land reforms green revolution, industrial development IPR 1956, foreign trade",
        "current-challenges-indian-economy": "economic reforms 1991 LPG liberalisation privatisation globalisation, human capital formation, rural development credit marketing, employment growth, sustainable development",
        "development-experience-comparative": "comparative development experiences of India Pakistan China, economic growth GDP growth rate, sector share, human development indicators HDI, demographic indicators",
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
