"""NCERT Class X Science chapter reference + keyword-based tagger.

Rule-based (not LLM): matches question text against a per-chapter keyword set.
Deliberately simple — the Question model has no chapter_id today (see
models/academic.py), so this exists to bridge that gap for the pilot subject
before a real curriculum-ingestion pass replaces it with graph-backed mapping.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter:
    id: str
    name: str
    keywords: tuple[str, ...]


CLASS_X_SCIENCE_CHAPTERS: tuple[Chapter, ...] = (
    Chapter("chemical-reactions-equations", "Chemical Reactions and Equations", (
        "chemical reaction", "chemical equation", "balanced equation", "decomposition reaction",
        "combination reaction", "displacement reaction", "double displacement", "oxidation",
        "reduction", "exothermic", "endothermic", "precipitate", "rancidity", "corrosion",
        "electrolysis of water",
    )),
    Chapter("acids-bases-salts", "Acids, Bases and Salts", (
        "acid", "base", "ph scale", "indicator", "litmus", "neutralization", "salt",
        "sodium hydroxide", "hydrochloric acid", "baking soda", "bleaching powder",
        "washing soda", "plaster of paris", "olfactory indicator",
    )),
    Chapter("metals-non-metals", "Metals and Non-metals", (
        "metal", "non-metal", "alloy", "ionic compound", "reactivity series", "ore", "metallurgy",
        "corrosion", "galvanization", "malleable", "ductile", "aqua regia", "roasting", "calcination",
    )),
    Chapter("carbon-compounds", "Carbon and its Compounds", (
        "carbon", "covalent bond", "hydrocarbon", "alkane", "alkene", "alkyne", "functional group",
        "ethanol", "ethanoic acid", "soap", "detergent", "catenation", "esterification", "saponification",
    )),
    Chapter("life-processes", "Life Processes", (
        "nutrition", "respiration", "photosynthesis", "transportation", "excretion", "digestion",
        "autotrophic", "heterotrophic", "stomata", "xylem", "phloem", "nephron", "alveoli", "haemoglobin",
    )),
    Chapter("control-coordination", "Control and Coordination", (
        "nervous system", "neuron", "reflex action", "hormone", "endocrine", "synapse",
        "tropism", "auxin", "cerebrum", "cerebellum", "spinal cord", "receptor",
    )),
    Chapter("reproduction", "How do Organisms Reproduce", (
        "reproduction", "asexual reproduction", "sexual reproduction", "budding", "fragmentation",
        "fertilization", "puberty", "menstrual cycle", "placenta", "pollination", "contraception",
    )),
    Chapter("heredity", "Heredity", (
        "heredity", "gene", "chromosome", "dominant trait", "recessive trait", "mendel",
        "variation", "sex determination", "evolution",
    )),
    Chapter("light-reflection-refraction", "Light – Reflection and Refraction", (
        "reflection", "refraction", "mirror", "lens", "focal length", "concave", "convex",
        "snell's law", "refractive index", "real image", "virtual image",
    )),
    Chapter("human-eye", "The Human Eye and the Colourful World", (
        "human eye", "retina", "myopia", "hypermetropia", "presbyopia", "power of accommodation",
        "dispersion", "scattering", "tyndall effect", "rainbow",
    )),
    Chapter("electricity", "Electricity", (
        "electric current", "potential difference", "resistance", "ohm's law", "resistivity",
        "series combination", "parallel combination", "electric power", "kilowatt hour", "fuse",
    )),
    Chapter("magnetic-effects", "Magnetic Effects of Electric Current", (
        "magnetic field", "magnetic field lines", "solenoid", "electromagnet", "electric motor",
        "electric generator", "fleming", "right hand rule", "induced current",
    )),
    Chapter("our-environment", "Our Environment", (
        "ecosystem", "food chain", "food web", "biodegradable", "non-biodegradable", "ozone layer",
        "trophic level", "decomposer", "biomagnification",
    )),
)

_BY_ID = {c.id: c for c in CLASS_X_SCIENCE_CHAPTERS}


def tag_chapter(question_text: str) -> tuple[str | None, float]:
    """Best-effort chapter match. Returns (chapter_id, confidence in [0,1]) or (None, 0.0).

    Confidence is keyword-hit density, not a calibrated probability — good enough
    to rank candidates and flag low-confidence tags for teacher review.
    """
    text = question_text.lower()
    best_id: str | None = None
    best_score = 0
    for chapter in CLASS_X_SCIENCE_CHAPTERS:
        score = sum(1 for kw in chapter.keywords if kw in text)
        if score > best_score:
            best_score = score
            best_id = chapter.id
    if best_id is None:
        return None, 0.0
    confidence = min(1.0, best_score / 3.0)
    return best_id, confidence


def chapter_name(chapter_id: str | None) -> str | None:
    if chapter_id is None:
        return None
    c = _BY_ID.get(chapter_id)
    return c.name if c else None
