"""Canonical subject names, so coverage reporting stops splitting on filenames.

CBSE's sample-paper filenames are not a controlled vocabulary. Measured across
the 132 downloaded pairs, the bank received 47 "subjects", and the same subject
arrives under several names for three different reasons:

  1. **Different classes, same subject.** `MathsStandard` and `MathsBasic` are
     Class X papers 041 and 241; `EnglishL` and `EnglishComm` are 184 and 101.
  2. **Different years, same subject, different spelling.**
     `CarnaticMelodicInstrument` in one year is `CarnaticMusicMelodicInstrument`
     in another.
  3. **Cosmetic drift.** `HomeScience` beside the board corpus's `Home Science`;
     `Applied-Maths` beside `Applied Mathematics`; `PolSci` beside
     `Political Science`.

Why this does not simply collapse variants
------------------------------------------
The obvious fix -- map every Mathematics variant to "Mathematics" -- would be
wrong, and wrong in the same way the marks cap was wrong before it was scoped.
Maths Standard and Maths Basic are DIFFERENT PAPERS with different difficulty,
and a school that sets Basic must not be shown Standard questions. Same for
English Communicative versus Language & Literature.

So this module produces two fields:

  * `canonical` -- the display name, which PRESERVES the variant:
    `"Mathematics (Standard)"`, `"English (Communicative)"`.
  * `family` -- the grouping key, which collapses it for reporting:
    `"Mathematics"`, `"English"`.

`family` is what a coverage report groups on, so a school asking "what
Mathematics coverage do I have" gets Standard and Basic together, while the
question bank still refuses to substitute one for the other.

Anything not in the table keeps its filename and is its own family, rather than
being guessed at -- an unmapped subject appearing under its own name is visible,
where a wrong merge is not. `unmapped()` exists so that list can be printed
rather than discovered later.
"""
from __future__ import annotations

# CBSE filename stem -> (display name, grouping family)
_SUBJECTS: dict[str, tuple[str, str]] = {
    # --- Mathematics: variants are distinct papers and stay distinct ---
    "MathsStandard": ("Mathematics (Standard)", "Mathematics"),
    "MathsStandardVIC": ("Mathematics (Standard, VIC)", "Mathematics"),
    "MathsBasic": ("Mathematics (Basic)", "Mathematics"),
    "MathsBasicVIC": ("Mathematics (Basic, VIC)", "Mathematics"),
    "Maths": ("Mathematics", "Mathematics"),
    "Applied-Maths": ("Applied Mathematics", "Applied Mathematics"),
    "AppliedMaths": ("Applied Mathematics", "Applied Mathematics"),

    # --- English: same reasoning. Core/Elective are XII papers 301/001;
    #     Communicative and Language & Literature are X papers 101/184. ---
    "EnglishL": ("English (Language & Literature)", "English"),
    "EnglishComm": ("English (Communicative)", "English"),
    "EnglishCore": ("English (Core)", "English"),
    "EnglishElective": ("English (Elective)", "English"),

    # --- Science, where the names already agree with the board corpus ---
    "Science": ("Science", "Science"),
    "Physics": ("Physics", "Physics"),
    "Chemistry": ("Chemistry", "Chemistry"),
    "Biology": ("Biology", "Biology"),
    "Biotechnology": ("Biotechnology", "Biotechnology"),
    "ComputerScience": ("Computer Science", "Computer Science"),
    "InformaticsPractices": ("Informatics Practices", "Informatics Practices"),
    "ComputerApplication": ("Computer Applications", "Computer Applications"),

    # --- Social science ---
    "SocialScience": ("Social Science", "Social Science"),
    "PolSci": ("Political Science", "Political Science"),
    "PoliticalScience": ("Political Science", "Political Science"),
    "History": ("History", "History"),
    "Geography": ("Geography", "Geography"),
    "Economics": ("Economics", "Economics"),
    "Sociology": ("Sociology", "Sociology"),
    "LegalStudies": ("Legal Studies", "Legal Studies"),

    # --- Commerce ---
    "Accountancy": ("Accountancy", "Accountancy"),
    "BusinessStudies": ("Business Studies", "Business Studies"),
    "Entrepreneurship": ("Entrepreneurship", "Entrepreneurship"),
    "ElementsBookKeepingAccountancy": ("Accountancy", "Accountancy"),
    "ElementsBusiness": ("Business Studies", "Business Studies"),

    # --- Cosmetic drift against the board corpus ---
    "HomeScience": ("Home Science", "Home Science"),
    "EnggGraphics": ("Engineering Graphics", "Engineering Graphics"),
    "Graphic": ("Graphic Design", "Graphic Design"),
    "Painting": ("Painting", "Painting"),
    "NCC": ("NCC", "NCC"),

    # --- Music, dance and the fine arts ---
    # Year-to-year spelling: one year says CarnaticMelodicInstrument, another
    # says CarnaticMusicMelodicInstrument. Same paper.
    "CarnaticMelodicInstrument": ("Carnatic Music (Melodic Instrument)",
                                  "Carnatic Music"),
    "CarnaticMusicMelodicInstrument": ("Carnatic Music (Melodic Instrument)",
                                       "Carnatic Music"),
    "CarnaticMusicPercussion": ("Carnatic Music (Percussion)", "Carnatic Music"),
    "CarnaticMusicVocal": ("Carnatic Music (Vocal)", "Carnatic Music"),
    "HindustaniMelodic": ("Hindustani Music (Melodic)", "Hindustani Music"),
    "HindustaniMusicMelodic": ("Hindustani Music (Melodic)", "Hindustani Music"),
    "HindustaniVocal": ("Hindustani Music (Vocal)", "Hindustani Music"),
    "HindustaniPercussion": ("Hindustani Music (Percussion)", "Hindustani Music"),
    "HindustaniMusicPercussion": ("Hindustani Music (Percussion)",
                                  "Hindustani Music"),
    "Bharatnatyam": ("Bharatanatyam", "Dance"),
    "Kathak": ("Kathak", "Dance"),
    "Kathakali": ("Kathakali", "Dance"),
    "Kuchipudi": ("Kuchipudi", "Dance"),
    "Odissi": ("Odissi", "Dance"),
    "Manipuri": ("Manipuri Dance", "Dance"),
    "ManipuriDance": ("Manipuri Dance", "Dance"),

    # --- Languages (MIL = Modern Indian Language) ---
    "Tangkhul": ("Tangkhul", "Tangkhul"),
    "TangkhulMIL": ("Tangkhul", "Tangkhul"),
    "Assamese": ("Assamese", "Assamese"),
    "Bengali": ("Bengali", "Bengali"),
    "Bodo": ("Bodo", "Bodo"),
    "Gujarati": ("Gujarati", "Gujarati"),
    "Kannada": ("Kannada", "Kannada"),
    "Malayalam": ("Malayalam", "Malayalam"),
    "Marathi": ("Marathi", "Marathi"),
    "Odia": ("Odia", "Odia"),
    "Persian": ("Persian", "Persian"),
    "Arabic": ("Arabic", "Arabic"),
    "Tibetan": ("Tibetan", "Tibetan"),
    "Lepcha": ("Lepcha", "Lepcha"),
    "Limboo": ("Limboo", "Limboo"),
    "Bhutia": ("Bhutia", "Bhutia"),
    "Bhoti": ("Bhoti", "Bhoti"),
    "Gurung": ("Gurung", "Gurung"),
    "Tamang": ("Tamang", "Tamang"),
    "Sherpa": ("Sherpa", "Sherpa"),
    "Nepali": ("Nepali", "Nepali"),
    "Mizo": ("Mizo", "Mizo"),
    "Kokborok": ("Kokborok", "Kokborok"),
    "Sanskrit": ("Sanskrit", "Sanskrit"),
    "Urdu": ("Urdu", "Urdu"),
    "Punjabi": ("Punjabi", "Punjabi"),
    "Tamil": ("Tamil", "Tamil"),
    "Telugu": ("Telugu", "Telugu"),
    "Sindhi": ("Sindhi", "Sindhi"),
}

# Hindi is listed so the mapping is complete, even though its PDFs currently
# yield nothing -- the encoding problem is in the source, not the naming.
for _stem in ("HindiCourseA", "HindiCourseB", "HindiCore", "HindiElective"):
    _SUBJECTS[_stem] = ("Hindi", "Hindi")

# --- added after `unmapped()` printed the list, rather than guessed at ------
# The function exists precisely so the gaps are visible; these were all named
# by it. Two things it caught that a hand-written table would have missed:
# `ODIA` in caps is a second spelling of `Odia`, and Sanskrit ships as three
# stems (Core, Comm, and a bare one).
_EXTRA: dict[str, tuple[str, str]] = {
    # Modern foreign languages -- genuinely different papers, own families.
    "French": ("French", "French"),
    "German": ("German", "German"),
    "Japanese": ("Japanese", "Japanese"),
    "Russian": ("Russian", "Russian"),
    "Spanish": ("Spanish", "Spanish"),

    # Indian languages the first pass missed.
    "Kashmiri": ("Kashmiri", "Kashmiri"),
    "Manipuri": ("Manipuri", "Manipuri"),
    "SanskritCore": ("Sanskrit (Core)", "Sanskrit"),
    "Sanskrit-Comm": ("Sanskrit (Communicative)", "Sanskrit"),
    "SanskritComm": ("Sanskrit (Communicative)", "Sanskrit"),
    "SanskritElective": ("Sanskrit (Elective)", "Sanskrit"),
    "ODIA": ("Odia", "Odia"),                   # caps variant of Odia

    # Other subjects the first pass missed.
    "Psychology": ("Psychology", "Psychology"),
    "BhashaMalyeu": ("Bhasha Malyeu", "Bhasha Malyeu"),
    "KTPI": ("Knowledge, Tradition and Practices of India",
             "Knowledge Tradition and Practices of India"),
    "RAI": ("RAI", "RAI"),
    "Agriculture": ("Agriculture", "Agriculture"),
    "FoodNutritionDietetics": ("Food, Nutrition and Dietetics",
                               "Food Nutrition and Dietetics"),
    # The last eight the printed list named. Ur\M stays one family because the
    # A/B/Core/Elective split is the same course at different depth, and Telugu
    # ships under the two regional spellings CBSE uses.
    "Sculpture": ("Sculpture", "Sculpture"),
    "Thai": ("Thai", "Thai"),
    "TeluguAP": ("Telugu (Andhra Pradesh)", "Telugu"),
    "TeluguTL": ("Telugu (Telangana)", "Telugu"),
    "UrduA": ("Urdu (A)", "Urdu"),
    "UrduB": ("Urdu (B)", "Urdu"),
    "UrduCore": ("Urdu (Core)", "Urdu"),
    "UrduElective": ("Urdu (Elective)", "Urdu"),
    "FashionStudies": ("Fashion Studies", "Fashion Studies"),
    "MassMediaStudies": ("Mass Media Studies", "Mass Media Studies"),
}
_SUBJECTS.update(_EXTRA)


def canonical_subject(stem: str) -> tuple[str, str]:
    """`(display_name, family)` for a CBSE filename stem.

    Unmapped stems are returned as their own name and their own family. That is
    deliberate: an unknown subject staying visible under its own name is a
    clue, whereas folding it into a near-match it does not belong to is a
    silent data error -- and this repository has already produced one of those
    by collapsing things that should have stayed apart.
    """
    key = (stem or "").strip()
    if key in _SUBJECTS:
        return _SUBJECTS[key]
    return key, key


def unmapped(stems: list[str]) -> list[str]:
    """The stems with no entry, so the table's gaps can be printed not guessed."""
    return sorted({s for s in stems if s and s not in _SUBJECTS})
