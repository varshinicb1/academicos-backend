"""Restore scientific notation flattened by PDF text extraction.

PDF text extraction throws away vertical position, so a source paper's
`CH₃COOH` arrives as `CH3COOH`, `10⁸` as `10 8`, and `cm²` as `cm2`. Printing
that on a school paper looks amateurish and, for chemistry, is simply wrong.

These rules are deliberately conservative — a false positive corrupts a
question, which is worse than leaving one formula flat. Digits only become
subscripts inside something that is unambiguously a chemical formula, and only
becomes a power where an exponent is the sole sensible reading.
"""
from __future__ import annotations

import re

_SUB = str.maketrans("0123456789+-", "₀₁₂₃₄₅₆₇₈₉₊₋")
_SUP = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")

# Element symbols, longest first so "Ca" wins over "C".
_ELEMENTS = (
    "Zn|Cu|Fe|Al|Mg|Ca|Na|Cl|Br|Si|Ag|Au|Pb|Sn|Hg|Mn|Ni|Cr|Ba|Sr|Li|Be|Ar|Ne|He|Kr|Xe"
    "|As|Se|Ti|Co|Pt|Sb|Bi|Cd|Ga|Ge|Rb|Cs|Ra|Th|Ur|Pd|Rh|Ir|Os|Ta|Nb|Zr|Sc|Va"
    "|H|C|N|O|P|S|K|F|I|B|U|W|V|Y"
)
# A formula is >= 2 element groups, at least one carrying a digit: H2O, CH3COOH,
# Al2O3, CaCO3. A bare "CO" or "No" must not match.
# Lookbehind rather than \b so a stoichiometric coefficient ("2CH3COOH") does
# not prevent the formula itself from matching.
_FORMULA = re.compile(rf"(?<![A-Za-z])(?:(?:{_ELEMENTS})\d*){{2,}}(?![A-Za-z])")
_ELEMENT_PART = re.compile(rf"({_ELEMENTS})(\d*)")

# Words that look like formulas but are ordinary English (or units).
_FORMULA_BLOCKLIST = {
    "CONCAVE", "CONVEX", "IN", "IS", "IT", "ON", "OF", "OR", "NO", "SO", "TO", "BE",
    "HI", "PH", "AS", "AN", "AT", "BY", "HAS", "HIS", "CAN", "CAP", "CUP", "HOP",
    "SUN", "SON", "SIN", "COS", "TAN", "CBSE", "NCERT", "IUPAC", "SI", "II", "III",
    "INC", "ICSE", "PVC", "LPG", "CNG", "DNA", "RNA", "ATP", "DC", "AC", "DPS",
}

_CHEM_HINT = re.compile(
    r"\b(reaction|acid|base|salt|oxide|compound|element|solution|gas|metal|"
    r"chemical|equation|mole|molecul|ion|precipitat|electrolys|combust|"
    r"hydroxide|carbonate|sulphate|sulphuric|chloride|nitrate|ester|alcohol)",
    re.I,
)


def _subscript_formula(match: re.Match) -> str:
    token = match.group(0)
    if token.upper() in _FORMULA_BLOCKLIST:
        return token
    if not any(ch.isdigit() for ch in token):
        return token          # no digits to lower; leave alone
    out: list[str] = []
    pos = 0
    for m in _ELEMENT_PART.finditer(token):
        if m.start() != pos:            # gap => not a clean formula, bail out
            return token
        symbol, digits = m.group(1), m.group(2)
        out.append(symbol + digits.translate(_SUB))
        pos = m.end()
    if pos != len(token):
        return token
    return "".join(out)


# Single-element molecules (H2, O2, Cl2, O3) have only one element group, so the
# general formula rule skips them; they are unambiguous enough to list.
_DIATOMIC = re.compile(r"(?<![A-Za-z])(H|O|N|Cl|Br|I|F|P|S)(2|3|4|8)(?![A-Za-z0-9])")
# A subscript following a bracketed group: (CH3COO)2 -> (CH₃COO)₂
_GROUP_SUBSCRIPT = re.compile(r"(\))(\d+)(?![A-Za-z0-9])")


def chemical_formulas(text: str) -> str:
    """CH3COOH -> CH₃COOH, Al2O3 -> Al₂O₃ (only in a chemistry context)."""
    chem = bool(_CHEM_HINT.search(text)) or bool(re.search(r"\b[A-Z][a-z]?\d", text))
    if not chem:
        return text
    text = _FORMULA.sub(_subscript_formula, text)
    if _CHEM_HINT.search(text) or "₂" in text or "₃" in text:
        text = _DIATOMIC.sub(lambda m: m.group(1) + m.group(2).translate(_SUB), text)
        text = _GROUP_SUBSCRIPT.sub(lambda m: m.group(1) + m.group(2).translate(_SUB), text)
    return text


# --- powers and units ------------------------------------------------------

# "3 × 10 8" / "3 x 10 8" / "2 108" following a multiplication sign.
_POWER_SPACED = re.compile(r"(?<![\d.])(\d+(?:·\d+)?)\s*[×x]\s*10\s+(\d{1,3})\b")
# "10 8 m/s" — a bare 10 followed by a small integer then a unit.
_POWER_UNIT = re.compile(r"\b10\s+(\d{1,3})\s*(?=(?:m|s|N|J|W|Hz|K|cm|mm|km|kg|g)\b)")
# "1.6 10 –19 C" — extraction drops the × entirely.
_POWER_SIGNED = re.compile(r"\b10\s*[–−-]\s*(\d{1,3})\b")
# cm2, m3, s-1, ms-2 -> cm², m³, s⁻¹
# "A" (Ampere) is deliberately excluded: CBSE geometry/trig questions name
# angles and triangle vertices A, B, C constantly ("sec A", "angle A"), and a
# unit exponent on Ampere essentially never appears in this corpus — matching
# it here turned "9 sec² A  9 tan² A" into "9 sec² A⁹ tan² A", eating the
# second term's coefficient into a bogus exponent on the variable.
_UNIT_POWER = re.compile(
    r"\b(cm|mm|km|m|s|kg|g|N|J|W|V|Ω|K|mol|L|ms|dm)\s*(-?\d)\b")
_UNIT_BLOCKLIST = {"m1", "s1", "g1", "m2b"}


def powers(text: str) -> str:
    text = _POWER_SPACED.sub(lambda m: f"{m.group(1)} × 10{m.group(2).translate(_SUP)}", text)
    text = _POWER_UNIT.sub(lambda m: f"10{m.group(1).translate(_SUP)} ", text)
    text = _POWER_SIGNED.sub(lambda m: f"10⁻{m.group(1).translate(_SUP)}", text)

    def unit_repl(m: re.Match) -> str:
        unit, exp = m.group(1), m.group(2)
        if f"{unit}{exp}".lower() in _UNIT_BLOCKLIST:
            return m.group(0)
        if exp.startswith("-"):
            return unit + "⁻" + exp[1:].translate(_SUP)
        return unit + exp.translate(_SUP)

    return _UNIT_POWER.sub(unit_repl, text)


# --- ions and charges ------------------------------------------------------

# A charge sign attached with no space: "Na+", "SO4²-", "Cl-".
_ION_ATTACHED = re.compile(r"\b([A-Z][a-z]?[₀-₉\d]*)(\d?)([+−–-])(?=[\s,.)]|$)")
# A spaced sign is ambiguous: in "Zn + 2CH3COOH" it means *plus*, in "H + ions"
# it is a charge the extractor separated. Only the latter reading is safe, so
# require the word "ion(s)" to follow. Getting this wrong rewrites the chemistry.
_ION_SPACED = re.compile(
    r"\b([A-Z][a-z]?[₀-₉\d]*)\s+(\d?)\s*([+−–])\s+(?=ions?\b)", re.I)


def _charge(sym: str, num: str, sign: str) -> str:
    return sym + (num.translate(_SUP) if num else "") + ("⁺" if sign == "+" else "⁻")


def ion_charges(text: str) -> str:
    if not _CHEM_HINT.search(text):
        return text
    text = _ION_ATTACHED.sub(lambda m: _charge(m.group(1), m.group(2), m.group(3)), text)
    text = _ION_SPACED.sub(lambda m: _charge(m.group(1), m.group(2), m.group(3)) + " ", text)
    return text


# --- reaction arrows -------------------------------------------------------

_ARROW_WORDS = re.compile(r"\s+(?:gives|yields|produces)\s+(?=[A-Z])")
_DOUBLE_ARROW = re.compile(r"<\s*[-=]+\s*>")
_SINGLE_ARROW = re.compile(r"(?<![<-])[-=]{2,}>")


def arrows(text: str) -> str:
    text = _DOUBLE_ARROW.sub(" ⇌ ", text)
    text = _SINGLE_ARROW.sub(" → ", text)
    return text


# --- fractions -------------------------------------------------------------

_COMMON_FRACTIONS = {
    ("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔", ("1", "4"): "¼",
    ("3", "4"): "¾", ("1", "5"): "⅕", ("2", "5"): "⅖", ("3", "5"): "⅗",
    ("1", "8"): "⅛", ("3", "8"): "⅜", ("5", "8"): "⅝", ("1", "6"): "⅙",
}
# "refractive index of glass and water is 2 3 and 3 4" — a stacked fraction
# flattens to two bare integers. Only convert where the phrasing makes a
# fraction the sole sensible reading, to avoid mangling "2 3" in a list.
# Only convert after a word that makes a fraction the sole sensible reading.
# A bare "is"/"of" is far too loose — "the cost is 2 3 rupees" is not ⅔.
_FRACTION_CTX = re.compile(
    r"\b(?:index|indices|ratio|fraction|probability)\s+(?:of\s+\w+\s+)?"
    r"(?:is|are|=)?\s*(\d)\s+(\d)\b(?!\s*\d)", re.I)


def fractions(text: str) -> str:
    def repl(m: re.Match) -> str:
        num, den = m.group(1), m.group(2)
        if num >= den:                       # 3 2 is not a proper fraction here
            return m.group(0)
        frac = _COMMON_FRACTIONS.get((num, den))
        whole = m.group(0)[: m.start(1) - m.start(0)]
        return f"{whole}{frac}" if frac else f"{whole}{num}/{den}"

    return _FRACTION_CTX.sub(repl, text)


def restore(text: str) -> str:
    """Apply every notation rule. Safe to run on any question stem."""
    if not text:
        return text
    text = arrows(text)
    text = chemical_formulas(text)
    text = ion_charges(text)
    text = powers(text)
    text = fractions(text)
    return text
