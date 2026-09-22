"""Characters a symbol font took with it, and the ones nothing can name.

A PDF that sets a character in a symbol font (Symbol, Wingdings, MT Extra)
stores it by the font's own code, not by its meaning, and an extractor with no
glyph table maps that code into the Unicode Private Use Area: Symbol's 0xB0
(degree) comes out as U+F0B0. Nothing draws a PUA character -- the web PDF
renderer drops it (assessment/pdf.py `_PRIVATE_USE`) and so does the phone's
`pdf_text_safety.dart` -- so the question prints with a hole where a symbol
belongs. Measured on the served bank at f90f42c: 45 records, 19 distinct code
points, among them cbe:q:Science9PS1, whose options print "W g", "W / g",
"W g /", "W / g" without their rho and so cannot be answered.

The table
---------
`SYMBOL` is the Adobe Symbol encoding (Adobe's own symbol.txt, the AFM
encoding vector for the Symbol typeface, as republished at
unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt): a PUA character U+F0xx
is Symbol code 0xxx. It covers every code the served records use except the
ones listed below.

What the table deliberately leaves out, and why
-----------------------------------------------
The PUA code says which byte the font was asked for, never WHICH font, so a
code is restored only where the served text agrees with Symbol's glyph at it:

  0x43   Symbol Chi, but SQP Social Science X 2023-24 Q2 prints "(2<F043>4=8)"
         -- a multiplication sign from another font.
  0x76   Symbol omega1, but SQP Telugu XII 2023-24 prints it inside Telugu text.
  0xA7   Symbol club, but the three SQP English papers that use it print a list
         bullet ("<F0A7> Strong Curriculum <F0A7> Lack of diversity"), which is
         Wingdings' glyph at 0xA7. Its neighbours 0xA8-0xAA are the other card
         suits and go with it.
  0xE0   Symbol lozenge, but cbe:q:Science10NG31c and SQP Chemistry XII 2025-26
         Q8 print "Ca + H2O <F0E0> Ca(OH)2" -- a reaction arrow.
  0xF0   undefined in Symbol (the Apple logo on a Mac), and printed as an
         implication arrow in cbe:q:Maths10PS2 and SQP Mathematics X 2024-25.

Left out as well: every code Adobe's own table maps back into the Private Use
Area -- the radical extender (0x60), the arrow extenders (0xBD, 0xBE), the
serif and sans register/copyright/trademark (0xD2-0xD4, 0xE2-0xE4), and the
bracket, brace and integral pieces (0xE6-0xEF, 0xF3-0xFE). A piece of a
four-line bracket has no meaning on its own, and there is no character to
restore it to.

A code the table does not cover is not guessed. `private_use_glyph` names the
first such character, and the merge excludes that record (private-use-glyph):
a question that prints a hole is one a student cannot answer, and a wrong
symbol is worse than no question (rule Q1's shape, applied to symbols).
"""
from __future__ import annotations

import re

# Adobe Symbol code -> the character it draws. Codes 0x20-0xFE, minus the ones
# the module docstring lists. Where Adobe's table gives two Unicode values for
# one code (0x44 Delta, 0x57 Omega, 0x6D mu), the letter is used rather than
# the technical symbol (U+2206, U+2126, U+00B5): these papers write Greek.
SYMBOL: dict[int, str] = {
    0x20: " ", 0x21: "!", 0x22: "∀", 0x23: "#", 0x24: "∃", 0x25: "%",
    0x26: "&", 0x27: "∋", 0x28: "(", 0x29: ")", 0x2a: "∗", 0x2b: "+",
    0x2c: ",", 0x2d: "−", 0x2e: ".", 0x2f: "/",
    0x30: "0", 0x31: "1", 0x32: "2", 0x33: "3", 0x34: "4", 0x35: "5", 0x36: "6",
    0x37: "7", 0x38: "8", 0x39: "9",
    0x3a: ":", 0x3b: ";", 0x3c: "<", 0x3d: "=", 0x3e: ">", 0x3f: "?",
    0x40: "≅",
    0x41: "Α", 0x42: "Β", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ",
    0x47: "Γ", 0x48: "Η", 0x49: "Ι", 0x4a: "ϑ", 0x4b: "Κ",
    0x4c: "Λ", 0x4d: "Μ", 0x4e: "Ν", 0x4f: "Ο", 0x50: "Π",
    0x51: "Θ", 0x52: "Ρ", 0x53: "Σ", 0x54: "Τ", 0x55: "Υ",
    0x56: "ς", 0x57: "Ω", 0x58: "Ξ", 0x59: "Ψ", 0x5a: "Ζ",
    0x5b: "[", 0x5c: "∴", 0x5d: "]", 0x5e: "⊥", 0x5f: "_",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε",
    0x66: "φ", 0x67: "γ", 0x68: "η", 0x69: "ι", 0x6a: "ϕ",
    0x6b: "κ", 0x6c: "λ", 0x6d: "μ", 0x6e: "ν", 0x6f: "ο",
    0x70: "π", 0x71: "θ", 0x72: "ρ", 0x73: "σ", 0x74: "τ",
    0x75: "υ", 0x77: "ω", 0x78: "ξ", 0x79: "ψ", 0x7a: "ζ",
    0x7b: "{", 0x7c: "|", 0x7d: "}", 0x7e: "∼",
    0xa0: "€", 0xa1: "ϒ", 0xa2: "′", 0xa3: "≤", 0xa4: "⁄",
    0xa5: "∞", 0xa6: "ƒ",
    0xab: "↔", 0xac: "←", 0xad: "↑", 0xae: "→", 0xaf: "↓",
    0xb0: "°", 0xb1: "±", 0xb2: "″", 0xb3: "≥", 0xb4: "×",
    0xb5: "∝", 0xb6: "∂", 0xb7: "•", 0xb8: "÷", 0xb9: "≠",
    0xba: "≡", 0xbb: "≈", 0xbc: "…", 0xbf: "↵",
    0xc0: "ℵ", 0xc1: "ℑ", 0xc2: "ℜ", 0xc3: "℘", 0xc4: "⊗",
    0xc5: "⊕", 0xc6: "∅", 0xc7: "∩", 0xc8: "∪", 0xc9: "⊃",
    0xca: "⊇", 0xcb: "⊄", 0xcc: "⊂", 0xcd: "⊆", 0xce: "∈",
    0xcf: "∉",
    0xd0: "∠", 0xd1: "∇", 0xd5: "∏", 0xd6: "√", 0xd7: "⋅",
    0xd8: "¬", 0xd9: "∧", 0xda: "∨", 0xdb: "⇔", 0xdc: "⇐",
    0xdd: "⇑", 0xde: "⇒", 0xdf: "⇓",
    0xe1: "〈", 0xe5: "∑", 0xf1: "〉", 0xf2: "∫",
}

# The Private Use Area a symbol-font code lands in (U+F000 + code for a symbol
# font; the whole area is read, so a glyph reference from any other font is
# caught too).
PRIVATE_USE = re.compile("[-]")
_RESTORABLE = {chr(0xf000 + code): ch for code, ch in SYMBOL.items()}


def restore_symbol_font(text: str) -> str:
    """`text` with every Symbol-font code point the table covers restored.

    Returns the text unchanged, character for character, when it holds none.
    """
    if not PRIVATE_USE.search(text):
        return text
    return PRIVATE_USE.sub(lambda m: _RESTORABLE.get(m.group(0), m.group(0)), text)


def private_use_glyph(text: str) -> str | None:
    """The first private-use character the table does not cover, or None."""
    for match in PRIVATE_USE.finditer(text):
        if match.group(0) not in _RESTORABLE:
            return match.group(0)
    return None
