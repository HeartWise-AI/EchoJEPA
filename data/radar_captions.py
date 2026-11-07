# data/radar_captions.py

"""Caption text rules for RADAR structure alignment (pure functions; `build_radar_captions.py` reads the reports).

- `caption_text`: a report cell as caption text; anything that is not a nonblank string (None, NaN, pandas NA,
  numbers) is missing and becomes "", never the literal "nan", and so is a cell holding only such a word. Control
  characters become spaces, except line breaks, which end a clause.
- `canonical_caption`: the form compared for exact-caption targets (E). Unicode NFC, one apostrophe, lower case,
  single spaces, trailing punctuation removed. Numbers, units, negations and uncertainty words are kept.
- `caption_id`: a stable signed 64-bit id of the canonical form (BLAKE2b), equal in every process and run, so ranks
  can compare captions without gathering strings. An empty caption has no id.
- `find_ef`: the EF statements of a text, read clause by clause (an EF expression never reaches into the next
  sentence): a point (`FEVG 55 %`), an interval (`entre 50 et 60 %`, `55-60 %`), or a bound (`< 55 %`,
  `supérieure à 55 %`). A statement is uncertain when its own wording hedges or negates (`FEVG environ 55 %`), and
  every statement of a clause is uncertain when the rest of the clause, outside its EF expressions, hedges, negates
  or qualifies a value (`Probablement FEVG 60 %`, `FEVG 60 % environ`, `FEVG 55 % ou moins`, `+/- 5`): the scope is
  the whole clause, before the EF label as well as after the percentage. That errs towards unknown (a clause that
  mixes an EF with a negated finding reads as uncertain here). `ef_expressions` gives the same expressions one by one,
  before that clause-wide rule, for `radar_normality.py`, which judges the words around them itself.
  Only EF expressions count (FEVG, FE VG, FE, fraction d'éjection), not any percentage; strain values are ignored.
- `lv_caption`: the LV caption with the report's structured visual EF appended when the text states none; a text EF
  statement more than `tolerance` points away from the structured value is a conflict (nothing appended, the text
  is kept).
- `text_clauses` and `split_clauses`: a caption's clauses, as written and in canonical form; `clause_template` has its
  numbers replaced by `#`.
- `ef_rule_flag`: normal only when every EF statement is certain and its whole range is at least `ef_normal_min`,
  abnormal only when its whole range is below it, and unknown otherwise (a straddling interval or bound, a hedge, or
  no threshold). It judges EF values only: `radar_normality.py` judges the caption.
"""

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass, replace

EF_SENTENCE = "FEVG estimée visuellement : {} %."
EF_RANGE = (5.0, 90.0)  # the ontology's range for a visual LVEF estimate

_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'"})
_NUMBER = r"\d{1,3}(?:[.,]\d+)?"
_EF_LABEL = r"(?:\bfevg\b|\bfe\s*vg\b|\bfe\b|fraction\s+d'\s*[ée]jection)"
_EF = re.compile(
    _EF_LABEL + r"(?P<gap>(?:\b[23]\s?d\b|[^%\d,]){0,60}?)"
    rf"(?<![-−–])(?P<first>{_NUMBER})\s*%?(?:\s*(?:-|–|à|a|et)\s*(?P<second>{_NUMBER}))?\s*%",
    re.IGNORECASE,
)
_EF_UNREAD = re.compile(_EF_LABEL + r"(?:[^.;]|\.(?=\d))*%", re.IGNORECASE)    # a value the EF expression did not take
_NOT_EF = re.compile(r"strain|\bsgl\b|\bgls\b", re.IGNORECASE)
_HEDGE = re.compile(r"\bnon\b|\bpas\b|\bimpossible|\bdifficile|\bprobabl|\bpossibl|\bsuspect|\bdouteu|\benviron\b"
                    r"|\bapprox|\bautour\b|\bvers\b|\bpeut[- ]?[êe]tre\b|\bsembl|±|\+\s*/\s*-|~|≈|\?",
                    re.IGNORECASE)
_QUALIFIER = re.compile(r"\b(?:ou|et)\s+(?:plus|moins)\b|\bmaxi|\bmini|\bmax\b|\bmin\b|\blimite", re.IGNORECASE)
_LOWER_STRICT = re.compile(r">(?!=)|\bsup[ée]rieure?s?\b|\bsup\b|\bplus de\b|\bau[- ]del[àa]\b", re.IGNORECASE)
_LOWER_CLOSED = re.compile(r">=|≥|\bau moins\b", re.IGNORECASE)
_UPPER_STRICT = re.compile(r"<(?!=)|\binf[ée]rieure?s?\b|\binf\b|\bmoins de\b", re.IGNORECASE)
_UPPER_CLOSED = re.compile(r"<=|≤|\bau plus\b", re.IGNORECASE)
_BETWEEN = re.compile(r"\bentre\b", re.IGNORECASE)
# A spreadsheet writes a line break as `_x000B_` or `_x000D_`: it ends a clause as much as a line break does.
_CLAUSE_SPLIT = re.compile(r"//|;|[\n\r\x0b\x0c\x85\u2028\u2029]|_x000[abcd]_|\.(?!\d)", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_EMPTY_CELL = frozenset({"na", "n/a", "#n/a", "<na>", "nan", "none", "null"})
_DIGITS = re.compile(_NUMBER)


@dataclass(frozen=True)
class EfStatement:
    """The EF values a statement allows: [low, high], open at an end for a strict bound; `certain` is False when
    the clause hedges or negates, or the wording is ambiguous."""
    low: float
    high: float
    low_open: bool = False
    high_open: bool = False
    certain: bool = True


def caption_text(value):
    """A report cell as caption text: a nonblank string, stripped, without control characters; anything else is
    missing (""), and so is a cell that only says it is empty."""
    if not isinstance(value, str):
        return ""
    text = _CONTROL.sub(" ", value.encode("utf-8", "replace").decode("utf-8")).strip()
    return "" if text.lower() in _EMPTY_CELL else text


def canonical_caption(text):
    if not isinstance(text, str):
        raise TypeError("A caption must be text; pass report cells through caption_text first.")
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES).lower()
    return " ".join(text.split()).strip(" .;:,")


def caption_id(text):
    canonical = canonical_caption(text)
    if not canonical:
        raise ValueError("An empty caption has no id: the structure has no training caption.")
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def _number(value):
    return float(value.replace(",", "."))


def _statement(gap, first, second):
    """The EfStatement of one match, or None when a value falls outside the EF range."""
    a = _number(first)
    b = _number(second) if second else None
    if not EF_RANGE[0] <= a <= EF_RANGE[1] or (b is not None and not EF_RANGE[0] <= b <= EF_RANGE[1]):
        return None
    certain = not _HEDGE.search(gap)
    lower = _LOWER_STRICT.search(gap) or _LOWER_CLOSED.search(gap)
    upper = _UPPER_STRICT.search(gap) or _UPPER_CLOSED.search(gap)
    if b is not None:
        return EfStatement(min(a, b), max(a, b), certain=certain and not (lower or upper))
    if _BETWEEN.search(gap) or (lower and upper):
        return EfStatement(a, a, certain=False)
    if lower:
        return EfStatement(a, EF_RANGE[1], low_open=bool(_LOWER_STRICT.search(gap)), certain=certain)
    if upper:
        return EfStatement(EF_RANGE[0], a, high_open=bool(_UPPER_STRICT.search(gap)), certain=certain)
    return EfStatement(a, a, certain=certain)


def _qualified(rest):
    """Whether the text of a clause outside its EF expressions hedges, negates, bounds or qualifies."""
    return any(p.search(rest) for p in (_HEDGE, _QUALIFIER, _LOWER_STRICT, _LOWER_CLOSED, _UPPER_STRICT,
                                        _UPPER_CLOSED, _BETWEEN))


def ef_value(first, second=None):
    """The EfStatement of a value or interval given without EF wording, or None outside the EF range."""
    return _statement("", first, second)


def ef_expressions(clause):
    """(regular expression match, EfStatement) of each EF expression of one clause, its certainty from its own wording
    alone. A strain value or a value outside the EF range is not an EF expression."""
    for m in _EF.finditer(clause):
        if _NOT_EF.search(m.group("gap")):
            continue
        statement = _statement(m.group("gap"), m.group("first"), m.group("second"))
        if statement is not None:
            yield m, statement


def find_ef(text):
    """The EF statements of `text`, clause by clause, in order."""
    text = unicodedata.normalize("NFC", caption_text(text)).translate(_APOSTROPHES)
    statements = []
    for clause in _CLAUSE_SPLIT.split(text):
        found = [statement for _, statement in ef_expressions(clause)]
        if found and _qualified(_EF.sub(" ", clause)):
            found = [replace(st, certain=False) for st in found]
        statements.extend(found)
    return statements


def finite(value):
    """`value` as a float, or None when it is missing, NaN or not a number."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _format_ef(value):
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}".replace(".", ",")


def lv_caption(text, structured_ef=None, tolerance=5.0):
    """(caption, status) for the LV. status: `stated` (the text has EF statements, none incompatible with the
    structured value), `appended` (no EF statement in the text; the structured one is appended), `conflict` (a text
    statement lies more than `tolerance` points from the structured value; the text is kept as is), `none` (no EF
    anywhere, or an EF the text words in a way not read here, so that a second one must not be written beside it). A
    structured value outside the EF range counts as missing. A missing text with a structured EF gives the appended
    sentence alone."""
    text = caption_text(text)
    stated = find_ef(text)
    structured_ef = finite(structured_ef)
    if structured_ef is not None and not EF_RANGE[0] <= structured_ef <= EF_RANGE[1]:
        structured_ef = None
    if not stated and _EF_UNREAD.search(unicodedata.normalize("NFC", text).translate(_APOSTROPHES)):
        return text, "none"
    if stated:
        if structured_ef is not None and any(not s.low - tolerance <= structured_ef <= s.high + tolerance
                                             for s in stated):
            return text, "conflict"
        return text, "stated"
    if structured_ef is not None:
        sentence = EF_SENTENCE.format(_format_ef(structured_ef))
        if not text:
            return sentence, "appended"
        return f"{text.rstrip(' .')}. {sentence}", "appended"
    return text, "none"


def text_clauses(text):
    """The clauses of a caption as written (split at //, ;, line breaks and full stops that are not decimal points)."""
    return [part.strip() for part in _CLAUSE_SPLIT.split(caption_text(text)) if part.strip()]


def split_clauses(text):
    """The canonical clauses of a caption."""
    return [c for c in map(canonical_caption, text_clauses(text)) if c]


def clause_template(clause):
    """The clause with every number replaced by `#`, to count wording apart from its values."""
    return _DIGITS.sub("#", canonical_caption(clause))


def ef_rule_flag(statements, ef_normal_min):
    """normal when every statement is certain and allows only EF >= ef_normal_min; abnormal when every statement is
    certain and allows only EF < ef_normal_min; unknown otherwise (or without a threshold)."""
    if ef_normal_min is None or not statements or not all(s.certain for s in statements):
        return "unknown"
    if all(s.low >= ef_normal_min for s in statements):
        return "normal"
    if all(s.high < ef_normal_min or (s.high == ef_normal_min and s.high_open) for s in statements):
        return "abnormal"
    return "unknown"
