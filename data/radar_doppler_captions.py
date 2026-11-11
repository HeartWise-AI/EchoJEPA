# data/radar_doppler_captions.py

"""Caption rules for structures that Doppler stills feed (pure functions; `build_radar_captions.py --doppler-captions`
calls them). Kept out of `radar_normality.py` and `radar_captions.py` on purpose: the normality policy's digest
covers those two files, so editing them would void the approval of the B-mode policy.

- `doppler_clauses`: the clauses of a report cell that `retain_caption` removes because only Doppler can ground them
  (stenosis, regurgitation, gradients, velocities, pressures), as written; a clause that names a date or a person,
  or compares with an earlier examination (`_PRIOR`), still leaves. Their wording is read by the same patterns as
  the B-mode rules (`_names_doppler`).
- `full_caption`: the B-mode caption followed by those clauses, without the B-mode clauses they already hold: the
  caption of a (study, structure) whose bag holds a
  Doppler still of that structure (scope `full`). Without such a still the B-mode caption is used, so a caption never
  says what the study's inputs cannot show.
- `hemo_caption`: the templated caption of Hemo, which has no report column: RVSP, RA pressure and mean E/e' from
  the report's structured fields, in the reports' language (French), only for values in a plausible range.
- `hemo_flag`: a provisional normal / abnormal / unknown flag of those values (RVSP >= 36 mmHg, RA pressure >= 8
  mmHg or E/e' > 14 is abnormal; all present values below is normal; nothing present is unknown). Its rows are
  marked `provisional`: training refuses A2 and A3 on them until a policy with Doppler scope is approved.
"""

import math
import re

import radar_captions as rc
import radar_normality as rn

HEMO_FIELDS = {
    "rvsp": ("RV Systolic Pressure", (5.0, 150.0), "Pression systolique ventriculaire droite estimée à {} mmHg"),
    "rap": ("RA Pressure", (0.0, 30.0), "Pression auriculaire droite estimée à {} mmHg"),
    "e_over_e_prime": (("MV E/e' (Lateral)", "MV E/e' (Septal)"), (1.0, 50.0), "Rapport E/e' moyen {}"),
}
HEMO_ABNORMAL = {"rvsp": 36.0, "rap": 8.0, "e_over_e_prime": 14.0}
FULL_STATUS = "provisional"


def hemo_columns():
    """The report columns `hemo_caption` reads."""
    columns = []
    for source, _, _ in HEMO_FIELDS.values():
        columns += list(source) if isinstance(source, tuple) else [source]
    return columns


def doppler_clauses(text):
    """The Doppler-only clauses of a report cell, as written (see the module docstring)."""
    text = rn._TITLE.sub(" docteur ", rn._CF.sub("cf", rc.caption_text(text)))
    kept = []
    for clause in rc.text_clauses(text):
        clause = " ".join(rn._ESCAPED_CHARACTER.sub(" ", clause).split()).strip(" .;,")
        if not clause or not re.search(r"[^\W\d_]", clause):
            continue
        plain = rn._plain(clause)
        if (rn._names_doppler(clause) and not rn._NOT_IN_THE_CLIP.search(plain) and not rn._DATE.search(clause)
                and not rn._PRIOR.search(plain)):
            kept.append(clause)
    return kept


def full_caption(bmode_caption, source):
    """The B-mode caption followed by the Doppler clauses of the report cell `source`; "" when there are none. A
    Doppler clause is kept whole, so it may hold what the B-mode caption kept of it (`Valve calcifiée` of `Valve
    calcifiée sans sténose`): a B-mode clause that a Doppler clause contains is not said twice."""
    clauses = doppler_clauses(source)
    if not clauses:
        return ""
    said = [rc.canonical_caption(c) for c in clauses]
    head = [c.strip(" .;,") for c in rc.text_clauses(bmode_caption or "")]
    head = [c for c in head if c and not any(rc.canonical_caption(c) in s for s in said)]
    return ". ".join([*head, *clauses]) + "."


def doppler_caption(source):
    """The Doppler clauses of the report cell `source` alone: the caption of a (study, structure) whose bag holds
    stills of the structure and no clip of it (scope `doppler`); "" when there are none."""
    clauses = doppler_clauses(source)
    return ". ".join(clauses) + "." if clauses else ""


def _number(value, digits):
    text = f"{value:.{digits}f}"
    return text.replace(".", ",")


def hemo_values(report):
    """{field: value} of the plausible structured haemodynamic values of a report row."""
    values = {}
    for name, (source, (low, high), _) in HEMO_FIELDS.items():
        if isinstance(source, tuple):
            parts = [rc.finite(report.get(c)) for c in source]
            value = sum(parts) / len(parts) if all(p is not None for p in parts) else None
        else:
            value = rc.finite(report.get(source))
        if value is not None and math.isfinite(value) and low <= value <= high:
            values[name] = value
    return values


def hemo_caption(report):
    """The templated Hemo caption of a report row, or "" when it holds no plausible value."""
    values = hemo_values(report)
    parts = [HEMO_FIELDS[k][2].format(_number(v, 1 if k == "e_over_e_prime" else 0)) for k, v in values.items()]
    return ". ".join(parts) + "." if parts else ""


def hemo_flag(report):
    values = hemo_values(report)
    if not values:
        return "unknown"
    if any(v >= HEMO_ABNORMAL[k] if k != "e_over_e_prime" else v > HEMO_ABNORMAL[k] for k, v in values.items()):
        return "abnormal"
    return "normal"
