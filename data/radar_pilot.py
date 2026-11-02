# data/radar_pilot.py

"""Choose the 2-3 report-linked training studies of the RADAR caption pilot, keep only the clips reviewers
should see, and build their column review candidates.

The pilot captions V1: B-mode videos of classified views, each reviewed in the column review (`form`
`column_review`). Color flow is left to V2, and a view the classifier labels `OTHER`, or does not classify, to
pretraining: neither reaches the pilot, so no clip goes to the view and type review (`view_type_review`).

Clips are removed in two places before any upload:
- **metadata rule** (`shortlist`): removes `OTHER` and unclassified views, clips without a B-mode or color-flow
  header mode, GE color flow, and clips of 3 frames or fewer (stills). Single-frame objects are absent.
- **the screen** (`data/clip_screen.py` on the shortlisted clips): excludes unreadable, still, analysis, and
  measurement clips; `select` then leaves out color flow read from another vendor's pixels and 3D or multi-panel
  displays named by their GE image code. Uncertain clips are listed separately.
Every clip of a shortlisted study left out is listed with its reasons: `removed_clips.csv` (shortlist) and
`not_kept_clips.csv` (select, both steps). Like every output they hold patient IDs and paths, so they stay in the
private --out-dir, which must be outside the repository. `funnel.csv` (select, also in `summary.md`) counts, from
the archive manifest to the pilot, the videos, studies and patients each step leaves and what it leaves out by
reason (each clip or study under its first: `non_train`, `other_view`, ..., `no_single_report`, ...): aggregate
only.

Example:
    python data/radar_pilot.py shortlist --videos <manifests>/videos.parquet --ef-labels <study_labels_ef.parquet> \\
        --rvsp-labels <study_labels_rvsp.parquet> --reports <df_report_clean.parquet> --pacs <pacs_index.csv> \\
        --out-dir <pilot_dir>
    python data/clip_screen.py features --clips <pilot_dir>/shortlist_clips.csv \\
        --out <pilot_dir>/screen/features.parquet
    python data/clip_screen.py features --clips <pilot_dir>/reference_clips.csv \\
        --out <pilot_dir>/screen/reference.parquet
    <OCR environment>/bin/python -E -s data/clip_screen.py text --features <pilot_dir>/screen/features.parquet \\
        --out <pilot_dir>/screen/features_text.parquet
    python data/clip_screen.py flag --features <pilot_dir>/screen/features_text.parquet \\
        --reference <pilot_dir>/screen/reference.parquet --out <pilot_dir>/screen/flags.parquet
    python data/radar_pilot.py select --out-dir <pilot_dir> --flags <pilot_dir>/screen/flags.parquet \\
        [--n-studies {2,3}] [--vendor-slots VENDOR ...]
    python data/radar_pilot.py candidates --out-dir <pilot_dir> --reports <df_report_clean.parquet> \\
        [--provisional-screen]

Console output and `summary.md` contain only aggregate information. `select` and `candidates` verify that clips have
not changed since screening. Media read failures stop processing, with sensitive details logged to `<out-dir>/media.log`.
"""

import argparse
import datetime
import hashlib
import hmac
import io
import itertools
import json
import math
import os
import re
import secrets
import sys
import unicodedata
import zlib

import numpy as np
import pandas as pd

import clip_screen
from build_manifests import REPO_ROOT, ensure_outside_repo, git_commit, normalize_id, sha256
from link_ef_labels import matches, same_patient, unique_or_drop

# Category order is the priority order: `--n-studies 2` takes the first two.
CATEGORIES = ("normal", "lv_dysfunction", "valve_disease")
# The vendors each --vendor value names: GE, the vendor names whose headers give the mode (`acq_vendor_validated`);
# GE Vingmed alone, the vendor the screen was mostly labelled on. Without --vendor, the categories and the vendor
# slots take any vendor.
VENDOR_GROUPS = {"GE": clip_screen.GE_HEADER_VENDORS, "GE Vingmed Ultrasound": ("GE Vingmed Ultrasound",)}
REQUIRED_VIEW_FAMILIES = ("A4C", "A2C", "PLAX", "PSAX")  # a study's view families by default (`--view-families`).
# The mode captioned (V1), and the header modes a clip may have before the screen reads another vendor's mode from
# its pixels.
CAPTIONED_MODE, IMAGING_MODES = "bmode", ("bmode", "color_flow")
STILL_MAX_FRAMES = 3  # a clip of this many frames or fewer counts as a still (`few_frames`).
# The GE image codes (fourth ImageType value, which the screen calls unusual) of 3D and multi-panel displays. All 22
# GE Vingmed clips with them, labelled blind (codes_01), were 3D renders (7) or displays of two or more
# panels (15, 11 of them measured); none was a single 2D view. Not every one is 3D, hence `3d_or_multipanel`. Another
# vendor's codes and other unusual ones were never labelled: such clips stay uncertain (`unusual_image_content`).
MULTIPANEL_3D_VENDOR, MULTIPANEL_3D_CODES = "GE Vingmed Ultrasound", ("0041", "0051")
VALVE_FIELDS = ("Mitral valve findings", "Aortic valve findings", "Tricuspid valve findings")
REPORT_NUMBERS = ("AV Mean Gradient",)
# The report's patient, and the PACS index columns that link a report's accession to its study (`link_reports`).
REPORT_PATIENT = "Dossier"
PACS_COLUMNS = {"AccessionNumber": "accession", "StudyInstanceUID": "study_id", "PatientID": "pacs_patient",
                "StudyDate": "pacs_date"}
LINK_REASONS = ("no_pacs_accession", "report_rows_differ", "no_report", "patient_or_date_disagree", "several_reports")
LINK_POLICY = "study_identity_and_reports_v2"
CLIP_COLUMNS = ("patient_id", "study_id", "view", "video_path", "dicom_path", "vendor", "model", "acq_category",
                "mode_source", "n_frames", "form", "global_key")
# A vendor slot's category in shortlist.csv and pilot_studies.csv: `vendor:<vendor>`.
VENDOR_SLOT = "vendor:"
LAYOUT_COLUMNS = ("vendor", "model", "acq_category")  # the screen's layouts before the frame size is known.
# What the screen read of each clip's media (clip_screen.changed_media checks the files still hold it).
MEDIA_COLUMNS = ("error", "video_sha256", "dicom_status", "dicom_regions", *clip_screen.HEADER_EVIDENCE, "height",
                 "width")
# --videos must come from the producer that joins the DICOM acquisition index: this branch's build_manifests.py
# keeps neither dicom_path, vendor and model nor the acq_* columns.
MANIFEST_PRODUCER = ("the archive manifest of build_manifests.py with the DICOM acquisition index (commit 57204d1, "
                     "branch icm/archive_pretraining; the one in experiments/icm_archive/manifests)")


def manifest_build(videos, columns, inputs):
    """The manifest `videos` checked to have `columns` (read into `inputs`), and the commit its build recorded in
    BUILD_COMPLETE.json beside it (None without one)."""
    import pyarrow.parquet as pq

    missing = sorted(set(columns) - set(pq.read_schema(io.BytesIO(inputs.read(videos))).names))
    if missing:
        raise SystemExit(f"--videos lacks the columns {missing}: it needs {MANIFEST_PRODUCER}.")
    record = os.path.join(os.path.dirname(os.path.abspath(videos)), "BUILD_COMPLETE.json")
    return inputs.json(record).get("git_commit") if os.path.exists(record) else None
DERIVED_COLUMNS = ("form", "global_key", "mode_source")  # computed here, not read from the videos table.
CAPTION_FORM, VIEW_TYPE_FORM = "column_review", "view_type_review"
SECRET_NAME = ".global_key_secret"
# What each step writes in --out-dir. A step first removes its own outputs and those of the steps after it, so a
# run that stops leaves nothing a later step could take for its result.
SHORTLIST_OUTPUTS = ("shortlist.csv", "shortlist_clips.csv", "reference_clips.csv", "removed_clips.csv",
                     "shortlist.json")
SELECTION_OUTPUTS = ("pilot_studies.csv", "pilot_clips.csv", "uncertain_clips.csv", "not_kept_clips.csv",
                     "funnel.csv", "selection.json", "summary.md")
CANDIDATE_OUTPUTS = ("candidates.json", "view_type_items.csv", "global_keys.csv", "candidates_summary.json")


def view_family(view):
    return view.str.split("_").str[0]


def _normalize(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Regurgitation in French and English, in full or abbreviated (texts are lowercased, accents removed).
LESION = r"insuf\w*|regurgit\w*|fuites?|incompeten\w*|leak\w*"
REGURGITATION = re.compile(rf"\b(?:(?:{LESION})(?:\s+valvulaires?)?\s+(?:mitral\w*|aorti\w*)"
                           rf"|(?:mitral\w*|aorti\w*)(?:\s+valves?)?\s+(?:{LESION})|im|ia|mr|ar|ai)\b")
OWN_VALVE_LESION = re.compile(rf"\b(?:{LESION})")  # in a valve's own field: "insuffisance modérée".
# A moderate or worse grade, in words or in Arabic or Roman numbers ("grade 3", "gr. III", "2/4", "II/IV", "3/3",
# "2 sur 4", "2+"; "1/4" and "I/IV" are mild); "significant" only when no negation precedes it, since "pas
# d'insuffisance significative" is how a report says there is none.
_NUM, _DEN = r"(?:[2-4]|ii|iii|iv)", r"(?:[34]|iii|iv)"
GRADE = re.compile(rf"\b(?:moder\w*|sever\w*|important\w*|massi\w*|torrenti\w*|(?:grade|gr)\s*{_NUM}\b"
                   rf"|{_NUM}\s*/\s*{_DEN}\b|{_NUM}\s+sur\s+(?:4|iv)\b|[2-4]\s*\+)")
SIGNIFICANT = re.compile(r"\bsignifica\w*")
# A grade is tied to the regurgitation only when nothing else in the part could carry it: another finding a severity
# describes ("insuffisance mitrale légère sur valve sévèrement calcifiée", "sténose", "dilatation", "prolapsus"), or
# a grade that is only an adverb ("sévèrement", "severely"), which qualifies an adjective, makes the part ambiguous.
OTHER_FINDING = re.compile(r"\b(?:calcif\w*|stenos\w*|stenot\w*|retreci\w*|epaissi\w*|thicken\w*|dilat\w*|"
                           r"prolaps\w*|hypertroph\w*|degenerat\w*|remani\w*|fibros\w*|scleros\w*|restrict\w*|"
                           r"ectasi\w*|anevrism\w*|aneurysm\w*|vegetation\w*|perforat\w*|ruptur\w*|flail\w*|"
                           r"gradient\w*|hypokin\w*|akines\w*)")
ADVERB_GRADE = re.compile(r"\b(?:moderement|severement|importamment|massivement|moderately|severely|significativement|"
                          r"significantly)\b")
NEGATED_SIGNIFICANT = re.compile(r"\b(?:pas|non|sans|aucune?|ni|absence|no|not|nor|without)\b.*\bsignifica")
# A negation or an uncertainty anywhere in the part makes it "possible": "pas d'insuffisance mitrale modérée",
# "non sévère", "not moderate or severe"; "possible moderate", "cannot exclude severe", "possiblement modérée",
# "ne peut être exclue", "modérée ?".
NEGATION = re.compile(r"\b(pas|non|sans|aucune?|ni|absence|no|not|nor|without)\b")
UNCERTAIN = re.compile(r"\b(possibl\w*|probabl\w*|suspect\w*|suspicion|soupcon\w*|peut[- ]etre|pourrai\w*|sembl\w*|"
                       r"eventuel\w*|douteu\w*|incertain\w*|equivoque|exclu\w*|suggest\w*|evoca\w*|evoqu\w*|"
                       r"compatible|may|might|could|cannot|likely|unlikely|questionabl\w*|uncertain\w*|rul\w* out)\b"
                       r"|\?")


def _graded(part):
    """A moderate or worse grade in this part of a clause (GRADE, or 'significant' without a negation before it)."""
    return bool(GRADE.search(part) or (SIGNIFICANT.search(part) and not NEGATED_SIGNIFICANT.search(part)))


def _unambiguous(part):
    """Whether the grade of this part can only be the regurgitation's: no other finding a severity describes
    (OTHER_FINDING), and a grade other than an adverb (ADVERB_GRADE) once those are left out."""
    return not OTHER_FINDING.search(part) and _graded(ADVERB_GRADE.sub(" ", part))


def regurgitation_grade(text, own_valve=False):
    """'significant' when one part of a clause (clauses end at '.', ';', '//' or a line break; parts at a comma or
    'et', 'and', 'avec', 'with', 'mais', 'but') names mitral or aortic regurgitation (REGURGITATION: 'insuffisance
    mitrale', 'régurgitation aortique', 'fuite mitrale', 'mitral regurgitation', 'aortic insufficiency', 'IM', 'IA',
    'MR', 'AR', 'AI'; with `own_valve`, the text of the mitral or aortic valve's own field, any regurgitation word)
    with a moderate or worse grade (`_graded`), without negation (NEGATION) or uncertainty (UNCERTAIN), and nothing
    else in the part the grade could describe (`_unambiguous`).

    'possible' when any clause states a moderate or worse grade otherwise: the grade of another lesion or of a
    regurgitation the rule does not recognize ('insuffisance mitrale légère, sténose aortique sévère',
    'insuffisance tricuspide modérée'), negated ('non sévère') or uncertain ('possible moderate mitral
    regurgitation', 'cannot exclude severe aortic regurgitation'), or of uncertain attachment ('insuffisance mitrale
    légère sur valve sévèrement calcifiée'). An unrecognized severity statement thus never lets a study count as
    normal. 'none' otherwise. Used to pick studies, never as a label: the reviewer judges
    every clip."""
    if not isinstance(text, str):
        return "none"
    lesion = re.compile(f"{REGURGITATION.pattern}|{OWN_VALVE_LESION.pattern}") if own_valve else REGURGITATION
    grade = "none"
    text = re.sub(r"\bgr\.", "gr ", _normalize(text))  # the abbreviation's dot does not end a sentence.
    for clause in re.split(r"//|;|\n|\.(?!\d)", text):
        parts = [p for p in re.split(r",|\b(?:et|and|avec|with|mais|but)\b", clause) if _graded(p)]
        if not parts:
            continue
        grade = "possible"
        for part in parts:
            if (lesion.search(part) and not NEGATION.search(part) and not UNCERTAIN.search(part)
                    and _unambiguous(part)):
                return "significant"
    return grade


def categorize(s):
    """Boolean columns per category. Missing values never satisfy a criterion. MR/AR/TR grade columns are
    not used: their codebook is not verified, so regurgitation comes from explicit report text, each valve field's
    own regurgitation words included for the mitral and aortic fields. A study whose text only possibly reports
    significant regurgitation, or states another moderate or worse finding in a valve field, is neither normal nor
    valve disease."""
    grades = pd.DataFrame({f: s[f].map(lambda t, own=f != "Tricuspid valve findings": regurgitation_grade(t, own))
                           for f in VALVE_FIELDS}, index=s.index)
    significant, possible = (grades == "significant").any(axis=1), (grades != "none").any(axis=1)
    return pd.DataFrame({
        "normal": (s.ef >= 55) & (s.av_mean_gradient < 20) & (s.rvsp < 35) & ~possible,
        "lv_dysfunction": s.ef <= 35,
        "valve_disease": (s.av_mean_gradient >= 40) | significant,
    }, index=s.index).fillna(False)


def review_form(clips):
    """The Labelbox form a clip would go to: the caption review for a classified view, the view and type review for
    `OTHER` or no view prediction (which the metadata rule removes from the pilot, `other_view`)."""
    unclassified = clips.view.fillna("").isin(["", "OTHER"])
    return pd.Series(np.where(unclassified, VIEW_TYPE_FORM, CAPTION_FORM), index=clips.index)


def mode_source(clips):
    """Where each clip's acquisition mode comes from: 'header' for a vendor whose region headers were validated
    against the decoded videos (`acq_vendor_validated`: GE's three vendor names), else 'pixels': Philips and Siemens
    headers declare one tissue region over a color-flow loop (every Philips clip of the archive reads as B-mode), so
    the screen reads the mode from the clip (`clip_screen` `pixel_mode`) and `select` takes that reading."""
    validated = clips.acq_vendor_validated.fillna(False).astype(bool)
    return pd.Series(np.where(validated, "header", "pixels"), index=clips.index)


def joined_reasons(index, *parts):
    """Per row of `index`, the reasons of `parts` (each an array of one reason or '' per row), ';'-joined as the
    screen's."""
    return pd.Series([";".join(r for r in row if r) for row in zip(*parts)], index=index, dtype=object)


def _count(name, n):
    """'name=n' for a funnel step's `left_out`, '' for none."""
    return f"{name}={int(n)}" if n else ""


def _counts_text(counts):
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    return "; ".join(f"{k}={int(n)}" for k, n in counts.items())


def reason_counts(reasons):
    """'reason=count; ...' of `reasons` (one entry per clip or study, its reasons ';'-joined in the rule's order, ''
    for one not left out), each entry under its first reason (as the screen's `exclude_reason`), most frequent
    first: the counts add up to what the step left out."""
    return _counts_text(reasons[reasons.fillna("") != ""].str.split(";").str[0].value_counts())


def each_reason_counts(reasons):
    """As `reason_counts`, each entry under every reason it has: '' when none has several, else the counts overlap."""
    every = reasons[reasons.fillna("") != ""].str.split(";").explode()
    if every.index.is_unique:
        return ""
    return "every reason counted: " + _counts_text(every.value_counts())


FUNNEL_COLUMNS = ("stage", "step", "videos", "studies", "patients", "left_out_unit", "left_out", "note")


def funnel_row(stage, step, clips, left_out="", unit="", note=""):
    """One step of the funnel (`funnel.csv`): the videos, studies and patients of `clips`, what is left after the
    step, and what the step left out by reason (`left_out`, counts of `unit`: clips or studies). Aggregate only."""
    return {"stage": stage, "step": step, "videos": int(len(clips)), "studies": int(clips.study_id.nunique()),
            "patients": int(clips.patient_id.nunique()), "left_out_unit": unit if left_out else "",
            "left_out": left_out, "note": note}


def clip_rule(clips):
    """Why each clip fails the metadata rule, ';'-joined ('' when it passes): `other_view`, a view the classifier
    labels `OTHER` or does not classify (pretraining only, never captioned); `acquisition`, a header declaring neither
    B-mode nor color-flow imaging, or a validated vendor's not read as such (`acq_confidence` other than header);
    `color_flow`, a validated vendor's color flow (another vendor's mode is read from the pixels, and `select` leaves
    out its color: `pilot_rule`); `few_frames`, STILL_MAX_FRAMES frames or fewer, a still (a clip without a frame
    count is left to the screen, which reads its frames)."""
    validated = clips.acq_vendor_validated.fillna(False).astype(bool)
    known = clips.acq_category.isin(IMAGING_MODES) & (~validated | (clips.acq_confidence == "header"))
    return joined_reasons(clips.index,
                          np.where(review_form(clips) == VIEW_TYPE_FORM, "other_view", ""),
                          np.where(~known, "acquisition", ""),
                          np.where(known & validated & (clips.acq_category == "color_flow"), "color_flow", ""),
                          np.where((clips.n_frames <= STILL_MAX_FRAMES).fillna(False), "few_frames", ""))


def removed_clips(every, short, clips, vendors, years):
    """The clips of the shortlisted studies that are not screened (not in `clips`), each with its `reasons`
    (';'-joined, as the screen's): the metadata rule's (`clip_rule`) and, for a category study's clip, `vendor`
    (not of --vendor's `vendors`, when given) and `year` (study date outside --years)."""
    removed = every[every.study_id.isin(set(short.study_id)) & ~every.video_path.isin(set(clips.video_path))]
    removed = removed.drop_duplicates("video_path")
    category = removed.study_id.isin(set(short.study_id[~short.category.str.startswith(VENDOR_SLOT)]))
    return removed.assign(reasons=joined_reasons(
        removed.index, removed.clip_rule,
        np.where(category & bool(vendors) & ~removed.vendor.isin(vendors or ()), "vendor", ""),
        np.where(category & ~removed.year.isin(years), "year", "")))


def study_shortfall(clips, min_clips, families):
    """Per study, why it fails the criteria (';'-joined, '' when it meets them): `min_clips`, fewer than `min_clips`
    of `clips`; `view_families`, a family of `families` missing among them. The clips are those the pilot would
    caption, B-mode clips of classified views already: those passing the clip rule (shortlist, where another
    vendor's clip counts whatever its mode until the screen reads it from the pixels) or kept after the screen and
    the pilot rule (select)."""
    per = clips.assign(family=view_family(clips.view))
    n = per.groupby("study_id").size()
    present = per.groupby("study_id").family.agg(frozenset)
    return joined_reasons(n.index, np.where(n < min_clips, "min_clips", ""),
                          np.where(present.map(lambda f: all(x in f for x in families)), "", "view_families"))


def study_ok(clips, min_clips, families):
    """Per study, whether it meets the criteria (`study_shortfall`)."""
    return study_shortfall(clips, min_clips, families) == ""


def vendor_slots(specs, vendors):
    """{vendor: studies to shortlist} from --vendor-studies items 'VENDOR=N', each of --vendor's `vendors` (any vendor
    when None)."""
    slots = {}
    for spec in specs:
        vendor, _, n = spec.rpartition("=")
        if not vendor or not n.isdigit() or int(n) < 1 or vendor in slots:
            raise SystemExit(f"--vendor-studies takes VENDOR=N items, N at least 1, each vendor once; got {spec!r}.")
        if vendors and vendor not in vendors:
            raise SystemExit(f"--vendor-studies names {vendor!r}, outside --vendor's vendors {list(vendors)}: drop "
                             "it, or leave --vendor out for any vendor.")
        slots[vendor] = int(n)
    return slots


def excluded_patients(paths, inputs):
    """The patients of the --exclude-patients lists (CSV files with patient_id, such as a validation list), read
    into `inputs`."""
    excluded = set()
    for path in paths:
        listed = inputs.csv(path, dtype=str)
        if "patient_id" not in listed:
            raise SystemExit(f"--exclude-patients {os.path.basename(path)} has no patient_id column.")
        excluded |= set(normalize_id(listed.patient_id.dropna()))
    return excluded


def link_reports(reports, pacs, studies):
    """One report per study of `studies` (study_id, patient_id, study_date), found through the PACS index (`pacs`:
    accession, study_id, pacs_patient, pacs_date) by its accession and checked as `link_ef_labels.py` checks its
    labels: an accession naming several PACS studies, patients or dates is dropped, and the report's patient
    (REPORT_PATIENT), the PACS patient and the manifest's must agree, as must the PACS and manifest dates. No EF
    label is needed. Report rows repeated exactly are one report; an accession whose rows differ, or a study with
    several reports, is left out rather than one of them chosen (`candidates` refuses them too). Returns the linked
    reports, one row per study with its study_id and accession, and per study of `studies` why it has none
    (LINK_REASONS, '' when linked)."""
    reports = reports.dropna(subset=["AccessionNumber"]).drop_duplicates()
    differ = reports.AccessionNumber.duplicated(keep=False)
    differing, reports = set(reports.AccessionNumber[differ]), reports[~differ]
    pacs = unique_or_drop(pacs.dropna(subset=["accession", "study_id"]), "accession",
                          ["study_id", "pacs_patient", "pacs_date"], "accessions")
    ids = set(studies.study_id)
    pacs = pacs[pacs.study_id.isin(ids)]
    linked = (pacs.merge(reports, left_on="accession", right_on="AccessionNumber")
              .merge(studies[["study_id", "patient_id", "study_date"]], on="study_id"))
    agree = (same_patient(normalize_id(linked[REPORT_PATIENT]), linked.pacs_patient)
             & same_patient(linked.pacs_patient, linked.patient_id)
             & matches(linked.pacs_date.str[:8] == linked.study_date.str[:8]))
    agreeing = linked[agree]
    # Ambiguity belongs to the study, before incompatible reports are filtered out. A valid accession must not
    # hide another report with conflicting rows or a patient/date mismatch.
    several = set(linked.study_id[linked.study_id.duplicated(keep=False)])
    in_pacs, with_report = set(pacs.study_id), set(linked.study_id)
    rows_differ = set(pacs.study_id[pacs.accession.isin(differing)])
    reason = pd.Series("", index=pd.Index(sorted(ids), name="study_id"), dtype=object)
    for name, left in zip(LINK_REASONS, (ids - in_pacs, rows_differ, in_pacs - with_report - rows_differ,
                                         with_report - set(agreeing.study_id), several)):
        reason.loc[sorted(left)] = name
    reason.loc[sorted(rows_differ)] = "report_rows_differ"
    one = agreeing[agreeing.study_id.map(reason) == ""].drop(
        columns=["AccessionNumber", REPORT_PATIENT, "pacs_patient", "pacs_date", "patient_id", "study_date"])
    return one.reset_index(drop=True), reason


def shortlist(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    clear_outputs(args.out_dir, SHORTLIST_OUTPUTS + SELECTION_OUTPUTS + CANDIDATE_OUTPUTS)
    vendors = VENDOR_GROUPS.get(args.vendor)  # None: any vendor.
    slots = vendor_slots(args.vendor_studies, vendors)
    rng = np.random.default_rng(args.seed)
    # Every input as read (Inputs): the record names those, and is written only if none changed meanwhile.
    inputs = clip_screen.Inputs(__file__)

    cols = [c for c in CLIP_COLUMNS if c not in DERIVED_COLUMNS] + ["split", "study_date", "acq_confidence",
                                                                     "acq_vendor_validated"]
    built_at = manifest_build(args.videos, cols, inputs)
    every = inputs.parquet(args.videos, columns=cols)
    # The funnel (`funnel_row`): what each step leaves and leaves out, from the manifest on (the steps before it are
    # in the manifest's summary.md).
    funnel = [funnel_row("shortlist", "archive manifest: TTE videos of every split", every)]
    split = every.split.astype("string").fillna("<none>")
    every = every[every.split == "train"].copy()  # the manifest holds TTE exams only.
    funnel.append(funnel_row("shortlist", "training split", every, _count("non_train", (split != "train").sum()),
                             "clips", "splits: " + reason_counts(split[split != "train"])))
    every["patient_id"], every["study_id"] = normalize_id(every.patient_id), normalize_id(every.study_id)
    # Validate all training clips before year/vendor/clip rules or patient exclusions can hide a conflicting row.
    # The later first() calls are safe only if the entire study has a single identity and date.
    identities = every.groupby("study_id")[["patient_id", "study_date"]].nunique(dropna=False)
    conflicting = (identities > 1).any(axis=1)
    if conflicting.any():
        raise SystemExit(f"{int(conflicting.sum())} training studies have conflicting patient IDs or study dates "
                         "in the manifest. Correct the source metadata and rebuild the manifest before shortlisting.")
    every["year"] = pd.to_numeric(every.study_date.astype("string").str[:4], errors="coerce")
    every["form"], every["clip_rule"], every["mode_source"] = review_form(every), clip_rule(every), mode_source(every)
    families = list(dict.fromkeys(args.view_families))
    unknown = set(families) - set(view_family(pd.Series(every.view[every.form == CAPTION_FORM].unique())))
    if unknown:
        raise SystemExit(f"--view-families {sorted(unknown)} name no classified view of the training clips (a "
                         "family is a view's name before its first '_': A4C_ZOOM is A4C).")
    # The patients of other lists (a validation list): none of their studies or clips is drawn, reference ones neither.
    excluded = excluded_patients(args.exclude_patients, inputs)
    listed = every.patient_id.isin(excluded)
    every = every[~listed]
    funnel.append(funnel_row("shortlist", f"--exclude-patients ({len(excluded)} patients listed)", every,
                             _count("listed_patient", listed.sum()), "clips"))
    v = every[every.year.isin(args.years)]
    funnel.append(funnel_row("shortlist", f"--years {' '.join(map(str, args.years))}", v,
                             _count("other_year", len(every) - len(v)), "clips"))
    in_scope = v.vendor.isin(vendors) if vendors else pd.Series(True, index=v.index)
    step = (f"--vendor {args.vendor}" + (f" ({', '.join(vendors)})" if vendors != (args.vendor,) else "")
            if vendors else "vendor: any (no --vendor)")
    funnel.append(funnel_row("shortlist", step, v[in_scope], _count("other_vendor", (~in_scope).sum()), "clips"))
    v = v[in_scope]
    kept = v[v.clip_rule == ""]
    funnel.append(funnel_row("shortlist", f"clip rule: B-mode clips of classified views, more than {STILL_MAX_FRAMES} "
                             "frames", kept, reason_counts(v.clip_rule), "clips", each_reason_counts(v.clip_rule)))

    ef = inputs.parquet(args.ef_labels, columns=["study_id", "accession", "ef"])
    rvsp = inputs.parquet(args.rvsp_labels, columns=["study_id", "accession", "rvsp"])
    for table in (ef, rvsp):
        table["study_id"], table["accession"] = normalize_id(table.study_id), normalize_id(table.accession)
    reports = inputs.parquet(args.reports)
    reports["AccessionNumber"] = normalize_id(reports.AccessionNumber)
    pacs = inputs.csv(args.pacs, dtype=str, usecols=list(PACS_COLUMNS)).rename(columns=PACS_COLUMNS)
    for c in ("accession", "study_id", "pacs_patient"):
        pacs[c] = normalize_id(pacs[c])
    # Each study's report through the PACS index, with no EF needed to link it; EF and RVSP where the label files
    # hold them for that report. A category asking for one leaves a study without it out (`categorize`).
    studies = every.groupby("study_id", as_index=False)[["patient_id", "study_date"]].first()
    linked, link_reason = link_reports(reports, pacs, studies)
    linked = (linked[["study_id", "accession", *VALVE_FIELDS, *REPORT_NUMBERS]]
              .merge(ef, on=["study_id", "accession"], how="left")
              .merge(rvsp, on=["study_id", "accession"], how="left"))
    linked["av_mean_gradient"] = pd.to_numeric(linked["AV Mean Gradient"], errors="coerce")
    linked_ids = set(linked.study_id)
    unlinked = link_reason.reindex(kept.study_id.unique())
    funnel.append(funnel_row("shortlist", "one report through the PACS index", kept[kept.study_id.isin(linked_ids)],
                             _count("no_single_report", (unlinked != "").sum()), "studies", reason_counts(unlinked)))
    shortfall = study_shortfall(kept, args.min_clips, families)
    ok = shortfall == ""
    s = linked[linked.study_id.isin(ok[ok].index)].sort_values("study_id").reset_index(drop=True)
    short_linked = shortfall[shortfall.index.isin(linked_ids)]
    funnel.append(funnel_row("shortlist", f"--min-clips {args.min_clips}, --view-families "
                             f"{' '.join(families) or '(none)'}", kept[kept.study_id.isin(set(s.study_id))],
                             reason_counts(short_linked), "studies", each_reason_counts(short_linked)))
    patients = kept.groupby("study_id").patient_id.first()
    s["patient_id"] = s.study_id.map(patients)
    cat = categorize(s)

    rows, pools = [], {}
    for c in CATEGORIES:
        pool = s.loc[cat[c], "study_id"].to_numpy()
        pools[c] = int(len(pool))
        for rank, sid in enumerate(rng.permutation(pool)[: args.per_category]):
            rows.append({"category": c, "rank": rank, "study_id": sid, "patient_id": patients[sid],
                         "accession": s.loc[s.study_id == sid, "accession"].iloc[0]})
    short = pd.DataFrame(rows, columns=["category", "rank", "study_id", "patient_id", "accession"])
    if short.empty:
        raise SystemExit("No study meets the pilot criteria; relax --years, --vendor or --min-clips.")
    category_studies = set(short.study_id)
    in_category = set(s.study_id[cat.any(axis=1)])
    funnel.append(funnel_row("shortlist", "report categories", kept[kept.study_id.isin(in_category)],
                             _count("no_category", len(s) - len(in_category)), "studies",
                             "pools (a study can be in several): " + ", ".join(f"{c}={n}" for c, n in pools.items())))
    funnel.append(funnel_row("shortlist", f"seeded draw (--per-category {args.per_category}, --seed {args.seed})",
                             kept[kept.study_id.isin(category_studies)],
                             _count("not_drawn", len(in_category - category_studies)), "studies"))

    # Vendor slots: per vendor named (of --vendor's, when given), report-linked studies of that vendor alone with the
    # required views, of any year and whatever their report category (a slot is for the vendor, not a category: select
    # never takes a slot's study for one), of patients not drawn in the categories or an earlier slot. Drawn from a
    # stream of their own so that the category draws are those of a shortlist without them. Their B-mode count waits
    # for select, once the screen has read the mode. Each slot's funnel row counts the shortlist so far, the slot's
    # own clips in its note.
    slot_rows, slot_pools, slot_clips, slot_kept = [], {}, [], []
    shortlisted = kept[kept.study_id.isin(category_studies)]
    vendors_per_study = every.groupby("study_id").vendor.nunique()
    for vendor, n in slots.items():
        w = every[(every.vendor == vendor) & (every.clip_rule == "")]
        slot_kept.append(w)
        w = w[w.study_id.map(vendors_per_study) == 1]
        ok_w = study_ok(w, args.min_clips, families)
        pool = linked[linked.study_id.isin(ok_w[ok_w].index)].drop_duplicates("study_id").sort_values("study_id")
        pool = pool.assign(patient_id=pool.study_id.map(w.groupby("study_id").patient_id.first()))
        taken = set(short.patient_id) | {r["patient_id"] for r in slot_rows}
        pool = pool[~pool.patient_id.isin(taken)].reset_index(drop=True)
        slot_pools[vendor] = int(len(pool))
        if pool.empty:
            raise SystemExit(f"No training study of {vendor} alone has a linked report and {args.min_clips} classified "
                             "clips of the required views, of patients not drawn otherwise: drop it from "
                             "--vendor-studies.")
        drawn = pool.iloc[np.random.default_rng([args.seed, zlib.crc32(vendor.encode())]).permutation(len(pool))[:n]]
        slot_rows += [{"category": VENDOR_SLOT + vendor, "rank": rank, "study_id": row.study_id,
                       "patient_id": row.patient_id, "accession": row.accession}
                      for rank, row in enumerate(drawn.itertuples())]
        slot_clips.append(w[w.study_id.isin(set(drawn.study_id))])
        shortlisted = pd.concat([shortlisted, slot_clips[-1]]).drop_duplicates("video_path")
        added = funnel_row("", "", slot_clips[-1])
        funnel.append(funnel_row("shortlist", f"vendor slot: {vendor}={n}", shortlisted,
                                 note=f"adds videos={added['videos']}, studies={added['studies']}, "
                                      f"patients={added['patients']}; pool={slot_pools[vendor]}"))
    short = pd.concat([short, pd.DataFrame(slot_rows, columns=short.columns)], ignore_index=True)

    outputs = [os.path.join(args.out_dir, n) for n in SHORTLIST_OUTPUTS[:4]]
    short.to_csv(outputs[0], index=False)
    clips = shortlisted
    secret = load_secret(args.out_dir)
    clips = clips.assign(global_key=[global_key(secret, p) for p in clips.video_path])
    clips[list(CLIP_COLUMNS)].to_csv(outputs[1], index=False)
    # Reference clips for the screen's layout baselines: other studies, the layouts the shortlist has; the vendor
    # slots' drawn apart, so that the categories' are those of a shortlist without them.
    def references(pool, listed):
        pool = pool[~pool.study_id.isin(set(short.study_id))]
        pool = pool.merge(listed[list(LAYOUT_COLUMNS)].drop_duplicates(), on=list(LAYOUT_COLUMNS))
        return (pool.sample(frac=1.0, random_state=args.seed)
                .groupby(list(LAYOUT_COLUMNS), dropna=False).head(args.reference_per_layout))

    reference = references(kept, clips[clips.study_id.isin(category_studies)])
    if slot_kept:
        reference = pd.concat([reference, references(pd.concat(slot_kept), clips[~clips.study_id.isin(
            category_studies)])]).drop_duplicates("video_path")
    reference = reference.assign(global_key=[global_key(secret, p) for p in reference.video_path])
    reference[list(CLIP_COLUMNS)].to_csv(outputs[2], index=False)
    funnel.append(funnel_row("shortlist", "shortlist: clips to screen", clips,
                             note=f"reference clips of other studies, for the screen's layout baselines="
                                  f"{len(reference)}"))
    # Every clip of a shortlisted study that the screen will not see, with its reasons and its global key.
    removed = removed_clips(every, short, clips, vendors, args.years)
    removed = removed.assign(global_key=[global_key(secret, p) for p in removed.video_path])
    removed[[*CLIP_COLUMNS, "reasons"]].to_csv(outputs[3], index=False)
    info = {
        "created_utc": _now(), "code_commit": git_commit(), "code_sha256": clip_screen.code_sha256(__file__),
        "seed": args.seed,
        "criteria": {"split": "train", "years": args.years, "vendor": args.vendor or "any",
                     "link_policy": LINK_POLICY,
                     "vendors": list(vendors) if vendors else "any",
                     "min_clips": args.min_clips,
                     "required_view_families": families,
                     "captioned": "B-mode videos of classified views (V1): OTHER and unclassified views, color flow "
                                  "and stills are left out, min_clips counts B-mode clips",
                     "still_max_frames": STILL_MAX_FRAMES,
                     "report_link": "one report per study through the PACS index by accession, report, PACS and "
                                    "manifest agreeing on the patient and PACS and manifest on the date; rows "
                                    "repeated exactly are one report, an accession whose rows differ or a study with "
                                    "several reports is left out; EF and RVSP where the label files hold them for "
                                    "that report",
                     "categories": {"normal": "EF >= 55, AV mean gradient < 20, RVSP < 35 (all present), no "
                                              "moderate or worse statement in a valve field of the report",
                                    "lv_dysfunction": "EF <= 35",
                                    "valve_disease": "AV mean gradient >= 40, or moderate/severe MR or AR in the "
                                                     "report text, the grade in the same part as the MR or AR"},
                     "vendor_studies": slots,
                     "vendor_slots": "per vendor (of --vendor's vendors, when given), training studies of that "
                                     "vendor alone with a linked report and min_clips classified clips of the required "
                                     "view families, any year, of patients drawn nowhere else; their B-mode clips are "
                                     "counted at select, on the mode the screen reads; select fills only the slots "
                                     "--vendor-slots names",
                     "exclude_patients": {os.path.abspath(p): inputs.digest(p) for p in args.exclude_patients},
                     "excluded_patients": len(excluded)},
        "inputs": {k: inputs.digest(p) for k, p in (("videos", args.videos), ("ef_labels", args.ef_labels),
                                                    ("rvsp_labels", args.rvsp_labels), ("reports", args.reports),
                                                    ("pacs", args.pacs))},
        "manifest_build_commit": built_at,
        "report_link": {"linked": int((link_reason == "").sum()),
                        **{r: int((link_reason == r).sum()) for r in LINK_REASONS}},
        "eligible_studies": int(len(s)), "pool_per_category": pools, "pool_per_vendor_slot": slot_pools,
        "clips_by_mode_source": {k: int(n) for k, n in clips.mode_source.value_counts().items()},
        "shortlisted_studies": int(short.study_id.nunique()), "clips_to_screen": int(len(clips)),
        "reference_clips": int(len(reference)), "reference_per_layout": args.reference_per_layout,
        "clips_removed": int(len(removed)),
        "clips_removed_by_metadata_rule": {k: int(n) for k, n in
                                           removed.reasons.str.split(";").explode().value_counts().items()},
        "funnel": funnel,  # select completes it in funnel.csv.
        "outputs": file_hashes(*outputs),
    }
    inputs.check(*outputs)
    _write_json(os.path.join(args.out_dir, "shortlist.json"), info)  # last: it marks a completed shortlist.
    print(f"[pilot]  {len(s)} eligible training studies; pools {pools}; vendor slot pools {slot_pools}; "
          f"{info['shortlisted_studies']} shortlisted; {len(clips)} clips to screen (mode from "
          f"{info['clips_by_mode_source']}); removed by metadata {info['clips_removed_by_metadata_rule']}.")
    print_funnel(funnel)


def print_funnel(funnel):
    for row in funnel:
        print(f"[funnel] {row['step']}: {row['videos']} videos, {row['studies']} studies, {row['patients']} patients"
              + (f"; left out ({row['left_out_unit']}) {row['left_out']}" if row["left_out"] else ""))


def screened_clips(clips, flags):
    """The shortlisted clips with the screen's decisions; `exclude` marks every clip the pilot leaves out: those
    the screen excludes and those it is uncertain about (`decision` other than keep), which are listed apart, never
    taken for invalid."""
    missing = set(clips.video_path) - set(flags.video_path)
    if missing:
        raise SystemExit(f"{len(missing)} shortlisted clips were not screened; run clip_screen.py on "
                         "shortlist_clips.csv first.")
    screened = clips.merge(flags, on="video_path", how="left", validate="one_to_one")
    return screened.assign(exclude=screened.decision != "keep")


def resolve_modes(clips):
    """The screened clips with the acquisition mode the review uses (`acq_category`): the header's for `mode_source`
    'header', the screen's reading of the pixels (`pixel_mode`) for 'pixels'; the header's own is kept as
    `header_acq_category`."""
    pixels = clips.mode_source == "pixels"
    return clips.assign(header_acq_category=clips.acq_category,
                        acq_category=clips.acq_category.where(~pixels, clips.pixel_mode))


def pilot_rule(clips):
    """Why `select` leaves out a screened clip at its resolved mode (`resolve_modes`), ';'-joined ('' for none):
    `color_flow`, color flow read from the pixels (`acquisition` for no mode read); `3d_or_multipanel`, a header of
    MULTIPANEL_3D_VENDOR naming an image code of MULTIPANEL_3D_CODES."""
    display = (clips.vendor == MULTIPANEL_3D_VENDOR) & clips.dicom_image_content.isin(MULTIPANEL_3D_CODES)
    mode = np.where(clips.acq_category == "color_flow", "color_flow",
                    np.where(clips.acq_category != CAPTIONED_MODE, "acquisition", ""))
    return joined_reasons(clips.index, mode, np.where(display, "3d_or_multipanel", ""))


def with_pilot_rule(clips):
    """The screened clips at their resolved mode, each that `pilot_rule` leaves out excluded with the screen's reasons
    and the rule's. The screen calls the image codes of 3D and multi-panel displays unusual (`unusual_image_content`,
    uncertain): `3d_or_multipanel` names that evidence and takes its place."""
    rule = pilot_rule(clips)
    reasons = [";".join(r for r in (*screen.split(";"), *own.split(";"))
                        if r and not (r == "unusual_image_content" and "3d_or_multipanel" in own))
               for screen, own in zip(clips.decision_reasons.fillna(""), rule)]
    clips = clips.assign(decision=clips.decision.where(rule == "", "exclude"),
                         decision_reasons=pd.Series(reasons, index=clips.index, dtype=object))
    return clips.assign(exclude=clips.decision != "keep")


def pilot_categories(short, n_studies, vendors=()):
    """The pilot's categories in priority order: the first `n_studies` report categories, then the vendor slots the
    shortlist holds (`vendor:<vendor>`) of `vendors` (none when empty: the other slots' studies are screened, not
    taken)."""
    slots = sorted({c for c in short.category if str(c).startswith(VENDOR_SLOT)})
    unknown = sorted(set(vendors) - {s[len(VENDOR_SLOT):] for s in slots})
    if unknown:
        raise SystemExit(f"--vendor-slots names {unknown}, which the shortlist has no slot for; its slots: "
                         f"{[s[len(VENDOR_SLOT):] for s in slots]}.")
    return [*CATEGORIES[:n_studies], *(s for s in slots if s[len(VENDOR_SLOT):] in vendors)]


def choose_studies(short, clips, min_clips, families, n_studies, vendors=()):
    """One study for each of the pilot's categories (`pilot_categories`: the first `n_studies`, then the vendor
    slots of `vendors`), each meeting the criteria on its kept clips (`exclude` false), all of different patients:
    of every such combination of shortlisted studies, the best-ranked in the categories' priority order (the best
    rank in the first category, then in the second, ...). The combinations are searched, not each category's best
    taken in turn: the best study of one category may be the only patient another category could use, while its
    second best leaves a pilot. Empty when no combination exists (`qualifying_studies` says how many studies each
    category has)."""
    categories = pilot_categories(short, n_studies, vendors)
    pools = [pool.to_dict("records")
             for pool in qualifying_studies(short, clips, min_clips, families, n_studies, vendors).values()]
    for combination in itertools.product(*pools):  # in rank order: the first of distinct patients is the best.
        if len({row["patient_id"] for row in combination}) == len(combination):
            return pd.DataFrame([{"category": c, **{k: row[k] for k in ("study_id", "patient_id", "accession")}}
                                 for c, row in zip(categories, combination)],
                                columns=["category", "study_id", "patient_id", "accession"])
    return pd.DataFrame(columns=["category", "study_id", "patient_id", "accession"])


def qualifying_studies(short, clips, min_clips, families, n_studies, vendors=()):
    """Per pilot category (`pilot_categories`), its shortlisted studies that meet the criteria on their kept clips,
    at their resolved mode, in rank order."""
    ok = study_ok(clips[~clips.exclude], min_clips, families)
    pools = {}
    for c in pilot_categories(short, n_studies, vendors):
        pool = short[short.category == c]
        pool = pool.assign(rank=pool["rank"].astype(int)).sort_values("rank")
        pools[c] = pool[pool.study_id.map(ok).fillna(False).astype(bool)]
    return pools


def select(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    clear_outputs(args.out_dir, SELECTION_OUTPUTS + CANDIDATE_OUTPUTS)  # an earlier selection is stale from here on.
    # The chain checked and the files then used are the same reads (Inputs), checked again before the record.
    inputs = clip_screen.Inputs(__file__)
    chain = screen_chain(args.out_dir, args.flags, inputs)
    listed = [os.path.join(args.out_dir, n)
              for n in ("shortlist.csv", "shortlist_clips.csv", "shortlist.json", "removed_clips.csv")]
    short = inputs.csv(listed[0], dtype=str)
    clips = inputs.csv(listed[1], dtype={"patient_id": str, "study_id": str})
    if "global_key" not in clips or "mode_source" not in clips:
        raise SystemExit("shortlist_clips.csv has no global_key or no mode_source (an older shortlist): rerun "
                         "shortlist.")
    record = inputs.json(listed[2])
    if os.path.abspath(listed[3]) not in record["outputs"] or "captioned" not in record["criteria"]:
        raise SystemExit("The shortlist lists no removed_clips.csv or was drawn before V1's rule (B-mode clips of "
                         "classified views only): an older shortlist, whose clips would reach the pilot in other "
                         "modes and views. Rerun shortlist, then the screen on its lists.")
    if record["criteria"].get("link_policy") != LINK_POLICY:
        raise SystemExit("The shortlist predates the study identity and report ambiguity checks: rerun shortlist, "
                         "then the screen on its lists.")
    removed = inputs.csv(listed[3], dtype={"patient_id": str, "study_id": str, "reasons": str})
    min_clips, families = record["criteria"]["min_clips"], record["criteria"]["required_view_families"]
    flags = inputs.parquet(args.flags)
    if not {"video_sha256", "decision", "pixel_mode"} <= set(flags):
        raise SystemExit("The flags do not record the video bytes they screened, the screen's decisions or its "
                         "reading of the mode (an older features or flag run): rerun clip_screen.py features and flag.")
    flags = flags[["video_path", "decision", "decision_reasons", "pixel_mode", *MEDIA_COLUMNS]]
    flag_info = inputs.json(args.flags + ".info.json")
    read = resolve_modes(screened_clips(clips, flags))
    clips = with_pilot_rule(read)
    categories = pilot_categories(short, args.n_studies, args.vendor_slots)
    chosen = choose_studies(short, clips, min_clips, families, args.n_studies, args.vendor_slots)
    if len(chosen) < len(categories):  # before any output: a pilot short of its studies is no completed selection.
        counts = {c: len(p) for c, p in
                  qualifying_studies(short, clips, min_clips, families, args.n_studies, args.vendor_slots).items()}
        reason = (f"none for {', '.join(c for c, n in counts.items() if not n)}" if not all(counts.values()) else
                  "but no combination of one study per category has different patients")
        decisions = {k: int(n) for k, n in clips.decision.value_counts().items()}
        raise SystemExit(f"No {len(categories)} studies of different patients, one per category and vendor slot, "
                         f"meet the criteria on the clips the screen keeps (qualifying studies per category: {counts}; "
                         f"{reason}; the shortlisted clips' decisions: {decisions}): no selection was written. "
                         "Shortlist more studies per category or vendor (shortlist --per-category, --vendor-studies), "
                         "or look into the uncertain and excluded clips (flags.parquet.info.json reasons).")
    screened = clips[clips.study_id.isin(set(chosen.study_id))]
    # The flags describe the videos and DICOM headers as features read them: a clip replaced in place since (re-masked,
    # re-exported) would reach the reviewers unscreened.
    changed = clip_screen.changed_media(screened, os.path.join(args.out_dir, "media.log"))
    if changed:
        raise SystemExit(f"{len(changed)} clips of the chosen studies no longer hold the video or DICOM regions the "
                         "screen read (replaced in place since features): no selection was written. Rerun "
                         "clip_screen.py features and flag on this shortlist.")
    outputs = [os.path.join(args.out_dir, n) for n in SELECTION_OUTPUTS[:5]]
    chosen.to_csv(outputs[0], index=False)
    pilot = screened[~screened.exclude]
    pilot[[*CLIP_COLUMNS, "header_acq_category", "video_sha256"]].to_csv(outputs[1], index=False)
    # Uncertain clips of every shortlisted study, for a separate look: left out of the pilot, not judged invalid.
    uncertain = clips[clips.decision == "uncertain"]
    uncertain[[*CLIP_COLUMNS, "decision_reasons"]].to_csv(outputs[2], index=False)
    # Every clip of every shortlisted study that the pilot leaves out, with its reasons: removed before the screen
    # (shortlist), excluded or uncertain after it (screen), or clean but belonging to an unchosen backup study.
    not_kept = pd.concat([
        removed.assign(step="shortlist", decision="removed", header_acq_category=removed.acq_category),
        clips[clips.exclude].assign(step="screen", reasons=lambda rows: rows.decision_reasons),
        clips[~clips.exclude & ~clips.study_id.isin(set(chosen.study_id))].assign(
            step="selection", decision="not_chosen", reasons="study_not_chosen")], ignore_index=True)
    not_kept = not_kept.assign(in_pilot=not_kept.study_id.isin(set(chosen.study_id)))
    not_kept[["step", "decision", "reasons", "in_pilot", *CLIP_COLUMNS, "header_acq_category"]].to_csv(
        outputs[3], index=False)
    # The shortlist's funnel (an older shortlist recorded none), then the screen, the pilot rule and the choice.
    by_screen = read.decision == "keep"
    by_screen_reasons, by_rule_reasons = read.decision_reasons.fillna("")[~by_screen], pilot_rule(read)[by_screen]
    slots_filled = [c[len(VENDOR_SLOT):] for c in categories if c.startswith(VENDOR_SLOT)]
    funnel = record.get("funnel", []) + [
        funnel_row("select", f"screen: {flag_info['thresholds']['policy']} policy, "
                   + ("calibrated thresholds" if flag_info.get("calibration") else "provisional thresholds"),
                   read[by_screen],
                   # The first reason decides (exclusions come first): 'exclude: still'.
                   reason_counts(read.decision[~by_screen] + ": " + by_screen_reasons), "clips",
                   each_reason_counts(by_screen_reasons)),
        funnel_row("select", "pilot rule: B-mode as read, no GE 3D or multi-panel image code", clips[~clips.exclude],
                   reason_counts(by_rule_reasons), "clips", each_reason_counts(by_rule_reasons)),
        funnel_row("select", f"study choice (--n-studies {args.n_studies}, vendor slots "
                   f"{', '.join(slots_filled) or 'none'})", screened[~screened.exclude],
                   _count("not_chosen", clips[~clips.exclude].study_id.nunique() - len(chosen)), "studies",
                   ", ".join(chosen.category))]
    pd.DataFrame(funnel, columns=list(FUNNEL_COLUMNS)).to_csv(outputs[4], index=False)
    removed_pilot = removed[removed.study_id.isin(set(chosen.study_id))]

    summary = {
        "created_utc": _now(), "code_commit": git_commit(),
        "code_sha256": clip_screen.code_sha256(__file__, clip_screen.__file__),
        "flags": os.path.abspath(args.flags), "inputs": chain, "outputs": file_hashes(*outputs),
        "screen_thresholds": flag_info["thresholds"], "screen_calibration": flag_info.get("calibration"),
        # Left out after the screen, on its readings (decision exclude, step screen in not_kept_clips.csv).
        "pilot_rule": {"captioned_mode": CAPTIONED_MODE,
                       "3d_or_multipanel": {"vendor": MULTIPANEL_3D_VENDOR, "image_codes": MULTIPANEL_3D_CODES}},
        "link_policy": LINK_POLICY,
        "n_studies": args.n_studies, "studies": int(len(chosen)), "categories": chosen.category.tolist(),
        "vendor_slots": [c for c in categories if c.startswith(VENDOR_SLOT)],
        # The modes read from the pixels against the header's where it gives them (agreement, not labels), and the
        # pilot clips by where their mode comes from; the reviewers confirm or correct each mode.
        "pixel_mode_vs_header": flag_info.get("pixel_mode_vs_header"),
        "clips_by_mode_source": {k: int(n) for k, n in pilot.mode_source.value_counts().items()},
        "clips_kept": int(len(pilot)), "clips_removed_before_screen": int(len(removed_pilot)),
        "clips_excluded": int((screened.decision == "exclude").sum()),
        "clips_uncertain": int((screened.decision == "uncertain").sum()),
        "removed_reasons_in_pilot_studies": {k: int(n) for k, n in removed_pilot.reasons.str.split(";").explode()
                                             .value_counts().items()},
        "not_kept_clips_shortlisted": {k: int(n) for k, n in not_kept.decision.value_counts().items()},
        "screen_decisions_in_pilot_studies": {k: int(n) for k, n in screened.decision.value_counts().items()},
        "screen_reasons_in_pilot_studies": {k: int(n) for k, n in screened.decision_reasons.str.split(";").explode()
                                            .replace("", np.nan).dropna().value_counts().items()},
        "uncertain_clips_shortlisted": int(len(uncertain)),
        "clips_by_acquisition": {k: int(n) for k, n in pilot.acq_category.value_counts().items()},
        "clips_by_form": {k: int(n) for k, n in pilot.form.value_counts().items()},
        "clips_by_view": {k: int(n) for k, n in pilot.view.fillna("<none>").value_counts().items()},
    }
    inputs.check(*outputs)
    _write_json(os.path.join(args.out_dir, "selection.json"), summary)
    lines = ["# RADAR pilot selection", "", "Aggregate only.", "",
             f"- Studies: {summary['studies']} ({', '.join(summary['categories'])})",
             f"- Clips kept: {summary['clips_kept']}; removed before the screen: "
             f"{summary['clips_removed_before_screen']}; excluded by the screen: {summary['clips_excluded']}; "
             f"uncertain (left out, listed in uncertain_clips.csv): {summary['clips_uncertain']}",
             f"- Every clip the shortlisted studies leave out, with its reasons: not_kept_clips.csv "
             f"({summary['not_kept_clips_shortlisted']})",
             f"- Reasons removed before the screen in the pilot studies: {summary['removed_reasons_in_pilot_studies']}",
             f"- Screen reasons in the pilot studies: {summary['screen_reasons_in_pilot_studies']}",
             f"- Screen thresholds: {summary['screen_thresholds']}, "
             + (f"calibrated ({summary['screen_calibration']})" if summary["screen_calibration"]
                else "provisional (no labelled calibration yet)"),
             f"- Clips by form: {summary['clips_by_form']}",
             f"- Clips by acquisition: {summary['clips_by_acquisition']} (mode from {summary['clips_by_mode_source']})",
             f"- Clips by view: {summary['clips_by_view']}",
             "", "## Funnel (funnel.csv)", "",
             "What each step leaves, and what it leaves out by reason (counted in clips or studies), each clip or "
             "study under its first reason in the rule's order; the note counts every reason where one has several.",
             "",
             "| Step | Videos | Studies | Patients | Left out | Note |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['step']} | {r['videos']:,} | {r['studies']:,} | {r['patients']:,} | "
              + (f"{r['left_out']} ({r['left_out_unit']})" if r["left_out"] else "") + f" | {r['note']} |"
              for r in funnel]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[pilot]  {summary['studies']} studies ({', '.join(summary['categories'])}), {summary['clips_kept']} "
          f"clips kept, {summary['clips_removed_before_screen']} removed before the screen, "
          f"{summary['clips_excluded']} excluded, {summary['clips_uncertain']} uncertain; "
          f"{summary['screen_reasons_in_pilot_studies']}.")
    print_funnel([r for r in funnel if r["stage"] == "select"])
    if not summary["screen_calibration"]:
        print("[pilot]  WARNING: the screen's thresholds are provisional (no labelled calibration yet), so its "
              "exclusions are not validated; candidates refuses this selection unless --provisional-screen.",
              file=sys.stderr)


def screen_chain(out_dir, flags_path, inputs=None):
    """Every file the flags stand on, as {path: SHA-256}, once the chain back to this shortlist is checked: the
    completed shortlist unchanged (its outputs, reference_clips.csv included), and the flags screened from its
    shortlist_clips.csv and reference_clips.csv as they are now (`clip_screen.screen_sources`: the flags and the
    features files behind them unchanged, each with the record of a completed run). Flags of an earlier shortlist,
    whose clips or DICOM paths differed, are refused even when they cover the same videos. `select` records the
    files, and `candidates` checks them and the chain again. Every file is read into `inputs` (clip_screen.Inputs),
    so a step that then uses them uses what was checked."""
    inputs = inputs or clip_screen.Inputs()
    record = os.path.join(out_dir, "shortlist.json")
    if not os.path.exists(record):
        raise SystemExit("No completed shortlist in --out-dir (shortlist.json is written last): run shortlist.")
    outputs = inputs.json(record)["outputs"]
    unchanged(outputs, "shortlist", inputs)
    sources = clip_screen.screen_sources(flags_path, inputs)
    lists = {"features": "shortlist_clips.csv", "reference": "reference_clips.csv"}
    stale = [name for role, name in lists.items()
             if sources[role]["clips"] != inputs.digest(os.path.join(out_dir, name))]
    if stale:
        raise SystemExit(f"The flags were not screened from this shortlist's {' and '.join(stale)} (another version "
                         "of it, or another list): run clip_screen.py features on them, then flag, again.")
    files = [record, flags_path, flags_path + ".info.json"]
    for source in sources.values():
        files += [source["path"], source["path"] + ".info.json"]
    return {**outputs, **file_hashes(*files, inputs=inputs)}


def global_key(secret, video_path):
    """Opaque, stable Labelbox global key: says nothing about the patient, study or path."""
    digest = hmac.new(secret, video_path.encode(), hashlib.sha256).hexdigest()
    return f"radar-pilot-{digest[:24]}"


def load_secret(out_dir):
    path = os.path.join(out_dir, SECRET_NAME)
    if not os.path.exists(path):
        with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip().encode()


def report_record(row):
    """One report row as the plain, JSON-ready dict the candidate builder reads: missing values become None and
    numpy scalars plain Python values."""
    out = {}
    for k, v in row.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = None if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)) else v
    return out


def file_hashes(*paths, inputs=None):
    """{path: SHA-256}: of the files as `inputs` (clip_screen.Inputs) read them, else as they are now."""
    return {os.path.abspath(p): inputs.digest(p) if inputs else sha256(p) for p in paths}


def unchanged(hashes, step, inputs=None):
    """Stop unless every file of `hashes` (file_hashes) is where `step` left it, unchanged (as read into `inputs`,
    when given)."""
    stale = [p for p, digest in hashes.items()
             if not os.path.exists(p) or (inputs.digest(p) if inputs else sha256(p)) != digest]
    if stale:
        raise SystemExit(f"{', '.join(os.path.basename(p) for p in stale)} changed or moved since {step}: rerun "
                         f"{step}.")


def clear_outputs(out_dir, names):
    clip_screen.clear_outputs(*(os.path.join(out_dir, n) for n in names))


def verify_selection(out_dir, inputs=None):
    """The SHA-256 of selection.json, after checking that the files of that selection are unchanged (the pilot
    studies and clips, and every file the flags stand on: the shortlist's files, the flags and the features files
    behind them, with their records) and that the screening chain still holds (`screen_chain`), each file read
    into `inputs` (clip_screen.Inputs)."""
    inputs = inputs or clip_screen.Inputs()
    path = os.path.join(out_dir, "selection.json")
    if not os.path.exists(path):
        raise SystemExit("No selection in --out-dir: the last select failed or never ran. Run select first.")
    selection = inputs.json(path)
    unchanged({**selection["inputs"], **selection["outputs"]}, "select", inputs)
    if not selection.get("studies") or selection["studies"] < selection.get("n_studies", math.inf):
        raise SystemExit("The selection holds fewer studies than it was asked for, or records no number asked for "
                         "(an older select run): rerun select.")
    if "pilot_rule" not in selection:
        raise SystemExit("The selection was made before V1's rule (B-mode clips of classified views only): rerun "
                         "shortlist, the screen and select.")
    if selection.get("link_policy") != LINK_POLICY:
        raise SystemExit("The selection predates the study identity and report ambiguity checks: rerun shortlist, "
                         "the screen and select.")
    if "flags" not in selection or screen_chain(out_dir, selection["flags"], inputs) != selection["inputs"]:
        raise SystemExit("The selection does not record the screening chain it was made from, or that chain "
                         "changed since: rerun select.")
    return inputs.digest(path)


def candidates(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    clear_outputs(args.out_dir, CANDIDATE_OUTPUTS)  # a refused or failed run leaves no earlier candidates.
    # The selection checked and the files then used are the same reads (Inputs), checked again before the record.
    column_review = os.path.join(REPO_ROOT, "tools", "labelbox", "column_review.py")
    inputs = clip_screen.Inputs(__file__, column_review)
    selection = verify_selection(args.out_dir, inputs)  # shortlist.json among its files: the report hash holds.
    calibration = inputs.json(os.path.join(args.out_dir, "selection.json")).get("screen_calibration")
    if not calibration and not args.provisional_screen:
        raise SystemExit("The selection's screen ran at provisional thresholds (no labelled calibration yet): its "
                         "automatic exclusions are not validated. Calibrate the screen first (pilot_selection.md), or "
                         "pass --provisional-screen for a trial run.")
    chosen_with = inputs.json(os.path.join(args.out_dir, "shortlist.json"))["inputs"]["reports"]
    reports_sha256 = inputs.digest(args.reports)
    if reports_sha256 != chosen_with:
        raise SystemExit("--reports is not the report file shortlist chose the studies with (its SHA-256 differs): "
                         "prefills from other report text would not match the studies' categories. Pass that file, "
                         "or rerun from shortlist.")
    sys.path.insert(0, REPO_ROOT)
    from tools.labelbox.column_review import SPEC, build_candidate

    # The review spec, read once: every candidate is built from it, and the summary names it.
    spec = inputs.json(SPEC)

    pilot = inputs.csv(os.path.join(args.out_dir, "pilot_clips.csv"), dtype={"patient_id": str, "study_id": str})
    changed = clip_screen.changed_media(pilot, os.path.join(args.out_dir, "media.log"), dicom=False)
    if changed:
        raise SystemExit(f"{len(changed)} pilot clips no longer hold the video bytes the screen read (replaced in "
                         "place since): rerun clip_screen.py features and flag, then select.")
    studies = inputs.csv(os.path.join(args.out_dir, "pilot_studies.csv"), dtype=str)
    reports = inputs.parquet(args.reports)
    reports["AccessionNumber"] = normalize_id(reports.AccessionNumber)
    reports = reports[reports.AccessionNumber.isin(set(studies.accession))].drop_duplicates()
    reports = reports.set_index("AccessionNumber")
    if reports.index.duplicated().any() or set(studies.accession) - set(reports.index):
        raise SystemExit("Every pilot study needs exactly one report (rows repeated exactly count once).")
    accession = studies.set_index("study_id").accession

    out, other, keys = [], [], []
    for _, clip in pilot.iterrows():
        key = clip.global_key  # made by shortlist: the same key named the clip on the screen's sheets.
        keys.append({"global_key": key, "form": clip.form, "study_id": clip.study_id, "video_path": clip.video_path,
                     "video_sha256": clip.video_sha256})  # the bytes screened: the upload checks the file holds them.
        if clip.form == VIEW_TYPE_FORM:
            other.append({"global_key": key})  # no report and no prefill: the reviewer gives view and type.
            continue
        report = report_record(reports.loc[accession[clip.study_id]].to_dict())
        out.append(build_candidate(report, clip.view, clip.acq_category, global_key=key, input_kind="video",
                                   input_provenance_verified=False, spec=spec))
    pd.DataFrame(other, columns=["global_key"]).to_csv(os.path.join(args.out_dir, "view_type_items.csv"), index=False)
    path = os.path.join(args.out_dir, "candidates.json")
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    keys_path = os.path.join(args.out_dir, "global_keys.csv")
    with open(os.open(keys_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        pd.DataFrame(keys).to_csv(f, index=False)
    n_columns = [len(c["columns"]) for c in out]
    code = clip_screen.code_sha256(__file__, column_review)
    inputs.check(*(os.path.join(args.out_dir, n) for n in CANDIDATE_OUTPUTS))
    summary = {"created_utc": _now(), "code_commit": git_commit(), "code_sha256": code, "selection": selection,
               "reports": reports_sha256, "spec": {os.path.relpath(SPEC, REPO_ROOT): inputs.digest(SPEC)},
               "screen_calibration": calibration,
               "provisional_screen": not calibration, "candidates": len(out),
               "view_type_items": len(other),
               # A candidate's modality is the header's or, for another vendor, the screen's pixel reading, which
               # the reviewer confirms or corrects like the view.
               "candidates_by_mode_source": {k: int(n) for k, n in
                                             pilot[pilot.form == CAPTION_FORM].mode_source.value_counts().items()},
               "prefilled_columns": int(sum(n_columns)),
               "candidates_without_prefill": int(sum(n == 0 for n in n_columns))}
    _write_json(os.path.join(args.out_dir, "candidates_summary.json"), summary)
    print(f"[pilot]  {len(out)} caption candidates, {summary['prefilled_columns']} prefilled columns, "
          f"{summary['candidates_without_prefill']} without any prefill; {len(other)} view and type items.")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("shortlist")
    s.add_argument("--videos", required=True, help="videos.parquet of the archive manifests (build_manifests.py: "
                                                   "TTE exams only; split, view, acquisition and frame count per "
                                                   "clip).")
    s.add_argument("--ef-labels", required=True, help="study_labels_ef.parquet (study_id, accession, ef): EF for the "
                                                     "categories, where it belongs to the study's linked report.")
    s.add_argument("--rvsp-labels", required=True, help="study_labels_rvsp.parquet (study_id, accession, rvsp): RVSP, "
                                                       "as EF.")
    s.add_argument("--reports", required=True, help="cleaned DeepECHO report parquet.")
    s.add_argument("--pacs", required=True, help="PACS index CSV (AccessionNumber, StudyInstanceUID, PatientID, "
                                                "StudyDate) linking each report to its study.")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--seed", type=int, default=20261014)
    s.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    s.add_argument("--vendor", choices=sorted(VENDOR_GROUPS),
                   help="the vendors drawn from (categories and vendor slots): GE, GE's three vendor names alone; "
                        "'GE Vingmed Ultrasound', that vendor alone; left out, any vendor, each clip's mode read from "
                        "its header or its pixels as its vendor allows.")
    s.add_argument("--vendor-studies", nargs="*", default=[], metavar="VENDOR=N",
                   help="vendor slots: N studies of each vendor named, of --vendor's when given, outside the "
                        "categories and --years (for example 'Philips Medical Systems=1'); screened, and taken by "
                        "select only when --vendor-slots names them.")
    s.add_argument("--exclude-patients", nargs="*", default=[],
                   help="CSV files with patient_id (a validation list) whose patients are drawn nowhere: no study, "
                        "clip or reference clip of theirs.")
    s.add_argument("--min-clips", type=int, default=20, help="kept B-mode clips of classified views a study needs.")
    s.add_argument("--view-families", nargs="*", default=list(REQUIRED_VIEW_FAMILIES), metavar="FAMILY",
                   help="view families a study needs among those clips (a view's name before its first '_': A4C_ZOOM "
                        "is A4C); given without a family: none.")
    s.add_argument("--per-category", type=int, default=5, help="studies shortlisted per category.")
    s.add_argument("--reference-per-layout", type=int, default=200,
                   help="reference clips per (vendor, model, acquisition) for the screen's layout baselines, from "
                        "eligible clips of studies not shortlisted.")
    t = sub.add_parser("select")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--flags", required=True, help="clip_screen.py flag output for shortlist_clips.csv.")
    t.add_argument("--n-studies", type=int, default=3, choices=(2, 3),
                   help="report categories, in priority order.")
    t.add_argument("--vendor-slots", nargs="*", default=[], metavar="VENDOR",
                   help="the shortlist's vendor slots to fill, one study each (default: none; the slots' studies "
                        "are screened, not taken).")
    c = sub.add_parser("candidates")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--reports", required=True)
    c.add_argument("--provisional-screen", action="store_true",
                   help="accept a selection whose screen ran at provisional thresholds: a trial run, not for upload.")
    args = p.parse_args(argv)
    umask = os.umask(0o077)  # every file written is the owner's only.
    try:
        {"shortlist": shortlist, "select": select, "candidates": candidates}[args.command](args)
    finally:
        os.umask(umask)


if __name__ == "__main__":
    main()

