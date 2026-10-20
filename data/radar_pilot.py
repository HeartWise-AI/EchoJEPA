# data/radar_pilot.py

"""Choose the 2-3 report-linked training studies of the RADAR caption pilot, keep only the clips reviewers
should see, and build their column review v2 candidates.

Every valid video of a pilot study is reviewed, in one of two Labelbox forms (`form` column):

- `column_review`: a classified view in B-mode or color Doppler gets the column review v2 and its caption;
- `view_type_review`: a video the view classifier labels `OTHER`, or does not classify, gets the view and type
  review and no caption (its anatomy is unknown, so no caption can be routed to it).

Clips are removed in two places before any upload:

- **metadata rule** (`shortlist`): for a classified view, an acquisition mode other than B-mode or color Doppler,
  or one not read from a validated header. Single-frame DICOM objects (still B-mode measurement frames, spectral
  Doppler stills, report pages) have no video and never appear. No clip is removed for being short.
- **pixel rules** (`data/clip_screen.py` on the shortlisted clips): still clips and visible measurements.

Studies are chosen on their classified clips left after both, so every pilot study still has the standard views.

Steps, every output in --out-dir, a folder outside the repository that only its owner can read (identities and
report text stay there). A step first removes its earlier outputs and those of the steps after it, and refuses
inputs that changed since the step that made them finished:
    shortlist   training-split studies with a linked report, per category, from the archive manifest (TTE exams
                only, from build_manifests.py); their clips after the metadata rules, each with its opaque global
                key (the sheet's and Labelbox's name for it), written to shortlist_clips.csv for the screen; and
                reference_clips.csv, a random sample of the same layouts from other studies, from which the screen
                learns where each layout normally has ink.
    select      after `clip_screen.py features` on shortlist_clips.csv and reference_clips.csv and `flag`: flags
                screened from other lists (an earlier shortlist) are refused. Remove every clip the screen's
                policy excludes (unreadable, a header without image regions, still; no one is asked), then take per
                category the best-ranked study that meets the criteria on its kept clips, one study per patient.
    candidates  column review v2 candidates for the pilot's classified clips and the list of its view and type
                review items, from a selection whose files are unchanged since select and the report file
                shortlist used. A screen at provisional thresholds (not yet calibrated on labelled clips) is
                refused unless --provisional-screen marks the run as a trial.

Example:
    python data/radar_pilot.py shortlist --videos <manifests>/videos.parquet --ef-labels <study_labels_ef.parquet> \\
        --rvsp-labels <study_labels_rvsp.parquet> --reports <df_report_clean.parquet> --out-dir <pilot_dir>
    python data/clip_screen.py features --clips <pilot_dir>/shortlist_clips.csv \\
        --out <pilot_dir>/screen/features.parquet
    python data/clip_screen.py features --clips <pilot_dir>/reference_clips.csv \\
        --out <pilot_dir>/screen/reference.parquet
    python data/clip_screen.py flag --features <pilot_dir>/screen/features.parquet \\
        --reference <pilot_dir>/screen/reference.parquet --out <pilot_dir>/screen/flags.parquet
    python data/radar_pilot.py select --out-dir <pilot_dir> --flags <pilot_dir>/screen/flags.parquet --n-studies 3
    python data/radar_pilot.py candidates --out-dir <pilot_dir> --reports <df_report_clean.parquet> \\
        [--provisional-screen]

Console output and summary.md are aggregate only: `select` and `candidates` check that the clips still hold the
media the screen read, and a failure to read one (its message names the clip's path) goes to media.log in
--out-dir.
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

import numpy as np
import pandas as pd

import clip_screen
from build_manifests import REPO_ROOT, ensure_outside_repo, git_commit, normalize_id, sha256

# Category order is the priority order: `--n-studies 2` takes the first two.
CATEGORIES = ("normal", "lv_dysfunction", "valve_disease")
REQUIRED_VIEW_FAMILIES = ("A4C", "A2C", "PLAX", "PSAX")
REVIEWED_ACQUISITIONS = ("bmode", "color_flow")
VALVE_FIELDS = ("Mitral valve findings", "Aortic valve findings", "Tricuspid valve findings")
REPORT_NUMBERS = ("AV Mean Gradient",)
CLIP_COLUMNS = ("patient_id", "study_id", "view", "video_path", "dicom_path", "vendor", "model", "acq_category",
                "n_frames", "form", "global_key")
LAYOUT_COLUMNS = ("vendor", "model", "acq_category")  # the screen's layouts before the frame size is known.
# What the screen read of each clip's media (clip_screen.changed_media checks the files still hold it).
MEDIA_COLUMNS = ("error", "video_sha256", "dicom_status", "dicom_regions", "height", "width")
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
DERIVED_COLUMNS = ("form", "global_key")  # computed here, not read from the videos table.
CAPTION_FORM, VIEW_TYPE_FORM = "column_review", "view_type_review"
SECRET_NAME = ".global_key_secret"
# What each step writes in --out-dir. A step first removes its own outputs and those of the steps after it, so a
# run that stops leaves nothing a later step could take for its result.
SHORTLIST_OUTPUTS = ("shortlist.csv", "shortlist_clips.csv", "reference_clips.csv", "shortlist.json")
SELECTION_OUTPUTS = ("pilot_studies.csv", "pilot_clips.csv", "selection.json", "summary.md")
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
    """The Labelbox form each clip goes to: the caption review for a classified view, the view and type review for
    `OTHER` or no view prediction."""
    unclassified = clips.view.fillna("").isin(["", "OTHER"])
    return pd.Series(np.where(unclassified, VIEW_TYPE_FORM, CAPTION_FORM), index=clips.index)


def clip_rule(clips):
    """Why each clip fails the metadata rule ('' when it passes). The acquisition rule is for the caption review
    only: the view and type review asks the reviewer for the acquisition mode."""
    validated = clips.acq_vendor_validated.fillna(False).astype(bool) & (clips.acq_confidence == "header")
    captioned = review_form(clips) == CAPTION_FORM
    failed = captioned & (~clips.acq_category.isin(REVIEWED_ACQUISITIONS) | ~validated)
    return pd.Series(np.where(failed, "acquisition", ""), index=clips.index)


def study_ok(clips, min_clips):
    """Per study, on its kept clips of classified views: enough of them, every required view family, and one family
    in both B-mode and color."""
    clips = clips[review_form(clips) == CAPTION_FORM]
    family = view_family(clips.view)
    per = clips.assign(family=family, bmode=clips.acq_category == "bmode", color=clips.acq_category == "color_flow")
    n = per.groupby("study_id").size()
    families = per.groupby("study_id").family.agg(frozenset)
    modes = per.groupby(["study_id", "family"])[["bmode", "color"]].any()
    paired = (modes.bmode & modes.color).groupby("study_id").any()
    ok = (n >= min_clips) & families.map(lambda f: all(x in f for x in REQUIRED_VIEW_FAMILIES)) & paired
    return ok.reindex(n.index).fillna(False)


def shortlist(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    clear_outputs(args.out_dir, SHORTLIST_OUTPUTS + SELECTION_OUTPUTS + CANDIDATE_OUTPUTS)
    rng = np.random.default_rng(args.seed)
    # Every input as read (Inputs): the record names those, and is written only if none changed meanwhile.
    inputs = clip_screen.Inputs(__file__)

    cols = [c for c in CLIP_COLUMNS if c not in DERIVED_COLUMNS] + ["split", "study_date", "acq_confidence",
                                                                     "acq_vendor_validated"]
    built_at = manifest_build(args.videos, cols, inputs)
    v = inputs.parquet(args.videos, columns=cols)
    v = v[v.split == "train"].copy()  # the manifest holds TTE exams only.
    v["patient_id"], v["study_id"] = normalize_id(v.patient_id), normalize_id(v.study_id)
    v["year"] = pd.to_numeric(v.study_date.astype("string").str[:4], errors="coerce")
    v = v[v.year.isin(args.years)]
    if args.vendor:
        v = v[v.vendor == args.vendor]
    v["form"] = review_form(v)
    v["clip_rule"] = clip_rule(v)
    kept = v[v.clip_rule == ""]

    ef = inputs.parquet(args.ef_labels, columns=["study_id", "accession", "ef"])
    ef["study_id"], ef["accession"] = normalize_id(ef.study_id), normalize_id(ef.accession)
    rvsp = inputs.parquet(args.rvsp_labels, columns=["study_id", "rvsp"])
    rvsp["study_id"] = normalize_id(rvsp.study_id)
    reports = inputs.parquet(args.reports, columns=["AccessionNumber", *VALVE_FIELDS, *REPORT_NUMBERS])
    reports["AccessionNumber"] = normalize_id(reports.AccessionNumber)
    reports = reports.drop_duplicates("AccessionNumber", keep=False)
    s = (ef.merge(rvsp, on="study_id", how="left")
         .merge(reports, left_on="accession", right_on="AccessionNumber", how="inner"))
    s["av_mean_gradient"] = pd.to_numeric(s["AV Mean Gradient"], errors="coerce")
    ok = study_ok(kept, args.min_clips)
    s = s[s.study_id.isin(ok[ok].index)].sort_values("study_id").reset_index(drop=True)
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
    outputs = [os.path.join(args.out_dir, n) for n in SHORTLIST_OUTPUTS[:3]]
    short.to_csv(outputs[0], index=False)
    clips = kept[kept.study_id.isin(set(short.study_id))]
    secret = load_secret(args.out_dir)
    clips = clips.assign(global_key=[global_key(secret, p) for p in clips.video_path])
    clips[list(CLIP_COLUMNS)].to_csv(outputs[1], index=False)
    # Reference clips for the screen's layout baselines: other studies, the layouts the shortlist has.
    pool = kept[~kept.study_id.isin(set(short.study_id))]
    pool = pool.merge(clips[list(LAYOUT_COLUMNS)].drop_duplicates(), on=list(LAYOUT_COLUMNS))
    reference = (pool.sample(frac=1.0, random_state=args.seed)
                 .groupby(list(LAYOUT_COLUMNS), dropna=False).head(args.reference_per_layout))
    reference = reference.assign(global_key=[global_key(secret, p) for p in reference.video_path])
    reference[list(CLIP_COLUMNS)].to_csv(outputs[2], index=False)

    removed = v[v.study_id.isin(set(short.study_id)) & (v.clip_rule != "")]
    info = {
        "created_utc": _now(), "code_commit": git_commit(), "code_sha256": clip_screen.code_sha256(__file__),
        "seed": args.seed,
        "criteria": {"split": "train", "years": args.years, "vendor": args.vendor,
                     "min_clips": args.min_clips,
                     "required_view_families": REQUIRED_VIEW_FAMILIES,
                     "categories": {"normal": "EF >= 55, AV mean gradient < 20, RVSP < 35 (all present), no "
                                              "moderate or worse statement in a valve field of the report",
                                    "lv_dysfunction": "EF <= 35",
                                    "valve_disease": "AV mean gradient >= 40, or moderate/severe MR or AR in the "
                                                     "report text, the grade in the same part as the MR or AR"}},
        "inputs": {k: inputs.digest(p) for k, p in (("videos", args.videos), ("ef_labels", args.ef_labels),
                                                    ("rvsp_labels", args.rvsp_labels), ("reports", args.reports))},
        "manifest_build_commit": built_at,
        "eligible_studies": int(len(s)), "pool_per_category": pools,
        "shortlisted_studies": int(short.study_id.nunique()), "clips_to_screen": int(len(clips)),
        "reference_clips": int(len(reference)), "reference_per_layout": args.reference_per_layout,
        "clips_removed_by_metadata_rule": {k: int(n) for k, n in removed.clip_rule.value_counts().items()},
        "outputs": file_hashes(*outputs),
    }
    inputs.check(*outputs)
    _write_json(os.path.join(args.out_dir, "shortlist.json"), info)  # last: it marks a completed shortlist.
    print(f"[pilot]  {len(s)} eligible training studies; pools {pools}; {info['shortlisted_studies']} shortlisted; "
          f"{len(clips)} clips to screen; removed by metadata {info['clips_removed_by_metadata_rule']}.")


def screened_clips(clips, flags):
    """The shortlisted clips with the screen's flags; `exclude` marks a clip the screen's policy excludes (its
    `exclude_reason`), which is removed."""
    missing = set(clips.video_path) - set(flags.video_path)
    if missing:
        raise SystemExit(f"{len(missing)} shortlisted clips were not screened; run clip_screen.py on "
                         "shortlist_clips.csv first.")
    screened = clips.merge(flags, on="video_path", how="left", validate="one_to_one")
    return screened.assign(exclude=screened.exclude.astype(bool))


def choose_studies(short, clips, min_clips, n_studies):
    """One study for each of the first `n_studies` categories, each meeting the criteria on its kept clips
    (`exclude` false), all of different patients: of every such combination of shortlisted studies, the best-ranked
    in the categories' priority order (the best rank in the first category, then in the second, ...). The
    combinations are searched, not each category's best taken in turn: the best study of one category may be the
    only patient another category could use, while its second best leaves a pilot. Empty when no combination
    exists (`qualifying_studies` says how many studies each category has)."""
    pools = [pool.to_dict("records") for pool in qualifying_studies(short, clips, min_clips, n_studies).values()]
    for combination in itertools.product(*pools):  # in rank order: the first of distinct patients is the best.
        if len({row["patient_id"] for row in combination}) == len(combination):
            return pd.DataFrame([{"category": c, **{k: row[k] for k in ("study_id", "patient_id", "accession")}}
                                 for c, row in zip(CATEGORIES, combination)],
                                columns=["category", "study_id", "patient_id", "accession"])
    return pd.DataFrame(columns=["category", "study_id", "patient_id", "accession"])


def qualifying_studies(short, clips, min_clips, n_studies):
    """Per category of the first `n_studies`, its shortlisted studies that meet the criteria on their kept clips,
    in rank order."""
    ok = study_ok(clips[~clips.exclude], min_clips)
    pools = {}
    for c in CATEGORIES[:n_studies]:
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
    listed = [os.path.join(args.out_dir, n) for n in ("shortlist.csv", "shortlist_clips.csv", "shortlist.json")]
    short = inputs.csv(listed[0], dtype=str)
    clips = inputs.csv(listed[1], dtype={"patient_id": str, "study_id": str})
    if "global_key" not in clips:
        raise SystemExit("shortlist_clips.csv has no global_key: rerun shortlist.")
    min_clips = inputs.json(listed[2])["criteria"]["min_clips"]
    flags = inputs.parquet(args.flags)
    if "video_sha256" not in flags:
        raise SystemExit("The flags do not record the video bytes they screened (an older features run): rerun "
                         "clip_screen.py features and flag.")
    flags = flags[["video_path", "exclude", "exclude_reason", *MEDIA_COLUMNS]]
    flag_info = inputs.json(args.flags + ".info.json")
    clips = screened_clips(clips, flags)
    chosen = choose_studies(short, clips, min_clips, args.n_studies)
    if len(chosen) < args.n_studies:  # before any output: a pilot short of its studies is no completed selection.
        counts = {c: len(p) for c, p in qualifying_studies(short, clips, min_clips, args.n_studies).items()}
        reason = (f"none for {', '.join(c for c, n in counts.items() if not n)}" if not all(counts.values()) else
                  "but no combination of one study per category has different patients")
        raise SystemExit(f"No {args.n_studies} studies of different patients, one per category, meet the criteria on "
                         f"the clips the screen keeps (qualifying studies per category: {counts}; {reason}): no "
                         "selection was written. Shortlist more studies per category (shortlist --per-category), or "
                         "check the screen's exclusions (flags.parquet.info.json counts, layouts without a "
                         "baseline).")
    screened = clips[clips.study_id.isin(set(chosen.study_id))]
    # The flags describe the videos and DICOM headers as features read them: a clip replaced in place since (re-masked,
    # re-exported) would reach the reviewers unscreened.
    changed = clip_screen.changed_media(screened, os.path.join(args.out_dir, "media.log"))
    if changed:
        raise SystemExit(f"{len(changed)} clips of the chosen studies no longer hold the video or DICOM regions the "
                         "screen read (replaced in place since features): no selection was written. Rerun "
                         "clip_screen.py features and flag on this shortlist.")
    outputs = [os.path.join(args.out_dir, n) for n in ("pilot_studies.csv", "pilot_clips.csv")]
    chosen.to_csv(outputs[0], index=False)
    pilot = screened[~screened.exclude]
    pilot[[*CLIP_COLUMNS, "video_sha256"]].to_csv(outputs[1], index=False)

    summary = {
        "created_utc": _now(), "code_commit": git_commit(),
        "code_sha256": clip_screen.code_sha256(__file__, clip_screen.__file__),
        "flags": os.path.abspath(args.flags), "inputs": chain, "outputs": file_hashes(*outputs),
        "screen_thresholds": flag_info["thresholds"], "screen_calibration": flag_info.get("calibration"),
        "n_studies": args.n_studies, "studies": int(len(chosen)), "categories": chosen.category.tolist(),
        "clips_kept": int(len(pilot)), "clips_excluded": int(screened.exclude.sum()),
        "screen_flags_in_pilot_studies": {k: int(n) for k, n in
                                          screened.exclude_reason.replace("", "kept").value_counts().items()},
        "clips_by_acquisition": {k: int(n) for k, n in pilot.acq_category.value_counts().items()},
        "clips_by_form": {k: int(n) for k, n in pilot.form.value_counts().items()},
        "clips_by_view": {k: int(n) for k, n in pilot.view.fillna("<none>").value_counts().items()},
    }
    inputs.check(*outputs)
    _write_json(os.path.join(args.out_dir, "selection.json"), summary)
    lines = ["# RADAR pilot selection", "", "Aggregate only.", "",
             f"- Studies: {summary['studies']} ({', '.join(summary['categories'])})",
             f"- Clips kept: {summary['clips_kept']}; excluded: {summary['clips_excluded']}",
             f"- Screen flags in the pilot studies: {summary['screen_flags_in_pilot_studies']}",
             f"- Screen thresholds: {summary['screen_thresholds']}, "
             + (f"calibrated ({summary['screen_calibration']})" if summary["screen_calibration"]
                else "provisional (no labelled calibration yet)"),
             f"- Clips by form: {summary['clips_by_form']}",
             f"- Clips by acquisition: {summary['clips_by_acquisition']}",
             f"- Clips by view: {summary['clips_by_view']}"]
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[pilot]  {summary['studies']} studies ({', '.join(summary['categories'])}), {summary['clips_kept']} "
          f"clips kept, {summary['clips_excluded']} excluded; {summary['screen_flags_in_pilot_studies']}.")
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
    reports = reports[reports.AccessionNumber.isin(set(studies.accession))].set_index("AccessionNumber")
    if reports.index.duplicated().any() or set(studies.accession) - set(reports.index):
        raise SystemExit("Every pilot study needs exactly one report row.")
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
    s.add_argument("--ef-labels", required=True, help="study_labels_ef.parquet (study_id, accession, ef).")
    s.add_argument("--rvsp-labels", required=True, help="study_labels_rvsp.parquet (study_id, rvsp).")
    s.add_argument("--reports", required=True, help="cleaned DeepECHO report parquet.")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--seed", type=int, default=20261014)
    s.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    s.add_argument("--vendor", default="GE Vingmed Ultrasound", help="Empty string for any vendor.")
    s.add_argument("--min-clips", type=int, default=20, help="kept clips a study needs.")
    s.add_argument("--per-category", type=int, default=8, help="studies shortlisted per category.")
    s.add_argument("--reference-per-layout", type=int, default=200,
                   help="reference clips per (vendor, model, acquisition) for the screen's layout baselines, from "
                        "eligible clips of studies not shortlisted.")
    t = sub.add_parser("select")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--flags", required=True, help="clip_screen.py flag output for shortlist_clips.csv.")
    t.add_argument("--n-studies", type=int, default=3, choices=(2, 3))
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
