# data/view_relabel.py

"""Draw and prepare the view relabelling set: clips prefilled with the classifier's predicted view, for one reviewer
question, to retrain the view classifier on the PSAX levels and PLAX_ZOOM_AV and to validate its other classes.

    python data/view_relabel.py draw --videos VIDEOS --out-dir DIR --exclude-patients LIST ... --exclude-clips LIST ...
    python data/clip_screen.py features --clips DIR/shortlist_clips.csv --out DIR/screen/features.parquet
    python data/clip_screen.py features --clips DIR/reference_clips.csv --out DIR/screen/reference.parquet
    <OCR environment>/bin/python -E -s data/clip_screen.py text --features DIR/screen/features.parquet \\
        --out DIR/screen/features_text.parquet
    python data/clip_screen.py flag --features DIR/screen/features_text.parquet \\
        --reference DIR/screen/reference.parquet --out DIR/screen/flags.parquet
    python data/view_relabel.py prepare --out-dir DIR --flags DIR/screen/flags.parquet --out DIR/upload

Then `tools.labelbox.pilot_upload --folder DIR/upload`, which goes to the view relabelling project.

`draw` takes train-split clips of imaging modes (B-mode and color Doppler), never a still, a listed patient or a listed
clip, and at most one clip per patient. It draws more than the set needs, because the screen removes some. `prepare`
keeps only what the calibrated screen keeps, leaves out GE 3D and multi-panel image codes as the pilot does, checks the
files still hold what the screen read, and writes the upload folder. Every file holding paths stays in DIR, outside the
repository; the console shows counts only.
"""

import argparse
import datetime
import hashlib
import hmac
import json
import os
import sys

import numpy as np
import pandas as pd

import clip_screen
import radar_pilot as rp
from build_manifests import REPO_ROOT, ensure_outside_repo, git_commit, normalize_id

sys.path.insert(0, REPO_ROOT)

UPSAMPLED = {"PSAX": 400, "PLAX_ZOOM": 200}  # the rest of the set is spread evenly over the other classes.
OTHER_CLASSES_TOTAL = 400
KEY_PREFIX, STUDY_KEY_PREFIX = "radar-view-", "radar-view-study-"
REMOVED_COLUMNS = (*rp.CLIP_COLUMNS, "reasons")
RECORD, PREPARE_RECORD, PICKED = "shortlist.json", "prepare.json", "picked_clips.csv"
MEDIA_LOG = "media.log"


def quotas(classes, upsampled=UPSAMPLED, others=OTHER_CLASSES_TOTAL):
    """How many clips each class gets: `upsampled`, and `others` split evenly over the rest, the remainder going to
    the first classes in sorted order so the total stays exact."""
    missing = set(upsampled) - set(classes)
    if missing:
        raise SystemExit(f"The manifest has no clips of {sorted(missing)}.")
    rest = sorted(set(classes) - set(upsampled))
    share, extra = divmod(others, len(rest))
    return {**upsampled, **{view: share + (i < extra) for i, view in enumerate(rest)}}


def path_tail(paths, parts=3):
    """The last folders and name of each path without its extension: the exclusion lists and the manifest root the same
    clips differently."""
    text = pd.Series(paths, dtype="string").str.replace("\\", "/", regex=False)
    return text.str.rsplit(".", n=1).str[0].str.split("/").str[-parts:].str.join("/")


def excluded_clips(paths, inputs):
    """The clips of the --exclude-clips lists (CSV files with avi_path or video_path), as `path_tail`s."""
    excluded = set()
    for path in paths:
        listed = inputs.csv(path, dtype=str)
        column = next((c for c in ("avi_path", "video_path") if c in listed), None)
        if column is None:
            raise SystemExit(f"--exclude-clips {os.path.basename(path)} has no avi_path or video_path column.")
        excluded |= set(path_tail(listed[column].dropna()))
    return excluded


def eligible_clips(every, excluded_patients, excluded_tails):
    """The train-split clips the set may hold. The pilot's metadata rule decides, without what only the caption review
    needs: color Doppler and `OTHER` stay, and a clip whose mode is unknown or that is a still does not."""
    clips = every[every.split == "train"].copy()
    clips["patient_id"], clips["study_id"] = normalize_id(clips.patient_id), normalize_id(clips.study_id)
    reasons = rp.clip_rule(clips).str.split(";")
    fails = reasons.map(lambda found: bool({"acquisition", "few_frames"} & set(found)))
    listed = clips.patient_id.isin(excluded_patients) | path_tail(clips.video_path).isin(excluded_tails).to_numpy()
    return clips[~fails & ~listed & clips.view.notna()]


def draw_clips(clips, wanted, seed):
    """`wanted` clips of each view, at most one clip per patient over all views, the rarest views choosing first."""
    shuffled = clips.sample(frac=1.0, random_state=seed)
    drawn, used = [], set()
    pools = shuffled.groupby("view", observed=True).size()
    for view in sorted(wanted, key=lambda v: (pools.get(v, 0), v)):
        pool = shuffled[(shuffled.view == view) & ~shuffled.patient_id.isin(used)].drop_duplicates("patient_id")
        taken = pool.head(wanted[view])
        drawn.append(taken)
        used |= set(taken.patient_id)
    return pd.concat(drawn)


def reference_clips(clips, drawn, per_layout, seed):
    """Clips of other patients for the screen's layout baselines: classified views, the layouts the draw has."""
    pool = clips[(clips.view != "OTHER") & ~clips.patient_id.isin(set(drawn.patient_id))]
    pool = pool.merge(drawn[list(rp.LAYOUT_COLUMNS)].drop_duplicates(), on=list(rp.LAYOUT_COLUMNS))
    return pool.sample(frac=1.0, random_state=seed).groupby(list(rp.LAYOUT_COLUMNS), dropna=False).head(per_layout)


def global_key(secret, video_path):
    digest = hmac.new(secret, video_path.encode(), hashlib.sha256).hexdigest()
    return f"{KEY_PREFIX}{digest[:24]}"


def set_key(secret):
    """The one study key of the set: an opaque key for its single batch."""
    return STUDY_KEY_PREFIX + hmac.new(secret, b"view relabelling set", hashlib.sha256).hexdigest()[:24]


def _with_columns(clips, secret):
    return clips.assign(form="view_review", mode_source=rp.mode_source(clips),
                        global_key=[global_key(secret, p) for p in clips.video_path])[list(rp.CLIP_COLUMNS)]


def draw(args):
    ensure_outside_repo(args.out_dir, "--out-dir")
    clip_screen.private_dir(args.out_dir)
    rp.clear_outputs(args.out_dir, ("shortlist_clips.csv", "reference_clips.csv", "removed_clips.csv", RECORD, PICKED))
    inputs = clip_screen.Inputs(__file__)
    columns = [c for c in rp.CLIP_COLUMNS if c not in rp.DERIVED_COLUMNS] + ["split", "acq_confidence",
                                                                             "acq_vendor_validated"]
    built_at = rp.manifest_build(args.videos, columns, inputs)
    every = inputs.parquet(args.videos, columns=columns)
    clips = eligible_clips(every, rp.excluded_patients(args.exclude_patients, inputs),
                           excluded_clips(args.exclude_clips, inputs))
    target = quotas(clips.view.unique())
    wanted = {view: max(round(n * args.oversample), args.min_draw) for view, n in target.items()}
    drawn = draw_clips(clips, wanted, args.seed)
    short = {v: int((drawn.view == v).sum()) for v, n in target.items() if (drawn.view == v).sum() < n}
    if short:
        raise SystemExit(f"Too few patients for the set in {short}: no list was written.")
    secret = rp.load_secret(args.out_dir)
    reference = reference_clips(clips, drawn, args.reference_per_layout, args.seed)
    outputs = [os.path.join(args.out_dir, n) for n in ("shortlist_clips.csv", "reference_clips.csv", "removed_clips.csv")]
    _with_columns(drawn, secret).to_csv(outputs[0], index=False)
    _with_columns(reference, secret).to_csv(outputs[1], index=False)
    pd.DataFrame(columns=list(REMOVED_COLUMNS)).to_csv(outputs[2], index=False)  # the set removes no clip of a study.
    inputs.check(*outputs)
    record = {"created_utc": _now(), "code_commit": git_commit(), "code_sha256": clip_screen.code_sha256(__file__),
              "seed": args.seed, "quotas": target, "drawn": {k: int(n) for k, n in drawn.view.value_counts().items()},
              "reference_clips": int(len(reference)), "manifest_commit": built_at,
              "inputs": {os.path.abspath(p): inputs.digest(p)
                         for p in (args.videos, *args.exclude_patients, *args.exclude_clips)},
              "outputs": rp.file_hashes(*outputs)}
    _write_private(os.path.join(args.out_dir, RECORD), json.dumps(record, indent=2) + "\n")
    print(f"[draw] {len(drawn)} clips to screen for a set of {sum(target.values())}, in {len(target)} classes; "
          f"{len(reference)} reference clips.")


def pick_clips(clips, target):
    """The first `target` clips of each view, in drawn order, among those the screen keeps and the pilot would keep
    for their image code. A view short of its quota stops the run."""
    kept = clips[(clips.decision == "keep") & ~rp.pilot_rule(clips).str.contains("3d_or_multipanel", regex=False)]
    picked = pd.concat([kept[kept.view == view].head(n) for view, n in target.items()])
    short = {v: f"{int((picked.view == v).sum())} of {n}" for v, n in target.items() if (picked.view == v).sum() < n}
    if short:
        raise SystemExit(f"The screen keeps too few clips of {short}: nothing was prepared. Run draw again with a "
                         "larger --oversample or --min-draw, then the screen.")
    return picked


def prepare(args):
    from tools.labelbox.pilot_upload import CLIPS, ITEMS, UploadError, read_plan, sha256
    from tools.labelbox.view_review import FORM_NAME
    from tools.labelbox.workflow_test import ENCODE, CopyError, _private_dir, _write_private as write_private, encode

    ensure_outside_repo(str(args.out), "--out")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit("--out must be a new or empty folder.")
    inputs = clip_screen.Inputs(__file__)
    chain = rp.screen_chain(args.out_dir, args.flags, inputs)
    info = inputs.json(args.flags + ".info.json")
    if not info.get("calibration"):
        raise SystemExit("The screen ran at provisional thresholds, so its decisions are not validated: nothing was "
                         "prepared.")
    record = inputs.json(os.path.join(args.out_dir, RECORD))
    clips = inputs.csv(os.path.join(args.out_dir, "shortlist_clips.csv"), dtype={"patient_id": str, "study_id": str})
    flags = inputs.parquet(args.flags)
    flags = flags[["video_path", "decision", "decision_reasons", "pixel_mode", *rp.MEDIA_COLUMNS]]
    screened = rp.resolve_modes(rp.screened_clips(clips, flags))
    picked = pick_clips(screened, record["quotas"])
    changed = clip_screen.changed_media(picked, os.path.join(args.out_dir, MEDIA_LOG))
    if changed:
        raise SystemExit(f"{len(changed)} picked clips no longer hold the video or DICOM regions the screen read: "
                         "nothing was prepared. Run the screen again.")
    secret = rp.load_secret(args.out_dir)
    if args.ffmpeg is None:
        import imageio_ffmpeg

        args.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = str(args.out)
    _private_dir(out)
    _private_dir(os.path.join(out, CLIPS))
    rows, views = [], []
    for number, clip in enumerate(picked.itertuples(), 1):
        target = os.path.join(out, CLIPS, f"{clip.global_key}.mp4")
        try:
            encode(args.ffmpeg, clip.video_path, target)
        except (CopyError, OSError) as exc:  # the message names the clip by number and key, never by path.
            detail = exc if isinstance(exc, CopyError) else type(exc).__name__
            raise SystemExit(f"Clip {number} ({clip.global_key}): {detail}.")
        rows.append({"global_key": clip.global_key, "form": "view_review", "study_key": set_key(secret),
                     "copy_sha256": sha256(target)})
        views.append({"global_key": clip.global_key, "view": clip.view})
        print(f"[prepare] {number}/{len(picked)}", end="\r", flush=True)
    items = os.path.join(out, ITEMS)
    _private_dir(items)
    write_private(os.path.join(items, "candidates.json"), "[]\n")
    write_private(os.path.join(items, "view_type_items.csv"), "global_key\n")
    write_private(os.path.join(items, "view_items.csv"), pd.DataFrame(views).to_csv(index=False))
    write_private(os.path.join(items, "global_keys.csv"), pd.DataFrame(rows).to_csv(index=False))
    try:
        read_plan(args.out)
    except UploadError as exc:
        raise SystemExit(f"The prepared folder fails the uploader's checks: {exc}")
    picked_path = os.path.join(args.out_dir, PICKED)
    write_private(picked_path, picked[[*rp.CLIP_COLUMNS, "video_sha256"]].to_csv(index=False))
    inputs.check(picked_path)
    write_private(os.path.join(out, PREPARE_RECORD), json.dumps({
        "created_utc": _now(), "code_commit": git_commit(), "form": FORM_NAME, "encode": ENCODE, "clips": len(rows),
        "views": {k: int(n) for k, n in picked.view.value_counts().items()},
        "modes": {k: int(n) for k, n in picked.acq_category.fillna("unknown").value_counts().items()},
        "vendors": {k: int(n) for k, n in picked.vendor.fillna("unknown").value_counts().items()},
        "screen": {"decisions": {k: int(n) for k, n in screened.decision.value_counts().items()},
                   "calibration": info["calibration"].get("date")},
        "chain": chain}, indent=2) + "\n")
    print(f"\n[prepare] {len(rows)} clips in {len(record['quotas'])} classes, in {out}.")


def _write_private(path, text):
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(text)


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draw", help="the clips to screen and the reference clips")
    d.add_argument("--videos", required=True, help="the archive manifest's videos table")
    d.add_argument("--out-dir", required=True, help="a folder outside the repository")
    d.add_argument("--seed", type=int, default=20261106)
    d.add_argument("--exclude-patients", nargs="*", default=[], help="CSV files with patient_id")
    d.add_argument("--exclude-clips", nargs="*", default=[], help="CSV files with avi_path or video_path")
    d.add_argument("--oversample", type=float, default=2.0, help="clips drawn per clip of the set")
    d.add_argument("--min-draw", type=int, default=50, help="clips drawn at least for each class")
    d.add_argument("--reference-per-layout", type=int, default=50)
    p = sub.add_parser("prepare", help="the upload folder of the clips the screen keeps")
    p.add_argument("--out-dir", required=True, help="the folder of `draw`")
    p.add_argument("--flags", required=True, help="clip_screen.py flag output")
    p.add_argument("--out", required=True, help="a new folder outside the repository")
    p.add_argument("--ffmpeg", help="ffmpeg to use; default: imageio_ffmpeg's")
    args = parser.parse_args(argv)
    if args.command == "draw":
        draw(args)
    else:
        from pathlib import Path

        args.out = Path(args.out)
        prepare(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
