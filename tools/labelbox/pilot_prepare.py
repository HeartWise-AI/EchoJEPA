"""Prepare the RADAR caption pilot's clips for upload (issue #22): playable copies, opaque study keys and the review
items `pilot_upload` reads.

Run after `data/radar_pilot.py candidates`:

    python -m tools.labelbox.pilot_prepare --pilot PILOT --out PILOT/upload

PILOT is the `radar_pilot.py` --out-dir; --out a new folder outside the repository. Then, in the Labelbox environment:

    python -m tools.labelbox.pilot_upload --folder PILOT/upload                          # dry run
    python -m tools.labelbox.pilot_upload --folder PILOT/upload --apply [--replace-form]  # the pilot projects

Before copying, verify the calibrated screen, committed column review, matching candidate/view/type/key lists,
and unchanged clip bytes. Use global clip keys and HMAC study IDs to avoid exposing identifiers or paths. Record
source hashes in `prepare.json` and validate with `pilot_upload`.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import shutil
import sys
from pathlib import Path

from tools.labelbox.column_review import ROOT, SPEC
from tools.labelbox.import_column_review import VERSION
from tools.labelbox.pilot_upload import (
    CLIPS,
    COLUMN_REVIEW,
    FORMS,
    ITEMS,
    PILOT,
    PILOT_STUDY_KEY_PREFIX,
    VIEW_TYPE_REVIEW,
    UploadError,
    read_plan,
    sha256,
)
from tools.labelbox.workflow_test import ENCODE, SECRET_NAME, CopyError, _private_dir, _write_private, encode, study_key

# What `radar_pilot.py candidates` writes, all read here.
CANDIDATE_FILES = ("candidates.json", "view_type_items.csv", "global_keys.csv", "candidates_summary.json")
RECORD = "prepare.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_candidates(pilot: Path) -> tuple[list[dict], list[dict[str, str]], list[dict[str, str]]]:
    """The pilot's candidates, view and type items and listed clips, after every check that reads no video."""
    try:
        summary = json.loads((pilot / "candidates_summary.json").read_text(encoding="utf-8"))
        candidates = json.loads((pilot / "candidates.json").read_text(encoding="utf-8"))
        other = _read_csv(pilot / "view_type_items.csv")
        listed = _read_csv(pilot / "global_keys.csv")
    except FileNotFoundError:
        raise SystemExit("--pilot holds no candidates: run `data/radar_pilot.py candidates` first.")
    if summary.get("provisional_screen") is not False:
        raise SystemExit("The candidates come from a screen at provisional thresholds (--provisional-screen): a trial "
                         "run, not for upload.")
    if sha256(SPEC) not in summary.get("spec", {}).values() or any(c.get("version") != VERSION for c in candidates):
        raise SystemExit(f"The candidates were built from another column review specification than the committed "
                         f"one (v{VERSION}): run `data/radar_pilot.py candidates` again with this code.")
    keys = [row.get("global_key", "") for row in listed]
    if not keys or len(keys) != len(set(keys)) or not all(PILOT.global_key.fullmatch(key) for key in keys):
        raise SystemExit("global_keys.csv needs at least one clip, each under its own pilot global key.")
    if any(row.get("form") not in FORMS or not all(row.get(c) for c in ("study_id", "video_path", "video_sha256"))
           for row in listed):
        raise SystemExit("Every listed clip needs a known review form, its study, its video and the video's hash.")
    for form, keys_of_form in ((COLUMN_REVIEW, [c.get("global_key") for c in candidates]),
                               (VIEW_TYPE_REVIEW, [row.get("global_key") for row in other])):
        if len(keys_of_form) != len(set(keys_of_form)) or set(keys_of_form) != {
                row["global_key"] for row in listed if row["form"] == form}:
            raise SystemExit(f"The {form} items and the clips global_keys.csv lists for that form differ.")
    return candidates, other, listed


def _new_folder_outside_repo(out: Path) -> None:
    if out.resolve().is_relative_to(ROOT):
        raise SystemExit("--out must be outside the repository: the copies are patient videos.")
    if out.exists() and any(out.iterdir()):
        raise SystemExit("--out must be a new or empty folder.")


def _holds(path: str, digest: str) -> bool:
    """Whether the file at `path` still holds the bytes of `digest`; a missing or unreadable file does not."""
    try:
        return sha256(path) == digest
    except OSError:
        return False


def prepare(pilot: Path, out: Path, *, ffmpeg: str | None = None, progress: bool = False) -> dict:
    """Write the upload folder of the pilot's candidates to `out` and return its record."""
    _new_folder_outside_repo(out)
    _, other, listed = read_candidates(pilot)
    changed = [row["global_key"] for row in listed if not _holds(row["video_path"], row["video_sha256"])]
    if changed:
        raise SystemExit(f"{len(changed)} clips no longer hold the video the screen read (moved or replaced since): "
                         "screen them again, then select and candidates.")
    try:
        secret = (pilot / SECRET_NAME).read_bytes().strip()
    except FileNotFoundError:
        raise SystemExit(f"--pilot has no {SECRET_NAME}, which named its clips: it is not a radar_pilot.py folder.")
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    _private_dir(str(out))
    _private_dir(str(out / CLIPS))
    rows, frames = [], 0
    for number, row in enumerate(listed, 1):
        key = row["global_key"]
        target = out / CLIPS / f"{key}.mp4"
        try:
            checked = encode(ffmpeg, row["video_path"], str(target))
            if not _holds(row["video_path"], row["video_sha256"]):
                raise CopyError("its video changed while it was copied")
        except (CopyError, OSError) as exc:  # the message names the clip by number and key, never by path.
            detail = exc if isinstance(exc, CopyError) else type(exc).__name__
            raise SystemExit(f"Clip {number} ({key}): {detail}.")
        frames += checked["frames"]
        rows.append({"global_key": key, "form": row["form"],
                     "study_key": study_key(secret, row["study_id"], PILOT_STUDY_KEY_PREFIX),
                     "copy_sha256": sha256(target)})
        if progress:
            print(f"[prepare] {number}/{len(listed)}", end="\r", flush=True)

    items = out / ITEMS
    _private_dir(str(items))
    shutil.copyfile(pilot / "candidates.json", items / "candidates.json")  # as built: the record hashes the source.
    os.chmod(items / "candidates.json", 0o600)
    _write_private(str(items / "view_type_items.csv"), _csv(["global_key"], [{"global_key": r["global_key"]}
                                                                            for r in other]))
    _write_private(str(items / "global_keys.csv"), _csv(["global_key", "form", "study_key", "copy_sha256"], rows))
    try:
        plan = read_plan(out)
    except UploadError as exc:
        raise SystemExit(f"The prepared folder fails the uploader's checks: {exc}")
    record = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "code_sha256": {name: sha256(Path(__file__).with_name(name))
                        for name in ("pilot_prepare.py", "workflow_test.py", "column_review.py")},
        "pilot": {name: sha256(pilot / name) for name in CANDIDATE_FILES},
        "column_review_version": VERSION,
        "encode": ENCODE,
        "clips": len(rows),
        "studies": len({row["study_key"] for row in rows}),
        "frames": frames,
        "forms": {form: len(plan.rows_of(form)) for form in plan.forms()},
    }
    _write_private(str(out / RECORD), json.dumps(record, indent=2) + "\n")
    return record


def _csv(fields: list[str], rows: list[dict]) -> str:
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, required=True, help="the `data/radar_pilot.py` --out-dir")
    parser.add_argument("--out", type=Path, required=True, help="a new folder outside the repository")
    args = parser.parse_args(argv)
    record = prepare(args.pilot, args.out, progress=True)
    forms = ", ".join(f"{n} {form}" for form, n in record["forms"].items())
    print(f"[prepare] {record['clips']} playable copies of {record['studies']} studies ({forms}) in {args.out}. "
          f"Next, in the Labelbox environment: python -m tools.labelbox.pilot_upload --folder {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
