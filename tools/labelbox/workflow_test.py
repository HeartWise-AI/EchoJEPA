"""Prepare hand-picked real clips for testing the RADAR Labelbox review workflow (issue #22).

Run three steps in the training environment (pandas, pyarrow, imageio-ffmpeg):

    python -m tools.labelbox.workflow_test studies --pilot PILOT --reports REPORTS
    python -m tools.labelbox.workflow_test copies --pilot PILOT --study valve_disease-7 --out TEST
    python -m tools.labelbox.workflow_test items --pilot PILOT --out TEST --picked picked.csv --reports REPORTS

`studies`: summarizes each shortlisted study's clips by review form, view, acquisition mode,
and available report prefills. Outputs counts only, without identifiers.

PILOT is the `data/radar_pilot.py shortlist` output directory; REPORTS is the report file used
for selection.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import hmac
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys

import numpy as np
import pandas as pd

from tools.labelbox.column_review import SPEC, build_candidate, load_spec
from tools.labelbox.pilot_upload import (
    CLIPS,
    COLUMN_REVIEW,
    ITEMS,
    STUDY_KEY_PREFIX,
    TEST_KEY_PREFIX,
    VIEW_TYPE_REVIEW,
    sha256,
)

PILOT_KEY = re.compile(r"radar-pilot-([0-9a-f]{24})")
SECRET_NAME = ".global_key_secret"
COPIES, STUDY, PAGE = "copies.csv", "study.json", "index.html"
CORE_VIEWS = ("A4C", "A2C", "PLAX", "PSAX")
FORM_NAMES = {COLUMN_REVIEW: "caption review", VIEW_TYPE_REVIEW: "view and type review"}
MODE_NAMES = {"bmode": "B-mode", "color_flow": "colour Doppler", "unknown": "mode unknown"}
# H.264 4:2:0 plays in every browser; CRF 18 is visually lossless for review. The pad adds one black line or column
# only to an odd side, which 4:2:0 cannot encode. Frames keep their timestamps (passthrough): none added or dropped.
ENCODE = ["-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_metadata:s:v", "-1",
          "-map_chapters", "-1", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "medium",
          "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-movflags", "+faststart",
          "-fflags", "+bitexact", "-flags:v", "+bitexact"]
# All an ENCODE copy may still name: the MP4 brand and ffmpeg's own stream labels.
ALLOWED_METADATA = {"major_brand", "minor_version", "compatible_brands", "handler_name", "vendor_id", "encoder"}


class CopyError(RuntimeError):
    """A playable copy could not be written or failed its checks."""


def to_test_key(pilot_key: str) -> str:
    """The test's global key of a shortlisted clip: the pilot key's opaque digest under the test prefix."""
    match = PILOT_KEY.fullmatch(pilot_key)
    if not match:
        raise ValueError("Not a pilot global key")
    return TEST_KEY_PREFIX + match.group(1)


def study_key(secret: bytes, study_id: str) -> str:
    """Opaque study key: an HMAC, because a study id has few unknown digits and a plain hash of it is reversible."""
    return STUDY_KEY_PREFIX + hmac.new(secret, study_id.encode(), hashlib.sha256).hexdigest()[:24]


def read_pilot(pilot: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shortlisted studies, each with a `label` (`<category>-<rank>`), and their clips."""
    read = dict(dtype=str, keep_default_na=False)
    studies = pd.read_csv(os.path.join(pilot, "shortlist.csv"), **read)
    clips = pd.read_csv(os.path.join(pilot, "shortlist_clips.csv"), **read)
    studies["label"] = studies["category"] + "-" + studies["rank"]
    return studies, clips


def pilot_hashes(pilot: str) -> dict[str, str]:
    return {name: sha256(os.path.join(pilot, name)) for name in ("shortlist.csv", "shortlist_clips.csv")}


def read_reports(pilot: str, path: str) -> pd.DataFrame:
    """The report table, after checking it is the file the shortlist chose its studies with."""
    with open(os.path.join(pilot, "shortlist.json"), encoding="utf-8") as f:
        chosen_with = json.load(f)["inputs"]["reports"]
    if sha256(path) != chosen_with:
        raise SystemExit("--reports is not the report file the shortlist used (its SHA-256 differs).")
    return pd.read_parquet(path)


def report_records(reports: pd.DataFrame, accessions) -> dict[str, dict]:
    """{accession: report row as plain values}: missing values become None, numpy scalars Python values. Each study
    needs exactly one report row, as for the pilot's candidates."""
    ids = reports["AccessionNumber"].astype("string").str.strip().str.strip("'\"")
    wanted = set(accessions)
    rows = reports[ids.isin(wanted)].assign(_accession=ids[ids.isin(wanted)])
    if rows["_accession"].duplicated().any() or wanted - set(rows["_accession"]):
        raise SystemExit("Every study needs exactly one report row.")
    out = {}
    for _, row in rows.iterrows():
        record = {}
        for key, value in row.drop(labels="_accession").items():
            if isinstance(value, np.generic):
                value = value.item()
            missing = value is None or value is pd.NA or value is pd.NaT
            record[key] = None if missing or (isinstance(value, float) and math.isnan(value)) else value
        out[row["_accession"]] = record
    return out


def study_table(studies: pd.DataFrame, clips: pd.DataFrame, reports: dict[str, dict], spec=None) -> list[dict]:
    """What a workflow test of each shortlisted study would cover; counts only."""
    spec = spec or load_spec()
    rows = []
    for _, study in studies.iterrows():
        own = clips[clips["study_id"] == study["study_id"]]
        caption = own[own["form"] == COLUMN_REVIEW]
        prefills = [len(build_candidate(reports[study["accession"]], clip["view"], clip["acq_category"],
                                        global_key="count-only", spec=spec)["columns"])
                    for _, clip in caption.iterrows()]
        prefilled = [n for n in prefills if n]
        families = {view.split("_")[0] for view in caption["view"]}
        rows.append({
            "study": study["label"],
            "clips": len(own),
            "caption": len(caption),
            "bmode": int((caption["acq_category"] == "bmode").sum()),
            "colour": int((caption["acq_category"] == "color_flow").sum()),
            "view_type": int((own["form"] == VIEW_TYPE_REVIEW).sum()),
            "views": caption["view"].nunique(),
            "core_views": sum(view in families for view in CORE_VIEWS),
            "prefilled_clips": len(prefilled),
            "findings_median": statistics.median(prefilled) if prefilled else 0,
            "findings_max": max(prefilled, default=0),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    header = ["study", "clips", "caption", "bmode", "colour", "view_type", "views", "core_views",
              "prefilled_clips", "findings_median", "findings_max"]
    widths = {h: max(len(h), *(len(f"{r[h]:g}" if isinstance(r[h], float) else str(r[h])) for r in rows))
              for h in header}
    print("  ".join(h.ljust(widths[h]) for h in header))
    for r in rows:
        cells = [f"{r[h]:g}" if isinstance(r[h], float) else str(r[h]) for h in header]
        print("  ".join(c.ljust(widths[h]) for c, h in zip(cells, header)))
    print("\ncaption = clips for the caption review (B-mode or colour, classified view); view_type = OTHER or "
          "unclassified clips;\ncore_views = how many of A4C, A2C, PLAX, PSAX it has; prefilled_clips = caption "
          "clips the report prefills;\nfindings = prefilled findings per prefilled clip. PSAX and SUBCOSTAL clips "
          "get no prefill (their slice is not known).")


def _private_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path: str, text: str) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o600)


def _metadata(text: str) -> dict[str, str]:
    """The `Metadata:` entries of ffmpeg's description of a file, global and per stream."""
    entries, indent = {}, None
    for line in text.splitlines():
        depth = len(line) - len(line.lstrip())
        if line.strip() == "Metadata:":
            indent = depth
        elif indent is not None and depth > indent and ":" in line:
            key, _, value = line.strip().partition(":")
            entries[key.strip()] = value.strip()
        else:
            indent = None
    return entries


def probe(ffmpeg: str, path: str) -> dict:
    """Streams, the video's codec, pixel format and size, and the metadata of a file, from ffmpeg's description."""
    text = subprocess.run([ffmpeg, "-hide_banner", "-nostdin", "-i", path], capture_output=True, text=True).stderr
    video = re.search(r"Stream #\d+:\d+.*?: Video: (\w+)[^,]*, (\w+)(?:\([^)]*\))?, (\d+)x(\d+)", text)
    if not video:
        raise CopyError("no video stream found")
    return {"streams": len(re.findall(r"^\s*Stream #", text, re.M)), "codec": video.group(1),
            "pix_fmt": video.group(2), "size": (int(video.group(3)), int(video.group(4))),
            "metadata": _metadata(text)}


def encode(ffmpeg: str, source: str, target: str) -> dict:
    """Write the playable copy of `source` to `target` and check it: one H.264 4:2:0 video stream of every source
    frame at the source size (an odd side padded by one), and no metadata beyond ffmpeg's own labels."""
    import imageio_ffmpeg

    source_info = probe(ffmpeg, source)
    done = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", source, *ENCODE, target],
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise CopyError("ffmpeg could not re-encode it")
    os.chmod(target, 0o600)
    info = probe(ffmpeg, target)
    width, height = (side + side % 2 for side in source_info["size"])
    if info["streams"] != 1 or info["codec"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise CopyError("the copy is not a single H.264 4:2:0 video stream")
    if info["size"] != (width, height):
        raise CopyError("the copy changed the frame size")
    extra = set(info["metadata"]) - ALLOWED_METADATA
    if extra or info["metadata"].get("handler_name", "VideoHandler") != "VideoHandler":
        raise CopyError(f"the copy still carries metadata ({', '.join(sorted(extra)) or 'handler_name'})")
    frames = imageio_ffmpeg.count_frames_and_secs(source)[0]
    copied = imageio_ffmpeg.count_frames_and_secs(target)[0]
    if copied != frames:
        raise CopyError(f"the copy has {copied} frames, the source {frames}")
    return {"frames": frames, "width": width, "height": height}


def page(records: list[dict], label: str) -> str:
    """The page to watch every copy and download the picked keys as `picked.csv`."""
    figures = []
    for r in records:
        mode = MODE_NAMES.get(r["acq_category"], r["acq_category"])
        caption = (f"#{r['number']} · {r['view'] if r['form'] == COLUMN_REVIEW else 'OTHER / unclassified'} · "
                   f"{mode} · {FORM_NAMES[r['form']]} · {r['frames']} frames")
        key = html.escape(r["global_key"])
        figures.append(
            f'<figure><video src="{CLIPS}/{key}.mp4" controls loop muted preload="metadata"></video>'
            f'<figcaption>{html.escape(caption)}<label><input type="checkbox" value="{key}" '
            f'data-form="{html.escape(r["form"])}"> pick</label></figcaption></figure>')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workflow test clips</title>
<style>
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 16px; background: #fff; color: #111; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }}
figure {{ margin: 0; border: 1px solid #ccc; border-radius: 6px; padding: 8px; }}
video {{ width: 100%; background: #000; }}
label {{ display: block; margin-top: 6px; font-weight: 600; }}
.bar {{ position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ccc;
        margin-bottom: 12px; }}
</style></head><body>
<h1>Workflow test: study {html.escape(label)}</h1>
<p>Pick 5–8 caption-review clips (several views, B-mode and colour) and 1–2 view and type clips.
Pick a clip only if it clearly moves, shows no measurement (calipers, traces, measurement text, a results box)
and is an ordinary echo image. Then download <code>picked.csv</code> and put it in the test folder.</p>
<div class="bar"><button id="download">Download picked.csv</button> <span id="count"></span></div>
<div class="grid">
{chr(10).join(figures)}
</div>
<script>
const boxes = [...document.querySelectorAll('input[type=checkbox]')];
const count = () => {{
  const picked = boxes.filter(b => b.checked);
  const caption = picked.filter(b => b.dataset.form === '{COLUMN_REVIEW}').length;
  document.getElementById('count').textContent =
    `${{picked.length}} picked: ${{caption}} caption review, ${{picked.length - caption}} view and type review`;
}};
boxes.forEach(b => b.addEventListener('change', count));
count();
document.getElementById('download').addEventListener('click', () => {{
  const keys = boxes.filter(b => b.checked).map(b => b.value);
  const blob = new Blob(['global_key\\n' + keys.map(k => k + '\\n').join('')], {{type: 'text/csv'}});
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'picked.csv';
  link.click();
}});
</script>
</body></html>
"""


def make_copies(pilot: str, label: str, out: str, *, ffmpeg: str | None = None, progress: bool = False) -> list[dict]:
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("--out must be a new or empty folder.")
    studies, clips = read_pilot(pilot)
    chosen = studies[studies["label"] == label]
    if len(chosen) != 1:
        raise SystemExit(f"No shortlisted study {label!r}: `studies` lists their names.")
    study = chosen.iloc[0]
    own = clips[clips["study_id"] == study["study_id"]].sort_values(["form", "view", "acq_category", "global_key"])
    if own.empty:
        raise SystemExit("The study has no shortlisted clips.")
    with open(os.path.join(pilot, SECRET_NAME)) as f:
        secret = f.read().strip().encode()
    if ffmpeg is None:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    _private_dir(out)
    _private_dir(os.path.join(out, CLIPS))
    records = []
    for number, (_, clip) in enumerate(own.iterrows(), 1):
        key = to_test_key(clip["global_key"])
        target = os.path.join(out, CLIPS, f"{key}.mp4")
        try:
            source_sha256 = sha256(clip["video_path"])
            checked = encode(ffmpeg, clip["video_path"], target)
        except (CopyError, OSError) as exc:  # the message names the clip by number and key, never by path.
            detail = exc if isinstance(exc, CopyError) else type(exc).__name__
            raise SystemExit(f"Clip {number} ({key}): {detail}.")
        records.append({"number": number, "global_key": key, "form": clip["form"], "view": clip["view"],
                        "acq_category": clip["acq_category"], **checked, "copy_sha256": sha256(target),
                        "source_sha256": source_sha256})
        if progress:
            print(f"[copies] {number}/{len(own)}", end="\r", flush=True)
    _write_private(os.path.join(out, COPIES), pd.DataFrame(records).to_csv(index=False))
    record = {"created_utc": _now(), "study": label, "study_key": study_key(secret, study["study_id"]),
              "clips": len(records), "pilot": pilot_hashes(pilot), "code_sha256": sha256(__file__),
              "encode": ENCODE}
    _write_private(os.path.join(out, STUDY), json.dumps(record, indent=2) + "\n")
    _write_private(os.path.join(out, PAGE), page(records, label))
    return records


def make_items(pilot: str, out: str, picked_path: str, reports_path: str) -> dict:
    try:
        with open(os.path.join(out, STUDY), encoding="utf-8") as f:
            study_record = json.load(f)
        copies = pd.read_csv(os.path.join(out, COPIES), dtype=str, keep_default_na=False)
    except FileNotFoundError:
        raise SystemExit("--out holds no copies: run `copies` first.")
    picked = pd.read_csv(picked_path, dtype=str, keep_default_na=False)
    if "global_key" not in picked.columns:
        raise SystemExit("picked.csv needs a global_key column.")
    keys = [key.strip() for key in picked["global_key"] if key.strip()]
    if not keys or len(keys) != len(set(keys)):
        raise SystemExit("picked.csv needs at least one key, each once.")
    copies = copies.set_index("global_key")
    unknown = [key for key in keys if key not in copies.index]
    if unknown:
        raise SystemExit(f"{len(unknown)} picked keys are not copies of this study.")
    chosen = copies.loc[keys]
    paths = {key: os.path.join(out, CLIPS, f"{key}.mp4") for key in keys}
    changed = [key for key, clip in chosen.iterrows()
               if not os.path.exists(paths[key]) or sha256(paths[key]) != clip["copy_sha256"]]
    if changed:
        raise SystemExit(f"{len(changed)} picked copies are missing or changed since `copies`: "
                         "rerun it in a new folder.")
    if pilot_hashes(pilot) != study_record["pilot"]:
        raise SystemExit("The shortlist changed since `copies`: rerun it in a new folder.")
    studies, _ = read_pilot(pilot)
    study = studies[studies["label"] == study_record["study"]].iloc[0]
    report = report_records(read_reports(pilot, reports_path), [study["accession"]])[study["accession"]]
    spec = load_spec()

    candidates, other, rows = [], [], []
    for key, clip in chosen.iterrows():
        if clip["form"] == COLUMN_REVIEW:
            candidates.append(build_candidate(report, clip["view"], clip["acq_category"], global_key=key,
                                              input_kind="video", input_provenance_verified=False, spec=spec))
        elif clip["form"] == VIEW_TYPE_REVIEW:
            other.append({"global_key": key})
        else:
            raise SystemExit(f"Unknown review form {clip['form']!r}.")
        rows.append({"global_key": key, "form": clip["form"], "study_key": study_record["study_key"],
                     "copy_sha256": clip["copy_sha256"]})

    folder = os.path.join(out, ITEMS)
    _private_dir(folder)
    _write_private(os.path.join(folder, "candidates.json"),
                   json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
    _write_private(os.path.join(folder, "view_type_items.csv"),
                   pd.DataFrame(other, columns=["global_key"]).to_csv(index=False))
    _write_private(os.path.join(folder, "global_keys.csv"), pd.DataFrame(rows).to_csv(index=False))
    n_columns = [len(c["columns"]) for c in candidates]
    summary = {"created_utc": _now(), "study": study_record["study"], "candidates": len(candidates),
               "view_type_items": len(other), "prefilled_columns": sum(n_columns),
               "candidates_without_prefill": sum(n == 0 for n in n_columns), "reports_sha256": sha256(reports_path),
               "spec_sha256": sha256(str(SPEC)), "code_sha256": sha256(__file__)}
    _write_private(os.path.join(folder, "items_summary.json"), json.dumps(summary, indent=2) + "\n")
    return summary


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("studies", help="counts per shortlisted study, to choose one")
    s.add_argument("--pilot", required=True)
    s.add_argument("--reports", required=True)
    c = sub.add_parser("copies", help="playable copies of one study's clips, and the page to pick them")
    c.add_argument("--pilot", required=True)
    c.add_argument("--study", required=True, help="a study name from `studies`, such as valve_disease-7")
    c.add_argument("--out", required=True, help="a new folder outside the repository")
    i = sub.add_parser("items", help="review items of the picked clips, for pilot_upload")
    i.add_argument("--pilot", required=True)
    i.add_argument("--out", required=True, help="the folder `copies` wrote")
    i.add_argument("--picked", required=True, help="CSV with a global_key column")
    i.add_argument("--reports", required=True)
    args = parser.parse_args(argv)

    if args.command == "studies":
        studies, clips = read_pilot(args.pilot)
        reports = report_records(read_reports(args.pilot, args.reports), studies["accession"])
        print_table(study_table(studies, clips, reports))
    elif args.command == "copies":
        records = make_copies(args.pilot, args.study, args.out, progress=True)
        forms = pd.Series([r["form"] for r in records]).value_counts()
        print(f"[copies] {len(records)} playable copies ({forms.get(COLUMN_REVIEW, 0)} caption review, "
              f"{forms.get(VIEW_TYPE_REVIEW, 0)} view and type review). Watch them in "
              f"{os.path.join(args.out, PAGE)}.")
    else:
        summary = make_items(args.pilot, args.out, args.picked, args.reports)
        print(f"[items] {summary['candidates']} caption review items ({summary['prefilled_columns']} prefilled "
              f"findings, {summary['candidates_without_prefill']} without any), {summary['view_type_items']} view "
              f"and type items, in {os.path.join(args.out, ITEMS)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
