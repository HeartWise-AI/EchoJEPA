# data/still_leak_audit.py

"""Is the measured value still readable on a prepared still? Counts only.

The heads regress the sonographer's value from a still on which that value was burned in. The preprocessing masks the
label boxes, crops to the spectral region and inpaints coloured overlays, but the still dataset stores the boxes of
the labels, not of the values, so a value printed inside the spectral region could survive. This tool reads prepared
stills with OCR and counts, per vendor and still type:

- `any_number`: stills on which OCR reads a number at all (axis ticks count);
- `value_read`: stills on which a number read equals one of the still's own measured values (in the stored unit or
  its hundredfold: m/s and cm/s, s and ms);
- `value_read_shuffled`: the same against the values of another still of the same type (drawn with `--seed`): how
  often a number matches by chance, since ticks and measurements share a range. A leak shows as `value_read` well
  above `value_read_shuffled`.

Each is counted on the crop at the screen's resolution (the easiest case for a reader) and on the 224-pixel input the
encoder sees. Counts from 1 to 4 are written `<5`.

Two steps, because the OCR environment has no torch and the preprocessing needs it:

    python data/still_leak_audit.py crops --stills <stills.csv> --split val --sample 2000 --work <folder> \\
        [--still_types cw_AV ...]
    <ocr python> data/still_leak_audit.py ocr --work <folder> --out <report.json>

`crops` writes the prepared crops (.npy) and a small table into `--work`; `ocr` reads them, writes the report and
deletes the crops. Both folders must be outside the repository: the crops are patient images.
"""

import argparse
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
SCALES = (1.0, 100.0, 1000.0, 0.01)        # stored unit, cm/s for m/s, ms for s, m/s for cm/s


def outside_repo(path, name):
    if os.path.commonpath([os.path.realpath(REPO), os.path.realpath(path)]) == os.path.realpath(REPO):
        raise SystemExit(f"{name} is inside the repository: prepared stills are patient images.")


def suppressed(n):
    return "<5" if 0 < n < 5 else int(n)


def numbers_in(texts):
    """The numbers of OCR text lines, as floats."""
    return [float(m.replace(",", ".")) for text in texts for m in NUMBER.findall(str(text))]


def matches(read, values, tolerance=0.011):
    """Whether a number read equals a measured value, in one of `SCALES`, within rounding (`tolerance`, relative).
    Values below 0.1 in absolute size are not compared: any small tick would match them."""
    for value in values:
        for scale in SCALES:
            target = abs(float(value)) * scale
            if target < 0.1:
                continue
            if any(abs(r - target) <= max(tolerance * target, 0.051) for r in read):
                return True
    return False


def resized(gray, size=224):
    import cv2
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def crops(args):
    import sys
    if REPO not in sys.path:
        sys.path.append(REPO)
    from PIL import Image

    from src.datasets.still_preprocess import prepare_crop
    outside_repo(args.work, "--work")
    table = pd.read_csv(args.stills, dtype={"still_id": str, "study_id": str}, keep_default_na=False)
    table = table[table.split.astype(str) == args.split]
    if args.still_types:
        table = table[table.still_type.astype(str).isin(args.still_types)]
    if len(table) > args.sample:
        table = table.sample(args.sample, random_state=args.seed)
    os.makedirs(os.path.join(args.work, "crops"), exist_ok=True)
    kept = []
    for row in table.to_dict("records"):
        try:
            with Image.open(row["png_path"]) as image:
                screen = np.asarray(image.convert("RGB"))
            box = tuple(int(float(row[c])) for c in ("spec_x0", "spec_y0", "spec_x1", "spec_y1"))
            gray, _ = prepare_crop(screen, box, json.loads(row["text_boxes"]) if row["text_boxes"] else [])
        except Exception:
            continue
        np.save(os.path.join(args.work, "crops", row["still_id"] + ".npy"), gray)
        values = [v for group in json.loads(row["signed_values"] or "{}").values() for v in group]
        kept.append({"still_id": row["still_id"], "manufacturer": row["manufacturer"],
                     "still_type": row["still_type"], "values": json.dumps(values)})
    pd.DataFrame(kept).to_csv(os.path.join(args.work, "crops.csv"), index=False)
    print(json.dumps({"stills_sampled": int(len(table)), "crops_written": len(kept)}))


def count(table, columns):
    return {c: suppressed(int(table[c].sum())) for c in columns} | {"stills": suppressed(len(table))}


def ocr(args):
    from rapidocr_onnxruntime import RapidOCR
    outside_repo(args.work, "--work")
    outside_repo(args.out, "--out")
    table = pd.read_csv(os.path.join(args.work, "crops.csv"), dtype={"still_id": str}, keep_default_na=False)
    engine = RapidOCR()
    rng = np.random.default_rng(args.seed)
    values = [json.loads(v) for v in table["values"]]
    other = {}
    for _, group in table.groupby("still_type"):                    # another still of the same type, never itself
        index = group.index.to_numpy()
        shift = int(rng.integers(1, len(index))) if len(index) > 1 else 0
        other.update(dict(zip(index, np.roll(index, shift))))
    rows = []
    for i, still in enumerate(table.still_id):
        gray = np.load(os.path.join(args.work, "crops", still + ".npy"))
        row = {}
        for name, image in (("full", gray), ("input", resized(gray))):
            result, _ = engine(np.stack([image] * 3, axis=-1))
            read = numbers_in([line[1] for line in result or []])
            row[f"any_number_{name}"] = bool(read)
            row[f"value_read_{name}"] = matches(read, values[i])
            row[f"value_read_shuffled_{name}"] = other[i] != i and matches(read, values[other[i]])
        rows.append(row)
    found = pd.concat([table[["manufacturer", "still_type"]], pd.DataFrame(rows)], axis=1)
    columns = list(rows[0]) if rows else []
    report = {"stills_read": len(found), "seed": args.seed, "all": count(found, columns),
              "by_vendor": {k: count(g, columns) for k, g in found.groupby("manufacturer")},
              "by_still_type": {k: count(g, columns) for k, g in found.groupby("still_type")}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    shutil.rmtree(os.path.join(args.work, "crops"), ignore_errors=True)
    print(json.dumps({"stills_read": report["stills_read"], "all": report["all"]}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    c = sub.add_parser("crops")
    c.add_argument("--stills", required=True)
    c.add_argument("--split", default="val")
    c.add_argument("--sample", type=int, default=2000)
    c.add_argument("--still_types", nargs="*", default=[], help="only these still types (default: all)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--work", required=True)
    o = sub.add_parser("ocr")
    o.add_argument("--work", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    (crops if args.step == "crops" else ocr)(args)


if __name__ == "__main__":
    main()
