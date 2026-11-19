# evals/radar_frozen/mine_prompts.py

"""Zero-shot prompts sampled from the training captions, in the schema of `src.models.radar.zero_shot`.

Why. The caption encoder is trained on whole structure captions, and a short hand-written sentence is not what it saw.
Both reference methods prompt with real report text for that reason:

- RADAR pools many report sentences per side of a finding (its supplement: AUC 0.894 with one prompt, 0.931 with 100,
  0.933 with 500 or more), and its inference code adds the structure's plain normal statement to every negative set.
- EchoCLIP-R, trained on whole reports, did badly with short prompts and well with "sampled prompts": 200 reports
  that carry the label for a binary task, up to ten per value for a regression task.

What. For every task of a prompt file (the task list: name, mode, structure, target, values, range), the hand-written
prompts are replaced by captions of that structure sampled from training studies with the wanted label:

- binary: up to `--per_set` captions of studies labelled 1 (positive set) and of studies labelled 0 (negative set);
- categorical: up to `--per_set` captions per grade;
- regression: up to `--per_value` captions per integer value of the rounded label inside the task's `range`, each
  with its value (`values`), instead of a `<#>` template.

Captions are drawn at random (seeded) from the rows, so a common wording is drawn more often, as in the reports. A
task is left as written, and listed under `kept_as_written`, when a side, a grade or the whole range has fewer than
`--min_prompts` captions. A caption longer than `--max_words` words is not drawn.

Only training studies are read: the caption table must be the builder's table of the split `--split` (its
`build_info.json` is checked), and label rows of other studies are ignored. The output holds report text, so it must
be outside the repository. Nothing printed holds a caption or a study identifier.

Usage:
    python -m evals.radar_frozen.mine_prompts --prompts configs/radar/prompts_v0.yaml --captions <captions.csv> \\
        --labels <study_targets.csv> --out <folder outside the repository>/prompts_mined.yaml
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import yaml

from src.datasets.radar_study_dataset import file_sha256, read_table
from src.models.radar.zero_shot import load_prompt_tasks

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _draw(captions, n, rng):
    """Up to n captions drawn without replacement from the rows, in a fixed order for a given seed."""
    captions = sorted(captions)
    return captions if len(captions) <= n else sorted(rng.sample(captions, n))


def mine_task(task, rows, per_set, per_value, min_prompts, rng):
    """The mined fields of one task ({} when there is too little to draw from). `rows`: caption and label [float] of
    the task's structure, for the training studies that have both."""
    captions, labels = rows.caption.tolist(), rows.label.to_numpy(dtype=float)

    def of(keep):
        return [c for c, k in zip(captions, keep) if k]

    if task.mode == "binary":
        positive, negative = of(labels == 1), of(labels == 0)
        if min(len(positive), len(negative)) < min_prompts:
            return {}
        return {"label_sources": _draw(positive, per_set, rng), "negative_sources": _draw(negative, per_set, rng)}
    if task.mode == "categorical":
        sources, values = [], []
        for grade in sorted(set(task.values)):
            members = of(labels == grade)
            if len(members) < min_prompts:
                return {}
            drawn = _draw(members, per_set, rng)
            sources += drawn
            values += [grade] * len(drawn)
        return {"label_sources": sources, "values": values}
    rounded = np.rint(labels)
    low, high = task.value_range if task.value_range else (rounded.min(), rounded.max())
    sources, values = [], []
    for value in range(int(low), int(high) + 1):
        drawn = _draw(of(rounded == value), per_value, rng)
        sources += drawn
        values += [value] * len(drawn)
    if len(sources) < min_prompts or len(set(values)) < 2:
        return {}
    return {"label_sources": sources, "values": values}


def mine(prompts, captions, labels, split="train", per_set=200, per_value=10, min_prompts=5, max_words=200, seed=0):
    """(prompt file content as a dict, counts per task). Paths in, nothing written."""
    tasks = load_prompt_tasks(prompts)
    with open(prompts, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    info_path = os.path.join(os.path.dirname(os.path.abspath(captions)), "build_info.json")
    if not os.path.exists(info_path):
        raise ValueError("No build_info.json beside the caption table: prompts are mined from the builder's table.")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    if info.get("captions_sha256") != file_sha256(captions) or info.get("split") != split:
        raise ValueError(f"The caption table is not the builder's table of the {split!r} split: prompts mined from "
                         "another split would hold the text of evaluated studies.")
    table = read_table(captions, id_columns=("study_id",))
    table = table[table.caption.str.split().str.len() <= max_words]
    label_table = pd.read_csv(labels, dtype=str, keep_default_na=False)
    if label_table.study_id.duplicated().any():
        raise ValueError("The label table repeats a study: one row per study.")
    label_table = label_table.set_index("study_id")
    rng = random.Random(f"prompts:{seed}")
    counts, kept = {}, []
    for name, task in tasks.items():
        if task.target not in label_table.columns:
            raise ValueError(f"{name}: the label table has no column {task.target!r}.")
        rows = table[table.structure == task.structure][["study_id", "caption"]].copy()
        text = rows.study_id.map(label_table[task.target]).fillna("").str.strip()
        rows["label"] = pd.to_numeric(text.where(text != ""), errors="coerce")
        rows = rows[np.isfinite(rows.label.to_numpy(dtype=float))]
        mined = mine_task(task, rows, per_set, per_value, min_prompts, rng)
        if not mined:
            kept.append(name)
            counts[name] = {"mined": False, "captions_with_label": int(len(rows))}
            continue
        spec = raw["tasks"][name]
        if task.mode == "regression":
            spec.pop("range", None)
        spec.update(mined)
        counts[name] = {"mined": True, "captions_with_label": int(len(rows)), "prompts": len(mined["label_sources"]),
                        "negative_prompts": len(mined.get("negative_sources", [])) or None}
    raw["mined_from"] = {"task_list_sha256": file_sha256(prompts), "captions_sha256": file_sha256(captions),
                         "labels_sha256": file_sha256(labels), "split": split, "per_set": per_set,
                         "per_value": per_value, "min_prompts": min_prompts, "max_words": max_words, "seed": seed,
                         "kept_as_written": kept, "rule_version": info.get("rule_version")}
    return raw, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", required=True, help="the task list: a prompt file whose prompts are replaced")
    parser.add_argument("--captions", required=True, help="the builder's caption table of the training split")
    parser.add_argument("--labels", required=True, help="study-level label table (data/make_study_targets.py)")
    parser.add_argument("--out", required=True, help="the mined prompt file, outside the repository")
    parser.add_argument("--split", default="train")
    parser.add_argument("--per_set", type=int, default=200)
    parser.add_argument("--per_value", type=int, default=10)
    parser.add_argument("--min_prompts", type=int, default=5)
    parser.add_argument("--max_words", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    root, target = os.path.realpath(REPO), os.path.realpath(args.out)
    if os.path.commonpath([root, target]) == root:
        raise ValueError("--out is inside the code repository; a mined prompt file holds report text.")
    content, counts = mine(args.prompts, args.captions, args.labels, args.split, args.per_set, args.per_value,
                           args.min_prompts, args.max_words, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, allow_unicode=True, sort_keys=False, width=1000)
    load_prompt_tasks(args.out)          # the file must read back as a prompt file
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
