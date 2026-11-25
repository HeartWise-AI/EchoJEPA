# /data/freeze_fresh_test.py

"""Freeze a fresh, uninspected confirmatory test set for archive-scale encoders.

The 10k cohort's test patients have been inspected (pilot results), so confirmatory results use
patients the shared split newly assigned to the test pool (`origin == "new"` in patient_splits.csv).
This draws them once, before any archive-scale encoder or probe exists, and makes the lists read-only:

- patients: a seeded sample of new test patients with at least one EF-labelled standard TTE study
  holding an A4C video (the EF probe's input view);
- EF studies: one such study per patient (seeded choice);
- RVSP studies: for the same patients, one RVSP-labelled study holding at least one RVSP slot view
  (the EF study when it qualifies, else a seeded choice); patients without one are absent.

Video-level verification happens later (build_manifests.py); a study whose videos all fail stays in
the frozen list and is reported as failed, never replaced.

Example:
    python data/freeze_fresh_test.py --metadata <snapshot.parquet> --splits <patient_splits.csv> \\
        --ef-labels <study_labels_ef.parquet> --rvsp-labels <study_labels_rvsp.parquet> --out-dir <fresh_test>
"""

import argparse
import datetime
import json
import os

import numpy as np
import pandas as pd

from build_manifests import ensure_outside_repo, git_commit, normalize_id, sha256

TTE = "ECHO CARDIAQUE AVEC DOPPLER"
EF_VIEWS = ("A4C",)
# Views the RVSP early-fusion probe can use (docs/rvsp_multiview_probe.md default slots).
RVSP_VIEWS = ("A4C", "RVINF", "PSAX", "PSAX_AV", "SUBCOSTAL")


def choose(df, key, seed):
    """One row per `key`, chosen by a seeded hash rank so the choice ignores row order."""
    rank = pd.util.hash_pandas_object(df.study_id.astype(str) + f"|{seed}", index=False).to_numpy()
    return df.assign(_rank=rank).sort_values([key, "_rank"]).drop_duplicates(key).drop(columns="_rank")


def freeze(meta, splits, ef, rvsp, n, seed):
    new_test = set(splits.loc[(splits.split == "test") & (splits.origin == "new"), "patient_id"])
    v = meta[meta.patient_id.isin(new_test)]
    a4c = set(v.loc[v.view.isin(EF_VIEWS), "study_id"])
    ef_studies = (v[["patient_id", "study_id"]].drop_duplicates()
                  .merge(ef[["study_id", "ef"]], on="study_id", how="inner"))
    ef_studies = ef_studies[ef_studies.study_id.isin(a4c)].sort_values(["patient_id", "study_id"])
    pool = np.sort(ef_studies.patient_id.unique())
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} new test patients qualify; asked for {n}.")
    patients = np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))
    ef_chosen = choose(ef_studies[ef_studies.patient_id.isin(set(patients))], "patient_id", seed)

    rvsp_views = set(v.loc[v.view.isin(RVSP_VIEWS), "study_id"])
    r = (v[["patient_id", "study_id"]].drop_duplicates().merge(rvsp[["study_id", "rvsp"]], on="study_id"))
    r = r[r.patient_id.isin(set(patients)) & r.study_id.isin(rvsp_views)]
    same = r[r.study_id.isin(set(ef_chosen.study_id))]
    rest = choose(r[~r.patient_id.isin(set(same.patient_id))], "patient_id", seed)
    rvsp_chosen = pd.concat([same, rest]).sort_values("patient_id")
    return pd.DataFrame({"patient_id": patients}), ef_chosen, rvsp_chosen, len(pool)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True, help="patient_splits.csv of the archive build.")
    p.add_argument("--ef-labels", required=True)
    p.add_argument("--rvsp-labels", required=True)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    ensure_outside_repo(args.out_dir, "--out-dir")
    if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"{args.out_dir} is not empty: the fresh test set is frozen once.")
    os.makedirs(args.out_dir, exist_ok=True)

    m = pd.read_parquet(args.metadata, columns=["mrn", "study_uid", "study_type", "avi_status", "predicted_class"])
    m = m[(m.study_type == TTE) & (m.avi_status == "success")]
    meta = pd.DataFrame({"patient_id": normalize_id(m.mrn), "study_id": normalize_id(m.study_uid),
                         "view": m.predicted_class})
    splits = pd.read_csv(args.splits, dtype=str)
    splits["patient_id"] = normalize_id(splits.patient_id)
    ef = pd.read_parquet(args.ef_labels).assign(study_id=lambda d: normalize_id(d.study_id))
    rvsp = pd.read_parquet(args.rvsp_labels).assign(study_id=lambda d: normalize_id(d.study_id))

    patients, ef_chosen, rvsp_chosen, pool = freeze(meta, splits, ef, rvsp, args.n, args.seed)
    if set(ef_chosen.patient_id) != set(patients.patient_id):
        raise AssertionError("Every frozen patient must have one EF study.")
    outputs = {"patients.csv": patients, "ef_studies.csv": ef_chosen[["patient_id", "study_id", "ef"]],
               "rvsp_studies.csv": rvsp_chosen[["patient_id", "study_id", "rvsp"]]}
    for name, df in outputs.items():
        path = os.path.join(args.out_dir, name)
        df.to_csv(path, index=False)
        os.chmod(path, 0o444)
    info = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(), "seed": args.seed, "n_patients": args.n,
        "rule": "new test patients (origin new) with an EF-labelled TTE study holding an A4C video; one EF study "
                "each; RVSP: the EF study if RVSP-labelled with an RVSP view, else a seeded other one",
        "eligible_new_test_patients": int(pool), "ef_studies": int(len(ef_chosen)),
        "rvsp_studies": int(len(rvsp_chosen)),
        "rvsp_same_study_as_ef": int(rvsp_chosen.study_id.isin(set(ef_chosen.study_id)).sum()),
        "ef_mean_sd": [float(ef_chosen.ef.mean()), float(ef_chosen.ef.std())],
        "ef_below_40": float((ef_chosen.ef < 40).mean()),
        "rvsp_mean_sd": [float(rvsp_chosen.rvsp.mean()), float(rvsp_chosen.rvsp.std())],
        "inputs_sha256": {k: sha256(getattr(args, k)) for k in ("metadata", "splits", "ef_labels", "rvsp_labels")},
        "outputs_sha256": {name: sha256(os.path.join(args.out_dir, name)) for name in outputs},
        "frozen_before": "any archive-scale encoder or probe exists; no model outcome has been viewed on it",
    }
    with open(os.path.join(args.out_dir, "fresh_test_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    os.chmod(os.path.join(args.out_dir, "fresh_test_info.json"), 0o444)
    print(json.dumps({k: v for k, v in info.items() if k not in ("inputs_sha256", "outputs_sha256")}, indent=2))


if __name__ == "__main__":
    main()
