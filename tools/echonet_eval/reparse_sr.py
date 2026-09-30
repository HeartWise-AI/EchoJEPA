"""Re-parse the GE structured reports of an existing results directory with the current sr_parse.py
(e.g. after adding AV/MV VTI or changing unit handling), without rerunning the models.
Writes <results>/sr_reparsed.parquet, which analyze.py uses instead of the stored kind == "sr" rows."""
import argparse, glob, os, sys
from multiprocessing import Pool
import pandas as pd, pydicom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr_parse


def one(args):
    uid, acc, sdir, fname = args
    try:
        f = os.path.join(sdir, fname) if fname else (sr_parse.find_sr_files(sdir) or [None])[0]
        if not f:
            return None
        meas = sr_parse.canonical_measurements(sr_parse.parse_sr_dataset(pydicom.dcmread(f)))
        return dict(StudyInstanceUID=uid, AccessionNumber=acc, kind="sr", model="ge_sr", file=os.path.basename(f), **{f"sr_{k}": v for k, v in meas.items()})
    except Exception as e:
        return dict(StudyInstanceUID=uid, kind="error", model="sr", error=str(e)[:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="/volume/echonet_eval/results")
    ap.add_argument("--cohort", default="/volume/echonet_eval/cohort_10k.parquet"); ap.add_argument("--procs", type=int, default=32)
    a = ap.parse_args()
    coh = pd.read_parquet(a.cohort)
    old = pd.concat([pd.read_parquet(f, columns=["StudyInstanceUID", "kind", "file"]) for f in glob.glob(os.path.join(a.results, "rows_shard*.parquet"))])
    srf = old[old.kind == "sr"].drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").file
    jobs = [(r.StudyInstanceUID, r.AccessionNumber, r.study_dir, srf.get(r.StudyInstanceUID)) for r in coh.itertuples() if r.StudyInstanceUID in srf.index]
    with Pool(a.procs) as p:
        out = [r for r in p.imap_unordered(one, jobs, chunksize=16) if r]
    df = pd.DataFrame(out); df.to_parquet(os.path.join(a.results, "sr_reparsed.parquet"), index=False)
    print(df.kind.value_counts().to_dict(), {c: int(df[c].notna().sum()) for c in df.columns if c.startswith("sr_")})
