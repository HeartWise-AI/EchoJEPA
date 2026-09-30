"""Turn analyze.py outputs into paper-style Markdown tables (for Notion / the PR)."""
import argparse, json, os
import pandas as pd


def fmt_row(r):
    ci = f"{r.r:.2f} [{r.r_ci_low:.2f}, {r.r_ci_high:.2f}]" if pd.notna(r.get("r")) else "-"
    mae = f"{r.mae:.2f}" if pd.notna(r.get("mae")) else "-"
    bias = f"{r.bias:+.2f} [{r.loa_low:+.2f}, {r.loa_high:+.2f}]" if pd.notna(r.get("bias")) else "-"
    return f"| {r.measurement} | {r.gt} | {int(r.n):,} | {ci} | {mae} | {bias} |"


def main(analysis_dir, out_path):
    tab = pd.read_csv(f"{analysis_dir}/agreement_table.csv")
    ef = pd.read_csv(f"{analysis_dir}/ef_table.csv")
    summ = json.load(open(f"{analysis_dir}/summary.json"))
    L = [f"**Studies analysed:** {summ['n_studies']:,}  ·  rows: {summ['n_rows']:,}  ·  " + ", ".join(f"{k} {v:,}" for k, v in summ["kinds"].items()), ""]
    hdr = ["| Measurement | Reference | n | Pearson r [95% CI] | MAE | Bias [LoA] |", "|---|---|---|---|---|---|"]
    for title, sel in [("2D calipers (EchoNet-Measurement, 3:4 sector canvas)", tab.model.isin(["ivs", "lvid", "lvpw", "la", "aorta", "aortic_root", "rv_base", "ivc", "pa"])),
                       ("Doppler peak velocities (EchoNet-Measurement)", tab.model.isin(["trvmax", "avvmax", "mrvmax", "lvotvmax", "latevel", "medevel"])),
                       ("LA area (EchoNet-Segmentation)", tab.model.eq("LA_AREA")),
                       ("VTI (EchoNet-Segmentation, YOLO + DeepLabV3)", tab.model.str.startswith("VTI_"))]:
        s = tab[sel]
        if len(s) == 0:
            continue
        L += [f"### {title}", *hdr, *[fmt_row(r) for _, r in s.iterrows()], ""]
    if len(ef):
        L += ["### EchoNet-Dynamic: visual EF < 50 % (report) from A4C clips", "| Variant | n | prevalence | AUROC [95% CI] | AUPRC | MAE (EF pts) | r | bias |", "|---|---|---|---|---|---|---|---|"]
        for _, r in ef.iterrows():
            L.append(f"| {r.variant} | {int(r.n):,} | {r.prevalence_ef_lt_50:.3f} | {r.auroc:.3f} [{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}] | {r.auprc:.3f} | {r.mae:.1f} | {r.r:.2f} | {r.bias:+.1f} |")
        L.append("")
    md = "\n".join(L)
    open(out_path, "w").write(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", default="/volume/echonet_eval/analysis"); ap.add_argument("--out", default="/volume/echonet_eval/analysis/report.md")
    a = ap.parse_args(); main(a.analysis, a.out)
