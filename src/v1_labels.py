"""Build the label+split manifest.

- Parse INDEX_general_PL -> pKd from Kd=/Ki=/IC50= token (molar -> -log10).
  Skip entries with ~ > < or ranges, or unparseable units.
- Join with LP_PDBBind.csv for new_split (leak-proof) and seq (cluster key for GroupKFold).
Writes manifest.csv (id,pKd,split,seq) and ids.txt (ids that have both label and split).
Reports dropped counts.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

UNITS = {"mM": 1e-3, "uM": 1e-6, "nM": 1e-9, "pM": 1e-12, "fM": 1e-15, "M": 1.0}
TOKEN = re.compile(r"(Kd|Ki|IC50)([=~<>])([0-9.]+)([a-zA-Z]+)")


def parse_index(path):
    rows, dropped = [], {"approx": 0, "unparse": 0}
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        code = parts[0]
        tok = parts[3]
        m = TOKEN.match(tok)
        if not m:
            dropped["unparse"] += 1
            continue
        kind, rel, val, unit = m.groups()
        if rel != "=" or unit not in UNITS:
            dropped["approx"] += 1
            continue
        try:
            molar = float(val) * UNITS[unit]
            pkd = -(molar and __import__("math").log10(molar))
        except Exception:
            dropped["unparse"] += 1
            continue
        if molar <= 0:
            dropped["unparse"] += 1
            continue
        rows.append((code, pkd))
    return pd.DataFrame(rows, columns=["id", "pKd"]), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--lp", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    idx, dropped = parse_index(a.index)
    lp = pd.read_csv(a.lp)
    idcol = lp.columns[0]
    lp = lp.rename(columns={idcol: "id"})[["id", "new_split", "seq"]]
    lp["id"] = lp["id"].str.lower()
    idx["id"] = idx["id"].str.lower()

    man = idx.merge(lp, on="id", how="inner").rename(columns={"new_split": "split"})
    man = man.dropna(subset=["pKd", "split"])
    man.to_csv(out / "manifest.csv", index=False)
    (out / "ids.txt").write_text("\n".join(man["id"]))

    print(f"INDEX parsed: {len(idx)} labelled, dropped approx={dropped['approx']} unparse={dropped['unparse']}")
    print(f"LP rows: {len(lp)}")
    print(f"manifest (label & split): {len(man)}")
    print(man["split"].value_counts().to_string())
    print(f"unique seq clusters: {man['seq'].nunique()}")


if __name__ == "__main__":
    main()
