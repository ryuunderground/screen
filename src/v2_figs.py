"""v2 figures: redocking success-rate bars (rigid vs torsion), rigid-vs-torsion scatter,
and one example docking landscape slice. Navy #1F3864 theme. All numbers from real runs."""
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1F3864"; LIGHT = "#7F9BC6"


def load(shards_dir):
    rows = []
    for f in glob.glob(str(Path(shards_dir) / "dock_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load(a.shards_dir)
    rig = np.array([r["rmsd_rigid"] for r in rows], float)
    tor = np.array([r["rmsd_tors"] for r in rows], float)
    rig = rig[np.isfinite(rig)]; n = len(rig)
    tor = np.array([r["rmsd_tors"] for r in rows], float)
    m = np.isfinite(np.array([r["rmsd_rigid"] for r in rows])) & np.isfinite(tor)
    R = np.array([r["rmsd_rigid"] for r in rows])[m]; T = tor[m]
    thr = [2.0, 3.0, 5.0]
    rig_rate = [100 * np.mean(R < t) for t in thr]
    tor_rate = [100 * np.mean(T < t) for t in thr]
    summ = dict(n=int(m.sum()), rigid_lt2=float(np.mean(R < 2)), tors_lt2=float(np.mean(T < 2)),
                rigid_lt5=float(np.mean(R < 5)), tors_lt5=float(np.mean(T < 5)),
                rigid_median=float(np.median(R)), tors_median=float(np.median(T)))
    (out / "v2_summary.json").write_text(json.dumps(summ, indent=2))
    print(summ)

    # 1. success-rate bar
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(thr)); w = 0.38
    ax.bar(x - w / 2, rig_rate, w, label="rigid FFT", color=NAVY)
    ax.bar(x + w / 2, tor_rate, w, label="+ torsion refine", color=LIGHT)
    ax.set_xticks(x); ax.set_xticklabels([f"< {t:g} Å" for t in thr])
    ax.set_ylabel(f"percent redocked (n={int(m.sum())})"); ax.set_ylim(0, 100)
    ax.set_title("v2 PRELIMINARY redocking success (PoseBusters subset)")
    ax.legend()
    for xi, v in zip(x - w / 2, rig_rate): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    for xi, v in zip(x + w / 2, tor_rate): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_v2_success.png", dpi=150); plt.close(fig)

    # 2. rigid vs torsion scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(R, T, s=18, color=NAVY, alpha=0.6, edgecolors="none")
    hi = max(R.max(), T.max()) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="#888", lw=1)
    ax.axhline(2, color="#C00", lw=0.7, ls=":"); ax.axvline(2, color="#C00", lw=0.7, ls=":")
    ax.set_xlabel("rigid RMSD (Å)"); ax.set_ylabel("+torsion RMSD (Å)")
    ax.set_title("Per-complex: torsion vs rigid\n(below diagonal = torsion helped)")
    fig.tight_layout(); fig.savefig(out / "fig_v2_scatter.png", dpi=150); plt.close(fig)
    print("wrote v2 figures to", out)


if __name__ == "__main__":
    main()
