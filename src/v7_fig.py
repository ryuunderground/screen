"""Funnel figures for the pose test: score vs RMSD for 3 weightings, example complexes
+ an aggregate panel. Navy/gold palette for the slides."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_features as V5
from v7_pose import (fit_v5_model, score_feats, rigid_decoys, compute8, _approx_move)

NAVY = "#1F3864"; GOLD = "#C8A94B"; GRAY = "#9AA3B2"
root = Path.home() / "screen"
model = fit_v5_model(root / "out/v5/v5_features.npz", root / "out/v1/manifest.csv")
names, mean, scale, coef, b = model
man = pd.read_csv(root / "out/v1/manifest.csv")
test_ids = man[man.split == "test"]["id"].tolist()
rng = np.random.default_rng(0); rng.shuffle(test_ids)

phys_w = np.array([0.5, 1.0, -2.0, 0.5, 0.8, 0.0, -0.8, -0.5])


def curves_for(code):
    paths = V5.find_complex(str(root / "data"), code)
    if paths is None or not paths[0].exists():
        return None
    lig, _ = V5.load_ligand(paths[1], paths[2])
    rec = V5.parse_pocket(paths[0])
    if lig is None or len(rec[0]) == 0:
        return None
    rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged = rec
    native = lig["xyz"]
    r = np.random.default_rng(abs(hash(code)) % (2**32))
    poses = [native] + rigid_decoys(native, 80, r)
    F, rmsd = [], []
    for nx in poses:
        lm = _approx_move(lig, nx)
        F.append(compute8(lm, rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged, nx))
        rmsd.append(float(np.sqrt(((nx - native) ** 2).sum(1).mean())))
    F = np.array(F); rmsd = np.array(rmsd)
    Z = (F - mean) / scale
    full = Z @ coef + b
    shape = score_feats(F, mean, scale, coef, b, mask=np.array([1, 1, 1, 0, 0, 0, 0, 0.0]))
    phys = Z @ phys_w
    return rmsd, full, shape, phys


# pick 3 example complexes that load
examples = []
for c in test_ids:
    r = curves_for(c)
    if r is not None and r[0].max() > 5:
        examples.append((c, r))
    if len(examples) == 3:
        break


def norm(s):
    s = np.asarray(s); return (s - s.min()) / (np.ptp(s) + 1e-9)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
titles = [("affinity weights (full v5)", "full", GRAY),
          ("contact only", "shape", NAVY),
          ("physical pose weights", "phys", GOLD)]
for ax, (code, (rmsd, full, shape, phys)) in zip(axes, examples):
    for label, s, col in [("affinity wts", full, GRAY), ("contact", shape, NAVY),
                          ("physical wts", phys, GOLD)]:
        ax.scatter(rmsd, norm(s), s=14, alpha=0.7, color=col, label=label, edgecolors="none")
    ax.axvline(2.0, color="#C00", ls="--", lw=1)
    ax.set_xlabel("RMSD to native (A)"); ax.set_ylabel("score (normalized)")
    ax.set_title(code)
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Pose scoring funnel: higher score should sit at low RMSD", y=1.02)
fig.tight_layout()
fig.savefig(root / "out/v7/fig_v7_funnel.png", dpi=150, bbox_inches="tight")
print("wrote fig_v7_funnel.png with", [e[0] for e in examples])

# aggregate bar: top1<2A for the three weightings (from the 250-complex summary)
import json
summ = json.loads((root / "out/v7/v7_summary.json").read_text())
fig, ax = plt.subplots(figsize=(5.5, 4))
labels = ["affinity\nweights", "contact\nonly", "physical\nweights"]
vals = [summ["full_top1_lt2"], summ["shape_top1_lt2"], summ["phys_top1_lt2"]]
ax.bar(labels, vals, color=[GRAY, NAVY, GOLD])
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"{v*100:.0f}%", ha="center", fontsize=11)
ax.set_ylim(0, 1.08); ax.set_ylabel("top-scored pose within 2 A  (n=%d)" % summ["n"])
ax.set_title("Native-pose recognition vs channel weighting")
fig.tight_layout(); fig.savefig(root / "out/v7/fig_v7_top1.png", dpi=150)
print("wrote fig_v7_top1.png")
