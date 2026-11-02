"""Ligand + receptor scalar-field channels as 3D CUBES (R^3 -> scalar).
Real atoms, Gaussian-smeared density per channel. Same cube box (ligand-centered)
so the two rows show complementary fields. Illustrative M&M figure."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path.home() / "screen/src"))
import v5_features as V5

ROOT = Path.home() / "screen"

def good_complex():
    man = pd.read_csv(ROOT / "out/v1/manifest.csv")
    ids = man[man.split == "test"]["id"].tolist()
    rng = np.random.default_rng(7); rng.shuffle(ids)
    for code in ids:
        p = V5.find_complex(str(ROOT / "data"), code)
        if p is None or not p[0].exists():
            continue
        try:
            lig, _ = V5.load_ligand(p[1], p[2])
            rec = V5.parse_pocket(p[0])
        except Exception:
            continue
        if lig is None or len(rec[0]) == 0:
            continue
        n = len(lig["xyz"]); nhb = int((lig["don"] | lig["acc"]).sum()); nhy = int(lig["hyd"].sum())
        if 18 <= n <= 34 and nhb >= 3 and nhy >= 4:
            return code, lig, rec
    raise SystemExit("none")

CODE, lig, rec = good_complex()
rxyz, rel, rdon, racc, rhyd = rec[0], rec[1], rec[2], rec[3], rec[4]
X = lig["xyz"]; c = X.mean(0)
R = max(6.5, np.abs(X - c).max() + 2.0)
n = 32
a = np.linspace(-R, R, n)
gx, gy, gz = np.meshgrid(a, a, a, indexing="ij")
G = np.stack([gx, gy, gz], -1) + c

def field(pts, sigma=1.3):
    if len(pts) == 0:
        return np.zeros((n, n, n))
    out = np.zeros((n, n, n))
    for p in pts:  # loop atoms = low memory
        out += np.exp(-((G - p) ** 2).sum(-1) / (2 * sigma ** 2))
    return out

# keep receptor atoms inside the box (+margin) only
inbox = (np.abs(rxyz - c) < R + 2).all(1)
rows = [
    ("Ligand", X, lig["don"] | lig["acc"], lig["hyd"]),
    ("Receptor", rxyz[inbox], (rdon | racc)[inbox], rhyd[inbox]),
]
cmaps = ["Reds", "Blues", "YlOrBr"]
titles = ["Structure", "H-bond", "Hydrophobic"]

fig = plt.figure(figsize=(12, 8.2))
for ri, (rowname, pts_all, mask_hb, mask_hy) in enumerate(rows):
    masks = [np.ones(len(pts_all), bool), mask_hb, mask_hy]
    for ci in range(3):
        f = field(pts_all[masks[ci]]); f /= f.max() + 1e-9
        ax = fig.add_subplot(2, 3, ri * 3 + ci + 1, projection="3d")
        thr = 0.12; sel = f > thr
        xs, ys, zs = gx[sel] + c[0], gy[sel] + c[1], gz[sel] + c[2]
        vals = f[sel]; order = np.argsort(vals)
        ax.scatter(xs[order], ys[order], zs[order], c=vals[order], cmap=cmaps[ci], s=22,
                   alpha=0.3, vmin=thr, vmax=1.0, edgecolors="none", depthshade=False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlim(c[0]-R, c[0]+R); ax.set_ylim(c[1]-R, c[1]+R); ax.set_zlim(c[2]-R, c[2]+R)
        ax.view_init(elev=18, azim=-60)
        if ri == 0:
            ax.set_title(titles[ci], fontsize=13, pad=0)
        if ci == 0:
            ax.text2D(-0.04, 0.5, rowname, transform=ax.transAxes, fontsize=14,
                      fontweight="bold", rotation=90, va="center")

fig.suptitle(f"Scalar field is 3D (R³→scalar): channels as cubes  ({CODE})",
             fontsize=14, y=0.98)
fig.tight_layout(rect=(0.02, 0, 1, 0.97))
fig.savefig(ROOT / "out/field_cube.png", dpi=150, bbox_inches="tight")
print("wrote field_cube.png", CODE, "| lig", len(X), "rec_in_box", int(inbox.sum()))
