"""Rebuild field atlas: keep ORIGINAL col0 (ligand/receptor photos), replace cols 1-5
with bigger 3D CUBE channels that also draw an xyz basis-vector triad.
Composited over the original image10.png so col0 is untouched. Output 1150x392."""
import sys, io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path.home() / "screen/src"))
import v5_features as V5

ROOT = Path.home() / "screen"
CODE = "6skb"
ORIG = ROOT / "out/orig_atlas.png"   # original image10.png placed here
p = V5.find_complex(str(ROOT / "data"), CODE)
lig, _ = V5.load_ligand(p[1], p[2])
rec = V5.parse_pocket(p[0])
rxyz, rel, rdon, racc, rhyd, rnbr, rq, rchg = rec

def fields_for(xyz, don, acc, hyd, q, chg, els):
    ext = np.abs(xyz - xyz.mean(0)).max()
    c = xyz.mean(0); Rv = ext + 0.6          # tighter box -> bigger blob
    n = 30; a = np.linspace(-ext - 1.5, ext + 1.5, n)
    gx, gy, gz = np.meshgrid(a, a, a, indexing="ij")
    G = np.stack([gx, gy, gz], -1) + c
    def gauss(mask, w=None, sigma=1.3):
        pts = xyz[mask]; wt = (w[mask] if w is not None else np.ones(mask.sum()))
        out = np.zeros((n, n, n))
        for pp, ww in zip(pts, wt):
            out += ww * np.exp(-((G - pp) ** 2).sum(-1) / (2 * sigma ** 2))
        return out
    polar = np.array([e in ("N", "O") for e in els])
    sig = np.where(polar & chg, 2.5, np.where(polar, 1.0, 0.0))
    F = {"Structure": (gauss(np.ones(len(xyz), bool)), "Reds", False),
         "H-bond": (gauss(don) - gauss(acc), "RdBu_r", True),
         "Hydrophobic": (gauss(hyd), "YlOrBr", False),
         "Electrostatics": (gauss(np.abs(q) > 1e-3, w=q), "RdBu_r", True),
         "Desolvation": (gauss(sig > 0, w=sig), "Purples", False)}
    return c, Rv, gx, gy, gz, F

rows = [fields_for(lig["xyz"], lig["don"], lig["acc"], lig["hyd"], lig["q"], lig["charged"], lig["els"]),
        fields_for(rxyz, rdon, racc, rhyd, rq, rchg, rel)]
order = ["Structure", "H-bond", "Hydrophobic", "Electrostatics", "Desolvation"]

fig = plt.figure(figsize=(11.5, 3.92), dpi=100)
gs = fig.add_gridspec(2, 6, wspace=0.0, hspace=0.0, left=0, right=1, top=1, bottom=0)

def triad(ax, c, Rv):
    o = c - 0.92 * Rv            # corner origin
    L = 0.5 * Rv
    for vec, col, lab in ([L,0,0],"#d11","x"), ([0,L,0],"#1a1","y"), ([0,0,L],"#15c","z"):
        ax.quiver(o[0], o[1], o[2], vec[0], vec[1], vec[2], color=col, lw=1.4,
                  arrow_length_ratio=0.25)
        ax.text(o[0]+vec[0]*1.25, o[1]+vec[1]*1.25, o[2]+vec[2]*1.25, lab,
                color=col, fontsize=7, ha="center", va="center")

for ri, (c, Rv, gx, gy, gz, F) in enumerate(rows):
    for ci, name in enumerate(order, start=1):
        f, cmap, div = F[name]
        if div:
            f = f / (np.abs(f).max() + 1e-9); thr = 0.12; sel = np.abs(f) > thr; vmn, vmx = -1, 1
        else:
            f = f / (f.max() + 1e-9); thr = 0.12; sel = f > thr; vmn, vmx = thr, 1.0
        ax = fig.add_subplot(gs[ri, ci], projection="3d")
        xs, ys, zs = gx[sel]+c[0], gy[sel]+c[1], gz[sel]+c[2]; vals = f[sel]
        o2 = np.argsort(np.abs(vals))
        ax.scatter(xs[o2], ys[o2], zs[o2], c=vals[o2], cmap=cmap, s=22, alpha=0.34,
                   vmin=vmn, vmax=vmx, edgecolors="none", depthshade=False)
        triad(ax, c, Rv)
        ax.set_xlim(c[0]-Rv, c[0]+Rv); ax.set_ylim(c[1]-Rv, c[1]+Rv); ax.set_zlim(c[2]-Rv, c[2]+Rv)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(18, -60)

buf = io.BytesIO(); fig.savefig(buf, dpi=100, format="png"); buf.seek(0)
rendered = Image.open(buf).convert("RGB").resize((1150, 392))
# composite: keep original col0 (x < 191), take cols1-5 from rendered
atlas = Image.open(ORIG).convert("RGB").resize((1150, 392))
x0 = round(0.16571 * 1150)  # col0/col1 boundary
atlas.paste(rendered.crop((x0, 0, 1150, 392)), (x0, 0))
atlas.save(ROOT / "out/atlas_cubes.png")
print("wrote atlas_cubes.png, col0 kept from original, split x=", x0)
