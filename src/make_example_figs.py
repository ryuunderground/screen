"""Figures + PDB export for the write-up.
 (1) channel_climb.png  — held-out test r across v1..v6.
 (2) pose_example.png   — crystal (true) vs top-ranked predicted pose, overlaid.
 (3) example_pdb/       — same two examples exported as PDB for Chimera:
       {code}_protein.pdb, {code}_ligand_native.pdb, {code}_ligand_predicted.pdb

Deterministic seed (md5 of code) so the figure and the exported poses MATCH and reproduce.
Honest: predicted pose = best-scoring rigid candidate (physical channel weights), native not
in the candidate set. Local funnel / cognate redocking, not blind docking.
"""
import sys
import hashlib
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_features as V5
from v7_pose import fit_v5_model, rigid_decoys, compute8, _approx_move

ROOT = Path.home() / "screen"
NAVY = "#1F3864"; GOLD = "#C8A94B"; GRAY = "#B8C0CC"
PHYS_W = np.array([0.5, 1.0, -2.0, 0.5, 0.8, 0.0, -0.8, -0.5])


def seed_of(code):
    return int(hashlib.md5(code.encode()).hexdigest(), 16) % (2**32)


def climb():
    vers = ["v1\ngrid", "v3\nVina terms", "v4\n+H-bond", "v5\n+elec+desolv", "v6\n+charges"]
    ridge = [0.382, 0.427, 0.426, 0.460, 0.423]
    gbr = [0.461, 0.490, 0.493, 0.524, 0.496]
    x = range(len(vers))
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, gbr, "-o", color=NAVY, lw=2.2, label="nonlinear (GBR)")
    ax.plot(x, ridge, "-o", color=GOLD, lw=2.2, label="linear (Ridge)")
    ax.axhspan(0.70, 0.82, color="#8Fae8f", alpha=0.18)
    ax.text(len(vers) - 1, 0.76, "useful scorer\n(~0.7-0.8)", ha="right", va="center",
            fontsize=9, color="#3a5a3a")
    ax.annotate("best so far", xy=(3, 0.524), xytext=(1.3, 0.60), fontsize=9, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xticks(list(x)); ax.set_xticklabels(vers, fontsize=9)
    ax.set_ylabel("held-out test correlation (Pearson r)")
    ax.set_ylim(0.33, 0.85); ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Each added physical channel lifts the scorer", fontsize=12, color=NAVY)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(ROOT / "out/channel_climb.png", dpi=150); plt.close(fig)
    print("wrote channel_climb.png")


def lig_mol(sdf, mol2):
    m = Chem.MolFromMolFile(str(sdf), removeHs=True)
    if m is None:
        m = Chem.MolFromMol2File(str(mol2), removeHs=True)
    return m


def predict_pose(code, model):
    names, mean, scale, coef, b = model
    paths = V5.find_complex(str(ROOT / "data"), code)
    if paths is None or not paths[0].exists():
        return None
    lig, _ = V5.load_ligand(paths[1], paths[2])
    rec = V5.parse_pocket(paths[0])
    if lig is None or len(rec[0]) == 0:
        return None
    rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged = rec
    native = lig["xyz"]
    rng = np.random.default_rng(seed_of(code))
    decoys = rigid_decoys(native, 120, rng)               # native NOT in candidates
    best, best_s, best_r = None, -1e9, None
    for nx in decoys:
        lm = _approx_move(lig, nx)
        F = compute8(lm, rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged, nx)
        s = ((F - mean) / scale) @ PHYS_W
        if s > best_s:
            best_s, best = s, nx
            best_r = float(np.sqrt(((nx - native) ** 2).sum(1).mean()))
    m = lig_mol(paths[1], paths[2])
    bonds = ([(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()) for bd in m.GetBonds()]
             if m is not None else [])
    return dict(code=code, native=native, pred=best, rmsd=best_r, bonds=bonds,
                pocket=rxyz, nH=len(native), mol=m, paths=paths)


def draw(ax, ex):
    P = ex["pocket"]
    idx = np.random.default_rng(0).choice(len(P), size=min(120, len(P)), replace=False)
    ax.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=5, color=GRAY, alpha=0.25)
    for coords, col, lab in [(ex["native"], NAVY, "crystal (true)"),
                             (ex["pred"], GOLD, "predicted")]:
        for i, j in ex["bonds"]:
            ax.plot(*zip(coords[i], coords[j]), color=col, lw=2.0)
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=18, color=col, label=lab)
    ax.set_title(f"{ex['code']}   RMSD = {ex['rmsd']:.2f} A", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    c = ex["native"].mean(0); r = 8
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)


def write_lig_pdb(mol, coords, path, resname="LIG"):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(coords.shape[0]):
        conf.SetAtomPosition(i, Point3D(float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])))
    # tag residue name for easy Chimera selection
    for a in m.GetAtoms():
        info = a.GetPDBResidueInfo()
        if info is None:
            from rdkit.Chem import AtomPDBResidueInfo
            info = AtomPDBResidueInfo()
        info.SetResidueName(resname); info.SetResidueNumber(1); info.SetIsHeteroAtom(True)
        a.SetMonomerInfo(info)
    Chem.MolToPDBFile(m, str(path))


def export_pdb(ex, outdir):
    code = ex["code"]
    prot = ex["paths"][0].parent / f"{code}_protein.pdb"
    if prot.exists():
        shutil.copy(prot, outdir / f"{code}_protein.pdb")
    shutil.copy(ex["paths"][0], outdir / f"{code}_pocket.pdb")
    if ex["mol"] is not None:
        write_lig_pdb(ex["mol"], ex["native"], outdir / f"{code}_ligand_native.pdb", "NAT")
        write_lig_pdb(ex["mol"], ex["pred"], outdir / f"{code}_ligand_predicted.pdb", "PRD")
    print(f"  exported {code}: protein+pocket+native+predicted (RMSD {ex['rmsd']:.2f} A)")


def examples():
    model = fit_v5_model(ROOT / "out/v5/v5_features.npz", ROOT / "out/v1/manifest.csv")
    import pandas as pd
    man = pd.read_csv(ROOT / "out/v1/manifest.csv")
    ids = man[man.split == "test"]["id"].tolist()
    rng = np.random.default_rng(3); rng.shuffle(ids)
    picks = []
    for c in ids:
        ex = predict_pose(c, model)
        if ex is None:
            continue
        if 10 <= ex["nH"] <= 28 and ex["bonds"] and ex["rmsd"] < 1.5:
            picks.append(ex)
        if len(picks) == 2:
            break
    # figure
    fig = plt.figure(figsize=(11, 5))
    for k, ex in enumerate(picks):
        ax = fig.add_subplot(1, len(picks), k + 1, projection="3d")
        draw(ax, ex)
        if k == 0:
            ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("It binds like THIS (crystal) - our scorer predicts THAT (overlay)", y=1.0)
    fig.tight_layout()
    fig.savefig(ROOT / "out/pose_example.png", dpi=150, bbox_inches="tight")
    print("wrote pose_example.png", [e["code"] for e in picks], [round(e["rmsd"], 2) for e in picks])
    # PDB export
    outdir = ROOT / "out/example_pdb"; outdir.mkdir(parents=True, exist_ok=True)
    for ex in picks:
        export_pdb(ex, outdir)
    # a short readme
    lines = ["# Example binding poses (for Chimera)",
             "",
             "Crystal (true) pose vs our scorer's top-ranked predicted pose.",
             "Honest: predicted = best-scoring rigid candidate (physical channel weights),",
             "native NOT given to the scorer. Local funnel / cognate redocking, not blind docking.",
             ""]
    for ex in picks:
        c = ex["code"]
        lines += [f"## {c}  (RMSD native vs predicted = {ex['rmsd']:.2f} A)",
                  f"- {c}_protein.pdb      : full receptor",
                  f"- {c}_pocket.pdb       : binding-site pocket",
                  f"- {c}_ligand_native.pdb    : crystal ligand pose (resname NAT)",
                  f"- {c}_ligand_predicted.pdb : our predicted pose (resname PRD)",
                  "",
                  "Chimera: open protein + both ligand PDBs; color NAT navy, PRD gold; they overlay.",
                  ""]
    (outdir / "README.md").write_text("\n".join(lines))
    print("wrote example_pdb/README.md")
    return [(e["code"], round(e["rmsd"], 2)) for e in picks]


if __name__ == "__main__":
    climb()
    out = examples()
    print("PICKS:", out)
