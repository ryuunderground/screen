"""v4 feature extraction: same as v3 but the H-bond channel is now GEOMETRIC.

v3's hbond was a blind distance-gated donor/acceptor count -> carried ~0 linear
weight (it just tracked contact count, collinear with gauss2). v4 gates each H-bond
by geometry so the channel becomes selective and orthogonal to bulk contact:

  hbond pair score = dist_fac · donor_fac · acceptor_fac   (each in [0,1])

Protonation:
  - ligand donors: RDKit AddHs(addCoords=True) -> REAL donor-H coordinates.
  - receptor: no protonation tool available (no reduce/obabel), so donor-H and
    acceptor lone-pair DIRECTIONS are approximated from heavy-atom geometry
    (the H / lone pair sits opposite the atom's bonded-neighbor centroid).

Geometry terms (standard H-bond geometry):
  dist_fac   : gaussian, peak at H···A = 2.0 Å (ligand donor) or D···A = 2.9 Å (rec donor)
  donor_fac  : ∠D–H···A near 180° (linear); when H unknown, A must sit opposite D's neighbor
  acceptor_fac: H/D must approach from the acceptor lone-pair side (opposite A's neighbor)

Other 5 channels (gauss1, gauss2, repulsion, hydrophobic, nrot) are IDENTICAL to v3.
Same complexes / splits / CV / labels. Only channel weights are learned.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "CL": 1.8, "BR": 2.0, "I": 2.2,
    "MG": 1.2, "MN": 1.2, "ZN": 1.2, "CA": 1.2, "FE": 1.2, "NA": 1.2, "K": 1.2,
}
HALOGENS = {"F", "CL", "BR", "I"}
PAIR_CUT = 8.0
HB_CUT = 4.0          # heavy-heavy cutoff for H-bond candidates
BOND_CUT = 1.9        # heavy-heavy bonded-neighbor distance

FEATURE_NAMES = ["gauss1", "gauss2", "repulsion", "hydrophobic", "hbond", "nrot"]

_DON = {
    ("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("TRP", "NE1"), ("ASN", "ND2"), ("GLN", "NE2"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
}
_ACC = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"),
}
_BB_DON = {"N"}
_BB_ACC = {"O", "OXT"}


def elem_from_name(atom_name):
    nm = atom_name.strip()
    if not nm:
        return ""
    two = nm[:2].upper()
    if two in ("CL", "BR", "ZN", "MG", "MN", "FE", "NA", "CA"):
        return two
    return nm[0].upper()


def _neighbor_centroid(xyz, tree):
    """For each atom, centroid of heavy atoms within BOND_CUT (its bonded neighbors).

    Returns (N,3) array; rows with no neighbor are NaN.
    """
    out = np.full_like(xyz, np.nan)
    groups = tree.query_ball_point(xyz, BOND_CUT)
    for i, g in enumerate(groups):
        g = [j for j in g if j != i]
        if g:
            out[i] = xyz[g].mean(0)
    return out


def parse_pocket_pdb(path):
    """Heavy atoms of pocket.pdb -> xyz, elements, is_don, is_acc, is_hyd, nbr_centroid."""
    xyz, els, don, acc, hyd = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            name = line[12:16].strip()
            el_col = line[76:78].strip()
            res = line[17:20].strip()
            el = el_col.upper() if el_col else elem_from_name(name)
            if el in ("H", "D"):
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            xyz.append((x, y, z)); els.append(el)
            don.append((res, name) in _DON or name in _BB_DON)
            acc.append((res, name) in _ACC or name in _BB_ACC)
            hyd.append(el in ("C", "S"))
    xyz = np.array(xyz, float)
    if len(xyz) == 0:
        return xyz, np.array([]), np.array([]), np.array([]), np.array([]), xyz
    nbr = _neighbor_centroid(xyz, cKDTree(xyz))
    return (xyz, np.array(els), np.array(don), np.array(acc), np.array(hyd), nbr)


def load_ligand(sdf, mol2):
    """Ligand -> dict of arrays incl. real donor-H coords. Try sdf then mol2."""
    mol, note = None, "sdf"
    for loader, src, tag in ((Chem.MolFromMolFile, sdf, "sdf"),
                             (Chem.MolFromMol2File, mol2, "mol2")):
        try:
            mol = loader(str(src), sanitize=True, removeHs=True)
        except Exception:
            mol = None
        if mol is not None:
            note = tag
            break
    if mol is None:
        mol = Chem.MolFromMolFile(str(sdf), sanitize=False, removeHs=True)
        note = "nosanitize"
        if mol is None:
            return None, "FAIL"
    try:
        nrot = int(Descriptors.NumRotatableBonds(mol))
    except Exception:
        nrot = 0
    # typing on the heavy mol; indices are preserved when we AddHs
    try:
        molH = Chem.AddHs(mol, addCoords=True)
    except Exception:
        molH = mol
    confH = molH.GetConformer()

    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    xyz, els, don, acc, hyd = [], [], [], [], []
    Hcoords = []   # list of (k,3) arrays per heavy atom
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        idx = a.GetIdx()
        p = confH.GetAtomPosition(idx)
        xyz.append((p.x, p.y, p.z))
        sym = a.GetSymbol().upper(); els.append(sym)
        nH = a.GetTotalNumHs(); fc = a.GetFormalCharge()
        don.append(sym in ("N", "O") and nH > 0 and fc >= 0)
        acc.append((sym == "O" and fc <= 0) or
                   (sym == "N" and fc <= 0 and a.GetTotalDegree() + nH < 4 and nH == 0))
        is_hyd = (sym == "C" and all(nb.GetSymbol() in ("C", "H") for nb in a.GetNeighbors())) \
            or sym in HALOGENS
        hyd.append(is_hyd)
        # real H coords: neighbors in molH that are hydrogens
        hs = []
        for nb in molH.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                q = confH.GetAtomPosition(nb.GetIdx())
                hs.append((q.x, q.y, q.z))
        Hcoords.append(np.array(hs, float) if hs else np.zeros((0, 3)))
    if not xyz:
        return None, "FAIL"
    xyz = np.array(xyz, float)
    nbr = _neighbor_centroid(xyz, cKDTree(xyz)) if len(xyz) > 1 else np.full_like(xyz, np.nan)
    return dict(xyz=xyz, els=np.array(els), don=np.array(don), acc=np.array(acc),
                hyd=np.array(hyd), H=Hcoords, nbr=nbr, nrot=nrot), note


def _ramp(d, lo, hi):
    return np.clip((hi - d) / (hi - lo), 0.0, 1.0)


def _cos(u, v):
    """Row-wise cosine between vector(s) u and v. u:(k,3) or (3,), v:(k,3)."""
    u = np.atleast_2d(u); v = np.atleast_2d(v)
    nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
    denom = np.where((nu * nv) == 0, 1e-9, nu * nv)
    return np.einsum("ij,ij->i", u, v) / denom


def _dir_fac(cosval, t):
    """Antiparallel preference: 1 at cos=-1 (vectors opposed), 0 at cos=-t (threshold).

    t is the positive threshold magnitude: donor linearity t=0.5 (angle>120 deg),
    acceptor/rec-donor directionality t=0.17 (angle>100 deg).
    """
    return np.clip((-t - cosval) / (1.0 - t), 0.0, 1.0)


def _hbond(lig, rxyz, rdon, racc, rnbr):
    """Geometric H-bond sum over ligand<->receptor donor/acceptor pairs."""
    lx = lig["xyz"]
    tree_r = cKDTree(rxyz)
    total = 0.0
    cand = tree_r.query_ball_point(lx, HB_CUT)
    for li in range(len(lx)):
        ri = np.asarray(cand[li], dtype=int)
        if ri.size == 0:
            continue
        D0 = lx[li]
        # --- ligand DONOR (real H) -> receptor ACCEPTOR ---
        if lig["don"][li] and lig["H"][li].shape[0] > 0:
            acc_i = ri[racc[ri]]
            if acc_i.size:
                A = rxyz[acc_i]                      # (k,3)
                best = np.zeros(acc_i.size)
                for H in lig["H"][li]:               # pick best H per acceptor
                    HA = A - H
                    dHA = np.linalg.norm(HA, axis=1)
                    dist_fac = np.exp(-((dHA - 2.0) / 0.5) ** 2)
                    don_fac = _dir_fac(_cos(D0 - H, HA), 0.5)   # angle D-H-A > 120
                    an = rnbr[acc_i]
                    acc_fac = np.ones(acc_i.size)
                    ok = ~np.isnan(an[:, 0])
                    if ok.any():
                        acc_fac[ok] = _dir_fac(_cos(an[ok] - A[ok], H - A[ok]), 0.17)
                    best = np.maximum(best, dist_fac * don_fac * acc_fac)
                total += float(best.sum())
        # --- receptor DONOR (no H) -> ligand ACCEPTOR ---
        if lig["acc"][li]:
            don_i = ri[rdon[ri]]
            if don_i.size:
                D = rxyz[don_i]                      # (k,3) receptor donors
                A = D0
                dDA = np.linalg.norm(A - D, axis=1)
                dist_fac = np.exp(-((dDA - 2.9) / 0.6) ** 2)
                dn = rnbr[don_i]
                don_fac = np.ones(don_i.size)
                ok = ~np.isnan(dn[:, 0])
                if ok.any():
                    don_fac[ok] = _dir_fac(_cos(dn[ok] - D[ok], A - D[ok]), 0.17)
                an = lig["nbr"][li]
                if not np.isnan(an[0]):
                    acc_fac = _dir_fac(_cos(an - A, D - A), 0.17)
                else:
                    acc_fac = np.ones(don_i.size)
                total += float((dist_fac * don_fac * acc_fac).sum())
    return total


def features_for(pocket_pdb, sdf, mol2):
    lig, note = load_ligand(sdf, mol2)
    if lig is None:
        return None, note
    rxyz, rel, rdon, racc, rhyd, rnbr = parse_pocket_pdb(pocket_pdb)
    if len(rxyz) == 0:
        return None, "NOREC"
    lx = lig["xyz"]
    lrad = np.array([RADII.get(e, 1.9) for e in lig["els"]])
    rrad = np.array([RADII.get(e, 1.9) for e in rel])

    tree_r = cKDTree(rxyz)
    pairs = tree_r.query_ball_point(lx, PAIR_CUT)
    g1 = g2 = rep = hphob = 0.0
    for li, ridxs in enumerate(pairs):
        if not ridxs:
            continue
        ridxs = np.asarray(ridxs)
        dd = np.linalg.norm(rxyz[ridxs] - lx[li], axis=1)
        d = dd - lrad[li] - rrad[ridxs]
        g1 += float(np.exp(-(d / 0.5) ** 2).sum())
        g2 += float(np.exp(-((d - 3.0) / 2.0) ** 2).sum())
        neg = d < 0
        rep += float((d[neg] ** 2).sum())
        if lig["hyd"][li]:
            hphob += float(_ramp(d[rhyd[ridxs]], 0.5, 1.5).sum())
    hb = _hbond(lig, rxyz, rdon, racc, rnbr)
    return np.array([g1, g2, rep, hphob, hb, float(lig["nrot"])]), note


def find_complex(data_root, code):
    for era in ("1981-2000", "2001-2010", "2011-2019"):
        d = Path(data_root) / "pdbbind/v2020R1/P-L" / era / code
        if d.is_dir():
            return (d / f"{code}_pocket.pdb", d / f"{code}_ligand.sdf", d / f"{code}_ligand.mol2")
    return None


def run_shard(data_root, ids, out_path, shard="1/1"):
    k, N = (int(x) for x in shard.split("/"))
    mine = [c for i, c in enumerate(ids) if i % N == (k - 1)]
    rows, idout, notes = [], [], []
    for j, code in enumerate(mine):
        paths = find_complex(data_root, code)
        if paths is None or not paths[0].exists():
            notes.append(f"{code}\tMISSING"); continue
        try:
            vec, note = features_for(*paths)
        except Exception as e:
            notes.append(f"{code}\tERR {type(e).__name__}: {e}"); continue
        if vec is None:
            notes.append(f"{code}\t{note}"); continue
        rows.append(vec); idout.append(code)
        if note != "sdf":
            notes.append(f"{code}\t{note}")
        if j % 200 == 0:
            print(f"shard {shard}: {j}/{len(mine)}", flush=True)
    X = np.array(rows) if rows else np.zeros((0, len(FEATURE_NAMES)))
    np.savez(out_path, ids=np.array(idout), X=X, feature_names=np.array(FEATURE_NAMES))
    Path(str(out_path) + ".notes").write_text("\n".join(notes))
    print(f"shard {shard} done: {len(idout)} ok, {len(notes)} notes -> {out_path}", flush=True)


def demo():
    """Self-check: a linear donor-H...acceptor scores high; a sideways one scores ~0."""
    # ligand donor D with H, receptor acceptor A collinear at H...A = 2.0 A
    D = np.array([0.0, 0, 0]); H = np.array([1.0, 0, 0]); A = np.array([3.0, 0, 0])
    An = np.array([4.0, 0, 0])  # acceptor neighbor beyond A (so H approaches opposite side)
    lig = dict(xyz=np.array([D]), don=np.array([True]), acc=np.array([False]),
               hyd=np.array([False]), H=[np.array([H])], nbr=np.array([[np.nan] * 3]), nrot=0,
               els=np.array(["N"]))
    s_lin = _hbond(lig, np.array([A]), np.array([False]), np.array([True]),
                   np.array([An]))
    # bent case: acceptor off-axis so D-H...A angle ~90
    A2 = np.array([1.0, 2.0, 0]); An2 = np.array([1.0, 3.0, 0])
    s_bent = _hbond(lig, np.array([A2]), np.array([False]), np.array([True]),
                    np.array([An2]))
    print(f"demo: linear hbond={s_lin:.3f}  bent hbond={s_bent:.3f}")
    assert s_lin > 0.5, "linear H-bond should score high"
    assert s_bent < s_lin, "bent H-bond should score lower than linear"
    print("demo OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "screen/data"))
    ap.add_argument("--ids")
    ap.add_argument("--out")
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        return demo()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.smoke:
        ids = ids[:5]
    run_shard(a.data_root, ids, a.out, a.shard)


if __name__ == "__main__":
    sys.exit(main())
