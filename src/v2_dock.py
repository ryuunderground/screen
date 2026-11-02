"""v2 (PRELIMINARY / UNVALIDATED): multi-channel FFT docking + internal torsion.

Rigid search: render receptor channels (core/skin/polar/hyd/charge) on a grid around
the known pocket (crystal-ligand centroid; cognate redocking, known site). For each of
N sampled rotations, render ligand channels, FFT-correlate each vs its receptor channel,
take the weighted-sum landscape peak -> translation. Best over rotations = predicted pose.

Torsion: after the rigid placement, greedily refine each "substantial" rotatable-bond
dihedral by scanning angles and maximizing the point-sampled field-overlap score (the
"point samples the curve" idea). One dihedral free per step, BFS outward. This is the
correct-but-simple research version.

Channel weights: v1 gave multicollinear (size-correlated) signs that are NOT usable as
docking weights, so v2 uses PHYSICALLY-SIGNED weights with v1-informed magnitudes:
clash is a hard penalty, contact/polar/hyd favorable, elec rewards charge complementarity.

Limits (be explicit): known binding box (not blind); greedy dihedral refinement (no
joint optimization, no root flexibility); nearest-voxel sampling at 0.8 A; symmetry-
corrected in-place RMSD; Gasteiger charges; no explicit H / H-bond directionality (that's v3).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms as rmt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_features import (RADII, POLAR, HYDROPHOBIC, REC_CHARGES, parse_pocket_pdb,
                         render, ball_mask, SPACING, SIGMA, SIGMA_Q, SKIN)

RDLogger.DisableLog("rdApp.*")

MARGIN = 6.0
W_CLASH = 8.0          # hard clash penalty (pilot value; v1 coef unusable, see docstring)
# v1-informed favorable magnitudes (|coef|); signs fixed physical
W = dict(contact=0.6, polar=0.6, hyd=0.15, elec=0.6)


def rand_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    R[0] = np.eye(3)
    return R


def load_mol(sdf):
    m = Chem.MolFromMolFile(str(sdf), sanitize=True, removeHs=True)
    if m is None:
        m = Chem.MolFromMolFile(str(sdf), sanitize=False, removeHs=True)
    return m


def mol_fields_input(mol):
    """Per-atom coords, element, gasteiger charge, polar/hyd masks."""
    try:
        AllChem.ComputeGasteigerCharges(mol); haveq = True
    except Exception:
        haveq = False
    conf = mol.GetConformer()
    xyz, els, q = [], [], []
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        xyz.append((p.x, p.y, p.z)); els.append(a.GetSymbol().upper())
        c = a.GetDoubleProp("_GasteigerCharge") if haveq else 0.0
        q.append(c if np.isfinite(c) else 0.0)
    return np.array(xyz, float), els, np.array(q, float)


def build_rec(protein_pdb, center, ext):
    rxyz, rel, rnm, rrs = parse_pocket_pdb(protein_pdb)
    near = np.linalg.norm(rxyz - center, axis=1) < ext + SKIN + 2
    rxyz, rel = rxyz[near], [e for e, k in zip(rel, near) if k]
    rnm = [m for m, k in zip(rnm, near) if k]; rrs = [r for r, k in zip(rrs, near) if k]
    origin = center - ext
    n = int(2 * ext / SPACING) + 1
    shape = (n, n, n)
    rrad = np.array([RADII.get(e, 1.9) for e in rel])
    core = ball_mask(shape, origin, rxyz, rrad)
    outer = ball_mask(shape, origin, rxyz, rrad + SKIN)
    skin = (outer & ~core).astype(np.float32); coref = core.astype(np.float32)
    pm = np.array([e in POLAR for e in rel]); hm = np.array([e in HYDROPHOBIC for e in rel])
    polar = render(shape, origin, rxyz[pm], SIGMA) if pm.any() else np.zeros(shape, np.float32)
    hyd = render(shape, origin, rxyz[hm], SIGMA) if hm.any() else np.zeros(shape, np.float32)
    rq = np.array([REC_CHARGES.get((rr, nm), 0.0) for rr, nm in zip(rrs, rnm)])
    charge = render(shape, origin, rxyz[rq != 0], SIGMA_Q, rq[rq != 0]) if np.any(rq != 0) else np.zeros(shape, np.float32)
    # scoring field for shape = favorable skin - penalty core
    score_shape = W["contact"] * skin - W_CLASH * coref
    return dict(origin=origin, shape=shape, score_shape=score_shape,
                polar=polar, hyd=hyd, charge=charge)


def _corr(a, b):
    return fftconvolve(a, b[::-1, ::-1, ::-1], mode="same")


def rigid_dock(rec, lig_xyz, lig_els, lig_q, n_rot=200, seed=0):
    L = lig_xyz - lig_xyz.mean(0)
    # seat ligand centroid at the box center so the FFT shift range covers the pocket
    shape, origin = rec["shape"], rec["origin"]
    vcenter = np.array(shape) // 2
    box_center = origin + vcenter * SPACING
    pm = np.array([e in POLAR for e in lig_els]); hm = np.array([e in HYDROPHOBIC for e in lig_els])
    R = rand_rotations(n_rot, seed)
    best = (-np.inf, None, None)
    for Ri in R:
        pos = box_center + L @ Ri.T
        occ = render(shape, origin, pos, SIGMA)
        total = _corr(rec["score_shape"], occ)
        if pm.any():
            total += W["polar"] * _corr(rec["polar"], render(shape, origin, pos[pm], SIGMA))
        if hm.any():
            total += W["hyd"] * _corr(rec["hyd"], render(shape, origin, pos[hm], SIGMA))
        if np.any(lig_q != 0):
            # complementarity: reward NEGATIVE product -> subtract overlap
            total -= W["elec"] * _corr(rec["charge"], render(shape, origin, pos, SIGMA_Q, lig_q))
        idx = np.unravel_index(np.argmax(total), total.shape)
        s = float(total[idx]); shift = (np.array(idx) - vcenter) * SPACING
        if s > best[0]:
            best = (s, Ri, shift)
    s, Rb, tb = best
    pred = box_center + L @ Rb.T + tb
    return pred, s


def sample_field(field, origin, xyz):
    """Nearest-voxel sample of a receptor field at atom positions."""
    ij = np.round((xyz - origin) / SPACING).astype(int)
    shp = np.array(field.shape)
    ok = np.all((ij >= 0) & (ij < shp), axis=1)
    out = np.zeros(len(xyz), np.float32)
    v = ij[ok]
    out[ok] = field[v[:, 0], v[:, 1], v[:, 2]]
    return out


def score_pose(rec, xyz, els, q):
    pm = np.array([e in POLAR for e in els]); hm = np.array([e in HYDROPHOBIC for e in els])
    o = rec["origin"]
    s = sample_field(rec["score_shape"], o, xyz).sum()
    if pm.any():
        s += W["polar"] * (sample_field(rec["polar"], o, xyz) * pm).sum()
    if hm.any():
        s += W["hyd"] * (sample_field(rec["hyd"], o, xyz) * hm).sum()
    s -= W["elec"] * (sample_field(rec["charge"], o, xyz) * q).sum()
    return float(s)


def substantial_rot_bonds(mol):
    """Rotatable bonds whose rotation sweeps a large arc (moving fragment reaches >~2.5 A
    from the bond axis => diameter >5 A), per the notes' ~5 A idea."""
    patt = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    conf = mol.GetConformer()
    P = conf.GetPositions()
    out = []
    for a1, a2 in mol.GetSubstructMatches(patt):
        b = mol.GetBondBetweenAtoms(a1, a2)
        em = Chem.FragmentOnBonds(mol, [b.GetIdx()], addDummies=False)
        frags = Chem.GetMolFrags(em)
        if len(frags) != 2:
            continue
        side = min(frags, key=len)          # smaller moving side
        if a2 in side:
            axis0, axis1 = a1, a2
        else:
            axis0, axis1 = a2, a1
        u = P[axis1] - P[axis0]; u = u / (np.linalg.norm(u) + 1e-9)
        reach = 0.0
        for at in side:
            v = P[at] - P[axis1]
            perp = np.linalg.norm(v - np.dot(v, u) * u)
            reach = max(reach, perp)
        if reach > 2.5 and len(side) >= 1:
            out.append((axis0, axis1))       # rotate side attached to axis1
    return out


def torsion_refine(rec, mol, pred_xyz, passes=2, n_ang=12):
    """Place pred_xyz into mol, greedily scan each substantial dihedral to maximize
    field overlap. Returns refined xyz (mol-atom order)."""
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, pred_xyz[i].tolist())
    bonds = substantial_rot_bonds(m)
    els = [a.GetSymbol().upper() for a in m.GetAtoms()]
    try:
        AllChem.ComputeGasteigerCharges(m); q = np.array([a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()])
        q = np.where(np.isfinite(q), q, 0.0)
    except Exception:
        q = np.zeros(m.GetNumAtoms())
    if not bonds:
        return conf.GetPositions(), 0
    for _ in range(passes):
        for a1, a2 in bonds:
            # need two reference neighbors to define a dihedral
            n1 = [n.GetIdx() for n in m.GetAtomWithIdx(a1).GetNeighbors() if n.GetIdx() != a2]
            n2 = [n.GetIdx() for n in m.GetAtomWithIdx(a2).GetNeighbors() if n.GetIdx() != a1]
            if not n1 or not n2:
                continue
            i, l = n1[0], n2[0]
            base = rmt.GetDihedralDeg(conf, i, a1, a2, l)
            best = (-np.inf, base)
            for d in np.linspace(0, 360, n_ang, endpoint=False):
                rmt.SetDihedralDeg(conf, i, a1, a2, l, base + d)
                sc = score_pose(rec, conf.GetPositions(), els, q)
                if sc > best[0]:
                    best = (sc, base + d)
            rmt.SetDihedralDeg(conf, i, a1, a2, l, best[1])
    return conf.GetPositions(), len(bonds)


def rmsd_sym(pred_xyz, ref_mol):
    """Symmetry-corrected in-place RMSD (no superposition)."""
    ref = ref_mol.GetConformer().GetPositions()
    if len(pred_xyz) != len(ref):
        return np.nan
    matches = ref_mol.GetSubstructMatches(ref_mol, uniquify=False, maxMatches=500)
    if not matches:
        matches = [tuple(range(len(ref)))]
    best = np.inf
    for mch in matches:
        d = pred_xyz - ref[list(mch)]
        r = np.sqrt((d * d).sum(1).mean())
        best = min(best, r)
    return float(best)


def dock_one(protein_pdb, start_sdf, cryst_sdf, n_rot=200):
    cry = load_mol(cryst_sdf)
    sta = load_mol(start_sdf)
    if cry is None or sta is None:
        return None
    if sta.GetNumAtoms() != cry.GetNumAtoms():
        return None
    center = cry.GetConformer().GetPositions().mean(0)
    ext = np.abs(cry.GetConformer().GetPositions() - center).max() + MARGIN
    rec = build_rec(protein_pdb, center, ext)
    lx, le, lq = mol_fields_input(sta)
    pred, score = rigid_dock(rec, lx, le, lq, n_rot=n_rot)
    rmsd_rigid = rmsd_sym(pred, cry)
    refined, nb = torsion_refine(rec, sta, pred)
    rmsd_tors = rmsd_sym(refined, cry)
    return dict(rmsd_rigid=rmsd_rigid, rmsd_tors=rmsd_tors, n_bonds=nb,
                grid=rec["shape"][0], score=score)


def iter_set(root, which):
    d = Path(root) / which
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        code = sub.name
        p = sub / f"{code}_protein.pdb"
        start = sub / f"{code}_ligand_start_conf.sdf"
        cry = sub / f"{code}_ligand.sdf"
        if p.exists() and start.exists() and cry.exists():
            yield code, p, start, cry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb-root", required=True)
    ap.add_argument("--which", default="posebusters_benchmark_set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rot", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    items = list(iter_set(a.pb_root, a.which))
    if a.limit:
        items = items[:a.limit]
    if a.smoke:
        items = items[:3]
    k, N = (int(x) for x in a.shard.split("/"))
    items = [c for i, c in enumerate(items) if i % N == (k - 1)]
    rows = []
    for j, (code, p, start, cry) in enumerate(items):
        try:
            r = dock_one(p, start, cry, n_rot=a.n_rot)
        except Exception as e:
            print(f"{code}: ERR {type(e).__name__} {e}", flush=True); continue
        if r is None:
            print(f"{code}: skip (parse/atom mismatch)", flush=True); continue
        r["id"] = code
        rows.append(r)
        print(f"{code}: rigid={r['rmsd_rigid']:.2f} tors={r['rmsd_tors']:.2f} "
              f"bonds={r['n_bonds']} grid={r['grid']}", flush=True)
    import json
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows))
    print(f"done: {len(rows)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
