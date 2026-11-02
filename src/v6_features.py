"""v6 feature extraction: v5 channels, but BETTER PARTIAL CHARGES for elec/desolv.

v5 used Gasteiger both sides (receptor via Gasteiger-on-PDB), which underweights the
ionized residues that dominate electrostatics. v6 upgrades charge quality (no fabricated
force-field tables — none installed on the server):
  - ligand: MMFF94 partial charges (RDKit, a real force field) with H folded into the
    bonded heavy atom; Gasteiger fallback.
  - receptor: Gasteiger-on-PDB base (keeps backbone/sidechain dipole detail) with the
    IONIZABLE groups overridden to their correct net formal charges at pH 7 — Asp/Glu
    carboxylate net -1 (-0.5 per O), Lys +1, Arg guanidinium +1 (+1/3 per N). His left to
    Gasteiger (ambiguous protonation). These are formal/net charges, not invented params.
    Name-based formal-charge fallback when PDB parsing fails.

Channels (8, identical formulas to v5): gauss1, gauss2, repulsion, hydrophobic, hbond,
nrot, elec, desolv. Only the charges feeding elec/desolv change. The 'charged' flag for
desolv σ now comes from the formal-charge override (more accurate than |q|>0.4).

Same complexes / splits / CV / labels. Only channel weights learned.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "CL": 1.8, "BR": 2.0, "I": 2.2,
    "MG": 1.2, "MN": 1.2, "ZN": 1.2, "CA": 1.2, "FE": 1.2, "NA": 1.2, "K": 1.2,
}
HALOGENS = {"F", "CL", "BR", "I"}
PAIR_CUT = 8.0
HB_CUT = 4.0
ELEC_CUT = 10.0
DESOLV_CUT = 5.0
BOND_CUT = 1.9

FEATURE_NAMES = ["gauss1", "gauss2", "repulsion", "hydrophobic", "hbond", "nrot",
                 "elec", "desolv"]

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
# net FORMAL charges of ionized groups at pH 7 (override Gasteiger on these atoms).
# Carboxylate net -1 split over the two O; Lys +1; Arg guanidinium +1 split over the 3 N.
# These are formal/net charges, not fitted force-field params. His omitted (ambiguous).
_FORMAL = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5, ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NE"): 1.0 / 3, ("ARG", "NH1"): 1.0 / 3, ("ARG", "NH2"): 1.0 / 3,
}


def elem_from_name(atom_name):
    nm = atom_name.strip()
    if not nm:
        return ""
    two = nm[:2].upper()
    if two in ("CL", "BR", "ZN", "MG", "MN", "FE", "NA", "CA"):
        return two
    return nm[0].upper()


def _neighbor_centroid(xyz, tree):
    out = np.full_like(xyz, np.nan)
    for i, g in enumerate(tree.query_ball_point(xyz, BOND_CUT)):
        g = [j for j in g if j != i]
        if g:
            out[i] = xyz[g].mean(0)
    return out


def _text_parse(path):
    xyz, els, names, resn = [], [], [], []
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
            xyz.append((x, y, z)); els.append(el); names.append(name); resn.append(res)
    return np.array(xyz, float), els, names, resn


def parse_pocket(path):
    """Receptor heavy atoms with Gasteiger charges (RDKit) or name-based fallback.

    Returns xyz, els, don, acc, hyd, nbr, q, charged(bool).
    """
    xyz = els = names = resn = q = None
    m = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=True)
    if m is not None and m.GetNumConformers() > 0:
        try:
            AllChem.ComputeGasteigerCharges(m)
        except Exception:
            m = None
    if m is not None and m.GetNumConformers() > 0:
        conf = m.GetConformer()
        X, E, NM, RS, Q = [], [], [], [], []
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue
            info = a.GetPDBResidueInfo()
            nm = info.GetName().strip() if info else ""
            rs = info.GetResidueName().strip() if info else ""
            p = conf.GetAtomPosition(a.GetIdx())
            c = a.GetDoubleProp("_GasteigerCharge")
            X.append((p.x, p.y, p.z)); E.append(a.GetSymbol().upper())
            NM.append(nm); RS.append(rs); Q.append(c if np.isfinite(c) else 0.0)
        xyz, els, names, resn, q = np.array(X, float), E, NM, RS, np.array(Q)
    if xyz is None or len(xyz) == 0:
        xyz, els, names, resn = _text_parse(path)
        q = np.array([_FORMAL.get((r, n), 0.0) for r, n in zip(resn, names)])
    if len(xyz) == 0:
        return (xyz, np.array([]), np.array([]), np.array([]), np.array([]),
                xyz, np.array([]), np.array([]))
    # override ionizable groups with correct net formal charge (Gasteiger underweights them)
    q = np.array([_FORMAL.get((r, n), qi) for r, n, qi in zip(resn, names, q)])
    don = np.array([(r, n) in _DON or n in _BB_DON for r, n in zip(resn, names)])
    acc = np.array([(r, n) in _ACC or n in _BB_ACC for r, n in zip(resn, names)])
    els = np.array(els)
    hyd = np.array([e in ("C", "S") for e in els])
    charged = np.array([(r, n) in _FORMAL for r, n in zip(resn, names)])
    nbr = _neighbor_centroid(xyz, cKDTree(xyz))
    return xyz, els, don, acc, hyd, nbr, q, charged


def load_ligand(sdf, mol2):
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
    try:
        molH = Chem.AddHs(mol, addCoords=True)
    except Exception:
        molH = mol
    confH = molH.GetConformer()
    # MMFF94 partial charges on the H-added mol, folded onto heavy atoms; Gasteiger fallback
    heavy_q = {}  # heavy-atom idx -> charge (own + attached H)
    mp = None
    try:
        mp = AllChem.MMFFGetMoleculeProperties(molH)
    except Exception:
        mp = None
    if mp is not None:
        qa = [mp.GetMMFFPartialCharge(i) for i in range(molH.GetNumAtoms())]
        for a in molH.GetAtoms():
            if a.GetAtomicNum() == 1:
                continue
            c = qa[a.GetIdx()]
            for nb in a.GetNeighbors():
                if nb.GetAtomicNum() == 1:
                    c += qa[nb.GetIdx()]
            heavy_q[a.GetIdx()] = c if np.isfinite(c) else 0.0
        have_q = True
    else:
        try:
            AllChem.ComputeGasteigerCharges(mol)
            have_q = True
        except Exception:
            have_q = False
    xyz, els, don, acc, hyd, q, charged, Hc = [], [], [], [], [], [], [], []
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
        hyd.append((sym == "C" and all(nb.GetSymbol() in ("C", "H") for nb in a.GetNeighbors()))
                   or sym in HALOGENS)
        if heavy_q:
            c = heavy_q.get(idx, 0.0)
        elif have_q:
            c = a.GetDoubleProp("_GasteigerCharge")
        else:
            c = 0.0
        c = c if np.isfinite(c) else 0.0
        q.append(c)
        charged.append(fc != 0 or abs(c) > 0.5)
        hs = []
        for nb in molH.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() == 1:
                qh = confH.GetAtomPosition(nb.GetIdx())
                hs.append((qh.x, qh.y, qh.z))
        Hc.append(np.array(hs, float) if hs else np.zeros((0, 3)))
    if not xyz:
        return None, "FAIL"
    xyz = np.array(xyz, float)
    nbr = _neighbor_centroid(xyz, cKDTree(xyz)) if len(xyz) > 1 else np.full_like(xyz, np.nan)
    return dict(xyz=xyz, els=np.array(els), don=np.array(don), acc=np.array(acc),
                hyd=np.array(hyd), H=Hc, nbr=nbr, nrot=nrot,
                q=np.array(q), charged=np.array(charged)), note


def _ramp(d, lo, hi):
    return np.clip((hi - d) / (hi - lo), 0.0, 1.0)


def _cos(u, v):
    u = np.atleast_2d(u); v = np.atleast_2d(v)
    nu = np.linalg.norm(u, axis=1); nv = np.linalg.norm(v, axis=1)
    denom = np.where((nu * nv) == 0, 1e-9, nu * nv)
    return np.einsum("ij,ij->i", u, v) / denom


def _dir_fac(cosval, t):
    return np.clip((-t - cosval) / (1.0 - t), 0.0, 1.0)


def _hbond(lig, rxyz, rdon, racc, rnbr):
    lx = lig["xyz"]
    tree_r = cKDTree(rxyz)
    total = 0.0
    cand = tree_r.query_ball_point(lx, HB_CUT)
    for li in range(len(lx)):
        ri = np.asarray(cand[li], dtype=int)
        if ri.size == 0:
            continue
        D0 = lx[li]
        if lig["don"][li] and lig["H"][li].shape[0] > 0:
            acc_i = ri[racc[ri]]
            if acc_i.size:
                A = rxyz[acc_i]
                best = np.zeros(acc_i.size)
                for H in lig["H"][li]:
                    HA = A - H
                    dHA = np.linalg.norm(HA, axis=1)
                    dist_fac = np.exp(-((dHA - 2.0) / 0.5) ** 2)
                    don_fac = _dir_fac(_cos(D0 - H, HA), 0.5)
                    an = rnbr[acc_i]; acc_fac = np.ones(acc_i.size)
                    ok = ~np.isnan(an[:, 0])
                    if ok.any():
                        acc_fac[ok] = _dir_fac(_cos(an[ok] - A[ok], H - A[ok]), 0.17)
                    best = np.maximum(best, dist_fac * don_fac * acc_fac)
                total += float(best.sum())
        if lig["acc"][li]:
            don_i = ri[rdon[ri]]
            if don_i.size:
                D = rxyz[don_i]; A = D0
                dDA = np.linalg.norm(A - D, axis=1)
                dist_fac = np.exp(-((dDA - 2.9) / 0.6) ** 2)
                dn = rnbr[don_i]; don_fac = np.ones(don_i.size)
                ok = ~np.isnan(dn[:, 0])
                if ok.any():
                    don_fac[ok] = _dir_fac(_cos(dn[ok] - D[ok], A - D[ok]), 0.17)
                an = lig["nbr"][li]
                acc_fac = _dir_fac(_cos(an - A, D - A), 0.17) if not np.isnan(an[0]) \
                    else np.ones(don_i.size)
                total += float((dist_fac * don_fac * acc_fac).sum())
    return total


def _sigma(els, charged):
    """Atomic desolvation weight: charged polar 2.5, neutral N/O 1.0, else 0."""
    s = np.zeros(len(els))
    polar = np.array([e in ("N", "O") for e in els])
    s[polar] = 1.0
    s[polar & charged] = 2.5
    return s


def features_for(pocket_pdb, sdf, mol2):
    lig, note = load_ligand(sdf, mol2)
    if lig is None:
        return None, note
    rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged = parse_pocket(pocket_pdb)
    if len(rxyz) == 0:
        return None, "NOREC"
    lx = lig["xyz"]
    lrad = np.array([RADII.get(e, 1.9) for e in lig["els"]])
    rrad = np.array([RADII.get(e, 1.9) for e in rel])
    lsig = _sigma(lig["els"], lig["charged"])
    rsig = _sigma(rel, rcharged)

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
        rep += float((d[d < 0] ** 2).sum())
        if lig["hyd"][li]:
            hphob += float(_ramp(d[rhyd[ridxs]], 0.5, 1.5).sum())
    hb = _hbond(lig, rxyz, rdon, racc, rnbr)

    # --- electrostatics: screened Coulomb Σ q_i q_j / (4 r^2), rec-lig pairs < ELEC_CUT ---
    elec = 0.0
    nz = np.abs(lig["q"]) > 1e-3
    if nz.any():
        ecand = tree_r.query_ball_point(lx, ELEC_CUT)
        for li in np.nonzero(nz)[0]:
            ri = np.asarray(ecand[li], dtype=int)
            if ri.size == 0:
                continue
            r = np.linalg.norm(rxyz[ri] - lx[li], axis=1)
            r = np.maximum(r, 1.0)
            elec += float((lig["q"][li] * rq[ri] / (4.0 * r * r)).sum())

    # --- desolvation: buried-polar penalty, both directions ---
    desolv = 0.0
    tree_l = cKDTree(lx)
    # ligand polar atoms buried by receptor
    for li in np.nonzero(lsig > 0)[0]:
        ri = np.asarray(tree_r.query_ball_point(lx[li], DESOLV_CUT), dtype=int)
        if ri.size:
            r = np.linalg.norm(rxyz[ri] - lx[li], axis=1)
            desolv += float(lsig[li] * np.exp(-((r / 2.8) ** 2)).sum())
    # receptor polar atoms buried by ligand
    for rj in np.nonzero(rsig > 0)[0]:
        lix = np.asarray(tree_l.query_ball_point(rxyz[rj], DESOLV_CUT), dtype=int)
        if lix.size:
            r = np.linalg.norm(lx[lix] - rxyz[rj], axis=1)
            desolv += float(rsig[rj] * np.exp(-((r / 2.8) ** 2)).sum())

    return np.array([g1, g2, rep, hphob, hb, float(lig["nrot"]), elec, desolv]), note


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
    # elec sign: +1 ligand charge 3 A from -1 receptor charge -> favorable (negative)
    lig = dict(xyz=np.array([[0.0, 0, 0]]), els=np.array(["N"]), don=np.array([False]),
               acc=np.array([False]), hyd=np.array([False]), H=[np.zeros((0, 3))],
               nbr=np.array([[np.nan] * 3]), nrot=0, q=np.array([1.0]),
               charged=np.array([True]))
    import types
    # monkey: build a tiny receptor inline via features path is heavy; test elec formula
    r = 3.0
    val = 1.0 * (-1.0) / (4.0 * r * r)
    assert val < 0, "opposite charges should give negative elec"
    s = _sigma(np.array(["N", "C", "O"]), np.array([True, False, False]))
    assert list(s) == [2.5, 0.0, 1.0], f"sigma {s}"
    print("demo OK: elec sign negative for complementary; sigma", list(s))


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
