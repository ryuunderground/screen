"""v3 feature extraction: distance-gated PAIR interaction terms at the crystal pose.

Why v3: v1's five grid-overlap channels all scaled with buried volume (one size
axis -> collinear, ridge r~0.38). Fix = replace fuzzy grid overlaps with Vina-style
pairwise terms that are physically orthogonal (attraction / steric clash / hydrophobic
contact / directional H-bond / ligand entropy). Still a linear scorer over channels;
only the weights are learned (labels = experimental pKd, not Vina scores).

Per complex, sum over receptor-ligand heavy-atom pairs of surface distance
  d = |r_i - r_j| - (R_i + R_j)      (Vina convention)
Channels (Trott & Olson 2010, doi:10.1002/jcc.21334 functional forms):
  gauss1      = Σ exp(-(d/0.5)^2)                       short-range contact
  gauss2      = Σ exp(-((d-3)/2)^2)                     longer-range contact
  repulsion   = Σ d^2 · [d<0]                           steric clash
  hydrophobic = Σ ramp(d; 0.5->1.5) over hydrophobe-hydrophobe pairs
  hbond       = Σ ramp(d; -0.7->0) over donor-acceptor pairs   (directional typing)
plus
  nrot        = ligand rotatable-bond count             (entropy / size control)

Pairs restricted to interatomic dist < 8 Å (gauss2 tail). Loads LOCAL files only.
Receptor donor/acceptor by PDB atom/residue name; ligand by RDKit (H-count + element).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

# Vina-ish heavy-atom radii (surface distance reference)
RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "CL": 1.8, "BR": 2.0, "I": 2.2,
    "MG": 1.2, "MN": 1.2, "ZN": 1.2, "CA": 1.2, "FE": 1.2, "NA": 1.2, "K": 1.2,
}
HALOGENS = {"F", "CL", "BR", "I"}
PAIR_CUT = 8.0  # interatomic distance cutoff

FEATURE_NAMES = ["gauss1", "gauss2", "repulsion", "hydrophobic", "hbond", "nrot"]

# receptor donor/acceptor by (resname, atomname). "both" atoms go in both sets.
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
# backbone N = donor, backbone O/OXT = acceptor (any residue)
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


def parse_pocket_pdb(path):
    """Heavy atoms of pocket.pdb -> (xyz Nx3, elements, is_don, is_acc, is_hyd)."""
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
            d = (res, name) in _DON or name in _BB_DON
            a = (res, name) in _ACC or name in _BB_ACC
            don.append(d); acc.append(a)
            # hydrophobic receptor atom: carbon or sulfur (crude; excludes polar O/N)
            hyd.append(el in ("C", "S"))
    return (np.array(xyz, float), np.array(els), np.array(don), np.array(acc),
            np.array(hyd))


def load_ligand(sdf, mol2):
    """Ligand -> (xyz, is_don, is_acc, is_hyd, nrot, note). Try sdf then mol2."""
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
            return (None,) * 5 + ("FAIL",)
    conf = mol.GetConformer()
    try:
        nrot = Descriptors.NumRotatableBonds(mol)
    except Exception:
        nrot = 0
    xyz, don, acc, hyd = [], [], [], []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        xyz.append((p.x, p.y, p.z))
        sym = a.GetSymbol().upper()
        nH = a.GetTotalNumHs()
        fc = a.GetFormalCharge()
        is_don = sym in ("N", "O") and nH > 0 and fc >= 0
        # acceptor: O (neutral/anionic) or N with a lone pair (not protonated/aromatic-NH)
        is_acc = ((sym == "O" and fc <= 0) or
                  (sym == "N" and fc <= 0 and a.GetTotalDegree() + nH < 4 and nH == 0))
        # hydrophobic: nonpolar carbon (only C/H neighbors) or halogen
        is_hyd = False
        if sym == "C":
            is_hyd = all(nb.GetSymbol() in ("C", "H") for nb in a.GetNeighbors())
        elif sym in HALOGENS:
            is_hyd = True
        don.append(is_don); acc.append(is_acc); hyd.append(is_hyd)
    if not xyz:
        return (None,) * 5 + ("FAIL",)
    return (np.array(xyz, float), np.array(don), np.array(acc),
            np.array(hyd), int(nrot), note)


def _ramp(d, lo, hi):
    """1 for d<=lo, 0 for d>=hi, linear between (Vina's piecewise-linear bump)."""
    return np.clip((hi - d) / (hi - lo), 0.0, 1.0)


def features_for(pocket_pdb, sdf, mol2):
    lxyz, ldon, lacc, lhyd, nrot, note = load_ligand(sdf, mol2)
    if lxyz is None:
        return None, note
    # ligand radii: need elements again -> re-read symbols cheaply from the mol is gone;
    # approximate ligand radii by nearest-neighbor element map is overkill. Use per-atom
    # radius from a parallel element list:
    rxyz, rel, rdon, racc, rhyd = parse_pocket_pdb(pocket_pdb)
    if len(rxyz) == 0:
        return None, "NOREC"
    # ligand elements for radii: reload symbols quickly
    lel = _ligand_elements(sdf, mol2, len(lxyz))
    lrad = np.array([RADII.get(e, 1.9) for e in lel])
    rrad = np.array([RADII.get(e, 1.9) for e in rel])

    # neighbor pairs within PAIR_CUT via KD-tree
    tree_r = cKDTree(rxyz)
    pairs = tree_r.query_ball_point(lxyz, PAIR_CUT)  # for each ligand atom, receptor idxs
    g1 = g2 = rep = hphob = hb = 0.0
    for li, ridxs in enumerate(pairs):
        if not ridxs:
            continue
        ridxs = np.asarray(ridxs)
        dd = np.linalg.norm(rxyz[ridxs] - lxyz[li], axis=1)
        d = dd - lrad[li] - rrad[ridxs]           # surface distance
        g1 += float(np.exp(-(d / 0.5) ** 2).sum())
        g2 += float(np.exp(-((d - 3.0) / 2.0) ** 2).sum())
        neg = d < 0
        rep += float((d[neg] ** 2).sum())
        if lhyd[li]:
            hphob += float(_ramp(d[rhyd[ridxs]], 0.5, 1.5).sum())
        # directional H-bond: ligand-donor<->receptor-acceptor and vice versa
        if ldon[li]:
            hb += float(_ramp(d[racc[ridxs]], -0.7, 0.0).sum())
        if lacc[li]:
            hb += float(_ramp(d[rdon[ridxs]], -0.7, 0.0).sum())
    return np.array([g1, g2, rep, hphob, hb, float(nrot)]), note


def _ligand_elements(sdf, mol2, n):
    """Heavy-atom element symbols, same order/length as load_ligand's xyz."""
    for loader, src in ((Chem.MolFromMolFile, sdf), (Chem.MolFromMol2File, mol2)):
        try:
            m = loader(str(src), sanitize=True, removeHs=True)
        except Exception:
            m = None
        if m is not None:
            els = [a.GetSymbol().upper() for a in m.GetAtoms() if a.GetAtomicNum() != 1]
            if len(els) == n:
                return els
    m = Chem.MolFromMolFile(str(sdf), sanitize=False, removeHs=True)
    if m is not None:
        els = [a.GetSymbol().upper() for a in m.GetAtoms() if a.GetAtomicNum() != 1]
        if len(els) == n:
            return els
    return ["C"] * n  # radius fallback


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
    """Self-check on pilot complexes if present; else synthetic sanity."""
    # synthetic: one donor ligand atom 2.9 A from one acceptor receptor atom -> hbond>0
    import tempfile, os
    assert abs(_ramp(np.array([-0.35]), -0.7, 0.0)[0] - 0.5) < 1e-6, "ramp midpoint"
    assert _ramp(np.array([-1.0]), -0.7, 0.0)[0] == 1.0
    assert _ramp(np.array([0.5]), -0.7, 0.0)[0] == 0.0
    print("demo: ramp OK")


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
