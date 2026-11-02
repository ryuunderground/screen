"""v1 feature extraction: multi-channel field-overlap scalars AT THE CRYSTAL POSE.

For each PDBbind complex: center a grid on the ligand, render receptor (pocket.pdb)
and ligand (sdf) as multi-channel fields, compute per-channel OVERLAP scalars
(sum(rec_ch * lig_ch) * dV). No docking search.

Channels (feature vector, one row per complex):
  contact  = Σ rec_skin   · lig_occ      (burial/shape contact, +)
  clash    = Σ rec_core   · lig_occ      (steric clash penalty, sign handled by model)
  polar    = Σ rec_polar  · lig_polar    (N/O vs N/O, +)
  hydrophob= Σ rec_hyd    · lig_hyd      (C/S/halogen vs same, +)
  elec     = Σ rec_charge · lig_charge   (signed; complementarity => negative is favorable)

Reuses render()/ball_mask() logic from fft_dock_pilot.py; loads LOCAL files only.

Receptor charge scheme (v1 proxy, by PDB atom name):
  Asp/Glu carboxylate O (OD1 OD2 OE1 OE2): -0.5 each
  Lys NZ: +1.0 ; Arg NE/NH1/NH2: +0.33 each ; His ND1/NE2: +0.5 each
  everything else: 0.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

RADII = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "CL": 1.8, "BR": 2.0, "I": 2.2,
    "MG": 1.2, "MN": 1.2, "ZN": 1.2, "CA": 1.2, "FE": 1.2, "NA": 1.2, "K": 1.2,
}
HALOGENS = {"F", "CL", "BR", "I"}
POLAR = {"N", "O"}
HYDROPHOBIC = {"C", "S", "F", "CL", "BR", "I"}

SPACING = 0.8
SIGMA = 0.8          # occupancy blob width
SIGMA_Q = 1.0        # charge spreads wider than density (notes/channels.md §3)
SKIN = 2.0           # receptor surface-shell thickness
MARGIN = 4.0         # receptor reach beyond ligand extent

REC_CHARGES = {
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
    ("LYS", "NZ"): 1.0,
    ("ARG", "NE"): 0.33, ("ARG", "NH1"): 0.33, ("ARG", "NH2"): 0.33,
    ("HIS", "ND1"): 0.5, ("HIS", "NE2"): 0.5,
}

FEATURE_NAMES = ["contact", "clash", "polar", "hydrophob", "elec"]


def elem_from_name(atom_name, resname):
    """Element from a PDB atom name (cols 13-16). Heavy atoms only; caller drops H."""
    nm = atom_name.strip()
    if not nm:
        return ""
    # two-letter elements present in protein/cofactors
    two = nm[:2].upper()
    if two in ("CL", "BR", "ZN", "MG", "MN", "FE", "NA", "CA"):
        return two
    return nm[0].upper()


def parse_pocket_pdb(path):
    """Return (xyz Nx3, elements, names, resnames) for heavy atoms of pocket.pdb."""
    xyz, els, names, resn = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            name = line[12:16]
            el_col = line[76:78].strip()
            res = line[17:20].strip()
            el = el_col.upper() if el_col else elem_from_name(name, res)
            if el == "H" or el == "D":
                continue
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                continue
            xyz.append((x, y, z)); els.append(el); names.append(name.strip()); resn.append(res)
    return np.array(xyz, float), els, names, resn


def load_ligand(sdf, mol2):
    """Load ligand -> (xyz, elements, gasteiger charges) heavy atoms. Try sdf, then mol2.

    Returns (xyz, els, charges, note). note flags fallbacks/failures.
    """
    mol = None
    note = "sdf"
    try:
        mol = Chem.MolFromMolFile(str(sdf), sanitize=True, removeHs=True)
    except Exception:
        mol = None
    if mol is None:
        try:
            mol = Chem.MolFromMol2File(str(mol2), sanitize=True, removeHs=True)
            note = "mol2"
        except Exception:
            mol = None
    if mol is None:
        # last resort: coords only, no charges
        mol = Chem.MolFromMolFile(str(sdf), sanitize=False, removeHs=False)
        note = "nosanitize"
        if mol is None:
            return None, None, None, "FAIL"
    conf = mol.GetConformer()
    try:
        AllChem.ComputeGasteigerCharges(mol)
        have_q = True
    except Exception:
        have_q = False
    xyz, els, q = [], [], []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        p = conf.GetAtomPosition(a.GetIdx())
        xyz.append((p.x, p.y, p.z))
        els.append(a.GetSymbol().upper())
        if have_q:
            c = a.GetDoubleProp("_GasteigerCharge")
            q.append(c if np.isfinite(c) else 0.0)
        else:
            q.append(0.0)
    if not xyz:
        return None, None, None, "FAIL"
    return np.array(xyz, float), els, np.array(q, float), note


def render(shape, origin, coords, sigma, weights=None):
    """Gaussian field on the grid; optional per-atom signed weight (for charge)."""
    f = np.zeros(shape, dtype=np.float32)
    cut = 4 * sigma
    if weights is None:
        weights = np.ones(len(coords))
    for a, w in zip(coords, weights):
        if w == 0:
            continue
        lo = np.floor((a - cut - origin) / SPACING).astype(int)
        hi = np.ceil((a + cut - origin) / SPACING).astype(int)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(shape) - 1)
        if np.any(lo > hi):
            continue
        sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        gx = origin[0] + np.arange(lo[0], hi[0] + 1) * SPACING
        gy = origin[1] + np.arange(lo[1], hi[1] + 1) * SPACING
        gz = origin[2] + np.arange(lo[2], hi[2] + 1) * SPACING
        d2 = ((gx - a[0])[:, None, None] ** 2 + (gy - a[1])[None, :, None] ** 2
              + (gz - a[2])[None, None, :] ** 2)
        f[sl] += (w * np.exp(-d2 / (2 * sigma * sigma))).astype(np.float32)
    return f


def ball_mask(shape, origin, coords, radii):
    m = np.zeros(shape, dtype=bool)
    for a, rad in zip(coords, radii):
        lo = np.floor((a - rad - origin) / SPACING).astype(int)
        hi = np.ceil((a + rad - origin) / SPACING).astype(int)
        lo = np.maximum(lo, 0); hi = np.minimum(hi, np.array(shape) - 1)
        if np.any(lo > hi):
            continue
        sl = tuple(slice(lo[d], hi[d] + 1) for d in range(3))
        gx = origin[0] + np.arange(lo[0], hi[0] + 1) * SPACING
        gy = origin[1] + np.arange(lo[1], hi[1] + 1) * SPACING
        gz = origin[2] + np.arange(lo[2], hi[2] + 1) * SPACING
        d2 = ((gx - a[0])[:, None, None] ** 2 + (gy - a[1])[None, :, None] ** 2
              + (gz - a[2])[None, None, :] ** 2)
        m[sl] |= d2 <= rad * rad
    return m


def features_for(pocket_pdb, sdf, mol2):
    """Compute the 5 overlap features for one complex. Returns (vec, note) or (None, note)."""
    lig_xyz, lig_el, lig_q, note = load_ligand(sdf, mol2)
    if lig_xyz is None:
        return None, note
    rec_xyz, rec_el, rec_names, rec_resn = parse_pocket_pdb(pocket_pdb)
    if len(rec_xyz) == 0:
        return None, "NOREC"

    centroid = lig_xyz.mean(0)
    ext = np.abs(lig_xyz - centroid).max() + MARGIN
    origin = centroid - ext
    n = int(2 * ext / SPACING) + 1
    shape = (n, n, n)
    dV = SPACING ** 3

    # keep receptor atoms near the box
    near = np.linalg.norm(rec_xyz - centroid, axis=1) < ext + SKIN + 2
    rxyz = rec_xyz[near]
    rel = [e for e, k in zip(rec_el, near) if k]
    rnm = [m for m, k in zip(rec_names, near) if k]
    rrs = [r for r, k in zip(rec_resn, near) if k]
    if len(rxyz) == 0:
        return None, "NOREC"
    rrad = np.array([RADII.get(e, 1.9) for e in rel])

    # receptor masks + fields
    core = ball_mask(shape, origin, rxyz, rrad)
    outer = ball_mask(shape, origin, rxyz, rrad + SKIN)
    skin = (outer & ~core).astype(np.float32)
    core = core.astype(np.float32)
    r_polar_m = np.array([e in POLAR for e in rel])
    r_hyd_m = np.array([e in HYDROPHOBIC for e in rel])
    rec_polar = render(shape, origin, rxyz[r_polar_m], SIGMA) if r_polar_m.any() else np.zeros(shape, np.float32)
    rec_hyd = render(shape, origin, rxyz[r_hyd_m], SIGMA) if r_hyd_m.any() else np.zeros(shape, np.float32)
    rq = np.array([REC_CHARGES.get((rr, nm), 0.0) for rr, nm in zip(rrs, rnm)])
    rec_charge = render(shape, origin, rxyz[rq != 0], SIGMA_Q, rq[rq != 0]) if np.any(rq != 0) else np.zeros(shape, np.float32)

    # ligand fields
    lig_occ = render(shape, origin, lig_xyz, SIGMA)
    l_polar_m = np.array([e in POLAR for e in lig_el])
    l_hyd_m = np.array([e in HYDROPHOBIC for e in lig_el])
    lig_polar = render(shape, origin, lig_xyz[l_polar_m], SIGMA) if l_polar_m.any() else np.zeros(shape, np.float32)
    lig_hyd = render(shape, origin, lig_xyz[l_hyd_m], SIGMA) if l_hyd_m.any() else np.zeros(shape, np.float32)
    lig_charge = render(shape, origin, lig_xyz, SIGMA_Q, lig_q)

    contact = float((skin * lig_occ).sum()) * dV
    clash = float((core * lig_occ).sum()) * dV
    polar = float((rec_polar * lig_polar).sum()) * dV
    hyd = float((rec_hyd * lig_hyd).sum()) * dV
    elec = float((rec_charge * lig_charge).sum()) * dV
    return np.array([contact, clash, polar, hyd, elec], float), note


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
            notes.append(f"{code}\tERR {type(e).__name__}"); continue
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "screen/data"))
    ap.add_argument("--ids", required=True, help="text file, one pdb code per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    if a.smoke:
        ids = ids[:3]
    run_shard(a.data_root, ids, a.out, a.shard)


if __name__ == "__main__":
    sys.exit(main())
