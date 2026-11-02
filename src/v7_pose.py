"""v7 pose test (moderate): does the v5 channel scorer recognize the native pose?

Not blind docking. A scoring-FUNNEL test on PDBbind test complexes we already have:
for each complex, take the crystal ligand (native, RMSD 0) and generate rigid-body decoy
poses spanning an RMSD ladder; score every pose with the frozen v5 scorer (8 pairwise
channels · learned weights); ask whether the score funnels toward native.

Metrics per complex:
  - top1<2A : is the single best-scoring pose within 2 A RMSD of native?
  - rho     : Spearman(score, RMSD) — want strongly NEGATIVE (score high when RMSD low).
Aggregated over the sample. Compares the FULL v5 scorer vs a shape-only ablation
(gauss1/gauss2/repulsion only) to see whether the physics channels (hbond/elec/desolv)
sharpen native-pose recognition — the place they should help, unlike affinity r.

Reuses v5_features for typing/charges/feature formulas (v5 untouched). RMSD is plain
in-place coordinate RMSD (no symmetry correction — moderate test).
"""
import argparse
import json
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v5_features as V5

RNG_SEED = 0


def fit_v5_model(feat_npz, manifest):
    """Freeze the v5 scorer: StandardScaler stats + Ridge coefs on the train split."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    d = np.load(feat_npz, allow_pickle=True)
    names = [str(x) for x in d["feature_names"]]
    df = pd.DataFrame(d["X"], columns=names); df["id"] = [str(i) for i in d["ids"]]
    man = pd.read_csv(manifest)
    m = man.merge(df, on="id").dropna(subset=names)
    tr = m[m.split == "train"]
    X = tr[names].values; y = tr["pKd"].values
    sc = StandardScaler().fit(X)
    ridge = Ridge(alpha=1.0).fit(sc.transform(X), y)
    return names, sc.mean_, sc.scale_, ridge.coef_, ridge.intercept_


def score_feats(F, mean, scale, coef, b, mask=None):
    """F: (...,8) raw features -> score. mask: bool over channels for ablation."""
    Z = (F - mean) / scale
    w = coef.copy()
    if mask is not None:
        w = w * mask
    return Z @ w + b


def rigid_decoys(xyz, n, rng, max_rot=np.pi, max_trans=5.0):
    """n rigid-body perturbations of xyz about its centroid. Returns list of (coords, R, t)."""
    c = xyz.mean(0)
    out = []
    for _ in range(n):
        # random axis-angle rotation, magnitude scaled to span small..large
        ang = rng.uniform(0, max_rot) * rng.uniform(0.2, 1.0)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        t = rng.normal(size=3); t = t / np.linalg.norm(t) * rng.uniform(0.3, max_trans)
        new = (xyz - c) @ R.T + c + t
        out.append(new)
    return out


def compute8(lig, rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged, lig_xyz):
    """The 8 v5 features for ligand at coords lig_xyz (typing/charges from lig dict)."""
    lrad = np.array([V5.RADII.get(e, 1.9) for e in lig["els"]])
    rrad = np.array([V5.RADII.get(e, 1.9) for e in rel])
    lsig = V5._sigma(lig["els"], lig["charged"])
    rsig = V5._sigma(rel, rcharged)
    tree_r = cKDTree(rxyz)
    pairs = tree_r.query_ball_point(lig_xyz, V5.PAIR_CUT)
    g1 = g2 = rep = hphob = 0.0
    for li, ridxs in enumerate(pairs):
        if not ridxs:
            continue
        ridxs = np.asarray(ridxs)
        dd = np.linalg.norm(rxyz[ridxs] - lig_xyz[li], axis=1)
        d = dd - lrad[li] - rrad[ridxs]
        g1 += float(np.exp(-(d / 0.5) ** 2).sum())
        g2 += float(np.exp(-((d - 3.0) / 2.0) ** 2).sum())
        rep += float((d[d < 0] ** 2).sum())
        if lig["hyd"][li]:
            hphob += float(V5._ramp(d[rhyd[ridxs]], 0.5, 1.5).sum())
    # build a transient lig view with moved coords for hbond (needs H + nbr moved too)
    hb = V5._hbond(lig, rxyz, rdon, racc, rnbr)
    elec = 0.0
    nz = np.abs(lig["q"]) > 1e-3
    if nz.any():
        ecand = tree_r.query_ball_point(lig_xyz, V5.ELEC_CUT)
        for li in np.nonzero(nz)[0]:
            ri = np.asarray(ecand[li], dtype=int)
            if ri.size == 0:
                continue
            r = np.maximum(np.linalg.norm(rxyz[ri] - lig_xyz[li], axis=1), 1.0)
            elec += float((lig["q"][li] * rq[ri] / (4.0 * r * r)).sum())
    desolv = 0.0
    tree_l = cKDTree(lig_xyz)
    for li in np.nonzero(lsig > 0)[0]:
        ri = np.asarray(tree_r.query_ball_point(lig_xyz[li], V5.DESOLV_CUT), dtype=int)
        if ri.size:
            r = np.linalg.norm(rxyz[ri] - lig_xyz[li], axis=1)
            desolv += float(lsig[li] * np.exp(-((r / 2.8) ** 2)).sum())
    for rj in np.nonzero(rsig > 0)[0]:
        lix = np.asarray(tree_l.query_ball_point(rxyz[rj], V5.DESOLV_CUT), dtype=int)
        if lix.size:
            r = np.linalg.norm(lig_xyz[lix] - rxyz[rj], axis=1)
            desolv += float(rsig[rj] * np.exp(-((r / 2.8) ** 2)).sum())
    return np.array([g1, g2, rep, hphob, hb, float(lig["nrot"]), elec, desolv])


def process_one(args):
    code, data_root, model = args
    names, mean, scale, coef, b = model
    paths = V5.find_complex(data_root, code)
    if paths is None or not paths[0].exists():
        return None
    try:
        lig, note = V5.load_ligand(paths[1], paths[2])
        if lig is None:
            return None
        rec = V5.parse_pocket(paths[0])
        if len(rec[0]) == 0:
            return None
        rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged = rec
        native = lig["xyz"]
        rng = np.random.default_rng(abs(hash(code)) % (2**32))
        poses = [native] + rigid_decoys(native, 60, rng)  # native first (RMSD 0)
        feats, rmsds = [], []
        for nx in poses:
            ligmoved = _approx_move(lig, nx)  # moves H/nbr too, for hbond geometry
            F = compute8(ligmoved, rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged, nx)
            feats.append(F)
            rmsds.append(float(np.sqrt(((nx - native) ** 2).sum(1).mean())))
        feats = np.array(feats); rmsds = np.array(rmsds)
        full = score_feats(feats, mean, scale, coef, b)
        shape_mask = np.array([1, 1, 1, 0, 0, 0, 0, 0.0])  # gauss1,gauss2,repulsion only
        shape = score_feats(feats, mean, scale, coef, b, mask=shape_mask)
        # physically-signed pose weights on standardized channels (higher = more native):
        # contact+, clash--, hydrophobic+, hbond+, elec favorable (neg feat) rewarded,
        # desolv (buried polar) penalized. nrot ~constant across rigid poses -> 0.
        Zp = (feats - mean) / scale
        phys_w = np.array([0.5, 1.0, -2.0, 0.5, 0.8, 0.0, -0.8, -0.5])
        phys = Zp @ phys_w

        def m(scores):
            best = int(np.argmax(scores))
            rho = spearmanr(scores, rmsds)[0] if len(scores) > 3 else np.nan
            return bool(rmsds[best] < 2.0), float(rho)
        f_top1, f_rho = m(full)
        s_top1, s_rho = m(shape)
        p_top1, p_rho = m(phys)
        return dict(id=code, n=len(rmsds), rmsd_min=float(rmsds.min()),
                    full_top1=f_top1, full_rho=f_rho,
                    shape_top1=s_top1, shape_rho=s_rho,
                    phys_top1=p_top1, phys_rho=p_rho)
    except Exception as e:
        return dict(id=code, err=f"{type(e).__name__}: {e}")


def _approx_move(lig, new_xyz):
    """Kabsch-fit rigid transform native->new_xyz, apply to H/nbr too (for hbond geometry)."""
    P = lig["xyz"]; Q = new_xyz
    cP = P.mean(0); cQ = Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = cQ - R @ cP
    out = dict(lig); out["xyz"] = new_xyz
    out["H"] = [((h) @ R.T + t) if h.shape[0] else h for h in lig["H"]]
    nb = lig["nbr"]; moved = np.full_like(nb, np.nan); ok = ~np.isnan(nb[:, 0])
    moved[ok] = (nb[ok]) @ R.T + t
    out["nbr"] = moved
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(Path.home() / "screen/data"))
    ap.add_argument("--manifest", default=str(Path.home() / "screen/out/v1/manifest.csv"))
    ap.add_argument("--v5-feats", default=str(Path.home() / "screen/out/v5/v5_features.npz"))
    ap.add_argument("--n", type=int, default=200, help="number of test complexes to sample")
    ap.add_argument("--procs", type=int, default=48)
    ap.add_argument("--out", default=str(Path.home() / "screen/out/v7"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    model = fit_v5_model(a.v5_feats, a.manifest)
    man = pd.read_csv(a.manifest)
    test_ids = man[man.split == "test"]["id"].tolist()
    rng = np.random.default_rng(RNG_SEED)
    rng.shuffle(test_ids)
    ids = test_ids[:(5 if a.smoke else a.n)]

    tasks = [(c, a.data_root, model) for c in ids]
    if a.smoke:
        rows = [process_one(t) for t in tasks]
    else:
        with Pool(a.procs) as p:
            rows = p.map(process_one, tasks)
    rows = [r for r in rows if r and "err" not in r]
    df = pd.DataFrame(rows)
    df.to_csv(out / "v7_pose_per_complex.csv", index=False)
    summ = dict(n=len(df),
                full_top1_lt2=float(df.full_top1.mean()),
                shape_top1_lt2=float(df.shape_top1.mean()),
                phys_top1_lt2=float(df.phys_top1.mean()),
                full_rho_median=float(df.full_rho.median()),
                shape_rho_median=float(df.shape_rho.median()),
                phys_rho_median=float(df.phys_rho.median()))
    (out / "v7_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
