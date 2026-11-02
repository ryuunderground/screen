"""4b: learn POSE weights (native=max) by pairwise ranking on native+decoys.
Supervision = RMSD to native (NOT affinity). Score E = Z@w, Z = affinity-model
standardized 8 channels -> learned w directly comparable to hand-set PHYS_W.
Steps: gen native+rigid decoys per complex -> (Z[P,8], rmsd[P]); pairwise logistic
(lower-RMSD must outscore higher-RMSD); eval held-out top1<2A + Spearman; vs PHYS_W.
"""
import argparse, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path.home() / "screen/src"))
import v5_features as V5
from v7_pose import fit_v5_model, compute8, rigid_decoys
from scipy.spatial import cKDTree

ROOT = Path.home() / "screen"
PHYS_W = np.array([0.5, 1.0, -2.0, 0.5, 0.8, 0.0, -0.8, -0.5])
NDEC = 60
_MODEL = None

def feats_for_complex(code):
    paths = V5.find_complex(str(ROOT / "data"), code)
    if paths is None or not paths[0].exists(): return None
    try:
        lig, _ = V5.load_ligand(paths[1], paths[2])
        if lig is None: return None
        rec = V5.parse_pocket(paths[0])
        if len(rec[0]) == 0: return None
    except Exception: return None
    rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged = rec
    native = lig["xyz"]
    rng = np.random.default_rng(int(__import__("hashlib").md5(code.encode()).hexdigest(), 16) % 2**32)
    poses = [native] + rigid_decoys(native, NDEC, rng)
    F, rmsd = [], []
    for P in poses:
        l2 = dict(lig); l2["xyz"] = P; l2["nbr"] = V5._neighbor_centroid(P, cKDTree(P))
        F.append(compute8(l2, rxyz, rel, rdon, racc, rhyd, rnbr, rq, rcharged, P))
        rmsd.append(float(np.sqrt(((P - native) ** 2).sum(1).mean())))
    return code, np.array(F), np.array(rmsd)

def gather(codes, procs):
    with Pool(procs) as pool:
        out = [r for r in pool.map(feats_for_complex, codes) if r is not None]
    return out

def standardize(F, model):
    _, mean, scale, _, _ = model
    return (F - mean) / scale

def learn_pose_w(data, model, max_pairs=150):
    Xd, yd = [], []
    for code, F, rmsd in data:
        Z = standardize(F, model)
        n = len(Z); rng = np.random.default_rng(0)
        pairs = []
        for _ in range(max_pairs):
            i, j = rng.integers(0, n, 2)
            if abs(rmsd[i] - rmsd[j]) < 0.5: continue
            lo, hi = (i, j) if rmsd[i] < rmsd[j] else (j, i)  # lo = better
            pairs.append(Z[lo] - Z[hi])
        for d in pairs:
            Xd.append(d); yd.append(1); Xd.append(-d); yd.append(0)
    Xd = np.array(Xd); yd = np.array(yd)
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(Xd, yd)
    return clf.coef_[0], len(yd)

def evaluate(data, model, w):
    top1, rhos = [], []
    for code, F, rmsd in data:
        Z = standardize(F, model); E = Z @ w
        best = np.argmax(E)
        top1.append(rmsd[best] < 2.0)
        rho = spearmanr(E, rmsd)[0]
        if np.isfinite(rho): rhos.append(rho)
    return 100 * np.mean(top1), float(np.median(rhos))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=1200)
    ap.add_argument("--procs", type=int, default=100)
    a = ap.parse_args()
    model = fit_v5_model(ROOT / "out/v5/v5_features.npz", ROOT / "out/v1/manifest.csv")
    man = pd.read_csv(ROOT / "out/v1/manifest.csv")
    ntr, nte = (6, 6) if a.smoke else (a.n_train, a.n_test)
    tr_ids = man[man.split == "train"]["id"].sample(ntr, random_state=1).tolist()
    te_ids = man[man.split == "test"]["id"].sample(nte, random_state=2).tolist()
    procs = 4 if a.smoke else a.procs
    print(f"gather train={ntr} test={nte} procs={procs} ...", flush=True)
    tr = gather(tr_ids, procs); te = gather(te_ids, procs)
    print(f"usable train={len(tr)} test={len(te)}", flush=True)
    w_pose, npairs = learn_pose_w(tr, model)
    np.save(ROOT / "out/w_pose.npy", w_pose)
    names = model[0]
    print("\nchannel           phys_w   learned_w")
    for nm, pw, lw in zip(names, PHYS_W, w_pose):
        print(f"  {nm:14s} {pw:7.2f}  {lw:8.3f}")
    print(f"\n(pairs used: {npairs})")
    for tag, w in [("PHYSICAL (hand)", PHYS_W), ("LEARNED (pose)", w_pose)]:
        t1, rho = evaluate(te, model, w)
        print(f"{tag:18s}  top1<2A = {t1:5.1f}%   median Spearman(E,RMSD) = {rho:+.3f}")

if __name__ == "__main__":
    main()
