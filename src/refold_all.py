"""Recompute all versions with val merged into train (test frozen). Reuses cached npz."""
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

man = pd.read_csv("out/v1/manifest.csv")

def met(y, p):
    return pearsonr(y, p)[0], spearmanr(y, p)[0], float(np.sqrt(np.mean((y-p)**2)))

def run(ver):
    d = np.load(f"out/{ver}/{ver}_features.npz", allow_pickle=True)
    NAMES = [str(x) for x in d["feature_names"]]
    feat = pd.DataFrame(d["X"], columns=NAMES); feat["id"] = [str(x) for x in d["ids"]]
    df = man.merge(feat, on="id", how="inner").dropna(subset=NAMES)
    tr = df[df.split.isin(["train","val"])].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    Xtr, ytr, gtr = tr[NAMES].values, tr["pKd"].values, tr["seq"].values
    Xte, yte = te[NAMES].values, te["pKd"].values
    gkf = GroupKFold(n_splits=5)
    orf = np.full(len(tr), np.nan); ogf = np.full(len(tr), np.nan)
    for itr, ite in gkf.split(Xtr, ytr, gtr):
        sc = StandardScaler().fit(Xtr[itr])
        orf[ite] = Ridge(alpha=1.0).fit(sc.transform(Xtr[itr]), ytr[itr]).predict(sc.transform(Xtr[ite]))
        ogf[ite] = GradientBoostingRegressor(random_state=0).fit(Xtr[itr], ytr[itr]).predict(Xtr[ite])
    sc = StandardScaler().fit(Xtr)
    rm = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    gm = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    pr, pg = rm.predict(sc.transform(Xte)), gm.predict(Xte)
    cr, cg, trr, tgr = met(ytr,orf), met(ytr,ogf), met(yte,pr), met(yte,pg)
    print(f"{ver}  n_tr={len(tr)} n_te={len(te)} | "
          f"CV Ridge {cr[0]:.3f}/GBR {cg[0]:.3f} | TEST Ridge {trr[0]:.3f} GBR {tgr[0]:.3f}")
    return ver, trr[0], tgr[0]

print("version  sizes | CV | TEST (val merged into train, test frozen)")
res = [run(v) for v in ["v1","v3","v4","v5","v6"]]
print("\nRIDGE =", {v: round(r,3) for v,r,g in res})
print("GBR   =", {v: round(g,3) for v,r,g in res})
