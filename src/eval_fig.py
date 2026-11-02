"""Slide-17 evaluation figure (2 panels):
 A) affinity: predicted vs experimental pKd (held-out test) -> Pearson r, RMSE
 B) pose: score vs RMSD funnel (learned w_pose) -> Spearman
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
sys.path.insert(0, str(Path.home()/"screen/src"))
import v5_features as V5
from v7_pose import fit_v5_model, compute8, rigid_decoys
ROOT=Path.home()/"screen"; NAVY="#1F3864"; GOLD="#C8A94B"
model=fit_v5_model(ROOT/"out/v5/v5_features.npz",ROOT/"out/v1/manifest.csv")
names,mean,scale,coef,b=model
W=np.load(ROOT/"out/w_pose.npy")

# --- Panel A: affinity scatter (GBR, held-out test) ---
d=np.load(ROOT/"out/v5/v5_features.npz",allow_pickle=True)
feat=pd.DataFrame(d["X"],columns=names); feat["id"]=[str(x) for x in d["ids"]]
man=pd.read_csv(ROOT/"out/v1/manifest.csv"); df=man.merge(feat,on="id").dropna(subset=names)
tr=df[df.split=="train"]; te=df[df.split=="test"]
gbr=GradientBoostingRegressor(random_state=0).fit(tr[names].values,tr["pKd"].values)
yte=te["pKd"].values; pte=gbr.predict(te[names].values)
rA=pearsonr(yte,pte)[0]; rmseA=float(np.sqrt(np.mean((yte-pte)**2)))

# --- Panel B: pose funnel (learned weights, ~150 test complexes, z-scored) ---
rng=np.random.default_rng(2); ids=man[man.split=="test"]["id"].sample(150,random_state=2).tolist()
allr=[]; allz=[]; rhos=[]
for code in ids:
    p=V5.find_complex(str(ROOT/"data"),code)
    if p is None or not p[0].exists(): continue
    try:
        lig,_=V5.load_ligand(p[1],p[2]); rec=V5.parse_pocket(p[0])
    except Exception: continue
    if lig is None or len(rec[0])==0: continue
    rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged=rec; native=lig["xyz"]
    r2=np.random.default_rng(int(__import__("hashlib").md5(code.encode()).hexdigest(),16)%2**32)
    poses=[native]+rigid_decoys(native,40,r2)
    sc=[]; rm=[]
    for P in poses:
        l2=dict(lig); l2["xyz"]=P; l2["nbr"]=V5._neighbor_centroid(P,cKDTree(P))
        F=compute8(l2,rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged,P)
        sc.append(((F-mean)/scale)@W); rm.append(float(np.sqrt(((P-native)**2).sum(1).mean())))
    sc=np.array(sc); rm=np.array(rm)
    z=(sc-sc.mean())/(sc.std()+1e-9)
    allr+=rm.tolist(); allz+=z.tolist()
    rh=spearmanr(sc,rm)[0]
    if np.isfinite(rh): rhos.append(rh)
medrho=float(np.median(rhos))

fig,ax=plt.subplots(1,2,figsize=(10.5,4.4))
# A
ax[0].scatter(yte,pte,s=7,alpha=0.25,color=NAVY,edgecolors="none")
lo,hi=min(yte.min(),pte.min()),max(yte.max(),pte.max())
ax[0].plot([lo,hi],[lo,hi],"--",color="#C0392B",lw=1.2)
ax[0].set_xlabel("experimental pKd"); ax[0].set_ylabel("predicted pKd")
ax[0].set_title(f"Affinity (held-out test)\nPearson r = {rA:.2f}   RMSE = {rmseA:.2f}",fontsize=11,color=NAVY)
# B
ax[1].scatter(allr,allz,s=5,alpha=0.15,color=GOLD,edgecolors="none")
ax[1].set_xlabel("RMSD to native (Å)"); ax[1].set_ylabel("score (z per complex)")
ax[1].set_title(f"Pose funnel (learned weights)\nmedian Spearman(score, RMSD) = {medrho:+.2f}",fontsize=11,color=NAVY)
ax[1].invert_yaxis()
fig.tight_layout(); fig.savefig(ROOT/"out/eval_metrics.png",dpi=150,bbox_inches="tight")
print(f"wrote eval_metrics.png  affinity r={rA:.3f} rmse={rmseA:.3f}  pose medrho={medrho:.3f}")
