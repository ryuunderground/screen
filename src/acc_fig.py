"""Two weight sets, two accuracies:
 Left  - w_affinity: predicted vs experimental pKd (held-out test) -> Pearson r, RMSE
 Right - w_pose: native-pose recognition (top-1<2A), learned pose weights vs affinity weights
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
import sys; sys.path.insert(0,str(Path.home()/"screen/src"))
from v7_pose import fit_v5_model
ROOT=Path.home()/"screen"; NAVY="#1F3864"; GRAY="#B8C0CC"
names,mean,scale,coef,b=fit_v5_model(ROOT/"out/v5/v5_features.npz",ROOT/"out/v1/manifest.csv")
d=np.load(ROOT/"out/v5/v5_features.npz",allow_pickle=True)
feat=pd.DataFrame(d["X"],columns=names); feat["id"]=[str(x) for x in d["ids"]]
man=pd.read_csv(ROOT/"out/v1/manifest.csv"); df=man.merge(feat,on="id").dropna(subset=names)
tr=df[df.split=="train"]; te=df[df.split=="test"]
gbr=GradientBoostingRegressor(random_state=0).fit(tr[names].values,tr["pKd"].values)
yte=te["pKd"].values; pte=gbr.predict(te[names].values)
rA=pearsonr(yte,pte)[0]; rmseA=float(np.sqrt(np.mean((yte-pte)**2)))
ev=json.load(open(ROOT/"out/pose_eval.json"))
va=ev["AFFINITY-fit w"]["top1"]; vp=ev["LEARNED (pose)"]["top1"]

fig,ax=plt.subplots(1,2,figsize=(10.4,4.5))
# left: affinity accuracy
ax[0].scatter(yte,pte,s=7,alpha=0.22,color=NAVY,edgecolors="none")
lo,hi=min(yte.min(),pte.min()),max(yte.max(),pte.max())
ax[0].plot([lo,hi],[lo,hi],"--",color="#C0392B",lw=1.2)
ax[0].set_xlabel("experimental pKd"); ax[0].set_ylabel("predicted pKd")
ax[0].set_title(f"Affinity weights  →  binding-strength accuracy\nPearson r = {rA:.2f}   RMSE = {rmseA:.2f}",fontsize=11,color=NAVY)
# right: pose accuracy
bars=ax[1].bar(["affinity-fit\nweights","learned\npose weights"],[va,vp],color=[GRAY,NAVY],width=0.58)
for bb,v in zip(bars,[va,vp]):
    ax[1].text(bb.get_x()+bb.get_width()/2,v+2.2,f"{v:.1f}%",ha="center",fontsize=13,fontweight="bold",color=NAVY if v>50 else "#777")
ax[1].set_ylim(0,108); ax[1].set_ylabel("native pose recognized (top-1 < 2 Å)")
ax[1].set_title("Pose weights  →  native-pose accuracy",fontsize=11,color=NAVY)
ax[1].spines["top"].set_visible(False); ax[1].spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(ROOT/"out/accuracy_two_weights.png",dpi=150,bbox_inches="tight")
print(f"wrote accuracy_two_weights.png  affinity r={rA:.3f} rmse={rmseA:.3f}  pose {va:.1f}/{vp:.1f}")
