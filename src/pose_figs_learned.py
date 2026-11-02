"""Regenerate pose figures with LEARNED w_pose:
 (1) torsion_curve.png  (2) pose_recognition.png (affinity vs learned)  (3) pose_example.png
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path.home() / "screen/src"))
import v5_features as V5
from v7_pose import fit_v5_model, compute8
ROOT = Path.home() / "screen"
NAVY="#1F3864"; GOLD="#C8A94B"; RED="#C0392B"
W = np.load(ROOT / "out/w_pose.npy")
model = fit_v5_model(ROOT/"out/v5/v5_features.npz", ROOT/"out/v1/manifest.csv")
names, mean, scale, coef, b_ = model

# ---------- (2) pose_recognition: affinity vs learned ----------
ev = json.load(open(ROOT/"out/pose_eval.json"))
labels=["affinity-fit\nweights","learned\npose weights"]
vals=[ev["AFFINITY-fit w"]["top1"], ev["LEARNED (pose)"]["top1"]]
fig,ax=plt.subplots(figsize=(5.4,4.2))
bars=ax.bar(labels,vals,color=["#B8C0CC",NAVY],width=0.6)
for b,v in zip(bars,vals):
    ax.text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=12,
            fontweight="bold", color=NAVY if v>50 else "#555")
ax.set_ylabel("native pose recognized  (top-1 < 2 Å)")
ax.set_ylim(0,105)
ax.set_title("Weights learned for the pose task recognize the native pose",
             fontsize=11.5, color=NAVY)
ax.axhline(0,color="#888",lw=0.8)
ax.text(0,8,"affinity weights\nfail at pose",ha="center",fontsize=8.5,color="#777")
fig.tight_layout(); fig.savefig(ROOT/"out/pose_recognition.png",dpi=150); plt.close(fig)
print("wrote pose_recognition.png", vals)

# ---------- (1) torsion_curve with learned W ----------
ROT=Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
def moving_set(mol,a,b):
    seen={a,b}; st=[b]; fr={b}
    while st:
        x=st.pop()
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            j=nb.GetIdx()
            if j not in seen: seen.add(j); fr.add(j); st.append(j)
    return fr
man=pd.read_csv(ROOT/"out/v1/manifest.csv"); ids=man[man.split=="test"]["id"].tolist()
rng=np.random.default_rng(11); rng.shuffle(ids); pick=None
for code in ids:
    p=V5.find_complex(str(ROOT/"data"),code)
    if p is None or not p[0].exists(): continue
    mol=Chem.MolFromMolFile(str(p[1]),removeHs=True) or Chem.MolFromMol2File(str(p[2]),removeHs=True)
    if mol is None or mol.GetNumConformers()==0: continue
    rb=[tuple(x) for x in mol.GetSubstructMatches(ROT)]
    if not (18<=mol.GetNumHeavyAtoms()<=34 and 3<=len(rb)<=7): continue
    for a,b in rb:
        fa=moving_set(mol,a,b); fb=moving_set(mol,b,a)
        mv,(aa,bb)=(fa,(a,b)) if len(fa)<=len(fb) else (fb,(b,a))
        if 5<=len(mv)<=13: pick=(code,p,mol,(aa,bb),mv); break
    if pick: break
code,paths,mol,(aa,bb),moving=pick
lig,_=V5.load_ligand(paths[1],paths[2]); rec=V5.parse_pocket(paths[0])
rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged=rec
X0=lig["xyz"].copy(); pa,pb=X0[aa],X0[bb]; axis=(pb-pa); axis/=np.linalg.norm(axis); mv3=np.array(sorted(moving))
def rot3(dd):
    th=np.deg2rad(dd); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K); Xr=X0.copy(); Xr[mv3]=(X0[mv3]-pb)@R.T+pb; return Xr
ang=np.arange(-180,181,6); E=[]
for dd in ang:
    Xr=rot3(dd); l2=dict(lig); l2["xyz"]=Xr; l2["nbr"]=V5._neighbor_centroid(Xr,cKDTree(Xr))
    F=compute8(l2,rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged,Xr); E.append(((F-mean)/scale)@W)
E=np.array(E); peak=ang[np.argmax(E)]; span=np.ptp(E)
fig,ax=plt.subplots(figsize=(6.6,4.0))
ax.plot(ang,E,"-",color=NAVY,lw=2.4); ax.axvline(0,color=GOLD,lw=2,ls="--")
ax.text(4,E.min()+0.04*span,"crystal pose\nθ=0",color="#9a7b1f",fontsize=8.5,va="bottom")
ax.scatter([peak],[E.max()],s=95,color=RED,zorder=5)
ax.annotate("peak = predicted pose + score",xy=(peak,E.max()),
            xytext=(peak-6,E.max()-0.20*span),fontsize=9,color=RED,ha="right",
            arrowprops=dict(arrowstyle="->",color=RED))
ax.set_xlabel("torsion angle θ (° from crystal) — one rotatable bond")
ax.set_ylabel("field-overlap score  E(θ)  [learned pose weights]")
ax.set_title("Binding is a curve over pose: E(θ) for one torsion",fontsize=11.5,color=NAVY)
ax.set_xticks(range(-180,181,60)); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(ROOT/"out/torsion_curve.png",dpi=150); plt.close(fig)
print("wrote torsion_curve.png (learned) peak",peak,"code",code)

# ---------- (3) pose_example with learned W ----------
import make_example_figs as M
M.PHYS_W = W
picks = M.examples()
print("POSE_EXAMPLE_PICKS", picks)
