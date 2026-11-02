"""Field-change figure: same ligand, rotate one torsion -> scalar field morphs.
Two cubes (Structure channel) at theta=0 and theta=rotated, with an arrow between."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path.home()/"screen/src"))
import v5_features as V5
ROOT=Path.home()/"screen"
ROT=Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
def moving_set(mol,a,b):
    seen={a,b}; st=[b]; fr={b}
    while st:
        x=st.pop()
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            j=nb.GetIdx()
            if j not in seen: seen.add(j); fr.add(j); st.append(j)
    return fr
# same ligand as fragmentation/curve (1xxi), same pivot selection
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
        if 5<=len(mv)<=13: pick=(code,p,aa,bb,mv); break
    if pick: break
code,paths,aa,bb,moving=pick
lig,_=V5.load_ligand(paths[1],paths[2]); X0=lig["xyz"].copy()
pa,pb=X0[aa],X0[bb]; axis=(pb-pa)/np.linalg.norm(pb-pa); mv=np.array(sorted(moving))
def rot(dd):
    th=np.deg2rad(dd); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K); Xr=X0.copy(); Xr[mv]=(X0[mv]-pb)@R.T+pb; return Xr

c=X0.mean(0); ext=np.abs(X0-c).max()+2.5; n=30
a=np.linspace(-ext,ext,n); gx,gy,gz=np.meshgrid(a,a,a,indexing="ij"); G=np.stack([gx,gy,gz],-1)+c
def field(Xr,sigma=1.2):
    out=np.zeros((n,n,n))
    for pt in Xr: out+=np.exp(-((G-pt)**2).sum(-1)/(2*sigma**2))
    return out/out.max()

fig=plt.figure(figsize=(6.0,3.0))
for k,(dd,lab) in enumerate([(0,"θ = 0"),(120,"θ rotated")]):
    Xr=rot(dd); f=field(Xr); ax=fig.add_subplot(1,2,k+1,projection="3d")
    sel=f>0.14; xs,ys,zs=gx[sel]+c[0],gy[sel]+c[1],gz[sel]+c[2]; v=f[sel]; o=np.argsort(v)
    ax.scatter(xs[o],ys[o],zs[o],c=v[o],cmap="Reds",s=14,alpha=0.3,vmin=0.14,vmax=1,edgecolors="none",depthshade=False)
    ax.set_xlim(c[0]-ext,c[0]+ext);ax.set_ylim(c[1]-ext,c[1]+ext);ax.set_zlim(c[2]-ext,c[2]+ext)
    ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(18,-60); ax.set_title(lab,fontsize=11)
fig.suptitle("Rotating a torsion changes the scalar field",fontsize=11,y=1.02)
# arrow between
fig.text(0.5,0.5,"→",fontsize=22,ha="center",va="center",color="#1F3864")
fig.tight_layout()
fig.savefig(ROOT/"out/field_change.png",dpi=150,bbox_inches="tight")
print("wrote field_change.png code",code)
