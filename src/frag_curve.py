"""Slide-15 figures: fragmentation (rigid fragments colored) + real E(theta) torsion curve."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path.home() / "screen/src"))
import v5_features as V5
from v7_pose import fit_v5_model, compute8

ROOT = Path.home() / "screen"
NAVY = "#1F3864"; GOLD = "#C8A94B"
PHYS_W = np.array([0.5, 1.0, -2.0, 0.5, 0.8, 0.0, -0.8, -0.5])
ROT = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
PAL = [(0.86,0.37,0.34),(0.33,0.49,0.74),(0.55,0.73,0.37),(0.80,0.63,0.26),
       (0.55,0.42,0.69),(0.35,0.70,0.67),(0.90,0.55,0.30)]

def moving_set(mol, a, b):
    seen={a,b}; st=[b]; frag={b}
    while st:
        x=st.pop()
        for nb in mol.GetAtomWithIdx(x).GetNeighbors():
            j=nb.GetIdx()
            if j not in seen: seen.add(j); frag.add(j); st.append(j)
    return frag

def pick():
    man = pd.read_csv(ROOT/"out/v1/manifest.csv")
    ids = man[man.split=="test"]["id"].tolist()
    rng = np.random.default_rng(11); rng.shuffle(ids)
    for code in ids:
        p = V5.find_complex(str(ROOT/"data"), code)
        if p is None or not p[0].exists(): continue
        mol = Chem.MolFromMolFile(str(p[1]), removeHs=True) or Chem.MolFromMol2File(str(p[2]), removeHs=True)
        if mol is None or mol.GetNumConformers()==0: continue
        rb = [tuple(x) for x in mol.GetSubstructMatches(ROT)]
        if not (18 <= mol.GetNumHeavyAtoms() <= 34 and 3 <= len(rb) <= 7): continue
        best=None
        for a,b in rb:
            fa=moving_set(mol,a,b); fb=moving_set(mol,b,a)
            mv,(aa,bb)=(fa,(a,b)) if len(fa)<=len(fb) else (fb,(b,a))
            if 5 <= len(mv) <= 13:
                best=(code,p,mol,rb,(aa,bb),mv); break
        if best: return best
    raise SystemExit("none")

CODE,paths,mol,rotbonds,(aa,bb),moving = pick()
print("picked",CODE,"rotbonds",len(rotbonds),"pivot",aa,bb,"moving",len(moving))

# ---- (1) fragmentation: color each rigid fragment ----
rw = Chem.RWMol(mol)
for a,b in rotbonds: rw.RemoveBond(a,b)
frags = Chem.GetMolFrags(rw, asMols=False, sanitizeFrags=False)
atom_col={}; 
for fi,atoms in enumerate(frags):
    for at in atoms: atom_col[at]=PAL[fi%len(PAL)]
bond_col={}; hb=[]
for bd in mol.GetBonds():
    i,j=bd.GetBeginAtomIdx(),bd.GetEndAtomIdx()
    if (i,j) in rotbonds or (j,i) in rotbonds: continue
    if atom_col.get(i)==atom_col.get(j):
        bond_col[bd.GetIdx()]=atom_col[i]; hb.append(bd.GetIdx())
rot_bidx=[mol.GetBondBetweenAtoms(a,b).GetIdx() for a,b in rotbonds]
for k in rot_bidx: bond_col[k]=(0.1,0.1,0.1); 
AllChem.Compute2DCoords(mol)
d=rdMolDraw2D.MolDraw2DCairo(620,470)
o=d.drawOptions(); o.highlightBondWidthMultiplier=12
rdMolDraw2D.PrepareAndDrawMolecule(d,mol,
    highlightAtoms=list(atom_col), highlightBonds=hb+rot_bidx,
    highlightAtomColors=atom_col, highlightBondColors=bond_col)
d.FinishDrawing(); open(ROOT/"out/fragmentation.png","wb").write(d.GetDrawingText())
print("wrote fragmentation.png | rigid fragments:",len(frags))

# ---- (2) torsion E(theta) ----
lig,_=V5.load_ligand(paths[1],paths[2]); rec=V5.parse_pocket(paths[0])
rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged=rec
X0=lig["xyz"].copy(); pa,pb=X0[aa],X0[bb]; axis=(pb-pa); axis/=np.linalg.norm(axis)
mv=np.array(sorted(moving))
def rot(dd):
    th=np.deg2rad(dd); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)+np.sin(th)*K+(1-np.cos(th))*(K@K); Xr=X0.copy(); Xr[mv]=(X0[mv]-pb)@R.T+pb; return Xr
names,mean,scale,coef,b_=fit_v5_model(ROOT/"out/v5/v5_features.npz",ROOT/"out/v1/manifest.csv")
ang=np.arange(-180,181,6); sc=[]
for dd in ang:
    Xr=rot(dd); l2=dict(lig); l2["xyz"]=Xr; l2["nbr"]=V5._neighbor_centroid(Xr,cKDTree(Xr))
    F=compute8(l2,rxyz,rel,rdon,racc,rhyd,rnbr,rq,rcharged,Xr); sc.append(((F-mean)/scale)@PHYS_W)
sc=np.array(sc); peak=ang[np.argmax(sc)]; span=np.ptp(sc)
fig,ax=plt.subplots(figsize=(6.6,4.0))
ax.plot(ang,sc,"-",color=NAVY,lw=2.4)
ax.axvline(0,color=GOLD,lw=2,ls="--")
ax.text(2,sc.min()+0.04*span,"crystal pose\nθ=0",color="#9a7b1f",fontsize=8.5,va="bottom")
ax.scatter([peak],[sc.max()],s=95,color="#C0392B",zorder=5)
ax.annotate("peak = predicted pose + score",xy=(peak,sc.max()),
            xytext=(peak-6,sc.max()-0.20*span),fontsize=9,color="#C0392B",ha="right",
            arrowprops=dict(arrowstyle="->",color="#C0392B"))
ax.set_xlabel("torsion angle θ (° from crystal) — one rotatable bond")
ax.set_ylabel("field-overlap score  E(θ)")
ax.set_title("Binding is a curve over pose: E(θ) for one torsion",fontsize=11.5,color=NAVY)
ax.set_xticks(range(-180,181,60)); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(ROOT/"out/torsion_curve.png",dpi=150); plt.close(fig)
print("wrote torsion_curve.png peak",peak)
