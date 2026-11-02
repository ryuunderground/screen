# Field-overlap docking: v1 scoring fit + v2 docking (PRELIMINARY)

Pipeline for multi-channel molecular-field docking. **v1** learns channel weights that
predict binding affinity from field-overlap features at the crystal pose (fully worked
out, validated on a leak-proof held-out test). **v2** extends the FFT docking pilot to a
full search + internal torsion and is **PRELIMINARY / UNVALIDATED**.

All numbers below were computed on the remote server (`~/screen`), not invented. Code in
`src/`, outputs in `out/{v1,v2}/`, figures in `figures_v1/` and `figures_v2/`.

---

## 1. Setup & data

- Compute: login(96) + node01(64) + node02(64) ~ 224 cores, conda env `fieldscreen`
  (python 3.11, rdkit 2026.09.1). CPU only.
- Structures: PDBbind v2020R1 P-L, 19037 complexes. Features use `*_pocket.pdb`
  (binding-site protein) + `*_ligand.sdf`.
- Labels: `INDEX_general_PL.2020R1.lst` parsed `Kd=/Ki=/IC50=` token -> molar -> pKd.
  **18767** parsed; dropped **263** approximate (`~ < >`) and **7** unparseable.
- Splits & clusters: `LP_PDBBind.csv` (Leak-Proof PDBbind) `new_split` + protein `seq`.
- Join (label AND split): **18137** complexes -- train 11044 / val 2340 / test 4753.
- Feature extraction succeeded on **18137/18137** (1 ligand needed a no-sanitize RDKit
  fallback, 0 hard failures).

## 2. v1 -- scoring-function fit at the crystal pose (VALIDATED)

### Method

For each complex: grid (0.8 A spacing) centered on the ligand; render receptor and
ligand as Gaussian fields; compute 5 per-channel **overlap scalars**
`sum(rec_ch * lig_ch)*dV` at the crystal pose (no docking search):

| feature | receptor side | ligand side | sign intent |
|---|---|---|---|
| `contact` | surface shell (skin) | occupancy | burial, + |
| `clash` | vdW core | occupancy | penalty |
| `polar` | N/O occupancy | N/O occupancy | + |
| `hydrophob` | C/S/halogen occ | C/S/halogen occ | + |
| `elec` | signed charge field | Gasteiger charge field | complementarity |

Receptor charge proxy by PDB atom name: Asp/Glu carboxylate O -0.5 each; Lys NZ +1.0;
Arg NE/NH1/NH2 +0.33 each; His ND1/NE2 +0.5 each; else 0. Charge rendered wider
(sigma=1.0) than density (sigma=0.8) per `notes/channels.md` section 3. Ligand Gasteiger
via RDKit.

Models: **Ridge** (primary; standardized coefficients = channel importances) and
**GradientBoosting** (nonlinear comparison). CV = 5-fold **GroupKFold grouped by protein
`seq`** on the LP train split (no same-protein leakage across folds). Then refit on full
train, **freeze**, evaluate LP val + test.

### Results (real, out-of-fold and frozen-test)

| metric | Ridge CV (OOF) | GBR CV (OOF) | Ridge test | GBR test |
|---|---|---|---|---|
| Pearson r | **0.342** | **0.509** | **0.382** | **0.461** |
| Spearman rho | 0.390 | 0.486 | 0.407 | 0.436 |
| RMSE (pKd) | 1.764 | 1.616 | 1.657 | 1.586 |

Per-fold Ridge r ranged 0.310-0.375 (tight); GBR 0.469-0.554. Val: Ridge r=0.315,
GBR r=0.455. Figures: `figures_v1/fig_v1_scatter.png` (CV + frozen test),
`fig_v1_channel_weights.png`, `fig_v1_folds.png`.

### Channel weights & an honest caveat

Frozen full-train Ridge coefficients (standardized): contact +0.59, clash +0.62,
polar -0.57, hydrophob -0.04, elec +0.21. **These signs are NOT physically
interpretable** -- the five overlap features are strongly collinear (all grow with
buried ligand volume), so Ridge splits a single "size/contact" signal across them with
arbitrary signs. This is exactly the sanity check `notes/channels.md` section 1 warns
about, and it fails: the learned linear signs do not match the designed interaction
table. The nonlinear GBR (r 0.51 vs 0.34 CV) extracts materially more signal, consistent
with the real information living in feature interactions rather than one linear size axis.

**Interpretation:** r~0.38 on a leak-proof held-out test from 5 hand-crafted overlap
features is a modest but genuine baseline (the scatter shows the classic
regression-to-the-mean compression around pKd 5-7). It is below production scoring
functions; the point of v1 is a validated, leakage-controlled reference, which it is.

## 3. v2 -- FFT docking + internal torsion (PRELIMINARY / UNVALIDATED)

### Method

- **Rigid search:** receptor channels rendered on a grid around the known pocket
  (crystal-ligand centroid -- cognate redocking, **known box, not blind**). For each of
  N sampled rotations, render ligand channels, FFT-correlate each vs its receptor
  channel, weighted-sum landscape -> translation peak; best over rotations = pose.
  The ligand centroid is seated at the box center so the FFT shift range covers the
  pocket (a bug caught in smoke-testing: the PoseBusters start conformer lives in a
  different coordinate frame 20-44 A away, unreachable otherwise).
- **Torsion:** after rigid placement, greedily refine each "substantial" rotatable-bond
  dihedral (moving fragment reaching >2.5 A from the bond axis, i.e. a >5 A swept
  diameter) by scanning 12 angles and maximizing the point-sampled field-overlap score,
  BFS outward, 2 passes. One dihedral free per step.
- **Channel weights:** v1's fitted signs are multicollinear and unusable for search, so
  v2 uses **physically-signed** weights with v1-informed magnitudes (clash a hard
  penalty, contact/polar/hyd favorable, elec rewards charge complementarity).
- Input = PoseBusters `*_ligand_start_conf.sdf` (wrong internal torsions); ground truth
  = `*_ligand.sdf`; RMSD is symmetry-corrected, in-place (no superposition).
- Ran 100 PoseBusters Benchmark complexes, 600 rotations, sharded across all 3 nodes.

### Results (n=99 docked; 1 skipped on ligand parse/atom mismatch)

| threshold | rigid FFT | + torsion refine |
|---|---|---|
| RMSD < 2 A | **1.0%** (1/99) | **1.0%** |
| RMSD < 3 A | 6.1% | 5.1% |
| RMSD < 5 A | **24.2%** | 21.2% |
| median RMSD | 7.81 A | 8.12 A |

Figures: `figures_v2/fig_v2_success.png` (success bars), `fig_v2_scatter.png`
(per-complex rigid vs torsion).

### Rotation-density sensitivity (controlled, same 23 complexes)

| rotations | rigid <2 A | rigid <5 A | median |
|---|---|---|---|
| 600 | 4.3% | 13.0% | 7.68 A |
| 4000 | 4.3% | 17.4% | 7.18 A |

Denser SO(3) sampling improves coarse placement (<5 A, and median) but leaves **<2 A
flat** -- the sub-2 A bottleneck is not rotation count. It is the torsion/scoring stage.

**Honest reading:** the method gets ~a quarter of ligands roughly into the pocket (<5 A)
but almost none to crystallographic accuracy (<2 A), and the greedy torsion refinement
does **not** improve aggregate accuracy (median slightly worse). The torsion step
maximizes field overlap, which is only loosely correlated with RMSD, so it can raise
score while worsening geometry. Pose accuracy, not scoring, is the current weak point.

## 4. What to check / next steps

- **Fix the <2 A bottleneck at the torsion/scoring stage, not rotation count** (the
  sensitivity table shows rotations alone do not help <2 A): replace greedy field-overlap
  dihedral scan with joint optimization plus an intramolecular strain term, and add local
  6-DOF refinement (Powell/BFGS) around each FFT peak.
- **v1 multicollinearity:** decorrelate channels (residualized features, or per-channel
  radial profiles instead of one scalar) so learned signs become interpretable and match
  the design table in `notes/channels.md`. Add Vina on the same split as a reference ceiling.
- **Charges & directionality:** Gasteiger + atom-name receptor charges are coarse; move
  to AM1-BCC / PDB2PQR, and add the l=1 H-bond donor/acceptor arrow channels (v2's stated
  design purpose) -- not yet implemented.
- **Independent benchmarks (not run, time):** BDB2020+ affinity test
  (`lp_pdbbind/dataset/BDB2020+.tgz`) and LIT-PCBA enrichment (EF@1%/BEDROC); both are
  staged to add.

## 5. What ran / failed / skipped

- Ran fully: v1 feature extraction (18137/18137), v1 CV + frozen test, all v1 figures;
  v2 rigid+torsion redocking (99/100), v2 figures, and a controlled rotation-density
  sensitivity check (600 vs 4000 rotations).
- One fixed bug: the start conformer's coordinate frame was outside the FFT shift range
  (caught during smoke-testing, not in production numbers).
- Skipped (time, not blockers): BDB2020+ external test, LIT-PCBA enrichment, explicit
  H-bond directional channels.
- No fabricated numbers. v2 is preliminary and its torsion method is not validated.
