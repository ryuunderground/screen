# v3 — Channel strengthening (pairwise physics)

Follow-up to report_v1_v2.md. v1/v2 diagnosis: the 5 grid-overlap channels all
scaled with buried volume → effectively one feature (collinear), honest held-out
Ridge r≈0.38. v3 replaces fuzzy grid overlaps with **distance-gated pair terms**
(Vina functional forms) at the crystal pose. Same 18,137 complexes, same LP-PDBBind
splits (train 11044 / val 2340 / test 4753), same cluster-aware GroupKFold(seq),
same labels (experimental pKd, NOT Vina scores). Only channel weights are learned.

## Channels (per complex, Σ over rec–lig heavy-atom pairs, surface dist d = r_ij − R_i − R_j)
- gauss1 = Σ exp(−(d/0.5)²)          short-range contact
- gauss2 = Σ exp(−((d−3)/2)²)        longer-range contact
- repulsion = Σ d²·[d<0]             steric clash
- hydrophobic = Σ ramp(d;0.5→1.5) over hydrophobe–hydrophobe pairs
- hbond = Σ ramp(d;−0.7→0) over donor–acceptor pairs (directional typing)
- nrot = ligand rotatable-bond count (entropy / size control)

## Results (vs v1)
| metric            | v1 grid overlaps | v3 pair terms |
|-------------------|------------------|---------------|
| CV Ridge r        | 0.342            | 0.477         |
| CV GBR r          | 0.509            | 0.587         |
| held-out test Ridge r | 0.382        | 0.427         |
| held-out test GBR r   | 0.461        | 0.490         |
| test Ridge RMSE   | 1.66             | 1.58          |

## De-collinearization (the point of v3) — VIF on train features
gauss1 5.5, gauss2 6.9 (these two correlate — both contact gaussians),
repulsion 1.2, hydrophobic 2.6, hbond 2.3, nrot 1.6.
→ 4 of 6 channels are now genuinely independent axes (v1 had effectively 1).

## Frozen Ridge weights (standardized) — what actually carries the signal
gauss2 +0.91 (dominant), nrot −0.54, hydrophobic +0.33,
repulsion −0.09, gauss1 −0.05, hbond −0.04.

Honest read:
- The gain is real but modest and comes mostly from TWO new axes: nrot (entropy,
  correct sign: more rotatable bonds → weaker) and hydrophobic contact.
- gauss2 still dominates = "amount of contact" = the old size axis, just
  better-shaped. Size is still the main predictor.
- **Directional H-bond carries ~0 linear weight.** The channel I expected to be the
  orthogonal win does not help affinity prediction here. Most likely cause: crude
  receptor donor/acceptor typing (PDB name-based, no protonation/geometry), and a
  raw H-bond *count* is a poor linear proxy for ΔG across a chemically diverse set.
- Scorer is still weak (test r ~0.43–0.49; a useful SF is ~0.7–0.8). The
  train→test gap (GBR CV 0.59 vs test 0.49) is the leak-proof LP split biting, as
  intended — it is honest, not a bug.

## Next lever (evidence-based)
1. Fix the H-bond channel properly: angular ℓ=1 geometry + real protonation
   (donor H positions, acceptor lone-pair directions) instead of name+count.
2. Add back a real electrostatics channel with actual partial charges both sides
   (v3 dropped the signed elec term; Vina also lacks explicit electrostatics).
3. gauss1/gauss2 are redundant — collapse to one contact term, free a slot.

## Artifacts (out/v3/)
v3_summary.json, v3_vif.csv, v3_channel_weights.csv, v3_cv_folds.csv,
v3_features.npz, fig_v3_corr.png, fig_v3_scatter.png, fig_v3_channel_weights.png.
Code: src/v3_features.py, src/v3_train.py, src/run_v3_features.sh.
v1/v2 untouched; frozen in snapshots/v1v2_2026-11-02.tar.gz.

---

# Dataset composition & evaluation protocol (applies to v1/v3/v4)

## Dataset composition
- **PDBbind v2020 general set** (INDEX_general_PL.2020R1): **19,037** labeled complexes.
- **Usable: 18,137** — kept only entries with (a) a clean `=` affinity (Kd / Ki / IC50,
  converted to pKd = −log10 M; approximate `~ < >` and ranges dropped) AND (b) an
  LP-PDBBind leak-proof split + sequence-cluster assignment. ~900 dropped.
- **Splits (LP-PDBBind, leak-proof):** train 11,044 / val 2,340 / test 4,753.
- **11,831 unique sequence clusters** → used as GroupKFold groups, so no CV fold shares
  a protein with another (prevents memorizing a protein family).
- Separate held-out benchmark sets (own files, not in the 18,137): PoseBusters (pose),
  LIT-PCBA (enrichment), COCONUT (unlabeled screening library).

## How accuracy is evaluated
Two distinct things, by stage:

**1. Scoring-function accuracy — what v1/v3/v4 report (affinity regression).**
Predict pKd, compare to experiment. Metrics:
- **Pearson r** (linear correlation pred vs true pKd) — primary "scoring power".
- **Spearman ρ** (rank correlation — matters for screening/ranking).
- **RMSE** in pKd units (absolute error).
Measured three ways:
- **Cluster-aware 5-fold CV** on train (GroupKFold on sequence cluster) → out-of-fold r/ρ/RMSE.
- **Frozen held-out test** (4,753, never seen) → the honest number we quote.
- **val** (2,340) as a secondary check.
Both a linear Ridge (interpretable channel weights) and GradientBoosting (nonlinear ceiling)
are reported. Reference: a *useful* empirical scorer reaches r ≈ 0.7–0.8 on PDBbind test;
we are currently well below that (v4 test Ridge/GBR ≈ see report_v4).

**2. Full docking-model accuracy — planned held-out (not all run yet).**
- **Pose accuracy:** fraction with ligand-RMSD < 2 Å vs crystal pose, on PoseBusters,
  plus PoseBusters physical-validity checks. (v2 pilot: ~1% <2 Å — weak, scorer-limited.)
- **Virtual-screening enrichment:** EF (enrichment factor) and BEDROC on LIT-PCBA
  actives vs decoys.
- **Positive control:** recover known TBK1 binders (e.g. betaine) in the screen.

## What each metric compares (predicted vs experimental)
All three metrics compare **our predicted pKd** (from the learned channel scorer)
against the **experimental pKd** (PDBbind measured affinity label), over held-out
test complexes:
- **Pearson r** — linear correlation of predicted vs experimental pKd (do the values track).
- **Spearman rho** — rank correlation (do we order complexes strongest->weakest correctly).
- **RMSE** — mean magnitude of |predicted - experimental| in pKd units (absolute error).
Not a comparison to Vina or another model: it is prediction vs ground-truth affinity.

## [FIGURE TO MAKE - dataset composition schematic] for slides
A flow diagram, navy/gold palette, to be generated later for the PPT:

  PDBbind v2020 general set ......... 19,037 labeled complexes
     [filter: clean "=" affinity Kd/Ki/IC50 -> pKd, AND an LP-PDBBind split]
  Usable set ....................... 18,137
     [LP-PDBBind leak-proof split]
  -> train 11,044 | val 2,340 | test 4,753 (frozen held-out)
     train -> cluster-aware 5-fold GroupKFold (11,831 sequence clusters;
              no fold shares a protein) -> fold0..fold4

  Side boxes (separate benchmark sets, NOT in the 18,137):
   - PoseBusters -> pose accuracy (RMSD < 2 A)
   - LIT-PCBA    -> enrichment (EF / BEDROC)
   - COCONUT     -> screening library (unlabeled; the deployment target)

  Caption: train/select on PDBbind (cluster CV + frozen test); independent held-out
  benchmarks measure pose and enrichment; COCONUT is the library we screen.
