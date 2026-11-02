# v5 — Electrostatics + Desolvation channels (the terms Vina lacks, Glide/PIPER have)

v3/v4 reproduced Vina's physics and hit Vina's ceiling. The two terms Vina omits and
Glide/PIPER add are exactly what we had not tried: charge-charge electrostatics and
polar desolvation. v5 adds both as pairwise channels (6 -> 8). Everything else identical
to v4. Same complexes / splits / CV / labels; only channel weights learned.

## New channels
- **elec**: screened Coulomb  Σ q_i q_j / (4 r^2) over rec-lig pairs < 10 A. Partial
  charges BOTH sides: Gasteiger (ligand; receptor via RDKit MolFromPDBFile+Gasteiger,
  name-based fallback ~8% of pockets). Complementary charge -> negative (favorable).
- **desolv**: buried-polar penalty. For each polar atom (either side), σ_a × contact
  density from the OTHER molecule. σ: neutral N/O = 1, charged = 2.5, C/S = 0. Encodes
  "burying a polar/charged group costs desolvation unless H-bond/salt-bridge pays it back".

## Results (held-out test; the climb across the channel work)
| version | channels                              | test Ridge r | test GBR r |
|---------|---------------------------------------|--------------|------------|
| v1      | 5 grid overlaps                       | 0.382        | 0.461      |
| v3      | 6 Vina-style pair terms               | 0.427        | 0.490      |
| v4      | + geometric H-bond                    | 0.426        | 0.493      |
| **v5**  | **+ electrostatics + desolvation**    | **0.460**    | **0.524**  |
CV Ridge OOF 0.495 / GBR 0.597. This is the biggest single-step gain of the channel work.

## Do the new channels carry weight? Yes.
Frozen Ridge weights (standardized): gauss2 +1.05, nrot -0.49, **elec -0.43**,
hydrophobic +0.28, **desolv +0.27**, gauss1 -0.17, repulsion -0.11, hbond +0.005.
- **elec is a suppressor**: big weight (-0.43) yet ~0 standalone corr with pKd (-0.006).
  Electrostatics alone does not predict affinity, but conditional on contact it corrects
  the fit — that is where the held-out gain comes from.
- **desolv** carries +0.27 (standalone corr +0.109) — a modest but real independent push.
- Contrast hbond, still ~0: so elec/desolv earned their place, H-bond count did not.
VIF: elec 3.1 (independent), desolv 6.6 (partly tracks size/polar burial), gauss1/gauss2
~7-8 (still a redundant pair).

## Honest status
Steady climb v1->v5: test Ridge 0.38 -> 0.46, GBR 0.46 -> 0.52. Still below a useful
scorer (~0.7-0.8). We now have essentially the Glide term set (contact, clash, hydrophobic,
directional H-bond, entropy, electrostatics, desolvation) as linear channels, and it tops
out around r~0.5 at the crystal pose. The remaining ceiling is the linear-channels framing
itself, plus charge quality.

## Next levers (evidence-based)
- Charge quality: real force-field charges / proper protonation instead of Gasteiger-on-PDB
  (elec is already a strong weight on crude charges -> better charges should help most).
- Collapse redundant gauss1/gauss2 (VIF ~7) into one contact term; free a slot.
- This is the point to try the LEARNED equivariant field (prior-work approach 2): we now
  know which physics matters, so a learned channel set has a target to beat (r~0.5 linear).

## Artifacts (out/v5/) — v4 untouched, both kept
v5_summary.json, v5_vif.csv, v5_channel_weights.csv, v5_cv_folds.csv, v5_features.npz,
fig_v5_corr.png, fig_v5_scatter.png, fig_v5_channel_weights.png.
Code: src/v5_features.py, src/v5_train.py, src/run_v5_features.sh.
Dataset composition & evaluation protocol: see report_v3.md (same for all versions).
