# v6 — Better partial charges (negative result, kept honestly)

Goal: improve charge quality feeding elec/desolv. v5 used Gasteiger both sides
(receptor via Gasteiger-on-PDB). v6: ligand MMFF94 (real force field, H folded into heavy
atoms); receptor Gasteiger-on-PDB with ionizable groups overridden to correct net formal
charges (Asp/Glu -1, Lys +1, Arg +1; His left to Gasteiger). No fabricated FF tables
(none installed on the server). Channel formulas identical to v5.

## The charges got more physical...
| quantity (train)         | v5 Gasteiger-on-PDB | v6 MMFF + formal |
|--------------------------|---------------------|------------------|
| elec mean                | +0.180              | -0.002           |
| fraction elec < 0 (favorable) | 2%             | **47%**          |
| standalone corr(elec, pKd)| -0.006 (no signal) | **-0.091** (right sign) |
| elec VIF                 | 3.1                 | 1.1 (independent) |
v6 electrostatics has the correct sign distribution (~half of native poses favorable) and
a real standalone correlation with affinity. v5's elec was essentially a zero-signal
suppressor that the model leaned on via collinearity.

## ...but held-out accuracy dropped
| version | test Ridge r | test GBR r |
|---------|--------------|------------|
| **v5**  | **0.460**    | **0.524**  |
| v6      | 0.423        | 0.496      |
Ridge weights v6: gauss2 +0.94, nrot -0.57, hydrophobic +0.34, **elec -0.16** (was -0.43),
**desolv +0.03** (was +0.27), gauss1 -0.11, repulsion -0.08, hbond -0.03.

Why it dropped:
- Honest elec gets a smaller weight (-0.43 -> -0.16); it no longer over-fits via collinearity.
- desolv collapsed (+0.27 -> +0.03) because the 'charged' flag changed (now formal-only,
  His and weak-Gasteiger atoms no longer flagged) -> its σ distribution shifted.
- Net: v5's larger (partly artifactual) elec/desolv weights happened to generalize to this
  test split slightly better. The drop is consistent across Ridge AND GBR, so it is real,
  not split noise.

## Interpretation (same lesson as the H-bond fix)
Making the electrostatics physically correct did NOT improve affinity regression. A
correct-sign electrostatics term's value should show up where geometry matters — docking
pose and actives/decoys enrichment — not in pKd correlation at a fixed crystal pose.
Affinity r is the wrong yardstick for charge quality too.

## Decision
- **v5 remains the best scorer** (test Ridge 0.46 / GBR 0.52). Keep it as the reference.
- Keep v6's charge code: the correct-sign electrostatics is the right thing to carry into
  the DOCKING search (v2) and enrichment tests, where it should pull poses correctly.
- Do NOT tune the 'charged' flag back just to recover desolv's weight — that would be
  fitting the metric, not the physics.

## Broader read of the channel arc (v1->v6)
test Ridge: 0.38 -> 0.43 (v3) -> 0.43 (v4) -> 0.46 (v5) -> 0.42 (v6).
We now have the full Glide-like physics as linear channels. It tops out ~r 0.5 at the
crystal pose. The ceiling is the linear-channels-at-native-pose framing itself, not any
single missing term. The evidenced next step is the LEARNED equivariant field (prior-work
approach 2): a learned channel set now has a concrete target to beat (v5's r~0.46 linear /
0.52 nonlinear), and the physics experiments show which interactions it must encode.

## Artifacts (out/v6/) — v5 untouched, both kept
v6_summary.json, v6_vif.csv, v6_channel_weights.csv, v6_cv_folds.csv, v6_features.npz,
fig_v6_*.png. Code: src/v6_features.py (MMFF + formal charges), src/v6_train.py.
