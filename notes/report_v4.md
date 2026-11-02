# v4 — Geometric H-bond channel (fixing v3's dead channel)

v3 finding: the H-bond channel carried ~0 weight. Hypothesis: a blind distance-gated
donor/acceptor COUNT is not selective (any donor near any acceptor counts, regardless
of orientation), so it just re-encodes contact and the model ignores it. v4 makes the
channel GEOMETRIC and tests that hypothesis.

## What changed (only the hbond channel; other 5 identical to v3)
Each candidate H-bond is gated by geometry, score = dist_fac · donor_fac · acceptor_fac:
- Ligand donors get REAL H coordinates (RDKit AddHs(addCoords=True)); receptor H/lone-pair
  directions approximated from heavy-atom geometry (no reduce/obabel on the server).
- dist_fac: gaussian peak at H···A=2.0 Å (ligand donor) or D···A=2.9 Å (receptor donor).
- donor_fac: ∠D–H···A near 180° (linear H-bond); when H unknown, acceptor must sit
  opposite the donor's bonded neighbor.
- acceptor_fac: donor/H must approach from the lone-pair side (opposite acceptor's neighbor).
Self-check: a collinear H-bond scores 1.0, a 90°-bent one scores 0.0.

## Result: the fix works as engineering but does NOT help the affinity scorer
| quantity                       | v3 (blind count) | v4 (geometric) |
|--------------------------------|------------------|----------------|
| corr(hbond, pKd) on train      | 0.020            | **0.085**      |  (4× — real H-bonds now)
| corr(hbond, gauss2)            | 0.388            | 0.517          |
| Ridge weight on hbond          | −0.042           | −0.032 (≈0)    |
| held-out test Ridge r          | 0.427            | 0.426          |
| held-out test GBR r            | 0.490            | 0.493          |
VIF unchanged (hbond 2.1, still independent of the others).

## Honest interpretation (this is a finding, not a bug)
- The geometry gate is correct and measurably improves the channel: its standalone
  correlation with affinity quadrupled (0.02 → 0.085) because non-H-bond contacts are
  now filtered out. So "fix the geometry" did exactly what it should to the channel.
- But the OVERALL scorer does not move, for two reasons:
  1. The absolute H-bond↔affinity signal is small (~0.085). **H-bond COUNT is a poor
     linear predictor of ΔG** — a protein-ligand H-bond largely replaces ligand-water
     and protein-water H-bonds, so its net enthalpic contribution tends to cancel. This
     is well established; our data reproduces it cleanly.
  2. What little signal it has overlaps with contact (corr 0.52 with gauss2): good
     H-bonds occur where atoms are in close contact, which gauss2 already captures.
     With gauss2 at weight +0.90, the model has no use for hbond → weight stays ~0.

## Consequence for the project (where the H-bond channel actually earns its place)
H-bonds govern WHERE a ligand sits and WHICH ligand fits (pose specificity, actives vs
decoys), not the MAGNITUDE of pKd. So the correct test for this channel is not affinity
regression r — it is:
  - pose accuracy (PoseBusters, RMSD<2 Å): does the geometric H-bond term pull the
    docked pose onto the right spot?
  - enrichment (LIT-PCBA, EF/BEDROC): does it help separate binders from decoys?
Keep the geometric H-bond channel (it is physically right and cheap) and judge it in the
docking/enrichment evaluation, where v2's pose search was scorer-limited. Affinity r was
the wrong yardstick for it.

## Still-open levers for the affinity scorer specifically
- Real electrostatics with partial charges both sides (v3/v4 have none).
- Desolvation / buried-polar-without-partner penalty (the term that makes H-bond COUNT
  stop looking free — may expose net H-bond signal).
- gauss1/gauss2 are redundant (VIF ~6); collapse to one, free a slot.

## Artifacts (out/v4/) — v3 untouched, both kept
v4_summary.json, v4_vif.csv, v4_channel_weights.csv, v4_cv_folds.csv, v4_features.npz,
fig_v4_corr.png, fig_v4_scatter.png, fig_v4_channel_weights.png.
Code: src/v4_features.py (geometric hbond), src/v4_train.py, src/run_v4_features.sh.
Dataset composition & evaluation protocol: see report_v3.md (same for v1/v3/v4).
