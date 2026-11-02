# v7 — Pose test (moderate): do the channels recognize the native pose?

Affinity r (v1-v6) could not reward H-bond/electrostatics. Pose is where they should pay
off. This is a scoring-FUNNEL test, NOT blind docking: on 250 PDBbind test complexes, take
the crystal ligand (native) + 60 rigid-body decoy poses spanning an RMSD ladder; score each
with the v5 channels under three weightings; ask whether the score funnels to low RMSD.
RMSD = in-place coordinate RMSD (no symmetry correction). Native is always in the set, so
this measures native RECOGNITION among local decoys, not global docking.

## Result (n=250)
| weighting                         | top-scored pose < 2 A | median Spearman(score, RMSD) |
|-----------------------------------|-----------------------|------------------------------|
| affinity-fit v5 weights (full)    | 2.4%                  | -0.13                        |
| contact channels only (shape)     | 52.8%                 | -0.65                        |
| physical pose weights (all 8 ch.) | 99.6%                 | -0.69                        |

## Reading
- **Affinity-regression weights are the WRONG weights for pose** (2.4%): they are tuned to
  rank affinity ACROSS complexes, not poses WITHIN one. gauss2's huge affinity weight plus
  per-pose noise from the other channels destroys the funnel. (v2 warned of exactly this.)
- **Contact alone already funnels** toward native (53%, rho -0.65): shape complementarity
  is a real pose signal.
- **With physically-signed weights, the full channel set recognizes the native pose almost
  always** (99.6%). So the physics channels (clash penalty, H-bond, electrostatics, desolv)
  DO help pose — the place they earn their keep, unlike affinity r.

## Honest caveats
- Local funnel / cognate redocking with the native always present; decoys are rigid
  perturbations, so 99.6% is "recognizes native among local decoys", not blind-docking
  success. Blind global docking (the v2 FFT engine on these channels) is the real next test.
- Physical weights were hand-set (not fitted); they encode correct signs, not tuned values.

## Takeaway for the project
Two different jobs need two different weightings of the SAME channels: affinity ranking
(fit) vs pose recognition (physical). The channel set is sound for pose; the open work is
(a) learn pose-appropriate weights, (b) move from this local funnel to blind FFT docking.

## Artifacts (out/v7/)
v7_summary.json, v7_pose_per_complex.csv, fig_v7_funnel.png, fig_v7_top1.png.
Code: src/v7_pose.py, src/v7_fig.py. Reuses v5_features (v5 untouched).
