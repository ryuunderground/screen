# v1–v4 overview — what each version is (for the slides)

These are NOT a live multi-stage pipeline and NOT running concurrently. They are
experiment versions: the same problem solved with improving channels. All finished;
only result files remain on disk. Two distinct TASKS:

## Task A — scoring function (predict binding affinity)
Give the KNOWN correct pose (crystal structure). Compute channel overlaps there,
learn channel weights, predict affinity (pKd). Question: "can channels predict how
strongly a ligand binds?" Metric: correlation of predicted vs experimental pKd.
- **v1** — channels = 5 fuzzy grid-overlap scalars. Held-out test Ridge r ≈ 0.38.
  Problem: all 5 channels scale with buried volume → one size axis (collinear).
- **v3** — SAME task, channels replaced by 6 Vina-style distance-gated PAIR terms
  (gauss1/2, repulsion, hydrophobic, hbond, nrot). Test r ≈ 0.43, collinearity broken.
- **v4** — SAME as v3, only the H-bond channel upgraded to a GEOMETRIC gate (angle +
  protonation). H-bond channel improves standalone, but overall r unchanged (≈0.43):
  H-bond COUNT does not predict affinity magnitude.
→ v1 → v3 → v4 are successive improvements of ONE task; each replaces the last.

## Task B — docking search (find the pose)
Do NOT give the pose. From the molecules alone, use FFT rotation search + torsion to
FIND where the ligand sits. Question: "does it dock to the right place?" Metric: pose
RMSD vs crystal.
- **v2** — FFT multi-channel search + greedy torsion. RMSD<2 Å ≈ 1%. Showed the
  bottleneck is the SCORING FUNCTION, not the search → motivated going back to Task A.

## How they relate
v2 (search) revealed the scorer is the weak link → v1 strengthened into v3/v4 (channels)
→ the better scorer is meant to go back into the v2 search. Nothing runs automatically
between them; each version is launched on demand.

## Current honest status
Scorer still weak (test r ≈ 0.43 linear / 0.49 nonlinear; a useful scorer is ≈0.7–0.8).
Pose search scorer-limited. H-bond channel physically correct but affinity is the wrong
yardstick for it (judge it on pose/enrichment instead).
