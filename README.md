# screen — field-based docking for virtual screening

A research prototype that scores protein–ligand binding as a **field-overlap landscape**
instead of a single number. Each molecule is rendered as a multi-channel field; binding is
the overlap of those fields over pose — a *curve* whose peak gives both the pose and the
score. Fields are rotation-equivariant, so the whole landscape is computable cheaply by FFT.

> Research in progress. The scorer is not yet at classical-docking accuracy; the goal of
> this repo is a principled line that improves as physical interactions are added. All
> numbers below are measured on held-out data. See [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md).

## Idea
- Fast embedding methods collapse a molecule to one **point** → the pose is lost.
- We represent each molecule as a **field** and score binding as **a weighted sum of
  per-interaction channel overlaps** (contact, clash, hydrophobic, H-bond, electrostatics,
  desolvation). Only the channel weights are learned; labels are **experimental affinity**.
- Prior work (ESF, arXiv 2312.04323) did scalar-field + SE(3) FFT correlation for **rigid**
  bodies; the intended contribution here adds **internal torsion** (fragment decomposition +
  kinematic tree) to that landscape.

## What's here (code only — no datasets)
```
src/                 feature extraction, scoring, docking, figures
  v1_*.py            v1 scorer: grid-overlap channels at the crystal pose
  v2_dock.py         FFT rigid search + greedy torsion (pose search)
  v4_features.py     geometric directional H-bond channel
  v5_features.py     + electrostatics + desolvation channels  (best scorer)
  v6_features.py     + better partial charges (MMFF + formal) — honest negative result
  v7_pose.py         pose scoring-funnel test (native recognition)
  v7_fig.py          pose figures
  make_example_figs.py   result figures + example PDB export
  *_labels.py / run_*.sh / make_*.py   label/split building, sharded runs, figures
presentation/        talk material (markdown + result figures + example PDBs for Chimera)
RESEARCH_SUMMARY.md  full write-up of the v1→v7 arc
```
Datasets (PDBbind, PoseBusters, LIT-PCBA, COCONUT) and large outputs are **not** in the
repo; scripts expect them under a local `data/` and write to `out/` (both git-ignored).

## Results so far (held-out)
Building the scorer up from shape overlaps to a full, Glide-like channel set:

| step | channels | test Ridge r | test GBR r |
|---|---|---|---|
| v1 | grid overlaps | 0.38 | 0.46 |
| v3 | Vina-style pair terms | 0.43 | 0.49 |
| v4 | + geometric H-bond | 0.43 | 0.49 |
| **v5** | **+ electrostatics + desolvation** | **0.46** | **0.52** |
| v6 | + better charges | 0.42 | 0.50 |

A *useful* scorer is ~0.7–0.8, so this is still a weak scorer — the point is the climb and
what it teaches. H-bond count and charge quality barely move affinity r (an honest finding:
affinity is the wrong yardstick for them). Where they pay off is the **pose**:

| weighting of the same channels | native pose recognized (top-1 < 2 Å, n=250) |
|---|---|
| affinity-fit weights | 2.4% |
| contact only | 53% |
| physical pose weights | 99.6% |

(Scoring-funnel / cognate-redocking test with the native always present — not blind docking.)

## Running
Python ≥3.11 with numpy, scipy, scikit-learn, pandas, matplotlib, rdkit (see
`pyproject.toml`). Feature extraction is sharded (`src/run_*.sh`); training and evaluation
read the shards and a label manifest. Point the scripts at a local `data/` holding the
structures. This repo is a prototype, not a packaged tool.

## Status & next
- Working: data pipeline, channels, FFT engine, torsion, cluster-aware CV evaluation.
- Gap: affinity scorer ~r 0.5 at the crystal pose; pose test is a local funnel, not blind docking.
- Next: learn pose-appropriate weights; blind FFT docking; PoseBusters / LIT-PCBA benchmarks;
  apply to an animal-biotechnology target (natural-product screening against TBK1).
