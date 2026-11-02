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
src/
  v1_features.py / v1_labels.py / v1_train.py   v1 scorer: grid-overlap channels at crystal pose
  v2_dock.py / v2_figs.py                        FFT rigid search + greedy torsion (pose search)
  v3_features.py                                 Vina-style distance-gated pair channels
  v4_features.py                                 + geometric directional H-bond channel
  v5_features.py                                 + electrostatics + desolvation  (best scorer)
  v6_features.py                                 + better partial charges (MMFF + formal) — negative result
  v{3..6}_train.py / run_v{1..6}_features.sh     training + sharded feature extraction per version
  v7_pose.py / v7_fig.py                         pose scoring-funnel test + figures
  pose_learn.py                                  learn pose-ranking weights (native > decoys by RMSD)
  make_example_figs.py                           result figures + example PDB export
  make_figures.py / make_slides.py               schematic figures / slide generation
  fourier_basics.py / toy_docking_2d.py / concepts_equivariance.py / torsion_spectrum.py /
  fft_dock_pilot.py / reexpansion_error.py       early concept & pilot studies
notes/               research log — per-version findings (report_v1_v2 … report_v7, versions_overview)
presentation/        talk material (markdown + result figures)
RESEARCH_SUMMARY.md  full write-up of the v1→v7 arc
```
The version-by-version research process is visible in `src/v{1..7}*` and the matching
`notes/report_v*.md`. Start with `notes/versions_overview.md`.
Datasets (PDBbind, PoseBusters, LIT-PCBA, COCONUT) and large outputs are **not** in the
repo; scripts expect them under a local `data/` and write to `out/` (both git-ignored).

## Data (what was used, how many, and what was left out)

**Used for the scorer — PDBbind v2020 "general" set.**
- Source index: **19,037** labeled protein–ligand complexes (`INDEX_general_PL.2020R1`).
- **Used: 18,137.** Kept only complexes that had **both** (a) a clean `=` affinity
  (Kd / Ki / IC50 → pKd = −log₁₀ M) **and** (b) an entry in the LP-PDBBind leak-proof split
  with a sequence cluster.
- **Dropped: 900.** Two reasons: (a) **270** had a non-clean affinity — approximate or a range
  (`~`, `<`, `>`), or an unparseable unit (263 approximate/range + 7 unparseable); (b) **630**
  were in LP-PDBBind but had **no leak-proof split assigned** (blank `new_split`). Excluded
  rather than guessed.
- Split (LP-PDBBind): **test 4,753 frozen**; the remaining **13,384** (train + val pooled) run
  through **cluster-aware 5-fold CV**, grouped by sequence cluster so no fold shares a protein.

**Downloaded and prepared but NOT YET used (benchmarks pending):**
- **PoseBusters** — intended pose-accuracy benchmark (RMSD < 2 Å); blind docking not run yet.
- **LIT-PCBA** — intended virtual-screening enrichment benchmark (EF / BEDROC); not run yet.
- **COCONUT** — natural-product library to screen in the application phase; not screened yet.

Datasets themselves are not redistributed here; see each source for licensing.

## Results so far (held-out)
Building the scorer up from shape overlaps to a full, Glide-like channel set:

| step | channels | test Ridge r | test GBR r |
|---|---|---|---|
| v1 | grid overlaps | 0.40 | 0.48 |
| v3 | Vina-style pair terms | 0.43 | 0.50 |
| v4 | + geometric H-bond | 0.43 | 0.50 |
| **v5** | **+ electrostatics + desolvation** | **0.47** | **0.54** |
| v6 | + better charges | 0.43 | 0.51 |

A *useful* scorer is ~0.7–0.8, so this is still a weak scorer — the point is the climb and
what it teaches. H-bond count and charge quality barely move affinity r (an honest finding:
affinity is the wrong yardstick for them). Where they pay off is the **pose**:

| weighting of the same channels | native pose recognized (top-1 < 2 Å, n=1200) |
|---|---|
| affinity-fit weights | 3.2% |
| **learned pose weights** | **94.0%** |

Pose weights are **learned separately** from affinity — by pairwise ranking (native scores
above decoys, supervised on RMSD, not affinity). Scoring-funnel / cognate-redocking test with
the native always present — not blind docking. Example overlays: RMSD 0.40 / 0.68 Å (4u0s, 3ay0).

## Running
Python ≥3.11 with numpy, scipy, scikit-learn, pandas, matplotlib, rdkit (see
`pyproject.toml`). Feature extraction is sharded (`src/run_*.sh`); training and evaluation
read the shards and a label manifest. Point the scripts at a local `data/` holding the
structures. This repo is a prototype, not a packaged tool.

## Status & next
- Working: data pipeline, channels, FFT engine, torsion, cluster-aware CV evaluation.
- Gap: affinity scorer ~r 0.5 at the crystal pose; pose test is a local funnel, not blind docking.
- Done: pose-appropriate weights are now learned (94% top-1 vs 3.2% for affinity-fit weights).
- Next: blind FFT docking; PoseBusters / LIT-PCBA benchmarks; apply to an
  animal-biotechnology target (natural-product screening against TBK1).
