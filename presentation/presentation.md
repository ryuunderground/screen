# 발표자료 생성 프롬프트 + 최종 콘텐츠
**Field-Based Docking for Virtual Screening — a research-in-progress talk**

이 파일 하나로 발표 슬라이드를 만든다. 아래 **형식 규칙**을 지키고, **슬라이드별 콘텐츠**를
그대로 쓴다. **흐름 순서가 핵심이므로 순서를 바꾸지 말 것.**

## 그림 안내 (중요)
- `figures/` 폴더에는 **실험해서 모델 성능을 보여주는 결과 그림 4개만** 있다
  (`channel_climb.png`, `pose_recognition.png`, `pose_example.png`, `pose_funnel.png`).
- Introduction/Objective/M&M의 **개념 도식**은 성능 figure가 아니라 폴더에 없다. 해당 슬라이드는
  `[개념 도식]`으로 표시 — 기존 `../slides/assets/`를 쓰거나 새로 그린다.
- **Results 슬라이드만 `figures/`의 실제 그림을 그대로 배치**한다.

## 형식 규칙
- 섹션: **Introduction → Objective(1장) → Materials & Methods → Results → Discussion(1장)
  → Further study(1장)**.
- 톤: **연구 진행 중**. "비나엔 아직 못 미치지만 물리를 더해가며 올리는 중."
- 수치는 전부 **실제 측정값**. 미확인 값은 쓰지 않는다. 생물학/인용 주장은 `[cite]`.
- 디자인: 4:3, Navy `#1F3864` / Gold `#C8A94B`, 흰 배경, Arial. 슬라이드당 가운데 그림 1 +
  불릿 ≤ 3.
- 평가 포인트: **Intro = 기존 문제가 왜 중요한가(도킹의 역할) + 흐름**, **Objective = 연구 질문 +
  가설**, **M&M = 디테일보다 rationale**, **Discussion = Limitation**, **Further study = 동물생명공학
  중요성(TBK1 예시)**.

---

# Introduction
*(아래 1→8 순서는 연구 서사의 논리 사슬 — 반드시 이 순서)*

## Slide 1 — Title
**From Points to Curves: Field-Based Docking that Predicts the Binding Pose**
- *Representing molecules as fields to recover the bound conformer, at screening scale*
- [Author] · [Lab / Affiliation] · Research in progress

## Slide 2 — (1) Docking is central — but slow
- Molecular docking predicts **whether and how** a ligand binds a target — a core tool for
  finding bioactive compounds `[cite]`
- Crucially it returns the **binding pose** (the 3D conformer), i.e. structural insight
- But classical docking is **computationally slow** — hard to run at library scale

🖼 `[개념 도식]` 도킹 = 포즈 + 친화도, 그러나 느림
_Note: 도킹이 왜 중요한지(포즈까지 준다) + 병목(느림)을 먼저 깐다._

## Slide 3 — (2→3) To go faster: DrugCLIP
- To screen faster, **DrugCLIP** embeds pocket & ligand as vectors; binding ≈ similarity
  `[cite: DrugCLIP, arXiv:2310.06367]`
- Extremely fast over huge libraries — its strength
- But it only answers **"binds or not"** — a single similarity score

🖼 `[개념 도식]` 리셉터·리간드 = 점, 점선 = similarity
_Note: 빠르게 하려고 대조학습(DrugCLIP) 등장. 단, 결합 여부만._

## Slide 4 — (4) What's lost: the conformer / how it binds
- DrugCLIP gives **no conformer** — you cannot see *how* the ligand sits in the pocket
- Why this matters: **classical docking returns that pose/conformer** — it's what structural
  interpretation and follow-up need
- So speed came at the cost of the pose

🖼 `[개념 도식]` 점수는 있는데 포즈가 없음(물음표 포즈)
_Note: 컨포머 정보 소실. 기존 도킹은 그것까지 주기 때문에 이게 손실이다._

## Slide 5 — (5) BindCLIP: shows how it binds, but still no pose
- **BindCLIP** was proposed to indicate *how* a ligand binds `[cite]`
- But it **still does not return the actual binding pose**
- The gap remains: fast methods can't give the bound 3D conformer

🖼 `[개념 도식]` BindCLIP — 방향은 알려주나 포즈 없음
_Note: 포즈를 주려 BindCLIP이 나왔지만 여전히 실제 포즈는 못 준다._

## Slide 6 — (6) A different representation: scalar fields, not points
- Another line represents a molecule as a **scalar field** instead of a point
  `[cite: ESF, arXiv:2312.04323]`
- A field carries the geometry a point throws away
- Binding scored by field overlap with SE(3) / FFT correlation

🖼 `[개념 도식]` 점 → 스칼라장 표현 (또는 `../slides/assets/field_channels.png`)
_Note: 점이 아니라 스칼라장으로 표현한 선행 연구(ESF 2312.04323)._

## Slide 7 — (7) Its limitation: rigid only, no internal torsion
- That field method uses **rigid-body** transforms
- So it **cannot see internal torsion** — the flexible rotatable bonds
- Real ligands flex; rigid scoring misses the true bound conformer

🖼 `[개념 도식]` 강체만 — 꺾이는 결합 표시 (또는 `../slides/assets/fragment_tree.png`)
_Note: 강체 변환이라 내부 torsion을 못 본다 = 핵심 공백._

## Slide 8 — (8) Our move: redefine binding as a curve to capture torsion
- We redefine binding as a **score curve over pose** (field-overlap landscape), not a single value
- The curve's axes **include internal torsion** → we can see *which flexible conformer* binds
- Its peak gives the pose; FFT draws the whole curve cheaply

🖼 `[개념 도식]` 점 → 곡선 (또는 `../slides/assets/point_vs_curve.png`)
_Note: 곡선으로 재정의해 내부 torsion까지 본다 — 우리 기여._

---

# Objective

## Slide 9 — Objective & hypothesis (1 slide)
- **Goal:** predict not just *whether* a ligand binds, but **in which pose** — explicitly
  accounting for **internal torsion**
- **Hypothesis:** accounting for the **docking pose** (pose + torsion), not just an affinity
  score, **improves accuracy**
- **Approach:** field-overlap curve + FFT + torsion; validate on held-out data

🖼 `[개념 도식]` 내부 torsion 고려 → 포즈 예측 (또는 `../slides/assets/point_vs_curve.png`)
_Note: 목표 = 내부 torsion 고려한 포즈 예측. 가설 = 포즈까지 보면 정확도 ↑._

---

# Materials & Methods
*(rationale 중심 — 세부 구현 생략)*

## Slide 10 — Data & evaluation (what we used)
- PDBbind v2020 general: index **19,037** labeled complexes → **used 18,137**
- Kept only: clean `=` affinity (Kd/Ki/IC50 → pKd) **and** an LP-PDBBind leak-proof split assigned
- **Dropped 900:** 270 non-clean affinity (`~`,`<`,`>`/unparseable) + 630 with no leak-proof split assigned
- **test 4,753 frozen**; remaining **13,384** (train+val pooled) → **cluster-aware 5-fold CV** (no protein leakage)
- Metric = **predicted vs experimental affinity** (Pearson r, Spearman ρ, RMSE); pose measured separately

🖼 `[개념 도식]` 클러스터 CV + held-out (또는 `../slides/assets/cv_split.png`)
_Note: 평가는 예측 pKd vs 실험 pKd. 누출 방지(같은 단백질 패밀리 암기 방지)가 핵심 rationale._

## Slide 11 — Representation & score: molecule as multi-channel field
**(1) 표현** — render receptor & ligand as fields on a grid, **one channel per interaction type**
(contact/shape, clash, hydrophobic, H-bond, electrostatics, desolvation)

**(2) 점수** — overlap the two molecules' fields and sum:
```
Vina :  score = Σ (atom-pair interaction terms)      — molecule as POINTS
Ours :  Score = Σ wᵢ · overlapᵢ  (per-channel field overlap)  — molecule as FIELDS
```
- Mathematically isomorphic to Vina's pair terms — but as fields, **FFT can sweep the whole
  pose at once** (→ Slide 12)
- **Only the weights wᵢ are learned; labels = experimental affinity (pKd), not binary bind/no-bind**

🖼 **`figures/field_cube.png`** — ligand & receptor rendered as 3D field cubes
(R³→scalar), one cube per channel: Structure / H-bond / Hydrophobic
_Note: (1) 분자를 여러 3D 장(큐브)으로 → (2) 두 장의 겹침 가중합이 점수. wᵢ만 학습, 라벨은 연속 친화도._
_(개념 도식: 실제 리간드/주머니 원자에서 계산한 장. 점수는 v5에선 pairwise이나, "이런 장을 겹친다"는 표현용.)_

## Slide 12 — Scoring the pose landscape by FFT + torsion (rationale)
- One FFT gives the overlap at **every translation** for a fixed orientation → max over
  orientation & translation = pose + score (the curve's peak)
- **Internal flexibility:** cut only bonds that move atoms, score fragments on a kinematic tree
- *Rationale:* FFT makes "draw the whole curve" affordable; torsion is the part prior fields missed

🖼 `[개념 도식]` FFT 상관 지형 / torsion 트리 (또는 `../slides/assets/corr_landscape.png`)
_Note: 왜 FFT·곡선·torsion인가에 집중._

---

# Results
*(아래부터는 `figures/`의 실제 실험 결과 그림을 그대로 배치)*

## Slide 13 — Adding physics lifts the scorer
- Built the scorer up from shape overlaps to a full, Glide-like channel set
- **Electrostatics + desolvation gave the biggest single lift** (best so far)
- Honest: still below a *useful* scorer (~0.7–0.8) — the point is the climb

🖼 **`figures/channel_climb.png`**  ← 실험 결과

| step | channels | test Ridge r | test GBR r |
|---|---|---|---|
| grid overlaps | baseline | 0.40 | 0.48 |
| Vina-style pair terms | +contact/H-bond/entropy | 0.43 | 0.50 |
| **+ electrostatics + desolvation** | **best** | **0.47** | **0.54** |

_Note: 예비 결과, held-out PDBbind test. 아직 비나급 아님 — 올라가는 중._

## Slide 14 — The pose is where it pays off (hypothesis check)
- Affinity-only channels barely move r — but **our hypothesis is about the POSE**
- On **native-pose recognition**, the same channels (physically weighted) succeed strongly
- Supports the hypothesis: using the pose, not just an affinity number, is where the gain is

🖼 **`figures/pose_recognition.png`**  ← 실험 결과

| weighting of the same channels | top-scored pose < 2 Å (n = 250) |
|---|---|
| affinity-fit weights | 2.4% |
| contact only | 53% |
| **physical pose weights** | **99.6%** |

_Note: 가설("포즈까지 보면 낫다")을 지지하는 결과. 디코이=native 강체 교란(국소 funnel, blind 아님)._

## Slide 15 — It binds like THIS; we predict THAT
- Crystal (true) pose vs our top-ranked predicted pose, overlaid in the pocket
- Held-out complexes; predicted pose picked by score (native not given)
- They nearly coincide — **RMSD 1.16 Å and 0.95 Å**

🖼 **`figures/pose_example.png`**  ← 실험 결과 (남색=실제 결정구조, 골드=예측)
_Note: 키메라용 PDB 제공 — `example_structures/`의 protein + ligand_native(NAT) + ligand_predicted(PRD)._

## Slide 16 — The scoring funnel toward the native pose (optional)
- Score vs RMSD for three weightings of the **same** channels
- Contact already funnels; physical weights sharpen it; affinity weights do not

🖼 **`figures/pose_funnel.png`**  ← 실험 결과
_Note: 시간 없으면 Slide 14·15로 충분._

---

# Discussion

## Slide 17 — Limitations (1 slide)
- **Scorer still weak:** affinity correlation ~r 0.5 at the crystal pose — below a useful
  scorer and not yet at classical-docking level
- **Pose result is a local funnel, not blind docking:** decoys are rigid perturbations with
  the native always present; true blind global docking is untested
- **Crude inputs:** Gasteiger-on-PDB / name-based atom typing; better force-field charges did
  **not** help affinity r (an honest negative result)
- **Cognate redocking on PDBbind only:** no external affinity test, no enrichment (LIT-PCBA)
  or PoseBusters pose benchmark yet

🖼 (텍스트 슬라이드)
_Note: 한계를 숨기지 않는다 — 약한 점수함수, 국소 funnel, 거친 입력, 벤치마크 미완._

---

# Further study

## Slide 18 — Further study & why it matters for animal biotechnology (1 slide)
- **Method:** learn pose-appropriate weights; move to **blind FFT docking**; run PoseBusters
  (RMSD<2 Å) and LIT-PCBA (EF/BEDROC) benchmarks
- **Animal-biotechnology relevance — example TBK1:** TBK1 is a target linked to **aging and
  fat accumulation** `[cite]`; a faithful screen could find natural-product modulators of
  animal metabolism and health
- **Plan:** screen natural products (COCONUT) against TBK1, recovering known binders such as
  **betaine** as a positive control `[cite]`

🖼 (텍스트 슬라이드; 원하면 TBK1 구조 도식)
_Note: 동물생명공학 중요성은 여기서 TBK1 예시로. 노화·지방축적 표적에 천연물 스크리닝, betaine 양성대조._

---

## figures/ 대응표 (실험 결과 그림 — 그대로 배치)
| 파일 | 슬라이드 | 내용 |
|---|---|---|
| channel_climb.png | 13 | 채널 추가에 따른 held-out test r 상승 |
| pose_recognition.png | 14 | 가중치별 native 포즈 인식률 (2.4 / 53 / 99.6%) |
| pose_example.png | 15 | 실제(결정) vs 예측 포즈 오버레이 (RMSD 1.16 / 0.95 Å) |
| pose_funnel.png | 16 | 점수 vs RMSD 깔때기 (가중치 3종) |

개념 도식(Intro/Objective/M&M)은 `../slides/assets/`를 쓰거나 새로 제작.
인용: DrugCLIP arXiv:2310.06367 · 스칼라장(ESF) arXiv:2312.04323 · BindCLIP `[cite 확인]` ·
TBK1/betaine 생물학 주장 `[cite 확인]`. 연구 과정 요약은 루트 `RESEARCH_SUMMARY.md`.
