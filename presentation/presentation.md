# 발표자료 생성 프롬프트 + 최종 콘텐츠
**Field-Based Docking for Virtual Screening — a research-in-progress talk**

이 파일 하나로 발표 슬라이드를 만든다. 아래 **형식 규칙**을 지키고, **슬라이드별 콘텐츠**를
그대로 쓴다.

## 그림 안내 (중요)
- `figures/` 폴더에는 **실험해서 모델 성능을 보여주는 결과 그림 4개만** 들어 있다
  (내가 데이터로 직접 돌려 만든 것): `channel_climb.png`, `pose_recognition.png`,
  `pose_example.png`, `pose_funnel.png`.
- **개념 도식**(깔때기, 점→곡선, 채널 모식도 등)은 성능 figure가 아니므로 이 폴더에 없다.
  해당 슬라이드는 `[개념 도식]`으로 표시해 두었으니, 기존 `../slides/assets/`의 그림을 쓰거나
  새로 그리면 된다. **결과/성능 슬라이드만 `figures/`의 실제 그림을 그대로 배치**한다.

## 형식 규칙
- 섹션: **Introduction → Objective(1장) → Materials & Methods → Results → Discussion(1장)
  → Further study(1장)**.
- 톤: 완성이 아니라 **연구 진행 중**. "비나엔 아직 못 미치지만 물리를 더해가며 올리는 중."
- 모든 수치는 **실제 측정값**(아래 표/숫자). 미확인 값(타 도구 수치 등)은 쓰지 않는다. 생물학적
  주장은 `[cite]`.
- 디자인: 4:3, Navy `#1F3864` / Gold `#C8A94B`, 흰 배경, Arial. 슬라이드당 가운데 그림 1 +
  불릿 ≤ 3.
- 평가 포인트: **Intro = 기존 문제 + 동물생명공학에서 왜 중요한가**, **Objective = 연구 질문 +
  가설**, **M&M = 디테일보다 rationale**, **Discussion = Limitation**.

---

# Introduction

## Slide 1 — Title
**From Points to Curves: Field-Based Docking for Virtual Screening**
- *Scoring molecular binding as a field-overlap landscape, not a single number*
- [Author] · [Lab / Affiliation] · Research in progress

## Slide 2 — Why this matters for animal biotechnology
- Bioactive natural compounds (feed additives, metabolic & anti-adipogenic modulators)
  are a largely un-screened resource for animal health and production `[cite]`
- Finding which compound binds which target is the rate-limiting step
- A faithful, scalable screen would let animal-biotech labs mine this space systematically

🖼 `[개념 도식]` 천연물 → 표적 스크리닝 병목 (또는 `../slides/assets/funnel.png`)
_Note: 천연물 기반 생리활성 물질(가축 대사·지방·항노화 조절 등) 탐색이 동물생명공학의 핵심인데,
어떤 분자가 어떤 표적에 붙는지 거르는 가상 스크리닝이 병목._

## Slide 3 — The existing problem: scale AND accuracy
- Compound/analog space is enormous (billions) — speed matters
- Classical docking is slow per compound **and** misses real binders `[cite]`
- Bottleneck is not only speed but **scoring faithfulness**

🖼 `[개념 도식]` 규모+정확도 두 병목 (또는 `../slides/assets/funnel.png`)
_Note: 느린 속도만이 아니라 진짜 결합자를 놓치는 점수함수 부정확성이 더 큰 문제._

## Slide 4 — Where current fast methods fall short
- Fast embedding methods compress a molecule to a single **point** → the 3D pose is lost
- A point can say "similar", but not **how** or **where** a molecule binds
- We want the pose a point throws away — while staying cheap enough to screen

🖼 `[개념 도식]` 점 → 곡선 (또는 `../slides/assets/point_vs_curve.png`)
_Note: 빠른 방법은 분자를 점 하나로 뭉개 포즈를 버린다. 포즈를 되살리되 값싸야 한다._

---

# Objective

## Slide 5 — Research question & hypothesis (1 slide)
- **Question:** can we score binding by the *shape of a pose landscape* (a curve), instead
  of a single affinity number, and does that make docking more faithful?
- **Hypothesis:** representing each molecule as a **field** and scoring with physically
  grounded **channels** — crucially using the **pose**, not just an affinity scalar — recovers
  binding geometry that point/affinity-only methods miss.
- **Test:** build the channel scorer; measure affinity correlation AND native-pose recognition
  on held-out data.

🖼 `[개념 도식]` 점→곡선 / 가설 요약 (또는 `../slides/assets/point_vs_curve.png`)
_Note: 가설 = "포즈까지 보면 더 낫다." 친화도 숫자만이 아니라 자세 지형을 본다._

---

# Materials & Methods
*(rationale 중심 — 세부 구현 생략)*

## Slide 6 — A molecule as a multi-channel field (rationale)
- Render receptor & ligand as fields on a grid, **one channel per interaction type**
- Channels mirror binding physics: contact/shape, clash, hydrophobic, H-bond,
  electrostatics, desolvation
- Score = **weighted sum of channel overlaps**; only the weights are learned (labels =
  experimental affinity)

🖼 `[개념 도식]` 다채널 장 표현 (또는 `../slides/assets/field_channels.png`)
_Note: 채널 = 물리 상호작용을 각각 장으로. 학습은 가중치만._

## Slide 7 — Scoring the pose landscape by FFT (rationale)
- One FFT gives the overlap at **every translation** for a fixed orientation
- Max over orientation & translation → docked pose + score (the curve's peak)
- Internal flexibility: cut only bonds that move atoms, score fragments on a kinematic tree
- *Rationale:* FFT makes "draw the whole curve" affordable at screening scale

🖼 `[개념 도식]` FFT 상관 지형 / torsion 트리 (또는 `../slides/assets/corr_landscape.png`)
_Note: 디테일보다 "왜 FFT/장인가"에 집중._

## Slide 8 — Data & evaluation (rationale)
- PDBbind: **18,137 complexes** (from 19,037; clean `=` affinity + leak-proof split)
- **Cluster-aware 5-fold CV** (grouped by sequence) + frozen held-out test → no protein leakage
- Metrics compare **predicted vs experimental affinity** (Pearson r, Spearman ρ, RMSE);
  pose measured separately
- *Rationale:* cluster splitting prevents "memorizing a protein family" — honest generalization

🖼 `[개념 도식]` 클러스터 CV + held-out (또는 `../slides/assets/cv_split.png`)
_Note: 평가는 예측 pKd vs 실험 pKd. 누출 방지가 핵심 rationale._

---

# Results
*(아래부터는 `figures/`의 실제 실험 결과 그림을 그대로 배치)*

## Slide 9 — Adding physics lifts the scorer
- Built the scorer up from shape overlaps to a full, Glide-like channel set
- **Electrostatics + desolvation gave the biggest single lift** (best so far)
- Honest: still below a *useful* scorer (~0.7–0.8) — the point is the climb

🖼 **`figures/channel_climb.png`**  ← 실험 결과 (채널 추가에 따른 test r 상승)

| step | channels | test Ridge r | test GBR r |
|---|---|---|---|
| grid overlaps | baseline | 0.38 | 0.46 |
| Vina-style pair terms | +contact/H-bond/entropy | 0.43 | 0.49 |
| **+ electrostatics + desolvation** | **best** | **0.46** | **0.52** |

_Note: 예비 결과, held-out PDBbind test. 아직 비나급 아님 — 올라가는 중._

## Slide 10 — H-bond & electrostatics pay off in the POSE, not affinity r
- These channels barely move affinity r — but **affinity is the wrong yardstick** for them
- On **native-pose recognition**, the same channels (physically weighted) succeed strongly
- Lesson: ranking affinity vs recognizing a pose are **two different jobs**

🖼 **`figures/pose_recognition.png`**  ← 실험 결과 (가중치별 native 포즈 인식률)

| weighting of the same channels | top-scored pose < 2 Å (n = 250) |
|---|---|
| affinity-fit weights | 2.4% |
| contact only | 53% |
| **physical pose weights** | **99.6%** |

_Note: 점수-RMSD funnel 테스트. 디코이 = native 강체 교란(국소 funnel, blind 도킹 아님) — 정직 표기._

## Slide 11 — It binds like THIS; we predict THAT
- Crystal (true) pose vs our top-ranked predicted pose, overlaid in the pocket
- Held-out complexes; predicted pose picked by score (native not given)
- They nearly coincide — **RMSD 1.16 Å and 0.95 Å**

🖼 **`figures/pose_example.png`**  ← 실험 결과 (남색=실제 결정구조, 골드=예측)
_Note: 사용자 요청 그림 — "실제로 이렇게 붙어있는데, 예측하니 이런 모양으로 붙더라"._
_Note: 키메라용 PDB도 제공 — `example_structures/{3r04,4u0s}_protein.pdb` + `_ligand_native.pdb`
(resname NAT) + `_ligand_predicted.pdb` (resname PRD). 직접 렌더링하면 더 깔끔한 그림 가능._

## Slide 12 — The scoring funnel (optional)
- Score vs RMSD for three weightings of the **same** channels
- Contact already funnels toward native; physical weights sharpen it; affinity weights do not

🖼 **`figures/pose_funnel.png`**  ← 실험 결과 (점수 vs RMSD 깔때기)
_Note: 낮은 RMSD가 높은 점수여야 함. 시간 없으면 Slide 10·11로 충분._

---

# Discussion

## Slide 13 — Limitations (1 slide)
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

## Slide 14 — Further study (1 slide)
- **Learn pose-appropriate field weights** (not affinity-fit weights) and move from the
  local funnel to **blind FFT docking**
- Run the held-out benchmarks: PoseBusters (RMSD<2 Å) and LIT-PCBA (EF/BEDROC enrichment)
- **Apply to an animal-biotech target:** screen natural products (COCONUT) against TBK1
  (aging / fat accumulation), recovering known binders such as betaine as a positive
  control `[cite]`

🖼 (텍스트 슬라이드)
_Note: 다음 = 포즈용 가중치 학습 → blind 도킹 → 벤치마크 → 동물생명공학 표적(TBK1/천연물) 응용._

---

## figures/ 대응표 (실험 결과 그림 — 그대로 배치)
| 파일 | 슬라이드 | 내용 |
|---|---|---|
| channel_climb.png | 9 | 채널 추가에 따른 held-out test r 상승 |
| pose_recognition.png | 10 | 가중치별 native 포즈 인식률 (2.4 / 53 / 99.6%) |
| pose_example.png | 11 | 실제(결정) vs 예측 포즈 오버레이 (RMSD 1.16 / 0.95 Å) |
| pose_funnel.png | 12 | 점수 vs RMSD 깔때기 (가중치 3종) |

개념 도식(Intro/Objective/M&M)은 `../slides/assets/`의 그림을 쓰거나 새로 제작.
연구 과정 전체 요약은 루트 `RESEARCH_SUMMARY.md`.
