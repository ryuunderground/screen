# Research summary — Field-based docking for virtual screening

지금까지의 연구 과정을 한 파일에 정리. (세부 버전별 리포트는 서버 `~/screen/notes/`에 보존.)

## 1. 아이디어 / 노블티
- 기존 빠른 스크리닝(대조학습 임베딩)은 분자를 **점** 하나로 뭉개 포즈를 버린다.
- 우리는 분자를 **장(field)**으로 표현하고, 결합을 **장 겹침의 자세 지형(곡선 E(θ))**으로 본다.
  곡선의 꼭대기가 포즈 + 점수. 장은 회전 등변이라 **FFT**로 곡선 전체를 값싸게 그린다.
- 선행 ESF(arXiv 2312.04323)는 스칼라장 + SE(3) FFT 상관이되 **강체만**(내부 torsion은 future work).
  우리 기여 = 그 곡선의 축에 **내부 torsion**(단편 분해 + 운동학 트리)을 추가.
- 점수함수 = **채널별 장 겹침의 가중합**(Vina 원자쌍 항과 수학적 동형). 학습은 **가중치만**,
  라벨은 **실험 친화도**(다른 도구 점수 아님).

## 2. 데이터 & 평가
- PDBbind v2020 general: 색인 19,037 → 사용 가능 **18,137** (깨끗한 `=` 친화도 + LP-PDBBind 분할).
- 분할(LP-PDBBind, 누출방지): **test 4,753 동결**, 나머지 **13,384**(train+val 병합)을 CV에 사용.
- **클러스터 인지 5-fold GroupKFold**(서열 클러스터로 그룹핑 → 어느 fold도 같은 단백질 공유 안 함) + 동결 held-out test.
- 지표 = **예측 pKd vs 실험 pKd**: Pearson r, Spearman ρ, RMSE. 포즈는 별도(아래 §4).
- 별도 벤치마크 준비됨(미실행): PoseBusters(포즈), LIT-PCBA(농축), COCONUT(스크리닝 라이브러리).
- 인프라: 서버 conda env `fieldscreen`, 224 CPU 코어 샤딩, GPU 노드 유휴.

## 3. 점수함수(친화도) 실험 — 채널을 물리적으로 강화한 궤적
각 버전은 **같은 과제(결정포즈에서 친화도 예측)**의 채널 개선판. held-out test:

| 버전 | 채널 | test Ridge r | test GBR r | 메모 |
|---|---|---|---|---|
| v1 | 격자 겹침 5채널 | 0.40 | 0.48 | 5채널이 묻힌부피축 1개로 공선 |
| v3 | Vina식 쌍 상호작용 6채널 | 0.43 | 0.50 | 공선 깨짐(VIF↓) |
| v4 | + 기하 H-결합 | 0.43 | 0.50 | H-결합 개별 상관 0.02→0.085, 전체 r은 불변 |
| **v5** | **+ 정전기 + 탈용매** | **0.47** | **0.54** | **최고. 단일 최대 상승** |
| v6 | + 전하 품질(MMFF+형식전하) | 0.43 | 0.51 | 정전기 부호는 물리적으로 개선됐으나 r은 하락(정직한 음성) |

**핵심 교훈**
- 정전기 + 탈용매가 가장 큰 기여(v5). elec은 단독 상관 ≈0인데 큰 가중치를 받는 **억제변수**로 작동.
- **H-결합 개수·전하 품질을 물리적으로 바로잡아도 친화도 r은 안 오른다.** 수소결합은 물과의
  결합을 대체해 순 엔탈피가 상쇄되는 등, 친화도 *크기*의 선형 예측자가 아님(분야 상식을 데이터가 재현).
- 결정포즈 + 선형 채널 틀에서 **r ~ 0.5가 천장**. 단일 누락항 문제가 아니라 틀 자체의 한계.

## 4. 포즈 실험 — 물리 채널이 값어치를 내는 곳
친화도 r이 못 본 H-결합·정전기의 값어치를 **포즈**에서 확인(scoring funnel, n=250 held-out).
native + 강체 교란 디코이를 v5 채널로 점수 → 점수가 저 RMSD로 깔때기지는지.

| 같은 8채널, 가중치만 다름 | top-scored pose < 2 Å | median Spearman(score, RMSD) |
|---|---|---|
| 친화도 적합 가중치 | 2.4% | −0.13 |
| 접촉만 | 52.8% | −0.65 |
| **물리 포즈 가중치** | **99.6%** | −0.69 |

- **친화도 회귀 가중치는 포즈엔 독**(2.4%). 포즈엔 물리적 부호 가중치가 맞음(99.6%).
- 예시: 실제(결정) vs 예측 포즈 오버레이 RMSD 1.16 / 0.95 Å (`out/pose_example.png`).
- **정직한 한계**: 디코이가 native 강체 교란이라 native가 항상 포함된 **국소 funnel / cognate
  redocking**, blind 글로벌 도킹 아님.

## 5. 현재 상태 (정직하게)
- **작동**: 데이터(18k), 채널, FFT 엔진, torsion, 클러스터 CV 평가 인프라 전부 돌아감.
- **약점**: 친화도 점수함수 ~r 0.5(쓸만한 수준 0.7–0.8에 못 미침, 비나급 아님).
- **입력 거칠다**: Gasteiger-on-PDB / 이름기반 원자 타이핑. 전하 개선은 친화도엔 도움 안 됨.
- **미실행 벤치마크**: 외부 친화도 테스트, LIT-PCBA 농축, PoseBusters 포즈.

## 6. 다음 단계
1. **포즈용 가중치 학습**(친화도 가중치 말고) → 국소 funnel에서 **blind FFT 도킹**으로.
2. 벤치마크 실행: PoseBusters(RMSD<2Å), LIT-PCBA(EF/BEDROC).
3. **학습된 등변 장**(선행 approach ②): 이제 넘어야 할 목표(v5 r~0.46/0.52)와 어떤 물리를 담아야
   하는지 근거가 생김.
4. 응용(동물생명공학): COCONUT 천연물을 TBK1(노화·지방축적) 표적에 스크리닝, betaine 양성대조 `[cite]`.

## 7. 산출물 위치
- 코드(서버+로컬 `src/`): `v1_features.py v1_labels.py v1_train.py`, `v2_dock.py`,
  `v4_features.py v5_features.py v6_features.py`, `v7_pose.py v7_fig.py`, `make_example_figs.py`.
- 결과(로컬 `out/`): `v3/ v4/ v5/ v6/ v7/`, `channel_climb.png`, `pose_example.png`.
- 발표자료: `presentation/presentation.md` + `presentation/figures/`.
- 서버 상세 리포트(보존): `~/screen/notes/report_v1_v2.md … report_v7.md`, `versions_overview.md`,
  v1/v2 동결 스냅샷 `~/screen/snapshots/v1v2_2026-11-02.tar.gz`.
