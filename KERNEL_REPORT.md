# S6E7 상위 커널 조사 최종 보고 (2026-07-14)

12개 커널 소스+메타를 `scratch_research/` 하위 폴더(georgy, beicicc, n06~n10, ichiro, router, yaminh, masa_stack, masa_ft, anhad)에 다운로드·분석.

## 1. 기법별 표

| 커널 | 무엇을 (모델/피처/기법) | CV·LB 효과 | 우리가 안 써본 것? |
|---|---|---|---|
| **georgymamarin (69표)** | LGBM + prior-correction(p/π). EDA 논문급 | OOF 0.9498 → LB 0.94988 | 아님 (우리 노선의 근거 문서) |
| **beicicc (Kun Zhang 클러스터)** | 모델 학습 0. 공개 제출 7파일 다수결 투표 + **148개 test id 하드코딩 라벨 오버라이드**(LB 프로빙 추정) | LB 0.95117 | 정직한 기법 아님 |
| **assemelqirsh N06–N10** | CatBoost + crosstab per-class TE(fold-fit) + sleep 5분위빈 + stress×sleep 교차. N06 pseudo-label(>0.99 확신, 단 증강데이터 OOF라 부풀려짐), N07 CB+XGB+HGB 균등블렌드 → Nelder-Mead 클래스가중, N08 KMeans-15 클러스터 + 3중교차 + 결측플래그, N10 5-seed 블렌드 vs TabNet vs DAE swap-noise | **출력 미저장 — 실측 점수 전무.** 주장뿐, 증거 없음 | 기법 전부 기지(旣知) 또는 증거 없음 |
| **ichirohippo** | "Stage-2 beta calibration" = OOF에서 최적화한 per-class 곱셈 가중 [0.113, 1.255, 1.506] 하드코딩 | 최고 V4 XGB **LB 0.95000** / CV 0.94945, 최종 V12 CatBoost CV 0.94912 / LB 0.94904 | 아님 (prior-correction 변형, 전부 우리 0.9504 이하) |
| **shamanthakreddymallu 라우터** | stress_level 4분할로 세그먼트별 최적모델 라우팅 (LGBM/CB/XGB/RealMLP/TabM/Lookup 6모델, 도전자는 세그먼트 내 CatBoost +0.002 초과해야 채택) | 출력 미저장. 코드에 "라우터가 전 세그먼트 CatBoost 선택 = 라우팅 무효" KILL 판정 로직 내장 | 시도 불필요 |
| **yaminh** | LGBM(GPU) + FT-Transformer 각자 β 튜닝 후 가중블렌드 그리드서치(0.02 간격) | 출력 미저장, 점수 미확인 | 아님 |
| **masayakawamata 스태커** | 자체 베이스 ~20개(XGB/RepLeaf/LGBM/FM/FFM/FwFM/GANDALF/GRN/LNN/DANet/TabR/ModernNCA/TabTransformer/RealMLP/FT-T, 전부 동일 7-fold rs42) → **cross-fitted vector scaling**(a·log p + b)으로 멤버 캘리브레이션 → **balanced logloss** 기준 nested Caruana greedy vs group-lasso logreg vs 전체 L2 중 승자 채택. β는 cross-fit | 베이스 최고 0.95063 → 스택 **CV 0.95067 (+0.00004)**. 20모델 스택도 사실상 못 올림 | 방법론(vector scaling, balanced logloss 선택기준)은 참고가치. OOF 0.9509 초과 아님 |
| **masayakawamata FT-T v2** | FT-Transformer + **13컬럼 전부(수치 포함) exact-value TE 39피처**(catstat TargetEncoder, fold내 OOF-fit) + plr-lite 수치임베딩 + 결측플래그 + 16에폭 고정(조기종료 금지) + β≈1.05. Ablation: **per-value TE +0.0012(5/5 fold) = 전부**, PLR +0.00003(무효), 클래스가중=β룰 대체재(+0.00001), 조기종료=+0.0003 낙관편향 | raw argmax 0.8919 → **CV 0.95063** (단독모델 공개 최고) | **예 — NN + 수치컬럼 exact-value TE 조합** |
| **anhadmahajan06** | 모델 학습 0. 공개 제출 CSV 자동수집 → 상관제거(>0.985) → LB점수 temperature-softmax 가중으로 pseudo-GT 생성 → 힐클라임/Optuna/LGBM·Ridge·LR 스택을 pseudo-GT에 최적화 → fit 클래스 비율 train 분포 강제정렬 | LB 0.95112 | 정직한 기법 아님 (pseudo-GT 자기참조 최적화) |

## 2. (a) 단독 CV 0.9504 초과 기법

**masayakawamata FT-Transformer v2 — CV 0.95063** 하나뿐 (+동급 RealMLP 0.95062, Mark Susol).
원천은 아키텍처가 아니라 **exact-value TE 39피처(수치 7컬럼 포함 13컬럼 × 3클래스)**:
- 그의 paired ablation에서 +0.0012 (5/5 fold) — v2 개선분의 사실상 전부
- georgy 커널도 Mark Susol 사례(+0.0009 OOF, CV→LB 하락 없음)로 독립 교차확인
- 우리 HistGB 0.9504와의 차이 = "NN + exact-value TE" 조합

## 3. (b) 정직한 블렌드 OOF 0.9509 초과 사례

**없음.** 관측된 정직 사례:
- masa 20-멤버 logreg 스태커: 0.95063 → **0.95067** (+0.00004)
- georgy 2-패밀리(LGBM+주기임베딩 MLP, 99% 일치): +0.0001
- Kłapiński XGB+RealMLP: 0.9505 → 0.9506 (+0.0001)

정직한 OOF 프런티어는 **0.9506–0.9507에서 정지**. 0.9509+ OOF는 공개 기록에 존재하지 않음.

## 4. (c) beta calibration / post-processing 이득 실재 여부

- prior-correction 자체(argmax 대비 **+0.072**)는 실재 — 이미 우리가 쓰는 것
- **그 이상은 허상**:
  - per-class 지수(β≠1) 튜닝: 최적 β≈1, 최대 +0.00015 (georgy 실측)
  - ichiro "stage-2" 가중 최적화: 최고 LB 0.95000 — 우리 0.9504 이하
  - N07 Nelder-Mead 가중: 점수 증거 자체가 없음
- LB 0.9511x (beicicc 0.95117, anhad 0.95112)는 전부 공개파일 투표 + 분포정렬 + id 오버라이드의 **public-split 과적합** — S6E6 전례상 private에서 무너질 가능성 높음

## 5. (d) georgymamarin 69표 커널 핵심 주장

1. **Public LB는 신기루**: 1,587팀 중 294팀이 0.9495–0.9500 한 빈에 밀집. S6E6에서 private top-20 전원이 public 149–495위, public 1위는 top-20 탈락 → 0.9511+ 공개앙상블 = public split 인플레이션.
2. **유일한 레버 = prior-correction**: argmax 0.878 → 0.950 (+0.072). 클래스가중과 동일효과("같은 방의 두 문"). per-class 지수튜닝 최적 β≈1, 최대 +0.00015.
3. **앙상블 무력**: LGBM + 주기임베딩 MLP 2-패밀리, 99% 일치, 블렌드 +0.0001. "노이즈는 평균으로 못 없앤다."
4. **천장 = 데이터**: broccoli beef의 3피처 규칙(sleep/stress/activity)이 원본 50k 데이터에서 BA ~0.99 → 타깃이 규칙+합성노이즈라면 규칙 복원 후 남는 건 학습불가 라벨노이즈. XGB/RepLeaf/RealMLP/LGBM/투표 전부 0.9496–0.9503 한 밴드(스프레드 0.0006).
5. **단 하나의 예외 = 표현(representation)**: Susol의 exact-value TE(cross-fitted)가 OOF +0.0009, LB 하락 없음. nybbler 천장분해도 ~+0.009가 프록시 신호복원에서 옴. "천장은 *주어진 표현에서의* 데이터다."
6. 결측플래그·imputation 효과 0 (결측플래그만으로 학습 시 BA 0.333 = 찬스레벨; n_missing 추가 시 ΔOOF 정확히 0.00000).

## 실행 시사점 (1줄)

유일하게 미시도 + 증거 있는 수 = **exact-value TE 39피처를 NN(FT-Transformer/RealMLP류)에 넣어 0.9506대 단독모델 확보 → 기존 HistGB와 cross-fit vector scaling 스택** (기대 +0.0000~0.0001). 그 이상은 데이터 천장.
