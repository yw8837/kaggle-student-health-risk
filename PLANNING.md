# 프로젝트 기획서 — 학생 건강 위험 조기경보

> Kaggle Playground S6E7 "Predicting Student Health Risk" 팀프로젝트 (발표 7/15)
> PPT 원고 겸용. 데이터 근거는 전부 실측 (EDA.html, oof/*.csv 참조)

---

## 1. 대회 소개 & 데이터 정의서

- **대회**: Kaggle Playground Series S6E7 (2026-07). 학생 생활데이터로 건강상태 3등급 분류
- **평가**: Balanced Accuracy (클래스별 recall 평균) — public LB는 test 20%, private 80%
- **데이터**: train 690,088행 / test 295,753행 / 13개 피처 (수치 7 + 범주 6)

| 구분 | 컬럼 | 값/범위 | 결측 |
|---|---|---|---|
| 타겟 | health_condition | at-risk 86% / unhealthy 8% / fit 6% | - |
| 수치 | sleep_duration | 3~10h | 11% |
| 수치 | heart_rate | 50~108 | 1% |
| 수치 | bmi | 16~35 | 2% |
| 수치 | calorie_expenditure | 1200~3580 | 8% |
| 수치 | step_count | 1k~15k | 2% |
| 수치 | exercise_duration | 0~100분 | 1% |
| 수치 | water_intake | 0.5~4.7L | 6% |
| 범주 | stress_level | low/medium/high | 12% |
| 범주 | physical_activity_level | active/moderate/sedentary | 5% |
| 범주 | sleep_quality | good/average/poor | 8% |
| 범주 | smoking_alcohol | no/occasional/yes | 4% |
| 범주 | diet_type | balanced/veg/non-veg | 1% |
| 범주 | gender | male/female/other | 3% |

## 2. 프로젝트 정의 (SCQA)

- **S (Situation)**: 대학생 생활데이터(수면·활동·설문)는 웨어러블·앱으로 쉽게 쌓이지만, 학생 건강관리는 여전히 "아프고 나서" 대응하는 사후적 체계다.
- **C (Complication)**: 정작 케어가 필요한 위험군은 소수(unhealthy 8%, 위험 경계군 포함해도 14%)라 전수 상담은 비효율적이고, 자칫 소수를 놓치면 건강 악화·중도이탈·의료비용으로 이어진다. 단순 "평균 정확도" 관점으론 다수(86%)만 맞혀도 잘하는 것처럼 보이는 착시까지 있다.
- **Q (Question)**: 설문·웨어러블 수준의 생활데이터만으로, 위험 학생을 놓치지 않고(소수클래스 recall) 미리 선별할 수 있는가?
- **A (Answer)**: 가능하다. 13개 생활지표로 balanced accuracy 0.95의 3등급 분류를 달성했고, 특히 **수면시간·스트레스·신체활동 3요인**이 건강등급을 사실상 결정함을 통계검정·모델중요도·데이터 역공학 3방향에서 일치 확인했다. → 이 3요인 중심의 **학생 건강 조기경보 시스템**을 제안한다.

### 왜 머신러닝인가
- 규칙 기반(if-then)으로도 일부 가능해 보이지만, **결측(최대 12%)과 노이즈**가 있는 실데이터에서는 단순 규칙이 무너짐 → 나머지 10개 상관 피처로 결측을 보완 추론하는 ML이 필요
- 86:8:6 극단 불균형에서 소수클래스 recall을 관리하려면 가중학습·임계값 조정 등 ML 프레임워크가 필수
- 신규 학생 데이터에 대해 자동·즉시 스코어링 (수기 판정 불가능한 규모: 수십만 명)

## 3. 과정 도식화 (프로세스 플로우)

```
[데이터 수집]      [EDA]                [전처리]            [모델링]              [평가/선택]         [서비스화]
 train 690k   →  분포·결측·불균형   →  층화추출          →  11개 모델 비교     →  Balanced Acc     →  조기경보
 test 296k       통계검정(χ²/ANOVA)     결측대치(median/     (동일조건,          5-fold 층화 CV       API/대시보드
                 효과크기 분석          missing 범주)        시간측정)           임계값 최적화        (액션플랜)
                 시각화 7종            원핫인코딩          최종모델 튜닝        SHAP 설명
                                       스케일링            (HistGB/부스팅)      제출·검증
```

## 4. ML 아키텍처 (Pipeline)

```
Pipeline(
  ColumnTransformer(
    수치 7개: SimpleImputer(median, add_indicator) → StandardScaler
    범주 6개: SimpleImputer(constant='missing')    → OneHotEncoder
  )
  → 분류기 (sklearn API: HistGradientBoosting / LGBMClassifier / XGBClassifier / CatBoost)
    · sample_weight='balanced' (불균형 보정)
    · eval_set 모니터링 + early stopping
)
검증: StratifiedKFold(5) → OOF balanced accuracy → (임계값/클래스가중 보정은 OOF에서만 튜닝)
```

- 실측 신뢰도: V1 로컬 CV 0.94962 vs Public LB 0.94952 (**갭 0.0001** → 검증체계 신뢰 가능)

## 5. 모델 비교 (11개, 층화 10만행 샘플, 동일조건)

| 순위 | 모델 | Balanced Acc | 학습시간 |
|---|---|---|---|
| 1 | HistGradientBoosting | 0.95183 | 4.2s |
| 2 | CatBoost | 0.94934 | 38.9s |
| 3 | RandomForest | 0.92500 | 5.2s |
| 4 | LightGBM | 0.92082 | 17.6s |
| 5 | LogisticRegression | 0.91995 | 0.8s |
| 6 | XGBoost | 0.91804 | 33.2s |
| 7 | GaussianNB | 0.87741 | 0.2s |
| 8 | MLP | 0.87026 | 27.2s |
| 9 | DecisionTree | 0.87019 | 1.4s |
| 10 | ExtraTrees | 0.83340 | 6.7s |
| 11 | KNeighbors | 0.67031 | 0.01s |

- 시간↔성능 시각화: `assets/model_zoo_time_vs_score.png`
- 해석 포인트: ① HistGB가 최고점+최단시간 ② 선형모델(LogReg)이 0.92 = 라벨 구조가 단순 규칙에 가깝다는 EDA 발견과 일치 ③ 비교는 동일조건(고정 하이퍼파라미터), 최종 제출모델은 별도 튜닝

## 6. 핵심 발견 (발표 하이라이트)

1. **지표의 반전**: 전부 at-risk로 찍으면 정확도 86%인데 balanced accuracy는 0.333 — "왜 이 지표인가"의 답
2. **3요인 지배**: 수면(η²=0.19)·스트레스(V=0.41)·활동(V=0.24)이 압도적. heart_rate·water_intake는 사실상 무관(효과크기 ≈0)
3. **삼각검증**: 통계검정 = 모델 feature importance = 커뮤니티의 원본데이터 역공학(깊이4 결정트리: 수면<6h+고스트레스→unhealthy / 수면≥7h+저스트레스+active→fit) 세 방법이 동일 결론
4. **CV↔LB 일치(0.0001)**: 검증 방법론의 신뢰성
5. **천장의 물리적 근거 (역공학 실측)**: 라벨 = depth-4 결정규칙(원본 50k에서 결정론 100%) + **합성노이즈 0.82%** + **핵심피처 결측 32%**. 완전관측행은 0.9715 도달, sleep/stress 결측행은 0.90 — 결측은 라벨 결정변수 소실이라 원리상 복원 불가 → 도달가능 상한 ~0.952
6. **최종 성적**: **최종 제출 = Co_10 (Aggressive Five-Fold Consensus), Public LB 0.95104** (동점 Co_5·9) / 정직 OOF 카드 **0.95088 → LB 0.95089** (N9) / 완전 독립 후보 최고 **0.95063** (Co_L3). 외부 제공·공개 컨센서스와 사실상 동일한 Co_16 0.95114는 성과에서 제외

## 7. 액션플랜 — 모델을 실제 개입으로 전환하는 HOW

**"캠퍼스 웰니스 조기경보 서비스" 개념안**

```
[입력]                    [스코어링]                  [출력/개입]
학기초 생활습관 설문   →   모델 API (predict_proba)  →  위험도 대시보드 (보건소/상담센터)
웨어러블 연동(선택)        · 주기적 배치 재평가          · unhealthy 예측 → 상담 우선 배정
학생앱 셀프체크            · 결측 허용 설계              · at-risk 경계군 → 수면/스트레스 프로그램 안내
                                                        · 본인 앱 알림: 3요인 개선 팁
```

- **대상/시점**: 신입생 및 재학생 중 동의자, 개강 1주차 설문 후 매주 월요일 09:00 재평가
- **우선순위 산정**: OOF에서 정한 임계값으로 `unhealthy → at-risk → fit` 순 상담 큐 생성. 단, 모델 라벨은 진단명이 아니라 상담 우선순위로만 사용
- **Red 실행**: 24시간 안에 보건소 간호사가 앱·문자 1차 접촉 → 10분 체크리스트 → 고위험 응답이면 72시간 안에 전문상담 연결
- **Amber 실행**: 14일 수면·스트레스·활동 미션 배정 → 7일차 자동 리마인드 → 14일차 재설문·재스코어링
- **Green 실행**: 월 1회 셀프체크와 예방 콘텐츠만 제공하며 상담 자원은 배정하지 않음
- **담당/도구**: 데이터팀(ETL·모델·모니터링), 보건소(임상 확인·상담), 학생지원팀(프로그램 운영), 개인정보책임자(동의·접근로그). 세부 RACI·KPI·중단기준은 `ACTION_PLAN.md`
- **파일럿**: 500명, 8주. 250명 즉시개입 vs 250명 통상안내 비교. 주 KPI는 Red 72시간 연결률·14일 등급개선율·unhealthy recall, 안전 KPI는 오탐 상담부담·중도이탈·민원
- **중단/재학습**: 그룹별 recall 격차 >10%p, 결측률 >20%, 월간 PSI >0.2, 상담 수용량 110% 초과 중 하나면 자동 확장 중단 후 원인 점검

## 7.5 PPT 레퍼런스 노트 (전 기수 장표에서 차용)

- **지표 슬라이드**: 수식 중앙 배치 + 용어 한 줄 정의 → Balanced Acc 수식 + "86% 정확도=0.33" 데모
- **모델 선정 등식 그래픽**: 원 3개 + = → `(11모델 동일조건 비교)+(시간대비 성능)+(결측 세그먼트 강건성)=HistGB`
- **실험 매트릭스 표**: 전체 실험 통합표(N넘버·모델·인코딩·피처·가중·OOF·LB·시간·판정) — EXPERIMENTS.md에서 생성
- 디자인: 미니멀·볼드 타이틀 (Beige Modern Minimal 계열)

## 8. 한계점 & 향후 계획

**한계**
- 합성데이터: 라벨이 결정규칙 + 합성노이즈 0.82%로 생성(원본 50k 실측) → 점수 천장 ~0.952 존재, 실세계 일반화 검증 아님. 결측 32%는 라벨 결정변수 소실이라 어떤 모델로도 복원 불가(프록시부스트 Δ+0.00000 실증)
- 결측 자체가 합성과정 산물이라 실제 결측 메커니즘(MNAR 등)과 다를 수 있음
- MLP·KNN은 sample_weight 미지원으로 불균형 보정 없이 비교됨(표에 명시)
- 인과가 아닌 상관: "수면 늘리면 fit이 된다"는 인과 주장은 불가 — 개입 설계는 별도 검증 필요

**향후 계획**
- 수치형 target encoding + HistGB 등 표현 개선으로 점수 갱신 (커뮤니티 검증 기법)
- SHAP 기반 개인별 설명("이 학생이 위험인 이유") → 상담 현장 활용성 강화
- 원본 실데이터(College Student Health Behavior Dataset)로 외부 타당성 점검
- 7월 중순 이후 개인 프로젝트로 확장: 조기경보 대시보드 프로토타입
