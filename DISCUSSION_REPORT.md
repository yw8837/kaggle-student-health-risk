# PS S6E7 디스커션 포럼 조사 보고 (2026-07-14)

hot/recent/votes 3개 정렬 전부 확인. WebFetch는 SPA 빈 렌더 → Jina Reader 경유로 스레드 본문 12개 확보.

## 1. 0.951+ 달성 방법 공유 — **없다**

- **Mark Susol "9-notebook research trail" (d/719199, 12표)**: 최고 LB **0.95065** (RealMLP+HGBC-TE 86/14 블렌드). 단 nested CV 개선 ~+0.0000이라 본인이 "not confirmed"라고 명시. 본인이 신뢰하는 모델은 v0.7 HGBC + exact-value TE (OOF 0.9502 / LB 0.95036, nested +0.0009).
  - repo: https://github.com/msusol/kaggle-playground-series-s6e7
- **Masaya Kawamata "Trust your CV" (d/718258, 9표)**: 13개 단일모델 중 최고 = XGB One-vs-Rest CV 0.95036 / LB **0.95040**. RealMLP류(GRN/GANDALF/LNN/ResNet)도 전부 0.949x.
- **Georgy Mamarin "Above ~0.950: skill, or the private draw?" (d/723666, 22표)**: 현재 **top 20이 0.0002 이내**, 보드 절반이 [0.949, 0.9515]. "0.950 위에서 아직 오르는 사람들, public split 피팅 말고 뭘 하는 거냐, 세 번째 레버 있으면 보여달라"고 공개 질문 → **아무도 답 안 함**. 0.9515+ 비법은 포럼에 없고, public 슬라이스 노이즈 피팅 가능성이 컨센서스.
- 기타 점수 공유는 전부 벽 아래: Rugved Bane LGBM+Optuna 0.95014 (d/724455, stress_activity_combo 상호작용·클래스 가중치 — 새 것 없음), naiQie 0.95002 (d/724493), yunsuxiaozi LLM-Agent 파이프라인 LB 0.94985 (d/724558), amirhossein 0.94943 (d/724303).

## 2. 라벨 생성 규칙 / 노이즈 — **기존 인식보다 정밀한 규칙 확정 (중요)**

**broccoli beef "Plausible generation model of the original dataset" (d/717222, 38표)**:
원본(ziya07 college-student-health-behavior) 라벨은 **완전 결정적 depth-4 트리, accuracy 1.0**:

```
sleep<6  & stress=high              → unhealthy
sleep<6  & stress≠high              → at-risk
sleep≥6  & stress≠low               → at-risk
sleep≥6  & low & activity≠active    → at-risk
sleep≥6  & low & active & sleep<7   → at-risk   ← 6≤sleep<7은 fit 아님!
sleep≥7  & low & active             → fit
```

- 기존에 알던 규칙과 대체로 일치하되, **fit 경계가 sleep<7에서 한 번 더 갈라진다**는 게 트리 export로 확정 (sleep 임계 6.0/7.0은 MI-over-thresholds 스캔으로도 재확인, Georgy v7 §10).
- 원본엔 노이즈 없음(결정적). 대회 데이터의 노이즈는 Kaggle 합성 과정에서 유입.
- 같은 제작자의 "enhanced" 데이터셋(대회와 무관)은 depth-6 트리 (mental_health_status, academic_pressure, screen_time 추가).
- **nybbler (d/716812·d/716792 댓글)**: "대회 = 3개 피처의 결측 복원 게임. 단 `stress_level`은 원본에서 자체가 랜덤 피처라 **결측 시 복원 불가능**" → 천장 추정 노트북: https://www.kaggle.com/code/nybbler/s6e7-estimating-the-score-ceiling
- **결측 신호 논쟁**:
  - naiQie (d/724493): pre-imputation 끄고 XGB 네이티브 NaN만으로 0.903→**0.95002** (CV 0.9091→0.9495). "MNAR 결측 패턴이 신호"라 주장.
  - 반증: Georgy — is-missing 플래그 13개만 학습 시 BA **0.333** (prior-correction 후에도 0.339, 신호 0). Kawamata — missing-count/is-missing 피처 효과 정확히 **0.00000**.
  - 정리: "**임퓨테이션이 신호를 파괴한다(하지 마라)**"는 참, "결측 자체가 추가 신호"는 반증됨. 결측률은 컬럼당 1~12%, sleep/stress에서 최고.

## 3. Balanced accuracy 후처리

- **prior-correction(`probs/class_priors`) ≈ class_weight** — 어느 하나만 쓰면 ~0.950. **둘 다 쌓으면 0.9047로 폭락** (이중 보정) — Georgy 4조합 ablation (d/717018 §8).
- **per-class threshold / weighted-argmax 튜닝: 음성 결과** (Susol v0.4, nested −0.0001, 미제출).
- **함정 (Busya PRIME, d/718911)**: native `lgb.train`의 params dict에 `class_weight="balanced"` 넣으면 **조용히 무시됨** (sklearn 래퍼 전용 인자) → "balanced baseline"이 사실 unweighted. 샘플 웨이트 배열로 제대로 주면 0.88→0.95, 표준 balanced 가중이 최적점.
- weighted logloss 서로게이트 검증: 이 데이터에선 불필요 — BA fold 산포 0.13%로 이미 안정, logloss가 오히려 1.5%로 더 출렁 (Georgy 실측, 본인 주장 자기정정).

## 4. Private LB 셰이크업 — **대형 셰이크업 예상이 지배적**

- **d/723666 (Georgy)**: S6E6 전례 — **우승자 = public 343등**(Optimistix, 자체 CV로 선택), private top20 전원이 public 149~495등, **public 1등(Crafty Code)은 private top20 밖**.
- **d/718258 (Kawamata)**: public = test의 20%(~59k). BA 특성상 minority 1개 뒤집힘 = public ±0.0001, minority recall 이항 노이즈 ±0.001~0.002. 13개 모델의 CV↔LB 갭 전부 **±0.0011 이내, 평균 −0.0001** (CV가 LB를 1:1 추적, 체계적 낙관 없음).
- 결론(공유된 규범): **최종 선택은 CV로, public LB는 구경꾼**. "블렌드 LB가 올라도 nested CV가 ~0이면 CV를 믿어라" (Susol).

## 5. exact-value TE 이후 새 피처/표현 기법 — **실질적으로 없다**

- 마지막 진짜 레버는 여전히 Susol의 **exact-value target encoding** (+0.0009, CV↔LB haircut 없음)과 **RealMLP** (주기 임베딩 NN, raw OOF 최고 0.95062이나 nested +0.0001).
- **앙상블 전멸**: 모델 간 예측 일치 97.8~99%, 블렌드 이득 ~+0.0001 (Georgy의 LGBM+주기임베딩NN, Szymon Kłapiński의 XGB+RealMLP 동일 규모). "남는 불일치는 공유 라벨 노이즈, decorrelation 실패".
- Nikunj Katta "14 tests" (d/724596): sleep×stress, bmi_per_exercise, hydration_deficit, hr_adjusted, log(calorie/step) 등 상호작용 피처 다수 제안하지만 본인 LB **0.94795**로 리더 그룹 미달 — 돌파 아님.
- yunsuxiaozi Agent 실험 (d/724558): LB 0.94985, 역시 벽 아래. 댓글 0.

## 한 줄 결론

기존에 아는 것(규칙 역공학, exact-value TE, prior-correction, 공개 최고 ~0.9507, 천장 ~0.952) 외의 **새 돌파구는 포럼에 없다**. 새로 건질 것:
1. depth-4 트리의 정확한 형태 — **6≤sleep<7 + low + active → at-risk** 경계 포함.
2. **stress_level 결측은 원리적으로 복원 불가** → nybbler 천장 추정 노트북 참고 가치.
3. **임퓨테이션 금지**(네이티브 NaN 처리)가 0.95 진입의 전제조건.
4. **이중 보정 함정**: class_weight + prior-correction 동시 사용 시 0.9047 폭락.
5. 셰이크업 대비: **CV 기준 최종 선택**이 유일한 통제 가능 엣지 (S6E6 우승자 = public 343등).
