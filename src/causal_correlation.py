"""13컬럼 인과 vs 상관 패턴 분석 (합성 라벨의 구조 규명)

가설: 라벨 = 3개 규칙피처(sleep_duration, stress_level, physical_activity)의 결정함수.
 -> 이 3개는 '인과'(라벨을 직접 결정), 나머지 10개는 '상관'(3개와의 상관 통한 간접 신호).
증명:
 A) 3 규칙피처만으로 학습한 트리 vs +10피처 -> 증분이 0에 수렴하면 10개는 인과 무관
 B) 조건부 가치: 3피처가 '관측됐을 때' vs '결측일 때' 10개 프록시의 라벨 예측 증분
    -> 결측 세그먼트에서만 프록시가 값을 갖는다면 = 결측대체용 프록시 (개선 여지 위치)

실행: python src/causal_correlation.py
출력: oof/causal_analysis.csv, assets/causal_correlation.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from config import DATA, OOF, CLASSES, TARGET

ASSETS = OOF.parent / "assets"
CAUSAL = ["sleep_duration", "stress_level", "physical_activity_level"]
PROXY = ["heart_rate", "bmi", "calorie_expenditure", "step_count", "exercise_duration",
         "water_intake", "diet_type", "sleep_quality", "smoking_alcohol", "gender"]
ORD = {"stress_level": {"low": 0, "medium": 1, "high": 2},
       "sleep_quality": {"poor": 0, "average": 1, "good": 2},
       "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
       "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
       "diet_type": {"veg": 0, "balanced": 1, "non-veg": 2},
       "gender": {"female": 0, "male": 1, "other": 2}}


def enc(df, cols):
    X = df[cols].copy()
    for c in cols:
        if c in ORD:
            X[c] = X[c].map(ORD[c]).astype(float)
    return X


def cv_ba(X, y, n=200_000, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    keep = [c for c in X.columns if X[c].nunique(dropna=True) >= 2]   # 전결측/상수 드랍
    X = X[keep]
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=int)
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                           class_weight="balanced",
                                           early_stopping=True, random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        pred[va] = m.predict(X.iloc[va])
    return balanced_accuracy_score(y, pred)


def main():
    tr = pd.read_csv(DATA / "train.csv")
    y = tr[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # A) 인과 3개 vs 전체 13개 vs 프록시 10개만
    ba_causal = cv_ba(enc(tr, CAUSAL), y)
    ba_all = cv_ba(enc(tr, CAUSAL + PROXY), y)
    ba_proxy = cv_ba(enc(tr, PROXY), y)
    print("=== A) 인과 vs 상관 (balanced accuracy) ===")
    print(f"  규칙 3피처만          : {ba_causal:.5f}")
    print(f"  전체 13피처           : {ba_all:.5f}  (증분 {ba_all-ba_causal:+.5f})")
    print(f"  프록시 10피처만       : {ba_proxy:.5f}  (규칙 없이도 이만큼 = 프록시가 규칙을 복원)")

    # B) 조건부 가치: 결측 세그먼트에서만 재보는 3피처 vs 3+프록시
    rows = []
    for seg, mask in [("전체", np.ones(len(tr), bool)),
                      ("스트레스결측", tr.stress_level.isna().to_numpy()),
                      ("수면결측", tr.sleep_duration.isna().to_numpy())]:
        sub = tr[mask]
        ys = y[mask]
        if mask.sum() < 5000:
            continue
        b3 = cv_ba(enc(sub, CAUSAL), ys, n=mask.sum())
        b13 = cv_ba(enc(sub, CAUSAL + PROXY), ys, n=mask.sum())
        rows.append(dict(segment=seg, n=int(mask.sum()),
                         causal3=round(b3, 4), all13=round(b13, 4),
                         proxy_gain=round(b13 - b3, 4)))
        print(f"  [{seg}] 규칙3={b3:.4f} 전체13={b13:.4f} 프록시증분={b13-b3:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OOF / "causal_analysis.csv", index=False)

    # 시각화
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(["rule 3\n(causal)", "proxy 10\n(correlational)", "all 13"],
           [ba_causal, ba_proxy, ba_all], color=["#ef4444", "#94a3b8", "#0ea5e9"])
    for i, v in enumerate([ba_causal, ba_proxy, ba_all]):
        a1.text(i, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")
    a1.set_ylim(0.9, 0.96); a1.set_ylabel("balanced accuracy")
    a1.set_title("A) Causal 3 features already near ceiling\nproxies only recover the same rule")
    if len(df):
        x = np.arange(len(df)); w = 0.35
        a2.bar(x - w/2, df.causal3, w, label="rule 3 only", color="#ef4444")
        a2.bar(x + w/2, df.all13, w, label="all 13", color="#0ea5e9")
        a2.set_xticks(x); a2.set_xticklabels(df.segment, fontsize=9)
        a2.set_ylim(0.85, 0.98); a2.legend()
        a2.set_title("B) Proxies add value ONLY where causal feature is missing")
    fig.tight_layout(); fig.savefig(ASSETS / "causal_correlation.png", dpi=120)
    print("saved assets/causal_correlation.png")


if __name__ == "__main__":
    main()
