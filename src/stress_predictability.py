"""스트레스(및 수면) 예측가능성 검증 — 결측 보완의 이론적 여지 측정

stress_level이 다른 12개 피처로 조금이라도 예측되면: 결측 행 개선 여지 있음
완전 랜덤(정확도 = 다수클래스 비율)이면: 결측 세그먼트 천장 확정 -> 결정 보정만이 답

실행: python src/stress_predictability.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from config import DATA

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
ORDINAL = {
    "stress_level":            {"low": 0, "medium": 1, "high": 2},
    "sleep_quality":           {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol":         {"no": 0, "occasional": 1, "yes": 2},
    "diet_type":               {"veg": 0, "balanced": 1, "non-veg": 2},
    "gender":                  {"female": 0, "male": 1, "other": 2},
}


def check(target_col, train, n=150_000):
    """target_col을 나머지 피처로 예측 시도 (타겟 라벨은 피처에서 제외 — 누수 방지)."""
    df = train[train[target_col].notna()].sample(n, random_state=0)
    feats = [c for c in NUM + list(ORDINAL) if c != target_col]
    X = df[feats].copy()
    for c, m in ORDINAL.items():
        if c in X:
            X[c] = X[c].map(m).astype("float64")
    if target_col in ORDINAL:
        y = df[target_col].map(ORDINAL[target_col]).to_numpy()
        chance = np.bincount(y).max() / len(y)          # 다수클래스 찍기 기준
        bal_chance = 1 / len(np.unique(y))              # balanced acc 기준(램덤)
    else:
        raise ValueError

    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                       early_stopping=True, random_state=0)
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    accs, baccs = [], []
    for tr, va in skf.split(X, y):
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[va])
        accs.append((p == y[va]).mean())
        baccs.append(balanced_accuracy_score(y[va], p))
    print(f"{target_col:26s} acc={np.mean(accs):.4f} (chance {chance:.4f}) | "
          f"bal_acc={np.mean(baccs):.4f} (random {bal_chance:.4f}) | "
          f"{'예측가능 신호 있음!' if np.mean(baccs) > bal_chance + 0.01 else '사실상 랜덤 - 보완 불가'}")


def main():
    train = pd.read_csv(DATA / "train.csv")
    # health_condition은 피처에 절대 넣지 않음 (test에 없음 = 누수)
    for col in ["stress_level", "physical_activity_level", "sleep_quality"]:
        check(col, train)
    # 수면시간(수치)은 이진화해서 확인: sleep<6 여부를 다른 피처로 맞출 수 있나
    df = train[train.sleep_duration.notna()].sample(150_000, random_state=0)
    y = (df.sleep_duration < 6).astype(int).to_numpy()
    feats = [c for c in NUM + list(ORDINAL) if c != "sleep_duration"]
    X = df[feats].copy()
    for c, m in ORDINAL.items():
        X[c] = X[c].map(m).astype("float64")
    mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                        early_stopping=True, random_state=0)
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    baccs = []
    for tr, va in skf.split(X, y):
        mm.fit(X.iloc[tr], y[tr])
        baccs.append(balanced_accuracy_score(y[va], mm.predict(X.iloc[va])))
    print(f"{'sleep_duration<6 (이진)':26s} bal_acc={np.mean(baccs):.4f} (random 0.5) | "
          f"{'예측가능 신호 있음!' if np.mean(baccs) > 0.51 else '사실상 랜덤 - 보완 불가'}")


if __name__ == "__main__":
    main()
