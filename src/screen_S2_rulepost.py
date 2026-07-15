"""S2: 규칙 사후확률 블렌드 스크린

가설: 라벨 = depth-4 규칙(sleep/stress/activity) + 노이즈.
결측행에서 "규칙 라벨의 사후확률"을 보조모델로 직접 추정(y 미사용→누수 0)해서
N5와 확률 블렌드하면, E실험(피처 주입 실패)과 달리 티끌이 나올 수 있다.

- aux: 3피처 모두 관측된 행(rule 라벨 확정)에서 rule_pred를 13피처로 학습
       → 전행 사후확률 (관측행은 사실상 정확, 결측행에서 일반화)
- 평가: KFold(5) 행분할 중첩 — 가중치는 tr에서만 적합 (정직)
기준선: N5 솔로 0.95045 / N7 2-way 0.95084

실행: python src/screen_S2_rulepost.py
"""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, CLASSES, TARGET
from train_v2 import base_encode, add_rule_features
from unified_ensemble import m_weight


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X = add_rule_features(base_encode(train))

    known = X["rule_pred"].notna()
    print(f"rule-known rows: {known.mean()*100:.1f}%")
    feats = [c for c in X.columns if c != "rule_pred"]
    aux = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.1, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.05, n_iter_no_change=20,
        class_weight="balanced", random_state=0)
    aux.fit(X.loc[known, feats], X.loc[known, "rule_pred"].astype(int))
    R = aux.predict_proba(X[feats])          # 규칙 사후확률 (전행)
    s_rule = balanced_accuracy_score(y, R.argmax(1))
    print(f"rule-posterior solo bacc={s_rule:.5f} (vs y, t={time.time()-t0:.0f}s)")

    N5 = np.load(OOF / "n5_oof.npy")
    print(f"N5 solo={balanced_accuracy_score(y, N5.argmax(1)):.5f}")

    P = np.stack([N5, R])
    kf = KFold(5, shuffle=True, random_state=0)
    honest, ws = [], []
    for tr, va in kf.split(y):
        s, w = m_weight(P, y, tr, va, n_draw=1500)
        honest.append(s)
        ws.append(w)
    wm = np.mean(ws, 0)
    print(f"[S2] 2-way honest={np.mean(honest):.5f} (std {np.std(honest):.5f}) "
          f"weights N5={wm[0]:.3f} RULE={wm[1]:.3f} "
          f"| baseline N5 0.95045 / N7 0.95084 | time={time.time()-t0:.0f}s")
    np.save(OOF / "s2_rulepost_oof.npy", R)


if __name__ == "__main__":
    main()
