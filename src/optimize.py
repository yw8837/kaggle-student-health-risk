"""Balanced Accuracy 결정 보정.

OOF 확률에 클래스별 곱셈 가중치 w=(w0,w1,w2)를 적용해
argmax(w * proba) 의 balanced accuracy를 최대화하는 w를 OOF에서만 탐색.
-> test 확률에 동일 w 적용 (누수 없음).

이유: 극단 불균형(86/8/6)에서 단순 argmax는 다수클래스로 쏠려 소수 recall이 죽는다.
balanced accuracy는 per-class recall 평균이라 소수클래스 결정경계를 넓히면 오른다.

실행: python src/optimize.py <oof.npy> <test.npy> [out.csv]
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from config import OOF, SUBS, DATA, ID, TARGET, CLASSES, N_CLASS
from features import load, encode_target


def ba(w, proba, y):
    return balanced_accuracy_score(y, (proba * w).argmax(1))


def optimize_weights(proba, y, restarts=12, seed=0):
    """Nelder-Mead 다중 재시작으로 per-class 가중치 탐색. w0는 1로 고정(스케일 자유도 제거)."""
    rng = np.random.default_rng(seed)
    base = ba(np.ones(N_CLASS), proba, y)
    best_w, best_s = np.ones(N_CLASS), base
    for r in range(restarts):
        x0 = np.ones(N_CLASS - 1) if r == 0 else rng.uniform(0.3, 4.0, N_CLASS - 1)
        def neg(x):
            w = np.concatenate([[1.0], x])
            return -ba(w, proba, y)
        res = minimize(neg, x0, method="Nelder-Mead",
                       options=dict(maxiter=2000, xatol=1e-4, fatol=1e-6))
        w = np.concatenate([[1.0], res.x])
        s = ba(w, proba, y)
        if s > best_s:
            best_s, best_w = s, w
    return best_w, base, best_s


def main():
    oof_path = sys.argv[1] if len(sys.argv) > 1 else OOF / "lgbm_s42_oof.npy"
    test_path = sys.argv[2] if len(sys.argv) > 2 else OOF / "lgbm_s42_test.npy"
    out = sys.argv[3] if len(sys.argv) > 3 else SUBS / "lgbm_s42_opt.csv"

    oof = np.load(oof_path)
    test = np.load(test_path)
    y = np.load(OOF / "y.npy")

    w, base, best = optimize_weights(oof, y)
    print(f"raw argmax OOF balanced_acc = {base:.5f}")
    print(f"opt weights = {np.round(w,4)}")
    print(f"opt OOF balanced_acc        = {best:.5f}  (+{best-base:.5f})")
    print("confusion (rows=true at-risk/unhealthy/fit):")
    print(confusion_matrix(y, (oof * w).argmax(1)))

    inv = {i: c for i, c in enumerate(CLASSES)}
    ids = pd.read_csv(DATA / "test.csv")[ID]
    pred = (test * w).argmax(1)
    sub = pd.DataFrame({ID: ids, TARGET: [inv[i] for i in pred]})
    sub.to_csv(out, index=False)
    print(f"saved {out} | dist {sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
