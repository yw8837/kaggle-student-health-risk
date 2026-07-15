"""S3: 라벨노이즈 교정(플립) 스크린 — S1(드랍)의 대조 변형

방법: N9급 블렌드 OOF(0.15 N5 + 0.57 RMLP + 0.28 K)가
      "자기 라벨 확률 < tau_own & 최대확률 > tau_max"로 강하게 반박하는 행을
      블렌드 argmax로 재라벨 → train fold에서만 적용 → HGB-K 재학습.
검증 라벨은 원본 유지 (정직).

실행: python src/screen_S3_noiseflip.py [tau_own] [tau_max] [seed]
기준선: v2_K s42 0.95039 / 게이트 +0.0005
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, SEED, CLASSES, N_CLASS, TARGET
from train_v2 import (NUM, base_encode, add_rule_features,
                      add_mres_features, add_domain_features)
from blend_library import load_library


def run(tau_own=0.15, tau_max=0.75, seed=SEED):
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    B = (0.15 * np.load(OOF / "n5_oof.npy")
         + 0.57 * load_library()[1]["RMLP"]
         + 0.28 * np.load(OOF / "v2_K_s42_oof.npy"))
    p_own = B[np.arange(len(y)), y]
    flip_to = B.argmax(1)
    flip_mask = (p_own < tau_own) & (B.max(1) > tau_max) & (flip_to != y)
    y_fix = y.copy()
    y_fix[flip_mask] = flip_to[flip_mask]
    print(f"flip: {flip_mask.sum()} rows ({flip_mask.mean()*100:.2f}%) "
          f"per-class-from {[int(flip_mask[y==c].sum()) for c in range(3)]}")

    X, X_test = base_encode(train), base_encode(test)
    X, X_test = add_rule_features(X), add_rule_features(X_test)
    X, X_test = add_mres_features(X, X_test)
    X, X_test = add_domain_features(X, X_test)
    te_cols = list(NUM) + [c for c in X.columns if c.startswith("bin_")]

    oof = np.zeros((len(X), N_CLASS))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        te = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)
        cols = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASS)]
        t_tr = te.fit_transform(X_tr[te_cols].fillna(-999.0), y_fix[tr])
        X_tr = pd.concat([X_tr.reset_index(drop=True),
                          pd.DataFrame(t_tr, columns=cols)], axis=1)
        tv = te.transform(X_va[te_cols].fillna(-999.0))
        X_va.reset_index(drop=True, inplace=True)
        X_va[cols] = tv
        m = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.05, max_leaf_nodes=63,
            early_stopping=True, validation_fraction=0.08, n_iter_no_change=50,
            class_weight="balanced", random_state=seed + fold)
        m.fit(X_tr, y_fix[tr])
        oof[va] = m.predict_proba(X_va)
        print(f"  fold{fold} {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f}",
              flush=True)
    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[S3 own<{tau_own} max>{tau_max} s{seed}] OOF={cv:.5f} "
          f"(baseline K 0.95039) time={time.time()-t0:.0f}s")
    np.save(OOF / f"s3_flip_s{seed}_oof.npy", oof)
    return cv


if __name__ == "__main__":
    a = sys.argv[1:]
    run(float(a[0]) if a else 0.15, float(a[1]) if len(a) > 1 else 0.75,
        int(a[2]) if len(a) > 2 else SEED)
