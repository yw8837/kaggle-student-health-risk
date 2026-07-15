"""S1: 라벨노이즈 필터링 스크린 (cleanlab 원리, N5 OOF 기반)

가설: 라벨 = 결정규칙 + 합성노이즈. OOF 확률로 "자기 라벨에 대한 신뢰도"가
바닥인 행 = 노이즈 후보. train fold에서만 제거(검증은 전행 유지 = 정직).

방법: n5_oof(10f7s 평균, 가장 안정)에서 p_own = P(자기라벨).
      클래스별 하위 q% & argmax != label 행을 드랍 후보로.
드랍률 q: CLI 인자 (기본 0.02)

실행: python src/screen_S1_noisedrop.py [q] [seed]
기준선: v2_K s42 OOF=0.95039 / 게이트 +0.0005
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


def run(q=0.02, seed=SEED):
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # 노이즈 후보 랭킹 (N5 OOF)
    P = np.load(OOF / "n5_oof.npy")
    p_own = P[np.arange(len(y)), y]
    wrong = P.argmax(1) != y
    drop_mask = np.zeros(len(y), bool)
    for c in range(N_CLASS):
        idx = np.where((y == c) & wrong)[0]
        k = int(len(np.where(y == c)[0]) * q)
        if k and len(idx):
            worst = idx[np.argsort(p_own[idx])[:k]]
            drop_mask[worst] = True
    print(f"drop candidates: {drop_mask.sum()} ({drop_mask.mean()*100:.2f}%) "
          f"per-class {[int(drop_mask[y==c].sum()) for c in range(3)]}")

    # K피처 (user3 제외 — S4 근거로 K 순정)
    X, X_test = base_encode(train), base_encode(test)
    X, X_test = add_rule_features(X), add_rule_features(X_test)
    X, X_test = add_mres_features(X, X_test)
    X, X_test = add_domain_features(X, X_test)
    te_cols = list(NUM) + [c for c in X.columns if c.startswith("bin_")]

    oof = np.zeros((len(X), N_CLASS))
    tp = np.zeros((len(X_test), N_CLASS))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        keep = tr[~drop_mask[tr]]                 # train fold에서만 제거
        X_tr, X_va, X_te = X.iloc[keep].copy(), X.iloc[va].copy(), X_test.copy()
        te = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)
        cols = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASS)]
        t_tr = te.fit_transform(X_tr[te_cols].fillna(-999.0), y[keep])
        X_tr = pd.concat([X_tr.reset_index(drop=True),
                          pd.DataFrame(t_tr, columns=cols)], axis=1)
        for Xp in (X_va, X_te):
            tv = te.transform(Xp[te_cols].fillna(-999.0))
            Xp.reset_index(drop=True, inplace=True)
            Xp[cols] = tv
        m = HistGradientBoostingClassifier(
            max_iter=2000, learning_rate=0.05, max_leaf_nodes=63,
            early_stopping=True, validation_fraction=0.08, n_iter_no_change=50,
            class_weight="balanced", random_state=seed + fold)
        m.fit(X_tr, y[keep])
        oof[va] = m.predict_proba(X_va)
        tp += m.predict_proba(X_te) / 5
        print(f"  fold{fold} {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f}",
              flush=True)
    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[S1 q={q} s{seed}] OOF={cv:.5f} (baseline K 0.95039, gate +0.0005) "
          f"time={time.time()-t0:.0f}s")
    np.save(OOF / f"s1_q{int(q*1000)}_s{seed}_oof.npy", oof)
    np.save(OOF / f"s1_q{int(q*1000)}_s{seed}_test.npy", tp)
    return cv


if __name__ == "__main__":
    q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    run(q, seed)
