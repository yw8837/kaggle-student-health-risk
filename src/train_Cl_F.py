"""Cl_F: 5모델 K-fold OOF 앙상블 파이프라인 (13컬럼 전부 + 파생 FE + Optuna)

구성 (지시사항 반영):
  - 13컬럼 전부: 주3(수면·스트레스·활동) + 자잘10(프록시/TE키/파생 재료)
  - 파생: 수면효율, 칼로리/걸음, BMI 범주화 + 검증된 K번들(mres TE + 도메인6종) + 규칙피처
  - exact-value TE (fold 내부 교차적합, 누수 0)
  - 5모델: hgb / lgbm / xgb / cat / nn — 각자 층화 5-fold OOF 저장
  - 최종 재학습(--refit): 채택 하이퍼파라미터로 풀데이터 학습 시 n_estimators x K/(K-1) 보정

실행:
  python src/train_Cl_F.py <model> [seed] [--gpu]     # 모델 하나 5-fold OOF
출력: oof/clf_<model>_s<seed>_{oof,test}.npy
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, SEED, CLASSES, N_CLASS, TARGET
from train_v2 import (NUM, CAT, base_encode, add_rule_features,
                      add_mres_features, add_domain_features)


def add_user_features(X_tr, X_te):
    """지시 파생 3종: 수면효율 / 칼로리 대비 활동량 / BMI 범주화."""
    for X in (X_tr, X_te):
        # sleep_quality 서수(poor0/avg1/good2), 0나눗셈 방지 +1
        X["sleep_efficiency"] = X["sleep_duration"] / (X["sleep_quality"] + 1)
        X["cal_per_step"] = X["calorie_expenditure"] / (X["step_count"] + 1)
        X["bmi_cat"] = pd.cut(X["bmi"], [0, 18.5, 23, 25, 30, 100],
                              labels=False).astype(float)
    return X_tr, X_te


def build_features(train, test):
    """전 파생 블록 적용. 반환: X, X_test, te_cols."""
    X, X_test = base_encode(train), base_encode(test)
    X, X_test = add_rule_features(X), add_rule_features(X_test)
    X, X_test = add_user_features(X, X_test)
    X, X_test = add_mres_features(X, X_test)
    X, X_test = add_domain_features(X, X_test)
    te_cols = list(NUM) + [c for c in X.columns if c.startswith("bin_")]
    return X, X_test, te_cols


# ---------- 모델 팩토리 (5모델) ----------
def make_model(kind, seed, gpu=False, params=None, iter_scale=1.0):
    p = params or {}
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=int(p.get("max_iter", 2000) * iter_scale),
            learning_rate=p.get("learning_rate", 0.05),
            max_leaf_nodes=p.get("max_leaf_nodes", 63),
            early_stopping=iter_scale == 1.0, validation_fraction=0.08,
            n_iter_no_change=50, class_weight="balanced", random_state=seed)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=int(p.get("n_estimators", 1200) * iter_scale),
            learning_rate=p.get("learning_rate", 0.05),
            num_leaves=p.get("num_leaves", 63),
            min_child_samples=p.get("min_child_samples", 40),
            class_weight="balanced", verbose=-1, n_jobs=-1, random_state=seed,
            device="gpu" if gpu else "cpu")
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=int(p.get("n_estimators", 1200) * iter_scale),
            learning_rate=p.get("learning_rate", 0.05),
            max_depth=p.get("max_depth", 8),
            min_child_weight=p.get("min_child_weight", 5),
            tree_method="hist", device="cuda" if gpu else "cpu",
            enable_categorical=False, n_jobs=-1, random_state=seed)
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=int(p.get("iterations", 2000) * iter_scale),
            learning_rate=p.get("learning_rate", 0.05),
            depth=p.get("depth", 8), auto_class_weights="Balanced",
            early_stopping_rounds=100 if iter_scale == 1.0 else None,
            task_type="GPU" if gpu else "CPU", verbose=0, random_seed=seed)
    raise ValueError(kind)   # nn은 train_nn.py 경로 사용 (torch/GPU)


def sample_weight_balanced(y):
    cnt = np.bincount(y)
    w = len(y) / (len(cnt) * cnt)
    return w[y]


def run(kind, seed=SEED, n_splits=5, gpu=False, params=None):
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X, X_test, te_cols = build_features(train, test)
    print(f"features: {X.shape[1]} cols, TE keys: {len(te_cols)}")

    oof = np.zeros((len(X), N_CLASS))
    test_pred = np.zeros((len(X_test), N_CLASS))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va, X_te = X.iloc[tr].copy(), X.iloc[va].copy(), X_test.copy()
        te = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)
        cols = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASS)]
        te_tr = te.fit_transform(X_tr[te_cols].fillna(-999.0), y[tr])
        X_tr = pd.concat([X_tr.reset_index(drop=True),
                          pd.DataFrame(te_tr, columns=cols)], axis=1)
        for Xp, name in ((X_va, "va"), (X_te, "te")):
            tv = te.transform(Xp[te_cols].fillna(-999.0))
            Xp.reset_index(drop=True, inplace=True)
            Xp[cols] = tv
        # 트리 결측 네이티브 처리; xgb만 sample_weight 별도
        m = make_model(kind, seed + fold, gpu=gpu, params=params)
        if kind == "xgb":
            m.fit(X_tr, y[tr], sample_weight=sample_weight_balanced(y[tr]))
        else:
            m.fit(X_tr, y[tr])
        oof[va] = m.predict_proba(X_va)
        test_pred += m.predict_proba(X_te) / n_splits
        print(f"  fold{fold} bal_acc="
              f"{balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f}")

    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[Cl_F/{kind} s{seed}] OOF={cv:.5f} time={time.time()-t0:.0f}s")
    np.save(OOF / f"clf_{kind}_s{seed}_oof.npy", oof)
    np.save(OOF / f"clf_{kind}_s{seed}_test.npy", test_pred)
    return cv


if __name__ == "__main__":
    kind = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else SEED
    run(kind, seed, gpu="--gpu" in sys.argv)
