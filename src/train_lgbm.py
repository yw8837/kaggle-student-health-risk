"""LightGBM 베이스라인. multiclass + balanced accuracy 대응(class weight).

실행: python src/train_lgbm.py
출력: oof/lgbm_oof.npy, oof/lgbm_test.npy, subs/lgbm.csv, CV 점수
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

from config import OOF, SUBS, SEED, SEEDS, N_CLASS, CLASSES, ID, TARGET, DATA
from features import load, prep
from cv import run_multiseed

PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASS,
    metric="multi_logloss",
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    min_child_samples=40,
    lambda_l1=1.0,
    lambda_l2=1.0,
    verbose=-1,
    n_jobs=-1,
)
N_EST = 3000


def make_fit(cats, class_weight):
    def fit_fold(X_tr, y_tr, X_va, y_va, X_test, fold):
        w = np.array([class_weight[c] for c in y_tr])
        wv = np.array([class_weight[c] for c in y_va])
        dtr = lgb.Dataset(X_tr, y_tr, weight=w, categorical_feature=cats)
        dva = lgb.Dataset(X_va, y_va, weight=wv, reference=dtr, categorical_feature=cats)
        model = lgb.train(
            PARAMS, dtr, num_boost_round=N_EST,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        return model.predict(X_va), model.predict(X_test)
    return fit_fold


def main():
    train, test = load()
    X, X_test, y, cats, mapping = prep(train, test)
    print(f"train={X.shape} test={X_test.shape} cats={len(cats)} classes={np.bincount(y)}")

    cw = compute_class_weight("balanced", classes=np.arange(N_CLASS), y=y)
    class_weight = {i: cw[i] for i in range(N_CLASS)}
    print("class_weight:", class_weight)

    oof, test_pred, cv = run_multiseed(X, y, X_test, make_fit(cats, class_weight), SEEDS)

    np.save(OOF / "lgbm_oof.npy", oof)
    np.save(OOF / "lgbm_test.npy", test_pred)

    # 제출 파일 생성
    inv = {v: k for k, v in mapping.items()}
    test_ids = pd.read_csv(DATA / "test.csv")[ID]
    sub = pd.DataFrame({ID: test_ids, TARGET: [inv[i] for i in test_pred.argmax(1)]})
    out = SUBS / "lgbm.csv"
    sub.to_csv(out, index=False)
    print(f"saved {out} | CV balanced_acc={cv:.5f}")

    # sanity: sample_submission 대비 검증
    ss = pd.read_csv(DATA / "sample_submission.csv")
    assert len(sub) == len(ss), f"행수 불일치 {len(sub)} vs {len(ss)}"
    assert list(sub.columns) == list(ss.columns), f"컬럼 불일치 {list(sub.columns)}"
    assert set(sub[TARGET]) <= set(CLASSES), f"라벨 이상 {set(sub[TARGET])}"
    print("submission sanity OK")


if __name__ == "__main__":
    main()
