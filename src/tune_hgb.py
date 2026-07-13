"""HGB Optuna 튜닝 (V2-C 피처 고정, 파라미터만 서치)

2단 구조:
  1) 층화 25만 서브샘플 · 3-fold로 40 trial 서치 (트라이얼당 ~1분)
  2) 상위 파라미터는 별도 스크립트로 풀데이터 5-fold 재검증 후에만 채택
실험조건 동일(유의점 7): 피처·폴드시드·가중 고정, 파라미터만 변수.

실행: python src/tune_hgb.py [n_trials]
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, SEED, CLASSES, N_CLASS, TARGET
from train_v2 import base_encode, add_rule_features, NUM

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare(n=250_000):
    train = pd.read_csv(DATA / "train.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y_all = train[TARGET].map(mapping).to_numpy()
    X_all = add_rule_features(base_encode(train))
    X, _, y, _ = train_test_split(X_all, y_all, train_size=n,
                                  stratify=y_all, random_state=SEED)
    return X.reset_index(drop=True), y


def cv_score(params, X, y):
    skf = StratifiedKFold(3, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in skf.split(X, y):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        te = TargetEncoder(target_type="multiclass", cv=5, random_state=SEED)
        cols = [f"te_{c}_{k}" for c in NUM for k in range(N_CLASS)]
        A = pd.DataFrame(te.fit_transform(X_tr[NUM].fillna(-999.0), y[tr]), columns=cols)
        B = pd.DataFrame(te.transform(X_va[NUM].fillna(-999.0)), columns=cols)
        X_tr = pd.concat([X_tr.reset_index(drop=True), A], axis=1)
        X_va = pd.concat([X_va.reset_index(drop=True), B], axis=1)
        m = HistGradientBoostingClassifier(
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.08, n_iter_no_change=40,
            max_iter=1500, random_state=SEED, **params)
        m.fit(X_tr, y[tr])
        scores.append(balanced_accuracy_score(y[va], m.predict(X_va)))
    return float(np.mean(scores))


def main(n_trials=40):
    X, y = prepare()
    base = cv_score(dict(learning_rate=0.05, max_leaf_nodes=63), X, y)
    print(f"baseline(현행 파라미터) subsample-CV = {base:.5f}")

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True),
            max_bins=trial.suggest_int("max_bins", 63, 255),
            max_features=trial.suggest_float("max_features", 0.5, 1.0),
        )
        return cv_score(params, X, y)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nbest subsample-CV = {study.best_value:.5f} (baseline {base:.5f}, "
          f"delta {study.best_value-base:+.5f})")
    print("best params:", study.best_params)
    top = sorted(study.trials, key=lambda t: -(t.value or 0))[:5]
    for i, t in enumerate(top):
        print(f"top{i+1}: {t.value:.5f} {t.params}")
    pd.DataFrame([dict(value=t.value, **t.params) for t in study.trials]) \
        .to_csv(OOF / "optuna_hgb_trials.csv", index=False)
    print("saved oof/optuna_hgb_trials.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
