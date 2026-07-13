"""CV 하네스 - StratifiedKFold + balanced_accuracy OOF 평가.

제출 지표(balanced accuracy)와 동일하게 로컬 점수를 낸다.
모델 학습 함수는 fold별 (X_tr, y_tr, X_va) -> (va_proba, test_proba) 를 반환하는
callable을 넘겨받는 형태로 모델-불가지론적으로 설계.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from config import N_SPLITS, N_CLASS


def run_cv(X, y, X_test, fit_fold, n_splits=N_SPLITS, seed=42, verbose=True):
    """
    fit_fold(X_tr, y_tr, X_va, y_va, X_test, fold) -> (va_proba[n_va, N_CLASS], test_proba[n_test, N_CLASS])
    반환: oof_proba[n, N_CLASS], test_proba[n_test, N_CLASS] (fold 평균), cv_score
    """
    n, n_test = len(X), len(X_test)
    oof = np.zeros((n, N_CLASS))
    test_pred = np.zeros((n_test, N_CLASS))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_va = X.iloc[va] if hasattr(X, "iloc") else X[va]
        y_tr, y_va = y[tr], y[va]
        va_proba, test_proba = fit_fold(X_tr, y_tr, X_va, y_va, X_test, fold)
        oof[va] = va_proba
        test_pred += test_proba / n_splits
        s = balanced_accuracy_score(y[va], va_proba.argmax(1))
        scores.append(s)
        if verbose:
            print(f"  fold{fold} balanced_acc={s:.5f}")

    cv = balanced_accuracy_score(y, oof.argmax(1))
    if verbose:
        print(f"[seed{seed}] OOF balanced_acc={cv:.5f} | fold mean={np.mean(scores):.5f} std={np.std(scores):.5f}")
    return oof, test_pred, cv


def run_multiseed(X, y, X_test, fit_fold, seeds, n_splits=N_SPLITS):
    """여러 시드 평균 -> 분산 축소, 셰이크업 저항."""
    oof_sum = np.zeros((len(X), N_CLASS))
    test_sum = np.zeros((len(X_test), N_CLASS))
    cvs = []
    for sd in seeds:
        oof, test_pred, cv = run_cv(X, y, X_test, fit_fold, n_splits, seed=sd)
        oof_sum += oof / len(seeds)
        test_sum += test_pred / len(seeds)
        cvs.append(cv)
    final_cv = balanced_accuracy_score(y, oof_sum.argmax(1))
    print(f"[multiseed] mean_cv={np.mean(cvs):.5f} | blended OOF balanced_acc={final_cv:.5f}")
    return oof_sum, test_sum, final_cv
