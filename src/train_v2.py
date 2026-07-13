"""V2: HistGradientBoosting + 규칙 피처 + 수치형 exact-value TargetEncoder + 원본데이터 증강

변형별 기여도 검증 (전부 층화 5-fold OOF, sklearn API):
  A: 서수 인코딩 13피처 (베이스)
  B: A + 규칙 피처 (sleep<6, sleep<7, rule_pred, 핵심 결측지시자)
  C: B + 수치형 exact-value TargetEncoder (검증된 +0.0009 기법)
  D: C + 원본 50k 증강 (train fold에만 추가, 검증은 대회 데이터로만)

실행: python src/train_v2.py A|B|C|D [seed]
출력: oof/v2_<variant>_s<seed>_{oof,test}.npy, 로그에 OOF balanced_acc
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, SEED, CLASSES, N_CLASS, TARGET, ID

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
CAT = list(ORDINAL)


def base_encode(df):
    X = df[NUM + CAT].copy()
    for c, m in ORDINAL.items():
        X[c] = X[c].map(m).astype("float64")
    return X


def add_rule_features(X):
    """역공학된 생성규칙 기반 피처. 결측이면 NaN 유지."""
    s, st, act = X["sleep_duration"], X["stress_level"], X["physical_activity_level"]
    X["sleep_lt6"] = np.where(s.isna(), np.nan, (s < 6).astype(float))
    X["sleep_lt7"] = np.where(s.isna(), np.nan, (s < 7).astype(float))
    # rule_pred: 0=at-risk 1=unhealthy 2=fit (CLASSES 순서와 무관한 피처값)
    rp = np.full(len(X), np.nan)
    known = ~(s.isna() | st.isna() | act.isna())
    sv, stv, av = s[known], st[known], act[known]
    r = np.zeros(known.sum())                                # default at-risk
    r[(sv < 6) & (stv == 2)] = 1.0                           # unhealthy
    r[(sv >= 7) & (stv == 0) & (av == 2)] = 2.0              # fit
    rp[known.to_numpy()] = r
    X["rule_pred"] = rp
    X["miss_sleep"] = s.isna().astype(float)
    X["miss_stress"] = st.isna().astype(float)
    X["miss_activity"] = act.isna().astype(float)
    return X


def make_hgb(seed):
    return HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.05, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.08, n_iter_no_change=50,
        class_weight="balanced", random_state=seed)


def run(variant="C", seed=SEED, n_splits=5):
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()

    X = base_encode(train)
    X_test = base_encode(test)
    if variant >= "B":
        X, X_test = add_rule_features(X), add_rule_features(X_test)

    orig = None
    if variant >= "D":
        o = pd.read_csv(DATA / "original" / "student_health_dataset_50k.csv")
        y_o = o[TARGET].map(mapping).to_numpy()
        X_o = base_encode(o)
        if variant >= "B":
            X_o = add_rule_features(X_o)
        orig = (X_o, y_o)
        print(f"augment: original 50k appended to train folds")

    oof = np.zeros((len(X), N_CLASS))
    test_pred = np.zeros((len(X_test), N_CLASS))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr].copy(), X.iloc[va].copy()
        y_tr = y[tr]
        X_te = X_test.copy()

        if orig is not None:
            X_tr = pd.concat([X_tr, orig[0]], ignore_index=True)
            y_tr = np.concatenate([y_tr, orig[1]])

        if variant >= "C":
            # 수치형 exact-value TE: 각 정확한 값 = 하나의 카테고리 (fold 내부 교차적합)
            te = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)
            sent = X_tr[NUM].fillna(-999.0)
            te_tr = te.fit_transform(sent, y_tr)
            te_va = te.transform(X_va[NUM].fillna(-999.0))
            te_te = te.transform(X_te[NUM].fillna(-999.0))
            cols = [f"te_{c}_{k}" for c in NUM for k in range(N_CLASS)]
            X_tr = pd.concat([X_tr.reset_index(drop=True),
                              pd.DataFrame(te_tr, columns=cols)], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True),
                              pd.DataFrame(te_va, columns=cols, index=range(len(X_va)))], axis=1)
            X_te = pd.concat([X_te.reset_index(drop=True),
                              pd.DataFrame(te_te, columns=cols)], axis=1)

        model = make_hgb(seed + fold)
        model.fit(X_tr, y_tr)
        oof[va] = model.predict_proba(X_va)
        test_pred += model.predict_proba(X_te) / n_splits
        s = balanced_accuracy_score(y[va], oof[va].argmax(1))
        scores.append(s)
        print(f"  fold{fold} bal_acc={s:.5f} iters={model.n_iter_}")

    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[V2-{variant} seed{seed}] OOF={cv:.5f} mean={np.mean(scores):.5f} "
          f"std={np.std(scores):.5f} time={time.time()-t0:.0f}s")
    np.save(OOF / f"v2_{variant}_s{seed}_oof.npy", oof)
    np.save(OOF / f"v2_{variant}_s{seed}_test.npy", test_pred)
    return cv


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "C"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    run(variant, seed)
