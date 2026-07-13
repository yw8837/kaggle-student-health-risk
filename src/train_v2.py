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


VARIANT_FEATURES = {
    "A": set(), "B": {"rule"}, "C": {"rule", "te"},
    "D": {"rule", "te", "orig"}, "E": {"rule", "te", "soft"},
    "F": {"rule", "te", "bundle"},   # 티끌 번들: 2D 상호작용 TE + 빈도인코딩 + 무신호 제거
    "G": {"rule", "te", "bundle"},   # F와 동일 피처, 조기중단만 기존 loss (절제실험)
}

DEAD_FEATURES = ["heart_rate", "water_intake"]   # 효과크기 0.000x — 번들에서 제거

def add_bundle_features(X_tr_all, X_te_all):
    """티끌 번들 (라벨 미사용 부분): 빈도 인코딩 + 2D 상호작용 키 생성.

    - count encoding: train+test 합산 값 빈도 (합성데이터의 '원본스러움' 신호)
    - interaction key: (수면 0.5h 구간 x 스트레스), (수면구간 x 활동) -> fold 내 TE 대상
    """
    both = pd.concat([X_tr_all, X_te_all], ignore_index=True)
    n_tr = len(X_tr_all)
    for c in ["sleep_duration", "step_count", "bmi", "calorie_expenditure"]:
        cnt = both[c].map(both[c].value_counts())
        X_tr_all[f"cnt_{c}"] = cnt.iloc[:n_tr].values
        X_te_all[f"cnt_{c}"] = cnt.iloc[n_tr:].values
    # 상호작용 키 (숫자 코드): NaN은 -1 코드
    sb = (both["sleep_duration"] * 2).round()          # 0.5h 구간
    st = both["stress_level"].fillna(-1)
    ac = both["physical_activity_level"].fillna(-1)
    k1 = sb.fillna(-1) * 10 + st                        # sleep_bin x stress
    k2 = sb.fillna(-1) * 10 + ac                        # sleep_bin x activity
    X_tr_all["ix_sleep_stress"] = k1.iloc[:n_tr].values
    X_te_all["ix_sleep_stress"] = k1.iloc[n_tr:].values
    X_tr_all["ix_sleep_act"] = k2.iloc[:n_tr].values
    X_te_all["ix_sleep_act"] = k2.iloc[n_tr:].values
    # cat x cat: 스트레스 x 수면질 (수면질=수면 프록시, 트리가 자동으로 못 찾는 결합)
    sq = both["sleep_quality"].fillna(-1)
    k3 = st * 10 + sq
    X_tr_all["ix_stress_quality"] = k3.iloc[:n_tr].values
    X_te_all["ix_stress_quality"] = k3.iloc[n_tr:].values
    # 비율 피처: 걸음당 칼로리 (트리가 스스로 못 만드는 나눗셈)
    cps = both["calorie_expenditure"] / (both["step_count"] + 1)
    X_tr_all["cal_per_step"] = cps.iloc[:n_tr].values
    X_te_all["cal_per_step"] = cps.iloc[n_tr:].values
    # 무신호 피처 제거
    X_tr_all = X_tr_all.drop(columns=DEAD_FEATURES)
    X_te_all = X_te_all.drop(columns=DEAD_FEATURES)
    return X_tr_all, X_te_all


def add_soft_impute(X_tr_all, X_te_all):
    """소프트 임퓨테이션: 결측 피처의 확률을 보조모델로 추정 (라벨 y 미사용 -> 누수 없음).

    - stress/activity: 3클래스 확률, sleep: P(<6h), P(>=7h)
    - 관측된 행은 실제값의 원핫/지시값으로 대체 (확률=확실성 1)
    - 소프트 규칙 사후확률: P(unh)=P(<6)*P(high), P(fit)=P(>=7)*P(low)*P(active)
    """
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    both = pd.concat([X_tr_all, X_te_all], ignore_index=True)
    n_tr = len(X_tr_all)

    def aux_proba(target, classes_n):
        feats = [c for c in NUM + CAT if c != target]
        obs_tr = X_tr_all[target].notna()
        m = HGB(max_iter=200, learning_rate=0.1, early_stopping=True, random_state=0)
        m.fit(X_tr_all.loc[obs_tr, feats], X_tr_all.loc[obs_tr, target].astype(int))
        P = m.predict_proba(both[feats])
        # 관측 행은 실제값 원핫으로 덮어씀
        obs = both[target].notna().to_numpy()
        vals = both[target].to_numpy()
        for k in range(classes_n):
            P[obs, k] = (vals[obs] == k).astype(float)
        return P

    P_st = aux_proba("stress_level", 3)          # [low, med, high]
    P_ac = aux_proba("physical_activity_level", 3)  # [sed, mod, act]

    def aux_binary(flag_col):
        feats = [c for c in NUM + CAT if c != "sleep_duration"]
        obs_tr = X_tr_all["sleep_duration"].notna()
        m = HGB(max_iter=200, learning_rate=0.1, early_stopping=True, random_state=0)
        m.fit(X_tr_all.loc[obs_tr, feats], X_tr_all.loc[obs_tr, flag_col].astype(int))
        p = m.predict_proba(both[feats])[:, 1]
        obs = both["sleep_duration"].notna().to_numpy()
        p[obs] = both.loc[obs, flag_col].to_numpy(dtype=float)
        return p

    both["sleep_ge7"] = np.where(both.sleep_duration.isna(), np.nan,
                                 (both.sleep_duration >= 7).astype(float))
    X_tr_all["sleep_ge7"] = both["sleep_ge7"].iloc[:n_tr].values
    p_lt6 = aux_binary("sleep_lt6")
    p_ge7 = aux_binary("sleep_ge7")

    out = pd.DataFrame({
        "p_stress_low": P_st[:, 0], "p_stress_high": P_st[:, 2],
        "p_act_active": P_ac[:, 2], "p_act_sed": P_ac[:, 0],
        "p_sleep_lt6": p_lt6, "p_sleep_ge7": p_ge7,
        "p_rule_unh": p_lt6 * P_st[:, 2],
        "p_rule_fit": p_ge7 * P_st[:, 0] * P_ac[:, 2],
    })
    A = pd.concat([X_tr_all.drop(columns=["sleep_ge7"]).reset_index(drop=True),
                   out.iloc[:n_tr].reset_index(drop=True)], axis=1)
    B = pd.concat([X_te_all.reset_index(drop=True),
                   out.iloc[n_tr:].reset_index(drop=True)], axis=1)
    return A, B


def make_hgb(seed, metric_aligned=False):
    """metric_aligned=True: 조기중단 기준을 대회지표(balanced accuracy)로 정렬
    (커뮤니티 실증: logloss 기준 조기중단은 지표와 어긋나 악화 사례 있음)"""
    return HistGradientBoostingClassifier(
        max_iter=2000, learning_rate=0.05, max_leaf_nodes=63,
        early_stopping=True, validation_fraction=0.08, n_iter_no_change=50,
        scoring="balanced_accuracy" if metric_aligned else "loss",
        class_weight="balanced", random_state=seed)


def make_model(kind, seed, metric_aligned=False):
    """보팅 다양성용 모델 팩토리 (전부 sklearn API, 동일 폴드에서 비교)."""
    if kind == "hgb":
        return make_hgb(seed, metric_aligned=metric_aligned)
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                                  auto_class_weights="Balanced",
                                  early_stopping_rounds=100, verbose=0, random_seed=seed)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=1500, learning_rate=0.05, num_leaves=63,
                              class_weight="balanced", verbose=-1, n_jobs=-1,
                              random_state=seed)
    raise ValueError(kind)


def run(variant="C", seed=SEED, n_splits=5, model_kind="hgb"):
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()

    flags = VARIANT_FEATURES[variant]
    X = base_encode(train)
    X_test = base_encode(test)
    if "rule" in flags:
        X, X_test = add_rule_features(X), add_rule_features(X_test)
    if "soft" in flags:
        X, X_test = add_soft_impute(X, X_test)
        print(f"soft-impute features added: {X.shape[1]} cols")
    if "bundle" in flags:
        X, X_test = add_bundle_features(X, X_test)
        print(f"bundle features added: {X.shape[1]} cols")

    # TE 대상: 남아있는 수치형 + (번들이면) 상호작용 키
    te_cols = [c for c in NUM if c in X.columns]
    if "bundle" in flags:
        # 커뮤니티 v0.7 원 레시피: 범주형 6개도 TE (우리 기존엔 수치만)
        te_cols += [c for c in CAT if c in X.columns]
        te_cols += ["ix_sleep_stress", "ix_sleep_act", "ix_stress_quality"]

    orig = None
    if "orig" in flags:
        o = pd.read_csv(DATA / "original" / "student_health_dataset_50k.csv")
        y_o = o[TARGET].map(mapping).to_numpy()
        X_o = base_encode(o)
        if "rule" in flags:
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

        if "te" in flags:
            # exact-value TE: 각 정확한 값 = 하나의 카테고리 (fold 내부 교차적합)
            te = TargetEncoder(target_type="multiclass", cv=5, random_state=seed)
            sent = X_tr[te_cols].fillna(-999.0)
            te_tr = te.fit_transform(sent, y_tr)
            te_va = te.transform(X_va[te_cols].fillna(-999.0))
            te_te = te.transform(X_te[te_cols].fillna(-999.0))
            cols = [f"te_{c}_{k}" for c in te_cols for k in range(N_CLASS)]
            X_tr = pd.concat([X_tr.reset_index(drop=True),
                              pd.DataFrame(te_tr, columns=cols)], axis=1)
            X_va = pd.concat([X_va.reset_index(drop=True),
                              pd.DataFrame(te_va, columns=cols, index=range(len(X_va)))], axis=1)
            X_te = pd.concat([X_te.reset_index(drop=True),
                              pd.DataFrame(te_te, columns=cols)], axis=1)

        model = make_model(model_kind, seed + fold,
                           metric_aligned=("bundle" in flags and variant != "G"))
        model.fit(X_tr, y_tr)
        oof[va] = model.predict_proba(X_va)
        test_pred += model.predict_proba(X_te) / n_splits
        s = balanced_accuracy_score(y[va], oof[va].argmax(1))
        scores.append(s)
        iters = getattr(model, "n_iter_", None) or getattr(model, "best_iteration_", "?")
        print(f"  fold{fold} bal_acc={s:.5f} iters={iters}")

    cv = balanced_accuracy_score(y, oof.argmax(1))
    tag = f"v2_{variant}_s{seed}" if model_kind == "hgb" else f"v2_{variant}_{model_kind}_s{seed}"
    print(f"[V2-{variant}/{model_kind} seed{seed}] OOF={cv:.5f} mean={np.mean(scores):.5f} "
          f"std={np.std(scores):.5f} time={time.time()-t0:.0f}s")
    np.save(OOF / f"{tag}_oof.npy", oof)
    np.save(OOF / f"{tag}_test.npy", test_pred)
    return cv


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "C"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else SEED
    model_kind = sys.argv[3] if len(sys.argv) > 3 else "hgb"
    run(variant, seed, model_kind=model_kind)
