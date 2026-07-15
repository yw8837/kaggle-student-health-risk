# Cl_F Part1: GPU trio (XGB-cuda / CatBoost-GPU / LGBM-CPU) 5-fold OOF
# 13 cols + derived (rule/user3/mres/domain) + exact-value TE (fold-fit, leak-free)
# outputs: clf_{kind}_s42_{oof,test}.npy + solo scores + trio honest blend + submission
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

T0 = time.time()


def find_data():
    import glob, os
    for hit in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        return os.path.dirname(hit)
    raise FileNotFoundError("train.csv not found. /kaggle/input: "
                            + str(glob.glob("/kaggle/input/*")))


DATA = find_data()
CLASSES = ["at-risk", "unhealthy", "fit"]
TARGET, SEED, NC = "health_condition", 42, 3
NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
ORDINAL = {
    "stress_level": {"low": 0, "medium": 1, "high": 2},
    "sleep_quality": {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol": {"no": 0, "occasional": 1, "yes": 2},
    "diet_type": {"veg": 0, "balanced": 1, "non-veg": 2},
    "gender": {"female": 0, "male": 1, "other": 2},
}


def encode(df):
    X = df[NUM + list(ORDINAL)].copy()
    for c, m in ORDINAL.items():
        X[c] = X[c].map(m).astype("float64")
    s, st, act = X["sleep_duration"], X["stress_level"], X["physical_activity_level"]
    X["sleep_lt6"] = np.where(s.isna(), np.nan, (s < 6).astype(float))
    X["sleep_lt7"] = np.where(s.isna(), np.nan, (s < 7).astype(float))
    rp = np.full(len(X), np.nan)
    known = ~(s.isna() | st.isna() | act.isna())
    sv, stv, av = s[known], st[known], act[known]
    r = np.zeros(known.sum())
    r[(sv < 6) & (stv == 2)] = 1.0
    r[(sv >= 7) & (stv == 0) & (av == 2)] = 2.0
    rp[known.to_numpy()] = r
    X["rule_pred"] = rp
    for c in ("sleep_duration", "stress_level", "physical_activity_level"):
        X[f"miss_{c.split('_')[0]}"] = X[c].isna().astype(float)
    # user3
    X["sleep_efficiency"] = X["sleep_duration"] / (X["sleep_quality"] + 1)
    X["cal_per_step"] = X["calorie_expenditure"] / (X["step_count"] + 1)
    X["bmi_cat"] = pd.cut(X["bmi"], [0, 18.5, 23, 25, 30, 100], labels=False).astype(float)
    # mres bins (TE keys)
    for c, step in [("sleep_duration", 0.5), ("sleep_duration", 0.25), ("bmi", 0.5),
                    ("step_count", 500), ("calorie_expenditure", 100),
                    ("exercise_duration", 5)]:
        X[f"bin_{c}_{str(step).replace('.', '_')}"] = (X[c] / step).round()
    # domain6
    X["total_activity_score"] = X["step_count"] + X["exercise_duration"] + X["calorie_expenditure"]
    X["metabolic_efficiency"] = X["calorie_expenditure"] / X["bmi"]
    X["cardio_stress_index"] = (X["heart_rate"] * X["bmi"]) / X["sleep_duration"]
    X["sleep_debt"] = 7.5 - X["sleep_duration"]
    X["hydration_ratio"] = X["water_intake"] / X["bmi"]
    X["stress_recovery_potential"] = (X["sleep_quality"] * X["sleep_duration"]) / (X["stress_level"] + 1)
    return X


train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
X, X_test = encode(train), encode(test)
TE_COLS = NUM + [c for c in X.columns if c.startswith("bin_")]
print(f"features={X.shape[1]} te_keys={len(TE_COLS)} t={time.time()-T0:.0f}s", flush=True)


def sw(y_):
    cnt = np.bincount(y_)
    w = len(y_) / (len(cnt) * cnt)
    return w[y_]


def make(kind, seed):
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=1200, learning_rate=0.05, max_depth=8,
                             min_child_weight=5, tree_method="hist", device="cuda",
                             n_jobs=-1, random_state=seed)
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=8,
                                  auto_class_weights="Balanced", task_type="GPU",
                                  verbose=0, random_seed=seed)
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=1200, learning_rate=0.05, num_leaves=63,
                              min_child_samples=40, class_weight="balanced",
                              verbose=-1, n_jobs=-1, random_state=seed)


PRIOR = np.bincount(y) / len(y)
results = {}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
splits = list(skf.split(X, y))
for kind in ("xgb", "cat", "lgbm"):
    t1 = time.time()
    oof = np.zeros((len(X), NC))
    tp = np.zeros((len(X_test), NC))
    for fold, (tr, va) in enumerate(splits):
        X_tr, X_va, X_te = X.iloc[tr].copy(), X.iloc[va].copy(), X_test.copy()
        te = TargetEncoder(target_type="multiclass", cv=5, random_state=SEED)
        cols = [f"te_{c}_{k}" for c in TE_COLS for k in range(NC)]
        t_tr = te.fit_transform(X_tr[TE_COLS].fillna(-999.0), y[tr])
        X_tr = pd.concat([X_tr.reset_index(drop=True),
                          pd.DataFrame(t_tr, columns=cols)], axis=1)
        for Xp in (X_va, X_te):
            tv = te.transform(Xp[TE_COLS].fillna(-999.0))
            Xp.reset_index(drop=True, inplace=True)
            Xp[cols] = tv
        m = make(kind, SEED + fold)
        if kind == "xgb":
            m.fit(X_tr, y[tr], sample_weight=sw(y[tr]))
        else:
            m.fit(X_tr, y[tr])
        oof[va] = m.predict_proba(X_va)
        tp += m.predict_proba(X_te) / 5
        print(f"  {kind} fold{fold} {balanced_accuracy_score(y[va], oof[va].argmax(1)):.5f}"
              f" t={time.time()-t1:.0f}s", flush=True)
    # xgb는 unweighted가 아니라 sample_weight로 tilt됨; 그래도 argmax 낮으면 prior-correct
    if balanced_accuracy_score(y, oof.argmax(1)) < 0.94:
        oof = (oof / PRIOR); oof /= oof.sum(1, keepdims=True)
        tp = (tp / PRIOR); tp /= tp.sum(1, keepdims=True)
    s = balanced_accuracy_score(y, oof.argmax(1))
    results[kind] = (oof, tp, s)
    np.save(f"clf_{kind}_s{SEED}_oof.npy", oof)
    np.save(f"clf_{kind}_s{SEED}_test.npy", tp)
    print(f"[{kind}] OOF={s:.5f} time={time.time()-t1:.0f}s total={time.time()-T0:.0f}s",
          flush=True)

# trio nested honest blend (행분할 중첩: 가중치는 tr에서만 적합)
names = list(results)
P = np.stack([results[n][0] for n in names])
rng = np.random.default_rng(0)
kf = KFold(5, shuffle=True, random_state=0)
honest, ws = [], []
for tr, va in kf.split(y):
    cand = np.vstack([rng.dirichlet(np.ones(3) * 0.7, 1500), np.eye(3),
                      np.full((1, 3), 1 / 3)])
    sc = [balanced_accuracy_score(y[tr], np.tensordot(w, P[:, tr], 1).argmax(1))
          for w in cand]
    w = cand[int(np.argmax(sc))]
    ws.append(w)
    honest.append(balanced_accuracy_score(y[va], np.tensordot(w, P[:, va], 1).argmax(1)))
wm = np.mean(ws, 0)
print(f"trio honest={np.mean(honest):.5f} weights={dict(zip(names, wm.round(3)))}", flush=True)

Ptest = np.stack([results[n][1] for n in names])
blend = np.tensordot(wm, Ptest, 1)
sub = pd.DataFrame({"id": test["id"],
                    TARGET: [CLASSES[i] for i in blend.argmax(1)]})
sub.to_csv("submission.csv", index=False)
print(f"done total={time.time()-T0:.0f}s", flush=True)
