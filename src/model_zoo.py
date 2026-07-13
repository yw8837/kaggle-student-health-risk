"""10+ 모델 비교 하네스 (과제 유의점 준수)

- scikit-learn 문법만 사용 (native API 금지 -> LGBMClassifier/XGBClassifier/CatBoostClassifier)
- 층화추출: train_test_split(stratify=y) 로 샘플링/분할
- 실험조건 동일: 같은 전처리(ColumnTransformer), 같은 split, 같은 class-balancing(sample_weight)
- valid_sets 모니터링: 부스팅 계열은 eval_set=[(X_va,y_va)] 지정
- 시간 측정: fit/predict 시간 기록 -> 시간 대비 성능 시각화용 CSV 저장

실행: python src/model_zoo.py [sample_size]  (기본 100_000)
출력: oof/model_zoo_results.csv, assets/model_zoo_time_vs_score.png
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import DATA, OOF, SEED, CLASSES
from features import load, encode_target

ASSETS = OOF.parent / "assets"

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality",
       "physical_activity_level", "smoking_alcohol", "gender"]


def build_preprocessor():
    """공통 전처리 (실험조건 동일): 결측 대치 + 스케일링 + 원핫."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])


def model_zoo():
    """비교 대상 11개. 전부 sklearn API. 실험조건 통일: 기본값 위주 + 동일 seed."""
    return {
        "LogisticRegression":  LogisticRegression(max_iter=2000, random_state=SEED),
        "GaussianNB":          GaussianNB(),
        "DecisionTree":        DecisionTreeClassifier(random_state=SEED),
        "KNeighbors":          KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
        "RandomForest":        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "ExtraTrees":          ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        "MLP":                 MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                             max_iter=200, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=500, early_stopping=True,
                                                               random_state=SEED),
        "LightGBM":            LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                                              verbose=-1, n_jobs=-1, random_state=SEED),
        "XGBoost":             XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=7,
                                             tree_method="hist", early_stopping_rounds=100,
                                             n_jobs=-1, random_state=SEED, verbosity=0),
        "CatBoost":            CatBoostClassifier(iterations=2000, learning_rate=0.05,
                                                  early_stopping_rounds=100, verbose=0,
                                                  random_seed=SEED),
    }


def main(sample_size=100_000):
    train, _ = load()
    y_all, mapping = encode_target(train)
    X_all = train[NUM + CAT]

    # 층화추출 (유의점 5): 클래스 비율 유지한 채 샘플링
    X_s, _, y_s, _ = train_test_split(
        X_all, y_all, train_size=sample_size, stratify=y_all, random_state=SEED)
    # 층화 train/valid 분할
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_s, y_s, test_size=0.25, stratify=y_s, random_state=SEED)
    print(f"sample={len(X_s):,} -> train={len(X_tr):,} valid={len(X_va):,}")
    print("class ratio kept:", np.bincount(y_tr) / len(y_tr))

    # 공통 전처리는 train에만 fit (누수 방지)
    pre = build_preprocessor()
    Xt_tr = pre.fit_transform(X_tr)
    Xt_va = pre.transform(X_va)

    # 클래스 불균형 보정 (실험조건 동일: 전 모델 sample_weight 'balanced')
    w_tr = compute_sample_weight("balanced", y_tr)

    rows = []
    for name, model in model_zoo().items():
        t0 = time.perf_counter()
        try:
            # valid_sets 모니터링 (유의점 3): eval_set 지원 모델에 지정
            if name == "XGBoost":
                model.fit(Xt_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(Xt_va, y_va)], verbose=False)
            elif name == "LightGBM":
                model.fit(Xt_tr, y_tr, sample_weight=w_tr,
                          eval_set=[(Xt_va, y_va)],
                          callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)])
            elif name == "CatBoost":
                model.fit(Xt_tr, y_tr, sample_weight=w_tr, eval_set=(Xt_va, y_va))
            elif name in ("MLP", "KNeighbors"):
                model.fit(Xt_tr, y_tr)          # sample_weight 미지원 -> 명시적 한계 기록
            else:
                model.fit(Xt_tr, y_tr, sample_weight=w_tr)
            fit_t = time.perf_counter() - t0

            t1 = time.perf_counter()
            proba = model.predict_proba(Xt_va)
            pred_t = time.perf_counter() - t1

            score = balanced_accuracy_score(y_va, proba.argmax(1))
            rows.append(dict(model=name, balanced_accuracy=round(score, 5),
                             fit_sec=round(fit_t, 2), predict_sec=round(pred_t, 3),
                             weighted="no" if name in ("MLP", "KNeighbors") else "yes"))
            print(f"{name:22s} bal_acc={score:.5f} fit={fit_t:7.1f}s pred={pred_t:6.2f}s")
        except Exception as e:
            rows.append(dict(model=name, balanced_accuracy=np.nan,
                             fit_sec=np.nan, predict_sec=np.nan, weighted="ERROR"))
            print(f"{name:22s} FAILED: {e}")

    df = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    out_csv = OOF / "model_zoo_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n", df.to_string(index=False))
    print(f"\nsaved {out_csv}")

    # 시간 대비 성능 시각화 (가이드라인 필수)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df.dropna(subset=["balanced_accuracy"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(ok.fit_sec, ok.balanced_accuracy, s=70, c="#0ea5e9", zorder=3)
    for _, r in ok.iterrows():
        ax.annotate(r.model, (r.fit_sec, r.balanced_accuracy),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("training time (sec, log scale)")
    ax.set_ylabel("balanced accuracy (valid)")
    ax.set_title(f"Model comparison: time vs score (stratified {len(X_s)//1000}k sample)")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_zoo_time_vs_score.png", dpi=110)
    print(f"saved {ASSETS / 'model_zoo_time_vs_score.png'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    main(n)
