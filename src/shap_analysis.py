"""SHAP 모델 설명 (과제 유의점 9: https://shap.readthedocs.io/en/latest/)

- LGBMClassifier(sklearn API) 학습 → TreeExplainer
- 범주형은 의미 순서로 서수 인코딩 → beeswarm 색이 '빨강=높음'으로 읽힘
- 산출:
    assets/shap_summary_bar.png        (3클래스 통합 중요도)
    assets/shap_beeswarm_unhealthy.png (unhealthy 방향 요인)
    assets/shap_beeswarm_fit.png       (fit 방향 요인)
    assets/shap_waterfall_student.png  (개인 설명: 왜 이 학생이 위험한가)

실행: python src/shap_analysis.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import balanced_accuracy_score

from config import SEED, CLASSES, OOF
from features import load, encode_target

ASSETS = OOF.parent / "assets"

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
# 서수 인코딩: 순서가 의미를 갖도록 (SHAP 색상 해석용)
ORDINAL = {
    "stress_level":            {"low": 0, "medium": 1, "high": 2},
    "sleep_quality":           {"poor": 0, "average": 1, "good": 2},
    "physical_activity_level": {"sedentary": 0, "moderate": 1, "active": 2},
    "smoking_alcohol":         {"no": 0, "occasional": 1, "yes": 2},
    "diet_type":               {"veg": 0, "balanced": 1, "non-veg": 2},
    "gender":                  {"female": 0, "male": 1, "other": 2},
}


def prepare(n_train=80_000, n_shap=3_000):
    train, _ = load()
    y, _ = encode_target(train)
    X = train[NUM + list(ORDINAL)].copy()
    for c, m in ORDINAL.items():
        X[c] = X[c].map(m)                     # NaN은 그대로 유지 (LGBM native 처리)

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=n_train, stratify=y, random_state=SEED)
    X_va, _, y_va, _ = train_test_split(
        X_rest, y_rest, train_size=20_000, stratify=y_rest, random_state=SEED)
    X_sh, _, y_sh, _ = train_test_split(
        X_va, y_va, train_size=n_shap, stratify=y_va, random_state=SEED)
    return X_tr, y_tr, X_va, y_va, X_sh, y_sh


def main():
    X_tr, y_tr, X_va, y_va, X_sh, y_sh = prepare()
    w = compute_sample_weight("balanced", y_tr)

    model = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                           verbose=-1, n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_tr, sample_weight=w,
              eval_set=[(X_va, y_va)],                       # valid_sets 모니터링
              callbacks=[early_stopping(100, verbose=False)])
    score = balanced_accuracy_score(y_va, model.predict(X_va))
    print(f"explainer model valid balanced_acc = {score:.5f}")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sh)          # (n, feat, class) 또는 [class별 (n, feat)]
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        sv_list = [sv[:, :, k] for k in range(sv.shape[2])]
    else:
        sv_list = list(sv)
    print("shap values:", len(sv_list), "classes x", sv_list[0].shape)

    # 1) 통합 summary (bar, 3클래스)
    plt.figure()
    shap.summary_plot(sv_list, X_sh, plot_type="bar",
                      class_names=CLASSES, show=False)
    plt.title("SHAP feature importance by class")
    plt.tight_layout()
    plt.savefig(ASSETS / "shap_summary_bar.png", dpi=110, bbox_inches="tight")
    plt.close("all")

    # 2) 클래스별 beeswarm (소수 클래스 2개가 관심사)
    for cls in ("unhealthy", "fit"):
        k = CLASSES.index(cls)
        plt.figure()
        shap.summary_plot(sv_list[k], X_sh, show=False, max_display=13)
        plt.title(f"SHAP beeswarm — toward '{cls}'\n(red = high feature value)")
        plt.tight_layout()
        plt.savefig(ASSETS / f"shap_beeswarm_{cls}.png", dpi=110, bbox_inches="tight")
        plt.close("all")

    # 3) 개인 설명: unhealthy로 예측된 학생 1명 waterfall
    proba = model.predict_proba(X_sh)
    k_un = CLASSES.index("unhealthy")
    idx = int(np.argmax(proba[:, k_un]))       # unhealthy 확률 최고 학생
    exp = explainer(X_sh)                      # Explanation (n, feat, class)
    one = exp[idx, :, k_un] if exp.values.ndim == 3 else exp[idx]
    plt.figure()
    shap.plots.waterfall(one, max_display=10, show=False)
    plt.title(f"Why is this student 'unhealthy'? (p={proba[idx, k_un]:.2f})")
    plt.tight_layout()
    plt.savefig(ASSETS / "shap_waterfall_student.png", dpi=110, bbox_inches="tight")
    plt.close("all")

    row = X_sh.iloc[idx]
    print(f"waterfall student: sleep={row.sleep_duration}, stress={row.stress_level}, "
          f"activity={row.physical_activity_level}, true={CLASSES[y_sh[idx]]}")
    print("saved 4 SHAP plots to", ASSETS)


if __name__ == "__main__":
    main()
