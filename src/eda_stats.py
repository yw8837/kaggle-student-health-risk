"""EDA 통계 검정 (가이드라인: 시각화 + 통계 검정)

- 범주형 6개 vs 타겟: 카이제곱 독립성 검정 + Cramer's V (효과크기)
- 수치형 7개 vs 타겟: 일원분산분석(ANOVA F) + Kruskal-Wallis(비모수) + eta^2 (효과크기)
p-value는 표본이 커서 전부 유의하게 나오므로, 효과크기로 "얼마나 강하게 가르는지"를 본다.

실행: python src/eda_stats.py
출력: oof/eda_stats_categorical.csv, oof/eda_stats_numeric.csv, assets/effect_size.png
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import DATA, OOF, TARGET

ASSETS = OOF.parent / "assets"

NUM = ["sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
       "step_count", "exercise_duration", "water_intake"]
CAT = ["diet_type", "stress_level", "sleep_quality",
       "physical_activity_level", "smoking_alcohol", "gender"]


def cramers_v(confusion):
    chi2 = stats.chi2_contingency(confusion)[0]
    n = confusion.to_numpy().sum()
    r, k = confusion.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def eta_squared(groups):
    """ANOVA 효과크기: 집단간 제곱합 / 전체 제곱합."""
    all_v = np.concatenate(groups)
    grand = all_v.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_v - grand) ** 2).sum()
    return ss_between / ss_total


def main():
    tr = pd.read_csv(DATA / "train.csv")

    # --- 범주형: 카이제곱 + Cramer's V ---
    rows = []
    for c in CAT:
        ct = pd.crosstab(tr[c], tr[TARGET])          # 결측 행은 자동 제외
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append(dict(feature=c, test="chi-square", statistic=round(chi2, 1),
                         p_value=p, effect_size=round(cramers_v(ct), 4),
                         effect_metric="Cramer's V"))
    cat_df = pd.DataFrame(rows).sort_values("effect_size", ascending=False)
    cat_df.to_csv(OOF / "eda_stats_categorical.csv", index=False)

    # --- 수치형: ANOVA + Kruskal + eta^2 ---
    rows = []
    for c in NUM:
        groups = [tr.loc[tr[TARGET] == k, c].dropna().to_numpy()
                  for k in tr[TARGET].unique()]
        f, p_a = stats.f_oneway(*groups)
        h, p_k = stats.kruskal(*groups)
        rows.append(dict(feature=c, anova_F=round(f, 1), anova_p=p_a,
                         kruskal_H=round(h, 1), kruskal_p=p_k,
                         effect_size=round(eta_squared(groups), 4),
                         effect_metric="eta^2"))
    num_df = pd.DataFrame(rows).sort_values("effect_size", ascending=False)
    num_df.to_csv(OOF / "eda_stats_numeric.csv", index=False)

    print("=== categorical (chi-square, Cramer's V) ===")
    print(cat_df.to_string(index=False))
    print("\n=== numeric (ANOVA/Kruskal, eta^2) ===")
    print(num_df.to_string(index=False))

    # --- 효과크기 통합 차트 (PPT용) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eff = pd.concat([
        cat_df[["feature", "effect_size"]].assign(kind="categorical (Cramer's V)"),
        num_df[["feature", "effect_size"]].assign(kind="numeric (eta^2)"),
    ]).sort_values("effect_size")
    colors = eff.kind.map({"categorical (Cramer's V)": "#8b5cf6",
                           "numeric (eta^2)": "#0ea5e9"})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(eff.feature, eff.effect_size, color=colors)
    for i, (_, r) in enumerate(eff.iterrows()):
        ax.text(r.effect_size + .003, i, f"{r.effect_size:.3f}", va="center", fontsize=9)
    ax.set_xlabel("effect size (association with health_condition)")
    ax.set_title("Statistical tests: which features truly separate the classes?\n"
                 "(all p-values < 1e-100 at n=690k; effect size is what matters)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#8b5cf6", label="categorical (Cramer's V)"),
                       Patch(color="#0ea5e9", label="numeric (eta^2)")], loc="lower right")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(ASSETS / "effect_size.png", dpi=110)
    print(f"\nsaved {ASSETS / 'effect_size.png'}")


if __name__ == "__main__":
    main()
