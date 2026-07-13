"""EDA 차트 생성 -> assets/*.png (HTML 리포트 임베드용).
차트 라벨은 영어(matplotlib 한글폰트 이슈 회피), 서술은 HTML에서 한국어.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from config import DATA, CLASSES

A = Path(__file__).resolve().parents[1] / "assets"; A.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi":110, "font.size":11, "axes.grid":True,
                     "grid.alpha":0.25, "axes.spines.top":False, "axes.spines.right":False})
C = {"at-risk":"#f59e0b", "unhealthy":"#ef4444", "fit":"#10b981"}
ORDER = ["at-risk","unhealthy","fit"]

tr = pd.read_csv(DATA/"train.csv"); te = pd.read_csv(DATA/"test.csv")
NUM = ["sleep_duration","heart_rate","bmi","calorie_expenditure","step_count","exercise_duration","water_intake"]
CAT = ["diet_type","stress_level","sleep_quality","physical_activity_level","smoking_alcohol","gender"]
stats = {}

# 1) class balance
vc = tr["health_condition"].value_counts().reindex(ORDER)
pct = (vc/len(tr)*100).round(2)
stats["class_pct"] = pct.to_dict(); stats["class_cnt"] = vc.to_dict()
fig, ax = plt.subplots(figsize=(7,3.6))
b = ax.bar(ORDER, vc.values, color=[C[c] for c in ORDER])
for i,c in enumerate(ORDER):
    ax.text(i, vc[c], f"{pct[c]}%\n({vc[c]:,})", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("count"); ax.set_title("Target class distribution (train) — heavily imbalanced")
ax.set_ylim(0, vc.max()*1.18)
fig.tight_layout(); fig.savefig(A/"class_balance.png"); plt.close(fig)

# 2) why balanced accuracy: naive-all-atrisk vs balanced
# naive: predict all 'at-risk' -> recall atrisk=1, others=0 -> bal_acc=1/3, acc=0.8587
naive_acc = pct["at-risk"]/100; naive_bacc = 1/3
fig, ax = plt.subplots(figsize=(7,3.6))
x = np.arange(2); w=0.36
ax.bar(x-w/2, [naive_acc, naive_bacc], w, label='"predict all at-risk"', color="#9ca3af")
ax.bar(x+w/2, [0.9509, 0.94952], w, label="V1 model", color="#2563eb")
ax.set_xticks(x); ax.set_xticklabels(["Plain Accuracy","Balanced Accuracy"])
ax.axhline(1/3, ls="--", lw=1, color="#ef4444")
ax.text(1.15, 1/3+0.01, "floor 0.333", color="#ef4444", fontsize=9)
for i,v in enumerate([naive_acc,naive_bacc]): ax.text(i-w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
for i,v in enumerate([0.9509,0.94952]): ax.text(i+w/2, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0,1.1); ax.set_ylabel("score"); ax.legend()
ax.set_title("Why the metric matters: naive 86% accuracy = only 0.333 balanced")
fig.tight_layout(); fig.savefig(A/"metric_demo.png"); plt.close(fig)

# 3) missingness
miss = (tr[NUM+CAT].isna().mean()*100).sort_values()
stats["missing_pct"] = miss.round(2).to_dict()
fig, ax = plt.subplots(figsize=(7,4.2))
ax.barh(miss.index, miss.values, color="#8b5cf6")
for i,(k,v) in enumerate(miss.items()): ax.text(v+0.1, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("% missing"); ax.set_title("Missing values per feature (train)")
fig.tight_layout(); fig.savefig(A/"missingness.png"); plt.close(fig)

# 4) numeric feature distribution by class (violin grid)
fig, axes = plt.subplots(2,4, figsize=(13,6)); axes=axes.ravel()
for j,f in enumerate(NUM):
    ax=axes[j]
    data=[tr.loc[tr.health_condition==c, f].dropna().values for c in ORDER]
    parts=ax.violinplot(data, showmeans=True, showextrema=False)
    for pc,c in zip(parts['bodies'], ORDER): pc.set_facecolor(C[c]); pc.set_alpha(.7)
    ax.set_xticks([1,2,3]); ax.set_xticklabels(ORDER, rotation=20, fontsize=8)
    ax.set_title(f, fontsize=10)
axes[-1].axis("off")
fig.suptitle("Numeric features by health class (which features separate classes?)", y=1.02)
fig.tight_layout(); fig.savefig(A/"numeric_by_class.png", bbox_inches="tight"); plt.close(fig)

# 5) categorical composition by class
fig, axes = plt.subplots(2,3, figsize=(13,6.5)); axes=axes.ravel()
for j,f in enumerate(CAT):
    ax=axes[j]
    ct=pd.crosstab(tr[f], tr["health_condition"], normalize="index").reindex(columns=ORDER)
    ct.plot(kind="bar", stacked=True, ax=ax, color=[C[c] for c in ORDER], legend=False, width=.8)
    ax.set_title(f, fontsize=10); ax.set_xlabel(""); ax.set_ylabel("class share")
    ax.tick_params(axis="x", rotation=15, labelsize=8)
axes[0].legend(ORDER, fontsize=8, loc="lower right")
fig.suptitle("Categorical features: health-class share within each category", y=1.02)
fig.tight_layout(); fig.savefig(A/"categorical_by_class.png", bbox_inches="tight"); plt.close(fig)

# 6) quick feature importance (single fast lgbm on subsample)
import lightgbm as lgb
from features import prep
sub = tr.sample(120000, random_state=0).reset_index(drop=True)
X,_,y,cats,_ = prep(sub, te.copy())
m = lgb.train(dict(objective="multiclass",num_class=3,learning_rate=0.05,num_leaves=63,verbose=-1),
              lgb.Dataset(X,y,categorical_feature=cats), num_boost_round=300)
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values()
stats["importance_top"] = imp.sort_values(ascending=False).round(0).to_dict()
fig, ax = plt.subplots(figsize=(7,4.6))
ax.barh(imp.index, imp.values/imp.values.sum()*100, color="#0ea5e9")
ax.set_xlabel("gain importance (%)"); ax.set_title("What drives the prediction? (LightGBM gain)")
fig.tight_layout(); fig.savefig(A/"importance.png"); plt.close(fig)

# 7) train vs test sanity (a couple of numerics)
fig, axes = plt.subplots(1,3, figsize=(12,3.4))
for ax,f in zip(axes, ["bmi","step_count","heart_rate"]):
    ax.hist(tr[f].dropna(), bins=60, density=True, alpha=.55, label="train", color="#2563eb")
    ax.hist(te[f].dropna(), bins=60, density=True, alpha=.55, label="test", color="#f59e0b")
    ax.set_title(f, fontsize=10)
axes[0].legend(fontsize=9)
fig.suptitle("Train vs Test distributions match (synthetic data sanity)", y=1.03)
fig.tight_layout(); fig.savefig(A/"train_test.png", bbox_inches="tight"); plt.close(fig)

import json
(Path(A)/"stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("charts saved to", A)
print(json.dumps(stats, ensure_ascii=False, default=str)[:800])
