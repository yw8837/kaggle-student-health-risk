# Cl_F Part2: PLR-lite MLP + exact-value TE on ALL 13 cols (masa FT-T v2 recipe)
# 근거: masayakawamata ablation — per-value TE 13cols +0.0012 (5/5 fold), 조기중단 금지(+0.0003 낙관 제거)
# 5-fold OOF, 고정 16에폭, class-weighted CE. outputs: clf_nn_s42_{oof,test}.npy + submission
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

T0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_data():
    import glob, os
    for hit in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        return os.path.dirname(hit)
    raise FileNotFoundError("train.csv not found. /kaggle/input: "
                            + str(glob.glob("/kaggle/input/*")))


DATA = find_data()
CLASSES = ["at-risk", "unhealthy", "fit"]
TARGET, SEED, NC = "health_condition", 42, 3
EPOCHS, BS, LR = 16, 4096, 2e-3
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
CAT = list(ORDINAL)
TE_COLS = NUM + CAT          # 13컬럼 전부 exact-value TE -> 39피처


def encode(df):
    X = df[NUM + CAT].copy()
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
    X["total_activity_score"] = X["step_count"] + X["exercise_duration"] + X["calorie_expenditure"]
    X["metabolic_efficiency"] = X["calorie_expenditure"] / X["bmi"]
    X["cardio_stress_index"] = (X["heart_rate"] * X["bmi"]) / X["sleep_duration"]
    X["sleep_debt"] = 7.5 - X["sleep_duration"]
    X["hydration_ratio"] = X["water_intake"] / X["bmi"]
    X["stress_recovery_potential"] = (X["sleep_quality"] * X["sleep_duration"]) / (X["stress_level"] + 1)
    return X


class PeriodicEmbed(nn.Module):
    def __init__(self, n_feats, n_freq=8, sigma=1.0):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_feats, n_freq) * sigma)

    def forward(self, x):
        v = 2 * torch.pi * x.unsqueeze(-1) * self.freq
        return torch.cat([torch.sin(v), torch.cos(v)], -1).flatten(1)


class Net(nn.Module):
    def __init__(self, n_num, n_total, n_freq=8, hidden=384):
        super().__init__()
        self.n_num = n_num
        self.pe = PeriodicEmbed(n_num, n_freq)
        d_in = n_num * 2 * n_freq + n_total
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden // 2, NC))

    def forward(self, x):
        return self.mlp(torch.cat([self.pe(x[:, :self.n_num]), x], 1))


def train_fold(Xtr, ytr, Xva, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(len(NUM), Xtr.shape[1]).to(DEVICE)
    cw = compute_class_weight("balanced", classes=np.arange(NC), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                  torch.tensor(ytr)), batch_size=BS, shuffle=True)
    for ep in range(EPOCHS):                     # 고정 에폭 — 조기중단 낙관편향 제거
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        sch.step()
    model.eval()
    outs = []
    for Xp in (Xva, Xte):
        ps = []
        with torch.no_grad():
            for i in range(0, len(Xp), 65536):
                t = torch.tensor(Xp[i:i + 65536], dtype=torch.float32).to(DEVICE)
                ps.append(torch.softmax(model(t), 1).cpu().numpy())
        outs.append(np.vstack(ps))
    return outs


train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
X, X_test = encode(train), encode(test)
print(f"features={X.shape[1]} device={DEVICE} t={time.time()-T0:.0f}s", flush=True)

oof = np.zeros((len(X), NC))
tp = np.zeros((len(X_test), NC))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for fold, (tr, va) in enumerate(skf.split(X, y)):
    X_tr, X_va, X_te = X.iloc[tr].copy(), X.iloc[va].copy(), X_test.copy()
    te = TargetEncoder(target_type="multiclass", cv=5, random_state=SEED)
    cols = [f"te_{c}_{k}" for c in TE_COLS for k in range(NC)]
    t_tr = te.fit_transform(X_tr[TE_COLS].fillna(-999.0), y[tr])
    X_tr = pd.concat([X_tr.reset_index(drop=True), pd.DataFrame(t_tr, columns=cols)], axis=1)
    for Xp in (X_va, X_te):
        tv = te.transform(Xp[TE_COLS].fillna(-999.0))
        Xp.reset_index(drop=True, inplace=True)
        Xp[cols] = tv
    # NaN: 지시자 + 평균대치 (NN은 NaN 불가), 통계는 train fold에서만
    for c in list(X_tr.columns):
        if X_tr[c].isna().any() or X_va[c].isna().any() or X_te[c].isna().any():
            for Xp in (X_tr, X_va, X_te):
                Xp[f"na_{c}"] = Xp[c].isna().astype(float)
            mu = X_tr[c].mean()
            mu = 0.0 if pd.isna(mu) else mu
            for Xp in (X_tr, X_va, X_te):
                Xp[c] = Xp[c].fillna(mu)
    sc = StandardScaler().fit(X_tr)
    pv, pt = train_fold(sc.transform(X_tr), y[tr], sc.transform(X_va),
                        sc.transform(X_te[X_tr.columns]), SEED + fold)
    oof[va] = pv
    tp += pt / 5
    print(f"  fold{fold} bal_acc={balanced_accuracy_score(y[va], pv.argmax(1)):.5f}"
          f" t={time.time()-T0:.0f}s", flush=True)

cv = balanced_accuracy_score(y, oof.argmax(1))
print(f"[NN-TE13] OOF={cv:.5f} time={time.time()-T0:.0f}s", flush=True)
np.save("clf_nn_s42_oof.npy", oof)
np.save("clf_nn_s42_test.npy", tp)
sub = pd.DataFrame({"id": test["id"], TARGET: [CLASSES[i] for i in tp.argmax(1)]})
sub.to_csv("submission.csv", index=False)
print("done", flush=True)
