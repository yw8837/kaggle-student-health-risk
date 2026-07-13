"""NN 카드: 주기 임베딩(PLR-lite) MLP — 트리와 상보적인 유일한 모델 계열

- 피처: V2-C와 동일(서수+규칙피처) + 표준화. 결측은 평균대치+지시자 (NN은 NaN 불가)
- 수치 피처마다 sin/cos 주기 임베딩 (RealMLP 핵심 아이디어의 경량판)
- class-weighted CrossEntropy (불균형 대응), 층화 K-fold
- 로컬: CPU 스모크 (--smoke) / 캐글: GPU 풀런 (노트북 변환용)

실행: python src/train_nn.py [--smoke]
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from config import DATA, OOF, SEED, CLASSES, N_CLASS, TARGET
from train_v2 import base_encode, add_rule_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PeriodicEmbed(nn.Module):
    """수치 피처별 sin/cos 주기 임베딩 (PLR-lite)."""
    def __init__(self, n_feats, n_freq=8, sigma=1.0):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_feats, n_freq) * sigma)

    def forward(self, x):                       # x: (B, F)
        v = 2 * torch.pi * x.unsqueeze(-1) * self.freq  # (B, F, K)
        return torch.cat([torch.sin(v), torch.cos(v)], -1).flatten(1)  # (B, F*2K)


class Net(nn.Module):
    def __init__(self, n_feats, n_freq=8, hidden=256):
        super().__init__()
        self.pe = PeriodicEmbed(n_feats, n_freq)
        d_in = n_feats * 2 * n_freq + n_feats   # 임베딩 + 원값
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden // 2, N_CLASS))

    def forward(self, x):
        return self.mlp(torch.cat([self.pe(x), x], 1))


def prep():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()
    X = add_rule_features(base_encode(train))
    X_test = add_rule_features(base_encode(test))
    # NN용 결측 처리: 지시자 + 평균 대치
    for c in X.columns:
        if X[c].isna().any() or X_test[c].isna().any():
            X[f"na_{c}"] = X[c].isna().astype(float)
            X_test[f"na_{c}"] = X_test[c].isna().astype(float)
            mu = X[c].mean()
            X[c] = X[c].fillna(mu); X_test[c] = X_test[c].fillna(mu)
    return X, X_test, y


def train_fold(X_tr, y_tr, X_va, y_va, X_te, seed, epochs=12, bs=4096, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net(X_tr.shape[1]).to(DEVICE)
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASS), y=y_tr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr)), batch_size=bs, shuffle=True)
    Xva_t = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)
    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    best_ba, best_va, best_te, patience = -1, None, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            pv = torch.softmax(model(Xva_t), 1).cpu().numpy()
        ba = balanced_accuracy_score(y_va, pv.argmax(1))
        if ba > best_ba:
            best_ba, patience = ba, 0
            best_va = pv
            with torch.no_grad():
                best_te = torch.softmax(model(Xte_t), 1).cpu().numpy()
        else:
            patience += 1
            if patience >= 3:
                break
    return best_va, best_te, best_ba


def main(smoke=False):
    t0 = time.time()
    X, X_test, y = prep()
    if smoke:
        idx = np.random.default_rng(0).choice(len(X), 30_000, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
        X_test = X_test.iloc[:5000]
        n_splits, epochs = 2, 4
    else:
        n_splits, epochs = 5, 12

    cols = X.columns
    oof = np.zeros((len(X), N_CLASS)); test_pred = np.zeros((len(X_test), N_CLASS))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X.iloc[tr])
        Xtr, Xva = sc.transform(X.iloc[tr]), sc.transform(X.iloc[va])
        Xte = sc.transform(X_test[cols])
        pv, pt, ba = train_fold(Xtr, y[tr], Xva, y[va], Xte, SEED + fold, epochs=epochs)
        oof[va] = pv; test_pred += pt / n_splits
        print(f"  fold{fold} best bal_acc={ba:.5f}")
    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"[NN{'-smoke' if smoke else ''}] OOF={cv:.5f} time={time.time()-t0:.0f}s device={DEVICE}")
    if not smoke:
        np.save(OOF / "nn_s42_oof.npy", oof); np.save(OOF / "nn_s42_test.npy", test_pred)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
