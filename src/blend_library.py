"""공개 OOF 라이브러리 블렌드 — 다자 가중 서치 (중첩검증, 확률/로짓 공간 비교)

멤버: 우리 N5·NN + 공개 4종 (RealMLP x2, LogReg스태커, FT-Transformer) [+CatBoost 완료 시]
방법: 행 5분할 중첩 — 4/5에서 Dirichlet 랜덤서치(+정제)로 가중치 적합, 1/5 정직 평가
비교 기준: N7 2-way (honest 0.95084)

실행: python src/blend_library.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, CLASSES, TARGET

BASE = OOF.parent / "consensus_src"


def load_library():
    train = pd.read_csv(DATA / "train.csv", usecols=[TARGET])
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    def csv_probs(path):
        df = pd.read_csv(path).sort_values("id")
        return df[["at-risk", "unhealthy", "fit"]].to_numpy()

    lib = {
        "N5":    np.load(OOF / "n5_oof.npy"),
        "RMLP":  csv_probs(BASE / "nawfeelrahman1124444_ps-s6-ep6-realmlp-0-95090/oof_preds.csv"),
        "STACK": csv_probs(BASE / "masayakawamata_s6e7-logreg-stacker-cv-0-95067/oof_stack.csv"),
        "FTT":   csv_probs(BASE / "masayakawamata_s6e7-ft-transformer-v2-cv-0-95063/oof_ftt.csv"),
        "YZMLP": np.load(BASE / "yunsuxiaozi_pss6e7-realmlp-cv-0-95063/oof_predictions.npy")[:, [0, 2, 1]],
        "NN":    np.load(OOF / "nn_s42_oof.npy"),
    }
    p_cat = OOF / "v2_C_cat_s42_oof.npy"
    if p_cat.exists():
        lib["CAT"] = np.load(p_cat)
    return y, lib


def search_weights(P, y_tr_idx, y, n_draw=3000, seed=0):
    """Dirichlet 랜덤서치 + 상위해 좌표 정제."""
    rng = np.random.default_rng(seed)
    M = P.shape[0]
    Ptr = P[:, y_tr_idx]
    ytr = y[y_tr_idx]

    def score(w):
        return balanced_accuracy_score(ytr, np.tensordot(w, Ptr, axes=1).argmax(1))

    cands = rng.dirichlet(np.ones(M) * 0.7, size=n_draw)
    # 솔로·균등도 후보에 포함
    cands = np.vstack([cands, np.eye(M), np.full((1, M), 1 / M)])
    scores = np.array([score(w) for w in cands])
    best_w, best_s = cands[scores.argmax()], scores.max()
    # 좌표 정제 2라운드
    for _ in range(2):
        for j in range(M):
            for d in (-0.1, -0.05, 0.05, 0.1):
                w = best_w.copy()
                w[j] = max(0.0, w[j] + d)
                w = w / w.sum()
                s = score(w)
                if s > best_s:
                    best_s, best_w = s, w
    return best_w, best_s


def main():
    y, lib = load_library()
    names = list(lib)
    for n in names:
        print(f"{n:6s} solo = {balanced_accuracy_score(y, lib[n].argmax(1)):.5f}")

    for space in ("prob", "logit"):
        P = np.stack([lib[n] for n in names])
        if space == "logit":
            P = np.log(np.clip(P, 1e-7, 1))
        kf = KFold(5, shuffle=True, random_state=0)
        honest, base_n7, ws = [], [], []
        for tr, va in kf.split(y):
            w, _ = search_weights(P, tr, y, seed=1)
            ws.append(w)
            honest.append(balanced_accuracy_score(
                y[va], np.tensordot(w, P[:, va], axes=1).argmax(1)))
            # N7 기준 (prob 공간 0.405/0.595)
            b = 0.405 * lib["N5"][va] + 0.595 * lib["RMLP"][va]
            base_n7.append(balanced_accuracy_score(y[va], b.argmax(1)))
        wm = np.mean(ws, axis=0)
        print(f"\n[{space}] weights:", {n: round(float(w), 3) for n, w in zip(names, wm)})
        print(f"[{space}] honest={np.mean(honest):.5f} (fold std {np.std(honest):.5f}) "
              f"| N7 기준={np.mean(base_n7):.5f} | gain={np.mean(honest)-np.mean(base_n7):+.5f}")


if __name__ == "__main__":
    main()
