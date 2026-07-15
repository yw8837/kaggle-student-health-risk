"""지시 #9 + #14: Optuna OOF 가중 블렌딩 + Rank/Power Averaging — 전 멤버 재실행

- 멤버: N5, RMLP, YZMLP, STACK, FTT, K(+CAT, NN) + 신규 s1/s2/s3 (있으면 자동 편입)
- 방법1: Optuna(TPE)로 가중치 탐색 — KFold(5) 행분할 중첩 (tr에서 목적함수, va로 정직 평가)
- 방법2: Rank Averaging (동일 중첩)
- 방법3: Power Averaging k∈{0.25..3} (동일 중첩)
기준선: N9 honest 0.95088

실행: python src/blend_optuna_full.py [n_trials]
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import optuna
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import OOF
from blend_library import load_library

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main(n_trials=400):
    t0 = time.time()
    y, lib = load_library()
    if (OOF / "v2_K_s42_oof.npy").exists():
        lib["K"] = np.load(OOF / "v2_K_s42_oof.npy")
    for tag, f in [("S1", "s1_q20_s42_oof.npy"), ("S2", "s2_rulepost_oof.npy"),
                   ("S3", "s3_flip_s42_oof.npy")]:
        p = OOF / f
        if p.exists():
            lib[tag] = np.load(p)
    names = list(lib)
    P = np.stack([lib[n] for n in names])
    M = len(names)
    print("members:", names)
    for n in names:
        print(f"  {n:6s} solo={balanced_accuracy_score(y, lib[n].argmax(1)):.5f}")

    kf = KFold(5, shuffle=True, random_state=0)
    res = {"optuna": [], "rank": [], "power": [], "n9": []}
    w_all = []
    n5, rmlp, K = lib["N5"], lib["RMLP"], lib["K"]
    for fi, (tr, va) in enumerate(kf.split(y)):
        Ptr, Pva = P[:, tr], P[:, va]
        ytr, yva = y[tr], y[va]

        # 1) Optuna 가중치 (tr에서만 최적화)
        def obj(trial):
            w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(M)])
            if w.sum() == 0:
                return 0.0
            w = w / w.sum()
            return balanced_accuracy_score(ytr, np.tensordot(w, Ptr, 1).argmax(1))
        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=fi))
        st.optimize(obj, n_trials=n_trials, n_jobs=1)
        w = np.array([st.best_params[f"w{i}"] for i in range(M)])
        w = w / w.sum()
        w_all.append(w)
        res["optuna"].append(
            balanced_accuracy_score(yva, np.tensordot(w, Pva, 1).argmax(1)))

        # 2) Rank Averaging
        R = np.stack([np.stack([rankdata(P[m, :, c]) for c in range(3)], 1)
                      for m in range(M)]) / len(y)
        res["rank"].append(balanced_accuracy_score(yva, R.mean(0)[va].argmax(1)))

        # 3) Power Averaging (k는 tr에서 선택)
        best_k, best_s = 1.0, -1
        for k in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            g = (np.clip(Ptr, 1e-7, 1) ** k).mean(0) ** (1 / k)
            s = balanced_accuracy_score(ytr, g.argmax(1))
            if s > best_s:
                best_s, best_k = s, k
        g = (np.clip(Pva, 1e-7, 1) ** best_k).mean(0) ** (1 / best_k)
        res["power"].append(balanced_accuracy_score(yva, g.argmax(1)))

        # 기준선 N9
        b = 0.15 * n5[va] + 0.57 * rmlp[va] + 0.28 * K[va]
        res["n9"].append(balanced_accuracy_score(yva, b.argmax(1)))
        print(f"fold{fi}: optuna={res['optuna'][-1]:.5f} rank={res['rank'][-1]:.5f} "
              f"power={res['power'][-1]:.5f}(k={best_k}) n9={res['n9'][-1]:.5f}",
              flush=True)

    base = np.mean(res["n9"])
    print(f"\n=== honest 5-fold (baseline N9={base:.5f}) ===")
    for k in ("optuna", "rank", "power"):
        m = np.mean(res[k])
        print(f"  {k:7s} = {m:.5f} (std {np.std(res[k]):.5f}) gain {m-base:+.5f}")
    wm = np.mean(w_all, 0)
    print("optuna mean weights:", {n: round(float(v), 3) for n, v in zip(names, wm)})
    print(f"time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
