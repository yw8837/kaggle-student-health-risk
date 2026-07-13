"""통합 앙상블 — 스케일 자동통일 후 4방법 5-fold OOF 비교 (지시사항 전량 반영)

핵심 발견: 공개 OOF들이 서로 다른 스케일 (raw posterior vs tilted).
 - tilted 멤버(class_weight로 학습, argmax가 이미 balanced-acc 최적): N5, RMLP, ...
 - raw 멤버(원확률, prior로 나눠야 tilt): XGB_OvR, LGBM, ... (MK 계열)
자동감지: raw argmax balanced_acc>0.94면 tilted 유지, 아니면 prior-correction으로 통일.

평가 4방법 (전부 KFold(5) 행분할 중첩검증, honest):
 1) 최적 가중치 탐색 (Dirichlet 랜덤서치 + 좌표정제)
 2) 순위 평균 (Rank Averaging) — 클래스별 표본순위 평균
 3) 거듭제곱 앙상블 (Power Averaging) — 멱평균 k 그리드
 4) LR-on-logits 스택 (S6E4/S6E6 우승공식)
비교 기준: N7 2-way (0.95084)

실행: python src/unified_ensemble.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, CLASSES, TARGET

BASE = OOF.parent / "consensus_src"
EPS = 1e-7


def prior_correct(P, prior):
    q = np.clip(P, EPS, 1) / prior
    return q / q.sum(1, keepdims=True)


def load_unified(y, prior, min_solo=0.945):
    """모든 멤버 로드 -> 스케일 자동통일(tilted) -> min_solo 이상만 채택."""
    raw = {}

    def add(name, oof, test):
        if oof is None or len(oof) != len(y):
            return
        raw[name] = (np.clip(oof, EPS, 1), None if test is None else np.clip(test, EPS, 1))

    add("N5", np.load(OOF / "n5_oof.npy"), np.load(OOF / "n5_test.npy"))
    add("NN", np.load(OOF / "nn_s42_oof.npy"), np.load(OOF / "nn_s42_test.npy"))
    if (OOF / "v2_C_cat_s42_oof.npy").exists():
        add("CAT", np.load(OOF / "v2_C_cat_s42_oof.npy"), np.load(OOF / "v2_C_cat_s42_test.npy"))

    def csvp(path, order=("at-risk", "unhealthy", "fit")):
        df = pd.read_csv(path)
        idc = "id" if "id" in df.columns else df.columns[0]
        return df.sort_values(idc)[list(order)].to_numpy()

    d = BASE / "nawfeelrahman1124444_ps-s6-ep6-realmlp-0-95090"
    add("RMLP", csvp(d / "oof_preds.csv"), csvp(d / "test_preds.csv"))
    yz = BASE / "yunsuxiaozi_pss6e7-realmlp-cv-0-95063"
    if (yz / "oof_predictions.npy").exists():
        add("YZMLP", np.load(yz / "oof_predictions.npy")[:, [0, 2, 1]],
            np.load(yz / "test_predictions.npy")[:, [0, 2, 1]])
    for d in sorted(BASE.glob("masayakawamata_s6e7-*")):
        for f in d.glob("oof_*.csv"):
            tf = d / f.name.replace("oof_", "testpred_")
            if not tf.exists():
                continue
            tag = "MK_" + f.stem.replace("oof_", "").upper()
            try:
                add(tag, csvp(f), csvp(tf))
            except Exception:
                pass

    # 스케일 통일 + 채택 필터
    lib = {}
    for name, (o, t) in raw.items():
        s_raw = balanced_accuracy_score(y, o.argmax(1))
        if s_raw >= 0.94:                       # 이미 tilted
            oo, tt, scale = o, t, "tilted"
        else:
            oo = prior_correct(o, prior)
            tt = prior_correct(t, prior) if t is not None else None
            scale = "prior-corrected"
        s = balanced_accuracy_score(y, oo.argmax(1))
        if s >= min_solo:
            lib[name] = (oo, tt)
        print(f"  {name:16s} raw={s_raw:.5f} -> {scale:16s} unified={s:.5f} "
              f"{'ADOPT' if s >= min_solo else 'drop'}")
    return lib


# ---------- 4 방법 ----------
def m_weight(P, y, tr, va, seed=1, n_draw=2500):
    rng = np.random.default_rng(seed)
    M = P.shape[0]
    def sc(w, idx): return balanced_accuracy_score(y[idx], np.tensordot(w, P[:, idx], axes=1).argmax(1))
    cand = np.vstack([rng.dirichlet(np.ones(M) * 0.6, n_draw), np.eye(M)])
    w = cand[np.argmax([sc(c, tr) for c in cand])]
    best = sc(w, tr)
    for _ in range(2):
        for j in range(M):
            for d in (-0.1, -0.05, 0.05, 0.1):
                w2 = w.copy(); w2[j] = max(0, w2[j] + d); w2 /= w2.sum()
                if sc(w2, tr) > best: best, w = sc(w2, tr), w2
    return sc(w, va), w


def m_rank(P, y, tr, va):
    # 클래스별 표본순위(0~1) 평균 -> argmax. tr에서 멤버 가중은 없음(순수 랭크평균)
    R = np.stack([np.stack([rankdata(P[m, :, c]) for c in range(3)], 1) for m in range(P.shape[0])])
    R = R / R.shape[1]
    avg = R.mean(0)
    return balanced_accuracy_score(y[va], avg[va].argmax(1))


def m_power(P, y, tr, va):
    best_k, best_s = 1.0, -1
    for k in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        g = (P ** k).mean(0) ** (1.0 / k)
        s = balanced_accuracy_score(y[tr], g[tr].argmax(1))
        if s > best_s: best_s, best_k = s, k
    g = (P ** best_k).mean(0) ** (1.0 / best_k)
    return balanced_accuracy_score(y[va], g[va].argmax(1)), best_k


def m_logit_stack(P, y, tr, va):
    L = np.hstack([np.log(P[m] / (1 - P[m])) for m in range(P.shape[0])])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    lr.fit(L[tr], y[tr])
    return balanced_accuracy_score(y[va], lr.predict(L[va]))


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=[TARGET])
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    prior = np.bincount(y) / len(y)
    print("=== 멤버 로드 & 스케일 통일 ===")
    lib = load_unified(y, prior)
    names = list(lib)
    P = np.stack([lib[n][0] for n in names])
    print(f"\n채택 멤버 {len(names)}개")

    kf = KFold(5, shuffle=True, random_state=0)
    res = {k: [] for k in ("weight", "rank", "power", "logit", "n7")}
    n5, rmlp = lib["N5"][0], lib["RMLP"][0]
    for tr, va in kf.split(y):
        res["weight"].append(m_weight(P, y, tr, va)[0])
        res["rank"].append(m_rank(P, y, tr, va))
        res["power"].append(m_power(P, y, tr, va)[0])
        res["logit"].append(m_logit_stack(P, y, tr, va))
        res["n7"].append(balanced_accuracy_score(y[va], (0.405 * n5[va] + 0.595 * rmlp[va]).argmax(1)))
    print("\n=== honest 5-fold 결과 (기준 N7=%.5f) ===" % np.mean(res["n7"]))
    for k in ("weight", "rank", "power", "logit"):
        m = np.mean(res[k]); print(f"  {k:8s} = {m:.5f} (std {np.std(res[k]):.5f}) | gain {m-np.mean(res['n7']):+.5f}")


if __name__ == "__main__":
    main()
