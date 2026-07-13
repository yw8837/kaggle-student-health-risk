"""LR-on-logits 결합 — S6E4/S6E6 우승 공식의 재현 (21멤버 라이브러리)

- 각 멤버의 3클래스 확률 -> 로짓 변환 -> LogisticRegression(class_weight='balanced')
- 정직 평가: 행 5분할 (4/5 적합, 1/5 평가) — 비교 기준 N7(0.95084)
- 주의: 멤버들의 원 폴드 구조가 상이 (자체 OOF 규율은 각자 담보) — 행분할 평가로 완화
- 전체 OOF 적합 -> test 로짓에 적용 = N8 후보 생성

실행: python src/stacker_logits.py
출력: oof/n8_test.npy, subs/N8 - logit stack.csv, 로그에 honest 점수
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, SUBS, CLASSES, TARGET, ID

BASE = OOF.parent / "consensus_src"
EPS = 1e-7


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_members():
    """OOF/test 확률 쌍 로드. 공개 CSV는 [at-risk,fit,unhealthy] -> 재정렬."""
    mem = {}

    def add(name, oof, test):
        if oof is not None and test is not None and len(oof) == 690088:
            mem[name] = (oof, test)

    add("N5", np.load(OOF / "n5_oof.npy"), np.load(OOF / "n5_test.npy"))
    add("NN", np.load(OOF / "nn_s42_oof.npy"), np.load(OOF / "nn_s42_test.npy"))
    p = OOF / "v2_C_cat_s42_oof.npy"
    if p.exists():
        add("CAT", np.load(p), np.load(OOF / "v2_C_cat_s42_test.npy"))

    def csv_pair(d, opat, tpat, reorder):
        po = list(d.glob(opat)); pt = list(d.glob(tpat))
        if not (po and pt):
            return None, None
        o = pd.read_csv(po[0]); t = pd.read_csv(pt[0])
        idc = "id" if "id" in o.columns else o.columns[0]
        o = o.sort_values(idc); t = t.sort_values("id" if "id" in t.columns else t.columns[0])
        cols = ["at-risk", "fit", "unhealthy"]
        return o[cols].to_numpy()[:, reorder], t[cols].to_numpy()[:, reorder]

    d = BASE / "nawfeelrahman1124444_ps-s6-ep6-realmlp-0-95090"
    o, t = csv_pair(d, "oof_preds.csv", "test_preds.csv", [0, 2, 1])
    add("RMLP", o, t)
    yz = BASE / "yunsuxiaozi_pss6e7-realmlp-cv-0-95063"
    if (yz / "oof_predictions.npy").exists():
        add("YZMLP", np.load(yz / "oof_predictions.npy")[:, [0, 2, 1]],
            np.load(yz / "test_predictions.npy")[:, [0, 2, 1]])
    for d in sorted(BASE.glob("masayakawamata_s6e7-*")):
        for f in d.glob("oof_*.csv"):
            tag = f.stem.replace("oof_", "").upper()
            tf = d / f.name.replace("oof_", "testpred_")
            o, t = csv_pair(d, f.name, tf.name, [0, 2, 1])
            add(f"MK_{tag}", o, t)
    return mem


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=[TARGET])
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    mem = load_members()
    print(f"members: {len(mem)}")
    for n, (o, _) in sorted(mem.items(), key=lambda kv: -balanced_accuracy_score(y, kv[1][0].argmax(1))):
        print(f"  {n:18s} solo={balanced_accuracy_score(y, o.argmax(1)):.5f}")

    Xo = np.hstack([logit(o) for o, _ in mem.values()])
    Xt = np.hstack([logit(t) for _, t in mem.values()])
    print("stack feature dim:", Xo.shape[1])

    # 정직 평가 (행 5분할)
    kf = KFold(5, shuffle=True, random_state=0)
    honest, base_n7 = [], []
    n5, rmlp = mem["N5"][0], mem["RMLP"][0]
    for tr, va in kf.split(y):
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        lr.fit(Xo[tr], y[tr])
        honest.append(balanced_accuracy_score(y[va], lr.predict(Xo[va])))
        base_n7.append(balanced_accuracy_score(
            y[va], (0.405 * n5[va] + 0.595 * rmlp[va]).argmax(1)))
    print(f"honest stack = {np.mean(honest):.5f} (fold std {np.std(honest):.5f})")
    print(f"vs N7 {np.mean(base_n7):.5f} -> gain {np.mean(honest)-np.mean(base_n7):+.5f}")

    # 전체 적합 -> N8 후보
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(Xo, y)
    pt = lr.predict_proba(Xt)
    np.save(OOF / "n8_test.npy", pt)
    oof_full = lr.predict_proba(Xo)          # (자기적합 — 기록용, 평가는 honest 사용)
    np.save(OOF / "n8_oof_selffit.npy", oof_full)

    ss = pd.read_csv(DATA / "sample_submission.csv")
    inv = {i: c for i, c in enumerate(CLASSES)}
    sub = pd.DataFrame({ID: ss[ID], TARGET: [inv[i] for i in pt.argmax(1)]})
    sub.to_csv(SUBS / "N8 - logit stack.csv", index=False)
    assert len(sub) == len(ss) and sub[TARGET].isna().sum() == 0
    print("saved subs/N8 - logit stack.csv | dist", sub[TARGET].value_counts().to_dict())


if __name__ == "__main__":
    main()
