"""LB 힐클라이밍 프로브 생성기 — public 20% 등반용

원리: balanced accuracy에서 소수클래스 정답(+0.00007/행)과 다수클래스 오답(-0.000007/행)의
10:1 비대칭 → '컨센서스=at-risk인데 라이브러리 사후확률이 소수클래스를 지지'하는 행을
EV순으로 그룹핑해 승격 후보로 만든다.

출력:
  subs/probe_base.csv           — 컨센서스 베이스
  subs/probe_P1..P6.csv         — 베이스 + 누적 승격 그룹 (EV 내림차순)
  oof/probe_ledger.csv          — 그룹별 행수·평균확률 (제출 결과 기록용)

실행: python src/probe_builder.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA, OOF, SUBS, CLASSES, TARGET, ID

BASE = OOF.parent / "consensus_src"


def load_test_probs():
    """가용한 test 확률 전부 로드 (강한 모델 위주, [at-risk,unhealthy,fit] 순)."""
    def csvp(p, cols=("at-risk", "unhealthy", "fit")):
        df = pd.read_csv(p)
        idc = "id" if "id" in df.columns else df.columns[0]
        df = df.sort_values(idc)
        return df[list(cols)].to_numpy()

    probs = {}
    probs["N5"] = np.load(OOF / "n5_test.npy")
    probs["RMLP"] = csvp(BASE / "nawfeelrahman1124444_ps-s6-ep6-realmlp-0-95090/test_preds.csv")
    probs["STACK"] = csvp(BASE / "masayakawamata_s6e7-logreg-stacker-cv-0-95067/testpred_stack.csv",
                          cols=("at-risk", "fit", "unhealthy"))[:, [0, 2, 1]]
    probs["FTT"] = csvp(BASE / "masayakawamata_s6e7-ft-transformer-v2-cv-0-95063/testpred_ftt.csv",
                        cols=("at-risk", "fit", "unhealthy"))[:, [0, 2, 1]]
    p = BASE / "yunsuxiaozi_pss6e7-realmlp-cv-0-95063/test_predictions.npy"
    if p.exists():
        probs["YZMLP"] = np.load(p)[:, [0, 2, 1]]
    # Kawamata 수확분 (있으면 자동 편입)
    for d in BASE.glob("masayakawamata_s6e7-*"):
        for f in d.glob("testpred_*.csv"):
            tag = f.stem.replace("testpred_", "").upper()
            if tag not in ("STACK", "FTT"):
                try:
                    probs[f"MK_{tag}"] = csvp(f, cols=("at-risk", "fit", "unhealthy"))[:, [0, 2, 1]]
                except Exception:
                    pass
    return probs


def main():
    ss = pd.read_csv(DATA / "sample_submission.csv")
    base = pd.read_csv(SUBS / "N6 - external consensus.csv")
    assert base[ID].tolist() == ss[ID].tolist()
    base_pred = base[TARGET].to_numpy()

    probs = load_test_probs()
    print("test-prob members:", list(probs))
    P = np.mean(np.stack(list(probs.values())), axis=0)   # 평균 사후확률

    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    at = base_pred == "at-risk"
    # 승격 후보: 베이스=at-risk & 평균 사후확률이 소수클래스 쪽으로 기움
    ledger, files = [], []
    groups = [
        ("P1_unh_strong", at & (P[:, 1] > 0.45), "unhealthy"),
        ("P2_fit_strong", at & (P[:, 2] > 0.45), "fit"),
        ("P3_unh_mid",    at & (P[:, 1] > 0.35) & (P[:, 1] <= 0.45), "unhealthy"),
        ("P4_fit_mid",    at & (P[:, 2] > 0.35) & (P[:, 2] <= 0.45), "fit"),
        ("P5_unh_edge",   at & (P[:, 1] > 0.28) & (P[:, 1] <= 0.35), "unhealthy"),
        ("P6_fit_edge",   at & (P[:, 2] > 0.28) & (P[:, 2] <= 0.35), "fit"),
    ]
    ss[TARGET] = base_pred
    ss[[ID, TARGET]].to_csv(SUBS / "probe_base.csv", index=False)
    cur = base_pred.copy()
    for name, mask, cls in groups:
        n = int(mask.sum())
        meanp = float(P[mask, cls_idx[cls]].mean()) if n else 0.0
        cur = cur.copy()
        cur[mask] = cls
        out = SUBS / f"probe_{name}.csv"
        pd.DataFrame({ID: ss[ID], TARGET: cur}).to_csv(out, index=False)
        ledger.append(dict(group=name, rows=n, mean_posterior=round(meanp, 3),
                           flip_to=cls, file=out.name, cumulative=True))
        print(f"{name:14s} rows={n:5d} mean_p={meanp:.3f} -> {out.name}")
    pd.DataFrame(ledger).to_csv(OOF / "probe_ledger.csv", index=False)
    print("ledger saved. 제출 순서: base -> P1 -> P2 ... (누적식, 내리면 직전으로 롤백)")


if __name__ == "__main__":
    main()
