"""V2 OOF 오답 분석 — 어디서 점수가 새는가

- 세그먼트별 balanced accuracy: 핵심 3피처(수면·스트레스·활동) 결측 여부 조합
- 혼동행렬: 어떤 클래스를 어떤 클래스로 틀리나
- 결론이 '결측이 병목'이면 → 결측 전용 피처 설계로 직행

실행: python src/error_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from config import DATA, OOF, CLASSES, TARGET

SEEDS = [42, 2026, 7, 101, 777]


def main():
    train = pd.read_csv(DATA / "train.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()

    oof = sum(np.load(OOF / f"v2_C_s{s}_oof.npy") for s in SEEDS) / len(SEEDS)
    pred = oof.argmax(1)
    print(f"blend OOF balanced_acc = {balanced_accuracy_score(y, pred):.5f}\n")

    print("=== confusion matrix (row=true, col=pred) [at-risk, unhealthy, fit] ===")
    cm = confusion_matrix(y, pred)
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    rec = cm.diagonal() / cm.sum(1)
    print("per-class recall:", {c: round(r, 4) for c, r in zip(CLASSES, rec)}, "\n")

    # 세그먼트: 핵심 3피처 결측 여부
    ms = train["sleep_duration"].isna()
    mt = train["stress_level"].isna()
    ma = train["physical_activity_level"].isna()

    segs = {
        "all_known (3피처 완전)": ~(ms | mt | ma),
        "miss_sleep only":  ms & ~mt & ~ma,
        "miss_stress only": ~ms & mt & ~ma,
        "miss_activity only": ~ms & ~mt & ma,
        "miss 2+ of core":  (ms.astype(int) + mt.astype(int) + ma.astype(int)) >= 2,
    }
    print("=== segment balanced accuracy ===")
    rows = []
    for name, m in segs.items():
        m = m.to_numpy()
        if m.sum() == 0:
            continue
        ba = balanced_accuracy_score(y[m], pred[m])
        err = (y[m] != pred[m]).mean()
        rows.append(dict(segment=name, n=int(m.sum()), share=f"{m.mean()*100:.1f}%",
                         bal_acc=round(ba, 4), err_rate=f"{err*100:.2f}%"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # 전체 오답 중 세그먼트 기여율
    wrong = y != pred
    print(f"\ntotal wrong: {wrong.sum():,} ({wrong.mean()*100:.2f}%)")
    for name, m in segs.items():
        m = m.to_numpy()
        contrib = (wrong & m).sum() / wrong.sum() * 100
        print(f"  {name:28s} 오답 기여 {contrib:5.1f}% (세그먼트 크기 {m.mean()*100:4.1f}%)")


if __name__ == "__main__":
    main()
