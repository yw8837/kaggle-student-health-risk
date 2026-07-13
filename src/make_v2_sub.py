"""V2 최종: C 변형 다중시드 평균 -> OOF 검증 -> 제출파일 생성

실행: python src/make_v2_sub.py
출력: subs/v2_hgb_te_5seed.csv + 블렌드 OOF balanced_acc
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from config import DATA, OOF, SUBS, CLASSES, TARGET, ID

SEEDS = [42, 2026, 7, 101, 777]


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=[TARGET])
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()

    oof_sum, test_sum, used = None, None, []
    for s in SEEDS:
        po, pt = OOF / f"v2_C_s{s}_oof.npy", OOF / f"v2_C_s{s}_test.npy"
        if not (po.exists() and pt.exists()):
            print(f"seed {s}: MISSING, skipped")
            continue
        o, t = np.load(po), np.load(pt)
        print(f"seed {s}: solo OOF={balanced_accuracy_score(y, o.argmax(1)):.5f}")
        oof_sum = o if oof_sum is None else oof_sum + o
        test_sum = t if test_sum is None else test_sum + t
        used.append(s)

    assert used, "no seed results found"
    oof, test = oof_sum / len(used), test_sum / len(used)
    cv = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\n[V2 blend x{len(used)} seeds] OOF balanced_acc = {cv:.5f}")

    inv = {i: c for i, c in enumerate(CLASSES)}
    ids = pd.read_csv(DATA / "test.csv", usecols=[ID])[ID]
    sub = pd.DataFrame({ID: ids, TARGET: [inv[i] for i in test.argmax(1)]})
    out = SUBS / "v2_hgb_te_5seed.csv"
    sub.to_csv(out, index=False)

    ss = pd.read_csv(DATA / "sample_submission.csv")
    assert len(sub) == len(ss) and list(sub.columns) == list(ss.columns)
    assert sub[ID].tolist() == ss[ID].tolist()
    assert set(sub[TARGET]) <= set(CLASSES) and sub[TARGET].isna().sum() == 0
    print(f"saved {out} | dist {sub[TARGET].value_counts().to_dict()} | sanity OK")


if __name__ == "__main__":
    main()
