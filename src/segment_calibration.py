"""세그먼트별 결정 보정 — 결측 패턴마다 클래스 가중 argmax를 따로 최적화

이론: 결측 행은 라벨의 해당 조건이 원리상 관측 불가 → 남는 건 '어느 쪽으로 찍는 게
balanced accuracy 기대값에 유리한가'의 결정 문제. 최적 가중치는 세그먼트마다 다르다.
(글로벌 가중 보정은 커뮤니티에서 ~0 확인 — 세그먼트별은 미개척)

방법:
- 상호배타 세그먼트 5개 (결측 패턴 파티션)
- 세그먼트별 (w_unhealthy, w_fit) 그리드, w_at-risk=1 고정 — 좌표하강 2라운드
- 목적함수는 '전체' balanced accuracy (세그먼트 간 상호작용 반영)
- **중첩 검증**: 행 5분할, 4/5 튜닝 → 1/5 정직 평가. 채택 기준 +0.0005

실행: python src/segment_calibration.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, SUBS, CLASSES, TARGET, ID

SEEDS = [42, 2026, 7, 101, 777]
GRID = np.linspace(0.5, 2.2, 18)


def get_segments(df):
    """상호배타 결측 패턴 파티션 (train/test 공통 로직)."""
    ms = df["sleep_duration"].isna().to_numpy()
    mt = df["stress_level"].isna().to_numpy()
    ma = df["physical_activity_level"].isna().to_numpy()
    n_miss = ms.astype(int) + mt.astype(int) + ma.astype(int)
    segs = {
        "none":       n_miss == 0,
        "sleep":      ms & (n_miss == 1),
        "stress":     mt & (n_miss == 1),
        "activity":   ma & (n_miss == 1),
        "multi":      n_miss >= 2,
    }
    assert sum(m.sum() for m in segs.values()) == len(df)
    return segs


def apply_weights(P, segs, W):
    """세그먼트별 가중 argmax."""
    pred = np.empty(len(P), dtype=int)
    for name, m in segs.items():
        w = np.array([1.0, W[name][0], W[name][1]])
        pred[m] = (P[m] * w).argmax(1)
    return pred


def tune(P, y, segs, rows_mask, rounds=2):
    """rows_mask 행만 목적함수에 사용해 세그먼트별 가중치 좌표하강."""
    W = {k: (1.0, 1.0) for k in segs}
    for _ in range(rounds):
        for name in segs:
            best = (balanced_accuracy_score(
                y[rows_mask], apply_weights(P, segs, W)[rows_mask]), W[name])
            for wu in GRID:
                for wf in GRID:
                    W_try = {**W, name: (wu, wf)}
                    s = balanced_accuracy_score(
                        y[rows_mask], apply_weights(P, segs, W_try)[rows_mask])
                    if s > best[0]:
                        best = (s, (wu, wf))
            W[name] = best[1]
    return W


def main():
    train = pd.read_csv(DATA / "train.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()
    P = sum(np.load(OOF / f"v2_C_s{s}_oof.npy") for s in SEEDS) / len(SEEDS)
    segs = get_segments(train)

    base = balanced_accuracy_score(y, P.argmax(1))
    print(f"base (argmax) OOF = {base:.5f}")

    # 중첩 검증: 4/5 튜닝 -> 1/5 평가
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    honest, base_folds = [], []
    for tr, va in kf.split(y):
        tr_mask = np.zeros(len(y), bool); tr_mask[tr] = True
        va_mask = np.zeros(len(y), bool); va_mask[va] = True
        W = tune(P, y, segs, tr_mask)
        honest.append(balanced_accuracy_score(y[va_mask], apply_weights(P, segs, W)[va_mask]))
        base_folds.append(balanced_accuracy_score(y[va_mask], P[va_mask].argmax(1)))
    gain = np.mean(honest) - np.mean(base_folds)
    print(f"nested honest: calibrated={np.mean(honest):.5f} vs base={np.mean(base_folds):.5f} "
          f"-> gain={gain:+.5f} (기준 +0.0005)")

    if gain < 0.0005:
        print("판정: 기각 — argmax 유지 (제출물 변경 없음)")
        return

    # 채택 시: 전체 OOF로 최종 가중치 적합 -> test 적용
    W = tune(P, y, segs, np.ones(len(y), bool))
    print("최종 세그먼트 가중치:", {k: (round(v[0], 2), round(v[1], 2)) for k, v in W.items()})
    T = sum(np.load(OOF / f"v2_C_s{s}_test.npy") for s in SEEDS) / len(SEEDS)
    test_df = pd.read_csv(DATA / "test.csv")
    segs_test = get_segments(test_df)
    pred = apply_weights(T, segs_test, W)
    inv = {i: c for i, c in enumerate(CLASSES)}
    sub = pd.DataFrame({ID: test_df[ID], TARGET: [inv[i] for i in pred]})
    out = SUBS / "n4_segment_calibrated.csv"
    sub.to_csv(out, index=False)
    ss = pd.read_csv(DATA / "sample_submission.csv")
    assert len(sub) == len(ss) and sub[ID].tolist() == ss[ID].tolist()
    print(f"saved {out} | dist {sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
