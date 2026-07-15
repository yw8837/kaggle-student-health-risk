"""모델 수 최적화 분석 — "3/4/5모델 중 뭐가 최적?"에 정량 답

1) 전 멤버 솔로 bacc
2) 페어와이즈 argmax 일치율 (다양성 실측)
3) greedy 전진선택: 멤버 1개→K개 늘리며 중첩검증 honest 곡선 → 플래토 지점 = 최적 모델 수
4) 오라클 상한: 표본별 최적 멤버 선택 시 bacc (느슨한 천장)

실행: python src/model_count_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, CLASSES, TARGET
from unified_ensemble import load_unified, m_weight


def main():
    train = pd.read_csv(DATA / "train.csv", usecols=[TARGET])
    y_full = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    prior = np.bincount(y_full) / len(y_full)
    print("=== 멤버 로드 ===", flush=True)
    lib = load_unified(y_full, prior, min_solo=0.945)
    names = list(lib)
    solo = {n: balanced_accuracy_score(y_full, lib[n][0].argmax(1)) for n in names}
    # 속도: 층화 150k 서브샘플로 탐색 (솔로/일치율은 풀데이터)
    rng = np.random.default_rng(0)
    idx = np.sort(np.concatenate([
        rng.choice(np.where(y_full == c)[0],
                   int(150_000 * (y_full == c).mean()), replace=False)
        for c in range(3)]))
    y = y_full[idx]
    P = np.stack([lib[n][0][idx] for n in names])
    P_full = np.stack([lib[n][0] for n in names])

    print("\n=== 1) 솔로 bacc (풀데이터) ===", flush=True)
    for n in sorted(solo, key=solo.get, reverse=True):
        print(f"  {n:16s} {solo[n]:.5f}")

    print("\n=== 2) 페어와이즈 argmax 일치율 (풀데이터) ===", flush=True)
    A = np.stack([P_full[m].argmax(1) for m in range(len(names))])
    top = sorted(range(len(names)), key=lambda i: solo[names[i]], reverse=True)[:8]
    hdr = "".join(f"{names[i][:9]:>10s}" for i in top)
    print(" " * 16 + hdr)
    for i in top:
        row = "".join(f"{(A[i] == A[j]).mean():10.4f}" for j in top)
        print(f"  {names[i]:14s}{row}")

    print("\n=== 3) greedy 전진선택 (150k 서브샘플, 중첩검증 honest) ===", flush=True)
    kf = KFold(5, shuffle=True, random_state=0)
    splits = list(kf.split(y))
    pool = sorted(range(len(names)), key=lambda i: solo[names[i]], reverse=True)[:10]
    selected = [pool[0]]
    print(f"  k=1: {names[selected[0]]:14s} honest={solo[names[selected[0]]]:.5f}",
          flush=True)
    prev = solo[names[selected[0]]]
    for k in range(2, 8):
        best_add, best_h = None, -1
        for c in pool:
            if c in selected:
                continue
            sub = P[selected + [c]]
            h = np.mean([m_weight(sub, y, tr, va, n_draw=300)[0]
                         for tr, va in splits[:2]])
            if h > best_h:
                best_h, best_add = h, c
        sub = P[selected + [best_add]]
        h5 = np.mean([m_weight(sub, y, tr, va, n_draw=1200)[0] for tr, va in splits])
        selected.append(best_add)
        print(f"  k={k}: +{names[best_add]:14s} honest={h5:.5f} "
              f"(gain {h5 - prev:+.5f})", flush=True)
        prev = h5

    print("\n=== 4) 오라클 상한 (표본별 최적 멤버, 풀데이터) ===")
    correct_any = np.zeros(len(y_full), bool)
    for m in range(len(names)):
        correct_any |= (A[m] == y_full)
    ub = np.mean([correct_any[y_full == c].mean() for c in range(3)])
    print(f"  oracle balanced acc 상한 = {ub:.5f}")


if __name__ == "__main__":
    main()
