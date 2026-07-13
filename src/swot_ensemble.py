"""SWOT 앙상블 — 모델×세그먼트 성능 매트릭스 + 상보성 정량화 + 가중 소프트 보팅

1. SWOT 매트릭스: 모델별 강/약 세그먼트 (결측 패턴 기준)
2. 상보성: 모델 쌍별 오답 겹침률·오답 상관 (낮을수록 보팅 이득)
3. 가중치 서치: 심플렉스 위 가중 평균, **중첩 검증** (OOF 5분할: 4/5 튜닝 → 1/5 평가)
   - 가중치 집중도 보고: 한 모델 95%+ 집중이면 보팅 무의미(솔로 채택)

실행: python src/swot_ensemble.py
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from config import DATA, OOF, CLASSES, TARGET

MODELS = {          # tag -> oof/test 파일 프리픽스 (seed 42, 동일 폴드)
    "HGB+TE":  "v2_C_s42",
    "CatBoost": "v2_C_cat_s42",
    "LGBM":    "v2_C_lgbm_s42",
}


def load():
    train = pd.read_csv(DATA / "train.csv")
    mapping = {c: i for i, c in enumerate(CLASSES)}
    y = train[TARGET].map(mapping).to_numpy()
    oofs, tests = {}, {}
    for name, tag in MODELS.items():
        po = OOF / f"{tag}_oof.npy"
        if not po.exists():
            print(f"{name}: missing, skipped")
            continue
        oofs[name] = np.load(po)
        tests[name] = np.load(OOF / f"{tag}_test.npy")
    segs = {
        "all_known":   ~(train.sleep_duration.isna() | train.stress_level.isna()
                         | train.physical_activity_level.isna()),
        "miss_sleep":  train.sleep_duration.isna(),
        "miss_stress": train.stress_level.isna(),
        "miss_activity": train.physical_activity_level.isna(),
    }
    segs = {k: v.to_numpy() for k, v in segs.items()}
    return y, oofs, tests, segs


def swot_matrix(y, oofs, segs):
    rows = []
    for name, p in oofs.items():
        pred = p.argmax(1)
        row = {"model": name, "overall": balanced_accuracy_score(y, pred)}
        for sname, m in segs.items():
            row[sname] = balanced_accuracy_score(y[m], pred[m])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model").round(4)
    print("=== SWOT: model x segment balanced accuracy ===")
    print(df.to_string())
    return df


def complementarity(y, oofs):
    names = list(oofs)
    print("\n=== 상보성: 오답 겹침 (jaccard) / 오답 상관 ===")
    wrongs = {n: (oofs[n].argmax(1) != y) for n in names}
    for a, b in itertools.combinations(names, 2):
        wa, wb = wrongs[a], wrongs[b]
        jac = (wa & wb).sum() / max((wa | wb).sum(), 1)
        corr = np.corrcoef(wa, wb)[0, 1]
        agree = (oofs[a].argmax(1) == oofs[b].argmax(1)).mean()
        print(f"{a:9s} vs {b:9s} | 오답겹침 {jac:.3f} | 오답상관 {corr:.3f} | 예측일치 {agree*100:.2f}%")


def weight_search(y, oofs, n_grid=21):
    """심플렉스 그리드 서치 + 중첩 검증 (행 5분할)."""
    names = list(oofs)
    P = np.stack([oofs[n] for n in names])          # (M, n, 3)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    honest, weights_list = [], []
    grid = np.linspace(0, 1, n_grid)
    combos = [w for w in itertools.product(grid, repeat=len(names))
              if abs(sum(w) - 1) < 1e-9]
    for tr, va in kf.split(y):
        best_w, best_s = None, -1
        for w in combos:
            blend = np.tensordot(np.array(w), P[:, tr], axes=1)
            s = balanced_accuracy_score(y[tr], blend.argmax(1))
            if s > best_s:
                best_s, best_w = s, w
        blend_va = np.tensordot(np.array(best_w), P[:, va], axes=1)
        honest.append(balanced_accuracy_score(y[va], blend_va.argmax(1)))
        weights_list.append(best_w)
    w_mean = np.mean(weights_list, axis=0)
    print("\n=== 가중 소프트 보팅 (중첩 검증) ===")
    for n, w in zip(names, w_mean):
        print(f"  {n:9s} weight = {w:.3f}")
    solo_best = max(balanced_accuracy_score(y, oofs[n].argmax(1)) for n in names)
    print(f"honest blend bal_acc = {np.mean(honest):.5f} (folds std {np.std(honest):.5f})")
    print(f"best solo            = {solo_best:.5f}")
    print(f"honest gain          = {np.mean(honest) - solo_best:+.5f}  (채택 기준 +0.0005)")
    conc = w_mean.max()
    print(f"가중치 집중도 = {conc:.2f} {'→ 사실상 솔로 (보팅 무의미)' if conc > 0.9 else '→ 실질 분산 (상보성 있음)'}")
    return dict(zip(names, w_mean)), np.mean(honest)


def main():
    y, oofs, tests, segs = load()
    if len(oofs) < 2:
        print("모델 2개 미만 — OOF 파일 생성 후 재실행")
        return
    swot_matrix(y, oofs, segs)
    complementarity(y, oofs)
    weight_search(y, oofs)


if __name__ == "__main__":
    main()
