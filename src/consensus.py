"""N6 - External Consensus: 상위 공개 제출 CSV들 + 우리 모델 하드보팅

- 행별 최빈값. 동률 tie-break: ①우리 예측 ②최고 LB 소스
- 진단: 소스 간 일치율 행렬, 우리 vs 컨센서스 차이 행 수
- ⚠️ CV 검증 불가 (public LB 추종 성격) — 최종 2개 헷지의 'public 카드'로만 사용

실행: python src/consensus.py <우리_제출.csv>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA, SUBS, CLASSES, TARGET, ID

ROOT = Path(__file__).resolve().parents[1]
CDIR = ROOT / "consensus_src"

# (경로, 라벨, public LB) — LB 내림차순 = tie-break 우선순위
SOURCES = [
    ("artkomissar_s6e7-external-consensus-mf-lb-0-95114/submission.csv",  "consensus114", 0.95114),
    ("hikari30_s6e7-external-ensemble-lb-chase-lb0-95113/submission.csv", "ensemble113",  0.95113),
    ("amanatar_s6e7-student-hearth-risk-lb-0-95112/submission.csv",       "hearth112",    0.95112),
    ("anhadmahajan06_s6e7-post-processing-ensemble-lb-0-95112/submission.csv", "postproc112", 0.95112),
    ("nawfeelrahman1124444_ps-s6-ep6-realmlp-0-95090/submission.csv",     "realmlp090",   0.95090),
    ("vad13irt_ps-s6e7-eda-ensemble-lb-0-95075/submission.csv",           "edaens075",    0.95075),
]


def main(ours_path):
    ss = pd.read_csv(DATA / "sample_submission.csv")
    ours = pd.read_csv(ours_path)
    assert ours[ID].tolist() == ss[ID].tolist(), "우리 파일 id 불일치"

    preds = {"OURS": ours[TARGET].to_numpy()}
    for rel, label, lb in SOURCES:
        df = pd.read_csv(CDIR / rel)
        assert len(df) == len(ss), f"{label} 행수 불일치"
        df = df.set_index(ID).loc[ss[ID]].reset_index()      # id 순서 정렬
        assert set(df[TARGET].unique()) <= set(CLASSES), f"{label} 라벨 이상"
        preds[label] = df[TARGET].to_numpy()

    names = list(preds)
    P = pd.DataFrame(preds)

    print("=== 소스 간 일치율 (%) ===")
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            agree.loc[a, b] = (P[a] == P[b]).mean() * 100
    print(agree.round(2).to_string())

    # 하드보팅: 최빈값. 동률 시 OURS -> 최고 LB 순으로 tie-break
    order = ["OURS"] + [lab for _, lab, _ in SOURCES]        # tie-break 우선순위
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    votes = np.zeros((len(P), len(CLASSES)), dtype=int)
    for n in names:
        col = P[n].map(cls_idx).to_numpy()
        votes[np.arange(len(P)), col] += 1
    maxv = votes.max(1)
    final = np.empty(len(P), dtype=object)
    tie_rows = 0
    for i in range(len(P)):
        winners = [CLASSES[k] for k in range(3) if votes[i, k] == maxv[i]]
        if len(winners) == 1:
            final[i] = winners[0]
        else:
            tie_rows += 1
            for src in order:                               # tie-break
                if P[src].iat[i] in winners:
                    final[i] = P[src].iat[i]
                    break

    changed = (final != P["OURS"].to_numpy()).sum()
    print(f"\n동률 행: {tie_rows:,} | 우리 예측과 달라진 행: {changed:,} "
          f"({changed/len(P)*100:.2f}%)")
    dist = pd.Series(final).value_counts()
    print("컨센서스 분포:", dist.to_dict())

    sub = pd.DataFrame({ID: ss[ID], TARGET: final})
    out = SUBS / "N6 - external consensus.csv"
    sub.to_csv(out, index=False)
    assert len(sub) == len(ss) and sub[TARGET].isna().sum() == 0
    print(f"saved {out} | sanity OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SUBS / "v2_hgb_te_5seed.csv")
