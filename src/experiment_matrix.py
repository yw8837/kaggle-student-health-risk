"""실험 통합 매트릭스 (전기수 장표 포맷) — 모든 실험을 한 표로

출력: oof/experiment_matrix.csv, assets/experiment_matrix.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import OOF

ASSETS = OOF.parent / "assets"

# (실험, 모델, 인코딩, 피처, 불균형, 데이터, 검증, BalAcc, LB, 시간s)
ROWS = [
    # model zoo (층화 10만, 홀드아웃 25%)
    ("zoo", "LogisticRegression", "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.91995, None, 0.8),
    ("zoo", "GaussianNB",         "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.87741, None, 0.2),
    ("zoo", "DecisionTree",       "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.87019, None, 1.4),
    ("zoo", "KNeighbors",         "OHE+Scale", "기본13", "무보정", "100k", "holdout", 0.67031, None, 0.01),
    ("zoo", "RandomForest",       "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.92500, None, 5.2),
    ("zoo", "ExtraTrees",         "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.83340, None, 6.7),
    ("zoo", "MLP",                "OHE+Scale", "기본13", "무보정", "100k", "holdout", 0.87026, None, 27.2),
    ("zoo", "HistGradientBoosting","OHE+Scale","기본13", "가중", "100k", "holdout", 0.95183, None, 4.2),
    ("zoo", "LightGBM",           "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.92082, None, 17.6),
    ("zoo", "XGBoost",            "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.91804, None, 33.2),
    ("zoo", "CatBoost",           "OHE+Scale", "기본13", "가중", "100k", "holdout", 0.94934, None, 38.9),
    # 제출 계열
    ("N1", "LightGBM",  "네이티브 범주", "기본13", "가중", "전체×5fold", "OOF", 0.94962, 0.94952, 335),
    ("V2-A", "HistGB",  "서수",         "기본13", "가중", "전체×5fold", "OOF", 0.94945, None, 112),
    ("V2-B", "HistGB",  "서수",         "+규칙피처", "가중", "전체×5fold", "OOF", 0.94950, None, 185),
    ("V2-C", "HistGB",  "서수+수치TE",  "+규칙피처", "가중", "전체×5fold", "OOF", 0.95028, None, 225),
    ("V2-D", "HistGB",  "서수+수치TE",  "+규칙+원본50k", "가중", "전체×5fold", "OOF", 0.94970, None, 204),
    ("N2", "HistGB",    "서수+수치TE",  "+규칙피처", "가중", "전체×5fold×5seed", "OOF", 0.95051, 0.94982, 1200),
    ("N4", "주기임베딩 MLP", "표준화+TE", "+규칙+도메인", "가중CE", "전체×5fold GPU", "OOF", 0.94883, 0.94906, 429),
    ("N5", "HistGB",    "서수+수치TE",  "+규칙피처", "가중", "전체×10fold×7seed", "OOF", 0.95045, 0.95007, 3361),
    ("N3", "HGB 3시드 보팅(폴백)", "서수+수치TE", "+규칙피처", "가중", "전체×5fold", "중첩검증", 0.95026, 0.95015, None),
    # 피처 변형 (전부 s42 단일시드 스크리닝)
    ("V2-E", "HistGB", "서수+TE", "+소프트임퓨트8종", "가중", "전체×5fold", "OOF", 0.94991, None, 1528),
    ("V2-F", "HistGB", "서수+확장TE", "+빈도·상호작용 번들", "가중", "전체×5fold", "OOF", 0.94989, None, 230),
    ("V2-H", "HistGB", "서수+다중해상도TE", "+거친구간 6종", "가중", "전체×5fold", "OOF", 0.95036, None, 500),
    ("V2-I", "HistGB", "서수+세그먼트TE", "+결측패턴 키 6종", "가중", "전체×5fold", "OOF", 0.94989, None, 485),
    ("V2-J", "HistGB", "서수+TE", "+도메인 파생 6종", "가중", "전체×5fold", "OOF", 0.95034, None, 616),
    ("V2-K", "HistGB", "서수+다중해상도TE", "H+J 결합", "가중", "전체×5fold", "OOF", 0.95039, None, 541),
    ("Cl_F", "HistGB", "서수+다중해상도TE", "K+파생3종", "가중", "전체×5fold", "OOF", 0.95022, None, 1111),
    # 튜닝·노이즈·구조 축
    ("Optuna", "HistGB", "서수+TE", "K피처 (25트라이얼)", "가중", "20만→전체 재검증", "OOF", 0.95039, None, 1500),
    ("디노이즈", "HistGB", "서수+TE", "원본규칙 라벨교정", "가중", "전체×5fold", "OOF", 0.94861, None, None),
    ("S1", "HistGB", "서수+TE", "노이즈 드랍 2%", "가중", "전체×5fold(부분)", "OOF", 0.95010, None, None),
    ("계층형", "RealMLP 2단 라우팅", "표준화+TE", "209피처", "가중CE", "fold4 GPU", "홀드아웃", 0.94544, None, 151),
    # 블렌드 (정직 = 중첩검증)
    ("N7", "N5+RealMLP 블렌드", "-", "가중 0.40/0.60", "-", "전체", "중첩검증", 0.95084, None, None),
    ("N9", "3중 블렌드", "-", "N5·RMLP·K 0.15/0.57/0.28", "-", "전체", "중첩검증", 0.95088, None, None),
]

def main():
    df = pd.DataFrame(ROWS, columns=["실험", "모델", "인코딩", "피처", "불균형",
                                     "데이터", "검증", "BalancedAcc", "PublicLB", "학습시간(s)"])
    df.to_csv(OOF / "experiment_matrix.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    # 장표용 렌더 (한글 폰트)
    for f in ["Malgun Gothic", "NanumGothic"]:
        if any(f in x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False

    show = df.fillna("—")
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(show) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)
    # 최고점은 비교가능한 검증(OOF/중첩검증) 행에서만 선정 (zoo 홀드아웃 제외)
    comparable = df["검증"].isin(["OOF", "중첩검증"])
    best = df.loc[comparable, "BalancedAcc"].max()
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f2937"); cell.set_text_props(color="white", fontweight="bold")
        elif comparable.iloc[r-1] and df.iloc[r-1]["BalancedAcc"] == best:
            cell.set_facecolor("#dcfce7")           # 최종 카드(N9) 강조
        elif df.iloc[r-1]["실험"] in ("N1", "N2", "N3", "N4", "N5", "N7", "N9"):
            cell.set_facecolor("#e0f2fe")           # 제출 라인 강조
        elif r % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("실험 통합 매트릭스 — 모델·인코딩·피처 조합별 성능 (Balanced Accuracy)",
                 fontsize=12, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(ASSETS / "experiment_matrix.png", dpi=130, bbox_inches="tight")
    print("saved", ASSETS / "experiment_matrix.png")

if __name__ == "__main__":
    main()
