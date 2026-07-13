"""PS S6E7 - Predicting Student Health Risk. 공통 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OOF = ROOT / "oof"
SUBS = ROOT / "subs"
for d in (OOF, SUBS):
    d.mkdir(exist_ok=True)

TARGET = "health_condition"
ID = "id"
CLASSES = ["at-risk", "unhealthy", "fit"]   # EDA 후 실제 라벨/순서 확정
N_CLASS = len(CLASSES)

SEED = 42
N_SPLITS = 5
SEEDS = [42, 2026, 7, 101, 777]   # 시드 평균용

COMP = "playground-series-s6e7"
