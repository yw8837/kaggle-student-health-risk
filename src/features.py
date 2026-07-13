"""데이터 로드 + 자동 dtype 처리 + 피처엔지니어링.

실제 31개 컬럼명은 데이터 다운로드 후 EDA에서 확정. 지금은 컬럼-불가지론적으로
- object/저카디널리티 -> 범주형(category)
- 나머지 -> 수치형
을 자동 판별. FE는 컬럼 확인 후 add_features()에 채운다.
"""
import numpy as np
import pandas as pd
from config import DATA, TARGET, ID, CLASSES


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, test


def get_feature_cols(train):
    return [c for c in train.columns if c not in (ID, TARGET)]


def detect_categorical(df, feat_cols, max_card=25):
    """비수치형(문자열/StringDtype/category) 이거나 저카디널리티 정수 -> 범주형."""
    from pandas.api.types import is_numeric_dtype
    cats = []
    for c in feat_cols:
        if not is_numeric_dtype(df[c]):
            cats.append(c)
        elif df[c].dtype.kind in "iu" and df[c].nunique() <= max_card:
            cats.append(c)
    return cats


def encode_target(train):
    """문자열 라벨 -> 0..K-1 (config.CLASSES 순서). test 예측은 다시 라벨로 역변환."""
    mapping = {c: i for i, c in enumerate(CLASSES)}
    assert set(train[TARGET].unique()) <= set(CLASSES), \
        f"라벨 불일치: {sorted(train[TARGET].unique())} vs {CLASSES}"
    y = train[TARGET].map(mapping).to_numpy()
    return y, mapping


def add_features(df):
    """FE 자리 - EDA로 컬럼 의미 확인 후 채운다 (비율/상호작용 등)."""
    return df


def prep(train, test):
    feat = get_feature_cols(train)
    cats = detect_categorical(train, feat)
    for c in cats:
        # train+test 합집합으로 카테고리 정렬 -> lgbm 코드 일관성 보장
        levels = pd.Categorical(
            pd.concat([train[c], test[c]], ignore_index=True)
        ).categories
        train[c] = pd.Categorical(train[c], categories=levels)
        test[c] = pd.Categorical(test[c], categories=levels)
    y, mapping = encode_target(train)
    return train[feat], test[feat], y, cats, mapping
