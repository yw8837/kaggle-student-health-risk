import time
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def load_and_preprocess():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    mapping = {'at-risk': 0, 'unhealthy': 1, 'fit': 2}
    y = train['health_condition'].map(mapping).values
    
    base_features = [c for c in train.columns if c not in ['id', 'health_condition']]
    X_train = train[base_features].copy()
    X_test = test[base_features].copy()
    
    return X_train, X_test, y, train, test

def add_advanced_features(X_train, X_test):
    n_tr = len(X_train)
    both = pd.concat([X_train, X_test], ignore_index=True)
    
    ms = both['sleep_duration'].isna().astype(int)
    mt = both['stress_level'].isna().astype(int)
    ma = both['physical_activity_level'].isna().astype(int)
    n = ms + mt + ma
    seg = np.select([n == 0, (ms == 1) & (n == 1), (mt == 1) & (n == 1), (ma == 1) & (n == 1)], [0, 1, 2, 3], default=4)
    both['seg_id'] = seg.astype(str)
    
    both['sleep_bin'] = (both['sleep_duration'] * 2).round().fillna(-1).astype(str)
    both['step_bin'] = (both['step_count'] / 500).round().fillna(-1).astype(str)
    both['bmi_bin'] = (both['bmi'] / 0.5).round().fillna(-1).astype(str)
    
    both['ix_sleep_stress'] = both['sleep_bin'] + "_" + both['stress_level'].fillna('NaN').astype(str)
    both['ix_sleep_act'] = both['sleep_bin'] + "_" + both['physical_activity_level'].fillna('NaN').astype(str)
    both['ix_stress_quality'] = both['stress_level'].fillna('NaN').astype(str) + "_" + both['sleep_quality'].fillna('NaN').astype(str)
    
    both['cal_per_step'] = both['calorie_expenditure'] / (both['step_count'] + 1)
    
    cat_cols = both.select_dtypes(['object']).columns
    for c in cat_cols:
        both[c] = both[c].fillna('Missing').astype('category')
        
    X_train = both.iloc[:n_tr].copy()
    X_test = both.iloc[n_tr:].copy()
    
    return X_train, X_test, list(cat_cols)

def apply_target_encoding(X_tr, X_va, X_te, y_tr, te_cols):
    te = TargetEncoder(target_type='multiclass', random_state=42, cv=5)
    
    X_tr_enc = te.fit_transform(X_tr[te_cols].astype(str), y_tr)
    X_va_enc = te.transform(X_va[te_cols].astype(str))
    X_te_enc = te.transform(X_te[te_cols].astype(str))
    
    cols = [f"te_{c}_{k}" for c in te_cols for k in range(3)]
    
    df_tr_enc = pd.DataFrame(X_tr_enc, columns=cols, index=X_tr.index)
    df_va_enc = pd.DataFrame(X_va_enc, columns=cols, index=X_va.index)
    df_te_enc = pd.DataFrame(X_te_enc, columns=cols, index=X_te.index)
    
    X_tr = pd.concat([X_tr, df_tr_enc], axis=1)
    X_va = pd.concat([X_va, df_va_enc], axis=1)
    X_te = pd.concat([X_te, df_te_enc], axis=1)
    
    return X_tr, X_va, X_te

def apply_rule(df, probs):
    preds = probs.argmax(1)
    s = df['sleep_duration']
    st = df['stress_level']
    act = df['physical_activity_level']
    
    c1 = (s < 6) & (st == 'high')
    c2 = (s >= 7) & (st == 'low') & (act == 'active')
    
    preds[c1] = 1 # unhealthy
    preds[c2] = 2 # fit
    return preds

def main():
    print("=== A_2.py: 0.952+ 극한 성능 최적화 (Multi-Model Ensemble) ===")
    t0 = time.time()
    
    X, X_test, y, train_raw, test_raw = load_and_preprocess()
    X, X_test, cat_cols = add_advanced_features(X, X_test)
    
    # Target Encoding 대상: 카테고리성 및 생성된 구간 피처들
    te_cols = cat_cols + ['seg_id', 'sleep_bin', 'step_bin', 'bmi_bin', 'ix_sleep_stress', 'ix_sleep_act', 'ix_stress_quality']
    # 중복 제거
    te_cols = list(set(te_cols))
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros((len(X), 3))
    oof_cat = np.zeros((len(X), 3))
    oof_xgb = np.zeros((len(X), 3))
    oof_hgb = np.zeros((len(X), 3))
    
    test_lgb = np.zeros((len(X_test), 3))
    test_cat = np.zeros((len(X_test), 3))
    test_xgb = np.zeros((len(X_test), 3))
    test_hgb = np.zeros((len(X_test), 3))
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        print(f"\n[Fold {fold}]")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        X_te = X_test.copy()
        
        # 1. Target Encoding
        X_tr, X_va, X_te = apply_target_encoding(X_tr, X_va, X_te, y_tr, te_cols)
        
        # 모델 1: LightGBM
        model_lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight='balanced', random_state=42, n_jobs=1, verbose=-1)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[LGBMClassifier.early_stopping(50, verbose=False)] if hasattr(LGBMClassifier, 'early_stopping') else None)
        oof_lgb[va_idx] = model_lgb.predict_proba(X_va)
        test_lgb = model_lgb.predict_proba(X_te)
        print(f" LGBM Base Acc: {balanced_accuracy_score(y_va, oof_lgb[va_idx].argmax(1)):.5f}")
        
        # 모델 2: CatBoost (조기 종료 추가하여 가볍게)
        model_cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights='Balanced', random_seed=42, verbose=0, cat_features=cat_cols, early_stopping_rounds=50, thread_count=1)
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        oof_cat[va_idx] = model_cat.predict_proba(X_va)
        test_cat = model_cat.predict_proba(X_te)
        print(f" CatBoost Base Acc: {balanced_accuracy_score(y_va, oof_cat[va_idx].argmax(1)):.5f}")
        
        # 모델 3: XGBoost (조기 종료 추가)
        model_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, enable_categorical=True, tree_method='hist', random_state=42, n_jobs=1, early_stopping_rounds=50)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_xgb[va_idx] = model_xgb.predict_proba(X_va)
        test_xgb = model_xgb.predict_proba(X_te)
        print(f" XGBoost Base Acc: {balanced_accuracy_score(y_va, oof_xgb[va_idx].argmax(1)):.5f}")
        
        # 모델 4: HistGradientBoosting (Scikit-Learn Native)
        model_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, class_weight='balanced', random_state=42, early_stopping=True, validation_fraction=0.1)
        model_hgb.fit(X_tr, y_tr)
        oof_hgb[va_idx] = model_hgb.predict_proba(X_va)
        test_hgb = model_hgb.predict_proba(X_te)
        print(f" HistGB Base Acc: {balanced_accuracy_score(y_va, oof_hgb[va_idx].argmax(1)):.5f}")
        break

    # Blending (가중치 튜닝: LGBM 0.3, CatBoost 0.3, XGBoost 0.2, HistGB 0.2)
    oof_blend = oof_lgb[va_idx] * 0.3 + oof_cat[va_idx] * 0.3 + oof_xgb[va_idx] * 0.2 + oof_hgb[va_idx] * 0.2
    y = y[va_idx]
    test_blend = test_lgb * 0.3 + test_cat * 0.3 + test_xgb * 0.2 + test_hgb * 0.2
    
    print(f"\n[Before Correction] Blended OOF Acc: {balanced_accuracy_score(y, oof_blend.argmax(1)):.5f}")
    
    # Prior-Correction
    priors = np.array([(y == 0).mean(), (y == 1).mean(), (y == 2).mean()])
    oof_corrected = oof_blend / priors
    test_corrected = test_blend / priors
    
    print(f"[After Correction] Blended OOF Acc: {balanced_accuracy_score(y, oof_corrected.argmax(1)):.5f}")
    
    # Rule Override (OOF는 검증용이므로 제외, Test에만 최종 적용)
    final_test_preds = apply_rule(test_raw, test_corrected)
    
    classes = ['at-risk', 'unhealthy', 'fit']
    sub = pd.DataFrame({'id': test_raw['id'], 'health_condition': [classes[p] for p in final_test_preds]})
    sub.to_csv('A_2.csv', index=False)
    print(f"\n최종 제출 파일 생성 완료: A_2.csv (총 소요시간: {time.time()-t0:.1f}초)")

if __name__ == '__main__':
    main()
