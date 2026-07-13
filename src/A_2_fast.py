import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from category_encoders import TargetEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    target_map = {'Healthy': 0, 'Fit': 1, 'At Risk': 2}
    y = train['Health_Risk'].map(target_map).values
    X = train.drop(['Health_Risk'], axis=1)
    
    cat_cols = X.select_dtypes(['object']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].fillna('Missing').astype('category')
        test[c] = test[c].fillna('Missing').astype('category')

    # Target Encoding features
    te_cols = ['Country', 'Gender', 'Occupation', 'Diet_Type', 'Sleep_Quality']
    
    # 80/20 Split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply TE
    encoder = TargetEncoder(cols=te_cols)
    X_tr_te = encoder.fit_transform(X_tr, y_tr)
    X_va_te = encoder.transform(X_va)
    X_te_te = encoder.transform(test)
    
    # 1. LGBM
    model_lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight='balanced', random_state=42, n_jobs=4, verbose=-1)
    model_lgb.fit(X_tr_te, y_tr, eval_set=[(X_va_te, y_va)], callbacks=[LGBMClassifier.early_stopping(50, verbose=False)] if hasattr(LGBMClassifier, 'early_stopping') else None)
    oof_lgb = model_lgb.predict_proba(X_va_te)
    test_lgb = model_lgb.predict_proba(X_te_te)
    
    # 2. CatBoost
    model_cat = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, auto_class_weights='Balanced', random_seed=42, verbose=0, cat_features=cat_cols, early_stopping_rounds=50, thread_count=4)
    model_cat.fit(X_tr_te, y_tr, eval_set=(X_va_te, y_va))
    oof_cat = model_cat.predict_proba(X_va_te)
    test_cat = model_cat.predict_proba(X_te_te)
    
    # 3. XGBoost
    model_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, enable_categorical=True, tree_method='hist', random_state=42, n_jobs=4, early_stopping_rounds=50)
    model_xgb.fit(X_tr_te, y_tr, eval_set=[(X_va_te, y_va)], verbose=False)
    oof_xgb = model_xgb.predict_proba(X_va_te)
    test_xgb = model_xgb.predict_proba(X_te_te)
    
    # 4. HistGB
    model_hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, class_weight='balanced', random_state=42, early_stopping=True, validation_fraction=0.1)
    model_hgb.fit(X_tr_te, y_tr)
    oof_hgb = model_hgb.predict_proba(X_va_te)
    test_hgb = model_hgb.predict_proba(X_te_te)
    
    # Blend
    oof_blend = oof_lgb * 0.3 + oof_cat * 0.3 + oof_xgb * 0.2 + oof_hgb * 0.2
    test_blend = test_lgb * 0.3 + test_cat * 0.3 + test_xgb * 0.2 + test_hgb * 0.2
    
    print(f"Blended Val Acc: {balanced_accuracy_score(y_va, oof_blend.argmax(1)):.5f}")
    
    rev_map = {0: 'Healthy', 1: 'Fit', 2: 'At Risk'}
    sub = pd.DataFrame({'id': test['id'] if 'id' in test.columns else range(len(test)), 'Health_Risk': pd.Series(test_blend.argmax(1)).map(rev_map)})
    sub.to_csv('A_2.csv', index=False)
    print("A_2.csv created successfully.")

if __name__ == '__main__':
    main()
