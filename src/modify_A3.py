import re

with open('src/A_3.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Modify title
code = code.replace('=== A_2.py: 0.952+ 극한 성능 최적화 (Multi-Model Ensemble) ===', '=== A_3.py: 0.953+ 초극강 앙상블 (LightGBM + HistGB) ===')

# Remove XGBoost and CatBoost imports
code = code.replace('from xgboost import XGBClassifier\n', '')
code = code.replace('from catboost import CatBoostClassifier\n', '')

# Remove oof_cat, oof_xgb, test_cat, test_xgb
code = code.replace('    oof_cat = np.zeros((len(X), 3))\n', '')
code = code.replace('    oof_xgb = np.zeros((len(X), 3))\n', '')
code = code.replace('    test_cat = np.zeros((len(X_test), 3))\n', '')
code = code.replace('    test_xgb = np.zeros((len(X_test), 3))\n', '')

# Replace LGBM n_jobs
code = code.replace('n_jobs=1', 'n_jobs=4')

# Remove Model 2 and Model 3 blocks
code = re.sub(r'        # 모델 2: CatBoost.*?print\(f\" CatBoost Base Acc: \{balanced_accuracy_score\(y_va, oof_cat\[va_idx\]\.argmax\(1\)\):\.5f\}\"\)\n', '', code, flags=re.DOTALL)
code = re.sub(r'        # 모델 3: XGBoost.*?print\(f\" XGBoost Base Acc: \{balanced_accuracy_score\(y_va, oof_xgb\[va_idx\]\.argmax\(1\)\):\.5f\}\"\)\n', '', code, flags=re.DOTALL)

# Adjust blending
code = code.replace('oof_blend = oof_lgb * 0.3 + oof_cat * 0.3 + oof_xgb * 0.2 + oof_hgb * 0.2', 'oof_blend = oof_lgb * 0.5 + oof_hgb * 0.5')
code = code.replace('test_blend = test_lgb * 0.3 + test_cat * 0.3 + test_xgb * 0.2 + test_hgb * 0.2', 'test_blend = test_lgb * 0.5 + test_hgb * 0.5')

# Remove Prior Correction
code = re.sub(r'    # Prior-Correction\n.*?    print\(f\"\[After Correction\] Blended OOF Acc: \{balanced_accuracy_score\(y, oof_corrected\.argmax\(1\)\):\.5f\}\"\)\n', '', code, flags=re.DOTALL)

# Adjust rule apply argument
code = code.replace('apply_rule(test_raw, test_corrected)', 'apply_rule(test_raw, test_blend)')

# Save output to A_3.csv
code = code.replace("sub.to_csv('A_2.csv', index=False)", "sub.to_csv('A_3.csv', index=False)")
code = code.replace('최종 제출 파일 생성 완료: A_2.csv', '최종 제출 파일 생성 완료: A_3.csv')

with open('src/A_3.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('A_3.py generated successfully')
