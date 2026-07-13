with open('src/A_2.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace test blending division
code = code.replace('test_lgb += model_lgb.predict_proba(X_te) / n_splits', 'test_lgb = model_lgb.predict_proba(X_te)')
code = code.replace('test_cat += model_cat.predict_proba(X_te) / n_splits', 'test_cat = model_cat.predict_proba(X_te)')
code = code.replace('test_xgb += model_xgb.predict_proba(X_te) / n_splits', 'test_xgb = model_xgb.predict_proba(X_te)')
code = code.replace('test_hgb += model_hgb.predict_proba(X_te) / n_splits', 'test_hgb = model_hgb.predict_proba(X_te)')

# Add break after first fold
target = 'print(f" HistGB Base Acc: {balanced_accuracy_score(y_va, oof_hgb[va_idx].argmax(1)):.5f}")'
code = code.replace(target, target + '\n        break')

# Calculate OOF only on the va_idx
code = code.replace('oof_blend = oof_lgb * 0.3 + oof_cat * 0.3 + oof_xgb * 0.2 + oof_hgb * 0.2', 'oof_blend = oof_lgb[va_idx] * 0.3 + oof_cat[va_idx] * 0.3 + oof_xgb[va_idx] * 0.2 + oof_hgb[va_idx] * 0.2\n    y = y[va_idx]')

with open('src/A_2_fast_2.py', 'w', encoding='utf-8') as f:
    f.write(code)
