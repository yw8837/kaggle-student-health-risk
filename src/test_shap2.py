import shap
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

X = pd.DataFrame(np.random.randn(100, 5), columns=list('ABCDE'))
y = np.random.randint(0, 3, 100)

model = XGBClassifier(base_score=0.5, random_state=42)
model.fit(X, y)

try:
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X[:10])
    print("Explainer with predict_proba worked!")
    print(shap_values.shape)
except Exception as e:
    print("Explainer with predict_proba failed:", e)

