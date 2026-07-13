import shap
import numpy as np
from xgboost import XGBClassifier
import pandas as pd

X = pd.DataFrame(np.random.randn(10, 5), columns=list('ABCDE'))
y = np.random.randint(0, 3, 10)

model = XGBClassifier(base_score=0.5, random_state=42)
model.fit(X, y)

try:
    explainer = shap.TreeExplainer(model)
    print("TreeExplainer worked!")
except Exception as e:
    print("TreeExplainer failed:", e)

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    print("Explainer worked!")
except Exception as e:
    print("Explainer failed:", e)
