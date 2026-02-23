# explain_model.py
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MODELS_DIR = "models"
TICKERS = ["GC=F", "ES=F", "NQ=F", "CL=F"]

def normalize_ticker(t):
    return t.replace('=','_').replace('^','').lower()

def plot_importance(ticker_raw):
    base = normalize_ticker(ticker_raw)
    for suffix in ["", "_low_vix", "_high_vix"]:
        regime = f"{base}{suffix}"
        choice_path = os.path.join(MODELS_DIR, f"{regime}_model_choice.joblib")
        if not os.path.exists(choice_path): continue

        choice = joblib.load(choice_path)
        m_type = choice['model_type']
        if m_type == 'none': continue

        print(f"\n--- Analyzing {regime} ({m_type}) ---")
        model = joblib.load(os.path.join(MODELS_DIR, f"{regime}_{m_type}_calibrated.joblib"))
        features = joblib.load(os.path.join(MODELS_DIR, f"{regime}_feature_list.joblib"))
        
        base_model = model.base_estimator
        importances = base_model.feature_importances_ if hasattr(base_model, 'feature_importances_') else np.abs(base_model.coef_[0])
        
        df_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        print(df_imp.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp['feature'], df_imp['importance'])
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 Features: {regime}")
        plt.show()

if __name__ == "__main__":
    for t in TICKERS: plot_importance(t)
