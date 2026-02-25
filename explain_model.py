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
        
        # Extract the underlying fitted model from CalibratedClassifierCV safely
        if hasattr(model, 'calibrated_classifiers_'):
            calibrated_clf = model.calibrated_classifiers_[0]
            # Handle scikit-learn version differences (base_estimator vs estimator)
            base_model = getattr(calibrated_clf, 'estimator', getattr(calibrated_clf, 'base_estimator', None))
        else:
            base_model = model
            
        if base_model is None:
            print(f"  [ERROR] Could not extract base estimator from {m_type}.")
            continue

        # Extract importances (Tree-based) or coefficients (Linear-based)
        importances = getattr(base_model, 'feature_importances_', None)
        if importances is None:
            if hasattr(base_model, 'get_feature_importance'):
                importances = base_model.get_feature_importance()
            else:
                coef = getattr(base_model, 'coef_', None)
                if coef is not None:
                    importances = np.abs(coef[0])
                else:
                    print(f"  [WARNING] No feature importances or coef_ found for {m_type}.")
                    importances = np.zeros(len(features))
        
        df_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False).head(10)
        print(df_imp.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        plt.barh(df_imp['feature'], df_imp['importance'])
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 Features: {regime}")
        plt.tight_layout()
        
        # Save plot to disk instead of plt.show() so the bash pipeline doesn't freeze
        plot_path = os.path.join(MODELS_DIR, f"{regime}_importance.png")
        plt.savefig(plot_path)
        print(f"  [OK] Saved importance plot to {plot_path}")
        plt.close()

if __name__ == "__main__":
    for t in TICKERS: plot_importance(t)
