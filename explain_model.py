# explain_model.py
import joblib, pandas as pd, numpy as np, os, matplotlib.pyplot as plt

MODELS_DIR = "models"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]

def plot_importance(ticker_raw):
    base = ticker_raw.replace('=','_').lower()
    for suffix in ["", "_low_vix", "_high_vix"]:
        rb = f"{base}{suffix}"
        path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
        if not os.path.exists(path): continue

        choice = joblib.load(path)
        
        # Structural Guardrail: Prevent processing of null model states
        if choice.get('model_type') == 'none' or not choice.get('trading_enabled'):
            continue

        m_list = ['xgb', 'lr', 'cb'] if choice['model_type'] == 'ensemble' else [choice['model_type']]
        
        for m_type in m_list:
            model_path = os.path.join(MODELS_DIR, f"{rb}_{m_type}_calibrated.joblib")
            if not os.path.exists(model_path):
                continue

            model = joblib.load(model_path)
            features = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
            base_model = model.calibrated_classifiers_[0].estimator
            
            imp = getattr(base_model, 'feature_importances_', None)
            if imp is None: imp = np.abs(base_model.coef_[0]) if hasattr(base_model, 'coef_') else np.zeros(len(features))
            
            df_imp = pd.DataFrame({'f': features, 'i': imp}).sort_values('i', ascending=False).head(10)
            plt.figure(figsize=(8, 5)); plt.barh(df_imp['f'], df_imp['i']); plt.title(f"{rb} | {m_type}")
            plt.savefig(os.path.join(MODELS_DIR, f"{rb}_{m_type}_imp.png")); plt.close()

if __name__ == "__main__":
    for t in TICKERS: plot_importance(t)
    print("✅ explain_model.py: Importance plots generated.")