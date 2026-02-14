import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MODELS_DIR = "models"
TICKERS = ["GC=F", "ES=F_high_vix", "NQ=F_low_vix"] # Add others as needed

def plot_importance(regime):
    print(f"--- Analyzing {regime} ---")
    
    # Load model and feature list
    try:
        choice = joblib.load(os.path.join(MODELS_DIR, f"{regime}_model_choice.joblib"))
        model_type = choice['model_type']
        
        if model_type == 'none':
            print("No valid model selected for this regime.")
            return

        model_path = os.path.join(MODELS_DIR, f"{regime}_{model_type}_calibrated.joblib")
        features_path = os.path.join(MODELS_DIR, f"{regime}_feature_list.joblib")
        
        # Load the base model (CalibratedClassifierCV wraps the actual model)
        calibrated_model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        
        # Extract the underlying estimator
        base_model = calibrated_model.base_estimator
        
        importances = None
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_ # XGBoost
        elif hasattr(base_model, 'coef_'):
            importances = np.abs(base_model.coef_[0]) # Logistic Regression
            
        if importances is not None:
            # Create DataFrame
            df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
            df_imp = df_imp.sort_values('importance', ascending=False).head(10)
            
            print(df_imp)
            
            # Simple Plot
            plt.figure(figsize=(10, 6))
            plt.barh(df_imp['feature'], df_imp['importance'])
            plt.gca().invert_yaxis()
            plt.title(f"Top 10 Features for {regime} ({model_type})")
            plt.show()
        else:
            print(f"Could not extract feature importances for {model_type}")

    except Exception as e:
        print(f"Error loading {regime}: {e}")

if __name__ == "__main__":
    for t in TICKERS:
        plot_importance(t)
