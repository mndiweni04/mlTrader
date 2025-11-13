# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import warnings
from sklearn.calibration import CalibratedClassifierCV
# --- NEW: Import Logistic Regression ---
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

DATA_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]

def load_data(ticker_base):
    """
    Tries to load data for a specific ticker and regime.
    """
    try:
        X_train = np.load(os.path.join(DATA_DIR, f"{ticker_base}_X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, f"{ticker_base}_y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, f"{ticker_base}_X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, f"{ticker_base}_y_val.npy"))
        features = joblib.load(os.path.join(MODELS_DIR, f"{ticker_base}_feature_list.joblib"))
        return (X_train, y_train, X_val, y_val, features)
    except FileNotFoundError:
        return None

for ticker in TICKERS:
    base = ticker.replace("=", "_").lower()
    
    found_model_for_ticker = False
    for suffix in REGIME_SUFFIXES:
        regime_base = f"{base}{suffix}"
        
        data = load_data(regime_base)
        if data is None:
            continue
            
        print(f"\n=== TRAINING {regime_base} ===")
        found_model_for_ticker = True
        
        X_train, y_train, X_val, y_val, features = data

        if y_train.size == 0 or len(np.unique(y_train)) < 2:
            print("  Skipping: No training data or only one class present.")
            continue

        # --- Calculate scale_pos_weight for XGBoost ---
        count_neg = np.sum(y_train == 0)
        count_pos = np.sum(y_train == 1)
        scale_pos_weight = count_neg / (count_pos + 1e-8)
        
        # --- 1. Train XGBoost Model ---
        print(f"  Using scale_pos_weight: {scale_pos_weight:.2f} (for XGB)")
        xgb_model_raw = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            scale_pos_weight=scale_pos_weight, 
            reg_alpha=0.5,
            reg_lambda=1.0
        )
        
        print("  Training raw XGBoost model...")
        xgb_model_raw.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        print("  Calibrating XGB probabilities on validation set...")
        xgb_calibrated = CalibratedClassifierCV(
            xgb_model_raw, 
            method='sigmoid', 
            cv='prefit'
        )
        xgb_calibrated.fit(X_val, y_val)
        
        joblib.dump(xgb_calibrated, os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib"))
        print(f"  ✅ Calibrated XGB model saved for {regime_base}.")

        # --- 2. Train Logistic Regression Model ---
        print("  Training raw Logistic Regression model...")
        # Use class_weight='balanced' as the LR equivalent of scale_pos_weight
        lr_model_raw = LogisticRegression(
            solver='liblinear', 
            class_weight='balanced', 
            random_state=42,
            C=0.1 # Add some regularization
        )
        # We train the LR model on the same (scaled) data
        lr_model_raw.fit(X_train, y_train) 
        
        print("  Calibrating LR probabilities on validation set...")
        # We must pre-fit the calibrator for LR on the validation set
        lr_calibrated = CalibratedClassifierCV(
            lr_model_raw, 
            method='sigmoid', 
            cv='prefit' # Use 'prefit' as we've already trained it
        )
        # Fit the calibrator
        lr_calibrated.fit(X_val, y_val)

        joblib.dump(lr_calibrated, os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib"))
        print(f"  ✅ Calibrated LR model saved for {regime_base}.")
        # --- END NEW ---

    if not found_model_for_ticker:
        print(f"\n--- No processed data files found for ticker {ticker} ---")


print("\n✅ All models trained and calibrated successfully.")