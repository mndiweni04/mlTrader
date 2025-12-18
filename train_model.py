import os
import joblib
import numpy as np
import xgboost as xgb
import warnings
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]

def load_data(ticker_base):
    """Loads preprocessed numpy files for training and validation."""
    try:
        X_train = np.load(os.path.join(DATA_DIR, f"{ticker_base}_X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, f"{ticker_base}_y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, f"{ticker_base}_X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, f"{ticker_base}_y_val.npy"))
        return X_train, y_train, X_val, y_val
    except FileNotFoundError:
        return None

# --- Main Training Loop ---
for ticker in TICKERS:
    base = ticker.replace("=", "_").lower()
    found_model = False

    for suffix in REGIME_SUFFIXES:
        regime_base = f"{base}{suffix}"
        data = load_data(regime_base)

        if data is None:
            continue

        found_model = True
        print(f"\n=== TRAINING {regime_base} ===")

        X_train, y_train, X_val, y_val = data

        if y_train.size == 0 or len(np.unique(y_train)) < 2:
            print("  Skipping: insufficient class diversity.")
            continue

        # 1. Calculate scale_pos_weight for imbalance
        count_neg = np.sum(y_train == 0)
        count_pos = np.sum(y_train == 1)
        scale_pos_weight = count_neg / (count_pos + 1e-8)
        print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")

        # --- 2. XGBoost Training ---
        # We train with early stopping first to find the best iteration
        xgb_raw = xgb.XGBClassifier(
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

        print("  Training XGB with early stopping...")
        xgb_raw.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        print("  Calibrating XGB probabilities (using 'prefit')...")
        # Use 'prefit' because we want to calibrate using the X_val set specifically
        xgb_calibrated = CalibratedClassifierCV(
            estimator=xgb_raw,
            method="sigmoid",
            cv="prefit"
        )
        xgb_calibrated.fit(X_val, y_val)

        joblib.dump(xgb_calibrated, os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib"))

        # --- 3. Logistic Regression Training ---
        print("  Training Logistic Regression...")
        lr_raw = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
            C=0.1
        )
        lr_raw.fit(X_train, y_train)

        print("  Calibrating LR probabilities...")
        lr_calibrated = CalibratedClassifierCV(
            estimator=lr_raw,
            method="sigmoid",
            cv="prefit"
        )
        lr_calibrated.fit(X_val, y_val)

        joblib.dump(lr_calibrated, os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib"))

        print(f"  ✅ Saved models for {regime_base}")

    if not found_model:
        print(f"\n--- No data found for {ticker} ---")

print("\n✅ All models trained successfully.")
