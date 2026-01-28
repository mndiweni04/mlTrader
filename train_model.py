# train_model.py
import os
import joblib
import json
import numpy as np
import xgboost as xgb
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_DIR, MODELS_DIR = "data/processed", "models"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]

# Adaptive Calibration Threshold
ISOTONIC_SAMPLE_THRESHOLD = 200  # Use isotonic if n_samples >= 200, else sigmoid

# Track training metadata for manifest
manifest = {}

for ticker in TICKERS:
    base = ticker.replace("=", "_").lower()
    for suffix in REGIME_SUFFIXES:
        rb = f"{base}{suffix}"
        try:
            X_train = np.load(os.path.join(DATA_DIR, f"{rb}_X_train.npy"))
            y_train = np.load(os.path.join(DATA_DIR, f"{rb}_y_train.npy"))
            X_val = np.load(os.path.join(DATA_DIR, f"{rb}_X_val.npy"))
            y_val = np.load(os.path.join(DATA_DIR, f"{rb}_y_val.npy"))
        except FileNotFoundError: 
            continue

        # Conditional Data Guard
        n_samples = len(y_train)
        if n_samples < 50:
            print(f"Skipping {rb}: insufficient samples ({n_samples})")
            continue

        # Determine calibration method based on sample size
        if n_samples >= ISOTONIC_SAMPLE_THRESHOLD:
            calibration_method = "isotonic"
            print(f"--- Training {rb} (Isotonic Calibration, n={n_samples}) ---")
        else:
            calibration_method = "sigmoid"
            print(f"--- Training {rb} (Sigmoid Calibration, n={n_samples}) ---")

        # Dynamic Cross-Validation
        n_splits = 5 if n_samples >= 100 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        iters, weights = [], []

        for train_idx, test_idx in tscv.split(X_train):
            Xt, Xv = X_train[train_idx], X_train[test_idx]
            yt, yv = y_train[train_idx], y_train[test_idx]
            
            # Fold Diversity Check
            if len(np.unique(yt)) < 2:
                print(f"  Warning: Skipping fold for {rb} (Single Class)")
                continue
            
            # Weight Optimization
            w = np.sum(yt==0) / (np.sum(yt==1) + 1e-8)
            weights.append(w)
            
            m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, 
                                  scale_pos_weight=w, early_stopping_rounds=20, eval_metric="logloss")
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            iters.append(m.best_iteration)

        if not iters: 
            continue
        avg_iter, avg_w = int(np.mean(iters)), np.mean(weights)

        # Final Training
        xgb_f = xgb.XGBClassifier(n_estimators=avg_iter, max_depth=4, scale_pos_weight=avg_w).fit(X_train, y_train)
        cal_xgb = CalibratedClassifierCV(xgb_f, method=calibration_method, cv="prefit").fit(X_val, y_val)
        joblib.dump(cal_xgb, os.path.join(MODELS_DIR, f"{rb}_xgb_calibrated.joblib"))
        
        lr_f = LogisticRegression(class_weight="balanced", C=0.1).fit(X_train, y_train)
        cal_lr = CalibratedClassifierCV(lr_f, method=calibration_method, cv="prefit").fit(X_val, y_val)
        joblib.dump(cal_lr, os.path.join(MODELS_DIR, f"{rb}_lr_calibrated.joblib"))
        
        print(f"  ✅ Saved models with {calibration_method} calibration (avg_iter={avg_iter}, scale_pos_weight={avg_w:.3f})")
        
        # Record metadata for manifest
        manifest[rb] = {
            "last_trained": datetime.now().isoformat(),
            "training_samples": int(n_samples),
            "validation_samples": int(len(y_val)),
            "calibration_method": calibration_method,
            "best_iteration": int(avg_iter),
            "scale_pos_weight": float(avg_w),
            "xgb_model": f"{rb}_xgb_calibrated.joblib",
            "lr_model": f"{rb}_lr_calibrated.joblib"
        }

# Write manifest to JSON for audit trail
try:
    manifest_path = os.path.join(MODELS_DIR, "model_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Model manifest written: {manifest_path} ({len(manifest)} regimes)")
except Exception as e:
    print(f"[WARNING] Failed to write manifest: {e}")

print("✅ train_model.py: Batch training complete.")