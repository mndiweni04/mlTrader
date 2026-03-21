# train_model.py
import os
import sys
import joblib
import json
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

DATA_DIR, MODELS_DIR = "data/processed", "models"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]
manifest = {}
missing_dependencies = []

# Phase 1: Dependency Verification
for ticker in TICKERS:
    base = ticker.replace("=", "_").lower()
    for suffix in REGIME_SUFFIXES:
        rb = f"{base}{suffix}"
        if not os.path.exists(os.path.join(DATA_DIR, f"{rb}_X_train.npy")) or \
           not os.path.exists(os.path.join(DATA_DIR, f"{rb}_y_train.npy")):
            missing_dependencies.append(rb)

if missing_dependencies:
    print(f"CRITICAL ERROR: Missing processed data for {len(missing_dependencies)} regimes.")
    print(f"Missing regimes: {', '.join(missing_dependencies)}")
    raise FileNotFoundError("process_data.py failed to produce required files. Halting training.")

# Phase 2: Model Training
for ticker in TICKERS:
    base = ticker.replace("=", "_").lower()
    for suffix in REGIME_SUFFIXES:
        rb = f"{base}{suffix}"
        
        X_train, y_train = np.load(os.path.join(DATA_DIR, f"{rb}_X_train.npy")), np.load(os.path.join(DATA_DIR, f"{rb}_y_train.npy"))
        X_val, y_val = np.load(os.path.join(DATA_DIR, f"{rb}_X_val.npy")), np.load(os.path.join(DATA_DIR, f"{rb}_y_val.npy"))

        n_samples = len(y_train)
        if n_samples < 50 or len(np.unique(y_train)) < 2: continue

        method = "isotonic" if n_samples >= 200 else "sigmoid"
        tscv = TimeSeriesSplit(n_splits=5 if n_samples >= 100 else 2)
        
        iters = []
        for train_idx, test_idx in tscv.split(X_train):
            Xt, yt = X_train[train_idx], y_train[train_idx]
            Xv, yv = X_train[test_idx], y_train[test_idx]
            if len(np.unique(yt)) < 2: continue
            m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, early_stopping_rounds=20, eval_metric="logloss")
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            iters.append(m.best_iteration)

        avg_iter = max(1, int(np.mean(iters))) if iters else 100

        # Prepare unified dataset with PredefinedSplit to bypass deprecated cv="prefit" constraint
        X_all = np.vstack((X_train, X_val))
        y_all = np.concatenate((y_train, y_val))
        test_fold = np.concatenate([-1 * np.ones(len(y_train)), np.zeros(len(y_val))])
        ps = PredefinedSplit(test_fold)

        # Train and Calibrate Ensemble Components
        for name, clf in [
            ("xgb", xgb.XGBClassifier(n_estimators=avg_iter, max_depth=4)),
            ("lr", LogisticRegression(class_weight="balanced", C=0.1)),
            ("cb", CatBoostClassifier(iterations=avg_iter, depth=4, auto_class_weights='Balanced', verbose=0))
        ]:
            cal = CalibratedClassifierCV(estimator=clf, method=method, cv=ps).fit(X_all, y_all)
            joblib.dump(cal, os.path.join(MODELS_DIR, f"{rb}_{name}_calibrated.joblib"))
        
        manifest[rb] = {"last_trained": datetime.now().isoformat(), "samples": int(n_samples), "calibration": method}

with open(os.path.join(MODELS_DIR, "model_manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2)
print("✅ train_model.py Complete")
