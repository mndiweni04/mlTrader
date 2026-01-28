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
ISOTONIC_SAMPLE_THRESHOLD = 200

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

        n_samples = len(y_train)
        if n_samples < 50:
            continue

        method = "isotonic" if n_samples >= ISOTONIC_SAMPLE_THRESHOLD else "sigmoid"
        n_splits = 5 if n_samples >= 100 else 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        iters, weights = [], []
        for train_idx, test_idx in tscv.split(X_train):
            Xt, Xv = X_train[train_idx], X_train[test_idx]
            yt, yv = y_train[train_idx], y_train[test_idx]
            if len(np.unique(yt)) < 2:
                continue
            
            w = np.sum(yt==0) / (np.sum(yt==1) + 1e-8)
            weights.append(w)
            m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, 
                                  scale_pos_weight=w, early_stopping_rounds=20, eval_metric="logloss")
            m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            iters.append(m.best_iteration)

        if not iters:
            continue
        avg_iter, avg_w = int(np.mean(iters)), np.mean(weights)

        # FINAL FIX FOR SKLEARN 1.6+: 
        # Use cv=None and ensemble=False to replicate 'prefit' behavior
        
        # XGBoost Calibration
        xgb_f = xgb.XGBClassifier(n_estimators=avg_iter, max_depth=4, scale_pos_weight=avg_w).fit(X_train, y_train)
        cal_xgb = CalibratedClassifierCV(estimator=xgb_f, method=method, cv="prefit")
        # Fallback for strict environments: if 'prefit' still fails, the fit() will catch it
        try:
            cal_xgb.fit(X_val, y_val)
        except ValueError:
            # Modern Sklearn fallback
            cal_xgb = CalibratedClassifierCV(estimator=xgb_f, method=method, cv=None, ensemble=False)
            cal_xgb.fit(X_val, y_val)
        joblib.dump(cal_xgb, os.path.join(MODELS_DIR, f"{rb}_xgb_calibrated.joblib"))
        
        # Logistic Regression Calibration
        lr_f = LogisticRegression(class_weight="balanced", C=0.1).fit(X_train, y_train)
        cal_lr = CalibratedClassifierCV(estimator=lr_f, method=method, cv="prefit")
        try:
            cal_lr.fit(X_val, y_val)
        except ValueError:
            # Modern Sklearn fallback
            cal_lr = CalibratedClassifierCV(estimator=lr_f, method=method, cv=None, ensemble=False)
            cal_lr.fit(X_val, y_val)
        joblib.dump(cal_lr, os.path.join(MODELS_DIR, f"{rb}_lr_calibrated.joblib"))
        
        manifest[rb] = {
            "last_trained": datetime.now().isoformat(),
            "training_samples": int(n_samples),
            "calibration_method": method,
            "xgb_model": f"{rb}_xgb_calibrated.joblib",
            "lr_model": f"{rb}_lr_calibrated.joblib"
        }

with open(os.path.join(MODELS_DIR, "model_manifest.json"), 'w') as f:
    json.dump(manifest, f, indent=2)
print("✅ train_model.py Complete")
