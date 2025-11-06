# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import itertools

warnings.filterwarnings('ignore')

DATA_DIR = "data/processed"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

TICKERS = ["CL=F"] # FOCUSED

# Anti-Overfitting Grid (36 tests)
PARAM_GRID = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.02],
    'subsample': [0.8],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.1]
}

def load_data(ticker):
    base = ticker.replace("=", "_").lower()
    try:
        X_train = np.load(os.path.join(DATA_DIR, f"{base}_X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, f"{base}_y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, f"{base}_X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, f"{base}_y_val.npy"))
        features = joblib.load(os.path.join(MODELS_DIR, f"{base}_feature_list.joblib"))
        return (X_train, y_train, X_val, y_val, features)
    except FileNotFoundError as e:
        print(f"  Error loading data for {ticker}: {e}")
        return None

def apply_smote(X, y):
    if y.size == 0: return X, y
    print(f"  Before SMOTE: {np.bincount(y)}")
    if np.min(np.bincount(y)) < 2:
        print("  Skipping SMOTE: not enough samples in minority class.")
        return X, y
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  After SMOTE: {np.bincount(y_res)}")
    return X_res, y_res

for ticker in TICKERS:
    print(f"\n=== TRAINING {ticker} ===")
    data = load_data(ticker)
    if data is None: continue
    X_train, y_train, X_val, y_val, features = data
    base = ticker.replace("=", "_").lower()

    if y_train.size == 0:
        print("  Skipping XGB: No training data.")
        continue

    print("  Applying SMOTE to training data...")
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    
    best_f1 = -1.0
    best_model = None
    best_params = {}

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"  Starting robust hyperparameter tuning ({len(param_combinations)} combinations)...")

    for params in param_combinations:
        print(f"    Testing params: {params}", end="")
        
        xgb_model = xgb.XGBClassifier(
            **params,
            n_estimators=500,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20
        )
        
        xgb_model.fit(X_train_bal, y_train_bal, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_val = xgb_model.predict(X_val)
        f1_val = f1_score(y_val, y_pred_val, pos_label=1, zero_division=0)
        print(f" -> F1: {f1_val:.4f}")
        
        if f1_val > best_f1:
            best_f1 = f1_val
            best_model = xgb_model
            best_params = params
            print(f"    🚀 New Best F1: {f1_val:.4f} (at iteration {best_model.best_iteration})")

    if best_model:
        print(f"\n  ✅ Best F1-Score: {best_f1*100:.2f}%")
        print(f"  ✅ Best Params: {best_params}")
        joblib.dump(best_model, os.path.join(MODELS_DIR, f"{base}_xgb_model.joblib"))
        print(f"  🧠 Best XGBoost model saved for {ticker}.")
    else:
        print("  Tuning failed, no model was saved.")

print("\n✅ All models tuned and saved successfully.")