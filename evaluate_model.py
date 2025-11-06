# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F"] # FOCUSED
PROC_DIR = "data/processed"
MODELS_DIR = "models"
RAW_DIR = "data/raw"

TRANSACTION_COSTS = {"CL=F": 0.0008}
THRESHOLDS_TO_TEST = [0.60, 0.65, 0.70, 0.75] 
PREDICTION_HORIZON = 5

def simple_backtest(y_true, y_pred, probs, prices, trans_cost, conf_threshold=0.6, horizon=5):
    pnl = []
    traded = []
    if len(y_true) == 0 or len(y_pred) == 0 or len(probs) == 0:
        return np.array([0.0]), np.array([0.0]), np.array([False])
        
    if len(prices) < len(y_true) + horizon:
        print(f"  Backtest Warning (Thresh {conf_threshold}): Not enough price data.")
        max_len = len(prices) - horizon
        if max_len <= 0: return np.array([0.0]), np.array([0.0]), np.array([False])
        y_true = y_true[:max_len]; y_pred = y_pred[:max_len]; probs = probs[:max_len]

    for i, pred in enumerate(y_pred):
        prob = probs[i]
        if prob < conf_threshold and (1 - prob) < conf_threshold: 
            pnl.append(0.0); traded.append(False); continue
        
        p0 = prices[i]; p1 = prices[i + horizon]
        if pred == 1: gross = (p1 / p0 - 1.0)
        else: gross = (p0 / p1 - 1.0)
        net = gross - trans_cost
        pnl.append(net); traded.append(True)
        
    pnl = np.array(pnl); cum = np.cumsum(pnl)
    return pnl, cum, np.array(traded)

for t in TICKERS:
    print(f"\n--- Evaluating {t} ---")
    base = t.replace('=','_').lower()
    model_xgb_file = os.path.join(MODELS_DIR, f"{base}_xgb_model.joblib")
    scaler_file = os.path.join(MODELS_DIR, f"{base}_scaler.joblib")
    feature_file = os.path.join(MODELS_DIR, f"{base}_feature_list.joblib")
    raw_csv = os.path.join(RAW_DIR, f"{base}_1d_data.csv")

    if not all(os.path.exists(f) for f in [scaler_file, feature_file, model_xgb_file, raw_csv]):
        print("  Missing one or more required files. Skipping.")
        continue

    scaler = joblib.load(scaler_file); features = joblib.load(feature_file)
    print("  CHOSEN MODEL FOR BACKTEST: XGB")
    Xte_file = os.path.join(PROC_DIR, f"{base}_X_test.npy")
    yte_file = os.path.join(PROC_DIR, f"{base}_y_test.npy")
    if not os.path.exists(Xte_file) or not os.path.exists(yte_file):
        print("  Missing test data files. Skipping."); continue
        
    model = joblib.load(model_xgb_file); Xte = np.load(Xte_file); yte = np.load(yte_file)
    if Xte.size == 0: print("  Test data is empty. Skipping."); continue
        
    probs = model.predict_proba(Xte)[:, 1]; ypred_base = (probs >= 0.5).astype(int)
    acc = accuracy_score(yte, ypred_base)
    print(f"  XGB Test Accuracy (at 50%): {acc*100:.2f}%")
    print(classification_report(yte, ypred_base, zero_division=0))
    cm = confusion_matrix(yte, ypred_base); print("  Confusion Matrix (at 50%):"); print(cm)

    if hasattr(model, 'feature_importances_'):
        try:
            fi = model.feature_importances_
            fi_list = sorted(zip(features, fi), key=lambda x: x[1], reverse=True)
            print("  Top features (XGB):")
            for name, val in fi_list[:8]: print(f"    {name}: {val:.4f}")
        except Exception: pass
            
    df_raw = pd.read_csv(raw_csv, index_col=0, parse_dates=True).sort_index()
    n_test = len(yte)
    if n_test == 0: print("  No test samples. Skipping backtest."); continue
    all_closes = df_raw['Close']
    if len(all_closes) < n_test + PREDICTION_HORIZON:
        print("  Not enough raw closes for price-aware backtest."); continue

    prices = all_closes.values[-(n_test + PREDICTION_HORIZON):]
    print("\n  --- Backtest Threshold Optimization ---")
    results = []
    for thresh in THRESHOLDS_TO_TEST:
        ypred_for_thresh = (probs >= 0.5).astype(int) 
        pnl, cum, traded = simple_backtest(yte, ypred_for_thresh, probs, prices, TRANSACTION_COSTS.get(t, 0.0005), thresh, PREDICTION_HORIZON)
        total = pnl.sum(); traded_count = traded.sum()
        avg = pnl[traded].mean() if traded_count > 0 else 0.0
        win_rate = np.mean(pnl[traded] > 0) if traded_count > 0 else 0.0
        results.append({
            "Threshold": f"{thresh*100:.0f}%", "Total Return": f"{total*100:.2f}%",
            "Trades": traded_count, "Avg Return": f"{avg*100:.3f}%", "Win Rate": f"{win_rate*100:.2f}%"
        })
    summary_df = pd.DataFrame(results); print(summary_df.to_string(index=False))