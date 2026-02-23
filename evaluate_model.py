# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]
MODEL_TYPES = ['xgb', 'lr', 'ensemble'] 

PROC_DIR = "data/processed"
MODELS_DIR = "models"

TRANSACTION_COSTS = {
    "CL=F": 0.0008, "GC=F": 0.0005, "SI=F": 0.0007, "NG=F": 0.0010,
    "ZC=F": 0.0010, "EURUSD=X": 0.0001, "JPYUSD=X": 0.0001,
    "ES=F": 0.0005, "NQ=F": 0.0005
}
PREDICTION_HORIZON = 5
THRESHOLDS_TO_TEST = [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40), (0.65, 0.35)]
DEFAULT_THRESHOLDS = (0.60, 0.40)

MIN_TOTAL_RETURN = 0.5
MIN_TRADES = 10 
MIN_RECALL_THRESHOLD = 0.05
MIN_RECALL_SYMMETRY = 0.15 # Minimum balance between Buy/Sell recall
MAX_PROFIT_FACTOR = 5.0

def backtest_5day(probs, prices, cost, bull_thresh, bear_thresh, horizon=5):
    pnl, traded = [], []
    price_indices = prices.index.intersection(probs.index)
    probs_aligned = probs.loc[price_indices].values
    prices_aligned = prices.loc[price_indices].values

    i = 0
    while i < len(probs_aligned):
        if i + horizon >= len(prices_aligned): break
        prob = probs_aligned[i]
        if prob >= bull_thresh:
            pnl.append((prices_aligned[i + horizon] / prices_aligned[i] - 1.0) - cost)
            traded.append(True); i += horizon 
        elif prob <= bear_thresh:
            pnl.append((prices_aligned[i] / prices_aligned[i + horizon] - 1.0) - cost)
            traded.append(True); i += horizon 
        else:
            i += 1 
    pnl = np.array(pnl)
    trades = int(np.sum(traded))
    win_rate = np.mean(pnl > 0) if trades > 0 else 0.0
    gains = np.sum(pnl[pnl > 0]); losses = np.abs(np.sum(pnl[pnl < 0]))
    pf = min(gains / (losses + 1e-12), MAX_PROFIT_FACTOR)
    return pnl, trades, win_rate, pf

def check_production_constraints(total_return, trades_count, recall_0, recall_1):
    failure_reasons = []
    if total_return <= MIN_TOTAL_RETURN: failure_reasons.append("LOW_RETURN")
    if trades_count < MIN_TRADES: failure_reasons.append("INSUFFICIENT_TRADES")
    if recall_0 < MIN_RECALL_THRESHOLD or recall_1 < MIN_RECALL_THRESHOLD: failure_reasons.append("ONE_SIDED")
    
    symmetry = min(recall_0, recall_1) / (max(recall_0, recall_1) + 1e-12)
    if symmetry < MIN_RECALL_SYMMETRY: failure_reasons.append("ASYMMETRIC")
    
    return len(failure_reasons) == 0, failure_reasons

full_test_prices = {}
for t in TICKERS:
    base = t.replace('=','_').lower()
    price_file = os.path.join(PROC_DIR, f"{base}_test_prices.csv")
    if os.path.exists(price_file):
        full_test_prices[t] = pd.read_csv(price_file, index_col=0, parse_dates=True).squeeze()

for t in TICKERS:
    for suffix in REGIME_SUFFIXES:
        base = t.replace('=','_').lower()
        rb = f"{base}{suffix}"
        
        files = [os.path.join(MODELS_DIR, f"{rb}_xgb_calibrated.joblib"), 
                 os.path.join(MODELS_DIR, f"{rb}_lr_calibrated.joblib"),
                 os.path.join(PROC_DIR, f"{rb}_X_test.npy"), 
                 os.path.join(PROC_DIR, f"{rb}_y_test.npy")]
        
        if not all(os.path.exists(f) for f in files): continue

        Xte, yte = np.load(files[2]), np.load(files[3])
        test_indices = joblib.load(os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib"))
        regime_prices = full_test_prices[t].loc[test_indices]
            
        model_probs = {}
        m_xgb = joblib.load(files[0])
        model_probs['xgb'] = pd.Series(m_xgb.predict_proba(Xte)[:, 1], index=regime_prices.index)
        m_lr = joblib.load(files[1])
        model_probs['lr'] = pd.Series(m_lr.predict_proba(Xte)[:, 1], index=regime_prices.index)
        model_probs['ensemble'] = (model_probs['xgb'] + model_probs['lr']) / 2.0

        best_score = -np.inf
        best_choice = {"model_type": "none", "trading_enabled": False}
        
        for m_type in MODEL_TYPES:
            probs = model_probs[m_type]
            for bull, bear in THRESHOLDS_TO_TEST:
                pnl, trades, win, pf = backtest_5day(probs, full_test_prices[t], TRANSACTION_COSTS.get(t, 0.0005), bull, bear)
                
                ypred = np.full(len(probs), -1)
                ypred[probs >= bull] = 1; ypred[probs <= bear] = 0
                recalls = recall_score(yte, ypred, labels=[0, 1], average=None, zero_division=0)
                
                passes, reasons = check_production_constraints(pnl.sum()*100, trades, recalls[0], recalls[1])
                
                if passes:
                    score = (pnl.sum()*100 * pf) / (1 + abs(pnl.sum()*100 - (win*100)))
                    if score > best_score:
                        best_score = score
                        best_choice = {"model_type": m_type, "thresholds": {"bull": bull, "bear": bear}, "trading_enabled": True, "production_score": score}

        joblib.dump(best_choice, os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib"))
        print(f"Evaluated {rb}: Best Model = {best_choice['model_type']}")
