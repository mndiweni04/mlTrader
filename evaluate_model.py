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
PREDICTION_HORIZON = 10
THRESHOLDS_TO_TEST = [(0.50, 0.50), (0.55, 0.45), (0.60, 0.40), (0.65, 0.35)]
DEFAULT_THRESHOLDS = (0.60, 0.40)

MIN_TOTAL_RETURN = 0.5 
MIN_TRADES = 10 
MIN_RECALL_THRESHOLD = 0.05 
MIN_RECALL_SYMMETRY = 0.15 
MAX_PROFIT_FACTOR = 5.0 

def backtest_strategy(probs, prices, cost, bull_thresh, bear_thresh, horizon=10):
    pnl = []
    traded = []
    
    price_indices = prices.index.intersection(probs.index)
    probs_aligned = probs.loc[price_indices].values
    prices_aligned = prices.loc[price_indices].values

    i = 0
    while i < len(probs_aligned):
        if i + horizon >= len(prices_aligned):
            break
            
        prob = probs_aligned[i]
        
        if prob >= bull_thresh: 
            p0 = prices_aligned[i]; p1 = prices_aligned[i + horizon]
            gross = (p1 / p0 - 1.0); net = gross - cost
            pnl.append(net); traded.append(True)
            i += horizon 
        
        elif prob <= bear_thresh: 
            p0 = prices_aligned[i]; p1 = prices_aligned[i + horizon]
            gross = (p0 / p1 - 1.0); net = gross - cost
            pnl.append(net); traded.append(True)
            i += horizon 
            
        else: 
            pnl.append(0.0); traded.append(False)
            i += 1 
        
    pnl = np.array(pnl)
    trades_count = int(np.sum(traded))
    win_rate = float(np.mean(pnl[traded] > 0)) if trades_count > 0 else 0.0
    
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(np.abs(np.sum(pnl[pnl < 0])))
    profit_factor = gains / (losses + 1e-12)
    profit_factor = min(profit_factor, MAX_PROFIT_FACTOR) 
    
    return pnl, trades_count, win_rate, profit_factor

def calculate_recall_metrics(ypred, yte):
    mask = ypred != -1
    if np.sum(mask) == 0:
        return 0.0, 0.0, 0.0
    
    ypred_trades = ypred[mask]
    yte_trades = yte[mask]
    
    try:
        recalls = recall_score(yte_trades, ypred_trades, labels=[0, 1], average=None, zero_division=0)
        recall_0 = float(recalls[0])
        recall_1 = float(recalls[1])
    except Exception:
        return 0.0, 0.0, 0.0
    
    denom = max(recall_0, recall_1)
    if denom == 0.0:
        recall_symmetry = 0.0
    else:
        recall_symmetry = float(min(recall_0, recall_1) / denom)
    
    return recall_0, recall_1, recall_symmetry

def check_production_constraints(total_return, trades_count, recall_0, recall_1, recall_symmetry):
    failure_reasons = []
    
    if total_return <= MIN_TOTAL_RETURN: failure_reasons.append("NEGATIVE_RETURN")
    if trades_count < MIN_TRADES: failure_reasons.append("INSUFFICIENT_TRADES")
    if recall_0 < MIN_RECALL_THRESHOLD or recall_1 < MIN_RECALL_THRESHOLD: failure_reasons.append("LOW_RECALL")
    if recall_symmetry < MIN_RECALL_SYMMETRY: failure_reasons.append("ASYMMETRIC_RECALL")
    
    passes = len(failure_reasons) == 0
    return passes, failure_reasons

def calculate_production_score(total_return, profit_factor, win_rate):
    numerator = total_return * profit_factor
    denominator = 1.0 + abs(total_return - (win_rate * 100.0))
    return float(numerator / denominator)

full_test_prices = {}
for t in TICKERS:
    base = t.replace('=','_').lower()
    price_file = os.path.join(PROC_DIR, f"{base}_test_prices.csv")
    if os.path.exists(price_file):
        full_test_prices[t] = pd.read_csv(price_file, index_col=0, parse_dates=True).squeeze()

for t in TICKERS:
    for suffix in REGIME_SUFFIXES:
        base = t.replace('=','_').lower()
        regime_base = f"{base}{suffix}"

        xgb_model_file = os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib")
        lr_model_file = os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib")
        choice_save_file = os.path.join(MODELS_DIR, f"{regime_base}_model_choice.joblib")
        Xte_file = os.path.join(PROC_DIR, f"{regime_base}_X_test.npy")
        yte_file = os.path.join(PROC_DIR, f"{regime_base}_y_test.npy")
        indices_file = os.path.join(MODELS_DIR, f"{regime_base}_test_indices.joblib")

        if not all(os.path.exists(f) for f in [xgb_model_file, lr_model_file, Xte_file, yte_file, indices_file]):
            continue
            
        if t not in full_test_prices:
            continue

        print(f"\n--- Evaluating {regime_base} ---")

        Xte = np.load(Xte_file)
        yte = np.load(yte_file)
        
        test_indices = joblib.load(indices_file)
        test_prices_full = full_test_prices[t]
        regime_test_prices = test_prices_full.loc[test_indices]
            
        model_probs = {}
        model_xgb = joblib.load(xgb_model_file)
        model_probs['xgb'] = pd.Series(model_xgb.predict_proba(Xte)[:, 1], index=regime_test_prices.index)
        
        model_lr = joblib.load(lr_model_file)
        model_probs['lr'] = pd.Series(model_lr.predict_proba(Xte)[:, 1], index=regime_test_prices.index)
        
        model_probs['ensemble'] = (model_probs['xgb'] + model_probs['lr']) / 2.0

        print("\n  --- Backtest Threshold & Model Optimization (Production-Safe) ---")
        
        best_production_score = -np.inf
        best_choice = {
            "model_type": "none", 
            "thresholds": {"bull": DEFAULT_THRESHOLDS[0], "bear": DEFAULT_THRESHOLDS[1]},
            "trading_enabled": False,
            "reason": "NO_VALID_MODEL"
        }
        
        all_results = []

        for model_type in MODEL_TYPES:
            probs = model_probs[model_type]
            
            for bull, bear in THRESHOLDS_TO_TEST:
                pnl, trades_count, win_rate, profit_factor = backtest_strategy(
                    probs, test_prices_full, TRANSACTION_COSTS.get(t, 0.0005),
                    bull, bear, PREDICTION_HORIZON
                )
                
                total_return = float(pnl.sum() * 100.0) 
                ypred = np.full(len(probs), -1) 
                ypred[probs >= bull] = 1 
                ypred[probs <= bear] = 0 
                
                recall_0, recall_1, recall_symmetry = calculate_recall_metrics(ypred, yte)
                passes_constraints, failure_reasons = check_production_constraints(
                    total_return, trades_count, recall_0, recall_1, recall_symmetry
                )
                
                if passes_constraints:
                    production_score = calculate_production_score(total_return, profit_factor, win_rate)
                else:
                    production_score = -np.inf
                
                if production_score > best_production_score:
                    best_production_score = production_score
                    best_choice = {
                        "model_type": model_type,
                        "thresholds": {"bull": bull, "bear": bear},
                        "trading_enabled": True,
                        "total_return": total_return,
                        "profit_factor": profit_factor,
                        "production_score": production_score
                    }

                fail_reason_str = "|".join(failure_reasons) if failure_reasons else "PASS"
                all_results.append({
                    "Model": model_type,
                    "Threshold": f"{bull*100:.0f}% / {bear*100:.0f}%",
                    "Total Return (%)": f"{total_return:.2f}",
                    "Profit Factor": f"{profit_factor:.3f}",
                    "Recall Class 0": f"{recall_0*100:.1f}%",
                    "Recall Class 1": f"{recall_1*100:.1f}%",
                    "Recall Symmetry": f"{recall_symmetry:.3f}",
                    "Trades": trades_count,
                    "Win Rate": f"{win_rate*100:.1f}%",
                    "Status": fail_reason_str,
                    "Prod Score": f"{production_score:.4f}" if production_score > -np.inf else "N/A"
                })
        
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
        
        if best_choice['model_type'] == 'none':
            print(f"\n  [WARNING] NO VALID MODEL FOUND")
        else:
            print(f"\n  [BEST] BEST MODEL SELECTED")
            print(f"     Model: {best_choice['model_type']}")
            print(f"     Optimal thresholds: {best_choice['thresholds']['bull']*100:.0f}% / {best_choice['thresholds']['bear']*100:.0f}%")
            print(f"     Total Return: {best_choice['total_return']:.2f}%")
            print(f"     Profit Factor: {best_choice['profit_factor']:.3f}")
            print(f"     Production Score: {best_choice['production_score']:.4f}")
        
        joblib.dump(best_choice, choice_save_file)
        print(f"  [OK] Saved model choice to {choice_save_file}")
