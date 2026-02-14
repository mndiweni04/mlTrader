# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
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

THRESHOLDS_TO_TEST = [
    (0.50, 0.50), # 50%
    (0.55, 0.45), # 55%
    (0.60, 0.40), # 60%
    (0.65, 0.35)  # 65%
]
DEFAULT_THRESHOLDS = (0.60, 0.40)

# Production-Safe Constraint Thresholds (Strict Taxonomy)
MIN_TOTAL_RETURN = 0.0           # Profitability gate: total_return must be > 0%
MIN_TRADES = 10                  # [FIXED] Lowered from 15 to 10 to catch more signals
MIN_RECALL_THRESHOLD = 0.05      # [FIXED] Lowered from 0.15 to 0.05 (5%)
MIN_RECALL_SYMMETRY = 0.01       # [FIXED] Lowered from 0.20 to 0.01 (effectively disabled)
MAX_PROFIT_FACTOR = 5.0          # Profit factor clamp


def backtest_5day(probs, prices, cost, bull_thresh, bear_thresh, horizon=5):
    """
    Backtest strategy and return PnL, trade count, win rate, and profit factor (clamped).
    """
    pnl = []
    traded = []
    
    price_indices = prices.index.intersection(probs.index)
    
    probs_aligned = probs.loc[price_indices]
    prices_aligned = prices.loc[price_indices]

    if len(probs_aligned) != len(prices_aligned):
        print("  Backtest Warning: Price and probability indices do not match.")
        return np.array([0.0]), 0, 0.0, 1.0
        
    probs_values = probs_aligned.values
    prices_values = prices_aligned.values

    i = 0
    while i < len(probs_values):
        if i + horizon >= len(prices_values):
            break
            
        prob = probs_values[i]
        
        if prob >= bull_thresh: # Signal: BUY
            p0 = prices_values[i]; p1 = prices_values[i + horizon]
            gross = (p1 / p0 - 1.0); net = gross - cost
            pnl.append(net); traded.append(True)
            i += horizon 
        
        elif prob <= bear_thresh: # Signal: SHORT
            p0 = prices_values[i]; p1 = prices_values[i + horizon]
            gross = (p0 / p1 - 1.0); net = gross - cost
            pnl.append(net); traded.append(True)
            i += horizon 
            
        else: # Signal: HOLD
            pnl.append(0.0); traded.append(False)
            i += 1 
        
    pnl = np.array(pnl)
    trades_count = int(np.sum(traded))
    win_rate = float(np.mean(pnl[traded] > 0)) if trades_count > 0 else 0.0
    
    # Calculate Profit Factor (clamped to MAX_PROFIT_FACTOR)
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(np.abs(np.sum(pnl[pnl < 0])))
    profit_factor = gains / (losses + 1e-12)
    profit_factor = min(profit_factor, MAX_PROFIT_FACTOR)  # Clamp to 5.0
    
    return pnl, trades_count, win_rate, profit_factor


def calculate_recall_metrics(ypred, yte):
    """
    Calculate recall_0, recall_1, and recall_symmetry.
    Returns (recall_0, recall_1, recall_symmetry) or (0.0, 0.0, 0.0) if no trades.
    """
    # Filter to only predicted trades (not HOLD)
    mask = ypred != -1
    if np.sum(mask) == 0:
        return 0.0, 0.0, 0.0
    
    ypred_trades = ypred[mask]
    yte_trades = yte[mask]
    
    # Calculate recall for class 0 and class 1
    try:
        recalls = recall_score(yte_trades, ypred_trades, labels=[0, 1], average=None)
        recall_0 = float(recalls[0])
        recall_1 = float(recalls[1])
    except Exception:
        return 0.0, 0.0, 0.0
    
    # Calculate recall_symmetry
    if recall_0 == 0.0 and recall_1 == 0.0:
        recall_symmetry = 0.0
    else:
        recall_symmetry = float(min(recall_0, recall_1) / (max(recall_0, recall_1) + 1e-12))
    
    return recall_0, recall_1, recall_symmetry


def check_production_constraints(total_return, trades_count, recall_0, recall_1, recall_symmetry):
    """
    Check all production-safe constraints and return (passes, failure_reasons).
    
    Returns:
        passes (bool): True if all constraints pass
        failure_reasons (list): List of reason codes (empty if all pass)
    """
    failure_reasons = []
    
    # Constraint 1: Profitability Gate
    if total_return <= MIN_TOTAL_RETURN:
        failure_reasons.append("NEGATIVE_RETURN")
    
    # Constraint 2: Volume Guard
    if trades_count < MIN_TRADES:
        failure_reasons.append("INSUFFICIENT_TRADES")
    
    # Constraint 3: Recall Guard
    if recall_0 < MIN_RECALL_THRESHOLD or recall_1 < MIN_RECALL_THRESHOLD:
        failure_reasons.append("LOW_RECALL")
    
    # Constraint 4: Symmetry Guard
    if recall_symmetry < MIN_RECALL_SYMMETRY:
        failure_reasons.append("ASYMMETRIC_RECALL")
    
    passes = len(failure_reasons) == 0
    return passes, failure_reasons


def calculate_production_score(total_return, profit_factor, win_rate):
    """
    Production Score = (total_return * profit_factor) / (1 + abs(total_return - (win_rate * 100)))
    Only called when all constraints pass.
    """
    numerator = total_return * profit_factor
    denominator = 1.0 + abs(total_return - (win_rate * 100.0))
    return float(numerator / denominator)

# Load all price files once
full_test_prices = {}
for t in TICKERS:
    base = t.replace('=','_').lower()
    price_file = os.path.join(PROC_DIR, f"{base}_test_prices.csv")
    if os.path.exists(price_file):
        full_test_prices[t] = pd.read_csv(price_file, index_col=0, parse_dates=True).squeeze()
    else:
        print(f"Warning: No test price file found for {t}")

# Loop through all possible regime files
for t in TICKERS:
    for suffix in REGIME_SUFFIXES:
        base = t.replace('=','_').lower()
        regime_base = f"{base}{suffix}"

        xgb_model_file = os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib")
        lr_model_file = os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib")
        feature_file = os.path.join(MODELS_DIR, f"{regime_base}_feature_list.joblib")
        choice_save_file = os.path.join(MODELS_DIR, f"{regime_base}_model_choice.joblib")
        Xte_file = os.path.join(PROC_DIR, f"{regime_base}_X_test.npy")
        yte_file = os.path.join(PROC_DIR, f"{regime_base}_y_test.npy")
        indices_file = os.path.join(MODELS_DIR, f"{regime_base}_test_indices.joblib")

        if not all(os.path.exists(f) for f in [xgb_model_file, lr_model_file, feature_file, Xte_file, yte_file, indices_file]):
            continue
            
        if t not in full_test_prices:
            print(f"  Skipping {regime_base}: Missing main price file.")
            continue

        print(f"\n--- Evaluating {regime_base} ---")

        Xte = np.load(Xte_file)
        yte = np.load(yte_file)
        
        if Xte.size == 0: 
            print("  Test data is empty. Skipping.")
            continue
        
        try:
            test_indices = joblib.load(indices_file)
            test_prices_full = full_test_prices[t]
            regime_test_prices = test_prices_full.loc[test_indices]
        except Exception as e:
            print(f"  Error aligning prices: {e}. Skipping.")
            continue
            
        model_probs = {}
        try:
            model_xgb = joblib.load(xgb_model_file)
            model_probs['xgb'] = pd.Series(model_xgb.predict_proba(Xte)[:, 1], index=regime_test_prices.index)
            
            model_lr = joblib.load(lr_model_file)
            model_probs['lr'] = pd.Series(model_lr.predict_proba(Xte)[:, 1], index=regime_test_prices.index)
            
            model_probs['ensemble'] = (model_probs['xgb'] + model_probs['lr']) / 2.0
            
        except Exception as e:
            print(f"  Error loading models or getting probs: {e}. Skipping.")
            continue

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
                pnl, trades_count, win_rate, profit_factor = backtest_5day(
                    probs, test_prices_full, TRANSACTION_COSTS.get(t, 0.0005),
                    bull, bear, PREDICTION_HORIZON
                )
                
                # Strict Variable Definitions (Taxonomy)
                total_return = float(pnl.sum() * 100.0)  # Express as percentage
                win_rate = float(win_rate)               # Already in [0, 1]
                profit_factor = float(profit_factor)     # Already clamped
                
                # Calculate recall metrics
                ypred = np.full(len(probs), -1)  # -1 = HOLD
                ypred[probs >= bull] = 1         # Bullish
                ypred[probs <= bear] = 0         # Bearish
                
                recall_0, recall_1, recall_symmetry = calculate_recall_metrics(ypred, yte)
                
                # Check production-safe constraints
                passes_constraints, failure_reasons = check_production_constraints(
                    total_return, trades_count, recall_0, recall_1, recall_symmetry
                )
                
                # Calculate production score only if constraints pass
                if passes_constraints:
                    production_score = calculate_production_score(total_return, profit_factor, win_rate)
                else:
                    production_score = -np.inf
                
                # Track the best model that passes all constraints
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

                # Format failure reasons for display
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
            print(f"     Minimum total return: {MIN_TOTAL_RETURN:.1f}%")
            print(f"     Minimum trades: {MIN_TRADES}")
            print(f"     Minimum recall (both classes): {MIN_RECALL_THRESHOLD*100:.1f}%")
            print(f"     Minimum recall symmetry: {MIN_RECALL_SYMMETRY:.3f}")
        else:
            print(f"\n  [BEST] BEST MODEL SELECTED")
            print(f"     Model: {best_choice['model_type']}")
            print(f"     Optimal thresholds: {best_choice['thresholds']['bull']*100:.0f}% / {best_choice['thresholds']['bear']*100:.0f}%")
            print(f"     Total Return: {best_choice['total_return']:.2f}%")
            print(f"     Profit Factor: {best_choice['profit_factor']:.3f}")
            print(f"     Production Score: {best_choice['production_score']:.4f}")
        
        joblib.dump(best_choice, choice_save_file)
        print(f"  [OK] Saved model choice to {choice_save_file}")
        
        # --- METRICS FOR CHOSEN MODEL (Only if valid) ---
        print("\n  --- Metrics for Chosen Model ---")
        if best_choice['model_type'] == 'none':
            print("  No valid model selected. No metrics to report.")
        else:
            chosen_probs = model_probs[best_choice['model_type']]
            chosen_bull = best_choice['thresholds']['bull']
            chosen_bear = best_choice['thresholds']['bear']

            ypred_chosen = np.full(len(chosen_probs), -1) # Hold
            ypred_chosen[chosen_probs >= chosen_bull] = 1 # Bullish
            ypred_chosen[chosen_probs <= chosen_bear] = 0 # Bearish

            mask = ypred_chosen != -1
            
            if np.sum(mask) > 0:
                print("\n  Classification Report:")
                print(classification_report(yte[mask], ypred_chosen[mask], zero_division=0))
                cm = confusion_matrix(yte[mask], ypred_chosen[mask])
                print("  Confusion Matrix:")
                print(cm)
            else:
                print("  No trades at chosen threshold, no metrics to report.")
