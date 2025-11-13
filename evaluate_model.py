# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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


def backtest_5day(probs, prices, trans_cost, bull_thresh, bear_thresh, horizon=5):
    pnl = []
    traded = []
    
    price_indices = prices.index.intersection(probs.index)
    
    probs_aligned = probs.loc[price_indices]
    prices_aligned = prices.loc[price_indices]

    if len(probs_aligned) != len(prices_aligned):
        print("  Backtest Warning: Price and probability indices do not match.")
        return np.array([0.0]), 0, 0.0
        
    probs_values = probs_aligned.values
    prices_values = prices_aligned.values

    i = 0
    while i < len(probs_values):
        if i + horizon >= len(prices_values):
            break
            
        prob = probs_values[i]
        
        if prob >= bull_thresh: # Signal: BUY
            p0 = prices_values[i]; p1 = prices_values[i + horizon]
            gross = (p1 / p0 - 1.0); net = gross - trans_cost
            pnl.append(net); traded.append(True)
            i += horizon 
        
        elif prob <= bear_thresh: # Signal: SHORT
            p0 = prices_values[i]; p1 = prices_values[i + horizon]
            gross = (p0 / p1 - 1.0); net = gross - trans_cost
            pnl.append(net); traded.append(True)
            i += horizon 
            
        else: # Signal: HOLD
            pnl.append(0.0); traded.append(False)
            i += 1 
        
    pnl = np.array(pnl); trades_count = np.sum(traded)
    win_rate = np.mean(pnl[traded] > 0) if trades_count > 0 else 0.0
    
    return pnl, trades_count, win_rate

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
            print("  Test data is empty. Skipping."); continue
        
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

        print("\n  --- Backtest Threshold & Model Optimization ---")
        
        best_overall_return = 0.0
        best_choice = {
            "model_type": "none", 
            "thresholds": {"bull": DEFAULT_THRESHOLDS[0], "bear": DEFAULT_THRESHOLDS[1]},
            "return": 0.0
        }
        
        all_results = []

        for model_type in MODEL_TYPES:
            probs = model_probs[model_type]
            
            for bull, bear in THRESHOLDS_TO_TEST:
                pnl, trades_count, win_rate = backtest_5day(
                    probs, test_prices_full, TRANSACTION_COSTS.get(t, 0.0005),
                    bull, bear, PREDICTION_HORIZON
                )
                total_return = pnl.sum()
                avg_return = np.mean(pnl[pnl != 0]) if trades_count > 0 else 0.0
                
                if total_return > best_overall_return:
                    best_overall_return = total_return
                    best_choice = {
                        "model_type": model_type,
                        "thresholds": {"bull": bull, "bear": bear},
                        "return": total_return
                    }

                all_results.append({
                    "Model": model_type,
                    "Threshold": f"{bull*100:.0f}% / {bear*100:.0f}%",
                    "Total Return": f"{total_return*100:.2f}%",
                    "Trades": trades_count,
                    "Win Rate": f"{win_rate*100:.2f}%"
                })
        
        summary_df = pd.DataFrame(all_results); 
        print(summary_df.to_string(index=False))
        
        print(f"\n  🏆 Best model found: {best_choice['model_type']}")
        print(f"     Optimal thresholds: {best_choice['thresholds']['bull']*100:.0f}% / {best_choice['thresholds']['bear']*100:.0f}%")
        print(f"     Best Return: {best_choice['return']*100:.2f}%")
        
        joblib.dump(best_choice, choice_save_file)
        print(f"  ✅ Saved optimal choice to {choice_save_file}")
        
        # --- End Backtest ---
        
        # --- START: CORRECTED METRICS BLOCK ---
        print("\n  --- Metrics for Chosen Model at Optimal Threshold ---")
        if best_choice['model_type'] == 'none':
            print("  No profitable model was chosen (Return = 0.00%). No metrics to report.")
        else:
            chosen_probs = model_probs[best_choice['model_type']]
            chosen_bull = best_choice['thresholds']['bull']
            chosen_bear = best_choice['thresholds']['bear']

            ypred_chosen = np.full(len(chosen_probs), -1) # Hold
            ypred_chosen[chosen_probs >= chosen_bull] = 1 # Bullish
            ypred_chosen[chosen_probs <= chosen_bear] = 0 # Bearish

            mask = ypred_chosen != -1
            
            # --- This block is now INSIDE the else statement ---
            if np.sum(mask) > 0:
                print(classification_report(yte[mask], ypred_chosen[mask], zero_division=0))
                cm = confusion_matrix(yte[mask], ypred_chosen[mask]); 
                print("  Confusion Matrix (Chosen):"); print(cm)
            else:
                print("  No trades at chosen threshold, no metrics to report.")
        # --- END: CORRECTED METRICS BLOCK ---