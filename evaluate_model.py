# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]
MODEL_TYPES = ['xgb', 'lr', 'cb', 'ensemble'] 
PROC_DIR, MODELS_DIR = "data/processed", "models"
HORIZON = 10

def backtest_strategy(probs, prices, cost, bull, bear):
    """
    Backtests signals against a continuous price history.
    probs: pd.Series with DatetimeIndex of signals
    prices: pd.Series with continuous DatetimeIndex of master price history
    """
    pnl, traded_count = [], 0
    
    # Structural Fix: Absolute Chronological Mapping
    # Map each signal date to its index in the continuous master price array
    price_indices = prices.index.get_indexer(probs.index)
    
    i = 0
    while i < len(probs):
        prob = probs.iloc[i]
        current_price_idx = price_indices[i]
        exit_price_idx = current_price_idx + HORIZON
        
        # Guardrail: Ensure exit date exists in master price calendar
        if current_price_idx == -1 or exit_price_idx >= len(prices):
            i += 1
            continue
            
        if prob >= bull:
            # Long position
            entry = prices.iloc[current_price_idx]
            exit_val = prices.iloc[exit_price_idx]
            pnl.append((exit_val / entry - 1.0) - cost)
            traded_count += 1
            # Logical Fix: Skip the actual calendar holding period
            exit_date = prices.index[exit_price_idx]
            next_indices = np.where(probs.index >= exit_date)[0]
            i = next_indices[0] if len(next_indices) > 0 else len(probs)
            
        elif prob <= bear:
            # Short position
            entry = prices.iloc[current_price_idx]
            exit_val = prices.iloc[exit_price_idx]
            pnl.append((entry / exit_val - 1.0) - cost)
            traded_count += 1
            exit_date = prices.index[exit_price_idx]
            next_indices = np.where(probs.index >= exit_date)[0]
            i = next_indices[0] if len(next_indices) > 0 else len(probs)
            
        else:
            pnl.append(0.0)
            i += 1
            
    return np.array(pnl), traded_count

# Structural Fix: Index Normalization Function
def normalize_index(idx):
    return pd.to_datetime(idx).tz_localize(None)

for t in TICKERS:
    base = t.replace('=','_').lower()
    price_file = os.path.join(PROC_DIR, f"{base}_test_prices.csv")
    
    if not os.path.exists(price_file):
        print(f"Skipping {t}: Price data file not found.")
        continue

    # Load and normalize master price history
    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True).squeeze()
    price_data.index = normalize_index(price_data.index)
    price_data = price_data.sort_index()
    
    for suffix in REGIME_SUFFIXES:
        rb = f"{base}{suffix}"
        model_paths = [os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib") for m in ['xgb', 'lr', 'cb']]
        index_path = os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib")
        
        if not (all(os.path.exists(f) for f in model_paths) and os.path.exists(index_path)):
            continue

        Xte = np.load(os.path.join(PROC_DIR, f"{rb}_X_test.npy"))
        raw_idx = joblib.load(index_path)
        
        # Normalize signal indices to match price data
        idx = normalize_index(raw_idx)
        
        m_probs = {}
        for m in ['xgb', 'lr', 'cb']:
            model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib"))
            m_probs[m] = pd.Series(model.predict_proba(Xte)[:, 1], index=idx)
            
        m_probs['ensemble'] = (m_probs['xgb'] + m_probs['lr'] + m_probs['cb']) / 3.0

        best_score, best_cfg = -np.inf, {"model_type": "none", "trading_enabled": False}
        
        for m_type in MODEL_TYPES:
            for b_up, b_dn in [(0.55, 0.45), (0.6, 0.4)]:
                pnl, count = backtest_strategy(m_probs[m_type], price_data, 0.0005, b_up, b_dn)
                score = pnl.sum() * 100
                
                if score > best_score and count >= 5:
                    best_score = score
                    # Strict profitability gate
                    is_profitable = score > 0
                    best_cfg = {
                        "model_type": m_type, 
                        "thresholds": {"bull": b_up, "bear": b_dn}, 
                        "trading_enabled": is_profitable
                    }

        joblib.dump(best_cfg, os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib"))
        status = f"Best {best_cfg['model_type']} ({best_score:.2f}%)" if best_cfg['trading_enabled'] else "No viable model found"
        print(f"Evaluated {rb}: {status}")

print("✅ evaluate_model.py Complete")
