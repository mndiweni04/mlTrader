# evaluate_model.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
REGIME_SUFFIXES = ["", "_low_vix", "_high_vix"]
MODEL_TYPES = ['xgb', 'lr', 'cb', 'ensemble'] 
PROC_DIR, MODELS_DIR = "data/processed", "models"
HORIZON = 10

def backtest_strategy(probs, prices, cost, bull, bear):
    pnl, traded, i = [], [], 0
    prices_aligned = prices.loc[probs.index].values
    probs_val = probs.values
    while i < len(probs_val) - HORIZON:
        prob = probs_val[i]
        if prob >= bull:
            pnl.append((prices_aligned[i+HORIZON] / prices_aligned[i] - 1.0) - cost)
            traded.append(True); i += HORIZON
        elif prob <= bear:
            pnl.append((prices_aligned[i] / prices_aligned[i+HORIZON] - 1.0) - cost)
            traded.append(True); i += HORIZON
        else:
            pnl.append(0.0); traded.append(False); i += 1
    return np.array(pnl), int(np.sum(traded))

for t in TICKERS:
    base = t.replace('=','_').lower()
    price_file = os.path.join(PROC_DIR, f"{base}_test_prices.csv")
    
    # Structural Guardrail: Validate file existence before read
    if not os.path.exists(price_file):
        print(f"Skipping {t}: Price data file not found ({price_file})")
        continue

    price_data = pd.read_csv(price_file, index_col=0, parse_dates=True).squeeze()
    
    for suffix in REGIME_SUFFIXES:
        rb = f"{base}{suffix}"
        files = [os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib") for m in ['xgb', 'lr', 'cb']]
        if not all(os.path.exists(f) for f in files + [os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib")]): continue

        Xte, yte = np.load(os.path.join(PROC_DIR, f"{rb}_X_test.npy")), np.load(os.path.join(PROC_DIR, f"{rb}_y_test.npy"))
        idx = joblib.load(os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib"))
        
        m_probs = {m: pd.Series(joblib.load(os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib")).predict_proba(Xte)[:, 1], index=idx) for m in ['xgb', 'lr', 'cb']}
        m_probs['ensemble'] = (m_probs['xgb'] + m_probs['lr'] + m_probs['cb']) / 3.0

        best_score, best_cfg = -np.inf, {"model_type": "none", "trading_enabled": False}
        for m_type in MODEL_TYPES:
            for b_up, b_dn in [(0.55, 0.45), (0.6, 0.4)]:
                pnl, count = backtest_strategy(m_probs[m_type], price_data, 0.0005, b_up, b_dn)
                score = pnl.sum() * 100
                if score > best_score and count >= 5:
                    best_score = score
                    # Structural Guardrail: Enforce strict positive profitability threshold
                    is_profitable = score > 0
                    best_cfg = {"model_type": m_type, "thresholds": {"bull": b_up, "bear": b_dn}, "trading_enabled": is_profitable}

        joblib.dump(best_cfg, os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib"))
        print(f"Evaluated {rb}: Best {best_cfg['model_type']} ({best_score:.2f}%) - Trading Enabled: {best_cfg['trading_enabled']}")
