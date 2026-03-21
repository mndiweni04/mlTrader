# predict.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
import warnings
from dynamic_features import get_current_state

warnings.filterwarnings('ignore')
VERSION = "v4.2"
MODELS_DIR, RAW_DIR = "models", "data/raw"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
MACRO_TICKERS = ["UUP", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
FRED_TICKERS = ["FRED_T10Y2Y", "FRED_UNRATE", "FRED_CPIAUCSL", "FRED_M2SL", "FRED_DGS10"]

def main():
    macro_df = pd.DataFrame({mt: yf.download(mt, period="60d", progress=False)['Close'] for mt in MACRO_TICKERS}).ffill()
    fred_df = pd.DataFrame({ft: pd.read_csv(os.path.join(RAW_DIR, f"{ft.replace('=','_').replace('^','').lower()}_1d_data.csv"), index_col=0, parse_dates=True)['Close'] for ft in FRED_TICKERS if os.path.exists(os.path.join(RAW_DIR, f"{ft.replace('=','_').replace('^','').lower()}_1d_data.csv"))}).ffill()
    
    current_vix = macro_df['^VIX'].iloc[-1] if '^VIX' in macro_df.columns else 20.0
    vix_regime = 1 if current_vix >= 20 else 0
    signals = []

    for ticker in TICKERS:
        df = yf.download(ticker, period="60d", progress=False)
        if df is None or len(df) < 50: continue
        
        latest_feat, atr, price = get_current_state(df, macro_df, fred_df)
        base = ticker.replace('=','_').lower()
        rb = f"{base}_high_vix" if vix_regime == 1 else f"{base}_low_vix"
        if not os.path.exists(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")): rb = base
        if not os.path.exists(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")): continue

        choice = joblib.load(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib"))
        if not choice.get("trading_enabled"): continue
        
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        f_list = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        X = scaler.transform(pd.DataFrame([latest_feat]).reindex(columns=f_list, fill_value=0.0).values)

        if choice['model_type'] == 'ensemble':
            probs = [joblib.load(os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib")).predict_proba(X)[0][1] for m in ['xgb', 'lr', 'cb']]
            prob = np.mean(probs)
        else:
            prob = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{choice['model_type']}_calibrated.joblib")).predict_proba(X)[0][1]

        signal_dir = "BULLISH" if prob >= choice['thresholds']['bull'] else "BEARISH" if prob <= choice['thresholds']['bear'] else "HOLD"
        if signal_dir != "HOLD":
            sl, tp = atr * 1.5, atr * 2.5
            signals.append({
                "ticker": ticker, "direction": signal_dir, "confidence": round(float(prob), 4),
                "entry_price": round(float(price), 6),
                "stop_loss": round(float(price - sl if signal_dir == "BULLISH" else price + sl), 6),
                "take_profit": round(float(price + tp if signal_dir == "BULLISH" else price - tp), 6),
                "model_regime": rb, "model_version": VERSION
            })

    print(json.dumps({"timestamp": datetime.now().isoformat(), "signals": signals}, indent=2))

if __name__ == "__main__": main()