# predict.py
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
import json

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MODELS_DIR = "models"
ATR_SL_MULT, ATR_TP_MULT = 1.5, 2.5

# Replicated Indicators for Zero Feature Drift
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def engineer_live(df):
    feat = pd.DataFrame(index=df.index)
    c = df['Close'].astype(float)
    feat['MA5'] = c.rolling(5).mean()
    feat['MA20'] = c.rolling(20).mean()
    feat['MA50'] = c.rolling(50).mean()
    feat['RSI14'] = calc_rsi(c, 14)
    feat['ATR'] = calc_atr(df)
    
    # Lag features by 1 to match training T-1 logic
    lagged = feat.shift(1)
    lagged['ATR_current'] = feat['ATR']
    return lagged

if __name__ == "__main__":
    signals = []
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    
    for t in TICKERS:
        base = t.replace('=','_').lower()
        choice_path = os.path.join(MODELS_DIR, f"{base}_model_choice.joblib")
        if not os.path.exists(choice_path): continue
        
        choice = joblib.load(choice_path)
        if choice['model_type'] == 'none': continue
        
        df = yf.download(t, period="100d", interval="1d", progress=False)
        feat_df = engineer_live(df)
        
        features = joblib.load(os.path.join(MODELS_DIR, f"{base}_feature_list.joblib"))
        X = feat_df.iloc[-1][features].values.reshape(1, -1)
        
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{base}_scaler.joblib"))
        model = joblib.load(os.path.join(MODELS_DIR, f"{base}_{choice['model_type']}_calibrated.joblib"))
        
        prob = model.predict_proba(scaler.transform(X))[0][1]
        
        direction = "HOLD"
        if prob >= choice['thresholds']['bull']: direction = "BULLISH"
        elif prob <= choice['thresholds']['bear']: direction = "BEARISH"
        
        if direction != "HOLD":
            last_close = float(df['Close'].iloc[-1])
            atr = float(feat_df.iloc[-1]['ATR_current'])
            signals.append({
                "ticker": t, "direction": direction, "confidence": prob,
                "entry": last_close, "sl": last_close - (atr * ATR_SL_MULT) if direction == "BULLISH" else last_close + (atr * ATR_SL_MULT),
                "tp": last_close + (atr * ATR_TP_MULT) if direction == "BULLISH" else last_close - (atr * ATR_TP_MULT)
            })
    print(json.dumps(signals, indent=2))
