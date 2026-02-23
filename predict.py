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
BASE_CAPITAL_ZAR = 500.00

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
    
    feat['VNM'] = c.diff(14) / (feat['ATR'] + 1e-12)
    direction = c.diff(14).abs()
    volatility = c.diff().abs().rolling(14).sum()
    feat['KER'] = direction / (volatility + 1e-12)
    
    lagged = feat.shift(1)
    lagged['ATR_current'] = feat['ATR']
    return lagged

if __name__ == "__main__":
    signals = []
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    
    for t in TICKERS:
        base = t.replace('=','_').lower()
        choice_path = os.path.join(MODELS_DIR, f"{base}_model_choice.joblib")
        
        df = yf.download(t, period="100d", interval="1d", progress=False)
        feat_df = engineer_live(df)
        
        feature_path = os.path.join(MODELS_DIR, f"{base}_feature_list.joblib")
        if not os.path.exists(feature_path): continue
        features = joblib.load(feature_path)
        
        last_row = feat_df.iloc[-1]
        for f in features:
            if f not in last_row:
                last_row[f] = 0.0 
                
        X = last_row[features].values.reshape(1, -1)
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{base}_scaler.joblib"))
        X_scaled = scaler.transform(X)
        
        xgb_model = joblib.load(os.path.join(MODELS_DIR, f"{base}_xgb_calibrated.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, f"{base}_lr_calibrated.joblib"))
        cb_model = joblib.load(os.path.join(MODELS_DIR, f"{base}_cb_calibrated.joblib"))
        
        prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]
        prob_lr = lr_model.predict_proba(X_scaled)[0][1]
        prob_cb = cb_model.predict_proba(X_scaled)[0][1]
        
        prob = (prob_xgb + prob_lr + prob_cb) / 3.0
        
        bull_threshold = 0.55
        bear_threshold = 0.45
        if os.path.exists(choice_path):
            choice = joblib.load(choice_path)
            bull_threshold = choice.get('thresholds', {}).get('bull', 0.55)
            bear_threshold = choice.get('thresholds', {}).get('bear', 0.45)
        
        direction = "HOLD"
        if prob >= bull_threshold: direction = "BULLISH"
        elif prob <= bear_threshold: direction = "BEARISH"
        
        if direction != "HOLD":
            last_close = float(df['Close'].iloc[-1])
            atr = float(feat_df.iloc[-1]['ATR_current'])
            sl_dist = atr * ATR_SL_MULT
            tp_dist = atr * ATR_TP_MULT
            
            b = tp_dist / sl_dist
            q = 1.0 - prob
            kelly_f = (prob * b - q) / b
            fractional_kelly = max(0.0, 0.25 * kelly_f)
            allocation_zar = BASE_CAPITAL_ZAR * fractional_kelly
            
            signals.append({
                "ticker": t, "direction": direction, "confidence": prob,
                "entry": last_close, 
                "sl": last_close - sl_dist if direction == "BULLISH" else last_close + sl_dist,
                "tp": last_close + tp_dist if direction == "BULLISH" else last_close - tp_dist,
                "allocation_zar": round(allocation_zar, 2),
                "kelly_percentage": round(fractional_kelly * 100, 2)
            })
    print(json.dumps(signals, indent=2))
