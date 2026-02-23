# predict.py
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
import json
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"]
MODELS_DIR = "models"
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def engineer_live(df, ticker_name, macros):
    feat = pd.DataFrame(index=df.index)
    c = df['Close'].astype(float)
    
    feat['MA5'] = c.rolling(5).mean()
    feat['MA20'] = c.rolling(20).mean()
    feat['MA50'] = c.rolling(50).mean()
    feat['MA200'] = c.rolling(200).mean()
    feat['MA_diff'] = feat['MA50'] - feat['MA200']
    feat['ATR'] = calc_atr(df)
    feat['RSI14'] = calc_rsi(c, 14)
    feat['ROC10'] = c.pct_change(periods=10) * 100
    feat['MACD'], _, feat['MACD_hist'] = calc_macd(c)
    
    feat['Day_Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
    feat['Dist_from_MA20'] = (df['Close'] - feat['MA20']) / (feat['MA20'] + 1e-12)
    
    if "^VIX" in macros:
        v_c = macros["^VIX"]['Close'].reindex(df.index, method='ffill').astype(float)
        feat['VIX_Close'] = v_c
        feat['VIX_Regime'] = (v_c > 20).astype(int)
    
    lagged = feat.shift(1)
    lagged['ATR_current'] = feat['ATR'] 
    return lagged

if __name__ == "__main__":
    signals_list = []
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    timestamp = now.isoformat()
    
    macros = {mt: yf.download(mt, period="50d", progress=False, auto_adjust=False) for mt in MACRO_TICKERS}
    for mt in macros:
        if isinstance(macros[mt].columns, pd.MultiIndex): macros[mt].columns = macros[mt].columns.get_level_values(0)
    
    vix_val = float(macros['^VIX']['Close'].iloc[-1]) if not macros['^VIX'].empty else 20.0
    vix_regime = 1 if vix_val > 20 else 0

    for t in TICKERS:
        base = t.replace('=','_').lower()
        suffix = ("_high_vix" if vix_regime else "_low_vix") if t in TICKERS_TO_SPLIT else ""
        rb = f"{base}{suffix}"
        
        choice_path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
        if not os.path.exists(choice_path): continue
        
        choice = joblib.load(choice_path)
        if choice['model_type'] == 'none': continue
        
        df = yf.download(t, period="250d", interval="1d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: continue
        
        feat_df = engineer_live(df, t, macros)
        features = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        
        latest = feat_df.iloc[-1]
        X_raw = latest[features].values.reshape(1, -1)
        if np.isnan(X_raw).any(): continue
        
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{choice['model_type']}_calibrated.joblib"))
        
        prob = model.predict_proba(scaler.transform(X_raw))[0][1]
        
        direction = "HOLD"
        if prob >= choice['thresholds']['bull']: direction = "BULLISH"
        elif prob <= choice['thresholds']['bear']: direction = "BEARISH"
        
        if direction != "HOLD":
            last_close = float(df['Close'].iloc[-1])
            atr = float(latest['ATR_current'])
            sl = last_close - (atr * ATR_SL_MULT) if direction == "BULLISH" else last_close + (atr * ATR_SL_MULT)
            tp = last_close + (atr * ATR_TP_MULT) if direction == "BULLISH" else last_close - (atr * ATR_TP_MULT)
            
            signals_list.append({
                "ticker": t, "direction": direction, "confidence": float(prob),
                "entry_price": last_close, "stop_loss": sl, "take_profit": tp,
                "model_regime": rb, "timestamp": timestamp
            })

    output = {"timestamp": timestamp, "signals": signals_list}
    print(json.dumps(output, indent=2))
