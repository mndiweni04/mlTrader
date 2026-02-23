# predict.py
import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from datetime import datetime
import pytz
import json

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"]
FRED_SERIES = {
    "FRED_T10Y2Y": "T10Y2Y", "FRED_UNRATE": "UNRATE",
    "FRED_CPIAUCSL": "CPIAUCSL", "FRED_M2SL": "M2SL", "FRED_DGS10": "DGS10"
}

MODELS_DIR = "models"
ATR_SL_MULT, ATR_TP_MULT = 1.5, 2.5
BASE_CAPITAL_ZAR = 500.00

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_bbands(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return mid + (std * std_dev), mid, mid - (std * std_dev)

def calc_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, np.abs(h-c.shift()), np.abs(l-c.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def get_live_macro():
    macro_df = pd.DataFrame()
    for mt in MACRO_TICKERS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            df = yf.download(mt, period="50d", interval="1d", progress=False)
        if not df.empty:
            c = df['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            macro_df[mt] = c
    return macro_df.ffill()

def get_live_fred():
    api_key = os.environ.get("FRED_API_KEY")
    fred_df = pd.DataFrame()
    if api_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=api_key)
            for name, s_id in FRED_SERIES.items():
                data = fred.get_series(s_id)
                fred_df[name] = data
        except Exception: pass
    return fred_df.ffill()

def engineer_live(df, macro_df, fred_df):
    feat = pd.DataFrame(index=df.index)
    c = df['Close'].astype(float)
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    
    feat['MA5'] = c.rolling(5).mean()
    feat['MA20'] = c.rolling(20).mean()
    feat['MA50'] = c.rolling(50).mean()
    feat['RSI14'] = calc_rsi(c, 14)
    feat['ATR'] = calc_atr(df)
    
    u, m, lo = calc_bbands(c, 20, 2)
    feat['BB_Width'] = (u - lo) / (m + 1e-12)
    feat['VNM'] = c.diff(14) / (feat['ATR'] + 1e-12)
    
    direction = c.diff(14).abs()
    volatility = c.diff().abs().rolling(14).sum()
    feat['KER'] = direction / (volatility + 1e-12)
    
    feat = feat.join(macro_df, how='left').join(fred_df, how='left').ffill()
    
    lagged = feat.shift(1)
    lagged['ATR_current'] = feat['ATR']
    return lagged

if __name__ == "__main__":
    signals = []
    macro_df = get_live_macro()
    fred_df = get_live_fred()
    vix_current = macro_df['^VIX'].iloc[-1] if '^VIX' in macro_df.columns and not macro_df['^VIX'].empty else 15.0
    
    for t in TICKERS:
        base = t.replace('=','_').lower()
        suffix = ""
        if t in TICKERS_TO_SPLIT:
            suffix = "_high_vix" if vix_current >= 20 else "_low_vix"
            
        rb = f"{base}{suffix}"
        
        feature_path = os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib")
        if not os.path.exists(feature_path):
            rb = base 
            feature_path = os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib")
            if not os.path.exists(feature_path): continue
            
        choice_path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            df = yf.download(t, period="100d", interval="1d", progress=False)
            
        if df.empty: continue
        feat_df = engineer_live(df, macro_df, fred_df)
        
        features = joblib.load(feature_path)
        last_row = feat_df.iloc[-1].to_dict()
        
        X_dict = {}
        for f in features:
            X_dict[f] = last_row.get(f, 0.0)
            if pd.isna(X_dict[f]): X_dict[f] = 0.0
                
        X = np.array([X_dict[f] for f in features]).reshape(1, -1)
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        X_scaled = scaler.transform(X)
        
        xgb_model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_xgb_calibrated.joblib"))
        lr_model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_lr_calibrated.joblib"))
        cb_model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_cb_calibrated.joblib"))
        
        prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]
        prob_lr = lr_model.predict_proba(X_scaled)[0][1]
        prob_cb = cb_model.predict_proba(X_scaled)[0][1]
        
        prob = (prob_xgb + prob_lr + prob_cb) / 3.0
        
        bull_threshold, bear_threshold = 0.55, 0.45
        if os.path.exists(choice_path):
            choice = joblib.load(choice_path)
            bull_threshold = choice.get('thresholds', {}).get('bull', 0.55)
            bear_threshold = choice.get('thresholds', {}).get('bear', 0.45)
        
        direction = "HOLD"
        if prob >= bull_threshold: direction = "BULLISH"
        elif prob <= bear_threshold: direction = "BEARISH"
        
        if direction != "HOLD":
            last_close = float(df['Close'].iloc[-1].item() if isinstance(df['Close'].iloc[-1], pd.Series) else df['Close'].iloc[-1])
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
                "kelly_percentage": round(fractional_kelly * 100, 2),
                "model_regime": suffix.replace('_', '') if suffix else 'standard'
            })
    print(json.dumps(signals, indent=2))
