# predict.py
import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
import json

# --- FAIL-FAST PROTOCOL ---
try:
    import talib
except ImportError:
    sys.exit("CRITICAL ERROR: TA-Lib is missing. Terminating predict.py to prevent feature drift.")

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_VERSION = "v4.0"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"]
MODELS_DIR = "models"
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
DATA_PERIOD = "250d"
INTERVAL = "1d"

def fetch_live(ticker):
    """Robust data fetching with retries."""
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=DATA_PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, timeout=10)
            if df is None or df.empty: raise Exception("Empty dataframe")
            
            # Formatting
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.title() for c in df.columns]
            
            # FX/Index cleanup
            if 'Volume' not in df.columns: df['Volume'] = 0
            if 'Open' not in df.columns: df['Open'] = df['Close']
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']
            
            # Timezone
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
            
            return df
        except Exception as e:
            time.sleep(1)
    return None

def engineer_live(df, ticker_name, sp_close=None, macro_data={}):
    """
    Strictly matches process_data.py math using TA-Lib.
    """
    df = df.copy().sort_index()
    
    # --- EXPLICIT DATA TYPING (Float64 for TA-Lib) ---
    c = df['Close'].astype(np.float64).values
    h = df['High'].astype(np.float64).values
    l = df['Low'].astype(np.float64).values
    v = df['Volume'].astype(np.float64).values
    
    # Helper for returns (used in correlations)
    ticker_ret = df['Close'].pct_change().fillna(0).astype(np.float64).values
    
    feat = pd.DataFrame(index=df.index)
    
    # --- TA-LIB INDICATORS ---
    feat['MA5'] = talib.MA(c, 5)
    feat['MA20'] = talib.MA(c, 20)
    feat['MA50'] = talib.MA(c, 50)
    feat['MA200'] = talib.MA(c, 200)
    feat['MA_diff'] = feat['MA50'] - feat['MA200']
    
    feat['ATR'] = talib.ATR(h, l, c, 14)
    u, m, lo = talib.BBANDS(c, 20, 2, 2, 0)
    feat['BB_Width'] = (u - lo) / (m + 1e-12)
    
    feat['RSI14'] = talib.RSI(c, 14)
    feat['ROC10'] = talib.ROC(c, 10)
    feat['MACD'], feat['MACD_signal'], feat['MACD_hist'] = talib.MACD(c)
    
    if np.sum(v) > 0:
        feat['OBV'] = talib.OBV(c, v)
    
    # --- NEW: Intraday Volatility Features for Quiet Markets (ES=F, NQ=F) ---
    # Day Range Pct: (High - Low) / Close (captures intraday volatility)
    feat['Day_Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
    
    # Distance from MA20: (Close - MA20) / MA20 (captures price position relative to trend)
    feat['Dist_from_MA20'] = (df['Close'] - feat['MA20']) / (feat['MA20'] + 1e-12)
        
    # --- MACRO LOGIC (Matches process_data.py) ---
    if "^VIX" in macro_data:
        v_c = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        feat['VIX_Close'] = v_c
        feat['VIX_Regime'] = (v_c > 20).astype(int)

    # Helper: Macro Correlation Calc
    def calc_corr(series_ret, name):
        # Align macro returns to ticker index
        aligned_macro = series_ret.reindex(df.index, method='ffill').fillna(0).astype(np.float64).values
        # Use TA-Lib CORREL
        return talib.CORREL(ticker_ret, aligned_macro, 10)

    if "DX=F" in macro_data:
        dxy = macro_data["DX=F"]['Close'].pct_change()
        feat['DXY_ret_1d'] = dxy.reindex(df.index, method='ffill')
        feat['Corr_DXY_10d'] = calc_corr(dxy, 'DXY')

    if "TLT" in macro_data:
        tlt = macro_data["TLT"]['Close'].pct_change()
        feat['TLT_ret_1d'] = tlt.reindex(df.index, method='ffill')
        feat['Corr_TLT_10d'] = calc_corr(tlt, 'TLT')

    if ticker_name == "CL=F" and "XLE" in macro_data:
        xle = macro_data["XLE"]['Close'].pct_change()
        feat['XLE_ret_1d'] = xle.reindex(df.index, method='ffill')
        feat['Corr_XLE_10d'] = calc_corr(xle, 'XLE')
        
    if ticker_name == "ZC=F":
        feat['Month'] = df.index.month
        if "ZS=F" in macro_data:
            feat['Corr_ZS_10d'] = calc_corr(macro_data["ZS=F"]['Close'].pct_change(), 'ZS')
        if "ZW=F" in macro_data:
            feat['Corr_ZW_10d'] = calc_corr(macro_data["ZW=F"]['Close'].pct_change(), 'ZW')
            
    if ticker_name in ["ES=F", "NQ=F"]:
        if "XLK" in macro_data:
            feat['XLK_ret_1d'] = macro_data["XLK"]['Close'].pct_change().reindex(df.index, method='ffill')
    
    if sp_close is not None and ticker_name != "ES=F":
         # Align SP500 returns
        aligned_sp = sp_close.reindex(df.index, method='ffill').fillna(0).astype(np.float64).values
        feat['Corr_SP500'] = talib.CORREL(ticker_ret, aligned_sp, 50)
    
    # --- CRITICAL: T-1 ALIGNMENT ---
    # Shift features by 1 to match "Yesterday's data predicts Today"
    lagged_feat = feat.shift(1)
    
    # --- EXIT LOGIC ---
    # We need CURRENT ATR for Stops/Targets, so we calculate it on the *unshifted* data
    # (ATR for Day T is known at Close of Day T)
    lagged_feat['ATR_current'] = feat['ATR'] 
    
    return lagged_feat

if __name__ == "__main__":
    signals_list = []
    
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    timestamp = now.isoformat()
    
    print(f"[PREDICT] Starting signal generation at {timestamp}")
    
    print("Fetching live macro data...")
    macros = {mt: fetch_live(mt) for mt in MACRO_TICKERS}
    
    vix_value = None
    vix_regime = 0
    if macros.get("^VIX") is not None:
        vix_value = float(macros["^VIX"]['Close'].iloc[-1])
        vix_regime = 1 if vix_value > 20 else 0
        print(f"[VIX] Value: {vix_value:.2f}, Regime: {vix_regime} (0=low, 1=high)")

    sp_df = fetch_live("ES=F")
    sp_close = sp_df['Close'].pct_change() if sp_df is not None else None

    for t in TICKERS:
        base = t.replace('=','_').lower()
        suffix = ("_high_vix" if vix_regime else "_low_vix") if t in TICKERS_TO_SPLIT else ""
        rb = f"{base}{suffix}"
        model_regime = rb
        
        choice_path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
        if not os.path.exists(choice_path): 
            continue
        choice = joblib.load(choice_path)
        if choice['model_type'] == 'none': 
            continue
        
        df = fetch_live(t)
        if df is None: 
            continue
        
        feat_df = engineer_live(df, t, sp_close, macros)
        
        features = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        
        # Ensure columns exist
        for f in features:
            if f not in feat_df.columns: 
                feat_df[f] = 0.0
            
        latest = feat_df.iloc[-1]
        current_atr = latest['ATR_current']
        
        X_raw = latest[features].values.reshape(1, -1)
        
        # Check for NaNs (e.g. if not enough history)
        if np.isnan(X_raw).any(): 
            print(f"[{t}] Skipping: NaNs in features (insufficient history)")
            continue
        
        scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{choice['model_type']}_calibrated.joblib"))
        
        prob = model.predict_proba(scaler.transform(X_raw))[0][1]
        
        direction = "HOLD"
        if prob >= choice['thresholds']['bull']: 
            direction = "BULLISH"
        elif prob <= choice['thresholds']['bear']: 
            direction = "BEARISH"

        if direction != "HOLD":
            last_close = float(df['Close'].iloc[-1])
            sl = last_close - (current_atr * ATR_SL_MULT) if direction == "BULLISH" else last_close + (current_atr * ATR_SL_MULT)
            tp = last_close + (current_atr * ATR_TP_MULT) if direction == "BULLISH" else last_close - (current_atr * ATR_TP_MULT)
            
            signal = {
                "ticker": t,
                "direction": direction,
                "confidence": float(prob),
                "entry_price": last_close,
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "model_regime": model_regime,
                "model_type": choice['model_type'],
                "model_version": MODEL_VERSION,
                "timestamp": timestamp,
                "vix_value": vix_value,
                "vix_regime": vix_regime,
                "atr_current": float(current_atr)
            }
            
            signals_list.append(signal)
            print(f"[{t}] {direction} signal detected (confidence: {prob:.2%})")

    # --- OUTPUT STRUCTURED JSON ---
    output_payload = {
        "timestamp": timestamp,
        "model_version": MODEL_VERSION,
        "vix_value": vix_value,
        "vix_regime": vix_regime,
        "signals_count": len(signals_list),
        "signals": signals_list
    }
    
    print(f"\n[OUTPUT] {len(signals_list)} signal(s) generated")
    print(json.dumps(output_payload, indent=2))