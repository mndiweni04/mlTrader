# predict.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import yfinance as yf
import time
import math
import talib

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except (ImportError, AttributeError):
    pass

# --- 1. CONFIGURATION ---
MODEL_VERSION = "v3.6" 
# -----------------------------

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"] 

INTERVAL = "1d"
DATA_PERIOD = "250d" 
MODELS_DIR = "models"

# --- EXITS ---
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5

DEFAULT_CONF_THRESH_BULLISH = 0.60
DEFAULT_CONF_THRESH_BEARISH = 0.40

# --- TA-Lib Free Functions ---
def compute_atr(df, n=14):
    high = df['High']; low = df['Low']; close = df['Close']
    tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def compute_rsi(series, period=14):
    delta = series.diff(); up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean(); ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-12); return 100 - (100 / (1 + rs))

def compute_bb_width(close, n=20, ndev=2):
    middle = close.rolling(n, min_periods=1).mean() 
    std = close.rolling(n, min_periods=1).std()
    upper = middle + (std * ndev); lower = middle - (std * ndev)
    return (upper - lower) / (middle + 1e-12)

def compute_macd(close, fast=12, slow=26, signal=9):
    exp1 = close.ewm(span=fast, adjust=False).mean(); exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2; macdsignal = macd.ewm(span=signal, adjust=False).mean(); macdhist = macd - macdsignal
    return macd, macdsignal, macdhist

def compute_roc(close, n=10):
    return (close - close.shift(n)) / (close.shift(n) + 1e-12)

def compute_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def fetch_live(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=DATA_PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, timeout=10)
            if df is None or df.empty: raise Exception("No data returned")
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.title() for c in df.columns]
            
            if "Adj Close" in df.columns and "Close" not in df.columns: df["Close"] = df["Adj Close"]
            if 'Open' not in df.columns: df['Open'] = df['Close']
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']
            if 'Volume' not in df.columns: df['Volume'] = 0
                
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
            return df 
        except Exception as e:
            print(f"  Fetch failed for {ticker} (Attempt {attempt+1}/3): {e}")
            time.sleep(2)
    return None

def engineer_live(df, ticker_name, sp_close=None, macro_data={}):
    df = df.copy().sort_index()
    close = df['Close']
    ticker_ret = close.pct_change() 
    features_df = pd.DataFrame(index=df.index)
    
    features_df['MA5'] = close.rolling(5).mean()
    features_df['MA20'] = close.rolling(20).mean()
    features_df['MA50'] = close.rolling(50).mean()
    features_df['MA200'] = close.rolling(200).mean()
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    
    features_df['ATR'] = compute_atr(df, n=14)
    features_df['BB_Width'] = compute_bb_width(close, n=20)
    features_df['RSI14'] = compute_rsi(close, period=14)
    features_df['ROC10'] = compute_roc(close, n=10) 
    features_df['MACD'], features_df['MACD_signal'], features_df['MACD_hist'] = compute_macd(close)
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features_df['OBV'] = compute_obv(close, df['Volume'])
    
    if "^VIX" in macro_data:
        vix_close = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill')
        features_df['VIX_Close'] = vix_close
        features_df['VIX_Regime'] = (vix_close > 20).astype(int) 

    if "DX=F" in macro_data:
        dxy_ret = macro_data["DX=F"]['PctChange'].reindex(df.index, method='ffill')
        features_df['DXY_ret_1d'] = dxy_ret
        features_df['Corr_DXY_10d'] = ticker_ret.rolling(10).corr(dxy_ret)

    if "TLT" in macro_data:
        tlt_ret = macro_data["TLT"]['PctChange'].reindex(df.index, method='ffill')
        features_df['TLT_ret_1d'] = tlt_ret
        features_df['Corr_TLT_10d'] = ticker_ret.rolling(10).corr(tlt_ret)

    if ticker_name == "CL=F" and "XLE" in macro_data:
        xle_ret = macro_data["XLE"]['PctChange'].reindex(df.index, method='ffill')
        features_df['XLE_ret_1d'] = xle_ret
        features_df['Corr_XLE_10d'] = ticker_ret.rolling(10).corr(xle_ret)

    if ticker_name == "ZC=F":
        features_df['Month'] = df.index.month
        if "ZS=F" in macro_data:
            zs_ret = macro_data["ZS=F"]['PctChange'].reindex(df.index, method='ffill')
            features_df['Corr_ZS_10d'] = ticker_ret.rolling(10).corr(zs_ret)
        if "ZW=F" in macro_data:
            zw_ret = macro_data["ZW=F"]['PctChange'].reindex(df.index, method='ffill')
            features_df['Corr_ZW_10d'] = ticker_ret.rolling(10).corr(zw_ret)

    if ticker_name == "ES=F":
        if "XLF" in macro_data:
            xlf_ret = macro_data["XLF"]['PctChange'].reindex(df.index, method='ffill')
            features_df['XLF_ret_1d'] = xlf_ret
        if "XLK" in macro_data:
            xlk_ret = macro_data["XLK"]['PctChange'].reindex(df.index, method='ffill')
            features_df['XLK_ret_1d'] = xlk_ret

    if ticker_name == "NQ=F":
        if "XLK" in macro_data:
            xlk_ret = macro_data["XLK"]['PctChange'].reindex(df.index, method='ffill')
            features_df['XLK_ret_1d'] = xlk_ret
    
    if sp_close is not None and ticker_name != "ES=F":
        sp_close_reindexed = sp_close.reindex(df.index, method='ffill')
        features_df['Corr_SP500'] = ticker_ret.rolling(50).corr(sp_close_reindexed)
    
    features_df = features_df.shift(1)
    
    features_df['Close'] = df['Close']
    features_df['ATR_current'] = compute_atr(df, n=14) 
    
    return features_df

if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("\n" + "="*70)
    print(" LIVE 5-DAY DIRECTIONAL REPORT (LEAKAGE-FREE) V3.6 - SIGNAL ONLY") 
    print("="*70)
    print(f"Generated (SAST): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    print(f"Model Version: {MODEL_VERSION}")

    print("Fetching live macro data...")
    live_macro_data = {}
    for mt in MACRO_TICKERS:
        df_m = fetch_live(mt)
        if df_m is not None:
            df_m['PctChange'] = df_m['Close'].pct_change()
            live_macro_data[mt] = df_m
    print("Macro data fetch complete.\n")
            
    sp_df = fetch_live("ES=F")
    sp_close = sp_df['Close'].pct_change() if sp_df is not None else None
    
    current_vix_regime = 0 
    if "^VIX" in live_macro_data:
        try:
            vix_val = live_macro_data["^VIX"]['Close'].iloc[-1]
            if vix_val > 20: current_vix_regime = 1
            print(f"  Current VIX: {vix_val:.2f}. Using {'HIGH' if current_vix_regime else 'LOW'}-VIX models.")
        except Exception: pass

    for t in TICKERS:
        print(f"--- {t} ---")
        base = t.replace('=','_').lower()
        suffix = ""
        if t in TICKERS_TO_SPLIT:
            suffix = "_high_vix" if current_vix_regime == 1 else "_low_vix"
            print(f"  (VIX REGIME: {'HIGH' if current_vix_regime else 'LOW'})")
        
        regime_base = f"{base}{suffix}"
        
        choice_file = os.path.join(MODELS_DIR, f"{regime_base}_model_choice.joblib")
        try:
            model_choice = joblib.load(choice_file)
            chosen_model_type = model_choice['model_type']
            conf_thresh_bullish = model_choice['thresholds']['bull']
            conf_thresh_bearish = model_choice['thresholds']['bear']
            print(f"  Loaded dynamic choice: Model={chosen_model_type}, Thresh={conf_thresh_bullish*100:.0f}% / {conf_thresh_bearish*100:.0f}%")
        except Exception:
            print(f"  Warning: Could not load model choice for '{regime_base}'. Holding.")
            continue
            
        if chosen_model_type == 'none':
            print(f"  No profitable model found for {regime_base}. Holding.")
            print("-" * 30); continue
            
        scaler_file = os.path.join(MODELS_DIR, f"{regime_base}_scaler.joblib")
        feature_file = os.path.join(MODELS_DIR, f"{regime_base}_feature_list.joblib")
        
        if not all(os.path.exists(f) for f in [scaler_file, feature_file]):
            print(f"  Missing scaler/features for '{regime_base}'."); continue
            
        scaler = joblib.load(scaler_file)
        features = joblib.load(feature_file)
        
        df = fetch_live(t)
        if df is None or df.empty:
            print(f"  No live data for {t}."); print("-" * 30); continue

        df_feat = engineer_live(df, t, sp_close, live_macro_data)
        for f in features:
            if f not in df_feat.columns: df_feat[f] = 0.0
        
        df_feat_ordered = df_feat[features]
        latest = df_feat_ordered.iloc[-1]  
        
        if latest.isnull().any():
            print("  Latest data has NaNs (from lagging/MAs), cannot predict."); continue
            
        X_latest = latest.values.reshape(1, -1)
        X_scaled = scaler.transform(X_latest)
        
        prob_bullish = 0.0
        try:
            if chosen_model_type == 'xgb':
                model = joblib.load(os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib"))
                prob_bullish = model.predict_proba(X_scaled)[0][1]
            elif chosen_model_type == 'lr':
                model = joblib.load(os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib"))
                prob_bullish = model.predict_proba(X_scaled)[0][1]
            elif chosen_model_type == 'ensemble':
                model_xgb = joblib.load(os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib"))
                model_lr = joblib.load(os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib"))
                prob_bullish = (model_xgb.predict_proba(X_scaled)[0][1] + model_lr.predict_proba(X_scaled)[0][1]) / 2.0
        except Exception as e:
            print(f"  Error loading chosen model: {e}. Skipping."); continue

        # --- REMOVED VIX VETO LOGIC ---

        model_name = f"{chosen_model_type.upper()} ({regime_base})"
        
        # --- FIX: Use raw DataFrame Close for Entry Price ---
        last_close = df['Close'].iloc[-1] 
        
        print(f"  Chosen model: {model_name}")
        
        direction = "HOLD"
        if prob_bullish >= conf_thresh_bullish: direction = "BULLISH"
        elif prob_bullish <= conf_thresh_bearish: direction = "BEARISH"

        if direction != "HOLD":
            print(f"  Prediction (5-Day Horizon): {direction} [TRADE SIGNAL]")
            conf_pct = prob_bullish if direction == "BULLISH" else (1 - prob_bullish)
            print(f"  Confidence: {conf_pct*100:.2f}% (>= {conf_thresh_bullish*100:.0f}%)")

            last_atr = df_feat.iloc[-1].get('ATR_current', np.nan)
            if np.isnan(last_atr) or last_atr <= 0:
                print("  Error: Invalid ATR. Skipping.")
                print("-" * 30); continue

            # --- EXITS ---
            atr_sl_dist = last_atr * ATR_SL_MULT
            atr_tp_dist = last_atr * ATR_TP_MULT
            
            if direction == "BULLISH":
                sl_price = last_close - atr_sl_dist
                tp_price = last_close + atr_tp_dist
            else:
                sl_price = last_close + atr_sl_dist
                tp_price = last_close - atr_tp_dist
            
            print(f"  Entry Price: {last_close:.4f}")
            print(f"  Suggested ATR SL ({ATR_SL_MULT}x): {sl_price:.4f}")
            print(f"  Suggested ATR TP ({ATR_TP_MULT}x): {tp_price:.4f}")
            
            # --- REMOVED RISK / LOT CALCULATIONS ---
        else:
            print(f"  Prediction: HOLD (Prob: {prob_bullish*100:.1f}%)")
            
        print("-" * 30)

    print("\n" + "="*70)
    print(" End of Live Market Direction Report")
    print("="*70)