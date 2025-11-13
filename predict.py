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
import time # <-- *** THIS IS THE FIX ***

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except (ImportError, AttributeError):
    pass

# --- 1. *** USER SETTINGS: EDIT THESE *** ---
ACCOUNT_CAPITAL = 50000.0 # Example: $50,000
RISK_PER_TRADE_PCT = 0.01   # Example: 1% risk per "full" trade
# --- END USER SETTINGS ---

# --- 2. Contract/Lot Specifications (Dollar value of a 1.0 point move) ---
TICKER_SPECS = {
    "CL=F": 1000.0,  # Crude Oil: $1000 per $1.00 move
    "GC=F": 100.0,   # Gold: $100 per $1.00 move
    "SI=F": 5000.0,  # Silver: $5000 per $1.00 move
    "NG=F": 10000.0, # Natural Gas: $10,000 per $1.00 move
    "ZC=F": 50.0,    # Corn: $50 per $1.00 move
    "ES=F": 50.0,    # E-mini S&P: $50 per $1.00 move
    "NQ=F": 20.0,    # E-mini NASDAQ: $20 per $1.00 move
    "EURUSD=X": 100000.0, # Forex: 1 Standard Lot = 100,000 units
    "JPYUSD=X": 100000.0, # Forex: 1 Standard Lot = 100,000 units
}
# --- END SPECS ---

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"] 

INTERVAL = "1d"
DATA_PERIOD = "250d" 
MODELS_DIR = "models"
ATR_MULTIPLIER = 1.5

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

# --- Position Sizing Function ---
def calculate_position_size(probability, min_thresh, max_thresh=1.0):
    if probability < min_thresh:
        return 0.0
    
    size = (probability - min_thresh) / (max_thresh - min_thresh)
    return np.clip(size, 0.0, 1.0) 

def fetch_live(ticker):
    # --- Add retry logic ---
    for attempt in range(3): # Try 3 times
        try:
            df = yf.download(ticker, period=DATA_PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, timeout=10) # 10-sec timeout
            if df is None or df.empty: 
                raise Exception("No data returned")
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.title() for c in df.columns]
            if "Adj Close" in df.columns and "Close" not in df.columns:
                df["Close"] = df["Adj Close"]
            if 'Open' not in df.columns: df['Open'] = df['Close']
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']
            if 'Volume' not in df.columns: df['Volume'] = 0
                
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None: df.index = df.index.tz_localize('UTC')
            else: df.index = df.index.tz_convert('UTC')
            
            return df # Success
        
        except Exception as e:
            print(f"  Fetch failed for {ticker} (Attempt {attempt+1}/3): {e}")
            time.sleep(2) # Wait 2 seconds before retrying
            
    print(f"  --- Giving up on {ticker} after 3 attempts. ---")
    return None # Return None after all attempts fail

def engineer_live(df, ticker_name, sp_close=None, macro_data={}):
    df = df.copy().sort_index()
    
    close = df['Close']
    ticker_ret = close.pct_change() 
    features_df = pd.DataFrame(index=df.index)
    
    # Trend
    features_df['MA5'] = close.rolling(5).mean()
    features_df['MA20'] = close.rolling(20).mean()
    features_df['MA50'] = close.rolling(50).mean()
    features_df['MA200'] = close.rolling(200).mean()
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    
    # Volatility
    features_df['ATR'] = compute_atr(df, n=14)
    features_df['BB_Width'] = compute_bb_width(close, n=20)
    
    # Momentum
    features_df['RSI14'] = compute_rsi(close, period=14)
    features_df['ROC10'] = compute_roc(close, n=10) 
    features_df['MACD'], features_df['MACD_signal'], features_df['MACD_hist'] = compute_macd(close)
    
    # Volume
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features_df['OBV'] = compute_obv(close, df['Volume'])
    
    # --- Macro Features ---
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
    print(" LIVE 5-DAY DIRECTIONAL REPORT (LEAKAGE-FREE) V3 - ENSEMBLE AWARE") 
    print("="*70)
    print(f"Generated (SAST): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    print("Fetching live macro data...")
    live_macro_data = {}
    for mt in MACRO_TICKERS:
        df_m = fetch_live(mt)
        if df_m is not None:
            df_m['PctChange'] = df_m['Close'].pct_change()
            live_macro_data[mt] = df_m
        else:
            print(f"  Warning: Failed to fetch live macro data for {mt}")
    print("Macro data fetch complete.\n")
            
    sp_df = fetch_live("ES=F")
    sp_close = sp_df['Close'].pct_change() if sp_df is not None else None
    
    current_vix_regime = 0 
    if "^VIX" in live_macro_data:
        try:
            last_vix_close = live_macro_data["^VIX"]['Close'].iloc[-1]
            if last_vix_close > 20:
                current_vix_regime = 1
                print(f"  Current VIX: {last_vix_close:.2f}. Using HIGH-VIX models.")
            else:
                print(f"  Current VIX: {last_vix_close:.2f}. Using LOW-VIX models.")
        except Exception as e:
            print(f"  Warning: Could not determine VIX regime. Defaulting to LOW. Error: {e}")
    else:
        print("  Warning: VIX data not found. Defaulting to LOW-VIX models.")

    for t in TICKERS:
        print(f"--- {t} ---")
        base = t.replace('=','_').lower()
        
        # --- START: DYNAMIC MODEL/FILE SELECTION ---
        suffix = ""
        if t in TICKERS_TO_SPLIT:
            if current_vix_regime == 1:
                suffix = "_high_vix"
                print("  (VIX REGIME: HIGH)")
            else:
                suffix = "_low_vix"
                print("  (VIX REGIME: LOW)")
        
        regime_base = f"{base}{suffix}"
        
        choice_file = os.path.join(MODELS_DIR, f"{regime_base}_model_choice.joblib")
        try:
            model_choice = joblib.load(choice_file)
            chosen_model_type = model_choice['model_type']
            conf_thresh_bullish = model_choice['thresholds']['bull']
            conf_thresh_bearish = model_choice['thresholds']['bear']
            print(f"  Loaded dynamic choice: Model={chosen_model_type}, Thresh={conf_thresh_bullish*100:.0f}% / {conf_thresh_bearish*100:.0f}%")
        except Exception as e:
            print(f"  Warning: Could not load model choice for '{regime_base}' ({e}). Holding.")
            print("-" * 30)
            continue
            
        if chosen_model_type == 'none':
            print(f"  No profitable model found for {regime_base}. Holding.")
            print("-" * 30)
            continue
            
        scaler_file = os.path.join(MODELS_DIR, f"{regime_base}_scaler.joblib")
        feature_file = os.path.join(MODELS_DIR, f"{regime_base}_feature_list.joblib")
        
        if not all(os.path.exists(f) for f in [scaler_file, feature_file]):
            print(f"  Missing scaler/features for '{regime_base}'. Check if model was trained.")
            print("-" * 30)
            continue
            
        scaler = joblib.load(scaler_file)
        features = joblib.load(feature_file)
        
        df = fetch_live(t)
        if df is None or df.empty:
            print(f"  No live data for {t}.")
            print("-" * 30)
            continue

        df_feat = engineer_live(df, t, sp_close, live_macro_data)
        
        for f in features:
            if f not in df_feat.columns:
                df_feat[f] = 0.0
        
        df_feat_ordered = df_feat[features]
        latest = df_feat_ordered.iloc[-1]  
        
        if latest.isnull().any():
            print("  Latest data has NaNs (from lagging/MAs), cannot predict.")
            continue
            
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
                
                prob_xgb = model_xgb.predict_proba(X_scaled)[0][1]
                prob_lr = model_lr.predict_proba(X_scaled)[0][1]
                prob_bullish = (prob_xgb + prob_lr) / 2.0
                
        except Exception as e:
            print(f"  Error loading chosen model file: {e}. Skipping.")
            print("-" * 30)
            continue

        model_name = f"{chosen_model_type.upper()} ({regime_base})"
        last_close = df_feat.iloc[-1]['Close'] 
        
        print(f"  Chosen model: {model_name}")
        
        current_rr_ratio = 1.5 
        raw_confidence = 0.0
        confidence_size_factor = 0.0 

        if prob_bullish >= conf_thresh_bullish: 
            direction = "BULLISH"
            raw_confidence = prob_bullish
            confidence_pct = raw_confidence * 100
            confidence_size_factor = calculate_position_size(raw_confidence, conf_thresh_bullish) 
            
            if raw_confidence >= 0.75:
                current_rr_ratio = 2.5
            else:
                current_rr_ratio = 1.5

            print(f"  Prediction (5-Day Horizon): {direction} [TRADE SIGNAL]")
            print(f"  Confidence: {confidence_pct:.2f}% (>= {conf_thresh_bullish*100:.0f}%)")
        
        elif prob_bullish <= conf_thresh_bearish: 
            direction = "BEARISH"
            raw_confidence = 1 - prob_bullish
            confidence_pct = raw_confidence * 100
            
            min_bear_conf = 1.0 - conf_thresh_bearish 
            confidence_size_factor = calculate_position_size(raw_confidence, min_bear_conf)
            
            if raw_confidence >= 0.75:
                current_rr_ratio = 2.5
            else:
                current_rr_ratio = 1.5

            print(f"  Prediction (5-Day Horizon): {direction} [TRADE SIGNAL]")
            print(f"  Confidence: {confidence_pct:.2f}% (>= {(1-conf_thresh_bearish)*100:.0f}%)")
        
        else:
            direction = "HOLD"
            print(f"  Prediction (5-Day Horizon): {direction} [HOLD / NO SIGNAL]")
            print(f"  (Prob: {prob_bullish*100:.1f}%, inside {conf_thresh_bearish*100:.0f}-{conf_thresh_bullish*100:.0f} dead-zone)")

        print(f"  Current Price: {last_close:.6f}")

        last_atr = df_feat.iloc[-1].get('ATR_current', np.nan) 
        if not np.isnan(last_atr) and last_atr > 0 and direction != "HOLD":
            atr_amount = last_atr * ATR_MULTIPLIER
            
            if direction == "BULLISH":
                atr_sl = last_close - atr_amount
                atr_tp = last_close + (atr_amount * current_rr_ratio) 
                print(f"  Suggested ATR SL ({ATR_MULTIPLIER}x): {atr_sl:.6f}")
                print(f"  Suggested ATR TP ({current_rr_ratio}x R/R): {atr_tp:.6f}")
            else: # BEARISH
                atr_sl = last_close + atr_amount
                atr_tp = last_close - (atr_amount * current_rr_ratio)
                print(f"  Suggested ATR SL ({ATR_MULTIPLIER}x): {atr_sl:.6f}")
                print(f"  Suggested ATR TP ({current_rr_ratio}x R/R): {atr_tp:.6f}")
            
            try:
                dollar_per_point = TICKER_SPECS.get(t)
                if dollar_per_point is None:
                    print(f"  Suggested Lots: N/A (No TICKER_SPECS entry for {t})")
                else:
                    risk_dollars_per_full_trade = ACCOUNT_CAPITAL * RISK_PER_TRADE_PCT
                    risk_per_contract = atr_amount * dollar_per_point
                    
                    if risk_per_contract <= 0:
                        print("  Suggested Lots: N/A (ATR risk is zero)")
                    else:
                        full_position_lots = risk_dollars_per_full_trade / risk_per_contract
                        final_lots = full_position_lots * confidence_size_factor
                        
                        unit = "Contracts" if "=F" in t else "Std. Lots"
                        print(f"  Suggested Lots: {final_lots:.2f} {unit}")
            except Exception as e:
                print(f"  Error in lot calculation: {e}")
        else:
            print("  ATR/TP/SL: N/A (Holding)")
            print("  Suggested Lots: 0.00 Contracts") 

        print("-" * 30)

    print("\n" + "="*70)
    print(" End of Live Market Direction Report")
    print("="*70)