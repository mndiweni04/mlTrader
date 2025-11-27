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
# Reads from Env or defaults to safe values
try:
    env_cap = os.environ.get("ACCOUNT_CAPITAL")
    env_risk = os.environ.get("RISK_PER_TRADE_PCT")
    ACCOUNT_CAPITAL = float(env_cap) if env_cap else 10000.0
    RISK_PER_TRADE_PCT = float(env_risk) if env_risk else 0.01
except ValueError:
    ACCOUNT_CAPITAL = 10000.0
    RISK_PER_TRADE_PCT = 0.01

MODEL_VERSION = "v3.4" # Bumped for tracking

# --- 2. CORRECT CONTRACT SPECS ---
# Value of a 1.0 point move in the asset's price
TICKER_SPECS = {
    "CL=F": 1000.0,  # Crude: $1000 per $1.00 move
    "GC=F": 100.0,   # Gold: $100 per $1.00 move
    "SI=F": 5000.0,  # Silver: $5000 per $1.00 move
    "NG=F": 10000.0, # Nat Gas: $10,000 per $1.00 move
    "ZC=F": 50.0,    # Corn: $50 per $1.00 move
    "ES=F": 50.0,    # ES: $50 per 1.00 points
    "NQ=F": 20.0,    # NQ: $20 per 1.00 points
    "EURUSD=X": 100000.0, # Forex: 1 Standard Lot = 100k units
    "JPYUSD=X": 100000.0, 
}

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

def compute_features_talib(df):
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Ensure float64
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    volume = volume.astype(np.float64)

    features = pd.DataFrame(index=df.index)
    features['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    return features

def fetch_live(ticker):
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=DATA_PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, timeout=10)
            if df is None or df.empty: raise Exception("No data")
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.title() for c in df.columns]
            if "Adj Close" in df.columns: df["Close"] = df["Adj Close"]
            
            return df 
        except:
            time.sleep(1)
    return None

def engineer_live(df, ticker_name, sp_close=None, macro_data={}):
    # (Keep your existing engineer_live logic, just ensure it returns ATR_current)
    # ... [Truncated for brevity, assuming standard logic] ...
    # RE-IMPLEMENTING MINIMAL FOR CLARITY - USE YOUR FULL VERSION IF NEEDED
    # For this snippet, I'll assume the standard one you have is fine, 
    # but I will force the ATR calculation here to be safe.
    
    df = df.copy().sort_index()
    close = df['Close'].values.astype(np.float64)
    high = df['High'].values.astype(np.float64)
    low = df['Low'].values.astype(np.float64)
    
    features_df = pd.DataFrame(index=df.index)
    # ... (Add all your MA/RSI features here as per process_data.py) ...
    # Important: Return the UNLAGGED ATR for the current day sizing
    features_df['ATR_current'] = talib.ATR(high, low, close, timeperiod=14)
    features_df['Close'] = df['Close']
    
    # Lag features for the model (except Close/ATR_current)
    # ...
    
    return features_df

# ... (Include your full engineer_live function here) ...
# For now, I will assume the one from process_data.py logic is used.
# Just ensure 'ATR_current' is the last row's ATR.

if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("\n" + "="*70)
    print(f" LIVE REPORT {MODEL_VERSION} - {now.strftime('%Y-%m-%d')}") 
    print("="*70)

    # ... (Fetch Macro Data Logic) ...
    # ... (VIX Logic) ...
    # (Assuming standard setup)
    current_vix_regime = 0 # Default Low

    # Placeholder for VIX check
    # ...

    for t in TICKERS:
        print(f"--- {t} ---")
        
        # ... (Load Model Logic) ...
        # For brevity, assuming 'model', 'scaler', 'features' are loaded
        # ...
        
        df = fetch_live(t)
        if df is None or df.empty: print("  No Data."); continue

        # Quick dirty fix to get ATR/Close without full engineer_live copy-paste
        # In production, use your full function
        close_val = df['Close'].iloc[-1]
        high_val = df['High'].iloc[-1]
        low_val = df['Low'].iloc[-1]
        # Calculate rough ATR on last 14 days if library fails, or use talib on series
        try:
            atr_series = talib.ATR(df['High'].values.astype(float), df['Low'].values.astype(float), df['Close'].values.astype(float), timeperiod=14)
            last_atr = atr_series[-1]
        except:
            last_atr = 0.0

        # --- VALIDATOR (STEP 2) ---
        if close_val <= 0:
            print(f"  ❌ REJECT: Invalid Price {close_val}")
            print("-" * 30); continue
        if last_atr <= 0 or np.isnan(last_atr):
            print(f"  ❌ REJECT: Invalid ATR {last_atr}")
            print("-" * 30); continue

        # ... (Run Prediction) ...
        # prob_bullish = ... 
        # direction = ...
        # conf_thresh_bullish = ...
        
        # DUMMY VALUES FOR EXAMPLE - Replace with your model.predict output
        prob_bullish = 0.5 # REPLACE
        direction = "HOLD" # REPLACE
        
        # --- OUTPUT ---
        print(f"  Current Price: {close_val:.4f}") # Log for debugging
        
        if direction != "HOLD":
            print(f"  Prediction (5-Day Horizon): {direction} [TRADE SIGNAL]")
            # print(f"  Confidence: ...")

            # --- SIZING LOGIC (STEP 3) ---
            atr_sl_dist = last_atr * ATR_SL_MULT
            atr_tp_dist = last_atr * ATR_TP_MULT
            
            if direction == "BULLISH":
                sl_price = close_val - atr_sl_dist
                tp_price = close_val + atr_tp_dist
            else:
                sl_price = close_val + atr_sl_dist
                tp_price = close_val - atr_tp_dist

            # Format for Parser
            print(f"  Entry Price: {close_val:.4f}") 
            print(f"  Suggested ATR SL ({ATR_SL_MULT}x): {sl_price:.4f}")
            print(f"  Suggested ATR TP ({ATR_TP_MULT}x): {tp_price:.4f}")

            dollar_per_point = TICKER_SPECS.get(t, 0.0)
            risk_dollars = ACCOUNT_CAPITAL * RISK_PER_TRADE_PCT
            
            # Risk per contract = Distance * Value_Per_Point
            risk_per_contract = atr_sl_dist * dollar_per_point
            
            final_lots = 0.0
            if risk_per_contract > 0:
                raw_lots = risk_dollars / risk_per_contract
                if "=F" in t:
                    final_lots = math.floor(raw_lots) # Integer for Futures
                else:
                    final_lots = round(raw_lots, 2)   # Float for FX

            print(f"  Risk Amount: ${risk_dollars:.2f}")
            print(f"  Suggested Lots: {final_lots}")
        else:
            print(f"  Prediction: HOLD")
            print("  Suggested Lots: 0.00")
        
        print("-" * 30)