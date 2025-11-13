# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import yfinance as yf
import talib # <-- NEW: Import TAlib

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]

TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"]

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
MODELS_DIR = "models"
PREDICTION_HORIZON = 5 
TEST_SIZE = 0.15       
VAL_SIZE = 0.15        
# (Train size will be 70%)

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- TA-Lib Free Functions (REMOVED) ---

def load_raw(ticker):
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    p = os.path.join(RAW_DIR, f"{safe_ticker}_1d_data.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0, parse_dates=True); return df.sort_index()

print("Loading all macro data...")
macro_data = {}
for mt in MACRO_TICKERS:
    df = load_raw(mt)
    if df is not None:
        macro_data[mt] = df
    else:
        print(f"Warning: Macro data for {mt} not found. Features depending on it will be 0.")

try:
    sp_df = load_raw("ES=F")
    if sp_df is None:
        print("Warning: ES=F data not found, attempting to download...")
        sp_df = yf.download("ES=F", period="10y", interval="1d", progress=False) # Match 10y
        sp_df.columns = [c.title() for c in sp_df.columns]
    SP_CLOSE = sp_df['Close'].pct_change()
except Exception as e:
    print(f"Warning: Could not load S&P 500 data. Correlation feature will be 0. Error: {e}")
    SP_CLOSE = None
            
for t in TICKERS:
    df = load_raw(t)
    if df is None:
        print(f"Skipping {t}: no raw data.")
        continue

    print(f"\nProcessing {t} ...")
    base = t.replace('=','_').lower() 

    df = df.copy().sort_index()
    
    # Save the *entire* price history for this asset
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))

    features_df = pd.DataFrame(index=df.index)
    
    # --- *** START: TALIB DATATYPE FIX *** ---
    # talib requires float64 (double) arrays.
    df['High'] = df['High'].astype(np.float64)
    df['Low'] = df['Low'].astype(np.float64)
    df['Close'] = df['Close'].astype(np.float64)
    df['Volume'] = df['Volume'].astype(np.float64)
    # --- *** END: TALIB DATATYPE FIX *** ---

    # --- START: NEW TALIB FEATURES ---
    # Get numpy arrays for talib
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values
    
    ticker_ret = talib.ROC(close, timeperiod=1) # 1-day % change
    
    # Trend
    features_df['MA5'] = talib.MA(close, timeperiod=5)
    features_df['MA20'] = talib.MA(close, timeperiod=20)
    features_df['MA50'] = talib.MA(close, timeperiod=50)
    features_df['MA200'] = talib.MA(close, timeperiod=200)
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    
    # Volatility
    features_df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    features_df['BB_Width'] = (bb_upper - bb_lower) / (bb_middle + 1e-12)
    
    # Momentum
    features_df['RSI14'] = talib.RSI(close, timeperiod=14)
    features_df['ROC10'] = talib.ROC(close, timeperiod=10) 
    features_df['MACD'], features_df['MACD_signal'], features_df['MACD_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Volume
    has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0
    if has_volume:
        features_df['OBV'] = talib.OBV(close, volume)
    # --- *** END: NEW TALIB FEATURES *** ---
    
    # --- Macro Features ---
    if "^VIX" in macro_data:
        vix_close = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        features_df['VIX_Close'] = vix_close
        features_df['VIX_Regime'] = (vix_close > 20).astype(int) 

    if "DX=F" in macro_data:
        dxy_ret = macro_data["DX=F"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
        features_df['DXY_ret_1d'] = dxy_ret
        features_df['Corr_DXY_10d'] = talib.CORREL(ticker_ret, dxy_ret.values, timeperiod=10)

    if "TLT" in macro_data:
        tlt_ret = macro_data["TLT"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
        features_df['TLT_ret_1d'] = tlt_ret
        features_df['Corr_TLT_10d'] = talib.CORREL(ticker_ret, tlt_ret.values, timeperiod=10)

    # Asset-Specific Features
    if t == "CL=F" and "XLE" in macro_data:
        xle_ret = macro_data["XLE"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
        features_df['XLE_ret_1d'] = xle_ret
        features_df['Corr_XLE_10d'] = talib.CORREL(ticker_ret, xle_ret.values, timeperiod=10)

    if t == "ZC=F":
        features_df['Month'] = df.index.month
        if "ZS=F" in macro_data:
            zs_ret = macro_data["ZS=F"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
            features_df['Corr_ZS_10d'] = talib.CORREL(ticker_ret, zs_ret.values, timeperiod=10)
        if "ZW=F" in macro_data:
            zw_ret = macro_data["ZW=F"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
            features_df['Corr_ZW_10d'] = talib.CORREL(ticker_ret, zw_ret.values, timeperiod=10)

    if t == "ES=F":
        if "XLF" in macro_data:
            xlf_ret = macro_data["XLF"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
            features_df['XLF_ret_1d'] = xlf_ret
        if "XLK" in macro_data:
            xlk_ret = macro_data["XLK"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
            features_df['XLK_ret_1d'] = xlk_ret

    if t == "NQ=F":
        if "XLK" in macro_data:
            xlk_ret = macro_data["XLK"]['Close'].pct_change().reindex(df.index, method='ffill').astype(np.float64)
            features_df['XLK_ret_1d'] = xlk_ret
    
    if SP_CLOSE is not None and t != "ES=F":
        features_df['Corr_SP500'] = talib.CORREL(ticker_ret, SP_CLOSE.reindex(df.index, method='ffill').astype(np.float64).values, timeperiod=50)

    feature_cols = list(features_df.columns) 

    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    df['label'] = (df['future_return'] > 0).astype(int) 
    
    df_full = features_df.join(df['label'])
    
    df_full[feature_cols] = df_full[feature_cols].shift(1)
    
    df_full = df_full.dropna()
    
    if df_full.empty:
        print(f"No processed rows for {t} after dropping NaNs.")
        continue
        
    df_full['label'] = df_full['label'].astype(int)

    # --- REGIME SPLITTING LOGIC (Unchanged) ---
    
    dfs_to_process = []
    
    if t in TICKERS_TO_SPLIT:
        print(f"  Found VIX_Regime and '{t}' is in split list. Splitting data...")
        df_low = df_full[df_full['VIX_Regime'] == 0]
        df_high = df_full[df_full['VIX_Regime'] == 1]
        dfs_to_process = [(df_low, "_low_vix"), (df_high, "_high_vix")]
    else:
        print(f"  '{t}' not in split list. Processing as single model.")
        dfs_to_process = [(df_full, "")] 

    for df_split, suffix in dfs_to_process:
        
        if df_split.empty:
            print(f"  Skipping regime '{suffix}': No data.")
            continue
            
        regime_base = f"{base}{suffix}"
        print(f"  Processing regime: {regime_base}")

        n = len(df_split)
        n_test = max(int(n * TEST_SIZE), 1)
        n_val = max(int((n - n_test) * VAL_SIZE), 1) 
        n_train = n - n_test - n_val 

        if n_train <= 0 or n_val <= 0 or n_test <= 0:
            print(f"  Skipping regime '{suffix}': Not enough data for train/val/test split ({n} rows).")
            continue

        df_train = df_split.iloc[:n_train]
        df_val = df_split.iloc[n_train:n_train + n_val]
        df_test = df_split.iloc[n_train + n_val:]
        
        X_train = df_train[feature_cols].values; y_train = df_train['label'].values
        X_val = df_val[feature_cols].values; y_val = df_val['label'].values
        X_test = df_test[feature_cols].values; y_test = df_test['label'].values

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        np.save(os.path.join(PROC_DIR, f"{regime_base}_X_train.npy"), X_train_s)
        np.save(os.path.join(PROC_DIR, f"{regime_base}_y_train.npy"), y_train)
        np.save(os.path.join(PROC_DIR, f"{regime_base}_X_val.npy"), X_val_s)
        np.save(os.path.join(PROC_DIR, f"{regime_base}_y_val.npy"), y_val)
        np.save(os.path.join(PROC_DIR, f"{regime_base}_X_test.npy"), X_test_s)
        np.save(os.path.join(PROC_DIR, f"{regime_base}_y_test.npy"), y_test)
        
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{regime_base}_scaler.joblib"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{regime_base}_feature_list.joblib"))
        joblib.dump(df_test.index, os.path.join(MODELS_DIR, f"{regime_base}_test_indices.joblib"))
        
        print(f"    Data shapes (train/val/test): {X_train.shape} / {X_val.shape} / {X_test.shape}")
        
        class_counts = np.bincount(y_train)
        print(f"    Class Balance (Train): {class_counts}")
        if len(class_counts) < 2 or np.min(class_counts) == 0:
            print(f"    WARNING: Regime '{suffix}' has only one class. Model will likely fail.")

print("\n" + "="*50)
print(" ✅ process_data.py (v6 - TALIB) FINISHED ")
print("="*50 + "\n")