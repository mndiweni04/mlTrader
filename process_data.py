# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
import yfinance as yf
import talib

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

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_raw(ticker):
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    p = os.path.join(RAW_DIR, f"{safe_ticker}_1d_data.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.sort_index()

print("Loading all macro data...")
macro_data = {}
for mt in MACRO_TICKERS:
    df = load_raw(mt)
    if df is not None:
        macro_data[mt] = df

try:
    sp_df = load_raw("ES=F")
    if sp_df is None:
        sp_df = yf.download("ES=F", period="10y", interval="1d", progress=False)
        sp_df.columns = [c.title() for c in sp_df.columns]
    SP_CLOSE = sp_df['Close'].pct_change()
except Exception as e:
    SP_CLOSE = None
            
for t in TICKERS:
    df = load_raw(t)
    if df is None: continue

    print(f"\nProcessing {t} ...")
    base = t.replace('=','_').lower() 
    df = df.copy().sort_index()
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))

    for col in ['High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(np.float64)

    features_df = pd.DataFrame(index=df.index)
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    ticker_ret = talib.ROC(close, timeperiod=1)
    
    features_df['MA5'] = talib.MA(close, 5)
    features_df['MA20'] = talib.MA(close, 20)
    features_df['MA50'] = talib.MA(close, 50)
    features_df['MA200'] = talib.MA(close, 200)
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    features_df['ATR'] = talib.ATR(high, low, close, 14)
    u, m, lo = talib.BBANDS(close, 20, 2, 2, 0)
    features_df['BB_Width'] = (u - lo) / (m + 1e-12)
    features_df['RSI14'] = talib.RSI(close, 14)
    features_df['ROC10'] = talib.ROC(close, 10)
    features_df['MACD'], features_df['MACD_signal'], features_df['MACD_hist'] = talib.MACD(close)
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features_df['OBV'] = talib.OBV(close, volume)
    
    features_df['Day_Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
    features_df['Dist_from_MA20'] = (df['Close'] - features_df['MA20']) / (features_df['MA20'] + 1e-12)

    # Handle indicator warm-up period to avoid dropping too much data
    features_df = features_df.ffill().bfill()

    if "^VIX" in macro_data:
        vix_c = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        features_df['VIX_Close'] = vix_c
        features_df['VIX_Regime'] = (vix_c > 20).astype(int)

    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    atr_values = talib.ATR(high, low, close, 14)
    barrier_threshold = atr_values * 0.20
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier_threshold, 'label'] = 1
    df.loc[df['future_return'] < -barrier_threshold, 'label'] = 0
    
    df_full = features_df.join(df[['label']])
    valid_count = df_full.dropna().shape[0]
    
    # Fallback labeling if ATR barrier is too sparse
    if valid_count < 100:
        print(f"  [INFO] Sparse data fallback for {t} ({valid_count} rows)")
        df['label'] = (df['future_return'] > 0).astype(int)
        df_full = features_df.join(df[['label']])

    feature_cols = list(features_df.columns)
    df_full[feature_cols] = df_full[feature_cols].shift(1)
    df_full = df_full.dropna()
    
    if df_full.empty: continue
    df_full['label'] = df_full['label'].astype(int)

    # Processing split for regime-specific assets
    dfs_to_process = []
    if t in TICKERS_TO_SPLIT:
        df_low = df_full[df_full['VIX_Regime'] == 0]
        df_high = df_full[df_full['VIX_Regime'] == 1]
        dfs_to_process = [(df_low, "_low_vix"), (df_high, "_high_vix")]
    else:
        dfs_to_process = [(df_full, "")] 

    for df_split, suffix in dfs_to_process:
        if df_split.empty: continue
            
        regime_base = f"{base}{suffix}"
        n = len(df_split)
        n_test = max(int(n * TEST_SIZE), 1)
        n_val = max(int((n - n_test) * VAL_SIZE), 1) 
        n_train = n - n_test - n_val 

        if n_train <= 0: continue

        df_train = df_split.iloc[:n_train]
        df_val = df_split.iloc[n_train:n_train + n_val]
        df_test = df_split.iloc[n_train + n_val:]
        
        y_train_full = df_train['label'].values
        mask_valid_train = ~np.isnan(y_train_full)
        df_train_labeled = df_train[mask_valid_train].copy()
        
        class_0_mask = df_train_labeled['label'].astype(int) == 0
        class_1_mask = df_train_labeled['label'].astype(int) == 1
        n_class_0, n_class_1 = np.sum(class_0_mask), np.sum(class_1_mask)
        
        if n_class_0 > 0 and n_class_1 > 0:
            min_size = min(n_class_0, n_class_1)
            df_c0 = resample(df_train_labeled[class_0_mask], n_samples=min_size, random_state=42, replace=False)
            df_c1 = resample(df_train_labeled[class_1_mask], n_samples=min_size, random_state=42, replace=False)
            df_train_bal = pd.concat([df_c0, df_c1], axis=0).sort_index()
            
            X_train = df_train_bal[feature_cols].values
            y_train = df_train_bal['label'].astype(int).values
            X_val = df_val[feature_cols].values
            y_val = df_val['label'].astype(int).values
            X_test = df_test[feature_cols].values
            y_test = df_test['label'].astype(int).values

            scaler = StandardScaler().fit(X_train)
            np.save(os.path.join(PROC_DIR, f"{regime_base}_X_train.npy"), scaler.transform(X_train))
            np.save(os.path.join(PROC_DIR, f"{regime_base}_y_train.npy"), y_train)
            np.save(os.path.join(PROC_DIR, f"{regime_base}_X_val.npy"), scaler.transform(X_val))
            np.save(os.path.join(PROC_DIR, f"{regime_base}_y_val.npy"), y_val)
            np.save(os.path.join(PROC_DIR, f"{regime_base}_X_test.npy"), scaler.transform(X_test))
            np.save(os.path.join(PROC_DIR, f"{regime_base}_y_test.npy"), y_test)
            
            joblib.dump(scaler, os.path.join(MODELS_DIR, f"{regime_base}_scaler.joblib"))
            joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{regime_base}_feature_list.joblib"))
            joblib.dump(df_test.index, os.path.join(MODELS_DIR, f"{regime_base}_test_indices.joblib"))

print("\n[OK] process_data.py Finished")
