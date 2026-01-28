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
    if df is not None: macro_data[mt] = df

try:
    sp_df = load_raw("ES=F")
    if sp_df is None:
        sp_df = yf.download("ES=F", period="10y", interval="1d", progress=False)
        sp_df.columns = [c.title() for c in sp_df.columns]
    SP_CLOSE = sp_df['Close'].pct_change()
except Exception:
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
    close = df['Close'].values
    
    # Technical Indicators
    features_df['MA5'] = talib.MA(close, 5)
    features_df['MA20'] = talib.MA(close, 20)
    features_df['MA50'] = talib.MA(close, 50)
    features_df['MA200'] = talib.MA(close, 200)
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    features_df['ATR'] = talib.ATR(df['High'].values, df['Low'].values, close, 14)
    u, m, lo = talib.BBANDS(close, 20, 2, 2, 0)
    features_df['BB_Width'] = (u - lo) / (m + 1e-12)
    features_df['RSI14'] = talib.RSI(close, 14)
    features_df['ROC10'] = talib.ROC(close, 10)
    features_df['MACD'], features_df['MACD_signal'], features_df['MACD_hist'] = talib.MACD(close)
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features_df['OBV'] = talib.OBV(close, df['Volume'].values)
    
    features_df['Day_Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
    features_df['Dist_from_MA20'] = (df['Close'] - features_df['MA20']) / (features_df['MA20'] + 1e-12)

    # Fill NaNs from indicators
    features_df = features_df.ffill().bfill()

    if "^VIX" in macro_data:
        vix_c = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        features_df['VIX_Close'] = vix_c
        features_df['VIX_Regime'] = (vix_c > 20).astype(int)

    # Labeling logic
    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    barrier = talib.ATR(df['High'].values, df['Low'].values, close, 14) * 0.20
    
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier, 'label'] = 1
    df.loc[df['future_return'] < -barrier, 'label'] = 0
    
    # Merge and handle empty data
    df_full = features_df.join(df[['label', 'future_return']])
    
    # If Triple Barrier is too restrictive, use simple binary returns
    if df_full['label'].dropna().shape[0] < 100:
        df_full['label'] = (df_full['future_return'] > 0).astype(float)
        print(f"  [INFO] Fallback to binary labeling for {t}")

    # Final cleanup: Shift features so we don't use today's data to predict today
    feature_cols = list(features_df.columns)
    df_full[feature_cols] = df_full[feature_cols].shift(1)
    
    # Drop rows where we have no label or no features (due to shift)
    df_full = df_full.dropna(subset=['label'] + feature_cols)
    
    if df_full.empty:
        print(f"  [WARNING] No data remains for {t}")
        continue

    df_full['label'] = df_full['label'].astype(int)

    # Split logic (ES=F, NQ=F, etc.)
    regimes = [(df_full[df_full['VIX_Regime'] == 0], "_low_vix"), 
               (df_full[df_full['VIX_Regime'] == 1], "_high_vix")] if t in TICKERS_TO_SPLIT else [(df_full, "")]

    for df_split, suffix in regimes:
        if len(df_split) < 50: continue
        rb = f"{base}{suffix}"
        
        n = len(df_split)
        n_test = max(int(n * TEST_SIZE), 1)
        n_val = max(int((n - n_test) * VAL_SIZE), 1)
        
        df_train = df_split.iloc[:-(n_test + n_val)]
        df_val = df_split.iloc[-(n_test + n_val):-n_test]
        df_test = df_split.iloc[-n_test:]
        
        # Balance Training Set
        c0, c1 = df_train[df_train['label']==0], df_train[df_train['label']==1]
        ms = min(len(c0), len(c1))
        if ms < 10: continue
        
        df_train_bal = pd.concat([resample(c0, n_samples=ms, random_state=42), 
                                  resample(c1, n_samples=ms, random_state=42)]).sort_index()
        
        scaler = StandardScaler().fit(df_train_bal[feature_cols])
        
        for name, data in [("X_train", df_train_bal), ("X_val", df_val), ("X_test", df_test)]:
            np.save(os.path.join(PROC_DIR, f"{rb}_{name}.npy"), scaler.transform(data[feature_cols]))
        for name, data in [("y_train", df_train_bal), ("y_val", df_val), ("y_test", df_test)]:
            np.save(os.path.join(PROC_DIR, f"{rb}_{name}.npy"), data['label'].values)
            
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        joblib.dump(df_test.index, os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib"))

print("\n[OK] process_data.py Finished")
