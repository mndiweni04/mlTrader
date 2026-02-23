# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X"]

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
MODELS_DIR = "models"
PREDICTION_HORIZON = 10 
TEST_SIZE = 0.15       
VAL_SIZE = 0.15        

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

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
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_bbands(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    return upper, mid, lower

def calc_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def calc_obv(close, volume):
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def load_raw(ticker):
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    p = os.path.join(RAW_DIR, f"{safe_ticker}_1d_data.csv")
    if not os.path.exists(p): return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.sort_index()

print("Loading macro data...")
macro_data = {mt: load_raw(mt) for mt in MACRO_TICKERS if load_raw(mt) is not None}

for t in TICKERS:
    df = load_raw(t)
    if df is None: 
        print(f"Skipping {t}: No raw data found.")
        continue

    print(f"Processing {t} ...")
    base = t.replace('=','_').lower() 
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))
    
    features_df = pd.DataFrame(index=df.index)
    close = df['Close'].astype(np.float64)
    
    features_df['MA5'] = close.rolling(window=5).mean()
    features_df['MA20'] = close.rolling(window=20).mean()
    features_df['MA50'] = close.rolling(window=50).mean()
    features_df['MA200'] = close.rolling(window=200).mean()
    features_df['MA_diff'] = features_df['MA50'] - features_df['MA200']
    features_df['ATR'] = calc_atr(df['High'], df['Low'], close, 14)
    u, m, lo = calc_bbands(close, 20, 2)
    features_df['BB_Width'] = (u - lo) / (m + 1e-12)
    features_df['RSI14'] = calc_rsi(close, 14)
    features_df['ROC10'] = close.pct_change(periods=10) * 100
    features_df['MACD'], _, features_df['MACD_hist'] = calc_macd(close)
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        features_df['OBV'] = calc_obv(close, df['Volume'])
    
    features_df['Day_Range_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-12)
    features_df['Dist_from_MA20'] = (df['Close'] - features_df['MA20']) / (features_df['MA20'] + 1e-12)

    features_df = features_df.ffill().bfill()

    if "^VIX" in macro_data:
        vix_c = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        features_df['VIX_Close'] = vix_c
        features_df['VIX_Regime'] = (vix_c > 20).astype(int)
    else:
        features_df['VIX_Regime'] = 0

    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    barrier = features_df['ATR'] * 0.40 
    
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier, 'label'] = 1
    df.loc[df['future_return'] < -barrier, 'label'] = 0
    
    valid_labels = df['label'].dropna().shape[0]
    if valid_labels < 100:
        print(f"  [INFO] Low labels for {t}. Using 0.2% fixed barrier.")
        df.loc[df['future_return'] > 0.002, 'label'] = 1
        df.loc[df['future_return'] < -0.002, 'label'] = 0

    feature_cols = list(features_df.columns)
    df_full = features_df.join(df[['label']])
    df_full[feature_cols] = df_full[feature_cols].shift(1)
    df_full = df_full.dropna()

    regimes = [(df_full[df_full['VIX_Regime'] == 0], "_low_vix"), 
               (df_full[df_full['VIX_Regime'] == 1], "_high_vix")] if t in TICKERS_TO_SPLIT else [(df_full, "")]

    for df_split, suffix in regimes:
        rb = f"{base}{suffix}"
        if len(df_split) < 50: continue
        
        n_test = max(int(len(df_split) * TEST_SIZE), 1)
        n_val = max(int((len(df_split) - n_test) * VAL_SIZE), 1)
        
        df_train = df_split.iloc[:-(n_test + n_val)]
        df_val = df_split.iloc[-(n_test + n_val):-n_test]
        df_test = df_split.iloc[-n_test:]
        
        c0, c1 = df_train[df_train['label']==0], df_train[df_train['label']==1]
        ms = min(len(c0), len(c1))
        if ms < 5: continue
        
        df_train_bal = pd.concat([resample(c0, n_samples=ms, random_state=42), 
                                  resample(c1, n_samples=ms, random_state=42)]).sort_index()
        
        scaler = StandardScaler().fit(df_train_bal[feature_cols])
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        joblib.dump(df_test.index, os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib"))
        
        for name, data in [("X_train", df_train_bal), ("X_val", df_val), ("X_test", df_test)]:
            np.save(os.path.join(PROC_DIR, f"{rb}_{name}.npy"), scaler.transform(data[feature_cols]))
        for name, data in [("y_train", df_train_bal), ("y_val", df_val), ("y_test", df_test)]:
            np.save(os.path.join(PROC_DIR, f"{rb}_{name}.npy"), data['label'].values.astype(int))
            
        print(f"  [OK] Saved data for {rb}")

print("\n[OK] process_data.py Finished")
