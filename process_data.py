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
PREDICTION_HORIZON = 10  # Increased from 5 to capture clearer trends
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
    return macd_line, signal_line, macd_line - signal_line

def calc_bbands(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return mid + (std * std_dev), mid, mid - (std * std_dev)

def calc_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

def load_raw(ticker):
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    p = os.path.join(RAW_DIR, f"{safe_ticker}_1d_data.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index() if os.path.exists(p) else None

macro_data = {mt: load_raw(mt) for mt in MACRO_TICKERS if load_raw(mt) is not None}

for t in TICKERS:
    df = load_raw(t)
    if df is None: continue
    
    base = t.replace('=','_').lower()
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))
    
    features_df = pd.DataFrame(index=df.index)
    c = df['Close'].astype(np.float64)
    features_df['MA5'] = c.rolling(5).mean()
    features_df['MA20'] = c.rolling(20).mean()
    features_df['MA50'] = c.rolling(50).mean()
    features_df['ATR'] = calc_atr(df['High'], df['Low'], c, 14)
    u, m, lo = calc_bbands(c, 20, 2)
    features_df['BB_Width'] = (u - lo) / (m + 1e-12)
    features_df['RSI14'] = calc_rsi(c, 14)
    
    # Labeling: Tightened Barrier to prevent "Buy-Only" models
    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    barrier = features_df['ATR'] * 0.40 
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier, 'label'] = 1
    df.loc[df['future_return'] < -barrier, 'label'] = 0
    
    df_full = features_df.join(df[['label']]).shift(1).join(df[['label']], rsuffix='_target').dropna()
    # ... (rest of split and balance logic remains same as original)
