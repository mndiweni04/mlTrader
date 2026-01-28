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
except Exception as e:
    SP_CLOSE = None
            
for t in TICKERS:
    df = load_raw(t)
    if df is None: continue

    print(f"\nProcessing {t} ...")
    base = t.replace('=','_').lower() 
    df = df.copy().sort_index()
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))

    # Convert to float64 for TA-Lib
    for col in ['High', 'Low', 'Close', 'Volume']:
        df[col] = df[col].astype(np.float64)

    features_df = pd.DataFrame(index=df.index)
    high, low, close, volume = df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values
    ticker_ret = talib.ROC(close, timeperiod=1)
    
    # Features
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

    # FIX: Warm-up period handling to prevent data loss from dropna()
    features_df = features_df.ffill().bfill()

    # Macro Features Alignment
    if "^VIX" in macro_data:
        vix_c = macro_data["^VIX"]['Close'].reindex(df.index, method='ffill').astype(np.float64)
        features_df['VIX_Close'] = vix_c
        features_df['VIX_Regime'] = (vix_c > 20).astype(int)

    # Labeling
    df['future_return'] = df['Close'].shift(-PREDICTION_HORIZON) / df['Close'] - 1
    atr_values = talib.ATR(high, low, close, 14)
    barrier_threshold = atr_values * 0.20
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier_threshold, 'label'] = 1
    df.loc[df['future_return'] < -barrier_threshold, 'label'] = 0
    
    # FIX: Fallback for sparse data
    df_full = features_df.join(df[['label']])
    if df_full.dropna().shape[0] < 100:
        print(f"  [INFO] Sparse data fallback for {t}")
        df['label'] = (df['future_return'] > 0).astype(int)
        df_full = features_df.join(df[['label']])

    feature_cols = list(features_df.columns)
    df_full[feature_cols] = df_full[feature_cols].shift(1)
    df_full = df_full.dropna()
    
    if df_full.empty: continue
    df_full['label'] = df_full['label'].astype(int)

    # Split and process regimes (Unchanged logic)
    # ... [Rest of the splitting/scaling logic from original file] ...
