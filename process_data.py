# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

TICKERS = ["CL=F"] # FOCUSED
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
MODELS_DIR = "models"
SEQUENCE_LENGTH = 48
PREDICTION_HORIZON = 5
MOVE_THRESHOLD = 0.0
TEST_SIZE = 0.10
VAL_SIZE = 0.10

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_raw(ticker):
    p = os.path.join(RAW_DIR, f"{ticker.replace('=','_').lower()}_1d_data.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

def compute_atr(df, n=14):
    try: import talib; return talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=n)
    except ImportError:
        high = df['High']; low = df['Low']; close = df['Close']
        tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=1).mean()

def compute_rsi(series, period=14):
    try: import talib; return talib.RSI(series, timeperiod=period)
    except ImportError:
        delta = series.diff(); up = delta.clip(lower=0); down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=period, min_periods=1).mean(); ma_down = down.rolling(window=period, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-12); return 100 - (100 / (1 + rs))

def compute_bb(close, n=20, ndev=2):
    middle = close.rolling(n, min_periods=1).mean() 
    try:
        import talib
        upper, middle_talib, lower = talib.BBANDS(close, timeperiod=n, nbdevup=ndev, nbdevdn=ndev, matype=0)
        upper = upper; middle = middle_talib; lower = lower
    except ImportError:
        std = close.rolling(n, min_periods=1).std(); upper = middle + (std * ndev); lower = middle - (std * ndev)
    bb_B = (close - lower) / (upper - lower + 1e-12); bb_W = (upper - lower) / (middle + 1e-12)
    return bb_B, bb_W

def compute_macd(close, fast=12, slow=26, signal=9):
    try:
        import talib
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    except ImportError:
        exp1 = close.ewm(span=fast, adjust=False).mean(); exp2 = close.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2; macdsignal = macd.ewm(span=signal, adjust=False).mean(); macdhist = macd - macdsignal
    return macd, macdsignal, macdhist

# Load GC=F for correlation feature
gc_df_path = os.path.join(RAW_DIR, "gc_f_1d_data.csv")
if os.path.exists(gc_df_path):
    gc_df = load_raw("GC=F")
else:
    print("GC=F raw data missing, attempting to download...")
    gc_df = yf.download("GC=F", period="5y", interval="1d", progress=False)
    if gc_df is None or gc_df.empty:
        print("Warning: GC=F download failed, correlation feature will be zero.")
        gc_df = None
    else:
        if isinstance(gc_df.columns, pd.MultiIndex):
            gc_df.columns = gc_df.columns.get_level_values(0)
        gc_df.columns = [c.title() for c in gc_df.columns]
        if "Adj Close" in gc_df.columns and "Close" not in gc_df.columns:
            gc_df["Close"] = gc_df["Adj Close"]
            
for t in TICKERS:
    df = load_raw(t)
    if df is None:
        print(f"Skipping {t}: no raw data.")
        continue

    print(f"\nProcessing {t} ...")
    df = df.copy().sort_index()

    df['ret_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ret_3d'] = np.log(df['Close'] / df['Close'].shift(3))
    df['ret_5d'] = np.log(df['Close'] / df['Close'].shift(5))
    df['ret_10d'] = np.log(df['Close'] / df['Close'].shift(10))
    df['ret_21d'] = np.log(df['Close'] / df['Close'].shift(21))
    df['roll_std_10'] = df['Close'].pct_change().rolling(10).std()
    df['roll_std_21'] = df['Close'].pct_change().rolling(21).std()
    df['roll_mean_10'] = df['Close'].pct_change().rolling(10).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_21'] = df['Close'].rolling(21).mean()
    df['ma_diff_10_21'] = (df['ma_10'] - df['ma_21']) / (df['ma_21'] + 1e-12)
    df['ATR'] = compute_atr(df, n=14)
    df['RSI_14'] = compute_rsi(df['Close'], period=14)
    df['ATR_pct'] = df['ATR'] / df['Close']
    df['bb_B'], df['bb_W'] = compute_bb(df['Close'], n=20)
    df['macd'], df['macdsignal'], df['macdhist'] = compute_macd(df['Close'])
    df['RSI_14_delta_1d'] = df['RSI_14'].diff(1)
    df['bb_B_delta_1d'] = df['bb_B'].diff(1)
    df['macdhist_delta_1d'] = df['macdhist'].diff(1)

    if 'Volume' in df.columns:
        df['vol_10d'] = df['Volume'].rolling(10).sum()
        df['vol_21d'] = df['Volume'].rolling(21).sum()
    df['dow'] = df.index.dayofweek

    if gc_df is not None and t != "GC=F":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            df_gc = gc_df['Close'].reindex(df.index).ffill()
        ret_gc = np.log(df_gc / df_gc.shift(1)).rename('gc_ret_1d')
        ret_here = df['ret_1d']
        corr = ret_here.rolling(21, min_periods=1).corr(ret_gc)
        df['corr_with_gc_21d'] = corr
    else:
        df['corr_with_gc_21d'] = 0.0

    df['future_close'] = df['Close'].shift(-PREDICTION_HORIZON)
    df['return_fwd'] = (df['future_close'] / df['Close'] - 1.0)
    df['label'] = np.nan
    df.loc[df['return_fwd'] > MOVE_THRESHOLD, 'label'] = 1
    df.loc[df['return_fwd'] < -MOVE_THRESHOLD, 'label'] = 0

    features = [
        'ret_1d','ret_3d','ret_5d','ret_10d','ret_21d',
        'roll_std_10','roll_std_21','roll_mean_10',
        'ma_diff_10_21','ATR','RSI_14','ATR_pct',
        'dow', 'corr_with_gc_21d',
        'bb_B', 'bb_W',
        'macd', 'macdsignal', 'macdhist',
        'RSI_14_delta_1d', 'bb_B_delta_1d', 'macdhist_delta_1d'
    ]
    if 'vol_10d' in df.columns:
        features += ['vol_10d','vol_21d']
    
    df = df.dropna(subset=features + ['label'])
    if df.empty:
        print(f"No processed rows for {t} after dropping NaNs.")
        continue
        
    df['label'] = df['label'].astype(int)

    n = len(df)
    n_test = max(int(n * TEST_SIZE), 1)
    n_val = max(int((n - n_test) * VAL_SIZE), 1)
    train_end = n - n_test - n_val
    val_end = n - n_test
    df_train = df.iloc[:train_end]; df_val = df.iloc[train_end:val_end]; df_test = df.iloc[val_end:]
    X_train = df_train[features].values; y_train = df_train['label'].values
    X_val = df_val[features].values; y_val = df_val['label'].values
    X_test = df_test[features].values; y_test = df_test['label'].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train); X_val_s = scaler.transform(X_val); X_test_s = scaler.transform(X_test)
    base = t.replace('=','_').lower()
    np.save(os.path.join(PROC_DIR, f"{base}_X_train.npy"), X_train_s)
    np.save(os.path.join(PROC_DIR, f"{base}_y_train.npy"), y_train)
    np.save(os.path.join(PROC_DIR, f"{base}_X_val.npy"), X_val_s)
    np.save(os.path.join(PROC_DIR, f"{base}_y_val.npy"), y_val)
    np.save(os.path.join(PROC_DIR, f"{base}_X_test.npy"), X_test_s)
    np.save(os.path.join(PROC_DIR, f"{base}_y_test.npy"), y_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{base}_scaler.joblib"))
    joblib.dump(features, os.path.join(MODELS_DIR, f"{base}_feature_list.joblib"))

    print(f"✅ {t}: total={n}, train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")