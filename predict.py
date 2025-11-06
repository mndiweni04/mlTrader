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

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

TICKERS = ["CL=F"] # FOCUSED
INTERVAL = "1d"
DATA_PERIOD = "100d"
MODELS_DIR = "models"
ATR_MULTIPLIER = 1.5
CONF_THRESHOLD = 0.60 # The 60% threshold is our proven winner
RR_RATIO = 1.5

# --- Feature Engineering Functions ---

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

# --- End of Feature Functions ---

def fetch_live(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval=INTERVAL, progress=False, auto_adjust=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.title() for c in df.columns]
        if "Adj Close" in df.columns and "Close" not in df.columns:
            df["Close"] = df["Adj Close"]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None: df.index = df.index.tz_localize('UTC')
        else: df.index = df.index.tz_convert('UTC')
        return df
    except Exception as e:
        print(f"Fetch failed: {e}")
        return None

def engineer_live(df, gc_close_series=None, is_gc_f=False):
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

    if gc_close_series is not None and not is_gc_f:
        ret_gc = np.log(gc_close_series / gc_close_series.shift(1))
        ret_here = df['ret_1d']
        df['corr_with_gc_21d'] = ret_here.rolling(21, min_periods=1).corr(ret_gc.reindex(df.index))
    else:
        df['corr_with_gc_21d'] = 0.0
    
    return df

if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("\n" + "="*70)
    print(" LIVE DAILY MARKET DIRECTION REPORT (5-Day Horizon)")
    print("="*70)
    print(f"Generated (SAST): {now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")

    gc_df = fetch_live("GC=F")
    gc_close = gc_df['Close'] if gc_df is not None else None

    for t in TICKERS:
        print(f"--- {t} ---")
        base = t.replace('=','_').lower()
        model_xgb_file = os.path.join(MODELS_DIR, f"{base}_xgb_model.joblib")
        scaler_file = os.path.join(MODELS_DIR, f"{base}_scaler.joblib")
        feature_file = os.path.join(MODELS_DIR, f"{base}_feature_list.joblib")

        if not all(os.path.exists(f) for f in [scaler_file, feature_file, model_xgb_file]):
            print("  Missing scaler/features/model — run process_data.py and train_model.py first.")
            continue

        scaler = joblib.load(scaler_file)
        features = joblib.load(feature_file)
        model = joblib.load(model_xgb_file)
        
        df = fetch_live(t)
        if df is None or df.empty:
            print("  No live data.")
            continue

        df_feat = engineer_live(df, gc_close, is_gc_f=(t == "GC=F"))
        if len(df_feat) < 30:
            print(f"  Not enough rows to compute features (need ~30, got {len(df_feat)}).")
            continue

        for f in features:
            if f not in df_feat.columns:
                print(f"  Warning: Feature {f} missing, setting to 0.")
                df_feat[f] = 0.0

        latest = df_feat.iloc[-1]
        
        if latest[features].isnull().any():
            print("  Latest data has NaNs, cannot predict.")
            continue
            
        X_latest = latest[features].values.reshape(1, -1)
        X_scaled = scaler.transform(X_latest)

        prob = model.predict_proba(X_scaled)[0][1]
        model_name = "XGB"
            
        direction = "BULLISH" if prob >= 0.5 else "BEARISH"
        confidence = prob*100 if prob >= 0.5 else (1-prob)*100
        last_close = latest['Close']
        
        print(f"  Chosen model: {model_name}")
        
        if confidence >= (CONF_THRESHOLD * 100):
            print(f"  Prediction (5-Day Horizon): {direction} [TRADE SIGNAL]")
            print(f"  Confidence: {confidence:.2f}% (>= {CONF_THRESHOLD*100:.1f}%)")
        else:
            print(f"  Prediction (5-Day Horizon): {direction} [HOLD / NO SIGNAL]")
            print(f"  Confidence: {confidence:.2f}% (< {CONF_THRESHOLD*100:.1f}%)")

        print(f"  Current Price: {last_close:.6f}")

        last_atr = latest.get('ATR', np.nan)
        if not np.isnan(last_atr):
            atr_amount = last_atr * ATR_MULTIPLIER
            
            if direction == "BULLISH":
                atr_sl = last_close - atr_amount
                atr_tp = last_close + (atr_amount * RR_RATIO)
                print(f"  Suggested ATR SL ({ATR_MULTIPLIER}x): {atr_sl:.6f}")
                print(f"  Suggested ATR TP ({RR_RATIO}x R/R): {atr_tp:.6f}")
            else: # BEARISH
                atr_sl = last_close + atr_amount
                atr_tp = last_close - (atr_amount * RR_RATIO)
                print(f"  Suggested ATR SL ({ATR_MULTIPLIER}x): {atr_sl:.6f}")
                print(f"  Suggested ATR TP ({RR_RATIO}x R/R): {atr_tp:.6f}")
        else:
            print("  ATR N/A")

        print("-" * 30)

    print("\n" + "="*70)
    print(" End of Live Market Direction Report")
    print("="*70)