# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def calculate_features(df):
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    features['volatility_20'] = features['returns'].rolling(20).std()
    
    # Simple Moving Averages
    features['sma_10'] = df['Close'].rolling(10).mean()
    features['sma_50'] = df['Close'].rolling(50).mean()
    features['sma_dist'] = features['sma_10'] / features['sma_50'] - 1
    
    # ATR Approximation 
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()
    
    return features.dropna()

def process_ticker(ticker):
    base = ticker.replace("=", "_").lower()
    raw_file = os.path.join(RAW_DIR, f"{base}.csv")
    
    # Accommodate naming discrepancies in raw downloaded files
    if not os.path.exists(raw_file):
        raw_file = os.path.join(RAW_DIR, f"{base}_1d_data.csv")
        
    if not os.path.exists(raw_file):
        print(f"Skipping {ticker}: Raw data not found.")
        return

    df = pd.read_csv(raw_file, index_col=0, parse_dates=True)
    if len(df) < 100:
        return

    features_df = calculate_features(df)
    
    # Target definition
    horizon = 10
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1.0
    
    # Structural Fix: Dynamic Asymmetric Volatility Thresholds
    # Uses a rolling 252-day window of historical 10-day returns to define dynamic quantiles
    historical_horizon_returns = df['Close'].pct_change(horizon)
    upper_barrier = historical_horizon_returns.rolling(252, min_periods=50).quantile(0.7).bfill()
    lower_barrier = historical_horizon_returns.rolling(252, min_periods=50).quantile(0.3).bfill()
    
    df['label'] = np.nan
    df.loc[df['future_return'] > upper_barrier, 'label'] = 1
    df.loc[df['future_return'] < lower_barrier, 'label'] = 0

    # Decouple Feature Shift from Label to prevent destructive data leakage
    features_shifted = features_df.shift(1)
    df_full = features_shifted.join(df[['label']].rename(columns={'label': 'label_target'})).dropna()

    if len(df_full) < 50:
        print(f"Insufficient data for {ticker} after processing.")
        return

    # Regime definition (VIX proxy using local volatility if VIX unavailable)
    median_vol = df_full['volatility_20'].median()
    regimes = {
        "": df_full,
        "_low_vix": df_full[df_full['volatility_20'] <= median_vol],
        "_high_vix": df_full[df_full['volatility_20'] > median_vol]
    }

    # Save test prices for evaluate_model.py
    test_split_idx = int(len(df_full) * 0.8)
    test_prices = df.loc[df_full.index[test_split_idx]:, 'Close']
    test_prices.to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))

    for suffix, regime_df in regimes.items():
        if len(regime_df) < 50:
            continue
            
        X = regime_df.drop(columns=['label_target', 'volatility_20']).values
        y = regime_df['label_target'].values

        train_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)

        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        test_indices = regime_df.index[val_idx:]

        rb = f"{base}{suffix}"
        np.save(os.path.join(PROC_DIR, f"{rb}_X_train.npy"), X_train)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_train.npy"), y_train)
        np.save(os.path.join(PROC_DIR, f"{rb}_X_val.npy"), X_val)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_val.npy"), y_val)
        np.save(os.path.join(PROC_DIR, f"{rb}_X_test.npy"), X_test)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_test.npy"), y_test)
        joblib.dump(test_indices, os.path.join("models", f"{rb}_test_indices.joblib"))

for ticker in TICKERS:
    process_ticker(ticker)

print("✅ process_data.py Complete")
