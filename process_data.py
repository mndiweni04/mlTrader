# process_data.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
from dynamic_features import generate_features

warnings.filterwarnings('ignore')

TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
MACRO_TICKERS = ["UUP", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
FRED_TICKERS = ["FRED_T10Y2Y", "FRED_UNRATE", "FRED_CPIAUCSL", "FRED_M2SL", "FRED_DGS10"]
TICKERS_TO_SPLIT = ["ES=F", "NQ=F", "NG=F", "JPYUSD=X", "BTC-USD", "ETH-USD", "TSLA", "NVDA", "RTY=F"]

RAW_DIR, PROC_DIR, MODELS_DIR = "data/raw", "data/processed", "models"
PREDICTION_HORIZON = 10 
TEST_SIZE, VAL_SIZE = 0.15, 0.15 

os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_raw(ticker):
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    p = os.path.join(RAW_DIR, f"{safe_ticker}_1d_data.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
        return df.loc[~df.index.duplicated(keep='last')]
    return None

macro_df = pd.DataFrame({mt: load_raw(mt)['Close'] for mt in MACRO_TICKERS if load_raw(mt) is not None}).ffill()
fred_df = pd.DataFrame({ft: load_raw(ft)['Close'] for ft in FRED_TICKERS if load_raw(ft) is not None}).ffill()

for t in TICKERS:
    df = load_raw(t)
    if df is None or df.empty: 
        continue
    
    base = t.replace('=','_').lower()
    
    # Structural Adjustment: Price extraction secured prior to indicator NaN filtering
    df['Close'].to_csv(os.path.join(PROC_DIR, f"{base}_test_prices.csv"))
    
    features_df = generate_features(df, macro_df, fred_df)
    
    c = df['Close'].astype(float)
    df['future_return'] = c.shift(-PREDICTION_HORIZON) / c - 1
    barrier = features_df['ATR'] * 0.40 
    df['label'] = np.nan
    df.loc[df['future_return'] > barrier, 'label'] = 1
    df.loc[df['future_return'] < -barrier, 'label'] = 0
    
    df_full = features_df.join(df[['label']]).shift(1).join(df[['label']], rsuffix='_target').dropna()

    regimes = {"": df_full}
    if t in TICKERS_TO_SPLIT and "^VIX" in features_df.columns:
        vix_values = features_df.loc[df_full.index, "^VIX"]
        regimes["_low_vix"] = df_full[vix_values < 20]
        regimes["_high_vix"] = df_full[vix_values >= 20]

    for suffix, regime_df in regimes.items():
        # Structural Adjustment: Lowered minimum sample threshold to 50 for volatile assets
        if len(regime_df) < 50: 
            continue
        
        rb = f"{base}{suffix}"
        
        # Save indices for evaluation alignment
        test_indices = regime_df.index[-int(len(regime_df) * TEST_SIZE):]
        joblib.dump(test_indices, os.path.join(MODELS_DIR, f"{rb}_test_indices.joblib"))

        y = regime_df['label_target'].values
        X_df = regime_df.drop(columns=['label', 'label_target'])
        feature_names = X_df.columns.tolist()
        X = X_df.values

        n_test, n_val = int(len(X) * TEST_SIZE), int(len(X) * VAL_SIZE)
        n_train = len(X) - n_test - n_val

        X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[-n_test:]
        y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[-n_test:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        joblib.dump(feature_names, os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))

        np.save(os.path.join(PROC_DIR, f"{rb}_X_train.npy"), X_train)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_train.npy"), y_train)
        np.save(os.path.join(PROC_DIR, f"{rb}_X_val.npy"), X_val)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_val.npy"), y_val)
        np.save(os.path.join(PROC_DIR, f"{rb}_X_test.npy"), X_test)
        np.save(os.path.join(PROC_DIR, f"{rb}_y_test.npy"), y_test)

print("✅ process_data.py Complete")