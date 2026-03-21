# dynamic_features.py
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def calc_bbands(series, period=20, std_dev=2):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return mid + (std * std_dev), mid, mid - (std * std_dev)

def calc_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def generate_features(df, macro_df=None, fred_df=None):
    """Generates base features without scaling."""
    feat = pd.DataFrame(index=df.index)
    c = df['Close'].astype(float)
    if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
    
    feat['MA5'] = c.rolling(5).mean()
    feat['MA20'] = c.rolling(20).mean()
    feat['MA50'] = c.rolling(50).mean()
    feat['RSI14'] = calc_rsi(c, 14)
    feat['ATR'] = calc_atr(df['High'], df['Low'], c, 14)
    
    u, m, lo = calc_bbands(c, 20, 2)
    feat['BB_Width'] = (u - lo) / (m + 1e-12)
    feat['VNM'] = c.diff(14) / (feat['ATR'] + 1e-12)
    
    direction = c.diff(14).abs()
    volatility = c.diff().abs().rolling(14).sum()
    feat['KER'] = direction / (volatility + 1e-12)
    
    if macro_df is not None:
        feat = feat.join(macro_df, how='left')
    if fred_df is not None:
        feat = feat.join(fred_df, how='left')
        
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.ffill(inplace=True)
    feat.fillna(0.0, inplace=True)
    return feat

def get_current_state(df, macro_df, fred_df):
    """Returns the latest feature vector, current ATR, and last price."""
    feat_df = generate_features(df, macro_df, fred_df)
    latest_features = feat_df.iloc[-1].to_dict()
    current_atr = feat_df['ATR'].iloc[-1]
    last_close = df['Close'].iloc[-1]
    return latest_features, current_atr, last_close