# predict.py
import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz
import warnings

warnings.filterwarnings('ignore')

VERSION = "v4.1"
MODELS_DIR = "models"
PROC_DIR = "data/processed"
TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = ["DX=F", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
FRED_TICKERS = ["FRED_T10Y2Y", "FRED_UNRATE", "FRED_CPIAUCSL", "FRED_M2SL", "FRED_DGS10"]

ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5

# --- Native Feature Engineering (Replaces TA-Lib to prevent drift) ---
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
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).rolling(window=period).mean()

def fetch_latest_data(ticker, period="60d"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
        
        rename_map = {'open':'Open','high':'High','low':'Low','close':'Close','adj close':'Adj Close','volume':'Volume'}
        df.columns = [col.lower() for col in df.columns]
        df.rename(columns=rename_map, inplace=True)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df
    except Exception:
        return None

def main():
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    signals = []
    
    # Fetch Macro Data
    macro_data = {}
    for mt in MACRO_TICKERS:
        df = fetch_latest_data(mt)
        if df is not None and not df.empty:
            c = df['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            macro_data[mt] = c
    macro_df = pd.DataFrame(macro_data).ffill()
    
    # Fetch FRED Data (from processed cache to avoid API latency during prediction)
    fred_data = {}
    for ft in FRED_TICKERS:
        safe_ft = ft.replace('=','_').replace('^','').lower()
        p = os.path.join("data/raw", f"{safe_ft}_1d_data.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
            c = d['Close']
            if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
            fred_data[ft] = c
    fred_df = pd.DataFrame(fred_data).ffill()

    current_vix = macro_df['^VIX'].iloc[-1] if '^VIX' in macro_df.columns else 20.0
    vix_regime_val = 1 if current_vix >= 20 else 0

    for ticker in TICKERS:
        df = fetch_latest_data(ticker)
        if df is None or len(df) < 50:
            continue

        base = ticker.replace('=', '_').lower()
        
        # Engineer Features
        features_df = pd.DataFrame(index=df.index)
        c = df['Close']
        if isinstance(c, pd.DataFrame): c = c.iloc[:, 0]
        c = c.astype(np.float64)
        
        features_df['MA5'] = c.rolling(5).mean()
        features_df['MA20'] = c.rolling(20).mean()
        features_df['MA50'] = c.rolling(50).mean()
        features_df['ATR'] = calc_atr(df['High'].squeeze(), df['Low'].squeeze(), c, 14)
        u, m, lo = calc_bbands(c, 20, 2)
        features_df['BB_Width'] = (u - lo) / (m + 1e-12)
        features_df['RSI14'] = calc_rsi(c, 14)
        
        features_df['VNM'] = c.diff(14) / (features_df['ATR'] + 1e-12)
        direction = c.diff(14).abs()
        volatility = c.diff().abs().rolling(14).sum()
        features_df['KER'] = direction / (volatility + 1e-12)

        features_df = features_df.join(macro_df, how='left').join(fred_df, how='left')
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.ffill(inplace=True)
        features_df.fillna(0.0, inplace=True)

        latest_features = features_df.iloc[-1:]
        current_price = float(c.iloc[-1])
        current_atr = float(features_df['ATR'].iloc[-1])

        # Determine Regime
        suffix = "_high_vix" if vix_regime_val == 1 else "_low_vix"
        
        # Fallback to standard model if regime model doesn't exist
        rb = f"{base}{suffix}"
        if not os.path.exists(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")):
            rb = base 
            if not os.path.exists(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")):
                continue

        try:
            choice = joblib.load(os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib"))
            if not choice.get("trading_enabled", False):
                continue

            feature_names = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
            scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
            
            # Ensure feature alignment
            X_raw = latest_features.reindex(columns=feature_names, fill_value=0.0).values
            X_scaled = scaler.transform(X_raw)

            model_type = choice['model_type']
            
            # Programmatic Ensemble Handling (Fixes FileNotFoundError)
            if model_type == 'ensemble':
                xgb_model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_xgb_calibrated.joblib"))
                lr_model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_lr_calibrated.joblib"))
                prob_xgb = xgb_model.predict_proba(X_scaled)[0][1]
                prob_lr = lr_model.predict_proba(X_scaled)[0][1]
                prob = float((prob_xgb + prob_lr) / 2.0)
            else:
                model = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{model_type}_calibrated.joblib"))
                prob = float(model.predict_proba(X_scaled)[0][1])

            bull_thresh = choice['thresholds']['bull']
            bear_thresh = choice['thresholds']['bear']

            signal_dir = "HOLD"
            if prob >= bull_thresh: signal_dir = "BULLISH"
            elif prob <= bear_thresh: signal_dir = "BEARISH"

            if signal_dir != "HOLD":
                sl_dist = current_atr * ATR_SL_MULT
                tp_dist = current_atr * ATR_TP_MULT
                
                sl = current_price - sl_dist if signal_dir == "BULLISH" else current_price + sl_dist
                tp = current_price + tp_dist if signal_dir == "BULLISH" else current_price - tp_dist

                signals.append({
                    "ticker": ticker,
                    "direction": signal_dir,
                    "confidence": round(prob, 4),
                    "entry_price": round(current_price, 6),
                    "stop_loss": round(sl, 6),
                    "take_profit": round(tp, 6),
                    "model_regime": rb,
                    "model_type": model_type,
                    "model_version": VERSION,
                    "atr_current": round(current_atr, 6)
                })

        except Exception as e:
            # Silent fail for individual tickers to preserve JSON output integrity for others
            continue

    payload = {
        "timestamp": timestamp,
        "model_version": VERSION,
        "vix_value": round(float(current_vix), 2),
        "vix_regime": vix_regime_val,
        "signals_count": len(signals),
        "signals": signals
    }

    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()