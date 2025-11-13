# acquire_data.py
import yfinance as yf
import pandas as pd
import os
import warnings
import time

# --- NEW: Added Macro/Cross-Instrument Tickers from your v2 memo ---
CORE_TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "EURUSD=X", "JPYUSD=X", "ES=F", "NQ=F"]
MACRO_TICKERS = [
    "DX=F",    # US Dollar Index
    "TLT",     # 20+ Year Treasury Bond ETF (Real Yield Proxy)
    "^VIX",    # VIX (Volatility Index)
    "XLE",     # Energy Sector ETF
    "ZS=F",    # Soybeans
    "ZW=F",    # Wheat
    "XLF",     # Financial Sector ETF
    "XLK"      # Technology Sector ETF
]
TICKERS = CORE_TICKERS + MACRO_TICKERS
# --- END NEW ---

PERIOD = "10y" 
INTERVAL = "1d"
EXPECTED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
MAX_RETRIES = 3 
DOWNLOAD_TIMEOUT = 30 

def download_data(ticker):
    print(f"Downloading {ticker} period={PERIOD} interval={INTERVAL} ...")
    
    for attempt in range(MAX_RETRIES):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                df = yf.download(
                    ticker, 
                    period=PERIOD, 
                    interval=INTERVAL, 
                    auto_adjust=False, 
                    progress=False,
                    timeout=DOWNLOAD_TIMEOUT 
                )
            
            if df.empty: raise Exception(f"No data returned for {ticker}.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
            rename_map = {'open':'Open','high':'High','low':'Low','close':'Close','adj close':'Adj Close','volume':'Volume'}
            df.columns = [col.lower() for col in df.columns]
            df.rename(columns=rename_map, inplace=True)
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            
            # Handle FX data which lacks OHLCV (and VIX)
            if 'Open' not in df.columns: df['Open'] = df['Close']
            if 'High' not in df.columns: df['High'] = df['Close']
            if 'Low' not in df.columns: df['Low'] = df['Close']
            if 'Volume' not in df.columns: df['Volume'] = 0
            
            missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing_cols: raise Exception(f"Missing columns for {ticker}: {missing_cols}")
                
            if "=F" in ticker:
                df = df[df['Volume'] > 0]
                if df.empty: raise Exception(f"No data with Volume > 0 for {ticker}.")

            return df[EXPECTED_COLUMNS]
            
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt + 1 < MAX_RETRIES: time.sleep(5)
            else: print(f"  All retries failed for {ticker}."); return None
        
    return None

def save_csv(df, ticker):
    if df is None or df.empty:
        print(f"No data to save for {ticker}.")
        return
    os.makedirs("data/raw", exist_ok=True)
    
    # --- NEW: Sanitize filename for tickers like ^VIX ---
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    file_path = os.path.join("data/raw", f"{safe_ticker}_{INTERVAL}_data.csv")
    # --- END NEW ---
    
    df.to_csv(file_path)
    print(f"✅ Saved: {file_path}")

if __name__ == "__main__":
    for t in TICKERS:
        df = download_data(t)
        save_csv(df, t)