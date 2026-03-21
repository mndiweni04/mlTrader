# acquire_data.py
import yfinance as yf
import pandas as pd
import os
import warnings
import time
from fredapi import Fred

# Added popular commodities, indices, crypto, and volatile stocks
CORE_TICKERS = ["CL=F", "GC=F", "SI=F", "NG=F", "ZC=F", "HG=F", "EURUSD=X", "JPYUSD=X", "BTC-USD", "ETH-USD", "ES=F", "NQ=F", "RTY=F", "TSLA", "NVDA"]
# Replaced delisted DX=F with UUP (Invesco DB US Dollar Index Bullish Fund)
MACRO_TICKERS = ["UUP", "TLT", "^VIX", "XLE", "ZS=F", "ZW=F", "XLF", "XLK"]
TICKERS = CORE_TICKERS + MACRO_TICKERS

FRED_SERIES = {
    "FRED_T10Y2Y": "T10Y2Y",
    "FRED_UNRATE": "UNRATE",
    "FRED_CPIAUCSL": "CPIAUCSL",
    "FRED_M2SL": "M2SL",
    "FRED_DGS10": "DGS10"
}

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
                    timeout=DOWNLOAD_TIMEOUT,
                    threads=False 
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

def download_fred_data():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY not found in environment. Skipping FRED macroeconomic data.")
        return
    
    fred = Fred(api_key=api_key)
    for name, series_id in FRED_SERIES.items():
        print(f"Downloading FRED series {name}...")
        try:
            data = fred.get_series(series_id)
            df = pd.DataFrame({'Close': data})
            df.index.name = 'Date'
            save_csv(df, name)
        except Exception as e:
            print(f"Failed to download FRED {name}: {e}")

def save_csv(df, ticker):
    if df is None or df.empty: return
    
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    os.makedirs("data/raw", exist_ok=True)
    safe_ticker = ticker.replace('=','_').replace('^','').lower()
    file_path = os.path.join("data/raw", f"{safe_ticker}_{INTERVAL}_data.csv")
    df.to_csv(file_path)
    print(f"✅ Saved: {file_path}")

if __name__ == "__main__":
    for t in TICKERS:
        df = download_data(t)
        save_csv(df, t)
    download_fred_data()