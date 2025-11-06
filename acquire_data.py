# acquire_data.py
import yfinance as yf
import pandas as pd
import os
import warnings
import time

TICKERS = ["CL=F"] # FOCUSED
PERIOD = "5y"
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
            missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
            if missing_cols: raise Exception(f"Missing columns for {ticker}: {missing_cols}")
                
            if "=F" in ticker:
                df = df[df['Volume'] > 0]
                if df.empty: raise Exception(f"No data with Volume > 0 for {ticker}.")

            return df[EXPECTED_COLUMNS]
            
        except Exception as e:
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt + 1 < MAX_RETRIES:
                time.sleep(5)
            else:
                print(f"  All retries failed for {ticker}.")
                return None
        
    return None

def save_csv(df, ticker):
    if df is None or df.empty:
        print(f"No data to save for {ticker}.")
        return
    os.makedirs("data/raw", exist_ok=True)
    file_path = os.path.join("data/raw", f"{ticker.replace('=','_').lower()}_{INTERVAL}_data.csv")
    df.to_csv(file_path)
    print(f"✅ Saved: {file_path}")

if __name__ == "__main__":
    for t in TICKERS:
        df = download_data(t)
        save_csv(df, t)