# monitor_trades.py
import os
import csv
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz
import time
import numpy as np 
import tempfile

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")
PREDICTION_HORIZON_DAYS = 5 

TICKER_SPECS = {
    "CL=F": 1000.0,  "GC=F": 100.0,   "SI=F": 5000.0, "NG=F": 10000.0,
    "ZC=F": 50.0,    "ES=F": 50.0,    "NQ=F": 20.0,
    "EURUSD=X": 100000.0, "JPYUSD=X": 100000.0,
}

def fetch_current_price(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, timeout=10)
        if df is None or df.empty: return None, None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            
        latest = df.iloc[-1]
        price_date = latest.name.strftime('%Y-%m-%d')
        return latest['Close'].item(), price_date
    except Exception as e:
        print(f"  [Monitor] Error fetching price for {ticker}: {e}")
        return None, None

def calculate_pnl(direction, entry_price, close_price, lots, dollar_per_point):
    try:
        entry = float(entry_price)
        close = float(close_price)
        lots_val = float(lots)
        if direction == "BULLISH":
            pnl = (close - entry) * lots_val * dollar_per_point
        elif direction == "BEARISH":
            pnl = (entry - close) * lots_val * dollar_per_point
        else: pnl = 0.0
        return round(pnl, 2)
    except: return 0.0

def monitor_open_trades():
    if not os.path.exists(TRADES_LOG_FILE):
        print("[Monitor] No trade log file found."); return

    print("[Monitor] Checking open trades...")
    trades = []
    fieldnames = []
    
    try:
        with open(TRADES_LOG_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames: fieldnames = list(reader.fieldnames)
            for row in reader: trades.append(row)
    except Exception as e:
        print(f"  [Monitor] Error reading log: {e}"); return

    if not trades: return

    open_trades = [row for row in trades if row['status'] == 'OPEN']
    open_tickers = list(set([row['ticker'] for row in open_trades]))
    
    if not open_tickers:
        print("[Monitor] No open trades found."); return
        
    print(f"  [Monitor] Found {len(open_trades)} open trade(s).")
    
    current_prices = {}
    for ticker in open_tickers:
        price, date = fetch_current_price(ticker)
        if price is not None: current_prices[ticker] = {'price': price, 'date': date}
        time.sleep(0.5)

    today = datetime.now(pytz.timezone("Africa/Johannesburg")).date()
    closed_count = 0
    
    for row in trades:
        if row['status'] != 'OPEN': continue
        ticker = row['ticker']
        if ticker not in current_prices: continue
            
        current_price = current_prices[ticker]['price']
        current_price_date = current_prices[ticker]['date']
        
        direction = row['direction']
        entry = float(row['entry_price'])
        sl = float(row['stop_loss'])
        tp = float(row['take_profit'])
        
        date_str = row['prediction_date'].split(' ')[0]
        pred_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        close_reason = None
        if direction == "BULLISH":
            if current_price <= sl: close_reason = "STOP-LOSS"
            elif current_price >= tp: close_reason = "TAKE-PROFIT"
        elif direction == "BEARISH":
            if current_price >= sl: close_reason = "STOP-LOSS"
            elif current_price <= tp: close_reason = "TAKE-PROFIT"
                
        days_open = np.busday_count(pred_date, today)
        if not close_reason and days_open > PREDICTION_HORIZON_DAYS:
            close_reason = "EXPIRED" 
            
        if close_reason:
            closed_count += 1
            dollar_per_point = TICKER_SPECS.get(ticker, 0.0)
            row['status'] = f"CLOSED ({close_reason})"
            row['close_date'] = current_price_date
            row['close_price'] = round(current_price, 6)
            
            lots_traded = row.get('lots', 1.0) 
            row['pnl'] = calculate_pnl(direction, entry, current_price, lots_traded, dollar_per_point)
            
            print(f"  [Monitor] CLOSING {row['ticker']} ({direction}): {close_reason} | P/L: ${row['pnl']}")

    if closed_count > 0:
        print(f"  [Monitor] Updating log file...")
        
        for new_col in ['close_date', 'close_price', 'pnl']:
            if new_col not in fieldnames:
                fieldnames.append(new_col)
                
        temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(TRADES_LOG_FILE))
        with os.fdopen(temp_fd, 'w', newline='') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)
        os.replace(temp_path, TRADES_LOG_FILE)
        print("  [Monitor] Log file updated.")
    else:
        print("[Monitor] No trades closed.")

if __name__ == "__main__":
    monitor_open_trades()
