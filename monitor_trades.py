# monitor_trades.py
"""
Reads the trade log, checks for SL/TP/Expiry,
and updates the log with closed trades.
"""

import os
import csv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import numpy as np # <-- Make sure numpy is imported

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")
PREDICTION_HORIZON_DAYS = 5 # 5-day model horizon

# --- Contract/Lot Specifications (from predict.py) ---
TICKER_SPECS = {
    "CL=F": 1000.0,  "GC=F": 100.0,   "SI=F": 5000.0, "NG=F": 10000.0,
    "ZC=F": 50.0,    "ES=F": 50.0,    "NQ=F": 20.0,
    "EURUSD=X": 100000.0, "JPYUSD=X": 100000.0,
}

def fetch_current_price(ticker):
    """
    Gets the most recent 'Close' price for a ticker.
    Uses '1d' interval to get the last *completed* day's close.
    """
    try:
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False, timeout=10)
        if df is None or df.empty:
            return None, None
        
        # Get the most recent row
        latest = df.iloc[-1]
        price_date = latest.name.strftime('%Y-%m-%d')
        
        # --- *** THIS IS THE FIX *** ---
        # Select the 'Close' price, which might be a pandas object
        price_object = latest['Close']
        # Use .item() to extract the raw Python float from the object
        price_float = price_object.item() 
        
        return price_float, price_date
        # --- *** END FIX *** ---
        
    except Exception as e:
        print(f"  [Monitor] Error fetching price for {ticker}: {e}")
        return None, None

def calculate_pnl(direction, entry_price, close_price, lots, dollar_per_point):
    """
    Calculates the P/L for a closed trade.
    """
    try:
        entry_price = float(entry_price)
        close_price = float(close_price)
        lots = float(lots)
        
        if direction == "BULLISH":
            pnl = (close_price - entry_price) * lots * dollar_per_point
        elif direction == "BEARISH":
            pnl = (entry_price - close_price) * lots * dollar_per_point
        else:
            pnl = 0.0
            
        return round(pnl, 2)
    except Exception as e:
        print(f"  [Monitor] Error calculating PNL: {e}")
        return 0.0

def monitor_open_trades():
    if not os.path.exists(TRADES_LOG_FILE):
        print("[Monitor] No trade log file found. Skipping.")
        return

    print("[Monitor] Checking open trades...")
    
    trades = []
    fieldnames = []
    try:
        with open(TRADES_LOG_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            # Ensure fieldnames are read correctly, even if file is empty
            if reader.fieldnames is None:
                print("[Monitor] Trade log is empty.")
                return
            fieldnames = reader.fieldnames # Save the header
            for row in reader:
                trades.append(row)
    except Exception as e:
        print(f"  [Monitor] Error reading log file: {e}")
        return

    if not trades:
        print("[Monitor] Trade log is empty.")
        return

    open_trades = [row for row in trades if row['status'] == 'OPEN']
    open_tickers = list(set([row['ticker'] for row in open_trades]))
    
    if not open_tickers:
        print("[Monitor] No open trades found.")
        return
        
    print(f"  [Monitor] Found {len(open_trades)} open trade(s) across {len(open_tickers)} assets: {open_tickers}")
    
    current_prices = {}
    print("  [Monitor] Fetching current prices for open trades...")
    for ticker in open_tickers:
        price, date = fetch_current_price(ticker)
        if price is not None:
            current_prices[ticker] = {'price': price, 'date': date}
        time.sleep(0.5) # Be nice to the API

    today = datetime.now(pytz.timezone("Africa/Johannesburg")).date()
    
    trades_closed_count = 0
    # --- Check each trade ---
    for row in trades:
        if row['status'] != 'OPEN':
            continue

        ticker = row['ticker']
        if ticker not in current_prices:
            print(f"  [Monitor] Skipping {ticker}: Could not get live price.")
            continue
            
        # Get data from log
        current_price = current_prices[ticker]['price'] # This is now a float
        current_price_date = current_prices[ticker]['date']
        
        direction = row['direction']
        entry_price = float(row['entry_price'])
        stop_loss = float(row['stop_loss'])
        take_profit = float(row['take_profit'])
        prediction_date = datetime.strptime(row['prediction_date'], '%Y-%m-%d').date()
        
        close_reason = None
        
        # --- Check SL/TP ---
        # This comparison will now work (float vs float)
        if direction == "BULLISH":
            if current_price <= stop_loss:
                close_reason = "STOP-LOSS"
            elif current_price >= take_profit:
                close_reason = "TAKE-PROFIT"
        elif direction == "BEARISH":
            if current_price >= stop_loss:
                close_reason = "STOP-LOSS"
            elif current_price <= take_profit:
                close_reason = "TAKE-PROFIT"
                
        # --- Check Expiry ---
        days_open = np.busday_count(prediction_date, today)
        
        if not close_reason and days_open > PREDICTION_HORIZON_DAYS:
            close_reason = "EXPIRED" 
            
        # --- Update Row if Closed ---
        if close_reason:
            trades_closed_count += 1
            dollar_per_point = TICKER_SPECS.get(ticker, 0.0)
            
            row['status'] = f"CLOSED ({close_reason})"
            row['close_date'] = current_price_date
            row['close_price'] = round(current_price, 6)
            row['pnl'] = calculate_pnl(direction, entry_price, current_price, row['lots'], dollar_per_point)
            
            print(f"  [Monitor] CLOSING {row['trade_id']} ({ticker}): {close_reason} | P/L: ${row['pnl']}")

    # --- Write all data back to the file ---
    if trades_closed_count > 0:
        print(f"  [Monitor] Writing {len(trades)} rows back to log file...")
        try:
            with open(TRADES_LOG_FILE, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames) # Use the fieldnames read at the start
                writer.writeheader()
                writer.writerows(trades)
            print("  [Monitor] Log file updated.")
        except Exception as e:
            print(f"  [Monitor] CRITICAL ERROR: Could not write updated log file! {e}")
    else:
        print("[Monitor] No trades hit SL/TP or expired. Log unchanged.")

if __name__ == "__main__":
    monitor_open_trades()