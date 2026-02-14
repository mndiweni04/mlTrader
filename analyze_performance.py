import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

LOG_FILE = "logs/live_trades_log.csv"

def analyze_log():
    if not os.path.exists(LOG_FILE):
        print("No trade log found.")
        return

    df = pd.read_csv(LOG_FILE)
    
    # Filter for OPEN trades that are old enough to have a result
    open_trades = df[df['status'] == 'OPEN'].copy()
    
    if open_trades.empty:
        print("No open trades to analyze.")
        return

    print(f"Analyzing {len(open_trades)} open trades...")

    for index, row in open_trades.iterrows():
        ticker = row['ticker']
        entry_date = pd.to_datetime(row['prediction_date']).tz_localize(None)
        
        # Skip if trade is less than 1 day old
        if (datetime.now() - entry_date).days < 1:
            continue

        # Fetch price history since entry
        data = yf.download(ticker, start=entry_date, progress=False)
        if data.empty: continue

        # Logic: Did we hit SL or TP?
        entry_price = float(row['entry_price'])
        tp = float(row['take_profit'])
        sl = float(row['stop_loss'])
        direction = row['direction']
        
        status = "OPEN"
        pnl = 0.0
        
        # Iterate through days to see what was hit first
        for i, daily_row in data.iterrows():
            high = float(daily_row['High'])
            low = float(daily_row['Low'])
            
            if direction == "BULLISH":
                if low <= sl:
                    status = "LOSS"
                    pnl = sl - entry_price
                    break
                elif high >= tp:
                    status = "WIN"
                    pnl = tp - entry_price
                    break
            elif direction == "BEARISH":
                if high >= sl:
                    status = "LOSS"
                    pnl = entry_price - sl
                    break
                elif low <= tp:
                    status = "WIN"
                    pnl = entry_price - tp
                    break
        
        # Update DataFrame if status changed
        if status != "OPEN":
            df.at[index, 'status'] = status
            df.at[index, 'realized_pnl'] = round(pnl * float(row['lots']) * 1000, 2) # Approx value
            print(f"Trade {row['trade_id']} ({ticker}): {status}")

    # Save back to CSV
    df.to_csv(LOG_FILE, index=False)
    print("Log updated.")

if __name__ == "__main__":
    analyze_log()
