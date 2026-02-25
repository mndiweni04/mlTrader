import pandas as pd
import yfinance as yf
import os
import tempfile
import pytz
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

    tz_local = pytz.timezone("Africa/Johannesburg")

    for index, row in open_trades.iterrows():
        ticker = row['ticker']
        
        entry_date_naive = pd.to_datetime(row['prediction_date']).tz_localize(None)
        entry_date_aware = tz_local.localize(entry_date_naive)
        
        # Skip if trade is less than 1 day old
        if (datetime.now(tz_local) - entry_date_aware).days < 1:
            continue

        # Fetch price history since entry
        data = yf.download(ticker, start=entry_date_naive, progress=False)
        if data.empty: continue

        # Flattens MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]

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
            allocation = float(row.get('allocation_zar', 0.0))
            df.at[index, 'realized_pnl'] = round(pnl * allocation, 2) 
            print(f"Trade {row.get('trade_id', 'UNKNOWN')} ({ticker}): {status}")

    # Atomic write to prevent file corruption
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(LOG_FILE))
    with os.fdopen(temp_fd, 'w') as tmp:
        df.to_csv(tmp, index=False)
    os.replace(temp_path, LOG_FILE)
    print("Log updated.")

if __name__ == "__main__":
    analyze_log()
