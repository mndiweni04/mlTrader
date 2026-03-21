# monitor_trades.py
import os, csv, yfinance as yf, pandas as pd, tempfile, time
from datetime import datetime

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")
PREDICTION_HORIZON = 10 # Aligned with process_data.py
POINT_VALUE = {"CL=F": 1000, "GC=F": 100, "SI=F": 5000, "NG=F": 10000, "ZC=F": 50, "EURUSD=X": 100000, "ES=F": 50, "NQ=F": 20}

def monitor_open_trades():
    if not os.path.exists(TRADES_LOG_FILE): return
    
    df = pd.read_csv(TRADES_LOG_FILE)
    open_indices = df[df['status'] == 'OPEN'].index
    if open_indices.empty: return

    for idx in open_indices:
        row = df.loc[idx]
        ticker = row['ticker']
        data = yf.download(ticker, period="5d", progress=False)
        if data.empty: continue
        
        curr_p = data.iloc[-1]['Close']
        entry, sl, tp = float(row['entry_price']), float(row['stop_loss']), float(row['take_profit'])
        days_open = (datetime.now() - pd.to_datetime(row['prediction_date']).replace(tzinfo=None)).days
        
        reason = None
        if row['direction'] == "BULLISH":
            if curr_p <= sl: reason = "SL"
            elif curr_p >= tp: reason = "TP"
        else:
            if curr_p >= sl: reason = "SL"
            elif curr_p <= tp: reason = "TP"
        
        if not reason and days_open >= PREDICTION_HORIZON: reason = "EXPIRED"

        if reason:
            pv = POINT_VALUE.get(ticker, 1)
            mult = 1 if row['direction'] == "BULLISH" else -1
            pnl = (curr_p - entry) * mult * float(row['lots']) * pv
            df.at[idx, 'status'], df.at[idx, 'pnl'], df.at[idx, 'close_price'] = f"CLOSED ({reason})", round(pnl, 2), round(curr_p, 6)

    temp_fd, temp_path = tempfile.mkstemp(dir=LOGS_DIR)
    with os.fdopen(temp_fd, 'w', newline='') as tmp:
        df.to_csv(tmp, index=False)
    os.replace(temp_path, TRADES_LOG_FILE)
    print("✅ monitor_trades.py: Log Updated.")

if __name__ == "__main__": monitor_open_trades()