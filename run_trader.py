# run_trader.py
import subprocess, sys, os, csv, hashlib, json
from datetime import datetime
from send_notification import send_email

ACCOUNT_BALANCE, RISK_PCT = 10000.0, 0.02
# Updated with required multipliers to maintain position-sizing logic integrity. 
POINT_VALUE = {"CL=F": 1000, "GC=F": 100, "SI=F": 5000, "NG=F": 10000, "ZC=F": 50, "HG=F": 25000, "EURUSD=X": 100000, "JPYUSD=X": 100000, "ES=F": 50, "NQ=F": 20, "RTY=F": 50, "BTC-USD": 1, "ETH-USD": 1, "TSLA": 1, "NVDA": 1}
TRADES_LOG = "logs/live_trades_log.csv"

def check_for_signals():
    result = subprocess.run([sys.executable, 'predict.py'], capture_output=True, text=True)
    if result.returncode != 0: return []
    payload = json.loads(result.stdout)
    processed = []
    for sig in payload.get('signals', []):
        pv = POINT_VALUE.get(sig['ticker'], 1)
        sl_dist = abs(sig['entry_price'] - sig['stop_loss'])
        lots = (ACCOUNT_BALANCE * RISK_PCT) / (sl_dist * pv) if sl_dist > 0 else 0
        sig_id = f"{sig['ticker']}_{sig['direction']}_{datetime.now().strftime('%Y%m%d')}"
        processed.append({**sig, 'lots': round(lots, 2), 'signal_hash': hashlib.md5(sig_id.encode()).hexdigest()[:8]})
    return processed

if __name__ == "__main__":
    sigs = check_for_signals()
    if sigs:
        existing = set()
        if os.path.exists(TRADES_LOG):
            with open(TRADES_LOG, 'r') as f:
                existing = {row['signal_hash'] for row in csv.DictReader(f)}
        
        new_sigs = [s for s in sigs if s['signal_hash'] not in existing]
        if new_sigs:
            with open(TRADES_LOG, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=new_sigs[0].keys())
                if os.path.getsize(TRADES_LOG) == 0: writer.writeheader()
                writer.writerows(new_sigs)
            send_email(f"Signals: {len(new_sigs)}", "\n".join([f"{s['ticker']} {s['direction']} ({s['lots']} lots)" for s in new_sigs]))