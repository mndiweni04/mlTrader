# run_trader.py
import subprocess
import sys
import os
import csv
import hashlib
from datetime import datetime
import pytz
import json
from send_notification import send_email

LOGS_DIR = "logs"
STATUS_LOG = os.path.join(LOGS_DIR, "bot_status.log")
TRADES_LOG = os.path.join(LOGS_DIR, "live_trades_log.csv")

# Ensure environment is ready for GitHub Action commit steps
os.makedirs(LOGS_DIR, exist_ok=True)
if not os.path.exists(TRADES_LOG):
    with open(TRADES_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trade_id', 'signal_hash', 'prediction_date', 'ticker', 'direction', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'lots', 'risk_dollars', 'model_regime', 'status'])
        writer.writeheader()

def log_audit(success, msg):
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    status = "[SUCCESS]" if success else "[ERROR]"
    with open(STATUS_LOG, "a") as f:
        f.write(f"{status} {timestamp} | {msg}\n")

def check_for_signals():
    try:
        result = subprocess.run([sys.executable, 'predict.py'], capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            log_audit(False, f"Predict failed: {result.stderr}")
            return []
        
        payload = json.loads(result.stdout)
        signals = payload.get('signals', [])
        
        processed = []
        for sig in signals:
            if sig.get('direction') == 'HOLD': continue
            
            # Position Sizing placeholder
            lots, risk = 0.1, 20.0 
            h_input = f"{sig['ticker']}_{sig['direction']}_{datetime.now().strftime('%Y%m%d')}"
            
            processed.append({
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': sig['ticker'], 'direction': sig['direction'], 'confidence': sig['confidence'],
                'entry_price': sig['entry_price'], 'stop_loss': sig['stop_loss'], 'take_profit': sig['take_profit'],
                'lots': lots, 'risk_dollars': risk, 'model_regime': sig['model_regime'],
                'signal_hash': hashlib.md5(h_input.encode()).hexdigest()[:8]
            })
        return processed
    except Exception as e:
        log_audit(False, f"Runner Error: {e}")
        return []

if __name__ == "__main__":
    sigs = check_for_signals()
    if sigs:
        # Check for duplicates and log
        existing = set()
        with open(TRADES_LOG, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader: existing.add(row['signal_hash'])
            
        new_sigs = [s for s in sigs if s['signal_hash'] not in existing]
        if new_sigs:
            with open(TRADES_LOG, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['trade_id', 'signal_hash', 'prediction_date', 'ticker', 'direction', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'lots', 'risk_dollars', 'model_regime', 'status'])
                for s in new_sigs:
                    s['status'] = 'OPEN'
                    writer.writerow(s)
            
            body = "\n".join([f"{s['ticker']} {s['direction']}: {s['lots']} lots" for s in new_sigs])
            send_email(f"ML Trader: {len(new_sigs)} New Signals", body)
            log_audit(True, f"Dispatched {len(new_sigs)} signals")
        else:
            log_audit(True, "No new signals (all duplicates)")
    else:
        log_audit(True, "No signals generated")
