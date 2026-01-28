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

ACCOUNT_BALANCE = 10000.0
RISK_PCT = 0.02
POINT_VALUE = {
    "CL=F": 1000, "GC=F": 100, "SI=F": 5000, "NG=F": 10000,
    "ZC=F": 50, "EURUSD=X": 100000, "JPYUSD=X": 100000, "ES=F": 50, "NQ=F": 20
} 

LOGS_DIR = "logs"
STATUS_LOG = os.path.join(LOGS_DIR, "bot_status.log")
TRADES_LOG = os.path.join(LOGS_DIR, "live_trades_log.csv")

# Ensure directory and file existence to prevent GitHub Action Git errors
os.makedirs(LOGS_DIR, exist_ok=True)
if not os.path.exists(TRADES_LOG):
    with open(TRADES_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['trade_id', 'signal_hash', 'prediction_date', 'ticker', 'direction', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'lots', 'risk_dollars', 'model_regime', 'status'])
        writer.writeheader()

def calculate_position_size(entry, sl, ticker):
    if ticker not in POINT_VALUE: return 0.0, 0.0
    dist = abs(entry - sl)
    if dist <= 0: return 0.0, 0.0
    risk_dollars = ACCOUNT_BALANCE * RISK_PCT
    lots = risk_dollars / (dist * POINT_VALUE[ticker])
    return round(lots, 2), round(risk_dollars, 2)

def log_signals_to_csv(signals):
    existing_hashes = set()
    if os.path.exists(TRADES_LOG):
        with open(TRADES_LOG, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader: existing_hashes.add(row['signal_hash'])

    unique_signals = [s for s in signals if s['signal_hash'] not in existing_hashes]
    if unique_signals:
        with open(TRADES_LOG, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['trade_id', 'signal_hash', 'prediction_date', 'ticker', 'direction', 'confidence', 'entry_price', 'stop_loss', 'take_profit', 'lots', 'risk_dollars', 'model_regime', 'status'])
            for s in unique_signals:
                writer.writerow({
                    'trade_id': f"{s['prediction_date']}-{s['ticker']}",
                    'signal_hash': s['signal_hash'],
                    'prediction_date': s['prediction_date'],
                    'ticker': s['ticker'],
                    'direction': s['direction'],
                    'confidence': f"{s['confidence']:.4f}",
                    'entry_price': f"{s['entry_price']:.4f}",
                    'stop_loss': f"{s['stop_loss']:.4f}",
                    'take_profit': f"{s['take_profit']:.4f}",
                    'lots': f"{s['lots']:.2f}",
                    'risk_dollars': f"{s['risk_dollars']:.2f}",
                    'model_regime': s['model_regime'],
                    'status': 'OPEN'
                })
    return unique_signals

def log_audit(success, msg):
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    status = "[SUCCESS]" if success else "[ERROR]"
    with open(STATUS_LOG, "a") as f:
        f.write(f"{status} {timestamp} | {msg}\n")

def check_for_signals():
    try:
        result = subprocess.run([sys.executable, 'predict.py'], capture_output=True, text=True, timeout=60)
        if result.returncode != 0: return []
        payload = json.loads(result.stdout)
        signals = payload.get('signals', [])
        
        processed = []
        for sig in signals:
            if sig.get('direction') == 'HOLD': continue
            lots, risk = calculate_position_size(sig['entry_price'], sig['stop_loss'], sig['ticker'])
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
    new_sigs = log_signals_to_csv(sigs)
    if new_sigs:
        body = "\n".join([f"{s['ticker']} {s['direction']}: {s['lots']} lots" for s in new_sigs])
        send_email(f"ML Trader: {len(new_sigs)} New Signals", body)
        log_audit(True, f"Dispatched {len(new_sigs)} signals")
