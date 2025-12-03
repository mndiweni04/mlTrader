# run_trader.py
"""
Runs the live prediction script and emails a notification if a trade is found.
This is the main file for your daily alerts.
"""

import subprocess
import sys
import os
from datetime import datetime
import pytz
from send_notification import send_email 
import csv 
import hashlib 

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

def run_predictions():
    print("Running predict.py...")
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "predict.py")
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return None, "File not found"

    try:
        process = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            timeout=600 
        )
        
        if process.returncode != 0:
            print("--- predict.py FAILED ---")
            print(process.stderr)
            print("-------------------------")
            return None, process.stderr
            
        return process.stdout, None
        
    except Exception as e:
        print(f"An error occurred while running subprocess: {e}")
        return None, str(e)

def generate_signal_hash(date, ticker, regime, direction, version, confidence):
    """
    Creates a cryptographically unique ID for the signal.
    """
    # Round confidence to 1 decimal to avoid floating point drift
    try:
        conf_val = float(confidence)
        conf_str = f"{conf_val:.1f}"
    except:
        conf_str = str(confidence)

    raw_str = f"{date}|{ticker}|{regime}|{direction}|{version}|{conf_str}"
    return hashlib.sha256(raw_str.encode()).hexdigest()

# --- RESTORED FUNCTION ---
def get_open_trades_state():
    """
    Reads the log to determine:
    1. Which tickers currently have an 'OPEN' trade.
    2. Which signal hashes have already been processed.
    """
    open_tickers = set()
    existing_hashes = set()
    
    if not os.path.exists(TRADES_LOG_FILE):
        return open_tickers, existing_hashes
    
    try:
        with open(TRADES_LOG_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return open_tickers, existing_hashes
            
            for row in reader:
                # Track existing hashes for deduplication
                if 'signal_hash' in row and row['signal_hash']:
                    existing_hashes.add(row['signal_hash'])
                
                # Track OPEN tickers to freeze them
                if row.get('status') == 'OPEN':
                    open_tickers.add(row['ticker'])
                    
    except Exception as e:
        print(f"Warning: Could not read log file: {e}")
        
    return open_tickers, existing_hashes
# -------------------------

def check_for_signals(output):
    if output is None: return []
    signals = []
    lines = output.split('\n')
    
    prediction_date = datetime.now(pytz.timezone("Africa/Johannesburg")).strftime('%Y-%m-%d')
    for line in lines:
        if "Generated (SAST):" in line:
            try:
                ts_str = line.replace("Generated (SAST):", "").strip()
                dt_obj = datetime.strptime(ts_str.split(' SAST')[0], '%Y-%m-%d %H:%M:%S')
                prediction_date = dt_obj.strftime('%Y-%m-%d')
            except: pass
            break
    
    def find_value_in_lines(start_idx, keyword, split_char=':'):
        for k in range(start_idx, min(start_idx + 15, len(lines))):
            if keyword in lines[k]:
                parts = lines[k].split(split_char)
                if len(parts) > 1:
                    return parts[-1].strip()
        return None

    for i, line in enumerate(lines):
        if "[TRADE SIGNAL]" in line:
            try:
                ticker_line = ""; model_line = ""
                for j in range(i - 1, max(0, i - 15), -1):
                    if lines[j].startswith("---") and lines[j].endswith("---"):
                        ticker_line = lines[j]; break
                    if "Chosen model:" in lines[j]: model_line = lines[j]
                
                if not ticker_line: continue
                current_ticker = ticker_line.strip().replace("---", "").strip()
                
                model_regime = "unknown"
                model_version = "v3.6" 
                
                if model_line:
                    if "(" in model_line: model_regime = model_line.split('(')[-1].replace(')', '').strip()
                    else: model_regime = model_line.split(':')[-1].strip()

                direction = line.split('[')[0].split(':')[-1].strip()
                
                confidence = "0.0"
                conf_val = find_value_in_lines(i, "Confidence", ':')
                if conf_val: confidence = conf_val.replace('%', '').split('(')[0].strip()
                
                entry_price = "0.0"
                price_val = find_value_in_lines(i, "Entry Price", ':')
                if price_val: entry_price = price_val
                
                stop_loss = "0.0"
                sl_val = find_value_in_lines(i, "Suggested ATR SL", ':')
                if sl_val: stop_loss = sl_val
                
                take_profit = "0.0"
                tp_val = find_value_in_lines(i, "Suggested ATR TP", ':')
                if tp_val: take_profit = tp_val
                
                lots = "1.00" # Default for ledger

                sig_hash = generate_signal_hash(prediction_date, current_ticker, model_regime, direction, model_version, confidence)

                signal_details = {
                    "prediction_date": prediction_date,
                    "ticker": current_ticker,
                    "model_regime": model_regime,
                    "model_version": model_version,
                    "direction": direction,
                    "confidence": confidence,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "lots": lots,
                    "risk_dollars": "MANUAL", 
                    "size_pct": f"{float(confidence):.2f}%", 
                    "signal_hash": sig_hash
                }
                signals.append(signal_details)
            except Exception as e:
                print(f"Error parsing signal for {current_ticker}: {e}")

    return signals

def log_signals_to_csv(signals_list):
    # --- RESTORED: Get open tickers ---
    open_tickers, existing_hashes = get_open_trades_state()

    fieldnames = [
        'trade_id', 'signal_hash', 'prediction_date', 'ticker', 'model_regime', 'model_version',
        'direction', 'entry_price', 'stop_loss', 'take_profit', 
        'lots', 'risk_dollars', 'confidence', 'status', 
        'close_date', 'close_price', 'pnl'
    ]
    
    rows_to_add = []
    valid_signals = []

    for sig in signals_list:
        # --- RESTORED RULE: Freeze OPEN trades ---
        if sig['ticker'] in open_tickers:
            print(f"  [Log] {sig['ticker']} is already OPEN. Skipping new signal.")
            continue
            
        # Rule 2: Idempotency
        if sig['signal_hash'] in existing_hashes:
            print(f"  [Log] Duplicate signal suppressed: {sig['ticker']}")
            continue

        status = 'OPEN'
        sig['lots'] = "1.00" 
        
        trade_id = f"{sig['prediction_date'].replace('-','')}-{sig['ticker']}"

        row = {
            'trade_id': trade_id,
            'signal_hash': sig['signal_hash'],
            'prediction_date': sig['prediction_date'],
            'ticker': sig['ticker'],
            'model_regime': sig['model_regime'],
            'model_version': sig.get('model_version', 'v3.6'),
            'direction': sig['direction'],
            'entry_price': sig['entry_price'],
            'stop_loss': sig['stop_loss'],
            'take_profit': sig['take_profit'],
            'lots': sig['lots'],
            'risk_dollars': sig['risk_dollars'],
            'confidence': sig['confidence'],
            'status': status,
            'close_date': '', 'close_price': '', 'pnl': ''
        }
        rows_to_add.append(row)
        
        if status == 'OPEN':
            valid_signals.append(sig)

    if rows_to_add:
        try:
            file_exists = os.path.isfile(TRADES_LOG_FILE)
            with open(TRADES_LOG_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                writer.writerows(rows_to_add)
            print(f"✅ Logged {len(rows_to_add)} new entries.")
        except Exception as e:
            print(f"--- ERROR writing to log: {e}")
            
    return valid_signals

if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("="*50)
    print(" ML TRADER BOT - DAILY SIGNAL CHECKER")
    print(f" {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*50)

    output, error = run_predictions()
    
    if error:
        print("\nCould not generate predictions.")
        send_email("ML Trader Alert: BOT FAILED", f"Error:\n{error}")
    else:
        print("\n--- Full Prediction Report ---")
        print(output)
        print("--- End of Report ---")
        
        signals = check_for_signals(output)
        
        if signals:
            print("\n" + "!"*50)
            print(" Processing Signals...")
            
            actionable = log_signals_to_csv(signals)
            
            if actionable:
                print(f" 🔔 NOTIFICATION: {len(actionable)} TRADES OPENED! 🔔")
                email_subject = f"ML Trader: {len(actionable)} New Trades"
                email_body = "New OPEN trades found (Manual Sizing Required):\n\n"
                for sig in actionable:
                    email_body += (
                        f"------------------------\n"
                        f"ASSET:         {sig['ticker']} ({sig['model_regime']})\n"
                        f"DIRECTION:     {sig['direction']}\n"
                        f"ENTRY:         {sig['entry_price']}\n"
                        f"STOP LOSS:     {sig['stop_loss']}\n"
                        f"TAKE PROFIT:   {sig['take_profit']}\n"
                        f"------------------------\n"
                    )
                send_email(subject=email_subject, body=email_body)
            else:
                print("\nAll signals suppressed (Duplicates or Already Open).")
        else:
            print("\n 💤 No new trade signals found.")