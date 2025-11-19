# run_trader.py
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

# --- FIX 1: Lock File Logic ---
def check_daily_lock():
    """Returns True if a lock file for today already exists."""
    today = datetime.now(pytz.timezone("Africa/Johannesburg")).strftime('%Y-%m-%d')
    lock_file = os.path.join(LOGS_DIR, f"daily_run_{today}.lock")
    if os.path.exists(lock_file):
        print(f"🔒 Lock file found for {today}. Skipping duplicate run.")
        return True, lock_file
    return False, lock_file

def create_daily_lock(lock_file):
    with open(lock_file, 'w') as f:
        f.write(f"Run completed at {datetime.now()}")
# ------------------------------

CORRELATION_GROUPS = [
    {'ES=F', 'NQ=F'},
    {'EURUSD=X', 'JPYUSD=X'},
    {'CL=F', 'NG=F'}
]

def run_predictions():
    print("Running predict.py...")
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "predict.py")
    if not os.path.exists(script_path): return None, "File not found"

    try:
        process = subprocess.run([python_exe, script_path], capture_output=True, text=True, timeout=600)
        if process.returncode != 0: return None, process.stderr
        return process.stdout, None
    except Exception as e: return None, str(e)

def get_open_tickers():
    if not os.path.exists(TRADES_LOG_FILE): return set()
    open_tickers = set()
    try:
        with open(TRADES_LOG_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'OPEN':
                    open_tickers.add(row['ticker'])
    except Exception: pass
    return open_tickers

def check_exposure_cap(new_ticker, open_tickers):
    for group in CORRELATION_GROUPS:
        if new_ticker in group:
            intersection = group.intersection(open_tickers)
            if intersection and new_ticker not in intersection:
                print(f"  [Risk] Blocking {new_ticker} due to correlated open trade: {intersection}")
                return True 
    return False

def generate_signal_hash(date, ticker, regime, direction):
    raw_str = f"{date}-{ticker}-{regime}-{direction}"
    return hashlib.sha256(raw_str.encode()).hexdigest()

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
            
    model_version = "unknown"
    for line in lines:
        if "Model Version:" in line: model_version = line.split(':')[-1].strip(); break

    for i, line in enumerate(lines):
        if "[TRADE SIGNAL]" in line:
            try:
                ticker_line = ""; model_line = ""
                for j in range(i - 1, max(0, i - 10), -1):
                    if lines[j].startswith("---") and lines[j].endswith("---"): ticker_line = lines[j]; break
                    if "Chosen model:" in lines[j]: model_line = lines[j]
                
                if not ticker_line: continue
                current_ticker = ticker_line.strip().replace("---", "").strip()
                
                model_regime = "unknown"
                if model_line:
                    if "(" in model_line: model_regime = model_line.split('(')[-1].replace(')', '').strip()
                    else: model_regime = model_line.split(':')[-1].strip()

                prediction_line = line
                confidence_line = lines[i+1]
                price_line = lines[i+2]
                sl_line = lines[i+3]
                tp_line = lines[i+4]
                
                lots_line = ""
                for k in range(i+5, i+8):
                    if "Suggested Lots:" in lines[k]: lots_line = lines[k]; break
                
                if not lots_line: continue
                lots_str = lots_line.split(':')[-1].strip().split(' ')[0]

                signal_details = {
                    "prediction_date": prediction_date,
                    "ticker": current_ticker,
                    "model_regime": model_regime,
                    "model_version": model_version,
                    "direction": prediction_line.split('[')[0].split(':')[-1].strip(),
                    "confidence": confidence_line.split(':')[-1].strip().replace('%', ''),
                    "entry_price": price_line.split(':')[-1].strip(),
                    "stop_loss": sl_line.split(':')[-1].strip(),
                    "take_profit": tp_line.split(':')[-1].strip(),
                    "lots": lots_str,
                    "signal_hash": generate_signal_hash(prediction_date, current_ticker, model_regime, prediction_line.split('[')[0].split(':')[-1].strip())
                }
                signals.append(signal_details)
            except Exception as e: print(f"Error parsing signal at line {i}: {e}")
    return signals

def log_signals_to_csv(signals_list):
    file_exists = os.path.isfile(TRADES_LOG_FILE)
    existing_hashes = set()
    if file_exists:
        try:
            with open(TRADES_LOG_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'signal_hash' in reader.fieldnames:
                    for row in reader: existing_hashes.add(row['signal_hash'])
        except Exception: pass

    open_tickers = get_open_tickers()
    fieldnames = [
        'trade_id', 'signal_hash', 'prediction_date', 'ticker', 'model_regime', 'model_version',
        'direction', 'entry_price', 'stop_loss', 'take_profit', 
        'lots', 'confidence', 'status', 'close_date', 'close_price', 'pnl'
    ]
    
    rows_to_add = []
    valid_signals = []

    for sig in signals_list:
        if sig['signal_hash'] in existing_hashes:
            print(f"  [Log] Duplicate signal suppressed: {sig['ticker']}")
            continue

        # --- FIX 2 & 6: Zero Lot & Status Check ---
        try: lots_val = float(sig['lots'])
        except: lots_val = 0.0
        
        status = 'OPEN'
        if lots_val <= 0:
            status = 'SKIPPED (Zero Size)'
            print(f"  [Log] Skipped {sig['ticker']}: 0.00 Lots.")
        elif check_exposure_cap(sig['ticker'], open_tickers):
            status = 'SKIPPED (Exposure Cap)'
            
        trade_id = f"{sig['prediction_date'].replace('-','')}-{sig['ticker']}"

        row = {
            'trade_id': trade_id, 'signal_hash': sig['signal_hash'],
            'prediction_date': sig['prediction_date'], 'ticker': sig['ticker'],
            'model_regime': sig['model_regime'], 'model_version': sig['model_version'],
            'direction': sig['direction'], 'entry_price': sig['entry_price'],
            'stop_loss': sig['stop_loss'], 'take_profit': sig['take_profit'],
            'lots': sig['lots'], 'confidence': sig['confidence'],
            'status': status, 'close_date': '', 'close_price': '', 'pnl': ''
        }
        rows_to_add.append(row)
        if status == 'OPEN': valid_signals.append(sig)

    if rows_to_add:
        try:
            with open(TRADES_LOG_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists: writer.writeheader()
                writer.writerows(rows_to_add)
            print(f"✅ Logged {len(rows_to_add)} entries.")
        except Exception as e: print(f"--- ERROR writing log: {e}")
            
    return valid_signals

if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("="*50)
    print(" ML TRADER BOT - DAILY SIGNAL CHECKER")
    print(f" {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*50)
    
    # --- LOCK CHECK ---
    is_locked, lock_file_path = check_daily_lock()
    if is_locked:
        print("Exiting to prevent duplicate run.")
        sys.exit(0)

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
                # Create lock only if we actually processed open trades
                create_daily_lock(lock_file_path)
                
                print(f" 🔔 NOTIFICATION: {len(actionable)} TRADES OPENED! 🔔")
                email_subject = f"ML Trader: {len(actionable)} New Trades"
                email_body = "New OPEN trades found:\n\n"
                for sig in actionable:
                    email_body += (
                        f"------------------------\n"
                        f"ASSET:     {sig['ticker']}\n"
                        f"DIRECTION: {sig['direction']}\n"
                        f"SIZE:      {sig['lots']} Lots\n"
                        f"ENTRY:     {sig['entry_price']}\n"
                        f"STOP:      {sig['stop_loss']}\n"
                        f"TARGET:    {sig['take_profit']}\n"
                        f"------------------------\n"
                    )
                send_email(subject=email_subject, body=email_body)
            else:
                print("\nAll signals skipped. No email sent.")
        else:
            print("\n 💤 No new trade signals found.")