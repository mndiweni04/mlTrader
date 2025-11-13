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
from send_notification import send_email # Imports the emailer
import csv # <-- NEW: Import csv module

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

def run_predictions():
    """
    Runs predict.py as a separate process and captures its output.
    """
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
            timeout=300 # 5-minute timeout
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

def check_for_signals(output):
    """
    Parses the text output from predict.py to find trade signals.
    """
    if output is None:
        return []
        
    signals = []
    lines = output.split('\n')
    
    # --- NEW: Get the timestamp from the report header ---
    report_timestamp_str = lines[2].replace("Generated (SAST):", "").strip()
    try:
        report_time = datetime.strptime(report_timestamp_str, '%Y-%m-%d %H:%M:%S %Z')
        prediction_date = report_time.strftime('%Y-%m-%d')
    except Exception:
        prediction_date = datetime.now(pytz.timezone("Africa/Johannesburg")).strftime('%Y-%m-%d')
    # --- END NEW ---

    for i, line in enumerate(lines):
        if "[TRADE SIGNAL]" in line:
            try:
                # Find the lines above the signal
                ticker_line = ""
                model_line = ""
                for j in range(i - 5, i): # Search 5 lines up
                    if lines[j].startswith("---"):
                        ticker_line = lines[j]
                    if "Chosen model:" in lines[j]:
                        model_line = lines[j]
                
                current_ticker = ticker_line.strip().replace("---", "").strip()
                chosen_model = model_line.split('(')[-1].replace(')', '').strip() # e.g., "ensemble (cl_f)" -> "cl_f"
                
                # Get the lines after the signal
                prediction_line = line
                confidence_line = lines[i+1]
                price_line = lines[i+2]
                sl_line = lines[i+3]
                tp_line = lines[i+4]
                size_line = lines[i+5]
                
                signal_details = {
                    "prediction_date": prediction_date,
                    "ticker": current_ticker,
                    "model_regime": chosen_model,
                    "direction": prediction_line.split('[')[0].split(':')[-1].strip(),
                    "confidence": confidence_line.split(':')[-1].strip().replace('%', ''),
                    "entry_price": price_line.split(':')[-1].strip(),
                    "stop_loss": sl_line.split(':')[-1].strip(),
                    "take_profit": tp_line.split(':')[-1].strip(),
                    "lots": size_line.split(':')[-1].strip().split(' ')[0],
                    "size_pct": confidence_line.split('(')[0].split(':')[-1].strip(), # Get the confidence %
                }
                signals.append(signal_details)
            except Exception as e:
                print(f"Error parsing signal: {e} (line {i})")

    return signals

# --- NEW: Function to write signals to CSV ---
def log_signals_to_csv(signals_list):
    """
    Appends a list of new signals to the trade log CSV.
    Creates the file and header if it doesn't exist.
    """
    file_exists = os.path.isfile(TRADES_LOG_FILE)
    
    # Define the fields for the CSV
    fieldnames = [
        'trade_id', 'prediction_date', 'ticker', 'model_regime', 'direction', 
        'entry_price', 'stop_loss', 'take_profit', 'lots', 'size_pct', 'confidence',
        'status', 'close_date', 'close_price', 'pnl'
    ]
    
    # Generate a unique trade ID for each signal
    # In a real system, you might use a database or a more robust ID
    # For now, we'll use timestamp + ticker
    
    rows_to_add = []
    for sig in signals_list:
        trade_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{sig['model_regime']}"
        row = {
            'trade_id': trade_id,
            'prediction_date': sig['prediction_date'],
            'ticker': sig['ticker'],
            'model_regime': sig['model_regime'],
            'direction': sig['direction'],
            'entry_price': sig['entry_price'],
            'stop_loss': sig['stop_loss'],
            'take_profit': sig['take_profit'],
            'lots': sig['lots'],
            'size_pct': sig['size_pct'],
            'confidence': sig['confidence'],
            'status': 'OPEN', # All new trades are open
            'close_date': '',
            'close_price': '',
            'pnl': ''
        }
        rows_to_add.append(row)

    try:
        with open(TRADES_LOG_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader() # Write header only if file is new
                
            writer.writerows(rows_to_add)
        print(f"✅ Successfully logged {len(rows_to_add)} new signal(s) to {TRADES_LOG_FILE}")
    except Exception as e:
        print(f"--- ERROR: Could not write to log file {TRADES_LOG_FILE} ---")
        print(f"   {e}")
# --- END NEW ---


if __name__ == "__main__":
    now = datetime.now(pytz.timezone("Africa/Johannesburg"))
    print("="*50)
    print(" ML TRADER BOT - DAILY SIGNAL CHECKER")
    print(f" {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*50)

    output, error = run_predictions()
    
    if error:
        print("\nCould not generate predictions. See error above.")
        send_email(
            subject="ML Trader Alert: BOT FAILED",
            body=f"Your trader script failed to run.\n\nError:\n{error}"
        )
    else:
        print("\n--- Full Prediction Report ---")
        print(output)
        print("--- End of Report ---")
        
        signals = check_for_signals(output)
        
        if signals:
            print("\n" + "!"*50)
            print(" 🔔 NOTIFICATION: NEW TRADE SIGNAL(S) FOUND! 🔔")
            print("!"*50)
            
            # --- NEW: Log signals to CSV file ---
            log_signals_to_csv(signals)
            # --- END NEW ---
            
            email_subject = f"ML Trader Alert: {len(signals)} New Signal(s)"
            email_body = "New trade signal(s) found:\n\n"
            
            for sig in signals:
                print(f"\n  ASSET:     {sig['ticker']} ({sig['model_regime']})")
                print(f"  DIRECTION: {sig['direction']}")
                print(f"  CONFIDENCE: {sig['confidence']}%")
                print(f"  SIZE (Lots): {sig['lots']}")
                print(f"  ENTRY:     {sig['entry_price']}")
                print(f"  STOP LOSS: {sig['stop_loss']}")
                print(f"  TAKE PROFIT: {sig['take_profit']}") 
                
                email_body += (
                    f"------------------------\n"
                    f"ASSET:     {sig['ticker']} ({sig['model_regime']})\n"
                    f"DIRECTION: {sig['direction']}\n"
                    f"CONFIDENCE: {sig['confidence']}%\n"
                    f"SIZE (Lots): {sig['lots']}\n"
                    f"ENTRY:     {sig['entry_price']}\n"
                    f"STOP LOSS: {sig['stop_loss']}\n"
                    f"TAKE PROFIT: {sig['take_profit']}\n" 
                    f"------------------------\n"
                )

            print("\n" + "!"*50)
            print("Sending email notification...")
            send_email(subject=email_subject, body=email_body)
            
        else:
            print("\n" + "="*50)
            print(" 💤 No new trade signals found. Holding cash.")
            print("="*50)