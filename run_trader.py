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
    for i, line in enumerate(lines):
        if "[TRADE SIGNAL]" in line:
            try:
                # Find the lines above the signal
                ticker_line = ""
                for j in range(i - 5, i): # Search 5 lines up
                    if lines[j].startswith("---"):
                        ticker_line = lines[j]
                        break
                
                current_ticker = ticker_line.strip().replace("---", "").strip()
                
                # Get the lines after the signal
                prediction_line = line
                confidence_line = lines[i+1]
                price_line = lines[i+2]
                sl_line = lines[i+3]
                tp_line = lines[i+4] # <-- THE NEW LINE
                
                signal_details = {
                    "ticker": current_ticker,
                    "prediction": prediction_line.split('[')[0].split(':')[-1].strip(),
                    "confidence": confidence_line.split(':')[-1].strip(),
                    "price": price_line.split(':')[-1].strip(),
                    "stop_loss": sl_line.split(':')[-1].strip(),
                    "take_profit": tp_line.split(':')[-1].strip() # <-- THE NEW LINE
                }
                signals.append(signal_details)
            except Exception as e:
                print(f"Error parsing signal: {e} (line {i})")

    return signals


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
            
            email_subject = f"ML Trader Alert: {len(signals)} New Signal(s)"
            email_body = "New trade signal(s) found:\n\n"
            
            for sig in signals:
                print(f"\n  ASSET:     {sig['ticker']}")
                print(f"  DIRECTION: {sig['prediction']}")
                print(f"  CONFIDENCE: {sig['confidence']}")
                print(f"  ENTRY:     {sig['price']}")
                print(f"  STOP LOSS: {sig['stop_loss']}")
                print(f"  TAKE PROFIT: {sig['take_profit']}") # <-- THE NEW LINE
                
                email_body += (
                    f"------------------------\n"
                    f"ASSET:     {sig['ticker']}\n"
                    f"DIRECTION: {sig['prediction']}\n"
                    f"CONFIDENCE: {sig['confidence']}\n"
                    f"ENTRY:     {sig['price']}\n"
                    f"STOP LOSS: {sig['stop_loss']}\n"
                    f"TAKE PROFIT: {sig['take_profit']}\n" # <-- THE NEW LINE
                    f"------------------------\n"
                )

            print("\n" + "!"*50)
            print("Sending email notification...")
            send_email(subject=email_subject, body=email_body)
            
        else:
            print("\n" + "="*50)
            print(" 💤 No new trade signals found. Holding cash.")
            print("="*50)