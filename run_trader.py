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

# --- POSITION SIZING & RISK MANAGEMENT ---
ACCOUNT_BALANCE = 10000.0      # Placeholder: $10k account
RISK_PCT = 0.02                # Risk 2% per trade
TICK_SIZE = {
    "CL=F": 0.01,    # Crude Oil: $10 per tick
    "GC=F": 0.1,     # Gold: $10 per tick
    "SI=F": 0.005,   # Silver: $25 per tick
    "NG=F": 0.001,   # Natural Gas: $10 per tick
    "ZC=F": 0.25,    # Corn: $12.50 per tick
    "EURUSD=X": 0.0001, # FX: 1 pip = variable
    "JPYUSD=X": 0.01,   # FX inverse
    "ES=F": 0.25,    # S&P 500: $12.50 per tick
    "NQ=F": 0.25,    # Nasdaq: $20 per tick
}

POINT_VALUE = {
    "CL=F": 1000,    # $1000 per contract per $1
    "GC=F": 100,     # $100 per contract per $1
    "SI=F": 5000,    # $5000 per contract per $1
    "NG=F": 10000,   # $10,000 per contract per $1
    "ZC=F": 50,      # $50 per contract per $0.25 (bushel)
    "EURUSD=X": 100000,  # 100k units standard lot
    "JPYUSD=X": 100000,  # 100k units
    "ES=F": 50,      # $50 per contract per point
    "NQ=F": 20,      # $20 per contract per point
} 

LOGS_DIR = "logs"
STATUS_LOG = os.path.join(LOGS_DIR, "bot_status.log")
TRADES_LOG = os.path.join(LOGS_DIR, "live_trades_log.csv")
os.makedirs(LOGS_DIR, exist_ok=True)

def calculate_position_size(entry_price, stop_loss, ticker):
    """
    Calculate position size (lots) based on risk management rules.
    
    Formula: Lots = (Account_Balance * Risk_Pct) / (abs(Entry - SL) * Point_Value)
    
    Args:
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        ticker (str): Ticker symbol
    
    Returns:
        tuple: (lots, risk_dollars) - Rounded to 2 decimals
    """
    if ticker not in POINT_VALUE:
        return 0.0, 0.0
    
    price_distance = abs(entry_price - stop_loss)
    if price_distance <= 0:
        return 0.0, 0.0
    
    risk_dollars = ACCOUNT_BALANCE * RISK_PCT
    point_value = POINT_VALUE[ticker]
    
    # Lots = Risk_Dollars / (Price_Distance * Point_Value)
    lots = risk_dollars / (price_distance * point_value)
    lots = round(lots, 2)
    
    return lots, round(risk_dollars, 2)

def dispatch_signal(signal):
    """
    Dispatch a trade signal to execution venue.
    
    Currently: Sends email notification
    Future: POST to execution webhook or broker API
    
    API Stub Example:
        response = requests.post(
            'https://api.execution.example.com/v1/orders',
            json=signal,
            headers={'Authorization': f'Bearer {API_KEY}'}
        )
        return response.status_code == 200
    
    Args:
        signal (dict): Signal dict with all execution parameters
    
    Returns:
        bool: True if dispatch successful
    """
    try:
        # --- CURRENT: Send email notification ---
        email_body = f"""
Trade Signal Generated:
Ticker: {signal['ticker']}
Direction: {signal['direction']}
Confidence: {signal['confidence']:.2%}
Entry: {signal['entry_price']:.4f}
Stop Loss: {signal['stop_loss']:.4f}
Take Profit: {signal['take_profit']:.4f}
Lots: {signal['lots']}
Risk: ${signal['risk_dollars']:.2f}
"""
        send_email(f"Trade Signal: {signal['ticker']}", email_body)
        
        # --- FUTURE: API Dispatch ---
        # import requests
        # response = requests.post(
        #     'https://broker-api.example.com/orders',
        #     json=signal,
        #     timeout=10
        # )
        # return response.status_code in [200, 201]
        
        return True
    except Exception as e:
        log_audit(False, f"Dispatch Error: {e}")
        return False

def log_signals_to_csv(signals):
    # Read-Check-Write protocol (IDEMPOTENCY PRESERVED)
    existing_hashes = set()
    if os.path.exists(TRADES_LOG):
        try:
            with open(TRADES_LOG, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_hashes.add(row['signal_hash'])
        except IOError as e:
            log_audit(False, f"CSV Lock Error: {e}")
            return []

    unique_signals = [s for s in signals if s['signal_hash'] not in existing_hashes]
    
    if signals and not unique_signals:
        log_audit(True, "Signals detected but already logged. Skipping write.")
        return []

    if unique_signals:
        try:
            # CSV Safety: Headers check and append mode
            write_header = not os.path.exists(TRADES_LOG)
            with open(TRADES_LOG, 'a', newline='') as f:
                fieldnames = ['trade_id', 'signal_hash', 'prediction_date', 'ticker', 'direction', 
                            'confidence', 'entry_price', 'stop_loss', 'take_profit', 
                            'lots', 'risk_dollars', 'model_regime', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header: 
                    writer.writeheader()
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
        except Exception as e:
            log_audit(False, f"File I/O Error: {e}")
    return unique_signals

def log_audit(success, msg):
    """Log audit trail to status log."""
    timestamp = datetime.now(pytz.timezone("Africa/Johannesburg")).isoformat()
    status_str = "[SUCCESS]" if success else "[ERROR]"
    try:
        with open(STATUS_LOG, "a") as f:
            f.write(f"{status_str} {timestamp} | {msg}\n")
    except Exception as e:
        print(f"[WARNING] Failed to write audit log: {e}")

def get_python_executable():
    """Return the current Python executable path."""
    return sys.executable

def check_for_signals():
    """
    Execute predict.py and parse JSON output for trade signals.
    
    Returns:
        list: Processed signals with position sizing
    """
    try:
        result = subprocess.run(
            [get_python_executable(), 'predict.py'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=60
        )
        
        if result.returncode != 0:
            log_audit(False, f"predict.py execution failed: {result.stderr[:200]}")
            return []
        
        # Parse JSON output from predict.py (v4.0 format)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            log_audit(False, f"Invalid JSON from predict.py: {str(e)[:200]}")
            return []
        
        signals = payload.get('signals', [])
        if not signals:
            log_audit(True, f"No signals generated. VIX={payload.get('vix_value', 'N/A')}, Regime={payload.get('vix_regime', 'N/A')}")
            return []
        
        # Convert JSON signals to CSV-ready format with position sizing
        processed_signals = []
        for sig in signals:
            if sig.get('direction') == 'HOLD':
                continue
                
            # Calculate position size
            lots, risk_dollars = calculate_position_size(
                sig['entry_price'], 
                sig['stop_loss'], 
                sig['ticker']
            )
            
            # Generate hash for deduplication
            hash_input = f"{sig['ticker']}_{sig['direction']}_{datetime.now().strftime('%Y%m%d')}"
            signal_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            
            processed_signals.append({
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ticker': sig['ticker'],
                'direction': sig['direction'],
                'confidence': sig['confidence'],
                'entry_price': sig['entry_price'],
                'stop_loss': sig['stop_loss'],
                'take_profit': sig['take_profit'],
                'lots': lots,
                'risk_dollars': risk_dollars,
                'model_regime': sig['model_regime'],
                'signal_hash': signal_hash
            })
        
        log_audit(True, f"Generated {len(processed_signals)} signals (VIX={payload.get('vix_value', 'N/A')}, Regime={payload.get('vix_regime', 'N/A')})")
        return processed_signals
        
    except subprocess.TimeoutExpired:
        log_audit(False, "predict.py timed out (60s)")
        return []
    except Exception as e:
        log_audit(False, f"Unexpected error in check_for_signals: {str(e)[:200]}")
        return []

if __name__ == "__main__":
    try:
        signals = check_for_signals()
        unique_signals = log_signals_to_csv(signals)
        
        if unique_signals:
            # Send summary email
            email_subject = f"ML Trader: {len(unique_signals)} New Signals"
            email_body = f"Generated {len(unique_signals)} new trade signals\n\n"
            for sig in unique_signals:
                email_body += f"{sig['ticker']} {sig['direction']}: {sig['lots']} lots (Risk: ${sig['risk_dollars']:.2f})\n"
            send_email(email_subject, email_body)
            log_audit(True, f"Dispatched {len(unique_signals)} signals")
        else:
            log_audit(True, "No new signals to log")
            
    except Exception as e:
        log_audit(False, f"Runner Error: {e}")