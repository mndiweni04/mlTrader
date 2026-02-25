# visualize_results.py
"""
Reads the live_trades_log.csv and prints a performance report.
"""

import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

LOGS_DIR = "logs"
TRADES_LOG_FILE = os.path.join(LOGS_DIR, "live_trades_log.csv")

def analyze_trade_log():
    if not os.path.exists(TRADES_LOG_FILE):
        print(f"No trade log file found at {TRADES_LOG_FILE}.")
        print("Run the bot to generate some trades first.")
        return

    print(f"Loading trade log from {TRADES_LOG_FILE}...\n")
    
    try:
        df = pd.read_csv(TRADES_LOG_FILE)
        if df.empty:
            print("Trade log is empty.")
            return
    except pd.errors.EmptyDataError:
        print("Trade log is empty.")
        return
    except Exception as e:
        print(f"Error reading trade log: {e}")
        return

    # Filter for closed trades
    df_closed = df[df['status'] != 'OPEN'].copy()

    # Legacy Schema Backward Compatibility
    if 'model_regime' not in df_closed.columns:
        df_closed['model_regime'] = 'standard'
    if 'kelly_percentage' not in df_closed.columns:
        df_closed['kelly_percentage'] = 0.0

    if df_closed.empty:
        print("No closed trades found yet. Run the monitor_trades.py script to update trade status.")
        
        # Show open trade count if any
        open_count = len(df[df['status'] == 'OPEN'])
        if open_count > 0:
            print(f"\nFound {open_count} trades still OPEN.")
        return

    # --- Start Analysis ---
    
    # Ensure PNL is numeric for calculations
    df_closed['pnl'] = pd.to_numeric(df_closed['pnl'], errors='coerce').fillna(0)

    total_trades = len(df_closed)
    total_pnl = df_closed['pnl'].sum()
    
    wins = df_closed[df_closed['pnl'] > 0]
    losses = df_closed[df_closed['pnl'] <= 0] # Treating 0 as a loss/scratch
    
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0.0
    avg_win = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    
    # --- FIX: Handle Profit Factor Calculation Safely ---
    total_loss_abs = abs(losses['pnl'].sum())
    if total_loss_abs > 0:
        profit_factor = wins['pnl'].sum() / total_loss_abs
    else:
        profit_factor = float('inf') # Use a float infinity, not a string
    # ----------------------------------------------------
    
    print("="*50)
    print(" 📈 LIVE TRADING PERFORMANCE REPORT")
    print("="*50)
    
    print("\n--- 1. OVERALL SUMMARY ---")
    print(f"Total Closed Trades: {total_trades}")
    print(f"Total Net P/L:       ${total_pnl:,.2f}")
    print(f"Win Rate:            {win_rate:.2f}% ({len(wins)} wins / {len(losses)} losses)")
    
    # Handle NaN formatting gracefully
    avg_win_str = f"${avg_win:,.2f}" if pd.notna(avg_win) else "N/A"
    print(f"Average Win:         {avg_win_str}")
    
    print(f"Average Loss:        ${avg_loss:,.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")

    # --- 2. Performance by Model Regime ---
    print("\n" + "-"*50)
    print("--- 2. PERFORMANCE BY MODEL (REGIME) ---")
    
    if not df_closed.empty:
        model_summary = df_closed.groupby('model_regime')['pnl'].agg(
            total_pnl='sum',
            trade_count='count',
            win_rate=lambda x: (x > 0).mean() * 100
        ).sort_values(by='total_pnl', ascending=False)
        
        print(model_summary.to_string(float_format="%.2f"))

    # --- 3. Performance by Ticker ---
    print("\n" + "-"*50)
    print("--- 3. PERFORMANCE BY TICKER ---")
    
    if not df_closed.empty:
        ticker_summary = df_closed.groupby('ticker')['pnl'].agg(
            total_pnl='sum',
            trade_count='count',
            win_rate=lambda x: (x > 0).mean() * 100
        ).sort_values(by='total_pnl', ascending=False)
        
        print(ticker_summary.to_string(float_format="%.2f"))

if __name__ == "__main__":
    analyze_trade_log()
