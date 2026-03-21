# visualize_results.py
import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

TRADES_LOG = "logs/live_trades_log.csv"

def display_performance_report():
    if not os.path.exists(TRADES_LOG) or os.path.getsize(TRADES_LOG) == 0:
        print("============================================================")
        print(" 📉 No live trades logged yet.")
        print("============================================================")
        return

    try:
        df = pd.read_csv(TRADES_LOG)
        if df.empty:
            print("============================================================")
            print(" 📉 No live trades logged yet.")
            print("============================================================")
            return
        
        print("\n============================================================")
        print(f" 📈 PRODUCTION PERFORMANCE REPORT | {len(df)} Trades Logged")
        print("============================================================\n")
        
        # Utilize Markdown output for structured terminal tables
        try:
            print(df.to_markdown(index=False, tablefmt="grid"))
        except ImportError:
            # Fallback formatting if tabulate is not installed
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(df.to_string(index=False))
            
    except Exception as e:
        print(f"Error parsing live trades log: {e}")

if __name__ == "__main__":
    display_performance_report()