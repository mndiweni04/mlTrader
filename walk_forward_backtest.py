# walk_forward_backtest.py
import os
import joblib
import pandas as pd
import numpy as np
from dynamic_features import generate_features, apply_rolling_scaling

DATA_DIR = "data/raw"
MODELS_DIR = "models"
LOOKBACK_WINDOW = 252
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.02

POINT_VALUE = {"NG=F": 10000, "ES=F": 50, "EURUSD=X": 100000} # Add others as needed

class WalkForwardBacktester:
    def __init__(self, tickers):
        self.tickers = tickers
        self.capital = INITIAL_CAPITAL
        self.open_positions = {}
        self.trade_history = []
        self.data_cache = {}
        self.models = {}

    def load_data_and_models(self):
        print("Loading data and generating rolling features...")
        for ticker in self.tickers:
            safe_ticker = ticker.replace('=','_').lower()
            file_path = os.path.join(DATA_DIR, f"{safe_ticker}_1d_data.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
                feat_df = generate_features(df)
                scaled_df = apply_rolling_scaling(feat_df, LOOKBACK_WINDOW).shift(1) # T-1
                
                # Combine scaled features with unscaled price/ATR for execution
                combined = scaled_df.join(df[['Close', 'High', 'Low']])
                combined['ATR_current'] = feat_df['ATR']
                self.data_cache[ticker] = combined.dropna()
                
                # Load models (assuming low_vix for standard backtest simplify)
                model_path = os.path.join(MODELS_DIR, f"{safe_ticker}_low_vix_xgb_calibrated.joblib")
                if os.path.exists(model_path):
                    self.models[ticker] = joblib.load(model_path)
                    
        # Align all indices
        common_idx = None
        for df in self.data_cache.values():
            if common_idx is None: common_idx = df.index
            else: common_idx = common_idx.intersection(df.index)
            
        for ticker in self.tickers:
            self.data_cache[ticker] = self.data_cache[ticker].loc[common_idx]
            
        self.timeline = common_idx

    def execute_bar(self, current_date):
        # 1. Manage existing positions
        closed_this_bar = []
        for ticker, pos in self.open_positions.items():
            current_data = self.data_cache[ticker].loc[current_date]
            high, low, close = current_data['High'], current_data['Low'], current_data['Close']
            
            if pos['direction'] == 'BULLISH':
                if low <= pos['sl']:
                    self._close_position(current_date, ticker, pos['sl'], "STOP_LOSS")
                    closed_this_bar.append(ticker)
                elif high >= pos['tp']:
                    self._close_position(current_date, ticker, pos['tp'], "TAKE_PROFIT")
                    closed_this_bar.append(ticker)
                    
        for t in closed_this_bar: del self.open_positions[t]

        # 2. Evaluate new signals
        for ticker in self.tickers:
            if ticker in self.open_positions or ticker not in self.models: continue
            
            current_data = self.data_cache[ticker].loc[current_date]
            feature_cols = [c for c in current_data.index if c not in ['Close', 'High', 'Low', 'ATR_current']]
            X = current_data[feature_cols].values.reshape(1, -1)
            
            prob = self.models[ticker].predict_proba(X)[0][1]
            if prob > 0.55:
                self._open_position(current_date, ticker, "BULLISH", current_data['Close'], current_data['ATR_current'], prob)

    def _open_position(self, date, ticker, direction, price, atr, prob):
        sl_dist = atr * ATR_SL_MULT
        tp_dist = atr * ATR_TP_MULT
        sl = price - sl_dist if direction == "BULLISH" else price + sl_dist
        tp = price + tp_dist if direction == "BULLISH" else price - tp_dist
        
        risk_dollars = self.capital * RISK_PCT
        pv = POINT_VALUE.get(ticker, 1000)
        lots = risk_dollars / ((atr * ATR_SL_MULT) * pv)
        
        self.open_positions[ticker] = {
            'direction': direction, 'entry': price, 'sl': sl, 'tp': tp,
            'lots': lots, 'prob': prob
        }

    def _close_position(self, date, ticker, exit_price, reason):
        pos = self.open_positions[ticker]
        pv = POINT_VALUE.get(ticker, 1000)
        
        if pos['direction'] == 'BULLISH':
            pnl = (exit_price - pos['entry']) * pv * pos['lots']
        else:
            pnl = (pos['entry'] - exit_price) * pv * pos['lots']
            
        self.capital += pnl
        self.trade_history.append({
            'date': date, 'ticker': ticker, 'direction': pos['direction'],
            'entry': pos['entry'], 'exit': exit_price, 'pnl': pnl, 'reason': reason,
            'capital_after': self.capital
        })

    def run(self):
        self.load_data_and_models()
        print(f"Starting backtest from {self.timeline[0]} to {self.timeline[-1]}")
        for date in self.timeline:
            self.execute_bar(date)
            
        print(f"Final Capital: ${self.capital:.2f}")
        df_res = pd.DataFrame(self.trade_history)
        if not df_res.empty:
            df_res.to_csv("logs/backtest_results.csv", index=False)
            print("Results saved to logs/backtest_results.csv")

if __name__ == "__main__":
    bt = WalkForwardBacktester(["NG=F", "ES=F", "EURUSD=X"])
    bt.run()