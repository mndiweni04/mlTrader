# walk_forward_backtest.py
import os
import joblib
import pandas as pd
import numpy as np
import warnings
from dynamic_features import generate_features

warnings.filterwarnings('ignore')

DATA_DIR, MODELS_DIR = "data/raw", "models"
INITIAL_CAPITAL = 10000.0
RISK_PCT = 0.02
ATR_SL_MULT, ATR_TP_MULT = 1.5, 2.5
POINT_VALUE = {"CL=F": 1000, "GC=F": 100, "SI=F": 5000, "NG=F": 10000, "ZC=F": 50, "EURUSD=X": 100000, "ES=F": 50, "NQ=F": 20}

class WalkForwardBacktester:
    def __init__(self, tickers):
        self.tickers = tickers
        self.capital = INITIAL_CAPITAL
        self.open_positions = {}
        self.trade_history = []
        self.data_cache = {}

    def load_environment(self):
        # Acquire macro data for VIX regime detection
        vix_path = os.path.join(DATA_DIR, "vix_1d_data.csv")
        self.vix_df = pd.read_csv(vix_path, index_col=0, parse_dates=True)['Close'] if os.path.exists(vix_path) else None

        for ticker in self.tickers:
            safe_ticker = ticker.replace('=','_').lower()
            file_path = os.path.join(DATA_DIR, f"{safe_ticker}_1d_data.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True).sort_index()
                # Use centralized feature logic (unscaled)
                feat_df = generate_features(df)
                combined = feat_df.join(df[['Close', 'High', 'Low']])
                self.data_cache[ticker] = combined.dropna()

    def _get_model_prediction(self, ticker, current_date, features):
        base = ticker.replace('=', '_').lower()
        vix = self.vix_df.loc[:current_date].iloc[-1] if self.vix_df is not None else 20
        rb = f"{base}_high_vix" if vix >= 20 else f"{base}_low_vix"
        
        # Fallback and load choice
        path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
        if not os.path.exists(path):
            rb = base
            path = os.path.join(MODELS_DIR, f"{rb}_model_choice.joblib")
            if not os.path.exists(path): return 0.5, "none"

        choice = joblib.load(path)
        if not choice.get("trading_enabled"): return 0.5, "none"

        scaler = joblib.load(os.path.join(MODELS_DIR, f"{rb}_scaler.joblib"))
        f_list = joblib.load(os.path.join(MODELS_DIR, f"{rb}_feature_list.joblib"))
        
        # Scale according to training parameters
        X = scaler.transform(pd.DataFrame([features]).reindex(columns=f_list, fill_value=0.0).values)

        if choice['model_type'] == 'ensemble':
            p = np.mean([joblib.load(os.path.join(MODELS_DIR, f"{rb}_{m}_calibrated.joblib")).predict_proba(X)[0][1] for m in ['xgb', 'lr', 'cb']])
        else:
            p = joblib.load(os.path.join(MODELS_DIR, f"{rb}_{choice['model_type']}_calibrated.joblib")).predict_proba(X)[0][1]
        
        return p, choice

    def execute_bar(self, date):
        for ticker, pos in list(self.open_positions.items()):
            bar = self.data_cache[ticker].loc[date]
            if (pos['dir'] == 'BULLISH' and bar['Low'] <= pos['sl']) or (pos['dir'] == 'BEARISH' and bar['High'] >= pos['sl']):
                self._close(date, ticker, pos['sl'], "SL")
            elif (pos['dir'] == 'BULLISH' and bar['High'] >= pos['tp']) or (pos['dir'] == 'BEARISH' and bar['Low'] <= pos['tp']):
                self._close(date, ticker, pos['tp'], "TP")

        for ticker in self.tickers:
            if ticker in self.open_positions or date not in self.data_cache[ticker].index: continue
            bar = self.data_cache[ticker].loc[date]
            prob, choice = self._get_model_prediction(ticker, date, bar.drop(['Close', 'High', 'Low', 'ATR']).to_dict())
            
            direction = "BULLISH" if prob >= choice.get('thresholds', {}).get('bull', 1.0) else "BEARISH" if prob <= choice.get('thresholds', {}).get('bear', 0.0) else None
            if direction:
                self._open(date, ticker, direction, bar['Close'], bar['ATR'], prob)

    def _open(self, date, ticker, direction, price, atr, prob):
        pv = POINT_VALUE.get(ticker, 1)
        sl_dist = atr * ATR_SL_MULT
        lots = (self.capital * RISK_PCT) / (sl_dist * pv)
        self.open_positions[ticker] = {
            'dir': direction, 'entry': price, 'lots': lots, 'prob': prob,
            'sl': price - sl_dist if direction == "BULLISH" else price + sl_dist,
            'tp': price + (atr * ATR_TP_MULT) if direction == "BULLISH" else price - (atr * ATR_TP_MULT)
        }

    def _close(self, date, ticker, exit_p, reason):
        p = self.open_positions.pop(ticker)
        mult = 1 if p['dir'] == "BULLISH" else -1
        pnl = (exit_p - p['entry']) * mult * p['lots'] * POINT_VALUE.get(ticker, 1)
        self.capital += pnl
        self.trade_history.append({'date': date, 'ticker': ticker, 'pnl': pnl, 'reason': reason, 'cap': self.capital})

    def run(self):
        self.load_environment()
        timeline = sorted(set().union(*(df.index for df in self.data_cache.values())))
        for date in timeline: self.execute_bar(date)
        pd.DataFrame(self.trade_history).to_csv("logs/backtest_results.csv", index=False)
        print(f"Final Capital: ${self.capital:.2f}")

if __name__ == "__main__":
    WalkForwardBacktester(["NG=F", "ES=F", "EURUSD=X"]).run()