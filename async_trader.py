# async_trader.py
import asyncio
import logging
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from ib_insync import IB, Future, Forex, MarketOrder, LimitOrder, StopOrder
from dynamic_features import get_current_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

MODELS_DIR = "models"
LOOKBACK_WINDOW = 252
ATR_SL_MULT = 1.5
ATR_TP_MULT = 2.5
RISK_PCT = 0.02
ACCOUNT_BALANCE = 10000.0  # Fetch dynamically in production

POINT_VALUE = {"NG": 10000, "ES": 50, "EUR": 100000} 

class AsyncTraderDaemon:
    def __init__(self, symbols):
        self.symbols = symbols
        self.ib = IB()
        self.contracts = {}
        self.historical_data = {}
        self.models = {}
        
    def load_models(self):
        for sym in self.symbols:
            # Simplification: Loading standard models. Implement regime logic as needed.
            path = os.path.join(MODELS_DIR, f"{sym.lower()}_f_low_vix_xgb_calibrated.joblib")
            if os.path.exists(path):
                self.models[sym] = joblib.load(path)
                logging.info(f"Loaded model for {sym}")

    async def connect(self):
        logging.info("Connecting to IBKR gateway...")
        # 7497 is default paper trading port. 7496 is live.
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        
        for sym in self.symbols:
            if sym == "EUR": contract = Forex('EURUSD')
            else: contract = Future(sym, '202603', 'GLOBEX') # Update expiry
            
            self.ib.qualifyContracts(contract)
            self.contracts[sym] = contract
            
            # Fetch exactly enough historical bars to prime the rolling scaler
            logging.info(f"Fetching prime data for {sym}")
            bars = await self.ib.reqHistoricalDataAsync(
                contract, endDateTime='', durationStr='260 D',
                barSizeSetting='1 day', whatToShow='MIDPOINT', useRTH=True
            )
            df = pd.DataFrame(bars)
            df.set_index('date', inplace=True)
            df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}, inplace=True)
            self.historical_data[sym] = df

    async def evaluate_and_trade(self, sym):
        df = self.historical_data[sym]
        if len(df) < LOOKBACK_WINDOW: return
        
        latest_features, atr, current_price = get_current_state(df, None, None, LOOKBACK_WINDOW)
        
        if sym not in self.models: return
        
        # Ensure feature columns match model training exactly
        X = np.array(list(latest_features.values())).reshape(1, -1)
        prob = self.models[sym].predict_proba(X)[0][1]
        
        direction = "HOLD"
        if prob > 0.55: direction = "BULLISH"
        elif prob < 0.45: direction = "BEARISH"
        
        if direction != "HOLD":
            logging.info(f"Signal: {sym} {direction} (Prob: {prob:.2f})")
            await self.execute_trade(sym, direction, current_price, atr)

    async def execute_trade(self, sym, direction, entry, atr):
        pv = POINT_VALUE.get(sym, 1000)
        sl_dist = atr * ATR_SL_MULT
        tp_dist = atr * ATR_TP_MULT
        
        sl_price = entry - sl_dist if direction == "BULLISH" else entry + sl_dist
        tp_price = entry + tp_dist if direction == "BULLISH" else entry - tp_dist
        
        risk_dollars = ACCOUNT_BALANCE * RISK_PCT
        lots = max(1, int(risk_dollars / (sl_dist * pv))) # Floor to 1 contract min
        
        action = 'BUY' if direction == "BULLISH" else 'SELL'
        contract = self.contracts[sym]
        
        # Bracket Order
        parent = MarketOrder(action, lots, transmit=False)
        sl_order = StopOrder('SELL' if action == 'BUY' else 'BUY', lots, sl_price, transmit=False, parentId=parent.orderId)
        tp_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', lots, tp_price, transmit=True, parentId=parent.orderId)
        
        self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, sl_order)
        self.ib.placeOrder(contract, tp_order)
        
        logging.info(f"Executed {lots} {sym} @ {entry}. SL: {sl_price}, TP: {tp_price}")

    async def run(self):
        self.load_models()
        await self.connect()
        
        # Schedule evaluation shortly after market close or on a specific cron schedule internally
        # For demonstration, executing immediately on script run
        for sym in self.symbols:
            await self.evaluate_and_trade(sym)
            
        logging.info("Sleeping to maintain API connection...")
        while self.ib.isConnected():
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Standard CME futures symbols
    SYMBOLS = ["NG", "ES", "EUR"] 
    daemon = AsyncTraderDaemon(SYMBOLS)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        logging.info("Shutting down.")