#!/bin/bash
# retrain_pipeline.sh

# This tells the script to exit immediately if any command fails
set -e

# 1. Clear old raw data
echo "--- 1. Clearing old raw data files ---"
# rm -f (force) is the equivalent of Remove-Item -ErrorAction SilentlyContinue
rm -f data/raw/*.csv
echo "Old .csv files deleted."
echo ""

# 2. Acquire raw data
echo "--- 2. Acquiring raw data ---"
python acquire_data.py
echo ""

# 3. Clear old processed data
echo "--- 3. Clearing old processed data (.npy, .csv) ---"
rm -f data/processed/*.npy
rm -f data/processed/*.csv 
echo "Old processed .npy and .csv files deleted."
echo ""

# 3.5. Clear old models
echo "--- 3.5. Clearing old models (.joblib) ---"
rm -f models/*.joblib
echo "Old .joblib model files deleted."
echo ""

# 4. Process data
echo "--- 4. Processing raw data ---"
python process_data.py
echo ""

# 5. Train models
echo "--- 5. Training models ---"
python train_model.py
echo ""

# 6. Evaluate models
echo "--- 6. Evaluating models ---"
python evaluate_model.py
echo ""

# 7. Monitor open trades (Check before opening new ones)
echo "--- 7. Monitoring open trades ---"
python monitor_trades.py
echo ""

# 8. Run trader (Find and log new signals)
echo "--- 8. Running trader (find/log new signals) ---"
python run_trader.py
echo ""

# 9. Analyze results (Show the final P/L)
echo "--- 9. Analyzing Live Performance ---"
python visualize_results.py
echo ""

echo "--- All Steps Complete for all tickers ---"