#!/bin/bash
# retrain_pipeline.sh

set -e

echo "--- 1. Clearing old raw data files ---"
rm -f data/raw/*.csv

echo "--- 2. Acquiring raw data ---"
python acquire_data.py

echo "--- 3. Clearing old processed data (.npy, .csv) ---"
rm -f data/processed/*.npy
rm -f data/processed/*.csv 

echo "--- 3.5. Clearing old models (.joblib) ---"
rm -f models/*.joblib

echo "--- 4. Processing raw data (Dynamic Features) ---"
python process_data.py

echo "--- 5. Training models ---"
python train_model.py

echo "--- 6. Evaluating models ---"
python evaluate_model.py

echo "--- 6.5. Explaining Model Decisions ---"
python explain_model.py || echo "Warning: Explanation failed."

echo "--- 7. Analyzing Historical Performance ---"
# Note: monitor_trades and run_trader are removed.
# Analyze performance is kept to review backtest/historical logs if applicable.
python analyze_performance.py
python visualize_results.py

echo "--- Retraining Complete ---"
echo "IMPORTANT: You must now restart your async_trader.py daemon to load the new models!"