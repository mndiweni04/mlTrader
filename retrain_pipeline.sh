#!/bin/bash
# retrain_pipeline.sh - Orchestrates the full ML update
set -e

echo "--- Phase 1: Data Acquisition ---"
python acquire_data.py

echo "--- Phase 2: Feature Engineering & Labeling ---"
# This now generates the critical test_indices.joblib
python process_data.py

echo "--- Phase 3: Model Training (XGB, LR, CatBoost) ---"
python train_model.py

echo "--- Phase 4: Strategy Evaluation & Model Selection ---"
# This selects the best model per regime and sets thresholds
python evaluate_model.py

echo "--- Pipeline Complete: System ready for live trading ---"