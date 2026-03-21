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

echo "--- Phase 3.5: Pipeline Integrity Verification ---"
# Pipeline Error Aggregation Check
python -c "
import json, sys
try:
    with open('models/model_manifest.json') as f:
        manifest = json.load(f)
    bases = set(k.split('_low_vix')[0].split('_high_vix')[0] for k in manifest.keys())
    if len(bases) < 15:
        print(f'CRITICAL ERROR: Manifest contains only {len(bases)}/15 base tickers. Upstream failure detected.')
        sys.exit(1)
    print('Manifest verification passed. All 15 base tickers accounted for.')
except Exception as e:
    print(f'CRITICAL ERROR reading manifest: {e}')
    sys.exit(1)
" || exit 1

echo "--- Phase 4: Strategy Evaluation & Model Selection ---"
# This selects the best model per regime and sets thresholds
python evaluate_model.py

echo "--- Pipeline Complete: System ready for live trading ---"
