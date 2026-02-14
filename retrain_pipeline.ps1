# mlTrader/retrain_pipeline.ps1

# Stop script on error
$ErrorActionPreference = "Stop"

# 1. Clear old raw data
Write-Host "--- 1. Clearing old raw data files ---" -ForegroundColor Cyan
Remove-Item -Path "data/raw/*.csv" -ErrorAction SilentlyContinue
Write-Host "Old .csv files deleted.`n"

# 2. Acquire raw data
Write-Host "--- 2. Acquiring raw data ---" -ForegroundColor Cyan
python acquire_data.py
Write-Host ""

# 3. Clear old processed data
Write-Host "--- 3. Clearing old processed data (.npy, .csv) ---" -ForegroundColor Cyan
Remove-Item -Path "data/processed/*.npy" -ErrorAction SilentlyContinue
Remove-Item -Path "data/processed/*.csv" -ErrorAction SilentlyContinue 
Write-Host "Old processed .npy and .csv files deleted.`n"

# 3.5. Clear old models
Write-Host "--- 3.5. Clearing old models (.joblib) ---" -ForegroundColor Cyan
Remove-Item -Path "models/*.joblib" -ErrorAction SilentlyContinue
Write-Host "Old .joblib model files deleted.`n"

# 4. Process data
Write-Host "--- 4. Processing raw data ---" -ForegroundColor Cyan
python process_data.py
Write-Host ""

# 5. Train models
Write-Host "--- 5. Training models ---" -ForegroundColor Cyan
python train_model.py
Write-Host ""

# 6. Evaluate models
Write-Host "--- 6. Evaluating models ---" -ForegroundColor Cyan
python evaluate_model.py
Write-Host ""

# 6.5. Explain models (NEW)
Write-Host "--- 6.5. Explaining Model Decisions ---" -ForegroundColor Cyan
try {
    python explain_model.py
    if ($LASTEXITCODE -ne 0) { throw "Script failed" }
} catch {
    Write-Host "Warning: Explanation script failed, but continuing pipeline." -ForegroundColor Yellow
}
Write-Host ""

# 7. Monitor open trades
Write-Host "--- 7. Monitoring open trades ---" -ForegroundColor Cyan
python monitor_trades.py
Write-Host ""

# 8. Run trader
Write-Host "--- 8. Running trader (find/log new signals) ---" -ForegroundColor Cyan
python run_trader.py
Write-Host ""

# 9. Analyze results
Write-Host "--- 9. Analyzing Live Performance ---" -ForegroundColor Cyan
python analyze_performance.py
python visualize_results.py
Write-Host ""

Write-Host "--- All Steps Complete for all tickers ---" -ForegroundColor Green