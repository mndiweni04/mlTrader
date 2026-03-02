# mlTrader/retrain_pipeline.ps1
$ErrorActionPreference = "Stop"

Write-Host "--- 1. Clearing old raw data files ---" -ForegroundColor Cyan
Remove-Item -Path "data/raw/*.csv" -ErrorAction SilentlyContinue

Write-Host "--- 2. Acquiring raw data ---" -ForegroundColor Cyan
python acquire_data.py

Write-Host "--- 3. Clearing old processed data (.npy, .csv) ---" -ForegroundColor Cyan
Remove-Item -Path "data/processed/*.npy" -ErrorAction SilentlyContinue
Remove-Item -Path "data/processed/*.csv" -ErrorAction SilentlyContinue 

Write-Host "--- 3.5. Clearing old models (.joblib) ---" -ForegroundColor Cyan
Remove-Item -Path "models/*.joblib" -ErrorAction SilentlyContinue

Write-Host "--- 4. Processing raw data (Dynamic Features) ---" -ForegroundColor Cyan
python process_data.py

Write-Host "--- 5. Training models ---" -ForegroundColor Cyan
python train_model.py

Write-Host "--- 6. Evaluating models ---" -ForegroundColor Cyan
python evaluate_model.py

Write-Host "--- 6.5. Explaining Model Decisions ---" -ForegroundColor Cyan
try {
    python explain_model.py
    if ($LASTEXITCODE -ne 0) { throw "Script failed" }
} catch {
    Write-Host "Warning: Explanation script failed, but continuing pipeline." -ForegroundColor Yellow
}

Write-Host "--- 7. Analyzing Historical Performance ---" -ForegroundColor Cyan
python analyze_performance.py
python visualize_results.py

Write-Host "--- Retraining Complete ---" -ForegroundColor Green
Write-Host "IMPORTANT: You must now restart your async_trader.py daemon to load the new models!" -ForegroundColor Yellow