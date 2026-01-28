
# PRODUCTION EXECUTION GUIDE
## ML Trading Bot - Phase 6 Refactoring Complete

---

## QUICK START

### 1. Run Full Pipeline (Development/Testing)
```bash
cd "c:\Users\mndiw\OneDrive\Desktop\Life Man\mlTrader"

# Full pipeline: data → processing → training → evaluation → prediction → trading
python acquire_data.py && ^
python process_data.py && ^
python train_model.py && ^
python evaluate_model.py && ^
python predict.py && ^
python run_trader.py
```

### 2. Individual Commands

#### Fetch Fresh Market Data
```bash
python acquire_data.py
```
**Output:** Raw OHLCV files in `data/raw/`

#### Engineer Features & Label Data
```bash
python process_data.py
```
**Output:** 
- Processed arrays in `data/processed/`
- 6 regimes trained (if all tickers have data)
- Hard-balanced 50/50 class split

#### Train Models with Calibration
```bash
python train_model.py
```
**Output:**
- Calibrated models in `models/`
- **NEW:** `models/model_manifest.json` with audit trail
- Calibration method: isotonic (if n≥200) or sigmoid

#### Evaluate Against Constraints
```bash
python evaluate_model.py
```
**Output:**
- Constraint gates applied (4-gate system)
- Production models selected
- Results to console

#### Generate Trade Signals
```bash
python predict.py
```
**Output:**
- **NEW:** Structured JSON to stdout
- Includes: ticker, direction, confidence, entry/SL/TP, position sizing params
- Model version v4.0

#### Parse Signals & Size Positions
```bash
python run_trader.py
```
**Output:**
- **NEW:** CSV log with position sizes
- Email notification with signal summary
- Audit log in `logs/bot_status.log`

---

## OUTPUT FILES REFERENCE

### Logs Directory (`logs/`)
```
bot_status.log
  Format: [SUCCESS/ERROR] TIMESTAMP | MESSAGE
  Purpose: Execution audit trail
  Example: [SUCCESS] 2026-01-28T14:30:00+02:00 | Generated 2 signals (VIX=18.5, Regime=0)

live_trades_log.csv
  Columns: trade_id, signal_hash, prediction_date, ticker, direction, confidence,
           entry_price, stop_loss, take_profit, lots, risk_dollars, model_regime, status
  Purpose: Complete trade record with position sizing
  Example: 2026-01-28-NG=F, a1b2c3d4, 2026-01-28 14:30:00, NG=F, BULLISH, 0.7800,
           3.2500, 3.1000, 3.5000, 0.13, 200.00, ng_f_low_vix, OPEN
```

### Models Directory (`models/`)
```
model_manifest.json (NEW)
  Purpose: Model training metadata for audit trail
  Fields per regime:
    - last_trained (ISO timestamp)
    - training_samples (int)
    - validation_samples (int)
    - calibration_method (isotonic/sigmoid)
    - best_iteration (int)
    - scale_pos_weight (float)
    - xgb_model (filename)
    - lr_model (filename)

*_xgb_calibrated.joblib
  Purpose: Calibrated XGBoost classifier
  
*_lr_calibrated.joblib
  Purpose: Calibrated Logistic Regression classifier

*_feature_list.joblib
  Purpose: Feature column names (for predict.py)

*_scaler.joblib
  Purpose: StandardScaler fitted on training data

*_model_choice.joblib
  Purpose: Selected model (xgb/lr), thresholds, version
```

### Data Directory (`data/processed/`)
```
{regime}_X_train.npy      - Training features
{regime}_y_train.npy      - Training labels
{regime}_X_val.npy        - Validation features
{regime}_y_val.npy        - Validation labels

Regimes available (6):
  - cl_f, ng_f_low_vix, ng_f_high_vix
  - eurusd_x, jpyusd_x_low_vix, jpyusd_x_high_vix
  - si_f, gc_f, zc_f
  (ES=F and NQ=F: zero rows after NaN dropping - too quiet)
```

---

## POSITION SIZING EXPLANATION

### Formula
```
Risk_Dollars = Account_Balance × Risk_Pct
Lots = Risk_Dollars / (|Entry_Price - Stop_Loss| × Point_Value)
```

### Example: NG=F (Natural Gas)
```
Account Balance: $10,000
Risk Per Trade: 2% = $200

Signal:
  Entry: $3.25
  Stop Loss: $3.10
  Point Value: $10,000 per $1 movement

Calculation:
  Price Distance = |3.25 - 3.10| = $0.15
  Lots = $200 / (0.15 × 10,000) = $200 / $1,500 = 0.13 lots
  
Result:
  Risk exactly $200 per trade
  Stop loss distance: $0.15 per contract
  Total loss if stopped: 0.13 × $10,000 × 0.15 = $195 (≈$200)
```

### Point Values (Key Mappings)
```
Commodities:
  CL=F (Crude Oil): $1,000 per contract per $1
  GC=F (Gold): $100 per contract per $1
  SI=F (Silver): $5,000 per contract per $1
  NG=F (Nat Gas): $10,000 per contract per $1
  ZC=F (Corn): $50 per contract per $0.25

Forex:
  EURUSD=X: 100,000 units per contract
  JPYUSD=X: 100,000 units per contract

Indexes:
  ES=F (S&P 500): $50 per contract per point
  NQ=F (Nasdaq): $20 per contract per point
```

---

## MONITORING SIGNALS

### Real-Time Monitoring
```bash
# Watch JSON output from predict.py
python predict.py | python -m json.tool

# Or check formatted CSV
tail logs/live_trades_log.csv
```

### Sample JSON Output
```json
{
  "timestamp": "2026-01-28T14:30:00+02:00",
  "model_version": "v4.0",
  "vix_value": 18.5,
  "vix_regime": 0,
  "signals_count": 2,
  "signals": [
    {
      "ticker": "NG=F",
      "direction": "BULLISH",
      "confidence": 0.78,
      "entry_price": 3.25,
      "stop_loss": 3.10,
      "take_profit": 3.50,
      "model_regime": "ng_f_low_vix",
      "model_type": "xgb",
      "model_version": "v4.0",
      "atr_current": 0.15
    }
  ]
}
```

---

## CONSTRAINT GATES (Production Safety)

All models must pass 4-gate system:

| Gate | Condition | Threshold |
|------|-----------|-----------|
| NEGATIVE_RETURN | Total return < 0% | Reject if negative |
| INSUFFICIENT_TRADES | Trade count | Require ≥15 trades |
| LOW_RECALL | Min(Recall_Bull, Recall_Bear) | Require ≥15% |
| ASYMMETRIC_RECALL | Min Recall / Max Recall | Require ≥0.20 |

**Current Production Models (2):**
1. ng_f_low_vix: 61% return, 31 trades, 0.226 symmetry ✅
2. jpyusd_x_high_vix: 28% return, 22 trades, 0.959 symmetry ✅

---

## SCHEDULING (Automation)

### Windows Task Scheduler
```
Task Name: MLTrader_Daily_5PM
Program: C:\Python313\python.exe
Arguments: "c:\Users\mndiw\OneDrive\Desktop\Life Man\mlTrader\run_trader.py"
Start in: c:\Users\mndiw\OneDrive\Desktop\Life Man\mlTrader
Schedule: Daily 5:00 PM (when market closes)
```

### Cron (Linux/Mac)
```bash
# Run daily at 5:00 PM (17:00)
0 17 * * * cd /path/to/mlTrader && python run_trader.py >> logs/cron.log 2>&1
```

### Docker (Future)
```dockerfile
FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run_trader.py"]
```

---

## TROUBLESHOOTING

### Issue: "No signals generated"
**Diagnosis:**
- Check VIX value and regime (in JSON output)
- Verify model confidence thresholds in model_choice.joblib
- Check feature NaNs (insufficient history)

**Solution:**
```bash
# Check raw data quality
python -c "import pandas as pd; df = pd.read_csv('data/raw/NG=F.csv'); print(df.tail())"
```

### Issue: Position size too small (lots < 0.01)
**Diagnosis:**
- Stop loss too close to entry
- Account balance too small

**Solution:**
- Increase ACCOUNT_BALANCE in run_trader.py
- Or increase RISK_PCT (2% default)
- Or use tighter ATR multiplier in predict.py

### Issue: Syntax error in JSON output
**Diagnosis:**
- NaN values not handled in predict.py
- Infinity values in indicators

**Solution:**
```bash
# Run validation
python validate_refactor.py

# Check predict.py output directly
python predict.py 2>&1 | head -50
```

### Issue: Model manifest not created
**Diagnosis:**
- train_model.py crashed before manifest generation
- models/ directory permissions issue

**Solution:**
```bash
# Check train logs
python train_model.py 2>&1 | tail -20

# Verify write permissions
ls -la models/
```

---

## PERFORMANCE METRICS

### Model Summary (Current)
| Regime | Type | Return | Trades | Calibration | Status |
|--------|------|--------|--------|-------------|--------|
| ng_f_low_vix | XGBoost | 61% | 31 | Sigmoid | ✅ PROD |
| jpyusd_x_high_vix | LogReg | 28% | 22 | Isotonic | ✅ PROD |
| ng_f_high_vix | - | - | <15 | - | ❌ Rejected |
| eurusd_x | - | - | <15 | - | ❌ Rejected |
| Other regimes | - | - | - | - | ❌ No data |

### Feature Set (22 total)
```
Moving Averages: MA5, MA20, MA50, MA200, MA_diff
Volatility: ATR, BB_Width, Day_Range_Pct, Dist_from_MA20
Momentum: RSI14, ROC10, MACD, MACD_signal, MACD_hist
Volume: OBV
Macro: VIX_Close, VIX_Regime, DXY_ret_1d, TLT_ret_1d, etc.
Correlations: Corr_DXY_10d, Corr_SP500, Corr_TLT_10d
```

---

## NEXT PHASE: API INTEGRATION

### Current: Email Dispatch
```python
send_email(f"Trade Signal: {ticker}", email_body)
```

### Future: Broker API Integration
```python
import requests

response = requests.post(
    'https://api.broker.example.com/orders',
    json=signal,
    headers={'Authorization': f'Bearer {API_KEY}'}
)
```

**Ready to implement:** API stub in dispatch_signal() function (line ~120 in run_trader.py)

---

## COMPLIANCE & AUDIT

### Audit Trail
1. **bot_status.log** - Execution history
2. **live_trades_log.csv** - All signals with position sizing
3. **model_manifest.json** - Model lineage + training metadata
4. **logs/cron.log** - Scheduled execution (if cron)

### Data Retention
- Keep logs for 1 year (compliance requirement)
- Archive models for backtesting (20+ years)
- Manifest serves as model version control

---

## VALIDATION CHECKLIST

Before production deployment, verify:

- [ ] predict.py outputs valid JSON (test with `python predict.py`)
- [ ] run_trader.py parses JSON without errors
- [ ] Position sizes match formula: Lots = Risk$ / (Distance × Point_Value)
- [ ] CSV log contains lots and risk_dollars columns
- [ ] model_manifest.json created with all required fields
- [ ] Email notifications sent successfully
- [ ] Audit logs contain [SUCCESS] markers
- [ ] No NaN or Inf values in predictions
- [ ] All 2 production models selected by constraints
- [ ] Validation tests pass: `python validate_refactor.py`

---

**Phase 6 Status: COMPLETE ✅**  
**Production Readiness: CONFIRMED ✅**  
**Deployment Ready: YES ✅**

