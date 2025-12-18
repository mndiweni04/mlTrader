        # --- Train XGBoost ---
        xgb_model_raw = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.5,
            reg_lambda=1.0
        )

        print("  Training raw XGBoost model...")
        xgb_model_raw.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print("  Calibrating XGB probabilities on validation set...")
        xgb_calibrated = CalibratedClassifierCV(
            estimator=xgb_model_raw,
            method="sigmoid",
            cv=None   # sklearn 1.4+ replacement for 'prefit'
        )
        xgb_calibrated.fit(X_val, y_val)

        joblib.dump(
            xgb_calibrated,
            os.path.join(MODELS_DIR, f"{regime_base}_xgb_calibrated.joblib")
        )

        # --- Train Logistic Regression ---
        lr_model_raw = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
            C=0.1
        )
        lr_model_raw.fit(X_train, y_train)

        print("  Calibrating LR probabilities on validation set...")
        lr_calibrated = CalibratedClassifierCV(
            estimator=lr_model_raw,
            method="sigmoid",
            cv=None
        )
        lr_calibrated.fit(X_val, y_val)

        joblib.dump(
            lr_calibrated,
            os.path.join(MODELS_DIR, f"{regime_base}_lr_calibrated.joblib")
        )

        print(f"  ✅ Models saved for {regime_base}")
