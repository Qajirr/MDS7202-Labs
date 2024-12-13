def train_model_task(data_path, execution_date, **kwargs):
    # Cargar los datos
    data = pd.read_parquet(data_path)

    # Selección de características
    selected_features = [
        'wallet_age', 'incoming_tx_count', 'outgoing_tx_count', 'net_incoming_tx_count',
        'risky_tx_count', 'outgoing_tx_sum_eth', 
        'incoming_tx_sum_eth', 'min_eth_ever', 'total_balance_eth', 
        'risk_factor', 'total_collateral_eth', 'total_available_borrows_eth', 
        'avg_weighted_risk_factor', 'risk_factor_above_threshold_daily_count', 
        'avg_risk_factor', 'max_risk_factor', 'borrow_amount_sum_eth', 
        'borrow_amount_avg_eth', 'deposit_count', 'deposit_amount_sum_eth', 
        'withdraw_amount_sum_eth', 'liquidation_amount_sum_eth', 
        'target'
    ]

    data = data[selected_features]

    # Dividir en características (X) y variable objetivo (y)
    X = data.drop(columns='target')
    y = data['target']

    # Dividir el dataset en 70%-20%-10%
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

    # Configurar el modelo con los mejores parámetros
    xgb_model = XGBClassifier(
        learning_rate=0.0066379026800933655,
        max_depth=13,
        n_estimators=1349,
        gamma=3.4613493503023274,
        subsample=0.8553835358702008,
        colsample_bytree=0.6028780401972087,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'  # Para evitar advertencias
    )

    # Entrenar el modelo
    xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluar el modelo en el conjunto de prueba
    y_pred = xgb_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Realizar el tracking en MLflow
    mlflow_tracking(xgb_model, model_name="XGBoost", model_path=f"model_{execution_date}.xgb")

    return xgb_model, report
