import os
import pickle
import json
import optuna
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

# Crear directorios necesarios
models_path = "models"
plots_path = "plots"
os.makedirs(models_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)

# Cargar los datos
def load_data():
    data_path = 'data/X_t1.parquet'
    data_path_y = 'data/y_t1.parquet'
    df = pd.read_parquet(data_path)
    dfy = pd.read_parquet(data_path_y)
    # Restablecer los índices para ambos DataFrames
    df = df.reset_index(drop=True)
    dfy = dfy.reset_index(drop=True)
    data = df.copy()
    data['target'] = dfy
    selected_features = [
        'wallet_age', 'incoming_tx_count', 'outgoing_tx_count', 'net_incoming_tx_count',
        'risky_tx_count', 'outgoing_tx_sum_eth', 'incoming_tx_sum_eth', 'min_eth_ever',
        'total_balance_eth', 'risk_factor', 'total_collateral_eth', 'total_available_borrows_eth',
        'avg_weighted_risk_factor', 'risk_factor_above_threshold_daily_count', 'avg_risk_factor',
        'max_risk_factor', 'borrow_amount_sum_eth', 'borrow_amount_avg_eth', 'deposit_count',
        'deposit_amount_sum_eth', 'withdraw_amount_sum_eth', 'liquidation_amount_sum_eth', 'target'
    ]
    
    data = data[selected_features]
    X = data.drop(columns='target')
    y = data['target']
    
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Función de optimización para Optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

    X_train, X_valid, y_train, y_valid = load_data()
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    valid_f1 = f1_score(y_valid, y_pred)

    # Loguear en MLflow
    with mlflow.start_run(nested=True, run_name=f"XGBoost Trial {trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metric("valid_f1", valid_f1)

    return valid_f1

# Función principal
def optimize_model():
    experiment_name = "XGBoost_Optimization"
    mlflow.set_experiment(experiment_name)

    study = optuna.create_study(direction="maximize")
    with mlflow.start_run(run_name="XGBoost Optimization"):
        study.optimize(objective, n_trials=50)

        # Mejor resultado
        best_trial = study.best_trial
        best_params = best_trial.params
        best_f1 = best_trial.value

        # Loguear resultados finales en MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_valid_f1", best_f1)

        # Guardar el mejor modelo
        X_train, X_valid, y_train, y_valid = load_data()
        best_model = xgb.XGBClassifier(**best_params, eval_metric="logloss")
        best_model.fit(X_train, y_train)

        best_model_path = os.path.join(models_path, "best_model.pkl")
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(best_model_path)

if __name__ == "__main__":
    optimize_model()