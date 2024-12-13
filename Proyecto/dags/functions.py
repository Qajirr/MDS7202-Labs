import os
import pandas as pd
import joblib
import shap
import optuna
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import mlflow
import subprocess
import mlflow.xgboost

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
if not mlflow.get_experiment_by_name("XGboost"):
    mlflow.create_experiment("XGboost")
mlflow.set_experiment("XGboost")

# Función para descargar los datos
def download_data(file_url, output_file, **kwargs):
    execution_date = kwargs['ds']
    folder_name = f"data_{execution_date}"
    raw_path = os.path.join(folder_name, 'raw')
    os.makedirs(raw_path, exist_ok=True)
    file_path = os.path.join(raw_path, output_file)

    result = subprocess.run(
        ['curl', '-o', file_path, file_url],
        check=True,
        capture_output=True
    )
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No se descargó {output_file}. Error: {result.stderr.decode()}")
    print(f"Archivo descargado en: {file_path}")

# Función para crear carpetas
def create_folders(execution_date):
    folder_name = f"data_{execution_date}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'preprocessed'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'splits'), exist_ok=True)
    os.makedirs(os.path.join(folder_name, 'models'), exist_ok=True)

def load_and_merge_parquet(execution_date):
    raw_path = os.path.join(f'data_{execution_date}', 'raw')
    data_files = [os.path.join(raw_path, f) for f in os.listdir(raw_path) if f.endswith('.parquet')]

    if not data_files:
        print("No se encontraron archivos Parquet para unir.")
        return None

    df = pd.concat([pd.read_parquet(file) for file in data_files], ignore_index=True)
    return df

# Función para análisis de drift de los datos
def data_drift_analysis(new_data, previous_data):
    drift_report = {}
    for column in new_data.columns:
        mean_diff = abs(new_data[column].mean() - previous_data[column].mean())
        std_diff = abs(new_data[column].std() - previous_data[column].std())
        drift_report[column] = {'mean_diff': mean_diff, 'std_diff': std_diff}
    return drift_report

# Función para reentrenar el modelo con Optuna
def re_train_model(X_train, y_train, X_val, y_val):
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e-1),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1),
        }

        model = XGBClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        return model.best_score_

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

    return model

# Función para el análisis SHAP
def shap_analysis(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

# Función para guardar el modelo
def save_model(model, model_path):
    joblib.dump(model, model_path)
    print(f"Modelo guardado en {model_path}")

def mlflow_tracking(model, model_name="XGBoost"):
    try:
        with mlflow.start_run():
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            model_path = "model.xgb"
            model.save_model(model_path)
            mlflow.log_artifact(model_path)
    except Exception as e:
        print(f"Error al realizar el tracking en MLflow: {str(e)}")


# Función principal de entrenamiento
def train_model(data_path):
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

    return xgb_model, report
