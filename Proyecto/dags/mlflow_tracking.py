import mlflow
import mlflow.xgboost
from mlflow.models import Model

def mlflow_tracking(model, model_name="XGBoost", model_path="model.xgb"):
    try:
        with mlflow.start_run():
            # Registrar parámetros del modelo
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            
            # Guardar el modelo XGBoost
            model.save_model(model_path)  # Guardamos el modelo en disco
            mlflow.log_artifact(model_path)  # Guardamos el modelo como artefacto en MLflow
            
            # También podemos registrar un modelo como un modelo de MLflow
            mlflow.xgboost.log_model(model, "model")  # Esto es muy útil para cargar después

            print(f"Modelo guardado y registrado en MLflow: {model_name}")
    except Exception as e:
        print(f"Error al realizar el tracking en MLflow: {str(e)}")
