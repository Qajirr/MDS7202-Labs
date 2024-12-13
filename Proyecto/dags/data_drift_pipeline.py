from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from functions import create_folders, download_data, load_and_merge_parquet, data_drift_analysis, re_train_model, save_model, mlflow_tracking
from sklearn.model_selection import train_test_split
import os

# Configuración inicial del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    dag_id='dynamic_pipeline_with_monitoring',
    default_args=default_args,
    schedule_interval='0 15 * * 0',
    catchup=True,
)

def branching_logic(**kwargs):
    execution_date = kwargs['execution_date']

    if isinstance(execution_date, str):
        execution_date = datetime.fromisoformat(execution_date)

    execution_date_naive = execution_date.replace(tzinfo=None)

    if execution_date_naive.hour % 2 == 0:
        return "train_initial_model"
    else:
        return "check_data_drift"

def train_initial_model(**kwargs):
    data = load_and_merge_parquet(kwargs['execution_date'])

    X_train, X_val, y_train, y_val = train_test_split(
        data.drop(columns='target'), data['target'], test_size=0.2, random_state=42
    )
    new_model = re_train_model(X_train, y_train, X_val, y_val)
    save_model(new_model, "models/initial_model.joblib")
    mlflow_tracking(new_model)
    return new_model

def download_all_data(execution_date):
    base_url = "https://gitlab.com/mds7202-2/proyecto-mds7202/-/raw/main/production_stage_files/"

    available_dates = ["2024-11-13", "2024-11-29", "2024-12-6"]

    dates_to_download = [
        date
        for date in available_dates
        if datetime.strptime(date, '%Y-%m-%d') <= execution_date
    ]

    for date in dates_to_download:
        download_data(
    file_url=f"{base_url}{date}-X.parquet",
    output_file=f"{date}-X.parquet",
    ds=execution_date
)
    download_data(
        file_url=f"{base_url}{date}-y.parquet",
        output_file=f"{date}-y.parquet",
        ds=execution_date
    )

def check_data_drift(current_data, reference_data, **kwargs):
    drift_report = data_drift_analysis(current_data, reference_data)
    drift_detected = any(
        [value['mean_diff'] > 0.1 or value['std_diff'] > 0.1 for value in drift_report.values()]
    )

    if drift_detected:
        return 'retrain_model'
    return 'no_action_needed'

def retrain_model(**kwargs):
    data = load_and_merge_parquet(kwargs['execution_date'])

    X_train, X_val, y_train, y_val = train_test_split(
        data.drop(columns='target'), data['target'], test_size=0.2, random_state=42
    )
    new_model = re_train_model(X_train, y_train, X_val, y_val)
    save_model(new_model, "models/new_model.joblib")
    mlflow_tracking(new_model)
    return new_model

# 1. Inicio del pipeline
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# 2. Crear carpetas
create_folders_task = PythonOperator(
    task_id='create_folders',
    python_callable=create_folders,
    provide_context=True,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    dag=dag
)

# 3. Branching
branching_task = BranchPythonOperator(
    task_id='branching',
    python_callable=branching_logic,
    provide_context=True,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    dag=dag
)

# Desacrgar datos
download_all_data_task = PythonOperator(
    task_id='download_all_data',
    python_callable=download_all_data,
    provide_context=True,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    dag=dag
)

load_local_data = PythonOperator(
    task_id='load_local_data',
    python_callable=load_and_merge_parquet,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    provide_context=True,
    dag=dag
)

retrain_model_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    provide_context=True,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    dag=dag
)

train_initial_model_task = PythonOperator(
    task_id='train_initial_model',
    python_callable=train_initial_model,
    provide_context=True,
    op_kwargs={'execution_date': '{{ execution_date }}'},
    dag=dag
)

check_data_drift_task = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    provide_context=True,
    op_kwargs={
        'current_data': '{{ task_instance.xcom_pull(task_ids="load_local_data") }}',
        'reference_data': '{{ task_instance.xcom_pull(task_ids="load_local_data") }}'
    },
    dag=dag
)

# Configuración del flujo del pipeline
start_pipeline >> create_folders_task >> branching_task

# Dependencias para el flujo de entrenamiento inicial o verificación de drift
branching_task >> train_initial_model_task >> download_all_data_task >> load_local_data
branching_task >> check_data_drift_task >> download_all_data_task >> load_local_data

# Flujo para cuando se detecta drift o no
check_data_drift_task >> [retrain_model_task, DummyOperator(task_id="no_action_needed")]

# Dependencia para retrain_model_task solo cuando hay drift
retrain_model_task >> download_all_data_task >> load_local_data
