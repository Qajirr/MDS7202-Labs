from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from datetime import datetime
from hiring_dynamic_functions import (
    create_folders,
    load_and_merge,
    split_data,
    train_model,
    evaluate_models,
)
import os
import subprocess

# Configuración inicial del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    dag_id='dynamic_hiring_pipeline',
    default_args=default_args,
    schedule_interval='0 15 5 * *',
    catchup=True,
)

# Función de branching
def branching_logic(execution_date, **kwargs):
    execution_date_naive = execution_date.replace(tzinfo=None)
    branch_task = 'download_data_2' if execution_date_naive >= datetime(2024, 11, 1) else 'download_data_1'
    return branch_task


# Función para descargar data
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
    dag=dag
)

# 3. Branching
branching_task = BranchPythonOperator(
    task_id='branching',
    python_callable=branching_logic,
    provide_context=True,
    dag=dag
)

# Descargar data
download_data_1 = PythonOperator(
    task_id='download_data_1',
    python_callable=download_data,
    op_kwargs={
        'file_url': 'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv',
        'output_file': 'data_1.csv',
    },
    provide_context=True,
    dag=dag
)

download_data_2 = PythonOperator(
    task_id='download_data_2',
    python_callable=download_data,
    op_kwargs={
        'file_url': 'https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv',
        'output_file': 'data_2.csv',
    },
    provide_context=True,
    dag=dag
)

# Marcador de unión después de branching
merge_branch = DummyOperator(
    task_id='merge_branch',
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag
)

# 4. Unir datos
merge_data_task = PythonOperator(
    task_id='merge_data',
    python_callable=load_and_merge,
    provide_context=True,
    dag=dag
)

# 5. Dividir datos
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    provide_context=True,
    dag=dag
)

# 6. Entrenar modelos
train_rf_task = PythonOperator(
    task_id='train_rf',
    python_callable=train_model,
    op_kwargs={'model': RandomForestClassifier()},
    provide_context=True,
    dag=dag
)

train_lr_task = PythonOperator(
    task_id='train_lr',
    python_callable=train_model,
    op_kwargs={'model': LogisticRegression()},
    provide_context=True,
    dag=dag
)

train_svc_task = PythonOperator(
    task_id='train_svc',
    python_callable=train_model,
    op_kwargs={'model': SVC()},
    provide_context=True,
    dag=dag
)

# 7. Evaluar modelos
evaluate_models_task = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    provide_context=True,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag
)

# Configuración del pipeline
start_pipeline >> create_folders_task >> branching_task
branching_task >> [download_data_1, download_data_2] >> merge_branch
merge_branch >> merge_data_task >> split_data_task
split_data_task >> [train_rf_task, train_lr_task, train_svc_task] >> evaluate_models_task
