from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime
from hiring_functions import create_folders, split_data, preprocess_and_train
import os

# Configuración inicial del DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 1),
    'retries': 0,
}

dag = DAG(
    dag_id='hiring_lineal',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# 1. Inicio del pipeline
start_task = DummyOperator(
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

# 3. Descargar datos
def download_data(**kwargs):
    execution_date = kwargs['ds']
    folder_name = f"data_{execution_date}"
    raw_path = os.path.join(folder_name, 'raw', 'data_1.csv')
    os.system(f"curl -o {raw_path} https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv")
    print(f"Archivo descargado en: {raw_path}")

download_data_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    provide_context=True,
    dag=dag
)

# 4. División de datos (hold-out)
split_data_task = PythonOperator(
    task_id='split_data',
    python_callable=split_data,
    dag=dag
)

# 5. Preprocesamiento y entrenamiento
preprocess_and_train_task = PythonOperator(
    task_id='preprocess_and_train',
    python_callable=preprocess_and_train,
    dag=dag
)

# 6. Lanzar interfaz Gradio
def launch_gradio():
    from hiring_functions import gradio_interface
    gradio_interface()

launch_gradio_task = PythonOperator(
    task_id='launch_gradio',
    python_callable=launch_gradio,
    dag=dag
)

# Configuración del pipeline
start_task >> create_folders_task >> download_data_task >> split_data_task >> preprocess_and_train_task >> launch_gradio_task
