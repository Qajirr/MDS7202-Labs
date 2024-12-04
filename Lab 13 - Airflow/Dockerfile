FROM python:3.10-slim

WORKDIR /airflow_project

ENV AIRFLOW_HOME=/airflow_project

RUN pip install apache-airflow
RUN pip install scikit-learn pandas numpy lightgbm

RUN airflow db init

EXPOSE 8080

RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

COPY ./dags /airflow_project/dags
COPY ./logs /airflow_project/logs
COPY ./plugins /airflow_project/plugins

CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]
