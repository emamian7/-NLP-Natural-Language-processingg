
from datetime import datetime, timedelta
from airflow.models.dag import DAG
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

now = datetime.now()
now_to_the_hour = (now - timedelta(0, 0, 0, 0, 0, 3)).replace(minute=0, second=0, microsecond=0)
START_DATE = now_to_the_hour
DAG_NAME = 'airflow'

default_args = {'owner': 'airflow', 'depends_on_past': True, 'start_date': days_ago(2)}
dag = DAG(DAG_NAME, schedule_interval='*/10 * * * *', default_args=default_args)

run_this_1 = DummyOperator(task_id='Bert-Pair_question', dag=dag)