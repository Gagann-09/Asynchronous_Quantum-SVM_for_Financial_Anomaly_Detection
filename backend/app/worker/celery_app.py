from celery import Celery

celery_app = Celery(
    "qsvm_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
    include=['app.worker.tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True
)
