from celery import Celery
from celery.signals import worker_ready
from config import Config
import os

# Create Celery app
celery_app = Celery(
    "knowledge_retrieval",
    broker=Config.REDIS_URL,
    backend=Config.REDIS_URL,
    include=['celery_tasks']
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout
    task_soft_time_limit=3300,  # 55 minutes soft timeout
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=True,
    result_expires=3600,  # Results expire after 1 hour
    # Fix CUDA multiprocessing issue
    worker_pool='solo',  # Use solo pool to avoid CUDA fork issues
    worker_concurrency=1,  # Single worker to avoid CUDA conflicts
    # Fix Redis serialization issues
    result_backend_transport_options={'master_name': 'mymaster'},
    result_compression='gzip',
    task_compression='gzip',
)

# Task routing
celery_app.conf.task_routes = {
    'celery_tasks.process_file': {'queue': 'file_processing'},
    'celery_tasks.process_question': {'queue': 'question_processing'},
}

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Called when worker is ready"""
    print("Celery worker is ready!")

if __name__ == '__main__':
    celery_app.start()
