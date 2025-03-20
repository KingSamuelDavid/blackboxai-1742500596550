from enum import Enum

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

def get_task_priority(file_size, processing_options):
    """Determine task priority based on size and complexity"""
    priority = Priority.MEDIUM
    
    # Adjust based on file size
    if file_size > 1000000000:  # 1GB
        priority = Priority.LOW
    elif file_size < 10000000:  # 10MB
        priority = Priority.HIGH
        
    # Adjust based on processing complexity
    if len(processing_options) > 2:
        priority = Priority(max(1, priority.value - 1))
        
    return priority.value

def configure_task_queues(app):
    """Configure Celery task queues with priorities"""
    app.conf.task_queues = {
        'high': {'exchange': 'high', 'routing_key': 'high'},
        'medium': {'exchange': 'medium', 'routing_key': 'medium'},
        'low': {'exchange': 'low', 'routing_key': 'low'},
    }
    
    app.conf.task_routes = {
        'tasks.image_to_video': {'queue': 'medium'},
        'tasks.super_resolution': {'queue': 'low'},
        'tasks.denoise': {'queue': 'medium'},
        'tasks.framerate_boost': {'queue': 'low'},
        'tasks.speech_to_text': {'queue': 'high'},
    }