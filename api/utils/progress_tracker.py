import time
from celery import current_task

class ProgressTracker:
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, step=1, status=""):
        """Update progress and estimate remaining time"""
        self.current_step += step
        elapsed = time.time() - self.start_time
        progress = (self.current_step / self.total_steps) * 100
        
        if self.current_step > 0:
            time_per_step = elapsed / self.current_step
            remaining = time_per_step * (self.total_steps - self.current_step)
        else:
            remaining = 0
            
        if current_task:
            current_task.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'current': self.current_step,
                    'total': self.total_steps,
                    'status': status,
                    'remaining': remaining
                }
            )