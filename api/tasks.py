from celery import Celery, states
from celery.exceptions import Ignore
import os
import logging
import subprocess
import json
from pathlib import Path
import shutil
import time
from config import *
from utils.memory_manager import MemoryManager
from utils.progress_tracker import ProgressTracker
from utils.cache_manager import CacheManager
from utils.task_priority import get_task_priority, configure_task_queues
from utils.error_recovery import ProcessingCheckpoint
from utils.resource_monitor import ResourceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'tasks.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Celery
app = Celery('tasks', 
             broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
             backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'))

# Configure task queues
configure_task_queues(app)

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=TASK_TIMEOUT,
    task_soft_time_limit=TASK_TIMEOUT - 30,
    task_track_started=True,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=MAX_RETRIES
)

# Initialize utilities
memory_manager = MemoryManager()
cache_manager = CacheManager()
resource_monitor = ResourceMonitor()
resource_monitor.start()

def cleanup_temp_files(temp_dir):
    """Clean up temporary files older than TTL"""
    try:
        current_time = time.time()
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.getctime(item_path) < (current_time - TEMP_FILE_TTL):
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {str(e)}")

def run_command(command, error_message, task=None, checkpoint=None, step_name=None):
    """
    Execute a command and handle its output with improved error handling and checkpointing.
    """
    try:
        if task:
            task.update_state(state='PROGRESS', meta={'current': 'Running command', 'command': ' '.join(command)})
        
        logger.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TASK_TIMEOUT
        )
        
        # Record successful step if checkpointing is enabled
        if checkpoint and step_name:
            checkpoint.step_completed(step_name, parameters={'command': command})
            
        logger.info(f"Command output: {result.stdout}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {TASK_TIMEOUT} seconds")
        if checkpoint and step_name:
            checkpoint.step_failed(step_name, "Command timeout")
        if task:
            task.update_state(state=states.FAILURE, meta={'error': 'Task timed out'})
        raise Ignore()
        
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_message}: {str(e)}")
        logger.error(f"Command stderr: {e.stderr}")
        if checkpoint and step_name:
            checkpoint.step_failed(step_name, str(e))
        if task:
            task.update_state(state=states.FAILURE, meta={'error': str(e)})
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during command execution: {str(e)}")
        if checkpoint and step_name:
            checkpoint.step_failed(step_name, str(e))
        if task:
            task.update_state(state=states.FAILURE, meta={'error': str(e)})
        return False

@app.task(name="tasks.image_to_video", bind=True)
def image_to_video(self, request):
    """
    Convert images to video and apply optional AI enhancements with improved processing.
    """
    try:
        # Create task-specific temp directory and checkpoint
        task_temp_dir = os.path.join(TEMP_DIR, self.request.id)
        os.makedirs(task_temp_dir, exist_ok=True)
        checkpoint = ProcessingCheckpoint(self.request.id, task_temp_dir)
        
        # Initialize progress tracker
        total_steps = 1 + sum(1 for opt in request.get("ai_options", {}).values() if opt)
        progress = ProgressTracker(total_steps)
        
        # Clean up old temp files
        cleanup_temp_files(TEMP_DIR)
        
        # Update task state
        self.update_state(state='PROGRESS', meta={'current': 'Starting conversion process'})
        
        logger.info(f"Starting image to video conversion process - Task ID: {self.request.id}")
        
        # Check cache for existing result
        cache_key = cache_manager.get_cache_key(
            str(request["input"]),
            "image_to_video",
            request
        )
        cached_result = cache_manager.get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Using cached result: {cached_result}")
            return {
                "status": "Complete",
                "output": cached_result,
                "cached": True
            }
        
        # Ensure output directory exists
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        
        # Base output path with task ID
        base_output = output_dir / f"video_output_{self.request.id}.mp4"
        current_input = " ".join(request["input"])
        
        # Initial conversion from images to video
        progress.update(status='Converting images to video')
        logger.info("Converting images to video")
        success = run_command(
            [
                "python3", "/app/processing/image_to_video.py",
                "--input", *request["input"],
                "--fps", str(request["fps"]),
                "--output", str(base_output)
            ],
            "Failed to convert images to video",
            self,
            checkpoint,
            "image_to_video"
        )
        
        if not success:
            self.update_state(state=states.FAILURE, meta={'error': 'Failed to convert images to video'})
            return {"status": "Error", "error": "Failed to convert images to video"}

        current_input = str(base_output)
        ai_options = request.get("ai_options", {})

        # Apply AI enhancements if requested
        if ai_options.get("superres"):
            progress.update(status='Applying super resolution')
            logger.info("Applying super resolution")
            temp_output = output_dir / "video_superres.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/super_resolution.py",
                    "--input", current_input,
                    "--output", str(temp_output),
                    "--resolution", ai_options["superres"]
                ],
                "Super resolution enhancement failed",
                self,
                checkpoint,
                "super_resolution"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Super resolution enhancement failed"}

        if ai_options.get("denoise"):
            progress.update(status='Applying denoising')
            logger.info("Applying denoising")
            temp_output = output_dir / "video_denoised.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/denoise.py",
                    "--input", current_input,
                    "--output", str(temp_output)
                ],
                "Denoising failed",
                self,
                checkpoint,
                "denoise"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Denoising failed"}

        if ai_options.get("speech2text"):
            progress.update(status='Extracting speech to text')
            logger.info("Extracting speech to text")
            success = run_command(
                [
                    "python3", "/app/processing/speech_to_text.py",
                    "--input", current_input,
                    "--output", str(output_dir / "transcription.txt")
                ],
                "Speech to text conversion failed",
                self,
                checkpoint,
                "speech_to_text"
            )
            if not success:
                return {"status": "Error", "error": "Speech to text conversion failed"}

        if ai_options.get("framerateboost"):
            progress.update(status='Boosting framerate')
            logger.info("Boosting framerate")
            temp_output = output_dir / "video_fps_boosted.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/framerate_boost.py",
                    "--input", current_input,
                    "--output", str(temp_output),
                    "--fps", str(ai_options["framerateboost"])
                ],
                "Framerate boost failed",
                self,
                checkpoint,
                "framerate_boost"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Framerate boost failed"}

        # Move the final result to the output location if it's not already there
        if current_input != str(base_output):
            os.replace(current_input, base_output)

        # Cache the result
        cache_manager.cache_result(cache_key, str(base_output))

        # Clean up task-specific temp directory and checkpoint
        shutil.rmtree(task_temp_dir, ignore_errors=True)
        checkpoint.cleanup()
        
        logger.info(f"Video processing completed successfully - Task ID: {self.request.id}")
        return {
            "status": "Complete",
            "output": str(base_output),
            "transcription": str(output_dir / "transcription.txt") if ai_options.get("speech2text") else None
        }

    except Exception as e:
        logger.error(f"Unexpected error during video processing: {str(e)}")
        if checkpoint:
            checkpoint.step_failed("processing", str(e))
        return {"status": "Error", "error": str(e)}

# Cleanup on worker shutdown
import atexit

@atexit.register
def cleanup():
    """Cleanup resources on worker shutdown"""
    resource_monitor.stop()
    resource_monitor.join()