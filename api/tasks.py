from celery import Celery
import os
import logging
import subprocess
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Celery
app = Celery('tasks', 
             broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
             backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'))

def run_command(command, error_message):
    """
    Execute a command and handle its output.
    
    Args:
        command (list): Command to execute as a list of arguments
        error_message (str): Error message to log if the command fails
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{error_message}: {str(e)}")
        logger.error(f"Command stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during command execution: {str(e)}")
        return False

@app.task(name="tasks.image_to_video")
def image_to_video(request):
    """
    Convert images to video and apply optional AI enhancements.
    
    Args:
        request (dict): Contains:
            - input: list of image paths
            - fps: desired frames per second
            - ai_options: dict of AI enhancement options
    
    Returns:
        dict: Status and output information
    """
    try:
        logger.info("Starting image to video conversion process")
        
        # Ensure output directory exists
        output_dir = Path("/app/videos")
        output_dir.mkdir(exist_ok=True)
        
        # Base output path
        base_output = output_dir / "video_output.mp4"
        current_input = " ".join(request["input"])
        
        # Initial conversion from images to video
        logger.info("Converting images to video")
        success = run_command(
            [
                "python3", "/app/processing/image_to_video.py",
                "--input", *request["input"],
                "--fps", str(request["fps"]),
                "--output", str(base_output)
            ],
            "Failed to convert images to video"
        )
        
        if not success:
            return {"status": "Error", "error": "Failed to convert images to video"}

        current_input = str(base_output)
        ai_options = request.get("ai_options", {})

        # Apply AI enhancements if requested
        if ai_options.get("superres"):
            logger.info("Applying super resolution")
            temp_output = output_dir / "video_superres.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/super_resolution.py",
                    "--input", current_input,
                    "--output", str(temp_output),
                    "--resolution", ai_options["superres"]
                ],
                "Super resolution enhancement failed"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Super resolution enhancement failed"}

        if ai_options.get("denoise"):
            logger.info("Applying denoising")
            temp_output = output_dir / "video_denoised.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/denoise.py",
                    "--input", current_input,
                    "--output", str(temp_output)
                ],
                "Denoising failed"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Denoising failed"}

        if ai_options.get("speech2text"):
            logger.info("Extracting speech to text")
            success = run_command(
                [
                    "python3", "/app/processing/speech_to_text.py",
                    "--input", current_input,
                    "--output", str(output_dir / "transcription.txt")
                ],
                "Speech to text conversion failed"
            )
            if not success:
                return {"status": "Error", "error": "Speech to text conversion failed"}

        if ai_options.get("framerateboost"):
            logger.info("Boosting framerate")
            temp_output = output_dir / "video_fps_boosted.mp4"
            success = run_command(
                [
                    "python3", "/app/processing/framerate_boost.py",
                    "--input", current_input,
                    "--output", str(temp_output),
                    "--fps", str(ai_options["framerateboost"])
                ],
                "Framerate boost failed"
            )
            if success:
                current_input = str(temp_output)
            else:
                return {"status": "Error", "error": "Framerate boost failed"}

        # Move the final result to the output location if it's not already there
        if current_input != str(base_output):
            os.replace(current_input, base_output)

        logger.info("Video processing completed successfully")
        return {
            "status": "Complete",
            "output": str(base_output),
            "transcription": str(output_dir / "transcription.txt") if ai_options.get("speech2text") else None
        }

    except Exception as e:
        logger.error(f"Unexpected error during video processing: {str(e)}")
        return {"status": "Error", "error": str(e)}