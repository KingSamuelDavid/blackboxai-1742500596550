from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
from tasks import image_to_video
import traceback
from config import *
import time
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} seconds"]
)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_file_size(file_path):
    """Check if file size is within limits"""
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb <= MAX_FILE_SIZE_MB

@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    })

def validate_request(data):
    """
    Validate the incoming request data.
    
    Args:
        data (dict): Request data containing input paths and processing options
    
    Returns:
        list: List of validation errors, empty if validation passes
    """
    errors = []
    
    # Check if input images are provided
    if not data.get('input') or not isinstance(data['input'], list):
        errors.append("'input' must be a non-empty list of image paths")
    else:
        # Verify each input file exists
        for img_path in data['input']:
            if not os.path.exists(img_path):
                errors.append(f"Image file not found: {img_path}")

    # Validate FPS
    fps = data.get('fps')
    if not fps or not isinstance(fps, (int, float)) or fps <= 0:
        errors.append("'fps' must be a positive number")

    # Validate AI options
    ai_options = data.get('ai_options')
    if not isinstance(ai_options, dict):
        errors.append("'ai_options' must be a dictionary")
    else:
        # Validate specific AI options if provided
        if 'superres' in ai_options and ai_options['superres']:
            if not isinstance(ai_options['superres'], str):
                errors.append("'superres' must be a string value (e.g., 'x4')")

        if 'framerateboost' in ai_options and ai_options['framerateboost']:
            if not isinstance(ai_options['framerateboost'], (int, float)) or ai_options['framerateboost'] <= 0:
                errors.append("'framerateboost' must be a positive number")

    return errors

@app.route('/image2video', methods=['POST'])
@limiter.limit(f"{RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} seconds")
def convert_images_to_video():
    """
    Convert a sequence of images to video with optional AI enhancements.
    
    Expected JSON payload:
    {
        "input": ["path/to/image1.jpg", "path/to/image2.jpg", ...],
        "fps": 24,
        "ai_options": {
            "superres": "x4",        // or false/null if not used
            "denoise": true,         // or false/null if not used
            "speech2text": true,     // or false/null if not used
            "framerateboost": 60     // or false/null if not used
        }
    }
    """
    try:
        # Log request received with request ID
        request_id = str(int(time.time() * 1000))
        logger.info(f"Received image2video conversion request - ID: {request_id}")
        
        # Validate request content type
        if not request.is_json:
            logger.error(f"Request {request_id}: Invalid content type")
            return jsonify({
                "status": "error",
                "code": "INVALID_CONTENT_TYPE",
                "message": "Content-Type must be application/json",
                "request_id": request_id
            }), 400

        # Get JSON data
        data = request.json
        logger.info(f"Request data: {data}")

        # Validate request data
        validation_errors = validate_request(data)
        if validation_errors:
            logger.error(f"Validation errors: {validation_errors}")
            return jsonify({
                "status": "error",
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": validation_errors,
                "request_id": request_id
            }), 400

        # Process the conversion
        try:
            # For initial testing, we'll use synchronous processing
            # In production, you might want to use task.delay() for async processing
            result = image_to_video(data)
            
            logger.info(f"Conversion completed successfully: {result}")
            return jsonify(result)

        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "code": "PROCESSING_ERROR",
                "message": "Processing failed",
                "details": str(e),
                "request_id": request_id
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "details": str(e),
            "request_id": request_id
        }), 500

if __name__ == '__main__':
    # Create videos directory if it doesn't exist
    os.makedirs('/app/videos', exist_ok=True)
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=8000)