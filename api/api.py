from flask import Flask, request, jsonify
import logging
import os
from tasks import image_to_video
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def validate_request(data):
    """Validate the incoming request data."""
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
        # Log request received
        logger.info("Received image2video conversion request")
        
        # Validate request content type
        if not request.is_json:
            logger.error("Request content-type is not application/json")
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400

        # Get JSON data
        data = request.json
        logger.info(f"Request data: {data}")

        # Validate request data
        validation_errors = validate_request(data)
        if validation_errors:
            logger.error(f"Validation errors: {validation_errors}")
            return jsonify({
                "error": "Validation failed",
                "details": validation_errors
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
                "error": "Processing failed",
                "details": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    # Create videos directory if it doesn't exist
    os.makedirs('/app/videos', exist_ok=True)
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=8000)