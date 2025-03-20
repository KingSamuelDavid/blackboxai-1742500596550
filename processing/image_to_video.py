import argparse
import cv2
import logging
import os
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_images(image_paths):
    """
    Validate that all images exist and can be opened.
    
    Args:
        image_paths (list): List of paths to image files
        
    Returns:
        tuple: (bool, str) - (True if all valid, error message if not)
    """
    for img_path in image_paths:
        if not os.path.exists(img_path):
            return False, f"Image not found: {img_path}"
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, f"Failed to read image: {img_path}"
        except Exception as e:
            return False, f"Error reading image {img_path}: {str(e)}"
    return True, ""

def get_output_dimensions(image_paths):
    """
    Determine the output video dimensions based on the first image.
    
    Args:
        image_paths (list): List of paths to image files
        
    Returns:
        tuple: (width, height) for the output video
    """
    first_image = cv2.imread(image_paths[0])
    height, width = first_image.shape[:2]
    return width, height

def convert_images_to_video(image_paths, output_path, fps):
    """
    Convert a sequence of images to a video file.
    
    Args:
        image_paths (list): List of paths to image files
        output_path (str): Path where the output video will be saved
        fps (float): Frames per second for the output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get dimensions from first image
        width, height = get_output_dimensions(image_paths)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error("Failed to create VideoWriter")
            return False
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
            
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
                
            # Resize if dimensions don't match
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            # Write frame
            out.write(frame)
        
        # Release resources
        out.release()
        
        # Verify the output file was created
        if not os.path.exists(output_path):
            logger.error("Output file was not created")
            return False
            
        logger.info(f"Successfully created video: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error during video creation: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert a sequence of images to video')
    parser.add_argument('--input', nargs='+', required=True, help='Input image files')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--fps', type=float, default=30.0, help='Frames per second')
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate input images
        valid, error_message = validate_images(args.input)
        if not valid:
            logger.error(error_message)
            exit(1)
        
        # Sort images to ensure consistent ordering
        sorted_images = sorted(args.input)
        
        # Convert images to video
        success = convert_images_to_video(sorted_images, args.output, args.fps)
        
        if not success:
            logger.error("Failed to convert images to video")
            exit(1)
        
        logger.info("Image to video conversion completed successfully")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()