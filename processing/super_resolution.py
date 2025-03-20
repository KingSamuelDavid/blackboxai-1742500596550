import argparse
import cv2
import logging
import os
import subprocess
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_frames(video_path, temp_dir):
    """
    Extract frames from video to temporary directory.
    
    Args:
        video_path (str): Path to input video
        temp_dir (str): Directory to save frames
        
    Returns:
        tuple: (success, frame_count, fps)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return False, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {frame_count} frames at {fps} FPS")
        return True, frame_count, fps

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False, 0, 0

def upscale_frames(input_dir, output_dir, resolution):
    """
    Upscale frames using RealESRGAN.
    
    Args:
        input_dir (str): Directory containing input frames
        output_dir (str): Directory to save upscaled frames
        resolution (str): Target resolution multiplier (e.g., 'x4')
        
    Returns:
        bool: True if successful
    """
    try:
        # Parse resolution multiplier
        scale = int(resolution.strip('x'))
        
        # Initialize RealESRGAN model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        
        # Download and load model weights
        model_path = load_file_from_url(
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            model_dir='weights'
        )
        
        # Initialize upscaler
        upscaler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,
            tile=0,  # Tile size (0 for no tiling)
            tile_pad=10,
            pre_pad=0,
            half=True  # Use half precision to save memory
        )

        # Process each frame
        frame_files = sorted(os.listdir(input_dir))
        for frame_file in frame_files:
            if not frame_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            input_path = os.path.join(input_dir, frame_file)
            output_path = os.path.join(output_dir, frame_file)
            
            # Read image
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            
            # Upscale
            try:
                output, _ = upscaler.enhance(img, outscale=scale)
                cv2.imwrite(output_path, output)
            except Exception as e:
                logger.error(f"Error upscaling frame {frame_file}: {str(e)}")
                continue

        return True

    except Exception as e:
        logger.error(f"Error during upscaling: {str(e)}")
        return False

def frames_to_video(frame_dir, output_path, fps):
    """
    Combine frames into a video.
    
    Args:
        frame_dir (str): Directory containing frames
        output_path (str): Path to save output video
        fps (float): Frames per second
        
    Returns:
        bool: True if successful
    """
    try:
        frame_files = sorted(os.listdir(frame_dir))
        if not frame_files:
            logger.error("No frames found")
            return False

        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
        height, width = first_frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames
        for frame_file in frame_files:
            if not frame_file.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()
        return True

    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upscale video resolution using RealESRGAN')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--resolution', default='x4', help='Resolution multiplier (e.g., x2, x4)')
    
    args = parser.parse_args()
    
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as input_temp_dir, \
             tempfile.TemporaryDirectory() as output_temp_dir:
            
            # Extract frames
            logger.info("Extracting frames from video...")
            success, frame_count, fps = extract_frames(args.input, input_temp_dir)
            if not success:
                logger.error("Frame extraction failed")
                exit(1)
            
            # Upscale frames
            logger.info("Upscaling frames...")
            if not upscale_frames(input_temp_dir, output_temp_dir, args.resolution):
                logger.error("Frame upscaling failed")
                exit(1)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Combine frames into video
            logger.info("Creating output video...")
            if not frames_to_video(output_temp_dir, args.output, fps):
                logger.error("Video creation failed")
                exit(1)
            
            logger.info("Super resolution processing completed successfully")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()