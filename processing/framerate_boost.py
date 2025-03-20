import argparse
import logging
import os
import subprocess
import torch
from pathlib import Path
import tempfile
import cv2
import numpy as np
from inference_rife import Model
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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
        tuple: (success, frame_count, original_fps)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return False, 0, 0

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {frame_count} frames at {original_fps} FPS")
        return True, frame_count, original_fps

    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
        return False, 0, 0

def interpolate_frames(input_dir, output_dir, target_fps, original_fps):
    """
    Interpolate frames using RIFE to achieve target FPS.
    
    Args:
        input_dir (str): Directory containing input frames
        output_dir (str): Directory to save interpolated frames
        target_fps (float): Desired output FPS
        original_fps (float): Original video FPS
        
    Returns:
        bool: True if successful
    """
    try:
        # Initialize RIFE model
        model = Model()
        model.load_model("train_log")
        model.eval()
        model.device()

        # Calculate number of frames to interpolate
        multiplier = target_fps / original_fps
        
        # Process frames
        frame_files = sorted(os.listdir(input_dir))
        frame_count = len(frame_files)
        
        for i in range(frame_count - 1):
            # Read consecutive frames
            img0 = cv2.imread(os.path.join(input_dir, frame_files[i]))
            img1 = cv2.imread(os.path.join(input_dir, frame_files[i + 1]))
            
            # Convert to RGB
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            
            # Calculate number of intermediate frames
            n_frames = int(multiplier - 1)
            
            # Save original frame
            cv2.imwrite(
                os.path.join(output_dir, f"frame_{i*multiplier:06d}.png"),
                cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            )
            
            # Generate and save intermediate frames
            for j in range(n_frames):
                progress = (j + 1) / (n_frames + 1)
                middle = model.inference(img0, img1, progress)
                middle = cv2.cvtColor(middle, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{i*multiplier+j+1:06d}.png"),
                    middle
                )
        
        # Save last frame
        cv2.imwrite(
            os.path.join(output_dir, f"frame_{(frame_count-1)*multiplier:06d}.png"),
            cv2.imread(os.path.join(input_dir, frame_files[-1]))
        )
        
        return True

    except Exception as e:
        logger.error(f"Error during frame interpolation: {str(e)}")
        return False

def frames_to_video(frame_dir, output_path, fps):
    """
    Combine frames into a video using FFmpeg with NVIDIA hardware acceleration.
    
    Args:
        frame_dir (str): Directory containing frames
        output_path (str): Path to save output video
        fps (float): Target frames per second
        
    Returns:
        bool: True if successful
    """
    try:
        # Construct FFmpeg command with NVIDIA acceleration
        command = [
            'ffmpeg',
            '-y',
            '-hwaccel', 'cuda',
            '-framerate', str(fps),
            '-pattern_type', 'sequence',
            '-i', os.path.join(frame_dir, 'frame_%06d.png'),
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-rc', 'vbr',
            '-cq', '15',
            '-b:v', '0',
            output_path
        ]
        
        # Execute FFmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
        return True

    except Exception as e:
        logger.error(f"Error creating video: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Boost video framerate using RIFE')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--fps', type=float, required=True, help='Target FPS')
    
    args = parser.parse_args()
    
    try:
        # Create temporary directories
        with tempfile.TemporaryDirectory() as input_temp_dir, \
             tempfile.TemporaryDirectory() as output_temp_dir:
            
            # Extract frames
            logger.info("Extracting frames from video...")
            success, frame_count, original_fps = extract_frames(args.input, input_temp_dir)
            if not success:
                logger.error("Frame extraction failed")
                exit(1)
            
            # Validate target FPS
            if args.fps <= original_fps:
                logger.error(f"Target FPS ({args.fps}) must be higher than original FPS ({original_fps})")
                exit(1)
            
            # Interpolate frames
            logger.info("Interpolating frames...")
            if not interpolate_frames(input_temp_dir, output_temp_dir, args.fps, original_fps):
                logger.error("Frame interpolation failed")
                exit(1)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Combine frames into video
            logger.info("Creating output video...")
            if not frames_to_video(output_temp_dir, args.output, args.fps):
                logger.error("Video creation failed")
                exit(1)
            
            logger.info("Framerate boost completed successfully")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()