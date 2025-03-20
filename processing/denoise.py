import argparse
import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def denoise_video(input_path, output_path, temporal_strength=4, spatial_strength=3):
    """
    Denoise video using FFmpeg's hqdn3d filter.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to save output video
        temporal_strength (int): Temporal denoise strength (1-10)
        spatial_strength (int): Spatial denoise strength (1-10)
        
    Returns:
        bool: True if successful
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construct FFmpeg command with CUDA acceleration
        # Using both hqdn3d for high-quality denoising and nlmeans for temporal consistency
        command = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-hwaccel', 'cuda',  # Use CUDA hardware acceleration
            '-i', input_path,
            '-vf', f'hqdn3d={spatial_strength}:{spatial_strength}:{temporal_strength}:{temporal_strength},nlmeans=10:7:15:15',
            '-c:v', 'h264_nvenc',  # Use NVIDIA encoder
            '-preset', 'p7',  # Highest quality preset for NVENC
            '-rc', 'vbr',  # Variable bitrate
            '-cq', '15',  # Constant quality factor (lower = better quality)
            '-b:v', '0',  # Let the encoder choose the bitrate
            '-c:a', 'copy',  # Copy audio stream without re-encoding
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(command)}")
        
        # Execute FFmpeg command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if the command was successful
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
        # Verify output file exists and has size > 0
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("Output file is missing or empty")
            return False
            
        logger.info("Video denoising completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during video denoising: {str(e)}")
        return False

def validate_input(input_path):
    """
    Validate input video file.
    
    Args:
        input_path (str): Path to input video
        
    Returns:
        bool: True if valid
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
        
    if os.path.getsize(input_path) == 0:
        logger.error(f"Input file is empty: {input_path}")
        return False
        
    # Check if file is a valid video using FFmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # FFmpeg prints format information to stderr
        if "Invalid data found" in result.stderr:
            logger.error(f"Invalid video file: {input_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating input file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Denoise video using FFmpeg')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output video file')
    parser.add_argument('--temporal-strength', type=int, default=4, 
                        help='Temporal denoising strength (1-10)')
    parser.add_argument('--spatial-strength', type=int, default=3,
                        help='Spatial denoising strength (1-10)')
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not validate_input(args.input):
            exit(1)
            
        # Validate strength parameters
        if not (1 <= args.temporal_strength <= 10 and 1 <= args.spatial_strength <= 10):
            logger.error("Strength parameters must be between 1 and 10")
            exit(1)
            
        # Process video
        if not denoise_video(
            args.input,
            args.output,
            args.temporal_strength,
            args.spatial_strength
        ):
            logger.error("Video denoising failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()