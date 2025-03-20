import argparse
import logging
import os
import subprocess
import whisper
import torch
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_audio(video_path, audio_path):
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path (str): Path to input video
        audio_path (str): Path to save extracted audio
        
    Returns:
        bool: True if successful
    """
    try:
        command = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit audio
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono audio
            audio_path
        ]
        
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
        logger.error(f"Error extracting audio: {str(e)}")
        return False

def transcribe_audio(audio_path, output_path, model_size='base'):
    """
    Transcribe audio using OpenAI's Whisper model.
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Path to save transcription
        model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        
    Returns:
        bool: True if successful
    """
    try:
        # Load Whisper model
        logger.info(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Using CUDA for transcription")
        else:
            logger.info("CUDA not available, using CPU")
        
        # Transcribe audio
        logger.info("Starting transcription...")
        result = model.transcribe(
            audio_path,
            fp16=torch.cuda.is_available(),  # Use FP16 if CUDA is available
            language='en',  # Default to English
            task='transcribe'
        )
        
        # Write transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write full transcription
            f.write("Full Transcription:\n")
            f.write(result['text'])
            f.write("\n\n")
            
            # Write segments with timestamps
            f.write("Segments with Timestamps:\n")
            for segment in result['segments']:
                start_time = format_timestamp(segment['start'])
                end_time = format_timestamp(segment['end'])
                f.write(f"[{start_time} --> {end_time}] {segment['text']}\n")
        
        logger.info(f"Transcription saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        return False

def format_timestamp(seconds):
    """
    Format time in seconds to HH:MM:SS.mmm.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

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
        
    # Check if file has audio stream using FFmpeg
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if "Stream #0:1: Audio" not in result.stderr:
            logger.error(f"No audio stream found in: {input_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating input file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Transcribe video audio using Whisper')
    parser.add_argument('--input', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output transcription file')
    parser.add_argument('--model', default='base', 
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size')
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not validate_input(args.input):
            exit(1)
            
        # Create temporary directory for audio extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract audio
            audio_path = os.path.join(temp_dir, "audio.wav")
            logger.info("Extracting audio from video...")
            if not extract_audio(args.input, audio_path):
                logger.error("Audio extraction failed")
                exit(1)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Transcribe audio
            logger.info("Transcribing audio...")
            if not transcribe_audio(audio_path, args.output, args.model):
                logger.error("Transcription failed")
                exit(1)
            
            logger.info("Speech-to-text conversion completed successfully")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()