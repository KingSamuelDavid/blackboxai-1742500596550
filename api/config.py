import os

# API Configuration
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', 100))
RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # 1 hour
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Processing Configuration
TASK_TIMEOUT = int(os.getenv('TASK_TIMEOUT', 3600))  # 1 hour
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
TEMP_FILE_TTL = int(os.getenv('TEMP_FILE_TTL', 3600))  # 1 hour

# Paths
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/app/videos')
TEMP_DIR = os.getenv('TEMP_DIR', '/app/temp')
LOG_DIR = os.getenv('LOG_DIR', '/app/logs')

# Create required directories
for directory in [OUTPUT_DIR, TEMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)