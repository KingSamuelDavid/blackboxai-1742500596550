import psutil
import logging
import torch

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, target_usage=0.8):
        self.target_usage = target_usage
        
    def get_available_memory(self):
        """Get available system and GPU memory"""
        system_memory = psutil.virtual_memory().available
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
        return system_memory, gpu_memory
        
    def estimate_frame_size(self, frame):
        """Estimate memory needed for a frame"""
        return frame.nbytes
        
    def calculate_batch_size(self, frame_size):
        """Calculate optimal batch size based on available memory"""
        sys_mem, gpu_mem = self.get_available_memory()
        available = min(sys_mem, gpu_mem) if gpu_mem else sys_mem
        return max(1, int((available * self.target_usage) / frame_size))