import psutil
import torch
import logging
from threading import Thread
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class ResourceMonitor(Thread):
    def __init__(self, interval=5, threshold_file="/app/config/thresholds.json"):
        super().__init__()
        self.interval = interval
        self.running = True
        self.thresholds = {
            'cpu_percent': 90,
            'memory_percent': 90,
            'gpu_util_percent': 90,
            'gpu_memory_percent': 90,
            'disk_percent': 90
        }
        self.history = []
        self.max_history = 100
        
    def get_gpu_stats(self):
        """Get GPU statistics if available"""
        if torch.cuda.is_available():
            try:
                gpu_util = torch.cuda.utilization()
                total_mem = torch.cuda.get_device_properties(0).total_memory
                used_mem = torch.cuda.memory_allocated()
                return {
                    'gpu_util': gpu_util,
                    'gpu_mem_total': total_mem,
                    'gpu_mem_used': used_mem,
                    'gpu_mem_percent': (used_mem / total_mem) * 100
                }
            except Exception as e:
                logger.error(f"Error getting GPU stats: {str(e)}")
                return None
        return None
        
    def get_disk_stats(self):
        """Get disk usage statistics"""
        try:
            disk = psutil.disk_usage('/')
            return {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
        except Exception as e:
            logger.error(f"Error getting disk stats: {str(e)}")
            return None
            
    def check_thresholds(self, stats):
        """Check if any resources exceed thresholds"""
        warnings = []
        
        if stats['cpu_percent'] > self.thresholds['cpu_percent']:
            warnings.append(f"CPU usage critical: {stats['cpu_percent']}%")
            
        if stats['memory_percent'] > self.thresholds['memory_percent']:
            warnings.append(f"Memory usage critical: {stats['memory_percent']}%")
            
        if 'gpu_util' in stats and stats['gpu_util'] > self.thresholds['gpu_util_percent']:
            warnings.append(f"GPU utilization critical: {stats['gpu_util']}%")
            
        if 'gpu_mem_percent' in stats and stats['gpu_mem_percent'] > self.thresholds['gpu_memory_percent']:
            warnings.append(f"GPU memory usage critical: {stats['gpu_mem_percent']}%")
            
        if 'disk_percent' in stats and stats['disk_percent'] > self.thresholds['disk_percent']:
            warnings.append(f"Disk usage critical: {stats['disk_percent']}%")
            
        return warnings
        
    def update_history(self, stats):
        """Update resource usage history"""
        self.history.append({
            'timestamp': time.time(),
            'stats': stats
        })
        
        # Keep history size limited
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
    def get_average_usage(self, minutes=5):
        """Get average resource usage over specified time period"""
        now = time.time()
        relevant_history = [
            h for h in self.history 
            if now - h['timestamp'] <= minutes * 60
        ]
        
        if not relevant_history:
            return None
            
        avg_stats = {}
        for key in relevant_history[0]['stats'].keys():
            if isinstance(relevant_history[0]['stats'][key], (int, float)):
                values = [h['stats'][key] for h in relevant_history]
                avg_stats[key] = sum(values) / len(values)
                
        return avg_stats
        
    def run(self):
        """Monitor system resources"""
        while self.running:
            try:
                # Collect current stats
                stats = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'timestamp': time.time()
                }
                
                # Add GPU stats if available
                gpu_stats = self.get_gpu_stats()
                if gpu_stats:
                    stats.update(gpu_stats)
                    
                # Add disk stats
                disk_stats = self.get_disk_stats()
                if disk_stats:
                    stats.update(disk_stats)
                    
                # Update history
                self.update_history(stats)
                
                # Check thresholds
                warnings = self.check_thresholds(stats)
                if warnings:
                    for warning in warnings:
                        logger.warning(warning)
                        
                # Log current usage
                logger.info(f"Resource Usage - CPU: {stats['cpu_percent']}%, "
                          f"Memory: {stats['memory_percent']}%, "
                          f"Disk: {stats.get('disk_percent', 'N/A')}%")
                if 'gpu_util' in stats:
                    logger.info(f"GPU Usage - Utilization: {stats['gpu_util']}%, "
                              f"Memory: {stats['gpu_mem_percent']:.2f}%")
                              
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                
            time.sleep(self.interval)
            
    def stop(self):
        """Stop monitoring"""
        self.running = False
        
    def get_current_stats(self):
        """Get most recent resource statistics"""
        return self.history[-1] if self.history else None