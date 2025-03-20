import os
import json
import hashlib
from pathlib import Path

class CacheManager:
    def __init__(self, cache_dir="/app/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_cache_key(self, input_path, operation, params):
        """Generate unique cache key based on input and parameters"""
        content_hash = hashlib.md5(open(input_path, 'rb').read()).hexdigest()
        params_hash = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
        return f"{operation}_{content_hash}_{params_hash}"
        
    def get_cached_result(self, cache_key):
        """Get cached result if available"""
        cache_path = self.cache_dir / f"{cache_key}.mp4"
        if cache_path.exists():
            return str(cache_path)
        return None
        
    def cache_result(self, cache_key, result_path):
        """Cache operation result"""
        cache_path = self.cache_dir / f"{cache_key}.mp4"
        os.symlink(result_path, cache_path)