import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingCheckpoint:
    def __init__(self, task_id, output_dir):
        self.task_id = task_id
        self.checkpoint_file = Path(output_dir) / f"{task_id}_checkpoint.json"
        self.load_checkpoint()
        
    def load_checkpoint(self):
        """Load existing checkpoint if available"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                'completed_steps': [],
                'current_step': None,
                'intermediate_files': {},
                'parameters': {},
                'error_count': {},
                'last_error': None
            }
            
    def save_checkpoint(self):
        """Save current processing state"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f)
            
    def step_completed(self, step_name, output_file=None, parameters=None):
        """Mark processing step as completed"""
        self.state['completed_steps'].append(step_name)
        if output_file:
            self.state['intermediate_files'][step_name] = output_file
        if parameters:
            self.state['parameters'][step_name] = parameters
        self.save_checkpoint()
        
    def step_failed(self, step_name, error):
        """Record step failure"""
        self.state['error_count'][step_name] = self.state['error_count'].get(step_name, 0) + 1
        self.state['last_error'] = {
            'step': step_name,
            'error': str(error),
            'count': self.state['error_count'][step_name]
        }
        self.save_checkpoint()
        
    def can_resume_from(self, step_name):
        """Check if processing can resume from a specific step"""
        if step_name in self.state['completed_steps']:
            output_file = self.state['intermediate_files'].get(step_name)
            if output_file and Path(output_file).exists():
                return True
        return False
        
    def should_retry(self, step_name, max_retries=3):
        """Determine if a failed step should be retried"""
        return self.state['error_count'].get(step_name, 0) < max_retries
        
    def get_step_parameters(self, step_name):
        """Get parameters used in a previous step"""
        return self.state['parameters'].get(step_name, {})
        
    def cleanup(self):
        """Clean up checkpoint file and intermediate files"""
        try:
            # Remove intermediate files
            for file_path in self.state['intermediate_files'].values():
                if Path(file_path).exists():
                    Path(file_path).unlink()
            
            # Remove checkpoint file
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                
        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {str(e)}")