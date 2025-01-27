import logging
import psutil
import os
# from functools import wraps
from typing import Optional

class MemoryLogger:
    def __init__(self, name: str, log_file: Optional[str] = "memory_usage.log"):
        # Configure logging
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            
            # Create formatters and handlers
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [Memory: %(memory_usage).2f MB] - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            # File handler
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            
            self.logger.addHandler(console_handler)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB
    
    def info(self, message: str) -> None:
        """Log info message with memory usage"""
        self.logger.info(
            message,
            extra={'memory_usage': self._get_memory_usage()}
        )
    
    # def debug(self, message: str) -> None:
    #     """Log debug message with memory usage"""
    #     self.logger.debug(
    #         message,
    #         extra={'memory_usage': self._get_memory_usage()}
    #     )
    
    # def warning(self, message: str) -> None:
    #     """Log warning message with memory usage"""
    #     self.logger.warning(
    #         message,
    #         extra={'memory_usage': self._get_memory_usage()}
    #     )
    
    # def error(self, message: str) -> None:
    #     """Log error message with memory usage"""
    #     self.logger.error(
    #         message,
    #         extra={'memory_usage': self._get_memory_usage()}
    #    )

# def log_memory(logger: MemoryLogger):
#     """Decorator to log memory usage before and after function execution"""
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             func_name = func.__name__
#             logger.info(f"Entering {func_name}")
            
#             result = func(*args, **kwargs)
            
#             logger.info(f"Exiting {func_name}")
#             return result
#         return wrapper
#     return decorator