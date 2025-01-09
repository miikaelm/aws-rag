import logging
import os
from datetime import datetime
from typing import Optional, Union, List
import json
from pathlib import Path
from logging.handlers import RotatingFileHandler

class Logger:
    """
    Logging utility that handles file logging with rotation
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the logger if it hasn't been initialized yet"""
        if self._initialized:
            return
            
        # Create logs directory if it doesn't exist
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Use a single log file
        self.log_file = self.logs_dir / "app.log"
        
        # Configure file logger
        self.file_logger = logging.getLogger("app")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.file_logger.handlers = []
        
        # Create file handler with rotation (10MB max size, keep 5 backups)
        file_handler = RotatingFileHandler(
            self.log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.file_logger.addHandler(file_handler)
        
        self._initialized = True

    def _format_message(self, message: Union[str, dict, list, Exception]) -> str:
        """Format message for logging"""
        if isinstance(message, (dict, list)):
            return json.dumps(message, indent=2)
        elif isinstance(message, Exception):
            return f"{type(message).__name__}: {str(message)}"
        return str(message)

    def debug(self, message: Union[str, dict, list]):
        """Log debug message"""
        formatted = self._format_message(message)
        self.file_logger.debug(formatted)

    def info(self, message: Union[str, dict, list]):
        """Log info message"""
        formatted = self._format_message(message)
        self.file_logger.info(formatted)

    def warning(self, message: Union[str, dict, list]):
        """Log warning message"""
        formatted = self._format_message(message)
        self.file_logger.warning(formatted)

    def error(self, message: Union[str, dict, list, Exception]):
        """Log error message"""
        formatted = self._format_message(message)
        self.file_logger.error(formatted)

    def get_log_contents(self, n_lines: int = None, level_filter: str = None) -> List[dict]:
        """
        Get contents of the log file
        
        Args:
            n_lines: Number of lines to return (None for all)
            level_filter: Optional level to filter by (DEBUG, INFO, WARNING, ERROR)
        """
        if not self.log_file.exists():
            return []
            
        logs = []
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                if n_lines:
                    lines = lines[-n_lines:]
                    
                for line in lines:
                    try:
                        # Parse log line
                        parts = line.strip().split(' - ', 2)
                        if len(parts) == 3:
                            timestamp, level, message = parts
                            
                            # Apply level filter if specified
                            if level_filter and level.strip() != level_filter:
                                continue
                                
                            logs.append({
                                'timestamp': timestamp,
                                'level': level.strip(),
                                'message': message
                            })
                    except Exception:
                        continue
                        
        except Exception as e:
            print(f"Error reading log file: {e}")
            
        return logs

    def clear_logfile(self):
        """Clear the contents of the log file"""
        try:
            with open(self.log_file, 'w') as f:
                f.write('')
            return True
        except Exception as e:
            print(f"Error clearing log file: {e}")
            return False