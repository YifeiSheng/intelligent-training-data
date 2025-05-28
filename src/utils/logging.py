import os
import logging
import time
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # If log file is specified, create file handler
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger

class DataTracer:
    """Utility for tracing data operations and maintaining lineage."""
    
    def __init__(self, log_dir: str = "logs/traces"):
        """
        Initialize data tracer.
        
        Args:
            log_dir: Directory to store trace logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            "data_tracer",
            os.path.join(log_dir, f"trace_{time.strftime('%Y%m%d')}.log")
        )
    
    def log_operation(self, operation: str, data_id: str, details: Optional[dict] = None) -> None:
        """
        Log a data operation.
        
        Args:
            operation: Name of the operation
            data_id: ID of the data being operated on
            details: Additional details about the operation
        """
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "data_id": data_id,
            "details": details or {}
        }
        
        self.logger.info(f"TRACE: {log_entry}")
    
    def get_lineage(self, data_id: str) -> list:
        """
        Get lineage for a specific data item.
        
        Args:
            data_id: ID of the data item
            
        Returns:
            List of operations performed on the data item
        """
        # In a real implementation, this would query a database or parse logs
        # For this example, we'll just return a placeholder
        return [
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "operation": "get_lineage",
                "data_id": data_id,
                "message": "Lineage retrieval not implemented in this example"
            }
        ]
