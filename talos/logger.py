import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FILE = Path("talos.log")

# Global logger instance
logger = logging.getLogger("talos")

def setup_logger():
    """Set up the global logger."""
    if logger.hasHandlers():
        return

    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler (writes to talos.log)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler (writes to console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def add_task_log_handler(task_output_dir: str) -> Optional[logging.FileHandler]:
    """Adds a file handler for a specific task."""
    try:
        log_path = Path(task_output_dir) / "task.log"
        task_handler = logging.FileHandler(log_path, encoding='utf-8')
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        task_handler.setFormatter(formatter)
        task_handler.setLevel(logging.INFO)
        
        logger.addHandler(task_handler)
        return task_handler
    except Exception as e:
        logger.error(f"Failed to add task log handler: {e}", exc_info=True)
        return None

def remove_task_log_handler(handler: Optional[logging.FileHandler]):
    """Removes a file handler for a specific task."""
    if handler:
        try:
            handler.close()
            logger.removeHandler(handler)
        except Exception as e:
            logger.error(f"Failed to remove task log handler: {e}", exc_info=True)

# Initialize the logger when the module is imported
setup_logger()

