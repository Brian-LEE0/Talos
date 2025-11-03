"""
Talos - AI-powered file management and task execution system

Key Features:
- Prompt input with mention functionality
- File CRUD operations (read/create/modify/delete)
- Parallel execution using CSV parameters
- Task management with independent folders
- Streamlit UI support
- CLI interface
"""

__version__ = "0.1.0"
__author__ = "GitHub Copilot"
__description__ = "AI-powered file management and task execution system"

# Convenience imports
from .core.task_manager import get_task_manager, TaskType, TaskStatus
from .core.context_manager import get_context_manager
from .core.file_manager import get_file_manager
from .core.vertex_ai_client import get_vertex_client
from .core.parallel_executor import get_parallel_executor

__all__ = [
    '__version__',
    '__author__',
    '__description__',
    'get_task_manager',
    'get_context_manager', 
    'get_file_manager',
    'get_ai_client',
    'get_parallel_executor',
    'TaskType',
    'TaskStatus'
]