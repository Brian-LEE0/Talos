"""
Task Management System
Handles task creation, ordering, and management of independent folders.
"""

import os
import json
import time
import uuid
import shutil
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from enum import Enum

from .file_manager import get_file_manager
from .context_manager import get_context_manager
from .vertex_ai_client import get_vertex_client
from .parallel_executor import get_parallel_executor
from ..logger import logger, add_task_log_handler, remove_task_log_handler
import re
import csv as csv_module


class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Task type"""
    SINGLE = "single"          # Single execution
    PARALLEL = "parallel"      # Parallel execution
    SEQUENTIAL = "sequential"  # Sequential execution


@dataclass
class TaskStep:
    """Task step"""
    step_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class Task:
    """Task information"""
    task_id: str
    name: str
    description: str
    prompt: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    
    # Directory information
    workspace_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Execution information
    steps: List[TaskStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Dependent task IDs
    
    # Parallel execution information (CSV-based)
    csv_parameters: List[Dict[str, Any]] = field(default_factory=list)
    parallel_batch_id: Optional[str] = None
    
    # Metadata
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    priority: int = 0  # Higher priority runs first
    
    # Result
    result: Optional[str] = None
    error: Optional[str] = None
    output_files: List[str] = field(default_factory=list)


class TaskManager:
    """Task Manager"""
    
    def __init__(self, base_workspace_dir: str = "workspaces"):
        """
        Initializes the TaskManager.
        
        Args:
            base_workspace_dir: The base directory for workspaces.
        """
        self.base_workspace_dir = Path(base_workspace_dir)
        self.base_workspace_dir.mkdir(exist_ok=True)
        
        # Components
        self.file_manager = get_file_manager()
        self.context_manager = get_context_manager()
        self.ai_client = get_vertex_client()
        self.parallel_executor = get_parallel_executor()
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[str] = []  # Execution queue
        
        # State file
        self.state_file = self.base_workspace_dir / "task_manager_state.json"
        self.load_state()
    
    def create_task(
        self,
        name: str,
        prompt: str,
        description: str = "",
        task_type: TaskType = TaskType.SINGLE,
        workspace_name: Optional[str] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None
    ) -> Task:
        """
        Creates a new task.
        
        Args:
            name: The name of the task.
            prompt: The prompt to execute.
            description: A description of the task.
            task_type: The type of the task.
            workspace_name: The name of the workspace (auto-generated if None).
            priority: The priority of the task.
            dependencies: A list of dependent task IDs.
            
        Returns:
            The created task.
        """
        try:
            # Create task ID
            task_id = f"task_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            # Create workspace directory
            if workspace_name is None:
                workspace_name = f"{name}_{int(time.time())}"
            
            # Sanitize directory name
            safe_workspace_name = "".join(c for c in workspace_name if c.isalnum() or c in "._-")
            workspace_dir = self.base_workspace_dir / safe_workspace_name
            workspace_dir.mkdir(exist_ok=True)
            
            # Create output directory
            output_dir = workspace_dir / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Analyze CSV parameters (for parallel tasks)
            csv_parameters = []
            if task_type == TaskType.PARALLEL:
                csv_params = self.parallel_executor.csv_parser.parse_csv_parameters(prompt)
                if csv_params:
                    csv_parameters = [asdict(param) for param in csv_params]
            
            # Create task
            task = Task(
                task_id=task_id,
                name=name,
                description=description,
                prompt=prompt,
                task_type=task_type,
                workspace_dir=str(workspace_dir),
                output_dir=str(output_dir),
                csv_parameters=csv_parameters,
                dependencies=dependencies or [],
                priority=priority
            )
            
            # Create initial steps
            self._create_initial_steps(task)
            
            # Register task
            self.tasks[task_id] = task
            
            # Add to queue if no dependencies
            if not task.dependencies:
                self.task_queue.append(task_id)
            
            # Save state
            self.save_state()
            
            logger.info(f"Task created: {task_id} - {name}")
            return task
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            raise
    
    def _create_initial_steps(self, task: Task):
        """Creates the initial steps for a task."""
        steps = []
        
        if task.task_type == TaskType.SINGLE:
            steps = [
                TaskStep("analyze_prompt", "Analyze prompt"),
                TaskStep("setup_context", "Set up context"),
                TaskStep("execute_task", "Execute task"),
                TaskStep("save_results", "Save results")
            ]
        
        elif task.task_type == TaskType.PARALLEL:
            steps = [
                TaskStep("analyze_csv_params", "Analyze CSV parameters"),
                TaskStep("create_batch", "Create execution batch"),
                TaskStep("execute_parallel", "Execute in parallel"),
                TaskStep("collect_results", "Collect results")
            ]
        
        elif task.task_type == TaskType.SEQUENTIAL:
            steps = [
                TaskStep("parse_sequence", "Parse sequential tasks"),
                TaskStep("execute_sequence", "Execute sequence"),
                TaskStep("merge_results", "Merge results")
            ]
        
        task.steps = steps
    
    def load_tasks_from_file(self, file_path: str) -> List[Task]:
        """
        Loads task definitions from a JSON file and creates them.

        Args:
            file_path: The path to the JSON file.

        Returns:
            A list of created tasks.
        """
        try:
            logger.info(f"Loading tasks from file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
            
            created_tasks = []
            for task_data in tasks_data:
                task = self.create_task(
                    name=task_data.get("name", task_data.get("id", "Unnamed Task")),
                    prompt=task_data.get("prompt", ""),
                    description=task_data.get("description", ""),
                    task_type=TaskType(task_data.get("type", "single")),
                    priority=task_data.get("priority", 0)
                )
                created_tasks.append(task)
            
            logger.info(f"Successfully loaded {len(created_tasks)} tasks.")
            return created_tasks
        except Exception as e:
            logger.error(f"Failed to load tasks from file: {e}", exc_info=True)
            raise

    def get_task(self, task_id: str) -> Optional[Task]:
        """Gets a task by its ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, status_filter: Optional[TaskStatus] = None) -> List[Task]:
        """Lists all tasks."""
        tasks = list(self.tasks.values())
        
        if status_filter:
            tasks = [t for t in tasks if t.status == status_filter]
        
        # Sort by priority and creation time
        tasks.sort(key=lambda t: (-t.priority, t.created_time))
        
        return tasks
    
    def execute_task(self, task_id: str) -> bool:
        """Executes a task."""
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task not found for execution: {task_id}")
            return False

        task_log_handler = None
        try:
            # Add a specific log handler for this task
            if task.output_dir:
                task_log_handler = add_task_log_handler(task.output_dir)

            # Check dependencies
            if not self._check_dependencies(task):
                logger.warning(f"Dependencies not met for task: {task_id}")
                return False
            
            # Update task status
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()
            
            logger.info(f"Starting task execution: {task_id} - {task.name}")
            
            # Execute by task type
            if task.task_type == TaskType.SINGLE:
                success = self._execute_single_task(task)
            elif task.task_type == TaskType.PARALLEL:
                success = self._execute_parallel_task(task)
            elif task.task_type == TaskType.SEQUENTIAL:
                success = self._execute_sequential_task(task)
            else:
                success = False
                task.error = f"Unsupported task type: {task.task_type}"
            
            # Process result
            task.end_time = time.time()
            
            if success:
                task.status = TaskStatus.COMPLETED
                logger.info(f"Task completed: {task_id}")
                
                # Activate dependent tasks
                self._activate_dependent_tasks(task_id)
            else:
                task.status = TaskStatus.FAILED
                logger.warning(f"Task failed: {task_id}")
            
            # Save state
            self.save_state()
            
            return success
            
        except Exception as e:
            if task:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = time.time()
                logger.error(f"Error during task execution ({task_id}): {e}", exc_info=True)
                self.save_state()
            return False
        finally:
            # Remove the task-specific log handler
            if task_log_handler:
                remove_task_log_handler(task_log_handler)
    
    def _execute_single_task(self, task: Task) -> bool:
        """Executes a single task."""
        try:
            # 1. Analyze prompt
            self._update_step_status(task, "analyze_prompt", TaskStatus.RUNNING)
            
            # Process mentions
            # base_dir: workspace directory, project_root: Talos root directory
            project_root = str(self.base_workspace_dir.parent)
            successful_mentions, failed_mentions = self.context_manager.process_mentions(
                task.prompt, 
                base_dir=task.workspace_dir,
                project_root=project_root
            )
            
            self._update_step_status(task, "analyze_prompt", TaskStatus.COMPLETED)
            
            # 2. Set up context
            self._update_step_status(task, "setup_context", TaskStatus.RUNNING)
            
            context_xml = self.context_manager.generate_context_xml()
            
            self._update_step_status(task, "setup_context", TaskStatus.COMPLETED)
            
            # 3. Execute task
            self._update_step_status(task, "execute_task", TaskStatus.RUNNING)
            
            # Pass prompt to AI
            full_prompt = f"""Context:
{context_xml}

User Request:
{task.prompt}

Working Directory: {task.workspace_dir}
Output Directory: {task.output_dir}

Please perform the task based on the context and request above."""
            
            result = self.ai_client.generate_with_context(
                task.prompt,
                context_xml,
                temperature=0.0,
                workspace_dir=task.workspace_dir
            )
            
            task.result = result
            
            self._update_step_status(task, "execute_task", TaskStatus.COMPLETED)
            
            # 4. Save results
            self._update_step_status(task, "save_results", TaskStatus.RUNNING)
            
            # Extract and save CSV files from result
            cleaned_result = self._extract_and_save_csv_files(result, task.output_dir, task)
            
            # Save result file (with CSV tags removed)
            result_file = os.path.join(task.output_dir, "result.txt")
            self.file_manager.create_file(result_file, cleaned_result, overwrite=True)
            task.output_files.append(result_file)
            
            # Save context
            context_file = os.path.join(task.output_dir, "context.xml")
            self.context_manager.save_context(context_file)
            task.output_files.append(context_file)
            
            self._update_step_status(task, "save_results", TaskStatus.COMPLETED)
            
            return True
            
        except Exception as e:
            task.error = str(e)
            logger.error(f"Single task execution failed for task {task.task_id}: {e}", exc_info=True)
            return False
    
    def _execute_parallel_task(self, task: Task) -> bool:
        """Executes a parallel task."""
        try:
            # 1. Analyze CSV parameters
            self._update_step_status(task, "analyze_csv_params", TaskStatus.RUNNING)
            
            if not task.csv_parameters:
                task.error = "No CSV parameters found"
                return False
            
            self._update_step_status(task, "analyze_csv_params", TaskStatus.COMPLETED)
            
            # 2. Create execution batch
            self._update_step_status(task, "create_batch", TaskStatus.RUNNING)
            
            # The base directory for resolving file paths in the prompt should be the root of the project,
            # where files like `sample_files.csv` are located.
            # The task's workspace_dir is for outputs and intermediate files.
            project_root = str(self.base_workspace_dir.parent)
            batch, errors = self.parallel_executor.create_execution_batch(
                task.prompt,
                base_dir=project_root,
                output_base_dir=task.output_dir
            )
            
            if errors or not batch:
                task.error = f"Failed to create batch: {errors}"
                return False
            
            task.parallel_batch_id = batch.batch_id
            
            self._update_step_status(task, "create_batch", TaskStatus.COMPLETED)
            
            # 3. Execute in parallel
            self._update_step_status(task, "execute_parallel", TaskStatus.RUNNING)
            
            # Define custom executor function
            def task_executor(prompt: str, parameters: Dict[str, Any], output_dir: str) -> str:
                # Set up context
                context_xml = self.context_manager.generate_context_xml()
                
                # Execute AI
                full_prompt = f"""Context:
{context_xml}

Task Request:
{prompt}

Parameters: {parameters}
Output Directory: {output_dir}

Please perform the individual task based on the information above."""
                
                result = self.ai_client.generate_with_context(
                    prompt, 
                    context_xml,
                    workspace_dir=output_dir
                )
                
                # Save result file
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create a minimal task object for CSV extraction
                    job_task = Task(
                        task_id=f"{task.task_id}_job",
                        name="job",
                        description="",
                        prompt="",
                        task_type=TaskType.SINGLE,
                        output_dir=output_dir
                    )
                    
                    # Extract and save CSV files from result
                    cleaned_result = self._extract_and_save_csv_files(result, output_dir, job_task)
                    
                    # Update main task's output files with job's output files
                    task.output_files.extend(job_task.output_files)
                    
                    # Save cleaned result
                    result_file = os.path.join(output_dir, "result.txt")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_result)
                
                return result
            
            # Asynchronous execution (waited synchronously)
            import asyncio
            
            async def run_parallel():
                return await self.parallel_executor.execute_batch_async(
                    batch.batch_id,
                    executor_func=task_executor
                )
            
            loop = asyncio.get_event_loop()
            success, batch_results = loop.run_until_complete(run_parallel())
            
            if not success:
                failed_jobs = [job for job in batch_results if job.status == "failed"]
                error_messages = [f"Job {job.job_id}: {job.error}" for job in failed_jobs]
                task.error = f"Parallel execution failed. Errors: {'; '.join(error_messages)}"
                logger.error(f"Parallel execution failed for task {task.task_id}: {task.error}", exc_info=True)
                return False
            
            self._update_step_status(task, "execute_parallel", TaskStatus.COMPLETED)
            
            # 4. Collect results
            self._update_step_status(task, "collect_results", TaskStatus.RUNNING)
            
            batch_status = self.parallel_executor.get_batch_status(batch.batch_id)
            if batch_status:
                task.result = f"Parallel execution completed: {batch_status['completed_jobs']}/{batch_status['total_jobs']} successful"
                
                # Collect output files
                if os.path.exists(task.output_dir):
                    for job_dir in os.listdir(task.output_dir):
                        job_path = os.path.join(task.output_dir, job_dir)
                        if os.path.isdir(job_path):
                            for file in os.listdir(job_path):
                                file_path = os.path.join(job_path, file)
                                if os.path.isfile(file_path):
                                    task.output_files.append(file_path)
            
            self._update_step_status(task, "collect_results", TaskStatus.COMPLETED)
            
            return True
            
        except Exception as e:
            task.error = str(e)
            logger.error(f"Parallel task execution failed for task {task.task_id}: {e}", exc_info=True)
            return False
    
    def _execute_sequential_task(self, task: Task) -> bool:
        """Executes a sequential task (to be implemented)."""
        # Currently treated the same as a single task
        return self._execute_single_task(task)
    
    def _update_step_status(self, task: Task, step_id: str, status: TaskStatus):
        """Updates the status of a step."""
        for step in task.steps:
            if step.step_id == step_id:
                step.status = status
                if status == TaskStatus.RUNNING:
                    step.start_time = time.time()
                elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    step.end_time = time.time()
                break
    
    def _extract_and_save_csv_files(self, result: str, output_dir: str, task: Task) -> str:
        """
        Extract CSV files from AI response and save them.
        
        Parses <create_csv filename="...">...</create_csv> tags from the result,
        saves the CSV content to files, and returns the cleaned result without tags.
        
        Args:
            result: AI response text
            output_dir: Directory to save CSV files
            task: Task object to track created files
            
        Returns:
            Cleaned result text without CSV tags
        """
        # Pattern to match <create_csv filename="...">content</create_csv>
        csv_pattern = re.compile(
            r'<create_csv\s+filename=["\']([^"\']+)["\']\s*>(.*?)</create_csv>',
            re.DOTALL | re.IGNORECASE
        )
        
        matches = csv_pattern.findall(result)
        
        for filename, csv_content in matches:
            try:
                # Ensure filename has .csv extension
                if not filename.endswith('.csv'):
                    filename += '.csv'
                
                # Create full path
                csv_path = os.path.join(output_dir, filename)
                
                # Clean up the content (remove leading/trailing whitespace)
                csv_content = csv_content.strip()
                
                # Save CSV file
                self.file_manager.create_file(csv_path, csv_content, overwrite=True)
                task.output_files.append(csv_path)
                
                logger.info(f"CSV file created: {csv_path}")
                
            except Exception as e:
                logger.error(f"Failed to create CSV file {filename}: {e}", exc_info=True)
        
        # Remove CSV tags from result
        cleaned_result = csv_pattern.sub('', result)
        
        return cleaned_result.strip()
    
    def _check_dependencies(self, task: Task) -> bool:
        """Checks task dependencies."""
        for dep_task_id in task.dependencies:
            dep_task = self.tasks.get(dep_task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _activate_dependent_tasks(self, completed_task_id: str):
        """Activates tasks that depend on the completed task."""
        for task in self.tasks.values():
            if (completed_task_id in task.dependencies and 
                task.status == TaskStatus.PENDING and
                self._check_dependencies(task)):
                
                if task.task_id not in self.task_queue:
                    self.task_queue.append(task.task_id)
    
    def execute_next_task(self) -> bool:
        """Executes the next task in the queue."""
        if not self.task_queue:
            return False
        
        # Select the task with the highest priority
        self.task_queue.sort(key=lambda tid: -self.tasks[tid].priority)
        
        task_id = self.task_queue.pop(0)
        return self.execute_task(task_id)
    
    def execute_all_pending_tasks(self) -> Dict[str, bool]:
        """Executes all pending tasks."""
        results = {}
        
        while self.task_queue:
            task_id = self.task_queue[0]
            success = self.execute_next_task()
            results[task_id] = success
            
            if not success:
                # Remove failed task from the queue
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancels a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            
            # Remove from queue
            if task_id in self.task_queue:
                self.task_queue.remove(task_id)
            
            self.save_state()
            logger.info(f"Task cancelled: {task_id}")
            return True
        
        return False
    
    def delete_task(self, task_id: str, delete_workspace: bool = False) -> bool:
        """Deletes a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Delete workspace
        if delete_workspace and task.workspace_dir and os.path.exists(task.workspace_dir):
            try:
                shutil.rmtree(task.workspace_dir)
            except Exception as e:
                logger.warning(f"Failed to delete workspace ({task.workspace_dir}): {e}")
        
        # Remove task
        del self.tasks[task_id]
        
        # Remove from queue
        if task_id in self.task_queue:
            self.task_queue.remove(task_id)
        
        self.save_state()
        logger.info(f"Task deleted: {task_id}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Returns the status information of a task."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "description": task.description,
            "task_type": task.task_type.value,
            "status": task.status.value,
            "workspace_dir": task.workspace_dir,
            "output_dir": task.output_dir,
            "created_time": task.created_time,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "duration": (task.end_time or time.time()) - (task.start_time or time.time()) if task.start_time else None,
            "priority": task.priority,
            "dependencies": task.dependencies,
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "status": step.status.value,
                    "start_time": step.start_time,
                    "end_time": step.end_time,
                    "duration": (step.end_time or time.time()) - (step.start_time or time.time()) if step.start_time else None
                }
                for step in task.steps
            ],
            "output_files": task.output_files,
            "result": task.result,
            "error": task.error
        }
    
    def save_state(self):
        """Saves the state to a file."""
        try:
            state_data = {
                "tasks": {
                    task_id: {
                        **asdict(task),
                        "status": task.status.value,
                        "task_type": task.task_type.value,
                        "steps": [
                            {
                                **asdict(step),
                                "status": step.status.value
                            }
                            for step in task.steps
                        ]
                    }
                    for task_id, task in self.tasks.items()
                },
                "task_queue": self.task_queue
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
    
    def load_state(self):
        """Loads the state from a file."""
        try:
            if not self.state_file.exists():
                return
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Restore tasks
            for task_id, task_data in state_data.get("tasks", {}).items():
                # Restore steps
                steps = []
                for step_data in task_data.get("steps", []):
                    step = TaskStep(**{
                        k: v for k, v in step_data.items() if k != "status"
                    })
                    step.status = TaskStatus(step_data["status"])
                    steps.append(step)
                
                # Prepare task arguments
                task_args = {
                    k: v for k, v in task_data.items() 
                    if k not in ["status", "steps"]
                }
                task_args["task_type"] = TaskType(task_args["task_type"])

                # Restore task
                task = Task(**task_args)
                task.status = TaskStatus(task_data["status"])
                task.steps = steps
                
                self.tasks[task_id] = task
            
            # Restore queue
            self.task_queue = state_data.get("task_queue", [])
            
            logger.info(f"State loaded successfully: {len(self.tasks)} tasks")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)


# Global TaskManager instance
_task_manager: Optional[TaskManager] = None


def get_task_manager(base_workspace_dir: str = "workspaces") -> TaskManager:
    """Returns the global TaskManager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(base_workspace_dir)
    return _task_manager