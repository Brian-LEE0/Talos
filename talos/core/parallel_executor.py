"""
CSV-based Parallel Execution Module
Parses <<{csv_filename}.{csv_field_name}>> parameters and executes in parallel
"""

import re
import csv
import asyncio
import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas as pd
from .file_manager import get_file_manager
from ..logger import logger


@dataclass
class CSVParameter:
    """CSV parameter information"""
    csv_file: str
    field_name: str
    original_text: str
    resolved_path: Optional[str] = None


@dataclass
class ExecutionJob:
    """Execution job information"""
    job_id: str
    prompt: str
    parameters: Dict[str, Any]
    row_index: int
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output_dir: Optional[str] = None


@dataclass
class ExecutionBatch:
    """Execution batch information"""
    batch_id: str
    original_prompt: str
    csv_parameters: List[CSVParameter]
    jobs: List[ExecutionJob]
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class CSVParameterParser:
    """CSV parameter parser"""
    
    def __init__(self):
        """Initialize parser"""
        self.file_manager = get_file_manager()
        self.logger = logger
        
        # CSV parameter pattern: <<filename.fieldname>>
        self.parameter_pattern = re.compile(r'<<([^>]+)>>')
        
        # Cached CSV data
        self.csv_cache: Dict[str, pd.DataFrame] = {}
    
    def parse_csv_parameters(self, text: str) -> List[CSVParameter]:
        """
        Extract CSV parameters from text.
        
        Args:
            text: Text to analyze.
            
        Returns:
            A list of CSV parameters.
        """
        parameters = []
        
        matches = self.parameter_pattern.finditer(text)
        for match in matches:
            param_text = match.group(1).strip()
            
            # Parse filename.fieldname format
            if '.' in param_text:
                parts = param_text.rsplit('.', 1)  # Split by the last dot
                csv_file = parts[0]
                field_name = parts[1]
                
                parameters.append(CSVParameter(
                    csv_file=csv_file,
                    field_name=field_name,
                    original_text=match.group(0)
                ))
        
        return parameters
    
    def resolve_csv_file_path(self, csv_file: str, base_dir: str = ".") -> Optional[str]:
        """Resolve CSV file path."""
        try:
            # Add .csv extension if not present
            if not csv_file.endswith('.csv'):
                csv_file += '.csv'
            
            # If absolute path
            if os.path.isabs(csv_file):
                return csv_file if os.path.exists(csv_file) else None
            
            # Handle relative path
            full_path = os.path.join(base_dir, csv_file)
            full_path = os.path.normpath(full_path)
            
            return full_path if os.path.exists(full_path) else None
            
        except Exception:
            return None
    
    def load_csv_data(self, csv_file_path: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """Load CSV file (with caching support)."""
        try:
            abs_path = os.path.abspath(csv_file_path)
            
            # Check cache
            if not force_reload and abs_path in self.csv_cache:
                return self.csv_cache[abs_path]
            
            # Read CSV file
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            # Save to cache
            self.csv_cache[abs_path] = df
            
            self.logger.info(f"CSV file loaded: {csv_file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV file ({csv_file_path}): {e}", exc_info=True)
            return None
    
    def validate_csv_parameters(self, parameters: List[CSVParameter], base_dir: str = ".") -> Tuple[List[CSVParameter], List[str]]:
        """
        Validate CSV parameters.
        
        Args:
            parameters: List of parameters to validate.
            base_dir: Base directory.
            
        Returns:
            A tuple containing a list of valid parameters and a list of error messages.
        """
        valid_parameters = []
        errors = []
        
        for param in parameters:
            # Resolve file path
            resolved_path = self.resolve_csv_file_path(param.csv_file, base_dir)
            
            if not resolved_path:
                errors.append(f"CSV file not found: {param.csv_file}")
                continue
            
            # Load CSV data
            df = self.load_csv_data(resolved_path)
            if df is None:
                errors.append(f"Failed to load CSV file: {resolved_path}")
                continue
            
            # Check field name
            if param.field_name not in df.columns:
                errors.append(f"Field not found: {param.field_name} in {param.csv_file}")
                continue
            
            # Add as a valid parameter
            param.resolved_path = resolved_path
            valid_parameters.append(param)
        
        return valid_parameters, errors
    
    def get_csv_field_values(self, param: CSVParameter) -> List[Any]:
        """Return field values for a CSV parameter."""
        if not param.resolved_path:
            return []
        
        df = self.load_csv_data(param.resolved_path)
        if df is None:
            return []
        
        if param.field_name not in df.columns:
            return []
        
        return df[param.field_name].tolist()
    
    def get_csv_row_data(self, param: CSVParameter, row_index: int) -> Optional[Dict[str, Any]]:
        """Return all data for a specific row."""
        if not param.resolved_path:
            return None
        
        df = self.load_csv_data(param.resolved_path)
        if df is None or row_index >= len(df):
            return None
        
        return df.iloc[row_index].to_dict()


class ParallelExecutor:
    """Parallel executor"""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of workers.
        """
        self.max_workers = max_workers
        self.csv_parser = CSVParameterParser()
        self.file_manager = get_file_manager()
        
        # Execution batches
        self.batches: Dict[str, ExecutionBatch] = {}
        
        # Logging setup
        self.logger = logger
    
    def create_execution_batch(
        self, 
        prompt: str, 
        base_dir: str = ".",
        output_base_dir: str = "outputs"
    ) -> Tuple[Optional[ExecutionBatch], List[str]]:
        """
        Create an execution batch.
        
        Args:
            prompt: Original prompt.
            base_dir: Base directory for CSV files.
            output_base_dir: Base directory for outputs.
            
        Returns:
            A tuple containing the execution batch and a list of error messages.
        """
        try:
            # Parse CSV parameters
            csv_parameters = self.csv_parser.parse_csv_parameters(prompt)
            
            if not csv_parameters:
                return None, ["No CSV parameters found."]
            
            # Validate parameters
            valid_params, errors = self.csv_parser.validate_csv_parameters(csv_parameters, base_dir)
            
            if errors:
                return None, errors
            
            if not valid_params:
                return None, ["No valid CSV parameters found."]
            
            # Create jobs based on the number of rows in the first CSV file
            primary_param = valid_params[0]
            df = self.csv_parser.load_csv_data(primary_param.resolved_path)
            
            if df is None or len(df) == 0:
                return None, ["CSV file is empty or could not be loaded."]
            
            # Create batch ID
            batch_id = f"batch_{int(time.time())}"
            
            # Create jobs
            jobs = []
            for row_index in range(len(df)):
                # Collect parameter values for each row
                row_parameters = {}
                
                for param in valid_params:
                    param_df = self.csv_parser.load_csv_data(param.resolved_path)
                    if param_df is not None and row_index < len(param_df):
                        if param.field_name in param_df.columns:
                            row_parameters[param.original_text] = param_df.iloc[row_index][param.field_name]
                
                # Set output directory
                output_dir = None
                if output_base_dir:
                    # Create a unique directory for each row (e.g., outputs/job_0, outputs/job_1)
                    output_dir = os.path.join(output_base_dir, f"job_{row_index}")
                
                # Create job
                job = ExecutionJob(
                    job_id=f"{batch_id}_job_{row_index}",
                    prompt=self._substitute_parameters(prompt, row_parameters),
                    parameters=row_parameters,
                    row_index=row_index,
                    output_dir=output_dir
                )
                
                jobs.append(job)
            
            # Create execution batch
            batch = ExecutionBatch(
                batch_id=batch_id,
                original_prompt=prompt,
                csv_parameters=valid_params,
                jobs=jobs,
                total_jobs=len(jobs)
            )
            
            self.batches[batch_id] = batch
            
            self.logger.info(f"Execution batch created: {batch_id} ({len(jobs)} jobs)")
            return batch, []
            
        except Exception as e:
            self.logger.error(f"Failed to create batch: {str(e)}", exc_info=True)
            return None, [f"Failed to create batch: {str(e)}"]
    
    def _substitute_parameters(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Substitute parameters in the prompt with actual values."""
        result = prompt
        
        for param_text, value in parameters.items():
            result = result.replace(param_text, str(value))
        
        return result
    
    async def execute_batch_async(
        self, 
        batch_id: str, 
        executor_func: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[bool, List[ExecutionJob]]:
        """
        Execute a batch asynchronously.
        
        Args:
            batch_id: Batch ID.
            executor_func: Execution function (takes prompt and returns result).
            progress_callback: Progress callback function.
            
        Returns:
            A tuple containing:
            - bool: Whether execution was successful overall.
            - List[ExecutionJob]: A list of all job objects with their final status.
        """
        try:
            batch = self.batches.get(batch_id)
            if not batch:
                return False, []
            
            batch.start_time = time.time()
            
            # Default execution function
            if executor_func is None:
                executor_func = self._default_executor
            
            # Create output directories
            for job in batch.jobs:
                if job.output_dir:
                    os.makedirs(job.output_dir, exist_ok=True)
            
            # Parallel execution using ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Start all jobs asynchronously
                futures = [
                    loop.run_in_executor(executor, self._execute_single_job, job, executor_func)
                    for job in batch.jobs
                ]
                
                # Wait for all jobs to complete
                await asyncio.gather(*futures)

            # Process results
            for job in batch.jobs:
                if job.status == "completed":
                    batch.completed_jobs += 1
                else:
                    batch.failed_jobs += 1
                    
                # Call progress callback
                if progress_callback:
                    progress = (batch.completed_jobs + batch.failed_jobs) / batch.total_jobs * 100
                    progress_callback(batch_id, job.job_id, job.status, progress)

            batch.end_time = time.time()
            
            success_rate = batch.completed_jobs / batch.total_jobs * 100 if batch.total_jobs > 0 else 0
            self.logger.info(f"Batch execution completed: {batch_id} (Success rate: {success_rate:.1f}%)")
            
            all_successful = batch.failed_jobs == 0
            return all_successful, batch.jobs
            
        except Exception as e:
            self.logger.error(f"Batch execution failed ({batch_id}): {e}", exc_info=True)
            if 'batch' in locals() and batch:
                for job in batch.jobs:
                    if job.status == "pending" or job.status == "running":
                        job.status = "failed"
                        job.error = f"Batch-level failure: {str(e)}"
                return False, batch.jobs
            return False, []
    
    def _execute_single_job(self, job: ExecutionJob, executor_func: Callable):
        """Execute a single job."""
        try:
            job.status = "running"
            job.start_time = time.time()
            
            # Execute job
            result = executor_func(job.prompt, job.parameters, job.output_dir)
            
            job.result = result
            job.status = "completed"
            
        except Exception as e:
            job.error = str(e)
            job.status = "failed"
            self.logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
        finally:
            job.end_time = time.time()
    
    def _default_executor(self, prompt: str, parameters: Dict[str, Any], output_dir: str) -> str:
        """Default execution function (Mock)."""
        try:
            # Simulation delay
            time.sleep(0.5)
            
            # Generate result
            result = f"Job completed:\nPrompt: {prompt[:100]}...\nParameters: {parameters}\nOutput directory: {output_dir}"
            
            # Create output file
            if output_dir:
                result_file = os.path.join(output_dir, "result.txt")
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(result)
            
            return result
            
        except Exception as e:
            return f"Execution failed: {str(e)}"
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Return batch status information."""
        batch = self.batches.get(batch_id)
        if not batch:
            return None
        
        return {
            "batch_id": batch_id,
            "total_jobs": batch.total_jobs,
            "completed_jobs": batch.completed_jobs,
            "failed_jobs": batch.failed_jobs,
            "pending_jobs": batch.total_jobs - batch.completed_jobs - batch.failed_jobs,
            "progress": (batch.completed_jobs + batch.failed_jobs) / batch.total_jobs * 100,
            "success_rate": batch.completed_jobs / max(1, batch.completed_jobs + batch.failed_jobs) * 100,
            "start_time": batch.start_time,
            "end_time": batch.end_time,
            "duration": (batch.end_time or time.time()) - (batch.start_time or time.time()) if batch.start_time else None,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "row_index": job.row_index,
                    "output_dir": job.output_dir,
                    "error": job.error
                }
                for job in batch.jobs
            ]
        }
    
    def get_job_result(self, batch_id: str, job_id: str) -> Optional[str]:
        """Return the result of a specific job."""
        batch = self.batches.get(batch_id)
        if not batch:
            return None
        
        for job in batch.jobs:
            if job.job_id == job_id:
                return job.result
        
        return None
    
    def clear_batch(self, batch_id: str) -> bool:
        """Delete a batch."""
        if batch_id in self.batches:
            del self.batches[batch_id]
            return True
        return False


# Global parallel executor instance
_parallel_executor: Optional[ParallelExecutor] = None


def get_parallel_executor(max_workers: int = 4) -> ParallelExecutor:
    """Return the global parallel executor."""
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelExecutor(max_workers)
    return _parallel_executor