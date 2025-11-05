"""
Vertex AI Client
Google Cloud Vertex AI API integration for LLM communication
"""

import json
import os
from typing import Optional, Dict, Any, List
from google.cloud import aiplatform
from google.auth import default
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part
from pathlib import Path
from ..i18n import t
from .file_manager import get_file_manager
from ..logger import logger
import re


class VertexAIClient:
    """Vertex AI client class for AI model interactions"""
    
    # Class-level cache for available models
    _model_cache = None
    _model_cache_timestamp = None
    _model_cache_ttl = 3600  # Cache for 1 hour
    
    def __init__(self, credentials_path: str = "vertex-ai-credentials.json"):
        """
        Initialize Vertex AI client
        
        Args:
            credentials_path: Path to service account key file
        """
        self.credentials_path = Path(credentials_path)
        self.project_id = None
        self.location = "us-central1"  # Default location
        self.model_name = "gemini-2.5-flash-lite"  # Updated to valid model version
        self._client = None
        self._model = None
        self.file_manager = get_file_manager()
        self.credentials = None
        
        self._initialize()
    
    def update_credentials(self, credentials_content: str, save_path: Optional[str] = None) -> bool:
        """
        Update Vertex AI credentials from JSON content
        
        Args:
            credentials_content: JSON string of service account credentials
            save_path: Optional path to save credentials file (defaults to current credentials_path)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            # Validate JSON
            creds_data = json.loads(credentials_content)
            
            # Validate required fields
            if 'project_id' not in creds_data:
                raise ValueError("project_id not found in credentials")
            if 'type' not in creds_data or creds_data['type'] != 'service_account':
                raise ValueError("Invalid credential type. Must be service_account")
            
            # Save to file
            target_path = Path(save_path) if save_path else self.credentials_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(creds_data, f, indent=2)
            
            # Update internal path and reinitialize
            self.credentials_path = target_path
            self._initialize()
            
            logger.info(f"Credentials updated successfully: {target_path}")
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to update credentials: {e}")
            return False
    
    def get_credentials_info(self) -> Dict[str, Any]:
        """
        Get current credentials information
        
        Returns:
            dict: Credentials info (project_id, location, model_name, credentials_path)
        """
        return {
            'project_id': self.project_id,
            'location': self.location,
            'model_name': self.model_name,
            'credentials_path': str(self.credentials_path),
            'credentials_exists': self.credentials_path.exists()
        }
    
    def update_default_settings(self, location: Optional[str] = None, model_name: Optional[str] = None) -> bool:
        """
        Update default location and model settings
        
        Args:
            location: New default location
            model_name: New default model name
            
        Returns:
            bool: True if update successful
        """
        try:
            if location:
                self.location = location
            if model_name:
                self.model_name = model_name
            
            # Reinitialize with new settings
            self._initialize()
            
            logger.info(f"Default settings updated: location={self.location}, model={self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update default settings: {e}")
            return False
    
    def _initialize(self):
        """Initialize the client"""
        try:
            # Load service account key file
            if not self.credentials_path.exists():
                raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
            
            with open(self.credentials_path, 'r', encoding='utf-8') as f:
                creds_data = json.load(f)
            
            self.project_id = creds_data.get('project_id')
            if not self.project_id:
                raise ValueError("project_id not found in credentials file")
            
            # Create service account credentials and store it
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=self.credentials
            )
            
            # Initialize generative model
            self._model = GenerativeModel(self.model_name)
            
            print(t('core.ai_client.init_success', project=self.project_id, model=self.model_name))
            
        except Exception as e:
            print(t('core.ai_client.init_failed', error=str(e)))
            raise
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 32000,
        top_p: float = 1.0,
        system_instruction: Optional[str] = None,
        model_name: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """
        Generate text using AI model
        
        Args:
            prompt: Input prompt
            temperature: Creativity control (0.0-1.0)
            max_tokens: Maximum token count
            top_p: Diversity control
            system_instruction: System instruction
            model_name: Model to use (defaults to self.model_name)
            location: Region to use (defaults to self.location)
            
        Returns:
            Generated text
        """
        try:
            # Use provided model_name and location or fall back to defaults
            use_model_name = model_name or self.model_name
            use_location = location or self.location
            
            # Reinitialize if location changed
            if use_location != self.location:
                vertexai.init(
                    project=self.project_id,
                    location=use_location,
                    credentials=self.credentials
                )
            
            # Log input
            logger.info("="*80)
            logger.info("LLM INPUT - START")
            logger.info("="*80)
            logger.info(f"Model: {use_model_name}, Location: {use_location}")
            if system_instruction:
                logger.info(f"System Instruction:\n{system_instruction}")
                logger.info("-"*80)
            logger.info(f"Prompt:\n{prompt}")
            logger.info(f"Temperature: {temperature}, Max Tokens: {max_tokens}, Top-P: {top_p}")
            logger.info("="*80)
            
            # Generation configuration
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p
            )
            
            # Recreate model with system instruction if provided
            if system_instruction:
                model = GenerativeModel(
                    use_model_name,
                    system_instruction=system_instruction
                )
            else:
                # Use specified model or default
                if use_model_name != self.model_name:
                    model = GenerativeModel(use_model_name)
                else:
                    model = self._model
            
            # Generate text
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text
            
            # Log output
            logger.info("="*80)
            logger.info("LLM OUTPUT - START")
            logger.info("="*80)
            logger.info(f"Response:\n{response_text}")
            logger.info("="*80)
            logger.info("LLM OUTPUT - END")
            logger.info("="*80)
            
            return response_text
            
        except Exception as e:
            error_msg = f"Text generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            # Re-raise the exception instead of returning error as string
            raise RuntimeError(error_msg) from e
    
    def generate_with_context(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.0,
        max_tokens: int = 32000,
        workspace_dir: Optional[str] = None,
        max_iterations: int = 100,
        model_name: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """
        Generate text with context using agentic loop.
        AI can request to read files or search for files autonomously.
        
        Args:
            prompt: User prompt
            context: Context information
            temperature: Creativity control
            max_tokens: Maximum token count
            workspace_dir: Working directory for file operations
            max_iterations: Maximum number of agent iterations
            model_name: Override default model name
            location: Override default location
            
        Returns:
            Generated text
        """
        return self._agent_execute(
            prompt=prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            workspace_dir=workspace_dir,
            max_iterations=max_iterations,
            model_name=model_name,
            location=location
        )
    
    def _agent_execute(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.0,
        max_tokens: int = 32000,
        workspace_dir: Optional[str] = None,
        max_iterations: int = 5,
        model_name: Optional[str] = None,
        location: Optional[str] = None
    ) -> str:
        """
        Execute AI agent with tool use capabilities.
        
        The agent can use the following tools:
        - <read_file path="...">: Read a file
        - <search_files pattern="..." directory="...">: Search for files
        
        Args:
            prompt: User prompt
            context: Initial context
            temperature: Creativity control
            max_tokens: Maximum token count
            workspace_dir: Working directory for file operations
            max_iterations: Maximum iterations
            model_name: Override default model name
            location: Override default location
            
        Returns:
            Final AI response
        """
        # System instruction for tool use
        system_instruction = """You are an intelligent AI assistant with the ability to manage files and tasks.

Available tools:

1. **read_file**: Read the contents of a file
<read_file path="path/to/file.py"></read_file>

2. **search_files**: Search for files matching a pattern
<search_files pattern="*.py" directory="."></search_files>

3. **create_file**: Create a new file with content (in workspace/outputs/ directory)
<create_file path="analysis_results.txt">
File content goes here...
Multiple lines are supported.
</create_file>

Important notes for create_file:
- Files are ALWAYS created in workspace/outputs/ directory
- Just specify the filename or subdirectory structure
- Example: "report.txt" creates workspace/outputs/report.txt
- Example: "data/results.csv" creates workspace/outputs/data/results.csv
- Parent directories are created automatically

4. **update_file**: Update specific lines in an existing file (in workspace/outputs/ directory)
<update_file path="analysis_results.txt" start_line="10" end_line="15">
New content for lines 10-15...
</update_file>

Important notes for update_file:
- Only files in workspace/outputs/ can be updated
- Line numbers are 1-indexed
- Specify the same path you used in create_file

5. **delete_file**: Delete a file (from workspace/outputs/ directory)
<delete_file path="temporary_file.txt"></delete_file>

Important notes for delete_file:
- Only files in workspace/outputs/ can be deleted
- Use the same path you used in create_file

6. **create_task**: Create a new sub-task for complex workflows
<create_task name="Task Name" description="Task description" type="single" priority="0">
Task prompt goes here...
</create_task>

7. **finish**: Mark the task as complete and stop iteration
<finish>
Task completed successfully. Summary: ...
</finish>

After using tools, you will receive results and can continue working.
Use multiple tools in one response if needed.
When you finish all work, use <finish> to complete the task."""

        # Build initial prompt
        if context:
            full_prompt = f"""User Request:
{prompt}

Context:
{context}

Please analyze the request and use the available tools if you need additional information."""
        else:
            full_prompt = f"""{prompt}

Please analyze the request and use the available tools if you need additional information."""
        
        conversation_history = []
        current_prompt = full_prompt
        
        for iteration in range(max_iterations):
            logger.info("="*80)
            logger.info(f"AGENT ITERATION {iteration + 1}/{max_iterations}")
            logger.info("="*80)
            
            # Generate AI response
            try:
                response = self.generate_text(
                    current_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_instruction=system_instruction,
                    model_name=model_name,
                    location=location
                )
            except Exception as e:
                logger.error(f"AI generation failed: {e}", exc_info=True)
                raise
            
            # Check for tool usage
            has_finish = '<finish' in response
            has_read_file = '<read_file' in response
            has_search_files = '<search_files' in response
            has_create_file = '<create_file' in response
            has_update_file = '<update_file' in response
            has_delete_file = '<delete_file' in response
            has_create_task = '<create_task' in response
            
            logger.info(f"Tool usage detected: finish={has_finish}, read_file={has_read_file}, search_files={has_search_files}, create_file={has_create_file}, update_file={has_update_file}, delete_file={has_delete_file}, create_task={has_create_task}")
            
            # If finish is called, stop iteration
            if has_finish:
                finish_pattern = re.compile(r'<finish>(.*?)</finish>', re.DOTALL | re.IGNORECASE)
                finish_match = finish_pattern.search(response)
                if finish_match:
                    finish_message = finish_match.group(1).strip()
                    logger.info(f"✅ Agent finished with message: {finish_message}")
                    # Remove finish tag from response
                    response = finish_pattern.sub('', response).strip()
                logger.info(f"✅ Agent completed successfully in {iteration + 1} iterations")
                logger.info("="*80)
                return response
            
            if not has_read_file and not has_search_files and not has_create_file and not has_update_file and not has_delete_file and not has_create_task:
                # No tools requested, return final response
                logger.info(f"✅ Agent completed successfully in {iteration + 1} iterations")
                logger.info("="*80)
                return response
            
            # Execute tools and build feedback
            tool_results = []
            
            # Handle read_file requests
            if has_read_file:
                read_pattern = re.compile(
                    r'<read_file\s+path=["\']([^"\']+)["\']\s*(?:/>|></read_file>)',
                    re.IGNORECASE
                )
                matches = list(read_pattern.finditer(response))
                logger.info(f"Found {len(matches)} read_file request(s)")
                for match in matches:
                    file_path = match.group(1)
                    result = self._execute_read_file(file_path, workspace_dir)
                    tool_results.append(f"<tool_result tool='read_file' path='{file_path}'>\n{result}\n</tool_result>")
            
            # Handle search_files requests
            if has_search_files:
                search_pattern = re.compile(
                    r'<search_files\s+pattern=["\']([^"\']+)["\']\s+directory=["\']([^"\']+)["\']\s*(?:/>|></search_files>)',
                    re.IGNORECASE
                )
                matches = list(search_pattern.finditer(response))
                logger.info(f"Found {len(matches)} search_files request(s)")
                for match in matches:
                    pattern = match.group(1)
                    directory = match.group(2)
                    result = self._execute_search_files(pattern, directory, workspace_dir)
                    tool_results.append(f"<tool_result tool='search_files' pattern='{pattern}' directory='{directory}'>\n{result}\n</tool_result>")
            
            # Handle create_file requests
            if has_create_file:
                create_file_pattern = re.compile(
                    r'<create_file\s+path=["\']([^"\']+)["\']\s*>(.*?)</create_file>',
                    re.DOTALL | re.IGNORECASE
                )
                matches = list(create_file_pattern.finditer(response))
                logger.info(f"Found {len(matches)} create_file request(s)")
                for match in matches:
                    file_path = match.group(1)
                    content = match.group(2).strip()
                    result = self._execute_create_file(file_path, content, workspace_dir)
                    tool_results.append(f"<tool_result tool='create_file' path='{file_path}'>\n{result}\n</tool_result>")
            
            # Handle update_file requests
            if has_update_file:
                update_file_pattern = re.compile(
                    r'<update_file\s+path=["\']([^"\']+)["\']\s+start_line=["\'](\d+)["\']\s+end_line=["\'](\d+)["\']\s*>(.*?)</update_file>',
                    re.DOTALL | re.IGNORECASE
                )
                matches = list(update_file_pattern.finditer(response))
                logger.info(f"Found {len(matches)} update_file request(s)")
                for match in matches:
                    file_path = match.group(1)
                    start_line = int(match.group(2))
                    end_line = int(match.group(3))
                    new_content = match.group(4).strip()
                    result = self._execute_update_file(file_path, start_line, end_line, new_content, workspace_dir)
                    tool_results.append(f"<tool_result tool='update_file' path='{file_path}'>\n{result}\n</tool_result>")
            
            # Handle delete_file requests
            if has_delete_file:
                delete_file_pattern = re.compile(
                    r'<delete_file\s+path=["\']([^"\']+)["\']\s*(?:/>|></delete_file>)',
                    re.IGNORECASE
                )
                matches = list(delete_file_pattern.finditer(response))
                logger.info(f"Found {len(matches)} delete_file request(s)")
                for match in matches:
                    file_path = match.group(1)
                    result = self._execute_delete_file(file_path, workspace_dir)
                    tool_results.append(f"<tool_result tool='delete_file' path='{file_path}'>\n{result}\n</tool_result>")
            
            # Handle create_task requests
            if has_create_task:
                create_task_pattern = re.compile(
                    r'<create_task\s+name=["\']([^"\']+)["\']\s+description=["\']([^"\']*)["\']\s+type=["\']([^"\']+)["\']\s+priority=["\'](\d+)["\']\s*>(.*?)</create_task>',
                    re.DOTALL | re.IGNORECASE
                )
                matches = list(create_task_pattern.finditer(response))
                logger.info(f"Found {len(matches)} create_task request(s)")
                for match in matches:
                    task_name = match.group(1)
                    task_description = match.group(2)
                    task_type = match.group(3)
                    priority = match.group(4)
                    task_prompt = match.group(5).strip()
                    result = self._execute_create_task(task_name, task_description, task_type, priority, task_prompt)
                    tool_results.append(f"<tool_result tool='create_task' name='{task_name}'>\n{result}\n</tool_result>")
            
            if not tool_results:
                # No valid tools found, return response
                logger.warning("⚠️ Tool tags found but couldn't parse them")
                return response
            
            # Log tool results
            logger.info(f"📦 Executed {len(tool_results)} tool(s)")
            for i, result in enumerate(tool_results, 1):
                logger.info(f"Tool Result {i}:")
                # Truncate long results for logging
                result_preview = result[:500] + "..." if len(result) > 500 else result
                logger.info(result_preview)
            
            # Build next prompt with tool results
            tool_results_text = "\n\n".join(tool_results)
            current_prompt = f"""Previous response:
{response}

Tool Results:
{tool_results_text}

Please continue your analysis with this new information. If you need more information, use the tools again. Otherwise, provide your final answer."""
            
            conversation_history.append({"iteration": iteration + 1, "response": response, "tools": tool_results})
            logger.info(f"Preparing next iteration with {len(tool_results)} tool result(s)")
            logger.info("="*80)
        
        # Max iterations reached
        logger.warning("="*80)
        logger.warning(f"⚠️ Agent reached maximum iterations ({max_iterations})")
        logger.warning("Returning last response")
        logger.warning("="*80)
        return response
    
    def _execute_read_file(self, file_path: str, workspace_dir: Optional[str]) -> str:
        """Execute read_file tool."""
        try:
            # Try multiple resolution strategies
            possible_paths = []
            
            # Strategy 1: Relative to workspace_dir
            if workspace_dir:
                possible_paths.append(os.path.join(workspace_dir, file_path))
            
            # Strategy 2: Absolute path
            if os.path.isabs(file_path):
                possible_paths.append(file_path)
                
            # Strategy 3: Relative to current working directory
            possible_paths.append(os.path.abspath(file_path))
            
            # Strategy 4: Relative to project root (parent of workspaces)
            if workspace_dir and 'workspaces' in workspace_dir:
                # Go up from workspaces directory to project root
                workspace_parent = Path(workspace_dir)
                while workspace_parent.name != 'workspaces' and workspace_parent.parent != workspace_parent:
                    workspace_parent = workspace_parent.parent
                if workspace_parent.name == 'workspaces':
                    project_root = workspace_parent.parent
                    possible_paths.append(os.path.join(str(project_root), file_path))
            
            # Try each possible path
            for full_path in possible_paths:
                full_path = os.path.normpath(full_path)
                if os.path.exists(full_path):
                    logger.info(f"Reading file: {full_path}")
                    content = self.file_manager.read_file(full_path)
                    return content
            
            # If no path worked, return error with attempted paths
            error_msg = f"Failed to find file '{file_path}'. Tried paths:\n" + "\n".join(f"  - {p}" for p in possible_paths)
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
            
        except Exception as e:
            error_msg = f"Failed to read file '{file_path}': {str(e)}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    def _execute_search_files(self, pattern: str, directory: str, workspace_dir: Optional[str]) -> str:
        """Execute search_files tool."""
        try:
            # Try multiple resolution strategies
            possible_dirs = []
            
            # Strategy 1: Absolute path
            if os.path.isabs(directory):
                possible_dirs.append(directory)
            
            # Strategy 2: Relative to workspace_dir
            if workspace_dir:
                possible_dirs.append(os.path.join(workspace_dir, directory))
            
            # Strategy 3: Relative to current working directory
            possible_dirs.append(os.path.abspath(directory))
            
            # Strategy 4: Relative to project root
            if workspace_dir and 'workspaces' in workspace_dir:
                workspace_parent = Path(workspace_dir)
                while workspace_parent.name != 'workspaces' and workspace_parent.parent != workspace_parent:
                    workspace_parent = workspace_parent.parent
                if workspace_parent.name == 'workspaces':
                    project_root = workspace_parent.parent
                    possible_dirs.append(os.path.join(str(project_root), directory))
            
            # Try each possible directory
            for full_dir in possible_dirs:
                full_dir = os.path.normpath(full_dir)
                if os.path.exists(full_dir) and os.path.isdir(full_dir):
                    logger.info(f"Searching files: pattern={pattern}, directory={full_dir}")
                    files = self.file_manager.search_files(full_dir, pattern, recursive=True)
                    
                    if not files:
                        continue  # Try next directory
                    
                    # Return list of files (relative to workspace if possible)
                    if workspace_dir:
                        try:
                            relative_files = []
                            for f in files:
                                try:
                                    rel = os.path.relpath(f, workspace_dir)
                                    relative_files.append(rel)
                                except:
                                    relative_files.append(f)
                            files = relative_files
                        except:
                            pass
                    
                    return f"Found {len(files)} file(s) in {full_dir}:\n" + "\n".join(f"- {f}" for f in files[:50])
            
            # If no directory worked
            error_msg = f"No files found matching pattern '{pattern}'. Tried directories:\n" + "\n".join(f"  - {d}" for d in possible_dirs)
            logger.warning(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Failed to search files with pattern '{pattern}': {str(e)}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
    
    def _execute_create_file(self, file_path: str, content: str, workspace_dir: Optional[str]) -> str:
        """Execute create_file tool - creates files in workspace/outputs/ directory."""
        try:
            # Determine the outputs directory
            if workspace_dir:
                outputs_dir = os.path.join(workspace_dir, "outputs")
            else:
                outputs_dir = os.path.join(os.getcwd(), "outputs")
            
            # Ensure outputs directory exists
            os.makedirs(outputs_dir, exist_ok=True)
            
            # Remove any leading path separators or "../" to prevent directory traversal
            file_path = file_path.lstrip('/\\').replace('..', '')
            
            # Construct full path (always inside outputs directory)
            full_path = os.path.join(outputs_dir, file_path)
            full_path = os.path.normpath(full_path)
            
            # Security check: ensure the final path is still inside outputs directory
            if not full_path.startswith(outputs_dir):
                error_msg = f"Security error: File path '{file_path}' attempts to escape outputs directory"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
            
            # Create parent directories within outputs
            parent_dir = os.path.dirname(full_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            # Write file
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Get relative path for display
                rel_path = os.path.relpath(full_path, workspace_dir if workspace_dir else os.getcwd())
                
                logger.info(f"✅ Created file: {rel_path} ({len(content)} bytes)")
                
                # Update context manager to include new file
                from .context_manager import get_context_manager
                context_manager = get_context_manager()
                context_manager.add_file_to_context(full_path, force_reload=True)
                
                return f"SUCCESS: File created in outputs directory\nPath: {rel_path}\nSize: {len(content)} bytes"
                
            except PermissionError as pe:
                error_msg = f"Permission denied writing to '{file_path}': {str(pe)}"
                logger.error(error_msg)
                return f"ERROR: {error_msg}"
            except Exception as we:
                error_msg = f"Failed to write file '{file_path}': {str(we)}"
                logger.error(error_msg, exc_info=True)
                return f"ERROR: {error_msg}"
            
        except Exception as e:
            error_msg = f"Failed to create file '{file_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"ERROR: {error_msg}"
    
    def _execute_update_file(self, file_path: str, start_line: int, end_line: int, new_content: str, workspace_dir: Optional[str]) -> str:
        """Execute update_file tool - updates files in workspace/outputs/ directory."""
        try:
            # Determine the outputs directory
            if workspace_dir:
                outputs_dir = os.path.join(workspace_dir, "outputs")
            else:
                outputs_dir = os.path.join(os.getcwd(), "outputs")
            
            # Remove any leading path separators or "../"
            file_path = file_path.lstrip('/\\').replace('..', '')
            
            # Try to find file in outputs directory
            possible_paths = [
                os.path.join(outputs_dir, file_path),  # Direct path in outputs
                os.path.normpath(os.path.join(outputs_dir, file_path))  # Normalized path
            ]
            
            resolved_path = None
            for p in possible_paths:
                if os.path.exists(p) and p.startswith(outputs_dir):
                    resolved_path = p
                    break
            
            if not resolved_path:
                error_msg = f"File not found in outputs directory: {file_path}"
                logger.warning(error_msg)
                return f"ERROR: {error_msg}\nNote: Files must be in workspace/outputs/ directory"
            
            # Read existing content
            with open(resolved_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Validate line numbers
            if start_line < 1 or end_line > len(lines) or start_line > end_line:
                return f"ERROR: Invalid line range. File has {len(lines)} lines, requested {start_line}-{end_line}"
            
            # Update lines (convert to 0-indexed)
            new_lines = new_content.split('\n')
            lines[start_line-1:end_line] = [line + '\n' for line in new_lines]
            
            # Write back
            with open(resolved_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Get relative path
            rel_path = os.path.relpath(resolved_path, workspace_dir if workspace_dir else os.getcwd())
            
            logger.info(f"✅ Updated file: {rel_path}, lines {start_line}-{end_line}")
            
            # Update context manager
            from .context_manager import get_context_manager
            context_manager = get_context_manager()
            context_manager.add_file_to_context(resolved_path, force_reload=True)
            
            return f"SUCCESS: Updated file in outputs directory\nPath: {rel_path}\nLines: {start_line}-{end_line} ({len(new_lines)} new lines)"
            
        except Exception as e:
            error_msg = f"Failed to update file '{file_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"ERROR: {error_msg}"
    
    def _execute_delete_file(self, file_path: str, workspace_dir: Optional[str]) -> str:
        """Execute delete_file tool - deletes files from workspace/outputs/ directory."""
        try:
            # Determine the outputs directory
            if workspace_dir:
                outputs_dir = os.path.join(workspace_dir, "outputs")
            else:
                outputs_dir = os.path.join(os.getcwd(), "outputs")
            
            # Remove any leading path separators or "../"
            file_path = file_path.lstrip('/\\').replace('..', '')
            
            # Try to find file in outputs directory
            possible_paths = [
                os.path.join(outputs_dir, file_path),
                os.path.normpath(os.path.join(outputs_dir, file_path))
            ]
            
            resolved_path = None
            for p in possible_paths:
                if os.path.exists(p) and p.startswith(outputs_dir):
                    resolved_path = p
                    break
            
            if not resolved_path:
                error_msg = f"File not found in outputs directory: {file_path}"
                logger.warning(error_msg)
                return f"ERROR: {error_msg}\nNote: Can only delete files in workspace/outputs/ directory"
            
            # Get relative path before deletion
            rel_path = os.path.relpath(resolved_path, workspace_dir if workspace_dir else os.getcwd())
            
            # Delete file
            os.remove(resolved_path)
            
            logger.info(f"✅ Deleted file: {rel_path}")
            
            # Remove from context manager
            from .context_manager import get_context_manager
            context_manager = get_context_manager()
            if resolved_path in context_manager.context_files:
                del context_manager.context_files[resolved_path]
            
            return f"SUCCESS: Deleted file from outputs directory\nPath: {rel_path}"
            
        except Exception as e:
            error_msg = f"Failed to delete file '{file_path}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"ERROR: {error_msg}"
    
    def _execute_create_task(self, name: str, description: str, task_type: str, priority: str, prompt: str) -> str:
        """Execute create_task tool."""
        try:
            # Import here to avoid circular dependency
            from .task_manager import get_task_manager, TaskType
            
            task_manager = get_task_manager()
            
            # Convert task_type string to TaskType enum
            task_type_map = {
                'single': TaskType.SINGLE,
                'parallel': TaskType.PARALLEL,
                'sequential': TaskType.SEQUENTIAL
            }
            task_type_enum = task_type_map.get(task_type.lower(), TaskType.SINGLE)
            
            # Create the task
            new_task = task_manager.create_task(
                name=name,
                prompt=prompt,
                description=description,
                task_type=task_type_enum,
                priority=int(priority)
            )
            
            logger.info(f"✅ Created new task: {new_task.task_id} - {name}")
            
            return f"SUCCESS: Task created successfully!\n" \
                   f"Task ID: {new_task.task_id}\n" \
                   f"Name: {name}\n" \
                   f"Type: {task_type}\n" \
                   f"Priority: {priority}\n" \
                   f"Workspace: {new_task.workspace_dir}\n" \
                   f"The task has been added to the queue and can be executed later."
            
        except Exception as e:
            error_msg = f"Failed to create task '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"ERROR: {error_msg}"
    
    def analyze_task_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt to extract task information
        
        Args:
            prompt: Prompt to analyze
            
        Returns:
            Task information dictionary
        """
        analysis_prompt = f"""Please analyze the following prompt and extract task information in JSON format:

Prompt: {prompt}

Information to extract:
1. tasks: List of tasks to perform (in order)
2. files_to_read: Files that need to be read
3. files_to_create: Files that need to be created
4. files_to_modify: Files that need to be modified
5. files_to_delete: Files that need to be deleted
6. parallel_execution: Whether parallel execution is needed
7. csv_parameters: Extract CSV parameters if any

Please respond only in JSON format."""

        try:
            response = self.generate_text(
                analysis_prompt,
                temperature=0.3,  # Low temperature for consistency
                max_tokens=1024
            )
            
            # Try JSON parsing
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Cannot parse as JSON format", "raw_response": response}
                
        except Exception as e:
            return {"error": f"Task analysis failed: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, str]:
        """Return model information"""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model_name": self.model_name,
            "status": "Connected" if self._model else "Disconnected"
        }
    
    def list_available_models(self, use_api: bool = True, use_cache: bool = True) -> List[str]:
        """
        Get list of available Gemini models.
        
        Args:
            use_api: If True, fetch models dynamically from Vertex AI API.
                     If False, return curated static list.
            use_cache: If True, use cached results if available and not expired.
        
        Returns:
            List of model names
        """
        # Check cache first
        if use_cache and self._is_cache_valid():
            logger.debug(f"Using cached model list ({len(self._model_cache)} models)")
            return self._model_cache.copy()
        
        if use_api:
            try:
                # Method 1: Try to use aiplatform API to list publisher models
                from google.cloud import aiplatform_v1
                
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                
                # Create client for model service
                client = aiplatform_v1.ModelServiceClient(credentials=credentials)
                
                # List publisher models (Google's official models)
                # Publisher models are in the format: publishers/google/models/*
                parent = "publishers/google/locations/us-central1"
                
                try:
                    request = aiplatform_v1.ListPublisherModelsRequest(parent=parent)
                    models = []
                    
                    # Fetch all publisher models
                    page_result = client.list_publisher_models(request=request)
                    
                    for model in page_result:
                        model_id = model.name.split('/')[-1]
                        # Filter only Gemini models
                        if 'gemini' in model_id.lower():
                            models.append(model_id)
                            logger.debug(f"Found model via API: {model_id}")
                    
                    if models:
                        # Sort by version (newest first)
                        models.sort(reverse=True)
                        logger.info(f"✅ Found {len(models)} Gemini models via Vertex AI API")
                        
                        # Update cache
                        self._update_cache(models)
                        
                        return models
                        
                except Exception as api_error:
                    logger.debug(f"Publisher model listing failed: {api_error}")
                
                # Method 2: Fallback - Try to instantiate known model patterns
                logger.info("Trying fallback method: testing model availability...")
                from vertexai.generative_models import GenerativeModel
                
                # Extended model patterns including future versions
                model_patterns = [
                    # Latest experimental/preview models
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash-exp",
                    "gemini-2.0-pro-exp",
                    
                    # Current stable models
                    "gemini-2.5-flash-lite",
                    "gemini-1.5-pro-002",
                    "gemini-1.5-flash-002",
                    "gemini-1.5-pro-001",
                    "gemini-1.5-flash-001",
                    
                    # Legacy models
                    "gemini-1.0-pro-002",
                    "gemini-1.0-pro-001",
                    "gemini-1.0-pro",
                    "gemini-pro",
                    
                    # Future-proofing patterns
                    "gemini-2.5-pro-002",
                    "gemini-2.5-pro-001",
                    "gemini-2.0-flash-002",
                    "gemini-2.0-flash-001",
                    "gemini-2.0-pro-002",
                    "gemini-2.0-pro-001",
                ]
                
                available_models = []
                
                # Test each model to see if it's available
                for model_name in model_patterns:
                    try:
                        # Quick validation by attempting to create model instance
                        test_model = GenerativeModel(model_name)
                        available_models.append(model_name)
                        logger.debug(f"✓ Model {model_name} is available")
                    except Exception as e:
                        logger.debug(f"✗ Model {model_name} not available: {e}")
                        continue
                
                if available_models:
                    logger.info(f"✅ Found {len(available_models)} available models via testing")
                    
                    # Update cache
                    self._update_cache(available_models)
                    
                    return available_models
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch models dynamically: {e}")
                logger.info("Falling back to static curated model list")
        
        # Fallback: Curated static list (updated regularly)
        # This list is used when API access fails or use_api=False
        # Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
        logger.info("Using static model list")
        static_models = [
            # Latest models (as of 2025)
            "gemini-2.0-flash-exp",
            "gemini-2.5-flash-lite",
            
            # Stable production models
            "gemini-1.5-pro-002",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-001",
            
            # Legacy models
            "gemini-1.0-pro-002",
            "gemini-1.0-pro-001",
        ]
        
        # Cache static list too
        self._update_cache(static_models)
        
        return static_models
    
    def _is_cache_valid(self) -> bool:
        """Check if model cache is valid and not expired."""
        if self._model_cache is None or self._model_cache_timestamp is None:
            return False
        
        import time
        elapsed = time.time() - self._model_cache_timestamp
        return elapsed < self._model_cache_ttl
    
    def _update_cache(self, models: List[str]):
        """Update model cache with new data."""
        import time
        VertexAIClient._model_cache = models.copy()
        VertexAIClient._model_cache_timestamp = time.time()
        logger.debug(f"Model cache updated with {len(models)} models")
    
    def refresh_model_cache(self) -> List[str]:
        """Force refresh the model cache by fetching from API."""
        logger.info("Force refreshing model cache...")
        return self.list_available_models(use_api=True, use_cache=False)
    
    def list_available_locations(self) -> List[str]:
        """
        Get list of available Google Cloud regions for Vertex AI.
        
        Returns:
            List of region names
        """
        # Available regions for Vertex AI Gemini API
        return [
            "us-central1",           # Iowa, USA
            "us-east4",              # Northern Virginia, USA
            "us-west1",              # Oregon, USA
            "us-west4",              # Nevada, USA
            "europe-west1",          # Belgium
            "europe-west2",          # London, UK
            "europe-west3",          # Frankfurt, Germany
            "europe-west4",          # Netherlands
            "asia-northeast1",       # Tokyo, Japan
            "asia-northeast3",       # Seoul, South Korea
            "asia-southeast1",       # Singapore
            "asia-south1",           # Mumbai, India
            "australia-southeast1",  # Sydney, Australia
        ]


# Global client instance (Singleton pattern)
_vertex_client: Optional[VertexAIClient] = None


def get_vertex_client() -> VertexAIClient:
    """Return the global Vertex AI client."""
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = VertexAIClient()
    return _vertex_client


def initialize_vertex_ai(credentials_path: str = "vertex-ai-credentials.json") -> VertexAIClient:
    """Initialize the Vertex AI client."""
    global _vertex_client
    _vertex_client = VertexAIClient(credentials_path)
    return _vertex_client