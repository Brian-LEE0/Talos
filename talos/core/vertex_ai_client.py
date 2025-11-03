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
        
        self._initialize()
    
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
            
            # Create service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials
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
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate text using AI model
        
        Args:
            prompt: Input prompt
            temperature: Creativity control (0.0-1.0)
            max_tokens: Maximum token count
            top_p: Diversity control
            system_instruction: System instruction
            
        Returns:
            Generated text
        """
        try:
            # Log input
            logger.info("="*80)
            logger.info("LLM INPUT - START")
            logger.info("="*80)
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
                    self.model_name,
                    system_instruction=system_instruction
                )
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
        max_iterations: int = 100
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
            
        Returns:
            Generated text
        """
        return self._agent_execute(
            prompt=prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            workspace_dir=workspace_dir,
            max_iterations=max_iterations
        )
    
    def _agent_execute(
        self,
        prompt: str,
        context: str = "",
        temperature: float = 0.0,
        max_tokens: int = 32000,
        workspace_dir: Optional[str] = None,
        max_iterations: int = 5
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
            
        Returns:
            Final AI response
        """
        # System instruction for tool use
        system_instruction = """You are an intelligent AI assistant with the ability to read files and search for files when needed.

When you need to read a file, use this format:
<read_file path="path/to/file.py"></read_file>

When you need to search for files, use this format:
<search_files pattern="*.py" directory="."></search_files>

After using these tools, you will receive the results and can continue your analysis.
You can use multiple tools in one response if needed.
When you have all the information you need, provide your final answer without any tool tags.

Available tools:
1. read_file: Read the contents of a file
2. search_files: Search for files matching a pattern

Use these tools wisely to gather the information you need to complete the task."""

        # Build initial prompt
        if context:
            full_prompt = f"""Context:
{context}

User Request:
{prompt}

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
                    system_instruction=system_instruction
                )
            except Exception as e:
                logger.error(f"AI generation failed: {e}", exc_info=True)
                raise
            
            # Check for tool usage
            has_read_file = '<read_file' in response
            has_search_files = '<search_files' in response
            
            logger.info(f"Tool usage detected: read_file={has_read_file}, search_files={has_search_files}")
            
            if not has_read_file and not has_search_files:
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