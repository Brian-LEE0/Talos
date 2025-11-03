"""
Context Management System
Manages file contents with <file> tags and provides mention functionality and context summarization
"""

import os
import re
import json
import time
import tiktoken
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from .file_manager import get_file_manager
from .vertex_ai_client import get_vertex_client


@dataclass
class FileContext:
    """File context information"""
    path: str
    content: str
    hash_md5: str
    size: int
    added_time: float
    last_accessed: float
    is_summary: bool = False
    original_size: int = 0


@dataclass 
class MentionedFile:
    """Mentioned file information"""
    path: str
    mention_type: str  # 'file', 'directory'
    context: str  # Context where mention was found
    line_number: int


class ContextManager:
    """Context manager"""
    
    def __init__(self, max_context_size: int = 32000, max_files: int = 50):
        """
        Initialize context manager
        
        Args:
            max_context_size: Maximum context size (token count)
            max_files: Maximum number of files
        """
        self.max_context_size = max_context_size
        self.max_files = max_files
        self.file_manager = get_file_manager()
        self.ai_client = get_vertex_client()
        
        # Context storage
        self.context_files: Dict[str, FileContext] = {}
        
        # Token calculator (based on GPT-3.5-turbo)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # If no tokenizer, approximate (4 chars = 1 token)
            return len(text) // 4
    
    def parse_mentions(self, text: str) -> List[MentionedFile]:
        """
        Parse file/directory mentions from text.

        Supported formats:
        - @filename.ext
        - @directory/
        """
        mentions = []
        lines = text.split('\n')
        
        # Define mention patterns
        patterns = [
            (r'@([^\s/]+\.[^\s/]+)', 'file'),  # @filename.ext
            (r'@(\./[^\s]+)', 'file'),  # @./path/to/file
            (r'@([^\s/]+/)', 'directory'),  # @directory_name/
            (r'@(\./[^\s]*/)', 'directory'),  # @./path/to/directory/
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, mention_type in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    path = match.group(1)
                    mentions.append(MentionedFile(
                        path=path,
                        mention_type=mention_type,
                        context=line.strip(),
                        line_number=line_num
                    ))
        
        return mentions
    
    def resolve_mention_path(self, mention_path: str, base_dir: str = ".", project_root: Optional[str] = None) -> Optional[str]:
        """
        Resolve a mention path to an actual file path.
        
        Tries multiple strategies:
        1. Absolute path
        2. Relative to base_dir (workspace)
        3. Relative to project_root
        """
        try:
            # Absolute path
            if os.path.isabs(mention_path):
                return mention_path if os.path.exists(mention_path) else None
            
            # Strategy 1: Relative to base_dir (workspace)
            if mention_path.startswith('./'):
                full_path = os.path.join(base_dir, mention_path[2:])
            else:
                full_path = os.path.join(base_dir, mention_path)
            
            full_path = os.path.normpath(full_path)
            if os.path.exists(full_path):
                return full_path
            
            # Strategy 2: Relative to project_root
            if project_root:
                if mention_path.startswith('./'):
                    root_path = os.path.join(project_root, mention_path[2:])
                else:
                    root_path = os.path.join(project_root, mention_path)
                
                root_path = os.path.normpath(root_path)
                if os.path.exists(root_path):
                    return root_path
            
            return None
            
        except Exception:
            return None
    
    def add_file_to_context(self, file_path: str, force_reload: bool = False) -> bool:
        """
        Add a file to the context.
        
        Args:
            file_path: File path.
            force_reload: Whether to force reload.
            
        Returns:
            Whether the addition was successful.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return False
            
            abs_path = os.path.abspath(file_path)
            
            # If already in context and not force_reload
            if abs_path in self.context_files and not force_reload:
                # Update access time only
                self.context_files[abs_path].last_accessed = time.time()
                return True
            
            # Read file
            content = self.file_manager.read_file(file_path)
            
            # Create file information
            file_info = self.file_manager.get_file_info(file_path)
            if not file_info:
                return False
            
            # Add to context
            file_context = FileContext(
                path=abs_path,
                content=content,
                hash_md5=file_info.hash_md5 or "",
                size=len(content),
                added_time=time.time(),
                last_accessed=time.time(),
                is_summary=False,
                original_size=len(content)
            )
            
            self.context_files[abs_path] = file_context
            
            # Manage context size
            self._manage_context_size()
            
            print(f"Added file to context: {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to add file to context ({file_path}): {e}")
            return False
    
    def add_directory_to_context(self, dir_path: str, pattern: str = "*", recursive: bool = False) -> int:
        """
        Add files from a directory to the context.
        
        Args:
            dir_path: Directory path.
            pattern: File pattern.
            recursive: Whether to include subdirectories.
            
        Returns:
            Number of files added.
        """
        try:
            if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
                return 0
            
            files = self.file_manager.search_files(dir_path, pattern, recursive)
            added_count = 0
            
            for file_path in files:
                if self.add_file_to_context(file_path):
                    added_count += 1
            
            print(f"Added {added_count} files from directory to context: {dir_path}")
            return added_count
            
        except Exception as e:
            print(f"Failed to add directory to context ({dir_path}): {e}")
            return 0
    
    def process_mentions(self, text: str, base_dir: str = ".", project_root: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Process mentions in text and add to context.
        
        Args:
            text: Text to analyze.
            base_dir: Base directory (workspace).
            project_root: Project root directory.
            
        Returns:
            A tuple containing a list of successful files and a list of failed files.
        """
        mentions = self.parse_mentions(text)
        successful = []
        failed = []
        
        for mention in mentions:
            real_path = self.resolve_mention_path(mention.path, base_dir, project_root)
            
            if real_path:
                if mention.mention_type == 'file':
                    if self.add_file_to_context(real_path):
                        successful.append(real_path)
                    else:
                        failed.append(mention.path)
                        
                elif mention.mention_type == 'directory':
                    count = self.add_directory_to_context(real_path)
                    if count > 0:
                        successful.append(f"{real_path} ({count} files)")
                    else:
                        failed.append(mention.path)
            else:
                failed.append(mention.path)
        
        return successful, failed
    
    def summarize_file(self, file_path: str) -> Optional[str]:
        """Summarize file content."""
        try:
            abs_path = os.path.abspath(file_path)
            
            if abs_path not in self.context_files:
                return None
            
            file_context = self.context_files[abs_path]
            
            # If already summarized
            if file_context.is_summary:
                return file_context.content
            
            # Summarize only if the file is large enough
            if file_context.size < 1000:  # Do not summarize if less than 1KB
                return file_context.content
            
            # Generate summary using AI
            summary_prompt = f"""Please summarize the following file content concisely:

File Name: {os.path.basename(file_path)}
Content:
{file_context.content[:2000]}...

Please include the main points, functions/classes, and core logic in the summary."""
            
            summary = self.ai_client.generate_text(
                summary_prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Update with summarized context
            file_context.content = summary
            file_context.is_summary = True
            file_context.size = len(summary)
            
            print(f"File summarized successfully: {file_path} ({file_context.original_size} -> {file_context.size} chars)")
            
            return summary
            
        except Exception as e:
            print(f"Failed to summarize file ({file_path}): {e}")
            return None
    
    def _manage_context_size(self):
        """Manage context size."""
        # Limit number of files
        if len(self.context_files) > self.max_files:
            self._remove_old_files()
        
        # Limit number of tokens
        current_tokens = self.get_context_token_count()
        if current_tokens > self.max_context_size:
            self._reduce_context_size()
    
    def _remove_old_files(self):
        """Remove old files."""
        # Sort by access time
        sorted_files = sorted(
            self.context_files.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Remove half
        to_remove = sorted_files[:len(sorted_files) // 2]
        
        for file_path, _ in to_remove:
            del self.context_files[file_path]
            print(f"Removed old file from context: {file_path}")
    
    def _reduce_context_size(self):
        """Reduce context size (by summarizing or removing)."""
        # Convert large files to summaries
        large_files = [
            (path, context) for path, context in self.context_files.items()
            if not context.is_summary and context.size > 1000
        ]
        
        # Sort by size
        large_files.sort(key=lambda x: x[1].size, reverse=True)
        
        for file_path, _ in large_files[:len(large_files) // 2]:
            self.summarize_file(file_path)
            
            # Re-check token count
            if self.get_context_token_count() <= self.max_context_size:
                break
    
    def get_context_token_count(self) -> int:
        """Get the total token count of the current context."""
        total_tokens = 0
        for file_context in self.context_files.values():
            total_tokens += self.count_tokens(file_context.content)
        return total_tokens
    
    def generate_context_xml(self, include_summary_info: bool = True) -> str:
        """Generate context in XML format."""
        xml_parts = []
        
        for file_path, file_context in self.context_files.items():
            rel_path = os.path.relpath(file_path)
            
            if include_summary_info and file_context.is_summary:
                xml_parts.append(f'<file path="{rel_path}" summary="true" original_size="{file_context.original_size}">')
            else:
                xml_parts.append(f'<file path="{rel_path}">')
            
            xml_parts.append(file_context.content)
            xml_parts.append('</file>')
            xml_parts.append('')  # Add a blank line
        
        return '\n'.join(xml_parts)
    
    def get_context_info(self) -> Dict[str, Any]:
        """Return context information."""
        total_size = sum(fc.size for fc in self.context_files.values())
        total_tokens = self.get_context_token_count()
        summary_count = sum(1 for fc in self.context_files.values() if fc.is_summary)
        
        return {
            "file_count": len(self.context_files),
            "total_size": total_size,
            "total_tokens": total_tokens,
            "summary_count": summary_count,
            "max_context_size": self.max_context_size,
            "max_files": self.max_files,
            "files": [
                {
                    "path": os.path.relpath(path),
                    "size": fc.size,
                    "is_summary": fc.is_summary,
                    "added_time": fc.added_time,
                    "last_accessed": fc.last_accessed
                }
                for path, fc in self.context_files.items()
            ]
        }
    
    def remove_file_from_context(self, file_path: str) -> bool:
        """Remove a file from the context."""
        try:
            abs_path = os.path.abspath(file_path)
            if abs_path in self.context_files:
                del self.context_files[abs_path]
                print(f"Removed file from context: {file_path}")
                return True
            return False
        except Exception as e:
            print(f"Failed to remove file from context ({file_path}): {e}")
            return False
    
    def clear_context(self):
        """Clear the entire context."""
        self.context_files.clear()
        print("Context cleared successfully.")
    
    def get_context_files(self) -> List[Dict[str, Any]]:
        """
        Return a list of files in the context.
        
        Returns:
            A list of file information.
        """
        files = []
        for path, context in self.context_files.items():
            files.append({
                'path': path,
                'size': context.size,
                'is_summary': context.is_summary,
                'original_size': context.original_size if context.is_summary else context.size,
                'added_time': context.added_time,
                'last_accessed': context.last_accessed
            })
        return files
    
    def save_context(self, file_path: str) -> bool:
        """Save the context to a file."""
        try:
            context_xml = self.generate_context_xml()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(context_xml)
            print(f"Context saved successfully: {file_path}")
            return True
        except Exception as e:
            print(f"Failed to save context: {e}")
            return False


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Return the global context manager."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager