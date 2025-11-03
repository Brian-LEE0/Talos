"""
File System Management Module
File CRUD operations with caching system
"""

import os
import shutil
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import aiofiles
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
import fnmatch
from ..i18n import t


@dataclass
class FileInfo:
    """File information class"""
    path: str
    size: int
    modified_time: float
    created_time: float
    is_directory: bool
    hash_md5: Optional[str] = None
    content_preview: Optional[str] = None


@dataclass
class CachedFile:
    """Cached file information"""
    path: str
    content: str
    hash_md5: str
    modified_time: float
    cached_time: float
    size: int


class FileCache:
    """File cache management class"""
    
    def __init__(self, cache_dir: str = ".cache", max_cache_size: int = 100):
        """
        Initialize cache
        
        Args:
            cache_dir: Cache directory path
            max_cache_size: Maximum number of cached files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index: Dict[str, CachedFile] = {}
        self.load_cache_index()
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def load_cache_index(self):
        """Load cache index"""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_index = {
                        k: CachedFile(**v) for k, v in data.items()
                    }
        except Exception as e:
            print(f"Cache index load failed: {e}")
            self.cache_index = {}
    
    def save_cache_index(self):
        """Save cache index."""
        try:
            data = {k: asdict(v) for k, v in self.cache_index.items()}
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save cache index: {e}")
    
    def is_file_cached(self, file_path: str) -> bool:
        """Check if a file is cached and valid."""
        abs_path = os.path.abspath(file_path)
        
        if abs_path not in self.cache_index:
            return False
        
        cached_file = self.cache_index[abs_path]
        
        # Check if the file exists
        if not os.path.exists(file_path):
            return False
        
        # Check if modification times are the same
        current_mtime = os.path.getmtime(file_path)
        if abs(current_mtime - cached_file.modified_time) > 1:  # Allow 1-second tolerance
            return False
        
        # Check if hashes are the same (optional)
        current_hash = self._get_file_hash(file_path)
        if current_hash != cached_file.hash_md5:
            return False
        
        return True
    
    def get_cached_content(self, file_path: str) -> Optional[str]:
        """Return cached file content."""
        abs_path = os.path.abspath(file_path)
        
        if self.is_file_cached(file_path):
            cached_file = self.cache_index[abs_path]
            
            # Read content from cache file
            cache_file_path = self.cache_dir / f"{cached_file.hash_md5}.txt"
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                # If cache file does not exist, remove from cache
                del self.cache_index[abs_path]
                self.save_cache_index()
        
        return None
    
    def cache_file(self, file_path: str, content: str):
        """Save a file to the cache."""
        abs_path = os.path.abspath(file_path)
        
        # Check cache size limit
        if len(self.cache_index) >= self.max_cache_size:
            self.cleanup_old_cache()
        
        # Collect file information
        file_hash = self._get_file_hash(file_path)
        modified_time = os.path.getmtime(file_path)
        file_size = len(content.encode('utf-8'))
        
        # Create cache file
        cache_file_path = self.cache_dir / f"{file_hash}.txt"
        try:
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Failed to save cache file: {e}")
            return
        
        # Update cache index
        cached_file = CachedFile(
            path=abs_path,
            content="",  # Actual content is stored in a separate file
            hash_md5=file_hash,
            modified_time=modified_time,
            cached_time=time.time(),
            size=file_size
        )
        
        self.cache_index[abs_path] = cached_file
        self.save_cache_index()
    
    def cleanup_old_cache(self):
        """Clean up old cache."""
        if len(self.cache_index) <= self.max_cache_size:
            return
        
        # Sort by cached time
        sorted_cache = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].cached_time
        )
        
        # Delete old cache
        to_remove = sorted_cache[:len(sorted_cache) - self.max_cache_size + 1]
        
        for file_path, cached_file in to_remove:
            # Delete cache file
            cache_file_path = self.cache_dir / f"{cached_file.hash_md5}.txt"
            try:
                cache_file_path.unlink(missing_ok=True)
            except Exception:
                pass
            
            # Remove from index
            del self.cache_index[file_path]
        
        self.save_cache_index()
    
    def clear_cache(self):
        """Clear the entire cache."""
        try:
            # Delete cache files
            for file in self.cache_dir.glob("*.txt"):
                file.unlink()
            
            # Initialize index
            self.cache_index = {}
            self.save_cache_index()
            
        except Exception as e:
            print(f"Failed to clear cache: {e}")


class FileManager:
    """File management class"""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: str = ".cache"):
        """
        Initialize file manager
        
        Args:
            cache_enabled: Whether to use cache
            cache_dir: Cache directory path
        """
        self.cache_enabled = cache_enabled
        self.cache = FileCache(cache_dir) if cache_enabled else None
    
    def read_file(self, file_path: str, use_cache: bool = True) -> str:
        """
        Read a file.
        
        Args:
            file_path: File path
            use_cache: Whether to use cache
            
        Returns:
            File content
        """
        try:
            # Check cache
            if use_cache and self.cache_enabled and self.cache:
                cached_content = self.cache.get_cached_content(file_path)
                if cached_content is not None:
                    print(f"Reading file from cache: {file_path}")
                    return cached_content
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Save to cache
            if use_cache and self.cache_enabled and self.cache:
                self.cache.cache_file(file_path, content)
            
            print(f"File read successfully: {file_path}")
            return content
            
        except Exception as e:
            error_msg = f"Failed to read file ({file_path}): {e}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
    
    async def read_file_async(self, file_path: str, use_cache: bool = True) -> str:
        """Read a file asynchronously."""
        try:
            # Check cache
            if use_cache and self.cache_enabled and self.cache:
                cached_content = self.cache.get_cached_content(file_path)
                if cached_content is not None:
                    return cached_content
            
            # Read file asynchronously
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Save to cache
            if use_cache and self.cache_enabled and self.cache:
                self.cache.cache_file(file_path, content)
            
            return content
            
        except Exception as e:
            error_msg = f"Failed to read file asynchronously ({file_path}): {e}"
            raise FileNotFoundError(error_msg)
    
    def create_file(self, file_path: str, content: str, overwrite: bool = False) -> bool:
        """
        Create a file.
        
        Args:
            file_path: File path
            content: File content
            overwrite: Whether to overwrite
            
        Returns:
            Whether creation was successful
        """
        try:
            # Check if file exists
            if os.path.exists(file_path) and not overwrite:
                raise FileExistsError(f"File already exists: {file_path}")
            
            # Create directory
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Create file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Update cache
            if self.cache_enabled and self.cache:
                self.cache.cache_file(file_path, content)
            
            print(f"File created successfully: {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to create file ({file_path}): {e}")
            return False
    
    async def create_file_async(self, file_path: str, content: str, overwrite: bool = False) -> bool:
        """Create a file asynchronously."""
        try:
            if os.path.exists(file_path) and not overwrite:
                raise FileExistsError(f"File already exists: {file_path}")
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            if self.cache_enabled and self.cache:
                self.cache.cache_file(file_path, content)
            
            return True
            
        except Exception as e:
            print(f"Failed to create file asynchronously ({file_path}): {e}")
            return False
    
    def update_file_lines(self, file_path: str, line_updates: Dict[int, str]) -> bool:
        """
        Update file lines.
        
        Args:
            file_path: File path
            line_updates: Dictionary of {line_number: new_content} (0-based)
            
        Returns:
            Whether update was successful
        """
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Update lines
            for line_no, new_content in line_updates.items():
                if 0 <= line_no < len(lines):
                    lines[line_no] = new_content + '\n' if not new_content.endswith('\n') else new_content
                elif line_no == len(lines):
                    # Add new line
                    lines.append(new_content + '\n' if not new_content.endswith('\n') else new_content)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Invalidate cache
            if self.cache_enabled and self.cache:
                abs_path = os.path.abspath(file_path)
                if abs_path in self.cache.cache_index:
                    del self.cache.cache_index[abs_path]
                    self.cache.save_cache_index()
            
            print(f"File updated successfully: {file_path}")
            return True
            
        except Exception as e:
            print(f"Failed to update file ({file_path}): {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: File path
            
        Returns:
            Whether deletion was successful
        """
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"File deleted successfully: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Directory deleted successfully: {file_path}")
            else:
                print(f"File/directory not found: {file_path}")
                return False
            
            # Remove from cache
            if self.cache_enabled and self.cache:
                abs_path = os.path.abspath(file_path)
                if abs_path in self.cache.cache_index:
                    cached_file = self.cache.cache_index[abs_path]
                    cache_file_path = self.cache.cache_dir / f"{cached_file.hash_md5}.txt"
                    cache_file_path.unlink(missing_ok=True)
                    del self.cache.cache_index[abs_path]
                    self.cache.save_cache_index()
            
            return True
            
        except Exception as e:
            print(f"Failed to delete file ({file_path}): {e}")
            return False
    
    def search_files(self, directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
        """
        Search for files.
        
        Args:
            directory: Directory to search
            pattern: File pattern (glob pattern)
            recursive: Whether to include subdirectories
            
        Returns:
            List of file paths
        """
        try:
            results = []
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return results
            
            if recursive:
                for file_path in dir_path.rglob(pattern):
                    if file_path.is_file():
                        results.append(str(file_path))
            else:
                for file_path in dir_path.glob(pattern):
                    if file_path.is_file():
                        results.append(str(file_path))
            
            return sorted(results)
            
        except Exception as e:
            print(f"Failed to search files: {e}")
            return []
    
    def search_in_files(self, directory: str, search_text: str, file_pattern: str = "*", recursive: bool = True) -> Dict[str, List[tuple]]:
        """
        Search for text within files.
        
        Args:
            directory: Directory to search
            search_text: Text to search for
            file_pattern: File pattern
            recursive: Whether to include subdirectories
            
        Returns:
            Dictionary of {file_path: [(line_number, line_content), ...]}
        """
        try:
            results = {}
            files = self.search_files(directory, file_pattern, recursive)
            
            for file_path in files:
                try:
                    content = self.read_file(file_path)
                    lines = content.split('\n')
                    
                    matches = []
                    for i, line in enumerate(lines):
                        if search_text.lower() in line.lower():
                            matches.append((i + 1, line.strip()))
                    
                    if matches:
                        results[file_path] = matches
                        
                except Exception:
                    continue  # Skip unreadable files
            
            return results
            
        except Exception as e:
            print(f"Failed to search in files: {e}")
            return {}
    
    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """Get file information."""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            stat = path.stat()
            
            file_info = FileInfo(
                path=str(path.absolute()),
                size=stat.st_size,
                modified_time=stat.st_mtime,
                created_time=stat.st_ctime,
                is_directory=path.is_dir()
            )
            
            # Add hash and preview for files
            if not file_info.is_directory and stat.st_size < 1024 * 1024:  # Less than 1MB
                try:
                    file_info.hash_md5 = self.cache._get_file_hash(file_path) if self.cache else ""
                    content = self.read_file(file_path)
                    file_info.content_preview = content[:200] + "..." if len(content) > 200 else content
                except Exception:
                    pass
            
            return file_info
            
        except Exception as e:
            print(f"Failed to get file info ({file_path}): {e}")
            return None
    
    def list_directory(self, directory: str, show_hidden: bool = False) -> List[FileInfo]:
        """List directory contents."""
        try:
            results = []
            dir_path = Path(directory)
            
            if not dir_path.exists() or not dir_path.is_dir():
                return results
            
            for item in dir_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                
                info = self.get_file_info(str(item))
                if info:
                    results.append(info)
            
            return sorted(results, key=lambda x: (not x.is_directory, x.path.lower()))
            
        except Exception as e:
            print(f"Failed to list directory ({directory}): {e}")
            return []


# Global file manager instance
_file_manager: Optional[FileManager] = None


def get_file_manager() -> FileManager:
    """Return the global file manager."""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileManager()
    return _file_manager