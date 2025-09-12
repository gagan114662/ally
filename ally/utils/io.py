"""
I/O utilities for Ally tools
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union
import orjson


def safe_read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely read JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary content or empty dict on error
    """
    try:
        with open(file_path, 'rb') as f:
            return orjson.loads(f.read())
    except Exception:
        return {}


def safe_write_json(data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
    """
    Safely write JSON file with error handling
    
    Args:
        data: Data to write
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return True
    except Exception:
        return False


def safe_read_text(file_path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Safely read text file
    
    Args:
        file_path: Path to text file
        encoding: File encoding
        
    Returns:
        File content or empty string on error
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception:
        return ""


def safe_write_text(content: str, file_path: Union[str, Path], encoding: str = 'utf-8') -> bool:
    """
    Safely write text file
    
    Args:
        content: Text content to write
        file_path: Output file path
        encoding: File encoding
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception:
        return False


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if needed
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def list_files(dir_path: Union[str, Path], pattern: str = "*", recursive: bool = False) -> list:
    """
    List files matching pattern
    
    Args:
        dir_path: Directory to search
        pattern: Glob pattern
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    try:
        path = Path(dir_path)
        if recursive:
            return list(path.rglob(pattern))
        else:
            return list(path.glob(pattern))
    except Exception:
        return []