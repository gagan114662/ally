"""
Hashing utilities for deterministic tool execution and caching
"""

import hashlib
import json
import xxhash
from typing import Any, Dict
import inspect
from pathlib import Path


def hash_inputs(inputs: Dict[str, Any], algorithm: str = "xxh64") -> str:
    """
    Create deterministic hash of tool inputs
    
    Args:
        inputs: Input dictionary to hash
        algorithm: Hashing algorithm ('xxh64', 'sha256', 'md5')
        
    Returns:
        Hex string hash of inputs
    """
    # Serialize inputs in deterministic order
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    
    if algorithm == "xxh64":
        return xxhash.xxh64(serialized.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(serialized.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(serialized.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def hash_code(func_or_file, algorithm: str = "xxh64") -> str:
    """
    Create hash of function source code or file contents
    
    Args:
        func_or_file: Function object or file path
        algorithm: Hashing algorithm
        
    Returns:
        Hex string hash of code
    """
    if callable(func_or_file):
        # Hash function source code
        try:
            source = inspect.getsource(func_or_file)
        except OSError:
            # Built-in function, use name as proxy
            source = func_or_file.__name__
    elif isinstance(func_or_file, (str, Path)):
        # Hash file contents
        with open(func_or_file, 'r') as f:
            source = f.read()
    else:
        source = str(func_or_file)
    
    if algorithm == "xxh64":
        return xxhash.xxh64(source.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(source.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(source.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def content_hash(data: bytes, algorithm: str = "xxh64") -> str:
    """
    Create hash of binary content
    
    Args:
        data: Binary data to hash
        algorithm: Hashing algorithm
        
    Returns:
        Hex string hash
    """
    if algorithm == "xxh64":
        return xxhash.xxh64(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest() 
    elif algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def create_cache_key(tool_name: str, inputs: Dict[str, Any], code_hash: str = None) -> str:
    """
    Create cache key for tool results
    
    Args:
        tool_name: Name of the tool
        inputs: Tool inputs
        code_hash: Optional hash of tool code
        
    Returns:
        Cache key string
    """
    inputs_hash = hash_inputs(inputs)
    
    if code_hash:
        combined = f"{tool_name}:{inputs_hash}:{code_hash}"
    else:
        combined = f"{tool_name}:{inputs_hash}"
    
    return hash_inputs({"key": combined}, algorithm="sha256")[:16]