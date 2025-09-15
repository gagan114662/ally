"""
Hashing utilities for deterministic tool execution and caching
"""

import hashlib
import json
from typing import Any, Dict
import inspect
from pathlib import Path

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False


def hash_inputs(inputs: Dict[str, Any], algorithm: str = "sha256") -> str:
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
    
    if algorithm == "xxh64" and HAS_XXHASH:
        return xxhash.xxh64(serialized.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(serialized.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(serialized.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(serialized.encode()).hexdigest()
    else:
        # Fallback to sha256 if xxh64 requested but not available
        if algorithm == "xxh64":
            return hashlib.sha256(serialized.encode()).hexdigest()
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def hash_code(func_or_file, algorithm: str = "sha256") -> str:
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
    
    if algorithm == "xxh64" and HAS_XXHASH:
        return xxhash.xxh64(source.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(source.encode()).hexdigest()
    elif algorithm == "md5":
        return hashlib.md5(source.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(source.encode()).hexdigest()
    else:
        # Fallback to sha256 if xxh64 requested but not available
        if algorithm == "xxh64":
            return hashlib.sha256(source.encode()).hexdigest()
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def content_hash(data: bytes, algorithm: str = "sha256") -> str:
    """
    Create hash of binary content
    
    Args:
        data: Binary data to hash
        algorithm: Hashing algorithm
        
    Returns:
        Hex string hash
    """
    if algorithm == "xxh64" and HAS_XXHASH:
        return xxhash.xxh64(data).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data).hexdigest() 
    elif algorithm == "md5":
        return hashlib.md5(data).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data).hexdigest()
    else:
        # Fallback to sha256 if xxh64 requested but not available
        if algorithm == "xxh64":
            return hashlib.sha256(data).hexdigest()
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def hash_payload(payload: Any, algorithm: str = "sha1") -> str:
    """
    Create SHA-1 hash of raw payload data for receipts
    
    Args:
        payload: Raw payload data (string, bytes, dict, etc.)
        algorithm: Hashing algorithm (default: sha1 for receipts)
        
    Returns:
        Hex string hash of payload
    """
    if isinstance(payload, dict):
        # For dictionaries, serialize deterministically
        serialized = json.dumps(payload, sort_keys=True, default=str)
        data = serialized.encode()
    elif isinstance(payload, str):
        data = payload.encode()
    elif isinstance(payload, bytes):
        data = payload
    else:
        # Convert other types to string first
        data = str(payload).encode()
    
    if algorithm == "sha1":
        return hashlib.sha1(data).hexdigest()
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