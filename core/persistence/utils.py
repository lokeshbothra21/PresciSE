"""
Utility functions for file tracking and persistence.

Priority: Reliability > Speed
- Uses SHA256 hashing for accurate change detection
- Robust error handling
"""

import hashlib
import os
from pathlib import Path
from typing import Optional


def get_file_hash(filepath: str) -> str:
    """
    Compute SHA256 hash of file content.
    
    This is RELIABLE but slower than modification time.
    Detects any content change, even if mtime is restored.
    
    Args:
        filepath: Path to file
        
    Returns:
        SHA256 hash as hex string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    sha256 = hashlib.sha256()
    
    try:
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(65536), b""):  # 64KB chunks
                sha256.update(chunk)
        return sha256.hexdigest()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")


def get_file_mtime(filepath: str) -> float:
    """
    Get file modification timestamp.
    
    Args:
        filepath: Path to file
        
    Returns:
        Modification time as Unix timestamp
    """
    return os.path.getmtime(filepath)


def ensure_dir(dirpath: str) -> None:
    """
    Ensure directory exists, create if not.
    
    Args:
        dirpath: Directory path to create
    """
    Path(dirpath).mkdir(parents=True, exist_ok=True)


def file_exists(filepath: str) -> bool:
    """
    Check if file exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists
    """
    return os.path.exists(filepath)
