"""
Utility functions for the Knowledge Retrieval API
"""
import os
import re
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from loguru import logger

def validate_file_upload(file) -> Dict[str, Any]:
    """Validate uploaded file"""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Import config here to avoid circular imports
    from config import Config
    
    # Check file size (configurable limit)
    max_size = Config.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
    file_size = getattr(file, 'size', 0)
    
    if file_size > max_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is {Config.MAX_FILE_SIZE_MB}MB. Your file is {file_size / (1024*1024):.1f}MB"
        )
    
    # Check file extension
    allowed_extensions = ['.sql', '.csv']
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    return {
        "filename": file.filename,
        "extension": file_extension,
        "size": file_size
    }

def sanitize_text(text: str) -> str:
    """Sanitize text input"""
    if not text:
        return ""
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)
    
    # Limit length
    max_length = 10000
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text.strip()

def validate_question(question: str) -> str:
    """Validate and sanitize question input"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    question = sanitize_text(question)
    
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters long")
    
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long. Maximum 1000 characters")
    
    return question

def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create required directory")

def safe_get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """Safely get environment variable with validation"""
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value

def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message for user display"""
    error_msg = str(error)
    
    # Don't expose internal details to users
    if "sqlite3" in error_msg.lower():
        return "Database error occurred. Please try again later."
    elif "connection" in error_msg.lower():
        return "Connection error. Please check your network and try again."
    elif "memory" in error_msg.lower():
        return "Insufficient memory. Please try with a smaller request."
    else:
        return f"An error occurred: {error_msg}"

def log_api_call(endpoint: str, method: str, status_code: int, duration: float = None):
    """Log API call details"""
    duration_str = f" ({duration:.3f}s)" if duration else ""
    logger.info(f"{method} {endpoint} - {status_code}{duration_str}")

def validate_database_connection(db_session) -> bool:
    """Validate database connection"""
    try:
        db_session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection validation failed: {e}")
        return False
