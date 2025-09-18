import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////app/data/knowledge_retrieval.db")
    
    # Vector Database
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/app/vector_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # LLM - Using a non-gated model that works well
    LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
    
    # Llama specific settings
    LLAMA_TEMPERATURE = float(os.getenv("LLAMA_TEMPERATURE", "0.7"))
    LLAMA_TOP_P = float(os.getenv("LLAMA_TOP_P", "0.9"))
    LLAMA_MAX_NEW_TOKENS = int(os.getenv("LLAMA_MAX_NEW_TOKENS", "256"))
    
    # Model cache paths (for volume mounting)
    TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "/app/models/transformers_cache")
    HF_HOME = os.getenv("HF_HOME", "/app/models/huggingface")
    TORCH_HOME = os.getenv("TORCH_HOME", "/app/models/torch")
    
    # Redis for Celery
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Chunking
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
    
    # Large file handling
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))  # 1GB max
    CSV_CHUNK_SIZE = int(os.getenv("CSV_CHUNK_SIZE", "10000"))  # Process 10k rows at a time
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))  # Database batch size (increased for speed)