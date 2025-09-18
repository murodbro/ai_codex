from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from models import Base
from config import Config
import time
import logging

logger = logging.getLogger(__name__)

# Create database engine with optimized settings for bulk operations and concurrency
if "sqlite" in Config.DATABASE_URL:
    connect_args = {
        "check_same_thread": False,
        "timeout": 60,  # Increase timeout for large operations
    }
    # Add SQLite performance optimizations with connection pooling
    engine = create_engine(
        Config.DATABASE_URL, 
        connect_args={
            **connect_args,
            "isolation_level": None  # Disable autocommit for better control
        },
        poolclass=StaticPool,
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections every hour
        echo=False  # Disable SQL logging for performance
    )
else:
    engine = create_engine(Config.DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables with optimized settings"""
    Base.metadata.create_all(bind=engine)
    
    # Optimize SQLite for bulk operations
    if "sqlite" in Config.DATABASE_URL:
        from sqlalchemy import text
        with engine.connect() as conn:
            # Enable WAL mode for better concurrency
            conn.execute(text("PRAGMA journal_mode=WAL"))
            # Increase cache size
            conn.execute(text("PRAGMA cache_size=10000"))
            # Disable synchronous writes for bulk operations (faster but less safe)
            conn.execute(text("PRAGMA synchronous=OFF"))
            # Increase page size
            conn.execute(text("PRAGMA page_size=4096"))
            # Enable memory-mapped I/O
            conn.execute(text("PRAGMA mmap_size=268435456"))  # 256MB
            conn.commit()

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_with_retry(max_retries=3, retry_delay=1):
    """Get database session with retry logic for handling locks"""
    for attempt in range(max_retries):
        try:
            db = SessionLocal()
            # Test the connection
            db.execute(text("SELECT 1"))
            return db
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database locked, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                logger.error(f"Database connection failed after {attempt + 1} attempts: {str(e)}")
                raise e
    return None

def execute_with_retry(db, operation, max_retries=3, retry_delay=1):
    """Execute database operation with retry logic"""
    for attempt in range(max_retries):
        try:
            return operation(db)
        except Exception as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database operation failed, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                logger.error(f"Database operation failed after {attempt + 1} attempts: {str(e)}")
                raise e
    return None
