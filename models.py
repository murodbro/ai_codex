from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Category(Base):
    __tablename__ = "category"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    parent_id = Column(Integer, ForeignKey("category.id"), nullable=True)
    
    # Relationships
    parent = relationship("Category", remote_side=[id], back_populates="children")
    children = relationship("Category", back_populates="parent")
    codexes = relationship("Codex", back_populates="category")

class Codex(Base):
    __tablename__ = "codex"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category_id = Column(Integer, ForeignKey("category.id"), nullable=True)
    
    # Relationships
    category = relationship("Category", back_populates="codexes")

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    answer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processing_steps = Column(Text, nullable=True)  # JSON string of processing steps
    is_ready = Column(Boolean, default=False)

class VectorChunk(Base):
    __tablename__ = "vector_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    codex_id = Column(Integer, ForeignKey("codex.id"), nullable=True)
    category_id = Column(Integer, ForeignKey("category.id"), nullable=True)
    vector_id = Column(String, nullable=False)  # ID in vector database
    created_at = Column(DateTime, default=datetime.utcnow)

class FileProcessingJob(Base):
    __tablename__ = "file_processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # sql, csv
    file_size = Column(Integer, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    current_step = Column(String, nullable=True)
    message = Column(Text, nullable=True)
    categories_count = Column(Integer, default=0)
    codexes_count = Column(Integer, default=0)
    celery_task_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)