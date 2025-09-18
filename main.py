from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional
import json
import os
import time
from datetime import datetime

from database import get_db, create_tables, get_db_with_retry, execute_with_retry
from models import Question, Category, Codex, FileProcessingJob
from data_parser import DataParser
from vector_db import vector_db
from config import Config
from logging_config import logger
from utils import validate_file_upload, validate_question, ensure_directory_exists, log_api_call
from celery_tasks import process_file, process_question

# Create FastAPI app
app = FastAPI(
    title="Llama-based Knowledge Retrieval API",
    description="API for querying knowledge base using Llama model with vector search",
    version="1.0.0"
)

# Add middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    log_api_call(request.url.path, request.method, response.status_code, process_time)
    return response

# Ensure database directory exists
os.makedirs("/app/data", exist_ok=True)

# Create database tables
create_tables()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question_id: int
    status: str
    message: str

class StatusResponse(BaseModel):
    question_id: int
    status: str
    progress: int
    step: str
    message: str
    is_ready: bool
    created_at: str
    updated_at: str

class AnswerResponse(BaseModel):
    question_id: int
    question: str
    answer: str
    status: str

class FileUploadResponse(BaseModel):
    job_id: int
    status: str
    message: str
    filename: str
    file_size_mb: float

class FileProcessingStatusResponse(BaseModel):
    job_id: int
    filename: str
    status: str
    progress: int
    current_step: str
    message: str
    categories_count: int
    codexes_count: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload file and start background processing"""
    try:
        logger.info(f"File upload started: {file.filename}")
        
        # Validate file
        file_info = validate_file_upload(file)
        logger.info(f"File validation passed: {file_info}")
        
        # Ensure uploads directory exists
        ensure_directory_exists("uploads")
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        file_size = file_info['size']
        
        if file_size > 50 * 1024 * 1024:  # If file is larger than 50MB
            logger.info(f"Large file detected ({file_size / (1024*1024):.1f}MB), using chunked upload...")
            await _save_large_file(file, file_path)
        else:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
        
        logger.info(f"File saved: {file_path} ({file_size} bytes)")
        
        # Determine file type
        if file.filename.endswith('.sql'):
            file_type = 'sql'
        elif file.filename.endswith('.csv'):
            file_type = 'csv'
        else:
            # Clean up file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create file processing job record with retry logic
        def create_job_operation(db_session):
            job = FileProcessingJob(
                filename=file.filename,
                file_path=file_path,
                file_type=file_type,
                file_size=file_size,
                status="pending"
            )
            db_session.add(job)
            db_session.commit()
            db_session.refresh(job)
            return job
        
        job = execute_with_retry(db, create_job_operation)
        
        logger.info(f"Created file processing job {job.id}")
        
        # Start background processing
        task = process_file.delay(file_path, file_type, job.id)
        
        # Update job with Celery task ID with retry logic
        def update_job_operation(db_session):
            job.celery_task_id = task.id
            db_session.commit()
            return job
        
        execute_with_retry(db, update_job_operation)
        
        logger.info(f"Started background processing for job {job.id}")
        
        return FileUploadResponse(
            job_id=job.id,
            status="pending",
            message="File uploaded successfully. Processing started in background.",
            filename=file.filename,
            file_size_mb=file_size / (1024 * 1024)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def _save_large_file(file: UploadFile, file_path: str):
    """Save large file in chunks to avoid memory issues"""
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(file_path, "wb") as buffer:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            buffer.write(chunk)

@app.post("/question", response_model=QuestionResponse)
async def submit_question(question_req: QuestionRequest, db: Session = Depends(get_db)):
    """Submit a question for processing"""
    try:
        logger.info(f"Question submission started: {question_req.question[:50]}...")
        
        # Validate question
        validated_question = validate_question(question_req.question)
        
        # Create question record
        question = Question(
            question_text=validated_question,
            status="pending"
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        
        logger.info(f"Question created with ID: {question.id}")
        
        # Start background processing with Celery
        task = process_question.delay(question.id)
        
        logger.info(f"Background processing started for question {question.id}")
        
        return QuestionResponse(
            question_id=question.id,
            status="pending",
            message="Question submitted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting question: {str(e)}")

@app.get("/question/{question_id}/status", response_model=StatusResponse)
async def get_question_status(question_id: int, db: Session = Depends(get_db)):
    """Get the status of a question"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Parse processing steps
    steps = {}
    if question.processing_steps:
        try:
            steps = json.loads(question.processing_steps)
        except:
            steps = {"step": "unknown", "progress": 0, "message": "Processing..."}
    
    return StatusResponse(
        question_id=question.id,
        status=question.status,
        progress=steps.get("progress", 0),
        step=steps.get("step", "unknown"),
        message=steps.get("message", "Processing..."),
        is_ready=question.is_ready,
        created_at=question.created_at.isoformat(),
        updated_at=question.updated_at.isoformat()
    )

@app.get("/question/{question_id}/response", response_model=AnswerResponse)
async def get_question_response(question_id: int, db: Session = Depends(get_db)):
    """Get the response for a completed question"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    if not question.is_ready:
        raise HTTPException(status_code=400, detail="Question is not ready yet")
    
    return AnswerResponse(
        question_id=question.id,
        question=question.question_text,
        answer=question.answer or "No answer available",
        status=question.status
    )

@app.get("/file-processing/{job_id}/status", response_model=FileProcessingStatusResponse)
async def get_file_processing_status(job_id: int, db: Session = Depends(get_db)):
    """Get the status of a file processing job"""
    job = db.query(FileProcessingJob).filter(FileProcessingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="File processing job not found")
    
    return FileProcessingStatusResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step or "unknown",
        message=job.message or "Processing...",
        categories_count=job.categories_count,
        codexes_count=job.codexes_count,
        created_at=job.created_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message
    )

@app.get("/file-processing/jobs")
async def list_file_processing_jobs(db: Session = Depends(get_db), limit: int = 10, offset: int = 0):
    """List file processing jobs"""
    jobs = db.query(FileProcessingJob).order_by(FileProcessingJob.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "jobs": [
            {
                "job_id": job.id,
                "filename": job.filename,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
            for job in jobs
        ],
        "total": db.query(FileProcessingJob).count()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        from sqlalchemy import text
        db = next(get_db())
        db.execute(text("SELECT 1"))
        db.close()
        
        # Check vector database
        vector_stats = vector_db.get_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "vector_db": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        db = next(get_db())
        
        # Get question statistics
        total_questions = db.query(Question).count()
        pending_questions = db.query(Question).filter(Question.status == "pending").count()
        completed_questions = db.query(Question).filter(Question.status == "completed").count()
        failed_questions = db.query(Question).filter(Question.status == "failed").count()
        
        # Get category and codex counts
        total_categories = db.query(Category).count()
        total_codexes = db.query(Codex).count()
        
        db.close()
        
        return {
            "api_status": "running",
            "timestamp": datetime.utcnow().isoformat(),
            "questions": {
                "total": total_questions,
                "pending": pending_questions,
                "completed": completed_questions,
                "failed": failed_questions
            },
            "knowledge_base": {
                "categories": total_categories,
                "codexes": total_codexes
            },
            "vector_db_stats": vector_db.get_stats()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
