from celery_app import celery_app
from database import get_db_with_retry, execute_with_retry
from models import Question, FileProcessingJob
from data_parser import DataParser
from vector_db import vector_db
from llm_service import llm_service
from logging_config import logger
import json
import os
from datetime import datetime

@celery_app.task(bind=True, name='celery_tasks.process_file')
def process_file(self, file_path: str, file_type: str, job_id: int):
    """Process uploaded file in background with progress tracking"""
    db = get_db_with_retry(max_retries=5, retry_delay=2)
    try:
        # Get job record with retry logic for database locks
        def get_job_operation(db_session):
            job = db_session.query(FileProcessingJob).filter(FileProcessingJob.id == job_id).first()
            if not job:
                logger.error(f"File processing job {job_id} not found")
                return None
            return job
        
        job = execute_with_retry(db, get_job_operation)
        if not job:
            return {"status": "failed", "message": "Job not found"}
        
        # Update job status to processing
        job.status = "processing"
        job.started_at = datetime.utcnow()
        
        # Update progress: File validation
        self.update_state(
            state='PROGRESS',
            meta={'step': 'validation', 'progress': 10, 'message': 'File validation completed'}
        )
        job.progress = 10
        job.current_step = 'validation'
        job.message = 'File validation completed'
        
        # Initialize parser
        parser = DataParser(db)
        
        # Update progress: File parsing
        self.update_state(
            state='PROGRESS',
            meta={'step': 'parsing', 'progress': 20, 'message': 'Parsing file...'}
        )
        job.progress = 20
        job.current_step = 'parsing'
        job.message = 'Parsing file...'
        
        # Commit all initial updates at once
        db.commit()
        
        # Parse file based on type
        if file_type == 'sql':
            logger.info(f"Processing SQL dump: {file_path}")
            categories, codexes = parser.parse_sql_dump(file_path)
        elif file_type == 'csv':
            logger.info(f"Processing CSV file: {file_path}")
            categories, codexes = parser.parse_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Update progress: Data validation
        self.update_state(
            state='PROGRESS',
            meta={'step': 'validation', 'progress': 40, 'message': f'Parsed {len(categories)} categories and {len(codexes)} codexes'}
        )
        job.progress = 40
        job.current_step = 'validation'
        job.message = f'Parsed {len(categories)} categories and {len(codexes)} codexes'
        job.categories_count = len(categories)
        job.codexes_count = len(codexes)
        
        # Update progress: Database loading
        self.update_state(
            state='PROGRESS',
            meta={'step': 'database', 'progress': 60, 'message': 'Loading data to database...'}
        )
        job.progress = 60
        job.current_step = 'database'
        job.message = 'Loading data to database...'
        
        # Commit validation and loading updates at once
        db.commit()
        
        # Load data to database
        parser.load_data_to_db(categories, codexes)
        
        # Update progress: Vector database
        self.update_state(
            state='PROGRESS',
            meta={'step': 'vector_db', 'progress': 80, 'message': 'Creating vector embeddings...'}
        )
        job.progress = 80
        job.current_step = 'vector_db'
        job.message = 'Creating vector embeddings...'
        
        # Update progress: Cleanup
        self.update_state(
            state='PROGRESS',
            meta={'step': 'cleanup', 'progress': 90, 'message': 'Cleaning up temporary files...'}
        )
        job.progress = 90
        job.current_step = 'cleanup'
        job.message = 'Cleaning up temporary files...'
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
        
        # Mark job as completed
        job.status = "completed"
        job.progress = 100
        job.current_step = 'completed'
        job.message = 'File processing completed successfully'
        job.completed_at = datetime.utcnow()
        
        # Commit all final updates at once
        db.commit()
        
        logger.info(f"File processing job {job_id} completed successfully")
        
        return {
            "status": "completed",
            "message": f"Successfully processed {len(categories)} categories and {len(codexes)} codexes",
            "categories_count": len(categories),
            "codexes_count": len(codexes)
        }
        
    except Exception as e:
        logger.error(f"Error in file processing job {job_id}: {str(e)}")
        
        # Update job status to failed with proper rollback handling
        if 'job' in locals():
            error_message = str(e)
            def update_failed_job(db_session):
                job.status = "failed"
                job.message = f"Processing failed: {error_message}"
                job.completed_at = datetime.utcnow()
                db_session.commit()
                return job
            
            try:
                execute_with_retry(db, update_failed_job)
            except Exception as commit_error:
                logger.error(f"Error updating job status: {commit_error}")
                try:
                    db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
        
        # Update Celery task state
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'step': 'error', 'progress': 0}
        )
        
        return {"status": "failed", "message": str(e)}
    
    finally:
        db.close()

@celery_app.task(bind=True, name='celery_tasks.process_question')
def process_question(self, question_id: int):
    """Process question in background with progress tracking"""
    db = get_db_with_retry(max_retries=5, retry_delay=2)
    try:
        def get_question_operation(db_session):
            question = db_session.query(Question).filter(Question.id == question_id).first()
            if not question:
                logger.error(f"Question {question_id} not found")
                return None
            return question
        
        question = execute_with_retry(db, get_question_operation)
        if not question:
            return {"status": "failed", "message": "Question not found"}
        
        # Update status to processing
        question.status = "processing"
        question.processing_steps = json.dumps({
            "step": "starting", 
            "progress": 10, 
            "message": "Qayta ishlash boshlandi"
        })
        db.commit()
        
        # Update Celery task state
        self.update_state(
            state='PROGRESS',
            meta={'step': 'starting', 'progress': 10, 'message': 'Qayta ishlash boshlandi'}
        )
        
        # Step 1: Vector search
        self.update_state(
            state='PROGRESS',
            meta={'step': 'vector_search', 'progress': 30, 'message': 'Vector bazasida qidirilmoqda...'}
        )
        
        question.processing_steps = json.dumps({
            "step": "vector_search", 
            "progress": 30, 
            "message": "Vector bazasida qidirilmoqda..."
        })
        db.commit()
        
        # Search for similar content
        search_results = vector_db.search(question.question_text, k=5)
        
        if not search_results:
            question.status = "completed"
            question.answer = "Kechirasiz, savolingizga mos ma'lumot topilmadi. Iltimos, boshqa savol bering."
            question.is_ready = True
            question.processing_steps = json.dumps({
                "step": "no_results", 
                "progress": 100, 
                "message": "Hech qanday mos ma'lumot topilmadi"
            })
            db.commit()
            
            self.update_state(
                state='SUCCESS',
                meta={'step': 'no_results', 'progress': 100, 'message': 'Hech qanday mos ma\'lumot topilmadi'}
            )
            return {"status": "completed", "message": "No results found"}
        
        # Step 2: Generate answer with LLM
        self.update_state(
            state='PROGRESS',
            meta={'step': 'llm_generation', 'progress': 60, 'message': 'LLM yordamida javob yaratilmoqda...'}
        )
        
        question.processing_steps = json.dumps({
            "step": "llm_generation", 
            "progress": 60, 
            "message": "LLM yordamida javob yaratilmoqda..."
        })
        db.commit()
        
        # Generate answer using LLM
        answer = llm_service.generate_answer(question.question_text, search_results)
        
        # Step 3: Complete processing
        self.update_state(
            state='PROGRESS',
            meta={'step': 'finalizing', 'progress': 90, 'message': 'Javob yakunlanmoqda...'}
        )
        
        question.processing_steps = json.dumps({
            "step": "finalizing", 
            "progress": 90, 
            "message": "Javob yakunlanmoqda..."
        })
        db.commit()
        
        # Save answer
        question.answer = answer
        question.is_ready = True
        question.status = "completed"
        question.processing_steps = json.dumps({
            "step": "completed", 
            "progress": 100, 
            "message": "Javob tayyor!"
        })
        db.commit()
        
        logger.info(f"Question {question_id} processed successfully")
        
        self.update_state(
            state='SUCCESS',
            meta={'step': 'completed', 'progress': 100, 'message': 'Javob tayyor!'}
        )
        
        return {"status": "completed", "answer": answer}
        
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {str(e)}")
        
        if 'question' in locals():
            try:
                question.status = "failed"
                question.processing_steps = json.dumps({
                    "step": "error", 
                    "progress": 0, 
                    "message": f"Xatolik: {str(e)}"
                })
                db.commit()
            except Exception as commit_error:
                logger.error(f"Error updating question status: {commit_error}")
                try:
                    db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
        
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'step': 'error', 'progress': 0}
        )
        
        return {"status": "failed", "message": str(e)}
    
    finally:
        db.close()
