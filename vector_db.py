import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from config import Config
import json
import time
from datetime import datetime, timedelta

class VectorDatabase:
    def __init__(self):
        # Check for GPU availability
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {self.device}")
        if self.device == 'cpu':
            print("⚠️  WARNING: Running on CPU. GPU acceleration would be much faster!")
            print("   Consider using a machine with CUDA-capable GPU for faster processing.")
        
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=self.device)
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.index = None
        self.chunk_metadata = []
        self.db_path = Config.VECTOR_DB_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Load existing index if available
        self.load_index()
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        metadata_path = os.path.join(self.db_path, "metadata.json")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            print(f"Loaded existing index with {self.index.ntotal} vectors")
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            self.chunk_metadata = []
            print("Created new FAISS index")
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        metadata_path = os.path.join(self.db_path, "metadata.json")
        
        faiss.write_index(self.index, index_path)
        with open(metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f)
        print(f"Saved index with {self.index.ntotal} vectors")
    
    def add_documents(self, documents: List[str], metadata: List[dict] = None):
        """Add documents to the vector database with detailed progress logging"""
        if not documents:
            return
        
        # Process in smaller batches for more frequent progress updates
        batch_size = min(100, len(documents))  # Reduced from 1000 to 100 for better progress tracking
        total_docs = len(documents)
        start_time = time.time()
        
        print(f"Processing {total_docs} documents in batches of {batch_size}...")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_end = min(i + batch_size, total_docs)
            batch_start_time = time.time()
            
            print(f"  Processing batch {i//batch_size + 1}: documents {i+1}-{batch_end}")
            
            # Log each document in the batch
            for j, doc in enumerate(batch_docs):
                doc_index = i + j + 1
                doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"    Processing document {doc_index}/{total_docs}: {doc_preview}")
            
            # Generate embeddings for this batch
            print(f"    Generating embeddings for batch {i//batch_size + 1}...")
            print(f"    Processing {len(batch_docs)} documents through neural network...")
            
            # Time the embedding generation
            embedding_start_time = time.time()
            
            # Enable progress bar and optimize settings
            embeddings = self.embedding_model.encode(
                batch_docs, 
                convert_to_tensor=False, 
                show_progress_bar=True,  # Enable progress bar
                batch_size=64 if self.device == 'cuda' else 16,  # Larger batches for GPU, smaller for CPU
                device=self.device  # Use the detected device
            )
            
            embedding_time = time.time() - embedding_start_time
            print(f"    ⚡ Embeddings generated in {embedding_time:.2f}s ({embedding_time/len(batch_docs):.3f}s per document)")
            
            # Normalize embeddings for cosine similarity
            print(f"    Normalizing embeddings for batch {i//batch_size + 1}...")
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to index
            print(f"    Adding {len(batch_docs)} vectors to FAISS index...")
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata for this batch
            if metadata is None:
                batch_metadata = [{"content": doc, "index": i + j} for j, doc in enumerate(batch_docs)]
            else:
                batch_metadata = metadata[i:i + batch_size]
            
            self.chunk_metadata.extend(batch_metadata)
            
            batch_time = time.time() - batch_start_time
            elapsed_time = time.time() - start_time
            progress_percent = (batch_end / total_docs) * 100
            
            # Estimate remaining time
            if batch_end > 0:
                avg_time_per_doc = elapsed_time / batch_end
                remaining_docs = total_docs - batch_end
                estimated_remaining = avg_time_per_doc * remaining_docs
                eta = datetime.now() + timedelta(seconds=estimated_remaining)
                print(f"    ✅ Batch {i//batch_size + 1} completed in {batch_time:.2f}s")
                print(f"    📊 Progress: {progress_percent:.1f}% ({batch_end}/{total_docs})")
                print(f"    ⏱️  Elapsed: {elapsed_time:.2f}s, ETA: {eta.strftime('%H:%M:%S')}")
            else:
                print(f"    ✅ Batch {i//batch_size + 1} completed in {batch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"✅ Added {total_docs} documents to vector database in {total_time:.2f}s")
        print(f"Average time per document: {total_time/total_docs:.3f}s")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx]
                results.append((metadata.get('content', ''), float(score), metadata))
        
        return results
    
    def get_stats(self):
        """Get database statistics"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "model": Config.EMBEDDING_MODEL
        }

# Global vector database instance
vector_db = VectorDatabase()
