import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from config import Config
import json

class VectorDatabase:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
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
        """Add documents to the vector database with optimized batch processing"""
        if not documents:
            return
        
        # Process in optimized batches for better performance
        batch_size = min(1000, len(documents))  # Process up to 1000 at once
        total_docs = len(documents)
        
        print(f"Processing {total_docs} documents in batches of {batch_size}...")
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_end = min(i + batch_size, total_docs)
            
            print(f"  Processing batch {i//batch_size + 1}: documents {i+1}-{batch_end}")
            
            # Generate embeddings for this batch
            embeddings = self.embedding_model.encode(batch_docs, convert_to_tensor=False, show_progress_bar=False)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata for this batch
            if metadata is None:
                batch_metadata = [{"content": doc, "index": i + j} for j, doc in enumerate(batch_docs)]
            else:
                batch_metadata = metadata[i:i + batch_size]
            
            self.chunk_metadata.extend(batch_metadata)
            
            print(f"    Added {len(batch_docs)} documents to vector database")
        
        print(f"✅ Added {total_docs} documents to vector database")
    
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
