import pandas as pd
import re
import os
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from models import Category, Codex
from vector_db import vector_db
from config import Config

class DataParser:
    def __init__(self, db: Session):
        self.db = db
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def parse_sql_dump(self, sql_file_path: str):
        """Parse SQL dump file and extract data"""
        print(f"Parsing SQL dump: {sql_file_path}")
        
        with open(sql_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Extract INSERT statements for categories
        category_pattern = r"INSERT INTO category.*?VALUES\s*\((.*?)\);"
        category_matches = re.findall(category_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Extract INSERT statements for codexes
        codex_pattern = r"INSERT INTO codex.*?VALUES\s*\((.*?)\);"
        codex_matches = re.findall(codex_pattern, content, re.DOTALL | re.IGNORECASE)
        
        # Parse categories
        categories = self._parse_insert_values(category_matches, ['id', 'name', 'parent_id'])
        # Parse codexes
        codexes = self._parse_insert_values(codex_matches, ['id', 'name', 'content', 'category_id'])
        
        return categories, codexes
    
    def parse_csv(self, csv_file_path: str):
        """Parse CSV file and extract data with chunked processing for large files"""
        print(f"Parsing CSV file: {csv_file_path}")
        
        # Get file size to determine processing strategy
        file_size = os.path.getsize(csv_file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        
        # For large files, use chunked processing
        if file_size_mb > 50:  # If file is larger than 50MB, use chunked processing
            return self._parse_csv_chunked(csv_file_path)
        else:
            return self._parse_csv_small(csv_file_path)
    
    def _parse_csv_small(self, csv_file_path: str):
        """Parse small CSV files (<= 50MB) in memory"""
        try:
            # Try different encodings for international text
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    print(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read CSV with any supported encoding")
                
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return [], []
        
        categories = []
        codexes = []
        
        print(f"CSV columns: {list(df.columns)}")
        print(f"CSV shape: {df.shape}")
        
        # Process the dataframe
        categories, codexes = self._process_dataframe(df)
        
        print(f"Parsed {len(categories)} categories and {len(codexes)} codexes")
        return categories, codexes
    
    def _parse_csv_chunked(self, csv_file_path: str):
        """Parse large CSV files using chunked processing"""
        print("Using chunked processing for large file...")
        
        categories = []
        codexes = []
        chunk_size = Config.CSV_CHUNK_SIZE
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1']
            encoding = None
            
            for enc in encodings:
                try:
                    # Test encoding by reading first few lines
                    pd.read_csv(csv_file_path, encoding=enc, nrows=5)
                    encoding = enc
                    print(f"Successfully detected encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if encoding is None:
                raise Exception("Could not read CSV with any supported encoding")
            
            # Process file in chunks
            chunk_count = 0
            for chunk_df in pd.read_csv(csv_file_path, encoding=encoding, chunksize=chunk_size):
                chunk_count += 1
                print(f"Processing chunk {chunk_count} ({len(chunk_df)} rows)...")
                
                # Process this chunk
                chunk_categories, chunk_codexes = self._process_dataframe(chunk_df)
                categories.extend(chunk_categories)
                codexes.extend(chunk_codexes)
                
                # Memory management - clear chunk from memory
                del chunk_df
                del chunk_categories
                del chunk_codexes
                
                # Force garbage collection every 10 chunks
                if chunk_count % 10 == 0:
                    import gc
                    gc.collect()
            
            print(f"Processed {chunk_count} chunks")
            print(f"Total parsed: {len(categories)} categories and {len(codexes)} codexes")
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return [], []
        
        return categories, codexes
    
    def _process_dataframe(self, df):
        """Process a dataframe to extract categories and codexes"""
        categories = []
        codexes = []
        
        # Check if it's a combined file with both categories and codexes
        has_parent_id = 'parent_id' in df.columns
        has_content = 'content' in df.columns
        has_category_id = 'category_id' in df.columns
        
        if has_parent_id or (has_content and has_category_id):
            # Combined file with both categories and codexes
            for _, row in df.iterrows():
                # Check if this row has content (making it a codex entry)
                content_value = row.get('content', '')
                has_meaningful_content = pd.notna(content_value) and str(content_value).strip() != ''
                
                if has_meaningful_content:
                    # This is a codex entry
                    codexes.append({
                        'id': self._safe_get_value(row, 'id'),
                        'name': self._safe_get_value(row, 'name', ''),
                        'content': str(content_value).strip(),
                        'category_id': self._safe_get_value(row, 'category_id')
                    })
                else:
                    # This is a category entry (no content or empty content)
                    name_value = self._safe_get_value(row, 'name', '')
                    if name_value:  # Only add if name is not empty
                        categories.append({
                            'id': self._safe_get_value(row, 'id'),
                            'name': name_value,
                            'parent_id': self._safe_get_value(row, 'parent_id')
                        })
        else:
            # Assume it's a codex-only file
            for _, row in df.iterrows():
                content_value = self._safe_get_value(row, 'content', '')
                if content_value:  # Only add if content is not empty
                    codexes.append({
                        'id': self._safe_get_value(row, 'id'),
                        'name': self._safe_get_value(row, 'name', ''),
                        'content': content_value,
                        'category_id': self._safe_get_value(row, 'category_id')
                    })
        
        return categories, codexes
    
    def _safe_get_value(self, row, column, default=None):
        """Safely get value from pandas row, handling NaN and None values"""
        try:
            value = row.get(column, default)
            if pd.isna(value):
                return default
            return value
        except Exception:
            return default
    
    def _parse_insert_values(self, matches: List[str], columns: List[str]) -> List[Dict[str, Any]]:
        """Parse INSERT VALUES and return list of dictionaries"""
        results = []
        
        for match in matches:
            # Split by comma, but be careful with quoted strings
            values = self._split_insert_values(match)
            
            if len(values) >= len(columns):
                row = {}
                for i, col in enumerate(columns):
                    if i < len(values):
                        value = values[i].strip().strip("'\"")
                        if value == 'NULL':
                            row[col] = None
                        else:
                            row[col] = value
                results.append(row)
        
        return results
    
    def _split_insert_values(self, values_str: str) -> List[str]:
        """Split INSERT VALUES string properly handling quotes"""
        values = []
        current_value = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(values_str):
            char = values_str[i]
            
            if char in ["'", '"'] and not in_quotes:
                in_quotes = True
                quote_char = char
                current_value += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current_value += char
            elif char == ',' and not in_quotes:
                values.append(current_value.strip())
                current_value = ""
            else:
                current_value += char
            
            i += 1
        
        if current_value.strip():
            values.append(current_value.strip())
        
        return values
    
    def load_data_to_db(self, categories: List[Dict], codexes: List[Dict]):
        """Load parsed data into database and vector store with batch processing"""
        print("Loading data to database...")
        
        try:
            # Clear existing data
            self.db.query(Codex).delete()
            self.db.query(Category).delete()
            self.db.commit()
            
            # Load categories using bulk insert with error handling
            category_map = {}
            batch_size = Config.BATCH_SIZE
            
            print(f"Loading {len(categories)} categories using bulk insert...")
            
            # Prepare bulk insert data
            category_data = []
            for cat_data in categories:
                category_data.append({
                    'id': cat_data.get('id'),
                    'name': cat_data.get('name', ''),
                    'parent_id': cat_data.get('parent_id')
                })
            
            # Bulk insert categories with error handling
            if category_data:
                try:
                    self.db.bulk_insert_mappings(Category, category_data)
                    self.db.commit()
                    print("Categories loaded successfully in one operation")
                except Exception as e:
                    print(f"Bulk insert failed, trying individual inserts: {e}")
                    self.db.rollback()
                    
                    # Fallback to individual inserts with upsert logic
                    for cat_data in category_data:
                        try:
                            # Use merge (upsert) instead of insert
                            existing = self.db.query(Category).filter(Category.id == cat_data['id']).first()
                            if existing:
                                existing.name = cat_data['name']
                                existing.parent_id = cat_data['parent_id']
                            else:
                                new_category = Category(**cat_data)
                                self.db.add(new_category)
                        except Exception as individual_error:
                            print(f"Error inserting category {cat_data['id']}: {individual_error}")
                            continue
                    
                    self.db.commit()
                    print("Categories loaded using individual inserts")
            
        except Exception as e:
            print(f"Error in category loading: {e}")
            self.db.rollback()
            raise e
        
        # Create category map for reference
        for cat_data in categories:
            category_map[cat_data.get('id')] = {'id': cat_data.get('id')}
        
        # Load codexes using optimized bulk operations with error handling
        documents = []
        metadata = []
        
        print(f"Loading {len(codexes)} codexes using optimized bulk operations...")
        
        # Process codexes in larger batches with fewer commits
        commit_frequency = 5  # Commit every 5 batches instead of every batch
        processed_batches = 0
        
        for i in range(0, len(codexes), batch_size):
            batch = codexes[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(codexes) + batch_size - 1) // batch_size
            
            print(f"Processing codex batch {batch_num}/{total_batches} ({len(batch)} records)...")
            
            # Prepare bulk insert data for this batch
            codex_data_list = []
            batch_documents = []
            batch_metadata = []
            
            for idx, codex_data in enumerate(batch):
                # Prepare for bulk insert
                codex_data_list.append({
                    'id': codex_data.get('id'),
                    'name': codex_data.get('name', ''),
                    'content': codex_data.get('content', ''),
                    'category_id': codex_data.get('category_id')
                })
                
                # Chunk content for vector storage (with progress tracking)
                content = codex_data.get('content', '')
                if content and len(content.strip()) > 0:
                    chunks = self._chunk_text(content)
                    for j, chunk in enumerate(chunks):
                        batch_documents.append(chunk)
                        batch_metadata.append({
                            'codex_id': codex_data.get('id'),
                            'codex_name': codex_data.get('name', ''),
                            'category_id': codex_data.get('category_id'),
                            'chunk_index': j,
                            'content': chunk
                        })
                
                # Show progress every 100 records
                if (idx + 1) % 100 == 0:
                    print(f"    Processed {idx + 1}/{len(batch)} records in batch {batch_num}")
            
            # Bulk insert this batch with error handling
            if codex_data_list:
                try:
                    self.db.bulk_insert_mappings(Codex, codex_data_list)
                    processed_batches += 1
                    
                    # Commit every few batches instead of every batch
                    if processed_batches % commit_frequency == 0 or batch_num == total_batches:
                        self.db.commit()
                        print(f"  Committed {processed_batches} batches to database")
                except Exception as e:
                    print(f"Bulk insert failed for batch {batch_num}, trying individual inserts: {e}")
                    self.db.rollback()
                    
                    # Fallback to individual inserts with upsert logic
                    for codex_data in codex_data_list:
                        try:
                            # Use merge (upsert) instead of insert
                            existing = self.db.query(Codex).filter(Codex.id == codex_data['id']).first()
                            if existing:
                                existing.name = codex_data['name']
                                existing.content = codex_data['content']
                                existing.category_id = codex_data['category_id']
                            else:
                                new_codex = Codex(**codex_data)
                                self.db.add(new_codex)
                        except Exception as individual_error:
                            print(f"Error inserting codex {codex_data['id']}: {individual_error}")
                            continue
                    
                    self.db.commit()
                    processed_batches += 1
                    print(f"  Committed batch {batch_num} using individual inserts")
            
            # Skip vector processing for now to speed up database operations
            # We'll create vectors later in a separate step
            if batch_documents:
                documents.extend(batch_documents)
                metadata.extend(batch_metadata)
                print(f"    Prepared {len(batch_documents)} chunks for vector processing")
            
            # Memory management
            del batch_documents
            del batch_metadata
            del codex_data_list
            
            # Force garbage collection every 10 batches
            if batch_num % 10 == 0:
                import gc
                gc.collect()
                print(f"  Memory cleanup completed for batch {batch_num}")
        
        # Process all vectors at once (much faster than batch by batch)
        if documents:
            print(f"Creating vector index for {len(documents)} chunks...")
            print("This may take a few minutes...")
            
            # Use optimized batch processing
            vector_db.add_documents(documents, metadata)
            
            print("Saving vector index...")
            vector_db.save_index()
            print("Vector index created successfully!")
        
        print(f"Loaded {len(categories)} categories and {len(codexes)} codexes")
        print(f"Created {len(documents)} text chunks for vector search")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks (optimized for speed)"""
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Simple chunking for speed - no sentence boundary detection
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
