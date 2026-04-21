import os
import faiss
import numpy as np
import json
import logging
from datetime import datetime
from ingestion.embeddings import generate_embeddings_batch, generate_embedding

# Setup logging correctly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, index_dir: str = "faiss_index/"):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.bin")
        self.metadata_path = os.path.join(index_dir, "metadata.json")
        self.index = None
        self.chunks = []
        self.doc_registry = {}
        
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
            
        self.load()

    def add_document(self, chunks: list[dict], doc_name: str):
        """
        Generates embeddings for chunks and adds them to the FAISS index.
        """
        if not chunks:
            return
            
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            # This will now raise ValueError if it fails
            embeddings = generate_embeddings_batch(texts)
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Dimension Validation
            dimension = embeddings_np.shape[1]
            if self.index is None:
                self.index = faiss.IndexFlatL2(dimension)
            elif self.index.d != dimension:
                logger.warning(f"Dimension shift (Index: {self.index.d}, New: {dimension}). Brain Resetting.")
                self.index = faiss.IndexFlatL2(dimension)
                self.chunks = []
                self.doc_registry = {}

            self.index.add(embeddings_np)
            self.chunks.extend(chunks)
            
            # Detailed tracking
            self.doc_registry[doc_name] = {
                "chunk_count": len(chunks),
                "added_at": datetime.now().isoformat(),
                "status": "indexed"
            }
            self.save()
            logger.info(f"Successfully indexed {doc_name} with {len(chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Indexing Engine Failure for {doc_name}: {e}")
            raise Exception(f"Intelligence System Error: {str(e)}")

    def search(self, query: str, top_k: int = 4, filter_doc: str = None, distance_threshold: float = 1.5) -> list[dict]:
        if self.index is None or not self.chunks:
            logger.error("Search attempted on empty index.")
            raise Exception("No intelligence context currently loaded. Please sync your documents.")
            
        try:
            logger.info(f"Searching index with dimension {self.index.d} for query: {query}")
            query_embedding = generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding.")
                return []
                
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            if query_embedding_np.shape[1] != self.index.d:
                logger.error(f"Intelligence Brain Mismatch: Query uses {query_embedding_np.shape[1]} dims, but Index uses {self.index.d} dims. Please re-sync documents.")
                return []

            # Fetch a moderate pool of candidates to filter from
            fetch_k = min(len(self.chunks), top_k * 2)
            distances, indices = self.index.search(query_embedding_np, fetch_k)
            
            results = []
            if len(indices) > 0:
                logger.info(f"FAISS fetched {len(indices[0])} candidates (threshold={distance_threshold}).")
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx == -1 or idx >= len(self.chunks):
                        continue
                        
                    chunk = self.chunks[idx].copy()
                    dist = float(distances[0][i])
                    chunk["score"] = dist
                    
                    # Document filter
                    if filter_doc and chunk.get("source_file") != filter_doc:
                        continue
                    
                    # ── Relevance gate ────────────────────────────────────
                    # L2 distance: lower = more similar.  Skip chunks that
                    # are too far from the query to be genuinely useful.
                    if dist > distance_threshold:
                        logger.info(f"Skipped (too distant): {chunk.get('source_file')} — dist={dist:.4f}")
                        continue
                    
                    logger.info(f"Match: {chunk.get('source_file')} — dist={dist:.4f} ✓")
                    results.append(chunk)
                    if len(results) >= top_k:
                        break
            
            if not results:
                logger.warning("No chunks passed the relevance threshold.")
            return results
        except Exception as e:
            logger.error(f"Search failure: {e}")
            raise Exception(f"Analysis engine failure: {str(e)}")

    def delete_document(self, doc_name: str):
        if doc_name not in self.doc_registry:
            return
            
        self.chunks = [c for c in self.chunks if c.get("source_file") != doc_name]
        if doc_name in self.doc_registry:
            del self.doc_registry[doc_name]
        
        if not self.chunks:
            self.index = None
        else:
            # Rebuild index
            texts = [chunk["text"] for chunk in self.chunks]
            try:
                embeddings = generate_embeddings_batch(texts)
                if embeddings:
                    embeddings_np = np.array(embeddings).astype('float32')
                    self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
                    self.index.add(embeddings_np)
                else:
                    self.index = None
            except:
                self.index = None
            
        self.save()

    def save(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        metadata = {"chunks": self.chunks, "doc_registry": self.doc_registry}
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load(self):
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except:
                self.index = None
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                try:
                    metadata = json.load(f)
                    self.chunks = metadata.get("chunks", [])
                    self.doc_registry = metadata.get("doc_registry", {})
                except:
                    self.chunks = []
                    self.doc_registry = {}
        
        # Integrity Check: Index and metadata must be in sync
        if self.index and len(self.chunks) != self.index.ntotal:
            logger.warning(f"Integrity Mismatch: Index has {self.index.ntotal} vectors but Metadata has {len(self.chunks)} chunks. Resetting context.")
            self.index = None
            self.chunks = []
            self.doc_registry = {}
                
    def get_documents(self) -> list[dict]:
        docs = []
        for name, info in self.doc_registry.items():
            docs.append({
                "doc_name": name,
                "chunk_count": info.get("chunk_count", 0),
                "added_at": info.get("added_at", datetime.now().isoformat())
            })
        return docs
