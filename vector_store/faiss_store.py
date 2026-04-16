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
            
            self.doc_registry[doc_name] = {
                "chunk_count": len(chunks),
                "added_at": datetime.now().isoformat()
            }
            self.save()
            logger.info(f"Successfully indexed {doc_name} with {len(chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Indexing Engine Failure for {doc_name}: {e}")
            raise Exception(f"Intelligence System Error: {str(e)}")

    def search(self, query: str, top_k: int = 5, filter_doc: str = None) -> list[dict]:
        if self.index is None:
            return []
            
        try:
            query_embedding = generate_embedding(query)
            if not query_embedding:
                return []
                
            query_embedding_np = np.array([query_embedding]).astype('float32')
            
            # Ensure query match dimension
            if query_embedding_np.shape[1] != self.index.d:
                logger.error(f"Search dimension mismatch: {query_embedding_np.shape[1]} vs {self.index.d}")
                return []

            distances, indices = self.index.search(query_embedding_np, top_k * 2)
            
            results = []
            if len(indices) > 0:
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    if idx == -1 or idx >= len(self.chunks):
                        continue
                        
                    chunk = self.chunks[idx].copy()
                    chunk["score"] = float(distances[0][i])
                    
                    if filter_doc and chunk["source_file"] != filter_doc:
                        continue
                        
                    results.append(chunk)
                    if len(results) >= top_k:
                        break
            return results
        except Exception as e:
            logger.error(f"Search failure: {e}")
            return []

    def delete_document(self, doc_name: str):
        if doc_name not in self.doc_registry:
            return
            
        self.chunks = [c for c in self.chunks if c["source_file"] != doc_name]
        del self.doc_registry[doc_name]
        
        if not self.chunks:
            self.index = None
        else:
            texts = [chunk["text"] for chunk in self.chunks]
            embeddings = generate_embeddings_batch(texts)
            if embeddings:
                embeddings_np = np.array(embeddings).astype('float32')
                dimension = embeddings_np.shape[1] if len(embeddings_np.shape) >= 2 else 768
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings_np)
            else:
                self.index = None
            
        self.save()

    def save(self):
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
                
    def get_documents(self) -> list[dict]:
        docs = []
        for name, info in self.doc_registry.items():
            docs.append({
                "doc_name": name,
                "chunk_count": info["chunk_count"],
                "added_at": info["added_at"]
            })
        return docs
