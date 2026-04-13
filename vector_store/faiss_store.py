import os
import faiss
import numpy as np
import json
from datetime import datetime
from ingestion.embeddings import generate_embeddings_batch, generate_embedding

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
        Generates embeddings for all chunks and adds them to the FAISS index.
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings_batch(texts)
        embeddings_np = np.array(embeddings).astype('float32')
        
        if self.index is None:
            # all-MiniLM-L6-v2 has dimension 384
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        
        self.doc_registry[doc_name] = {
            "chunk_count": len(chunks),
            "added_at": datetime.now().isoformat()
        }
        
        self.save()

    def search(self, query: str, top_k: int = 5, filter_doc: str = None) -> list[dict]:
        """
        Searches the FAISS index for the most relevant chunks.
        """
        if self.index is None:
            return []
            
        query_embedding = generate_embedding(query)
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search for more than top_k to allow for filtering
        distances, indices = self.index.search(query_embedding_np, top_k * 2)
        
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:
                continue
                
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(distances[0][i])
            
            if filter_doc and chunk["source_file"] != filter_doc:
                continue
                
            results.append(chunk)
            if len(results) >= top_k:
                break
                
        return results

    def delete_document(self, doc_name: str):
        """
        Removes all chunks belonging to a document and rebuilds the index.
        """
        if doc_name not in self.doc_registry:
            return
            
        # Filter chunks
        self.chunks = [c for c in self.chunks if c["source_file"] != doc_name]
        del self.doc_registry[doc_name]
        
        # Rebuild index
        if not self.chunks:
            self.index = None
        else:
            texts = [chunk["text"] for chunk in self.chunks]
            embeddings = generate_embeddings_batch(texts)
            embeddings_np = np.array(embeddings).astype('float32')
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            
        self.save()

    def save(self):
        """
        Saves the FAISS index and metadata to disk.
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            
        metadata = {
            "chunks": self.chunks,
            "doc_registry": self.doc_registry
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.chunks = metadata.get("chunks", [])
                self.doc_registry = metadata.get("doc_registry", {})
                
    def get_documents(self) -> list[dict]:
        """
        Returns a list of registered documents.
        """
        docs = []
        for name, info in self.doc_registry.items():
            docs.append({
                "doc_name": name,
                "chunk_count": info["chunk_count"],
                "added_at": info["added_at"]
            })
        return docs
