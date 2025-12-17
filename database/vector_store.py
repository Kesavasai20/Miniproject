"""
Vector Store for Semantic Search
Uses FAISS and ChromaDB for embedding storage and retrieval
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from sentence_transformers import SentenceTransformer

import sys
sys.path.append('..')
from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Hybrid vector store using FAISS for fast similarity search
    and ChromaDB for persistent storage with metadata filtering
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_chroma: bool = True,
        use_faiss: bool = True
    ):
        self.embedding_model_name = embedding_model
        self.use_chroma = use_chroma and CHROMA_AVAILABLE
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize stores
        self.faiss_index = None
        self.faiss_id_map = {}  # Maps FAISS index position to document ID
        self.chroma_client = None
        self.chroma_collection = None
        
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize FAISS index and ChromaDB collection"""
        
        # FAISS Index
        if self.use_faiss:
            self._init_faiss()
        
        # ChromaDB
        if self.use_chroma:
            self._init_chroma()
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        index_path = settings.FAISS_INDEX_PATH
        
        if Path(f"{index_path}.index").exists():
            # Load existing index
            self.faiss_index = faiss.read_index(f"{index_path}.index")
            with open(f"{index_path}_map.json", "r") as f:
                self.faiss_id_map = json.load(f)
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (IVF for scalability)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine)
            logger.info("Created new FAISS index")
    
    def _init_chroma(self):
        """Initialize ChromaDB collection"""
        persist_dir = str(settings.CHROMA_PERSIST_DIR)
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection for float data
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="argo_floats",
                metadata={"description": "ARGO float summaries and metadata"}
            )
            
            logger.info(f"ChromaDB initialized with {self.chroma_collection.count()} documents")
        except Exception as e:
            logger.warning(f"Could not initialize ChromaDB: {e}")
            self.use_chroma = False
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embedder.encode(text, normalize_embeddings=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id"
    ) -> int:
        """
        Add documents to the vector store
        
        Args:
            documents: List of dicts with text and metadata
            text_field: Key for the text content
            id_field: Key for document ID
        
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        texts = [doc[text_field] for doc in documents]
        ids = [str(doc[id_field]) for doc in documents]
        embeddings = self.embed_texts(texts)
        
        # Add to FAISS
        if self.use_faiss and self.faiss_index is not None:
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embeddings.astype(np.float32))
            
            for i, doc_id in enumerate(ids):
                self.faiss_id_map[str(start_idx + i)] = doc_id
            
            self._save_faiss()
        
        # Add to ChromaDB
        if self.use_chroma and self.chroma_collection is not None:
            metadatas = []
            for doc in documents:
                meta = {k: v for k, v in doc.items() if k not in [text_field, id_field]}
                # ChromaDB only supports str, int, float, bool
                meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v 
                        for k, v in meta.items()}
                metadatas.append(meta)
            
            self.chroma_collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                ids=ids,
                metadatas=metadatas
            )
        
        logger.info(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
        use_chroma_filter: bool = True
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter_metadata: Metadata filters (ChromaDB only)
            use_chroma_filter: Whether to use ChromaDB for filtered search
        
        Returns:
            List of (document_id, score, metadata) tuples
        """
        query_embedding = self.embed_text(query)
        results = []
        
        # If filtering needed and ChromaDB available, use it
        if filter_metadata and use_chroma_filter and self.use_chroma:
            chroma_results = self.chroma_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filter_metadata
            )
            
            if chroma_results["ids"][0]:
                for i, doc_id in enumerate(chroma_results["ids"][0]):
                    score = 1.0 - chroma_results["distances"][0][i] if chroma_results["distances"] else 0.0
                    meta = chroma_results["metadatas"][0][i] if chroma_results["metadatas"] else {}
                    results.append((doc_id, score, meta))
        
        # Otherwise use FAISS for fast unfiltered search
        elif self.use_faiss and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(top_k, self.faiss_index.ntotal)
            )
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    doc_id = self.faiss_id_map.get(str(idx), str(idx))
                    score = float(distances[0][i])
                    results.append((doc_id, score, {}))
        
        return results
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """
        Search and return context string for RAG
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant context found."
        
        # Get actual documents from ChromaDB
        if self.use_chroma and self.chroma_collection:
            doc_ids = [r[0] for r in results]
            docs = self.chroma_collection.get(ids=doc_ids)
            
            context_parts = []
            for i, doc in enumerate(docs.get("documents", [])):
                if doc:
                    context_parts.append(f"[{i+1}] {doc}")
            
            return "\n\n".join(context_parts)
        
        return "Context retrieved but details unavailable."
    
    def _save_faiss(self):
        """Save FAISS index to disk"""
        if self.faiss_index is not None:
            index_path = settings.FAISS_INDEX_PATH
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.faiss_index, f"{index_path}.index")
            with open(f"{index_path}_map.json", "w") as f:
                json.dump(self.faiss_id_map, f)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        stats = {
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dim,
            "faiss_enabled": self.use_faiss,
            "chroma_enabled": self.use_chroma,
        }
        
        if self.use_faiss and self.faiss_index:
            stats["faiss_vectors"] = self.faiss_index.ntotal
        
        if self.use_chroma and self.chroma_collection:
            stats["chroma_documents"] = self.chroma_collection.count()
        
        return stats
    
    def clear(self):
        """Clear all vectors from stores"""
        if self.use_faiss:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_id_map = {}
            self._save_faiss()
        
        if self.use_chroma and self.chroma_collection:
            # Delete and recreate collection
            self.chroma_client.delete_collection("argo_floats")
            self.chroma_collection = self.chroma_client.create_collection(
                name="argo_floats",
                metadata={"description": "ARGO float summaries and metadata"}
            )
        
        logger.info("Vector store cleared")


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
