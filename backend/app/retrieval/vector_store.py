"""
Vector store manager using FAISS and SentenceTransformers.
Encapsulates embedding generation, indexing, serialization, and semantic search.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from backend.app.config import settings
from backend.app.models.schemas import Chunk, EvidenceItem

logger = logging.getLogger("graphrag.retrieval.vector_store")

class VectorStore:
    _instance: Optional['VectorStore'] = None

    def __new__(cls) -> 'VectorStore':
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        """Initializes the embedding model and loads the local FAISS index if present."""
        if getattr(self, "_initialized", False):
            return
            
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {str(e)}")
            raise e
            
        self.index_dir = settings.VECTOR_STORE_PATH
        self.index_file = os.path.join(self.index_dir, "index.faiss")
        self.metadata_file = os.path.join(self.index_dir, "chunks_metadata.json")
        
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 (default)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        
        self._load_index()
        self._initialized = True

    def _load_index(self) -> None:
        """Loads FAISS index and metadata ledger from disk if they exist."""
        os.makedirs(self.index_dir, exist_ok=True)
        
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Successfully loaded vector index containing {len(self.metadata)} items.")
            except Exception as e:
                logger.error(f"Error loading local vector store files: {str(e)}. Initializing empty index.")
                self.index = None
                self.metadata = []
                
        if self.index is None:
            # IndexFlatIP computes inner products. Normalizing vectors yields Cosine similarity.
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
            logger.info("Initialized new empty FAISS IndexFlatIP index.")

    def _save_index(self) -> None:
        """Saves the index and metadata ledger to disk."""
        os.makedirs(self.index_dir, exist_ok=True)
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info("Saved vector index and metadata ledger to disk.")
        except Exception as e:
            logger.error(f"Failed to serialize vector index: {str(e)}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Embeds a list of document chunks, normalizes the vectors,
        adds them to the FAISS index, and saves to disk.
        """
        self.initialize()
        if not chunks:
            return
            
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} text chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings_arr = np.array(embeddings).astype('float32')
        
        # L2 normalization for inner-product to perform Cosine Similarity
        faiss.normalize_L2(embeddings_arr)
        
        # Add to index
        self.index.add(embeddings_arr)
        
        # Append metadata
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "section_title": chunk.section_title,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset
            })
            
        self._save_index()

    def search(self, query: str, top_k: int = 5) -> List[EvidenceItem]:
        """
        Semantic search on chunk embeddings.
        Returns:
            List[EvidenceItem]: List of retrieved EvidenceItems matching the query.
        """
        self.initialize()
        if self.index.ntotal == 0:
            logger.warning("Search called on empty vector index.")
            return []
            
        # Embed and normalize query vector
        query_vector = self.model.encode([query], show_progress_bar=False)
        query_arr = np.array(query_vector).astype('float32')
        faiss.normalize_L2(query_arr)
        
        # Perform query search
        top_k_adj = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_arr, top_k_adj)
        
        evidence_items = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            meta = self.metadata[idx]
            # Convert inner product score (cosine similarity range [-1, 1]) to a confidence float
            confidence = float(max(0.0, min(1.0, score)))
            
            # Format content to include section information if available
            section_info = f"[Section: {meta['section_title']}] " if meta.get("section_title") else ""
            content = f"{section_info}{meta['text']}"
            
            evidence_items.append(EvidenceItem(
                type="chunk",
                content=content,
                source_ids=[meta["document_id"]],
                confidence=confidence
            ))
            
        return evidence_items

    def clear(self) -> None:
        """Clears the current vector store index and deletes files on disk."""
        self.initialize()
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        if os.path.exists(self.index_file):
            try:
                os.remove(self.index_file)
            except Exception:
                pass
        if os.path.exists(self.metadata_file):
            try:
                os.remove(self.metadata_file)
            except Exception:
                pass
        logger.info("Cleared vector store index and files.")

# Singleton helper
vector_store = VectorStore()
