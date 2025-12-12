# app/retrieval/bm25_retriever.py
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

BM25_CACHE_DIR = os.getenv("BM25_CACHE_DIR", "bm25_index")
BM25_REBUILD_INTERVAL = int(os.getenv("BM25_REBUILD_INTERVAL", "300"))



class BM25Retriever:
    """BM25 sparse retrieval for keyword-based search."""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.cache_dir = Path(BM25_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / f"{collection_name}_bm25.pkl"
        self.docs_path = self.cache_dir / f"{collection_name}_docs.pkl"
        
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self._load_index()
    
    def _load_index(self):
        """Load BM25 index from cache if available."""
        if self.index_path.exists() and self.docs_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(self.docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                self.bm25 = None
                self.documents = []
    
    def _save_index(self):
        """Save BM25 index to cache."""
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.bm25, f)
            with open(self.docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            logger.info(f"Saved BM25 index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return
        
        self.documents = documents
        tokenized_corpus = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self._save_index()
        logger.info(f"Built BM25 index for {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with 'id', 'text', 'metadata', 'score'
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25 index not available")
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                doc = self.documents[idx].copy()
                doc["score"] = float(scores[idx])
                results.append(doc)
        
        logger.info(f"BM25 retrieved {len(results)} results for query: {query[:50]}")
        return results
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """Update or add a single document to the index."""
        # Check if document exists
        existing_idx = None
        for idx, doc in enumerate(self.documents):
            if doc.get("id") == doc_id:
                existing_idx = idx
                break
        
        new_doc = {"id": doc_id, "text": text, "metadata": metadata}
        
        if existing_idx is not None:
            self.documents[existing_idx] = new_doc
        else:
            self.documents.append(new_doc)
        
        # Rebuild index (could be optimized for incremental updates)
        self.build_index(self.documents)


async def start_bm25_rebuild_task(chroma_client, collection_name: str = "documents", interval_seconds: int = BM25_REBUILD_INTERVAL):
    """Periodically rebuild BM25 index when new documents are detected in Chroma.
    
    Args:
        chroma_client: ChromaDB client instance
        collection_name: Name of the ChromaDB collection to monitor
        interval_seconds: Seconds between rebuild checks (default from env BM25_REBUILD_INTERVAL)
    """
    bm25_retriever = BM25Retriever(collection_name=collection_name)
    last_doc_count = -1
    
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as e:
        logger.warning(f"BM25 rebuild task: could not access collection '{collection_name}': {e}")
        return
    
    while True:
        try:
            data = collection.get()
            ids = data.get("ids", []) or []
            docs = data.get("documents", []) or []
            metas = data.get("metadatas", []) or []
            current_count = len(ids)
            
            if current_count != last_doc_count and current_count > 0:
                bm25_docs = []
                for i, id_ in enumerate(ids):
                    bm25_docs.append({
                        "id": str(id_),
                        "text": str(docs[i]) if i < len(docs) else "",
                        "metadata": metas[i] if i < len(metas) else {},
                    })
                bm25_retriever.build_index(bm25_docs)
                last_doc_count = current_count
                logger.info(f"BM25 periodic rebuild complete: {current_count} documents")
            else:
                logger.debug(f"BM25 periodic rebuild: no changes (count={current_count})")
        except Exception as e:
            logger.error(f"BM25 periodic rebuild loop error: {e}")
        
        await asyncio.sleep(interval_seconds)

