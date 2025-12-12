# app/retrieval/reranker.py
import os
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import torch

logger = logging.getLogger(__name__)

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "32"))


class Reranker:
    """Cross-encoder reranker for improving retrieval quality."""
    
    def __init__(self, model_name: str = RERANKER_MODEL, device: str = RERANKER_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = None
        logger.info(f"Reranker initialized with model: {model_name}, device: {device}")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker model loaded successfully")
        return self._model
    
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of dicts with 'text' field
            top_k: Number of top results to return (None = return all)
            
        Returns:
            Reranked list of documents with added 'rerank_score' field
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, doc.get("text", "")) for doc in documents]
        
        # Get reranking scores in batches
        try:
            scores = self.model.predict(pairs, batch_size=RERANKER_BATCH_SIZE, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents
        
        # Add scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        
        reranked = sorted(documents, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        if top_k is not None:
            reranked = reranked[:top_k]
        
        logger.info(f"Reranked {len(documents)} documents, returned top {len(reranked)}")
        return reranked
