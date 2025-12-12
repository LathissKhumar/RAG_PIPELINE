# app/retrieval/hybrid_retriever.py
import os
import logging
from typing import List, Dict, Any, Optional
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.reranker import Reranker
from app.vector_store.chroma_client import get_chroma_client, query_texts

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))  # Weight for dense retrieval (0=BM25 only, 1=vector only)
USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"


class HybridRetriever:
    """Production-grade hybrid retriever combining BM25 and dense retrieval with reranking."""
    
    def __init__(
        self, 
        collection_name: str = CHROMA_COLLECTION,
        alpha: float = HYBRID_ALPHA,
        use_reranker: bool = USE_RERANKER
    ):
        """
        Args:
            collection_name: ChromaDB collection name
            alpha: Weight for dense retrieval (0-1). 0=BM25 only, 1=vector only, 0.5=balanced
            use_reranker: Whether to use cross-encoder reranking
        """
        self.collection_name = collection_name
        self.alpha = alpha
        self.use_reranker = use_reranker
        
        self.bm25 = BM25Retriever(collection_name)
        self.chroma_client = get_chroma_client()
        self.reranker = Reranker() if use_reranker else None
        
        logger.info(
            f"HybridRetriever initialized: collection={collection_name}, "
            f"alpha={alpha}, reranker={use_reranker}"
        )
    
    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents.
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
        """
        self.bm25.build_index(documents)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        bm25_top_k: Optional[int] = None,
        vector_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with optional reranking.
        
        Args:
            query: Search query
            top_k: Final number of results to return
            bm25_top_k: Number of BM25 results (default: top_k * 3)
            vector_top_k: Number of vector results (default: top_k * 3)
            rerank_top_k: Number of results after reranking (default: top_k)
            where: ChromaDB metadata filter
            where_document: ChromaDB document filter
            
        Returns:
            List of documents with scores
        """
        # Default retrieval counts
        bm25_top_k = bm25_top_k or (top_k * 3)
        vector_top_k = vector_top_k or (top_k * 3)
        rerank_top_k = rerank_top_k or top_k
        
        # 1. Dense retrieval (ChromaDB vector search)
        dense_results = []
        try:
            res = query_texts(
                self.chroma_client,
                self.collection_name,
                query,
                top_k=vector_top_k,
                where=where,
                where_document=where_document,
            )
            ids = res.get("ids", [[]])[0] if isinstance(res.get("ids"), list) else []
            docs = res.get("documents", [[]])[0] if isinstance(res.get("documents"), list) else []
            metas = res.get("metadatas", [[]])[0] if isinstance(res.get("metadatas"), list) else []
            dists = res.get("distances", [[]])[0] if isinstance(res.get("distances"), list) else []
            
            for i, id_ in enumerate(ids):
                dense_results.append({
                    "id": str(id_),
                    "text": str(docs[i]) if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "vector_score": 1.0 / (1.0 + float(dists[i])) if i < len(dists) else 0.0,  # Convert distance to similarity
                })
            logger.info(f"Dense retrieval: {len(dense_results)} results")
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            if "dimension" in str(e).lower() or "size" in str(e).lower():
                logger.error(
                    "Embedding dimension mismatch detected. This usually happens when:\n"
                    "  1. The embedding model changed (e.g., bge-m3 to a different model)\n"
                    "  2. Chroma DB was created with a different embedding dimension\n"
                    "Solution: Clear chroma_db/ and re-ingest documents, or ensure EMBED_MODEL is consistent"
                )
            import traceback
            logger.debug(traceback.format_exc())
        
        # 2. Sparse retrieval (BM25)
        sparse_results = []
        try:
            sparse_results = self.bm25.search(query, top_k=bm25_top_k)
            logger.info(f"BM25 retrieval: {len(sparse_results)} results")
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
        
        # 3. Hybrid fusion using Reciprocal Rank Fusion (RRF)
        merged = self._reciprocal_rank_fusion(
            dense_results, 
            sparse_results, 
            alpha=self.alpha,
            k=60  # RRF constant
        )
        
        logger.info(f"Hybrid fusion: {len(merged)} unique results")
        
        # 4. Reranking (optional)
        if self.use_reranker and self.reranker and merged:
            try:
                merged = self.reranker.rerank(query, merged, top_k=rerank_top_k)
                logger.info(f"Reranking complete: {len(merged)} results")
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
        
        # Return top-k
        return merged[:top_k]
    
    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        alpha: float = 0.5,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion for combining ranked lists.
        
        RRF score = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_sparse))
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            alpha: Weight for dense retrieval (0-1)
            k: RRF constant (default: 60)
            
        Returns:
            Merged and sorted results
        """
        # Create rank maps
        doc_scores = {}
        doc_data = {}
        
        # Process dense results
        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            rrf_score = alpha / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
                doc_data[doc_id]["retrieval_sources"] = []
            doc_data[doc_id]["retrieval_sources"].append("vector")
            doc_data[doc_id]["vector_rank"] = rank + 1
        
        # Process sparse results
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            rrf_score = (1 - alpha) / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = doc
                doc_data[doc_id]["retrieval_sources"] = []
            doc_data[doc_id]["retrieval_sources"].append("bm25")
            doc_data[doc_id]["bm25_rank"] = rank + 1
            if "bm25_score" in doc:
                doc_data[doc_id]["bm25_score"] = doc["bm25_score"]
        
        # Sort by RRF score
        merged = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_data[doc_id]
            doc["hybrid_score"] = score
            merged.append(doc)
        
        return merged
