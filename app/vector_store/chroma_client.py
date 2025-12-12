# app/vector_store/chroma_client.py
import os
from typing import Any, Dict, List, Optional
import chromadb
import logging
from app.embeddings.ollama_embeddings import embed_texts

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_METRIC = os.getenv("CHROMA_METRIC", "cosine")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

_client = None

def get_chroma_client():
    """Get or create ChromaDB persistent client. Uses modern PersistentClient API."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info(f"ChromaDB PersistentClient initialized at {CHROMA_PERSIST_DIR}")
    return _client


def _get_or_create_collection(client, collection_name: str):
    try:
        return client.get_collection(collection_name)
    except Exception:
        return client.create_collection(name=collection_name, metadata={"distance": CHROMA_METRIC})


def filter_missing_ids(client, collection_name: str, ids: List[str]) -> List[str]:
    """Return the subset of ids that are not yet stored in the collection.

    If the collection is empty or lookup fails, conservatively return all ids (so we re-ingest).
    """
    if not ids:
        return []
    try:
        collection = _get_or_create_collection(client, collection_name)
        res = collection.get(ids=ids)
        existing = set(res.get("ids", [])) if isinstance(res, dict) else set()
        return [i for i in ids if i not in existing]
    except Exception as e:
        logger.warning(f"Chroma missing-id check failed for {collection_name}: {e}")
        return ids


def query_texts(client, collection_name: str, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None):
    """Query Chroma with a text question using the configured embed model."""
    try:
        collection = _get_or_create_collection(client, collection_name)
        
        # Check if collection is empty
        count = collection.count()
        if count == 0:
            logger.warning(f"Collection '{collection_name}' is empty, returning empty results")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Validate query
        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        # Generate query embedding
        embedding_result = embed_texts([query.strip()], model=EMBED_MODEL)
        
        # Validate embedding result
        if not embedding_result or len(embedding_result) == 0:
            logger.error(f"Embedding generation returned empty result for query: {query[:50]}")
            raise ValueError(f"Failed to generate embedding for query using model {EMBED_MODEL}")
        
        query_embedding = embedding_result[0].tolist()
        
        # Validate embedding is not empty
        if not query_embedding or len(query_embedding) == 0:
            logger.error(f"Generated embedding is empty for query: {query[:50]}")
            raise ValueError(f"Empty embedding generated for query using model {EMBED_MODEL}")
        
        logger.debug(f"Query embedding dimension: {len(query_embedding)}")
        
        # Filter out empty/None where filters (ChromaDB doesn't accept empty dicts)
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),  # Don't request more than available
        }
        
        if where and len(where) > 0:
            query_params["where"] = where
        
        if where_document and len(where_document) > 0:
            query_params["where_document"] = where_document
        
        # Query collection
        res = collection.query(**query_params)
        return res
    except Exception as e:
        logger.error(f"Query failed for collection '{collection_name}': {e}")
        # Check for dimension mismatch
        error_str = str(e).lower()
        if "dimension" in error_str or "size" in error_str or "shape" in error_str or "index" in error_str:
            if 'query_embedding' in locals() and query_embedding:
                logger.error(f"Query embedding dimension: {len(query_embedding)}")
            else:
                logger.error("Query embedding was not generated or is empty")
            try:
                # Get a sample embedding from collection to compare
                sample = collection.get(limit=1, include=["embeddings"])
                if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                    stored_dim = len(sample["embeddings"][0])
                    logger.error(f"Stored embedding dimension: {stored_dim}")
            except Exception:
                pass
        raise

def ingest_batch(client, collection_name: str, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
    logger.info(f"Ingesting batch of {len(ids)} items to Chroma collection '{collection_name}'")
    
    # Validate input lengths match
    if not ids:
        logger.warning("Empty batch - nothing to ingest")
        return
    
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        logger.error(f"Length mismatch: ids={len(ids)}, docs={len(documents)}, metas={len(metadatas)}, embeds={len(embeddings)}")
        raise ValueError("All input lists must have the same length")
    
    # Validate embeddings are not empty
    for i, emb in enumerate(embeddings):
        if not emb or len(emb) == 0:
            logger.error(f"Empty embedding at index {i} for id {ids[i]}")
            raise ValueError(f"Empty embedding found at index {i}")
    
    # Ensure collection exists
    collection = _get_or_create_collection(client, collection_name)
    
    try:
        # Attempt direct add
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        logger.info(f"Successfully added {len(ids)} items to Chroma")
    except Exception as e:
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "already exists" in error_msg:
            logger.warning(f"Duplicate IDs detected, attempting upsert via delete-then-add: {e}")
            try:
                # Delete existing items then re-add
                for id_ in ids:
                    try:
                        collection.delete(ids=[id_])
                    except Exception:
                        pass
                collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
                logger.info(f"Successfully upserted {len(ids)} items to Chroma after delete")
            except Exception as e2:
                logger.error(f"Chroma upsert failed: {e2}")
                raise
        else:
            logger.error(f"Chroma add failed with error: {e}")
            raise
    
    logger.info(f"Batch ingestion complete - {len(ids)} items stored in {collection_name}")