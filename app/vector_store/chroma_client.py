# app/vector_store/chroma_client.py
import os
from typing import Any, Dict, List
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_METRIC = os.getenv("CHROMA_METRIC", "cosine")

_client = None

def get_chroma_client():
    global _client
    if _client is None:
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PERSIST_DIR)
        _client = chromadb.Client(settings)
    return _client

def ingest_batch(client, collection_name: str, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
    logger.info(f"Ingesting batch of {len(ids)} items to Chroma collection '{collection_name}'")
    # ensure collection exists
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name, metadata={"distance": CHROMA_METRIC})
    # deduplicate: try to add, but if ids already present, update them
    # chroma will error on duplicate ids; safer to call add with only new ids
    existing_ids = set()
    try:
        # query collection for existing ids by metadata is not trivial; attempt naive safe add
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        logger.info(f"Successfully added {len(ids)} items to Chroma")
    except Exception as e:
        logger.warning(f"Chroma add failed, attempting delete-then-add: {e}")
        try:
            # try remove then add for duplicates (best-effort)
            for id_ in ids:
                try:
                    collection.delete(ids=[id_])
                except Exception:
                    pass
            collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            logger.info(f"Successfully re-added {len(ids)} items to Chroma after delete")
        except Exception as e:
            logger.error(f"Chroma ingest error: {e}")
    # persist to disk
    try:
        client.persist()
        logger.info("Chroma data persisted to disk")
    except Exception as e:
        logger.warning(f"Chroma persist failed: {e}")