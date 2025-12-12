# app/embeddings/ollama_embeddings.py
import os
import time
import numpy as np
from typing import List
import requests
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Ollama embeddings endpoint
OLLAMA_EMBED_PATH = os.getenv("OLLAMA_EMBED_PATH", "/api/embeddings")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

def _build_request(text: str, model: str):
    url = OLLAMA_URL.rstrip("/") + OLLAMA_EMBED_PATH
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    # Ollama /api/embeddings endpoint uses "prompt" not "input"
    payload = {"model": model, "prompt": text}
    return url, headers, payload

def _parse_response(resp_json) -> List[float]:
    # Ollama embeddings endpoint returns {"embedding": [...]} (single vector)
    if isinstance(resp_json, dict):
        if "embedding" in resp_json:
            emb = resp_json["embedding"]
            if emb and isinstance(emb, list) and len(emb) > 0:
                return emb
            logger.error(f"Empty or invalid embedding in response: {resp_json}")
        if "data" in resp_json:
            return resp_json["data"]
    logger.error(f"Unexpected Ollama response format: {resp_json}")
    raise ValueError(f"Unexpected Ollama response format: {resp_json}")

def embed_texts(texts: List[str], model: str = "bge-m3", retries: int = 3, backoff: float = 1.0) -> List[np.ndarray]:
    if not texts:
        return []
    
    # Filter out empty texts
    valid_texts = [(idx, text) for idx, text in enumerate(texts) if text and text.strip()]
    if len(valid_texts) != len(texts):
        logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty text(s) from embedding request")
    
    if not valid_texts:
        logger.warning("All texts were empty after filtering")
        return []
    
    logger.info(f"Embedding {len(valid_texts)} texts with model {model}")
    embeddings: List[np.ndarray] = []
    for list_idx, (orig_idx, text) in enumerate(valid_texts):
        attempt = 0
        while True:
            attempt += 1
            try:
                url, headers, payload = _build_request(text.strip(), model)
                logger.debug(f"Ollama request: POST {url} with payload: {payload}")
                resp = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
                resp.raise_for_status()
                
                resp_data = resp.json()
                logger.debug(f"Ollama response status: {resp.status_code}, keys: {list(resp_data.keys())}")
                
                vector = _parse_response(resp_data)
                
                # Validate vector is not empty
                if not vector or len(vector) == 0:
                    logger.error(f"Empty embedding in response. Full response: {resp_data}")
                    raise ValueError(f"Ollama returned empty embedding for text: {text[:50]}...")
                
                embeddings.append(np.array(vector, dtype=np.float32))
                logger.debug(f"Successfully embedded text, dimension: {len(vector)}")
                break
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt} failed for item {list_idx+1}/{len(valid_texts)}: {e}")
                if attempt >= retries:
                    logger.error(f"Embedding failed after {retries} retries for item {list_idx+1}/{len(valid_texts)}")
                    raise
                time.sleep(backoff * (2 ** (attempt - 1)))
    logger.info(f"Successfully embedded {len(embeddings)} texts")
    return embeddings