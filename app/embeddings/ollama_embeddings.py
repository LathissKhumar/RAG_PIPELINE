# app/embeddings/ollama_embeddings.py
import os
import time
import numpy as np
from typing import List
import requests
import logging

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_PATH = os.getenv("OLLAMA_EMBED_PATH", "/embed")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", None)
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

def _build_request(texts: List[str], model: str):
    url = OLLAMA_URL.rstrip("/") + OLLAMA_EMBED_PATH
    params = {"model": model}
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    payload = {"input": texts}
    return url, params, headers, payload

def _parse_response(resp_json) -> List[List[float]]:
    # Accept either {"embeddings": [...]} or list-of-vectors or {"data": [...]}
    if isinstance(resp_json, dict):
        if "embeddings" in resp_json:
            return resp_json["embeddings"]
        if "data" in resp_json:
            return resp_json["data"]
    if isinstance(resp_json, list):
        return resp_json
    raise ValueError("Unexpected Ollama response format")

def embed_texts(texts: List[str], model: str = "bge-m3", retries: int = 3, backoff: float = 1.0) -> List[np.ndarray]:
    if not texts:
        return []
    logger.info(f"Embedding {len(texts)} texts with model {model}")
    attempt = 0
    while True:
        attempt += 1
        try:
            url, params, headers, payload = _build_request(texts, model)
            resp = requests.post(url, params=params, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            json_data = resp.json()
            vectors = _parse_response(json_data)
            logger.info(f"Successfully embedded {len(vectors)} texts on attempt {attempt}")
            return [np.array(v, dtype=np.float32) for v in vectors]
        except Exception as e:
            logger.warning(f"Embedding attempt {attempt} failed: {e}")
            if attempt >= retries:
                logger.error(f"Embedding failed after {retries} retries")
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))