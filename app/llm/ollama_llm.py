# app/llm/ollama_llm.py
import os
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama3")
DEFAULT_TIMEOUT = float(os.getenv("OLLAMA_LLM_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("OLLAMA_LLM_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("OLLAMA_LLM_BACKOFF", "1.5"))


def generate_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate an answer using Ollama LLM with retrieved context chunks.
    
    Args:
        question: User's question
        context_chunks: List of dicts with 'text' and optional 'metadata'
        model: Ollama model name (default: llama3)
        system_prompt: Optional system prompt override
        
    Returns:
        Generated answer string
    """
    if not context_chunks:
        return "I don't have enough context to answer this question."
    
    # Build context from chunks
    context_parts = []
    for idx, chunk in enumerate(context_chunks, start=1):
        text = chunk.get("text", "")
        if text:
            context_parts.append(f"[Context {idx}]\n{text}")
    
    context_str = "\n\n".join(context_parts)
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a helpful AI assistant. Answer the user's question based on the provided context. "
            "If the context doesn't contain enough information, say so. Be concise and accurate."
            "If you don't know the answer, respond with 'I don't have enough context to answer this question.'" \
            "If the context includes multiple parts, you can reference them in your answer." \
            "Always cite the context parts you used."
            "Do not make up information."
            
        )
    
    # Build user prompt
    user_prompt = f"""Context:
{context_str}

Question: {question}

Answer:"""
    
    # Call Ollama API with simple retries for robustness
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False,
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.info(f"Generating answer with {model} (attempt {attempt}) for question: {question[:50]}...")
            response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            logger.info(f"Generated answer ({len(answer)} chars)")
            return answer
        except Exception as e:
            last_err = e
            logger.warning(f"LLM generation attempt {attempt} failed: {e}")
            if attempt <= MAX_RETRIES:
                import time
                time.sleep(RETRY_BACKOFF ** attempt)
            else:
                break
    logger.error(f"LLM generation failed after retries: {last_err}")
    raise last_err