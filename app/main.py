# app/main.py
from fastapi import FastAPI, HTTPException
from app.routers import upload
from app.embeddings import worker
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import requests
from app.vector_store.chroma_client import get_chroma_client
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25_retriever import start_bm25_rebuild_task
from app.llm import generate_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

app = FastAPI()
app.include_router(upload.router)

# Health check models
class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    workers_running: bool

class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    use_llm: bool = True
    where: Optional[dict] = None
    where_document: Optional[dict] = None

class QueryResult(BaseModel):
    id: str
    text: str
    metadata: dict
    distance: float

class AskResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    top_k: int
    results: List[QueryResult]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        response = requests.head(f"{ollama_url}/api/tags", timeout=2)
        ollama_available = response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama health check failed: {e}")
        ollama_available = False
    
    return HealthResponse(
        status="healthy",
        ollama_available=ollama_available,
        workers_running=worker._started
    )

@app.post("/ask", response_model=AskResponse)
async def ask(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # Use production-grade hybrid retrieval (BM25 + vector + optional reranker)
    try:
        global _hybrid_retriever
        if '_hybrid_retriever' not in globals() or _hybrid_retriever is None:
            _hybrid_retriever = HybridRetriever(collection_name=CHROMA_COLLECTION)
        hybrid_docs = _hybrid_retriever.retrieve(
            query=req.question.strip(),
            top_k=max(1, req.top_k),
            where=req.where,
            where_document=req.where_document,
        )
    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Map hybrid results to QueryResult
    results: List[QueryResult] = []
    for doc in hybrid_docs:
        results.append(
            QueryResult(
                id=str(doc.get('id', '')),
                text=str(doc.get('text', '')),
                metadata=doc.get('metadata', {}) or {},
                distance=float(doc.get('hybrid_score', 0.0)),
            )
        )

    # Generate LLM answer if requested
    answer = None
    if req.use_llm and results:
        try:
            context_chunks = [{"text": r.text, "metadata": r.metadata} for r in results]
            answer = generate_answer(req.question.strip(), context_chunks)
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"

    return AskResponse(question=req.question, answer=answer, top_k=len(results), results=results)

@app.on_event("startup")
async def startup_event():
    worker.start_workers()
    # Initialize hybrid retriever and start BM25 periodic rebuild task
    global _hybrid_retriever
    _hybrid_retriever = HybridRetriever(collection_name=CHROMA_COLLECTION)
    logger.info("HybridRetriever initialized")
    # Start BM25 rebuild background task
    import asyncio
    client = get_chroma_client()
    asyncio.create_task(start_bm25_rebuild_task(chroma_client=client, collection_name=CHROMA_COLLECTION))
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    await worker.stop_workers()
    logger.info("Application shutdown complete")
