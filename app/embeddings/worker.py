# app/embeddings/worker.py
import asyncio
import os
import time
import threading
import numpy as np
from typing import Any, Dict, List, Tuple
from app.embeddings.cache import Cache, compute_hash
from app.embeddings.ollama_embeddings import embed_texts
from app.vector_store.chroma_client import get_chroma_client, ingest_batch
import logging

logger = logging.getLogger(__name__)  

EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "3"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_BATCH_WAIT_MS = int(os.getenv("EMBED_BATCH_WAIT_MS", "200"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_CACHE_PATH = os.getenv("EMBED_CACHE_PATH", "embeddings_cache.sqlite3")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

_cache = Cache(EMBED_CACHE_PATH)
_queue: "asyncio.Queue[Tuple[str,str,Dict[str,Any]]]" = asyncio.Queue()
_worker_tasks: List[asyncio.Task] = []
_started = False
_started_lock = threading.Lock()  # Thread-safe flag access

async def _gather_batch(initial_item, max_size: int, wait_ms: int):
    batch = [initial_item]
    deadline = time.monotonic() + (wait_ms / 1000.0)
    while len(batch) < max_size:
        timeout = max(0, deadline - time.monotonic())
        try:
            item = await asyncio.wait_for(_queue.get(), timeout=timeout)
            if item is None:
                # re-insert sentinel for other workers and break
                await _queue.put(None)
                break
            batch.append(item)
        except asyncio.TimeoutError:
            break
    return batch

async def _worker_loop(worker_index: int):
    chroma_client = get_chroma_client()  # initialize per worker (lightweight)
    logger.info(f"Worker {worker_index} started")
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            break
        try:
            batch = await _gather_batch(item, EMBED_BATCH_SIZE, EMBED_BATCH_WAIT_MS)
            # batch items: tuples (chunk_id, text, metadata)
            ids = [it[0] for it in batch]
            texts = [it[1] for it in batch]
            metas = [it[2] or {} for it in batch]
            logger.info(f"Worker {worker_index} processing batch of {len(batch)} chunks")

            hashes = [compute_hash(t, EMBED_MODEL) for t in texts]
            cached = _cache.bulk_get(hashes)

            embeddings: List[np.ndarray] = [None] * len(batch)
            missing_indices = []
            missing_texts = []

            for i, h in enumerate(hashes):
                if h in cached:
                    embeddings[i] = cached[h]
                else:
                    missing_indices.append(i)
                    missing_texts.append(texts[i])

            if missing_texts:
                new_vectors = embed_texts(missing_texts, model=EMBED_MODEL)
                # place vectors into embeddings by mapping
                for idx, vec in zip(missing_indices, new_vectors):
                    embeddings[idx] = vec
                # bulk_set into cache
                cache_items = []
                for i in missing_indices:
                    cache_items.append((hashes[i], EMBED_MODEL, texts[i], embeddings[i]))
                _cache.bulk_set(cache_items)
                logger.info(f"Worker {worker_index} cached {len(missing_texts)} new embeddings")

            # Now store to vector store (Chroma)
            # ingest_batch should accept lists: ids, texts, metas, embeddings (numpy arrays)
            ingest_batch(chroma_client, CHROMA_COLLECTION, ids, texts, metas, [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings])
            logger.info(f"Worker {worker_index} ingested {len(ids)} chunks to Chroma")
        except Exception as e:
            logger.error(f"Embedding worker error: {e}")
        finally:
            for _ in batch:
                _queue.task_done()

async def _start_all_workers():
    global _worker_tasks
    _worker_tasks = []
    for i in range(EMBED_WORKERS):
        t = asyncio.create_task(_worker_loop(i))
        _worker_tasks.append(t)

async def _stop_all_workers():
    # send sentinel per worker
    for _ in _worker_tasks:
        await _queue.put(None)
    await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _cache.close()

def start_workers(loop: asyncio.AbstractEventLoop = None):
    global _started
    with _started_lock:
        if _started:
            return
        _started = True
    loop = loop or asyncio.get_event_loop()
    loop.create_task(_start_all_workers())

async def stop_workers():
    global _started
    with _started_lock:
        if not _started:
            return
        _started = False
    await _stop_all_workers()

# sync helper for chunker (works both from async context and sync code)
def enqueue_chunk_sync(chunk_id: str, text: str, metadata: Dict[str, Any] = None):
    metadata = metadata or {}
    logger.info(f"Enqueuing chunk {chunk_id} for embedding (text length: {len(text)})")
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    item = (chunk_id, text, metadata)
    if loop and loop.is_running():
        loop.call_soon_threadsafe(_queue.put_nowait, item)
    else:
        # background thread that pumps into the running event loop if any
        def _put():
            import asyncio
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(_queue.put(item))
            new_loop.close()
        import threading
        t = threading.Thread(target=_put, daemon=True)
        t.start()