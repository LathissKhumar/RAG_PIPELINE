# app/embeddings/cache.py
import sqlite3
import hashlib
import numpy as np
import threading
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB = "embeddings_cache.sqlite3"

def compute_hash(text: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()

class Cache:
    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    id INTEGER PRIMARY KEY,
                    model TEXT NOT NULL,
                    hash TEXT NOT NULL UNIQUE,
                    text TEXT,
                    dim INTEGER,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def bulk_get(self, hashes: List[str]) -> Dict[str, np.ndarray]:
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        q = f"SELECT hash, dim, embedding FROM embeddings_cache WHERE hash IN ({placeholders})"
        with self._lock:
            cur = self.conn.execute(q, hashes)
            rows = cur.fetchall()
        out: Dict[str, np.ndarray] = {}
        for hsh, dim, blob in rows:
            arr = np.frombuffer(blob, dtype=np.float32)
            if dim is not None:
                arr = arr.reshape((dim,))
            out[hsh] = arr
        logger.info(f"Cache bulk_get: requested {len(hashes)}, found {len(out)} cached embeddings")
        return out

    def set(self, hash_: str, model: str, text: str, vector: np.ndarray):
        dim = int(vector.size)
        blob = vector.astype(np.float32).tobytes()
        with self._lock:
            try:
                with self.conn:
                    self.conn.execute(
                        "INSERT INTO embeddings_cache (model, hash, text, dim, embedding) VALUES (?, ?, ?, ?, ?)",
                        (model, hash_, text, dim, blob),
                    )
            except sqlite3.IntegrityError:
                # already exists
                pass

    def bulk_set(self, items: List[Tuple[str, str, str, np.ndarray]]):
        if not items:
            return
        with self._lock:
            cur = self.conn.cursor()
            for hash_, model, text, vector in items:
                dim = int(vector.size)
                blob = vector.astype(np.float32).tobytes()
                try:
                    cur.execute(
                        "INSERT INTO embeddings_cache (model, hash, text, dim, embedding) VALUES (?, ?, ?, ?, ?)",
                        (model, hash_, text, dim, blob),
                    )
                except sqlite3.IntegrityError:
                    continue
            self.conn.commit()
        logger.info(f"Cache bulk_set: stored {len(items)} embeddings")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass