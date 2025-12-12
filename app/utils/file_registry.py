# app/utils/file_registry.py
import sqlite3
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple
import os

logger = logging.getLogger(__name__)

FILE_REGISTRY_DB = os.getenv("FILE_REGISTRY_DB", "file_registry.db")


class FileRegistry:
    """SQLite-backed registry for tracking PDF file hashes and conversion state."""
    
    def __init__(self, db_path: str = FILE_REGISTRY_DB):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the file registry database with schema if not exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_hash TEXT NOT NULL,
                        md_output_path TEXT,
                        converted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                logger.info(f"FileRegistry initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize FileRegistry: {e}")
            raise
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            raise
    
    def get_file_entry(self, file_path: str) -> Optional[Tuple[str, str]]:
        """Get registry entry for a file.
        
        Returns:
            Tuple of (file_hash, md_output_path) or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_hash, md_output_path FROM files WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                return row if row else None
        except Exception as e:
            logger.error(f"Failed to query FileRegistry for {file_path}: {e}")
            return None
    
    def register_file(self, file_path: str, file_hash: str, md_output_path: str) -> bool:
        """Register or update a file in the registry.
        
        Args:
            file_path: Full path to the PDF file
            file_hash: SHA256 hash of the file
            md_output_path: Path to the generated markdown file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO files (file_path, file_hash, md_output_path, converted_at, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(file_path) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        md_output_path = excluded.md_output_path,
                        updated_at = CURRENT_TIMESTAMP
                """, (file_path, file_hash, md_output_path))
                conn.commit()
                logger.info(f"Registered file {file_path} with hash {file_hash[:12]}...")
                return True
        except Exception as e:
            logger.error(f"Failed to register file {file_path}: {e}")
            return False
    
    def should_skip_conversion(self, file_path: str, current_hash: str) -> bool:
        """Check if file conversion should be skipped (hash match in registry).
        
        Args:
            file_path: Full path to the PDF file
            current_hash: Current SHA256 hash of the file
            
        Returns:
            True if file hash matches registry (skip conversion), False otherwise
        """
        entry = self.get_file_entry(file_path)
        if entry:
            stored_hash, md_path = entry
            if stored_hash == current_hash and md_path:
                logger.info(f"File {file_path} unchanged (hash match); skipping conversion")
                return True
        return False
    
    def cleanup(self):
        """Close database connection."""
        if hasattr(self, 'db_path'):
            logger.info("FileRegistry cleanup complete")

# Global singleton instance
_registry: Optional[FileRegistry] = None

def get_file_registry() -> FileRegistry:
    """Get or create the global FileRegistry instance."""
    global _registry
    if _registry is None:
        _registry = FileRegistry()
    return _registry
