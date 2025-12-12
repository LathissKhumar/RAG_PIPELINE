#!/usr/bin/env python3
"""Reset all caches and databases - useful after fixing embedding issues."""

import os
import shutil
import sys

def reset_all_caches():
    """Remove all cache databases and vector stores."""
    
    print("=" * 70)
    print("RESET ALL CACHES AND DATABASES")
    print("=" * 70)
    
    items_to_remove = [
        ("embeddings_cache.sqlite3", "Embeddings cache"),
        ("embeddings_cache.sqlite3-shm", "Embeddings cache (shared memory)"),
        ("embeddings_cache.sqlite3-wal", "Embeddings cache (write-ahead log)"),
        ("file_registry.db", "File registry"),
        ("chroma_db/", "ChromaDB vector store"),
        ("bm25_index/", "BM25 sparse index"),
    ]
    
    removed = []
    not_found = []
    errors = []
    
    for item, description in items_to_remove:
        try:
            if os.path.exists(item):
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                removed.append(f"  ✓ Removed: {description} ({item})")
            else:
                not_found.append(f"  - Not found: {description} ({item})")
        except Exception as e:
            errors.append(f"  ✗ Error removing {item}: {e}")
    
    print("\nRemoved:")
    for msg in removed:
        print(msg)
    
    if not_found:
        print("\nNot found (already clean):")
        for msg in not_found:
            print(msg)
    
    if errors:
        print("\nErrors:")
        for msg in errors:
            print(msg)
    
    print("\n" + "=" * 70)
    print("RESET COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Restart the FastAPI server if running")
    print("  2. Re-upload your documents: POST /convert/")
    print("  3. Embeddings will be generated fresh with correct Ollama API")
    print("  4. Test retrieval: POST /ask")
    print()

if __name__ == "__main__":
    response = input("This will delete all cached data. Continue? [y/N]: ")
    if response.lower() in ['y', 'yes']:
        reset_all_caches()
    else:
        print("Aborted.")
        sys.exit(0)
