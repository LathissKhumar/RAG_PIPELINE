#!/usr/bin/env python3
"""Diagnostic script for dense retrieval dimension mismatch issues."""

import os
import sys
sys.path.insert(0, '/home/lathiss/Projects/RAG_PIPELINE')

from app.vector_store.chroma_client import get_chroma_client
from app.embeddings.ollama_embeddings import embed_texts

def diagnose_chroma_dimensions():
    """Check Chroma collection and embedding dimensions."""
    
    print("=" * 70)
    print("CHROMA DIMENSION DIAGNOSTIC")
    print("=" * 70)
    
    # Get environment config
    embed_model = os.getenv("EMBED_MODEL", "bge-m3")
    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
    
    print(f"\n1. CONFIGURATION")
    print(f"   EMBED_MODEL: {embed_model}")
    print(f"   CHROMA_COLLECTION: {collection_name}")
    print(f"   OLLAMA_URL: {os.getenv('OLLAMA_URL', 'http://localhost:11434')}")
    
    # Test query embedding
    print(f"\n2. QUERY EMBEDDING TEST")
    try:
        test_query = "test query for dimension check"
        query_emb = embed_texts([test_query], model=embed_model)
        query_dim = len(query_emb[0].tolist())
        print(f"   ✓ Query embedding dimension: {query_dim}")
    except Exception as e:
        print(f"   ✗ Failed to generate query embedding: {e}")
        return
    
    # Check Chroma collection
    print(f"\n3. CHROMA COLLECTION CHECK")
    try:
        client = get_chroma_client()
        print(f"   ✓ Chroma client initialized")
        
        try:
            collection = client.get_collection(collection_name)
            print(f"   ✓ Collection '{collection_name}' exists")
        except Exception:
            print(f"   ✗ Collection '{collection_name}' does not exist")
            print(f"   → Create collection by uploading documents first")
            return
        
        # Get collection count
        count = collection.count()
        print(f"   ✓ Collection has {count} documents")
        
        if count == 0:
            print(f"   ⚠ Collection is empty - no dimension mismatch possible yet")
            print(f"   → Upload documents to populate the collection")
            return
        
        # Get sample embedding
        sample = collection.get(limit=1, include=["embeddings"])
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            stored_dim = len(sample["embeddings"][0])
            print(f"   ✓ Stored embedding dimension: {stored_dim}")
            
            # Compare dimensions
            print(f"\n4. DIMENSION COMPARISON")
            if query_dim == stored_dim:
                print(f"   ✓ DIMENSIONS MATCH ({query_dim} == {stored_dim})")
                print(f"   → Dense retrieval should work correctly")
            else:
                print(f"   ✗ DIMENSION MISMATCH!")
                print(f"      Query:  {query_dim} dimensions")
                print(f"      Stored: {stored_dim} dimensions")
                print(f"\n   SOLUTION:")
                print(f"   1. Backup important data")
                print(f"   2. Clear Chroma DB: rm -rf chroma_db/")
                print(f"   3. Restart workers: pkill -f worker.py")
                print(f"   4. Re-upload documents with correct EMBED_MODEL={embed_model}")
                print(f"   5. Or change EMBED_MODEL to match stored dimension")
        else:
            print(f"   ✗ Could not retrieve embeddings from collection")
            
    except Exception as e:
        print(f"   ✗ Error checking Chroma: {e}")
        import traceback
        traceback.print_exc()
    
    # Test actual query
    print(f"\n5. ACTUAL QUERY TEST")
    try:
        from app.vector_store.chroma_client import query_texts
        result = query_texts(client, collection_name, test_query, top_k=2)
        result_count = len(result.get("ids", [[]])[0])
        print(f"   ✓ Query successful! Retrieved {result_count} results")
    except Exception as e:
        print(f"   ✗ Query failed: {e}")
        import traceback
        print(traceback.format_exc())
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    diagnose_chroma_dimensions()
