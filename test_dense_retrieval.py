#!/usr/bin/env python3
"""Test dense retrieval fix for IndexError."""

import sys
import os
sys.path.insert(0, '/home/lathiss/Projects/RAG_PIPELINE')

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

from app.vector_store.chroma_client import get_chroma_client, query_texts

def test_query_validation():
    """Test query_texts with various edge cases."""
    
    print("=" * 60)
    print("TESTING DENSE RETRIEVAL QUERY VALIDATION")
    print("=" * 60)
    
    client = get_chroma_client()
    collection_name = os.getenv("CHROMA_COLLECTION", "documents")
    
    # Test 1: Empty collection
    print("\n[1] Testing with potentially empty collection...")
    try:
        result = query_texts(client, collection_name, "test query", top_k=5)
        result_count = len(result.get("ids", [[]])[0])
        print(f"   ✓ Query returned {result_count} results (empty collection handled)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 2: Empty query
    print("\n[2] Testing with empty query...")
    try:
        result = query_texts(client, collection_name, "", top_k=5)
        result_count = len(result.get("ids", [[]])[0])
        print(f"   ✓ Empty query handled, returned {result_count} results")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 3: Whitespace query
    print("\n[3] Testing with whitespace query...")
    try:
        result = query_texts(client, collection_name, "   ", top_k=5)
        result_count = len(result.get("ids", [[]])[0])
        print(f"   ✓ Whitespace query handled, returned {result_count} results")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test 4: Normal query
    print("\n[4] Testing with normal query...")
    try:
        result = query_texts(client, collection_name, "What is machine learning?", top_k=5)
        result_count = len(result.get("ids", [[]])[0])
        print(f"   ✓ Normal query successful, returned {result_count} results")
        
        if result_count > 0:
            print(f"   Sample result ID: {result['ids'][0][0]}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_query_validation()
