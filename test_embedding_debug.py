#!/usr/bin/env python3
"""Quick test with debug logging enabled."""

import sys
import logging

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, '/home/lathiss/Projects/RAG_PIPELINE')

from app.embeddings.ollama_embeddings import embed_texts

print("Testing Ollama embedding with DEBUG logging...\n")

try:
    result = embed_texts(["test query"], model="bge-m3")
    if result and len(result) > 0:
        print(f"\n✓ SUCCESS! Embedding dimension: {len(result[0])}")
    else:
        print("\n✗ Empty result")
except Exception as e:
    print(f"\n✗ FAILED: {e}")
