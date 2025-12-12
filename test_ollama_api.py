#!/usr/bin/env python3
"""Test Ollama API connectivity and response format."""

import sys
import json
import requests

sys.path.insert(0, '/home/lathiss/Projects/RAG_PIPELINE')

OLLAMA_URL = "http://localhost:11434"

def test_ollama_connectivity():
    """Test basic Ollama connectivity and API format."""
    
    print("=" * 70)
    print("OLLAMA API CONNECTIVITY TEST")
    print("=" * 70)
    
    # Test 1: Check if Ollama is running
    print("\n[1] Checking if Ollama service is running...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            print(f"   ✓ Ollama is running")
            models = resp.json().get("models", [])
            print(f"   Available models: {len(models)}")
            for model in models[:5]:
                print(f"     - {model.get('name', 'unknown')}")
        else:
            print(f"   ✗ Ollama returned status {resp.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Cannot connect to Ollama: {e}")
        print(f"   Make sure Ollama is running: ollama serve")
        return
    
    # Test 2: Try embedding with "prompt" field
    print("\n[2] Testing /api/embeddings with 'prompt' field...")
    try:
        payload = {"model": "bge-m3", "prompt": "test embedding"}
        print(f"   Request: {json.dumps(payload)}")
        resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=30)
        print(f"   Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Response keys: {list(data.keys())}")
            
            if "embedding" in data:
                emb = data["embedding"]
                if emb and len(emb) > 0:
                    print(f"   ✓ Success! Embedding dimension: {len(emb)}")
                    print(f"   First 5 values: {emb[:5]}")
                else:
                    print(f"   ✗ Empty embedding returned")
                    print(f"   Full response: {data}")
            else:
                print(f"   ✗ No 'embedding' key in response")
                print(f"   Full response: {json.dumps(data, indent=2)[:500]}")
        else:
            print(f"   ✗ Request failed: {resp.text[:200]}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Try with "input" field (alternative)
    print("\n[3] Testing /api/embeddings with 'input' field (alternative)...")
    try:
        payload = {"model": "bge-m3", "input": "test embedding"}
        resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=30)
        print(f"   Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if "embedding" in data and data["embedding"]:
                print(f"   ✓ 'input' field also works!")
            else:
                print(f"   ✗ 'input' field did not work")
        else:
            print(f"   ✗ Request failed with 'input' field")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Check if model needs to be pulled
    print("\n[4] Checking if bge-m3 model is available...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any("bge-m3" in name for name in model_names):
                print(f"   ✓ bge-m3 model is available")
            else:
                print(f"   ✗ bge-m3 model not found")
                print(f"   Available models: {', '.join(model_names[:5])}")
                print(f"   Run: ollama pull bge-m3")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_ollama_connectivity()
