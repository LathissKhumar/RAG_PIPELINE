# RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, and Ollama. This pipeline provides intelligent document processing, hybrid retrieval (BM25 + vector search), and LLM-powered question answering.

## 🚀 Features

- **PDF Processing**: Advanced PDF-to-Markdown conversion using Docling with OCR support
- **Hybrid Retrieval**: Combines BM25 sparse retrieval and dense vector search with Reciprocal Rank Fusion (RRF)
- **Smart Chunking**: Intelligent document chunking with support for code, markdown, and hybrid fallback strategies
- **Embedding Cache**: SQLite-based embedding cache to avoid redundant computations
- **Async Workers**: Parallel embedding generation with configurable worker pools
- **Vector Store**: ChromaDB for efficient similarity search
- **LLM Integration**: Ollama integration for answer generation with context
- **Reranking**: Optional cross-encoder reranking for improved relevance
- **RESTful API**: FastAPI-based API with automatic documentation
- **CLI Tool**: Command-line interface for easy interaction

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [CLI Tool](#cli-tool)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🏗 Architecture

```
┌─────────────────┐
│   FastAPI API   │
└────────┬────────┘
         │
         ├─────────────┬─────────────┬──────────────┐
         │             │             │              │
    ┌────▼────┐   ┌───▼────┐   ┌────▼─────┐   ┌───▼─────┐
    │ Upload  │   │  Ask   │   │ Health   │   │   CLI   │
    │ Router  │   │Endpoint│   │  Check   │   │  Client │
    └────┬────┘   └───┬────┘   └──────────┘   └─────────┘
         │            │
         │            │
    ┌────▼────────────▼────┐
    │  Docling Converter   │
    │  (PDF → Markdown)    │
    └──────────┬───────────┘
               │
    ┌──────────▼──────────┐
    │  Chunking System    │
    │  - Markdown         │
    │  - Code             │
    │  - Hybrid Fallback  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Embedding Workers  │
    │  (Async Pool)       │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Embedding Cache    │
    │  (SQLite)           │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   ChromaDB Store    │
    └─────────────────────┘
               │
    ┌──────────▼──────────┐
    │ Hybrid Retriever    │
    │  - BM25 (Sparse)    │
    │  - Vector (Dense)   │
    │  - RRF Fusion       │
    │  - Reranking        │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │    Ollama LLM       │
    │  (Answer Gen)       │
    └─────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running
- GPU recommended for faster processing (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/LathissKhumar/RAG_PIPELINE.git
cd RAG_PIPELINE
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Ollama

```bash
# Install Ollama (visit https://ollama.ai/ for instructions)

# Pull required models
ollama pull bge-m3      # Embedding model
ollama pull llama3      # LLM model (or your preferred model)
```

### Step 5: Initialize Directories

The application will automatically create required directories on first run:
- `uploaded_pdfs/` - Stores uploaded PDF files
- `converted_mds/` - Stores converted Markdown files
- `chroma_db/` - ChromaDB vector store
- `bm25_index/` - BM25 index files

## ⚙️ Configuration

Configure the application using environment variables. Create a `.env` file in the root directory:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=bge-m3
LLM_MODEL=llama3

# Embedding Worker Configuration
EMBED_WORKERS=3
EMBED_BATCH_SIZE=64
EMBED_BATCH_WAIT_MS=200
EMBED_CACHE_PATH=embeddings_cache.sqlite3

# ChromaDB Configuration
CHROMA_COLLECTION=documents

# Hybrid Retrieval Configuration
HYBRID_ALPHA=0.5  # 0=BM25 only, 1=vector only, 0.5=balanced
USE_RERANKER=1    # 1=enabled, 0=disabled

# LLM Configuration
OLLAMA_LLM_TIMEOUT=180
OLLAMA_LLM_RETRIES=3
OLLAMA_LLM_BACKOFF=1.5

# API Configuration
API_BASE_URL=http://localhost:8000
ASK_TOP_K=5
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_MODEL` | `bge-m3` | Ollama embedding model |
| `LLM_MODEL` | `llama3` | Ollama LLM model for answers |
| `EMBED_WORKERS` | `3` | Number of parallel embedding workers |
| `EMBED_BATCH_SIZE` | `64` | Batch size for embedding generation |
| `HYBRID_ALPHA` | `0.5` | Weight for dense vs sparse retrieval |
| `USE_RERANKER` | `1` | Enable cross-encoder reranking |
| `CHROMA_COLLECTION` | `documents` | ChromaDB collection name |

## 🚀 Usage

### Starting the API Server

```bash
# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "ollama_available": true,
  "workers_running": true
}
```

#### 2. Upload and Convert PDFs

```bash
curl -X POST "http://localhost:8000/convert/" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response:**
```json
{
  "successful": [
    {
      "filename": "document1.pdf",
      "md_path": "converted_mds/document1/document1.md",
      "chunks_count": 42
    }
  ],
  "failed": [],
  "total_processed": 1,
  "total_successful": 1,
  "total_failed": 0
}
```

**Features:**
- Supports multiple file upload (max 10 files per request)
- Maximum file size: 50MB per file
- Automatic OCR for scanned documents
- Smart chunking and embedding
- Duplicate detection and deduplication

#### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 5,
    "use_llm": true
  }'
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "Based on the provided context, the main topic...",
  "top_k": 5,
  "results": [
    {
      "id": "document1__001",
      "text": "Relevant text chunk...",
      "metadata": {
        "source_md": "converted_mds/document1/document1.md",
        "chunk_index": 1
      },
      "distance": 0.85
    }
  ]
}
```

**Query Parameters:**
- `question` (required): The question to ask
- `top_k` (optional, default: 10): Number of chunks to retrieve
- `use_llm` (optional, default: true): Generate LLM answer
- `where` (optional): Metadata filters (ChromaDB format)
- `where_document` (optional): Document content filters

### CLI Tool

The CLI provides a convenient way to interact with the RAG system from the terminal.

#### Installation

```bash
# The CLI is installed with the package dependencies
python -m app.cli --help
```

#### Ask Questions

```bash
# Basic question
python -m app.cli ask "What is the main topic?"

# Retrieve more results
python -m app.cli ask "Explain the methodology" --top-k 10

# Show source chunks
python -m app.cli ask "What are the key findings?" --show-sources

# Disable LLM and show only retrieved chunks
python -m app.cli ask "Find information about X" --no-llm
```

**CLI Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--top-k` | Number of results to return | 5 |
| `--no-llm` | Disable LLM answer generation | False |
| `--show-sources` | Show source chunks with answer | False |

**Environment Variables for CLI:**
```bash
export API_BASE_URL=http://localhost:8000
export ASK_TOP_K=5
```

## 📚 API Documentation

### Request/Response Models

#### QueryRequest
```python
{
  "question": str,           # Required: Question to ask
  "top_k": int = 10,        # Optional: Number of results
  "use_llm": bool = True,   # Optional: Generate LLM answer
  "where": dict = None,     # Optional: Metadata filter
  "where_document": dict = None  # Optional: Document filter
}
```

#### AskResponse
```python
{
  "question": str,          # Echo of the question
  "answer": str | None,     # LLM-generated answer
  "top_k": int,            # Number of results returned
  "results": [             # Retrieved chunks
    {
      "id": str,           # Chunk ID
      "text": str,         # Chunk text
      "metadata": dict,    # Chunk metadata
      "distance": float    # Similarity score
    }
  ]
}
```

### Metadata Filtering

Filter results by metadata fields:

```python
# Find chunks from a specific document
{
  "question": "What is X?",
  "where": {
    "source_md": {"$eq": "converted_mds/document1/document1.md"}
  }
}

# Find recent chunks (if timestamp metadata exists)
{
  "question": "Recent updates?",
  "where": {
    "timestamp": {"$gte": "2024-01-01"}
  }
}
```

## 🔧 Development

### Project Structure

```
RAG_PIPELINE/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── cli.py                  # CLI client
│   ├── routers/
│   │   └── upload.py          # Upload endpoints
│   ├── embeddings/
│   │   ├── worker.py          # Async embedding workers
│   │   ├── cache.py           # Embedding cache
│   │   └── ollama_embeddings.py
│   ├── vector_store/
│   │   └── chroma_client.py   # ChromaDB client
│   ├── retrieval/
│   │   ├── hybrid_retriever.py # Hybrid retrieval
│   │   ├── bm25_retriever.py   # BM25 retrieval
│   │   └── reranker.py         # Cross-encoder reranker
│   ├── llm/
│   │   └── ollama_llm.py      # LLM integration
│   └── utils/
│       ├── docling_converter.py # PDF conversion
│       ├── file_registry.py    # File tracking
│       └── chunker/
│           ├── markdown_chunker.py
│           ├── code_chunker.py
│           ├── hybrid_fallback.py
│           └── optimizer.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Test Ollama API connection
python test_ollama_api.py

# Test dense retrieval
python test_dense_retrieval.py

# Test embedding with debug info
python test_embedding_debug.py

# Diagnose ChromaDB issues
python diagnose_chroma.py
```

### Reset Caches

If you need to clear caches and rebuild:

```bash
# Clear all caches and databases
python reset_caches.py

# This will remove:
# - embeddings_cache.sqlite3
# - chroma_db/
# - bm25_index/
# - file_registry.db
```

### Code Quality

```bash
# Format code (if using ruff or black)
ruff format .

# Lint code
ruff check .
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Ollama Connection Failed

**Error:** `Ollama health check failed` or `Connection refused`

**Solution:**
- Ensure Ollama is running: `ollama serve`
- Check the URL: `OLLAMA_BASE_URL=http://localhost:11434`
- Verify models are pulled: `ollama list`

#### 2. Embedding Dimension Mismatch

**Error:** `Embedding dimension mismatch detected`

**Solution:**
- Clear ChromaDB and re-ingest: `python reset_caches.py`
- Ensure consistent `EMBED_MODEL` across runs
- Re-upload all documents after clearing

#### 3. Out of Memory

**Error:** `CUDA out of memory` or system hangs

**Solution:**
- Reduce `EMBED_BATCH_SIZE` (try 32 or 16)
- Reduce `EMBED_WORKERS` (try 1 or 2)
- Use CPU-only mode: `onnxruntime` instead of `onnxruntime-gpu`

#### 4. Slow PDF Processing

**Solution:**
- Reduce image quality in Docling settings
- Use GPU acceleration for OCR
- Process files in smaller batches

#### 5. No Results from Search

**Solution:**
- Check if documents were successfully ingested
- Verify ChromaDB collection: `python diagnose_chroma.py`
- Check BM25 index exists: `ls bm25_index/`
- Rebuild indexes if needed

### Logs

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This project is available for use and modification. Please check the repository for specific license information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub or contact the maintainer.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Docling](https://github.com/DS4SD/docling) - PDF conversion
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Embedding model

