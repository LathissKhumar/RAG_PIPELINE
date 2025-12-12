# app/utils/vector_store.py

import os
import json
from typing import List, Dict

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Where all FAISS DBs will be stored
VECTOR_DIR = "vectorstores"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(chunks_json_path: str, store_name: str) -> str:
    """
    Load chunked JSON file and create a FAISS vectorstore.
    Returns the path to the stored DB.
    """
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks: List[Dict] = json.load(f)

    texts = [c["content"] for c in chunks]
    metadata = [{"header_path": c["header_path"]} for c in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadata
    )

    save_path = os.path.join(VECTOR_DIR, store_name)
    vectorstore.save_local(save_path)
    return save_path


def load_vectorstore(store_name: str):
    """
    Load a FAISS vectorstore from disk.
    """
    path = os.path.join(VECTOR_DIR, store_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
