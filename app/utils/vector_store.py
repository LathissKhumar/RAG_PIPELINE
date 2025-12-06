import os
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

EMBED_DIR = "embeddings_store"
os.makedirs(EMBED_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_chunks_to_vectorstore(json_chunks_path: str, store_name: str):
    with open(json_chunks_path, "r", encoding="utf-8") as jf:
        chunks = json.load(jf)

    texts = [c["content"] for c in chunks]
    metadata = [{"header_path": c["header_path"]} for c in chunks]

    vectorstore = FAISS.from_texts(texts=texts, embedding=embedding_model, metadatas=metadata)
    save_path = os.path.join(EMBED_DIR, store_name)

    vectorstore.save_local(save_path)
    return save_path


def load_vectorstore(store_name: str):
    path = os.path.join(EMBED_DIR, store_name)
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
