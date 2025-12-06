import os
from langchain_community.llms import Ollama
from .vector_store import load_vectorstore

# Make the model configurable via environment variable
llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama3"))


def rag_query(question: str, store_name: str = "default_store"):
    """
    Query the RAG system with a question using context from the vector store.

    Args:
        question: The question to answer.
        store_name: Name of the vector store to query (default: "default_store").

    Returns:
        The generated answer from the LLM based on retrieved context.
    """
    try:
        # Load vector store
        vs = load_vectorstore(store_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store '{store_name}': {e}")

    try:
        # Retrieve similar documents
        docs = vs.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
        raise RuntimeError(f"Failed during similarity search: {e}")

    # Build the prompt
    prompt = f"""
Use the following context to answer the question.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    try:
        # Call the LLM
        return llm(prompt)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")
