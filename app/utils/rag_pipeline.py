from langchain.llms import Ollama
from .vector_store import load_vectorstore

llm = Ollama(model="llama3")

def rag_query(question: str, store_name: str = "default_store"):
    vs = load_vectorstore(store_name)

    docs = vs.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the following context to answer the question.

### Context:
{context}

### Question:
{question}

### Answer:
"""

    return llm(prompt)
