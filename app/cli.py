# cli.py
import typer
from app.utils.vector_store import build_vectorstore, load_vectorstore
from langchain_community.llms import Ollama

app = typer.Typer()

llm = Ollama(model="llama3")

@app.command()
def build(json_path: str, name: str = "default_store"):
    """
    Build a FAISS vectorstore from header-chunked JSON.
    """
    store_path = build_vectorstore(json_path, name)
    typer.echo(f"Vectorstore saved at: {store_path}")


@app.command()
def ask(question: str, db: str = "default_store"):
    """
    Query the vectorstore using RAG.
    """
    store = load_vectorstore(db)
    docs = store.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Use the below context to answer the question.

### CONTEXT
{context}

### QUESTION
{question}

### ANSWER
"""

    answer = llm(prompt)
    typer.echo(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    app()
