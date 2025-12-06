import typer
from app.utils.rag_pipeline import rag_query

app = typer.Typer()

@app.command()
def ask(question: str):
    """Ask a question using the RAG system."""
    answer = rag_query(question)
    typer.echo(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    app()
