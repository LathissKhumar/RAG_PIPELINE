import os
import sys
import requests
import typer
from typing import Optional

app = typer.Typer(help="Terminal client for the RAG API")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_TOP_K = int(os.getenv("ASK_TOP_K", "5"))


def _print_result(idx: int, item: dict):
    distance = item.get("distance")
    dist_txt = f"distance={distance:.4f}" if isinstance(distance, (int, float)) else ""
    typer.echo(f"\n[{idx}] {item.get('id', '')} {dist_txt}")
    text = item.get("text", "") or ""
    typer.echo(text.strip())


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the RAG service"),
    top_k: int = typer.Option(DEFAULT_TOP_K, help="Number of results to return"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM answer generation, show only retrieved chunks"),
    show_sources: bool = typer.Option(False, "--show-sources", help="Show source chunks along with answer"),
):
    """Send a question to the /ask endpoint and print the LLM-generated answer."""
    url = API_BASE_URL.rstrip("/") + "/ask"
    payload = {"question": question, "top_k": top_k, "use_llm": not no_llm}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        typer.echo(f"Request failed: {e}", err=True)
        raise typer.Exit(code=1)

    data = resp.json()
    results = data.get("results", []) if isinstance(data, dict) else []
    answer = data.get("answer")
    
    if not results:
        typer.echo("No results returned")
        raise typer.Exit(code=0)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"Question: {data.get('question', question)}")
    typer.echo(f"{'='*60}\n")
    
    # Display LLM answer if available
    if answer:
        typer.secho("Answer:", fg="green", bold=True)
        typer.echo(answer)
        typer.echo()
    
    # Optionally show source chunks
    if show_sources or not answer:
        typer.secho(f"Source Chunks ({len(results)}):", fg="cyan", bold=True)
        for idx, item in enumerate(results, start=1):
            _print_result(idx, item)


if __name__ == "__main__":
    app()
